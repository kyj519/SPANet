#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored Run-II dataset builder
---------------------------------
- Keeps your hashing-based fold assignment (cppyy + splitmix64)
- Loads minimal branches via uproot/awkward in parallel
- Normalizes helper names, removes globals, adds type hints & docstrings
- Makes saving deterministic and explicit (no hidden global dicts)
- Optional ratio-balanced equal class sampling per era

Usage (default: writes inclusive_RunII_fold{K}.h5 in CWD):

    python refactored_dataset_builder.py \
        --outdir ./out \
        --workers 8 \
        --nbuckets 3 \
        --max-jet 9

Notes
-----
- The code assumes your input ROOT trees contain the branches accessed below
  (run, lumi, event, Jet_*, Lepton_*, Met_*, genTtbarId, decay_mode, etc.).
- If a branch is missing the script will raise a KeyError; you can soften
  this by adding try/except in `load_tree_minimal`.
- Era indexing is expected to be present as `era_index` in the tree. If not,
  adapt `attach_fixed_fields(...)` to inject it from the filename/era arg.
"""

from __future__ import annotations

import os
import shutil
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import awkward as ak
import uproot
import h5py
import cppyy
from concurrent.futures import ThreadPoolExecutor

# -----------------------------------------------------------------------------
# C++ fold hashing (unchanged logic, wrapped nicely)
# -----------------------------------------------------------------------------

def _compile_cpp_once() -> None:
    cppyy.cppdef(r"""
static inline uint64_t splitmix64(uint64_t x){
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
} // https://prng.di.unimi.it/splitmix64.c

static inline uint32_t rle_bucket(uint64_t run, uint64_t lumi, uint64_t event, uint32_t nbuckets){
    uint64_t seed = run  * 0x9E3779B97F4A7C15ULL
                  ^ lumi * 0xBF58476D1CE4E5B9ULL
                  ^ event* 0x94D049BB133111EBULL;
    uint64_t h = splitmix64(seed);

    __uint128_t prod = (__uint128_t)h * (__uint128_t)nbuckets;
    return (uint32_t)(prod >> 64);  
}

extern "C" void rle_bucket_bulk_addr(uintptr_t run_addr,
                                     uintptr_t lumi_addr,
                                     uintptr_t event_addr,
                                     size_t n,
                                     uint32_t nbuckets,
                                     uintptr_t out_addr) {
    const uint64_t* run   = reinterpret_cast<const uint64_t*>(run_addr);
    const uint64_t* lumi  = reinterpret_cast<const uint64_t*>(lumi_addr);
    const uint64_t* event = reinterpret_cast<const uint64_t*>(event_addr);
    uint32_t* out         = reinterpret_cast<uint32_t*>(out_addr);
    for (size_t i = 0; i < n; ++i) {
        out[i] = rle_bucket(run[i], lumi[i], event[i], nbuckets);
    }
}
    """)


def compute_fold_cpp(run: ak.Array | np.ndarray,
                     lumi: ak.Array | np.ndarray,
                     event: ak.Array | np.ndarray,
                     nbuckets: int = 3) -> np.ndarray:
    """Return fold indices using the exact C++ logic on raw pointers."""
    arun   = np.ascontiguousarray(ak.to_numpy(run),   dtype=np.uint64)
    alumi  = np.ascontiguousarray(ak.to_numpy(lumi),  dtype=np.uint64)
    aevent = np.ascontiguousarray(ak.to_numpy(event), dtype=np.uint64)
    n = arun.shape[0]
    out = np.empty(n, dtype=np.uint32)
    cppyy.gbl.rle_bucket_bulk_addr(
        int(arun.ctypes.data),
        int(alumi.ctypes.data),
        int(aevent.ctypes.data),
        n,
        int(nbuckets),
        int(out.ctypes.data),
    )
    return out

# -----------------------------------------------------------------------------
# I/O + transformation helpers
# -----------------------------------------------------------------------------

def attach_fixed_fields(arr: ak.Array,
                        lepton_m: float,
                        etag: int,
                        utag: int,
                        y: int,
                        nbuckets: int) -> ak.Array:
    """Attach constant/derived fields and the fold index to an awkward table."""
    n = len(arr["run"])  # type: ignore[index]
    row_index = ak.local_index(arr["run"])  # analogous to rdfentry_
    fold = compute_fold_cpp(arr["run"], arr["lumi"], arr["event"], nbuckets=nbuckets)

    arr = ak.with_field(arr, ak.Array(np.full(n, float(lepton_m))), "Lepton_M")
    arr = ak.with_field(arr, ak.Array(np.full(n, int(etag), dtype=np.int32)), "etag")
    arr = ak.with_field(arr, ak.Array(np.full(n, int(utag), dtype=np.int32)), "utag")
    arr = ak.with_field(arr, ak.Array(np.full(n, int(y),    dtype=np.int32)), "y")
    arr = ak.with_field(arr, row_index, "row_index")
    arr = ak.with_field(arr, ak.Array(fold), "fold")
    return arr


def load_tree_minimal(file_path: str,
                      tree_path: str,
                      lepton_m: float,
                      etag: int,
                      utag: int,
                      y: int,
                      nbuckets: int) -> ak.Array:
    """Read a training tree and attach derived fields/fold.

    Parameters
    ----------
    file_path : path to ROOT file
    tree_path : e.g. "El/Training_Tree" or "Mu/Training_Tree"
    lepton_m  : mass to attach in GeV (0.00051 e, 0.10566 mu)
    etag/utag : extra integer tags
    y         : initial class label (will be remapped later by renew_y)
    nbuckets  : number of folds
    """
    with uproot.open(file_path) as f:
        tree = f[tree_path]
        arr = tree.arrays(library="ak")
    return attach_fixed_fields(arr, lepton_m, etag, utag, y, nbuckets)


import awkward as ak

def renew_y(arr: ak.Array) -> ak.Array:
    """
    Remap class label y using the same scheme as build_input_tuple:
      0: W→c b̄  (merge old 0 & 1: both reco-correct and reco-wrong; identified by decay_mode==45)
      1: CX + extra b   (CS/CD with additional b radiation)
      2: UX + extra b   (US/UD with additional b radiation)
      3: CX + extra c   (CS/CD with additional c radiation)
      4: CX + light     (CS/CD with no extra heavy flavor)
      5: UX + not BB    (US/UD with CC or light; i.e., anything that is not extra b)
    Anything not covered keeps its original y.
    """
    # Inputs
    dm  = arr["decay_mode"]
    gid = arr["genTtbarId"]
    gent = gid % 100

    # ---- decay-mode masks (match your _cuts) ----
    is_cb = (dm == 45)                           # W→cb
    is_cx = ((dm // 10) % 10 == 4) & (~is_cb)    # W→cs/cd only (exclude cb=45)
    is_ux = ((dm // 10) % 10 == 2)               # W→us/ud

    # ---- HF radiation masks from genTtbarId % 100 ----
    extra_b = (gent >= 51) & (gent <= 55)        # BB
    extra_c = (gent >= 41) & (gent <= 45)        # CC
    light   = ~(extra_b | extra_c)               # no extra HF (includes gent == 0)

    # start from existing labels; overwrite where masks apply
    y_new = ak.ones_like(arr["y"]) * -1  # default: -1 (ignore)
    y_new = ak.where(is_cb,                0, y_new)  # merge (0,1) -> 0
    y_new = ak.where(is_cx & extra_b,      1, y_new)  # CX + BB
    y_new = ak.where(is_ux & extra_b,      2, y_new)  # UX + BB
    y_new = ak.where(is_cx & extra_c,      3, y_new)  # CX + CC
    y_new = ak.where(is_cx & light,        4, y_new)  # CX + light (!HF)
    y_new = ak.where(is_ux & (~extra_b),   5, y_new)  # UX + (CC or light) == not BB

    if ak.all(y_new == -1):
        raise ValueError("All events got y=-1 after remapping; check the logic.")
    return ak.with_field(arr, y_new, "y")

import awkward as ak
import numpy as np
from typing import Dict

def coverage_stats(arr: ak.Array) -> Dict[str, int | bool]:
    dm   = arr["decay_mode"]
    gent = arr["genTtbarId"] % 100

    # decay-mode 분기
    is_cb = (dm == 45)                               # W→c b̄  (class 0)
    is_cx = ((dm // 10) % 10 == 4) & (~is_cb)        # W→c(s/d) (45 제외)
    is_ux = ((dm // 10) % 10 == 2)                   # W→u(s/d)

    # HF 라디에이션
    extra_b = (gent >= 51) & (gent <= 55)            # BB
    extra_c = (gent >= 41) & (gent <= 45)            # CC
    light   = ~(extra_b | extra_c)                   # no extra HF

    # 클래스 마스크
    m0 = is_cb
    m2 = is_cx & extra_b
    m3 = is_ux & extra_b
    m4 = is_cx & extra_c
    m5 = is_cx & light
    m6 = is_ux & (~extra_b)  # UX + (CC or light) == not BB

    covered = m0 | m2 | m3 | m4 | m5 | m6

    # --- 중복 검사: 각 이벤트가 몇 개 마스크에 걸렸는지 카운트 ---
    cnt = (ak.values_astype(m0, np.int8) + ak.values_astype(m2, np.int8) +
           ak.values_astype(m3, np.int8) + ak.values_astype(m4, np.int8) +
           ak.values_astype(m5, np.int8) + ak.values_astype(m6, np.int8))
    dup_mask = cnt > 1
    num_duplicates = int(ak.sum(dup_mask))

    if num_duplicates > 0:
        # 필요하면 디버그용으로 중복 인덱스 몇 개 뽑아볼 수도 있음:
        # dup_indices = ak.to_numpy(ak.where(dup_mask)[0])[:10]
        # raise ValueError(f"{num_duplicates} duplicates. examples: {dup_indices}")
        raise ValueError(f"Found {num_duplicates} duplicate events across class masks (an event matched >1 class).")

    total = int(ak.count(dm))
    num_uncovered = int(ak.sum(~covered))
    return {
        "all_covered": bool(ak.all(covered)),
        "num_uncovered": num_uncovered,
        "total": total,
        "num_duplicates": num_duplicates,
    }



# -----------------------------------------------------------------------------
# Balanced sampling by era/class
# -----------------------------------------------------------------------------

def _proportional_with_caps(counts: np.ndarray,
                            weights: np.ndarray,
                            total: int,
                            rng: np.random.Generator) -> np.ndarray:
    counts  = np.asarray(counts, dtype=int)
    weights = np.asarray(weights, dtype=float)
    n = np.zeros_like(counts, dtype=int)
    remaining = int(total)

    avail = (weights > 0) & (counts > 0)
    if remaining <= 0 or not avail.any():
        return n

    while remaining > 0 and avail.any():
        w = weights * avail
        s = w.sum()

        if s == 0:
            idxs = np.flatnonzero(avail)
            cap  = counts[idxs] - n[idxs]
            if cap.sum() <= 0:
                break
            order = rng.permutation(idxs)
            for k in order:
                if remaining == 0:
                    break
                if n[k] < counts[k]:
                    n[k] += 1
                    remaining -= 1
            avail = (weights > 0) & (n < counts)
            continue

        alloc_real = remaining * w / s
        base = np.floor(alloc_real).astype(int)
        cap = counts - n
        base = np.minimum(base, cap)

        n += base
        remaining -= int(base.sum())
        if remaining == 0:
            break

        rem = alloc_real - base
        eligible = np.flatnonzero(avail & (n < counts))
        if eligible.size == 0:
            break
        order = eligible[np.argsort(rem[eligible])[::-1]]
        for k in order:
            if remaining == 0:
                break
            if n[k] < counts[k]:
                n[k] += 1
                remaining -= 1
        avail = (weights > 0) & (n < counts)
    return n


def balanced_equal_class_counts_by_era(era_index: ak.Array | np.ndarray,
                                       y: ak.Array | np.ndarray,
                                       ratio: Optional[Sequence[float]] = None,
                                       seed: int = 42,
                                       strict: bool = False,
                                       return_plan: bool = False):
    era = ak.to_numpy(era_index) if isinstance(era_index, ak.Array) else np.asarray(era_index)
    cls = ak.to_numpy(y)         if isinstance(y,         ak.Array) else np.asarray(y)
    assert era.shape == cls.shape

    if ratio is None:
        ratio = np.ones(4, dtype=float)
    else:
        ratio = np.asarray(ratio, dtype=float)
        assert ratio.shape == (4,)

    rng = np.random.default_rng(seed)
    classes = np.unique(cls)

    pools  = {int(c): [np.flatnonzero((cls == c) & (era == k)) for k in range(4)] for c in classes}
    counts = {int(c): np.array([len(p) for p in pools[int(c)]], dtype=int)       for c in classes}

    weights = {c: ratio * (counts[c] > 0) for c in counts}

    if strict:
        bad = [(c, int(k)) for c in counts for k in range(4) if ratio[k] > 0 and counts[c][k] == 0]
        if bad:
            raise ValueError(f"ratio>0인데 표본이 없는 (class, era): {bad}")

    T_max = {}
    for c in counts:
        w = weights[c]
        if w.sum() == 0:
            T_max[c] = 0
            continue
        s_candidates = [counts[c][k] / w[k] for k in range(4) if w[k] > 0]
        s_max = min(s_candidates) if s_candidates else 0.0
        take_max = np.floor(np.minimum(counts[c], s_max * w)).astype(int)
        T_max[c] = int(take_max.sum())

    T_target = min(T_max.values()) if len(T_max) > 0 else 0

    sel_all = []
    plan = {"T_target": int(T_target), "per_class": {}}

    for c in counts:
        w = weights[c]
        cnt = counts[c]
        if w.sum() == 0 or T_target == 0:
            take = np.zeros(4, dtype=int)
        else:
            take = _proportional_with_caps(cnt, w, T_target, rng)

        took = []
        for k in range(4):
            n = int(take[k])
            if n > 0:
                took.append(rng.choice(pools[c][k], size=n, replace=False))
        if took:
            sel_c = np.concatenate(took)
            sel_all.append(sel_c)

        plan["per_class"][c] = {
            "counts": cnt.tolist(),
            "weights": w.tolist(),
            "take": take.tolist(),
            "T_max": T_max[c],
        }

    if sel_all:
        sel = np.concatenate(sel_all)
        rng.shuffle(sel)
    else:
        sel = np.empty(0, dtype=int)

    if return_plan:
        plan["total_selected"] = int(sel.size)
        return sel, plan
    return sel

# -----------------------------------------------------------------------------
# Feature tensors
# -----------------------------------------------------------------------------

def prepare_MET(tbl: ak.Array) -> Dict[str, np.ndarray]:
    Met = {}
    Met['met']     = ak.to_numpy(tbl['Met_Pt'])
    Met['cos_phi'] = ak.to_numpy(np.cos(tbl['Met_Phi']))
    Met['sin_phi'] = ak.to_numpy(np.sin(tbl['Met_Phi']))
    Met['MASK']    = np.ones(len(Met['met']), dtype=bool)
    return Met


def prepare_Lepton(tbl: ak.Array) -> Dict[str, np.ndarray]:
    L = {}
    L['pt']      = ak.to_numpy(tbl['Lepton_Pt'])
    L['eta']     = ak.to_numpy(tbl['Lepton_Eta'])
    L['cos_phi'] = ak.to_numpy(np.cos(tbl['Lepton_Phi']))
    L['sin_phi'] = ak.to_numpy(np.sin(tbl['Lepton_Phi']))
    L['utag']    = ak.to_numpy(tbl['utag'])
    L['etag']    = ak.to_numpy(tbl['etag'])
    L['Era']     = ak.to_numpy(tbl['era_index'])
    L['mass']    = ak.to_numpy(tbl['Lepton_M'])
    L['mask']    = np.ones(len(L['pt']), dtype=bool)
    return L


def _pad_to_np(tbl: ak.Array, field: str, max_jet: int, fill: float = 0.0, dtype=np.float32) -> np.ndarray:
    arr_pad  = ak.pad_none(tbl[field], max_jet, clip=True)
    arr_fill = ak.fill_none(arr_pad, fill)
    return ak.to_numpy(arr_fill).astype(dtype, copy=False)


def prepare_Jets(tbl: ak.Array, max_jet: int = 9) -> Dict[str, np.ndarray]:
    mask_ak = ~ak.is_none(ak.pad_none(tbl["Jet_Pt"], max_jet, clip=True), axis=1)
    mask_np = ak.to_numpy(mask_ak)

    pt    = _pad_to_np(tbl, "Jet_Pt",   max_jet)
    eta   = _pad_to_np(tbl, "Jet_Eta",  max_jet)
    phi   = _pad_to_np(tbl, "Jet_Phi",  max_jet)
    mass  = _pad_to_np(tbl, "Jet_Mass", max_jet)
    btag  = _pad_to_np(tbl, "Jet_BvsC", max_jet)
    cvsl  = _pad_to_np(tbl, "Jet_CvsL", max_jet)
    cvsb  = _pad_to_np(tbl, "Jet_CvsB", max_jet)

    cos_phi = np.cos(phi); sin_phi = np.sin(phi)
    cos_phi[~mask_np] = 0.0
    sin_phi[~mask_np] = 0.0

    era_vec = ak.to_numpy(tbl["era_index"]).astype(np.int32)
    Era = np.broadcast_to(era_vec[:, None], mask_np.shape).copy()
    Era[~mask_np] = 0

    return {
        "Mask": mask_np,
        "pt": pt,
        "eta": eta,
        "cos_phi": cos_phi,
        "sin_phi": sin_phi,
        "mass": mass,
        "qtag": mask_np.astype(np.float32),
        "btag": btag,
        "cvsl": cvsl,
        "cvsb": cvsb,
        "Era":  Era,
    }

# -----------------------------------------------------------------------------
# Edge (pairwise jet) tensors
# -----------------------------------------------------------------------------

def prepare_Edges(Momenta: Mapping[str, np.ndarray], event_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Build pairwise (N, J, J) edge features from dense jet tensors.

    Returned keys:
      - mask   : bool   (N,J,J) where both jets exist and i!=j
      - deltaR : float32(N,J,J)
      - invM   : float32(N,J,J) invariant mass of dijet system
      - z      : float32(N,J,J) min(pTi,pTj)/(pTi+pTj)
      - kT     : float32(N,J,J) min(pTi,pTj) * ΔR_ij

    Notes:
      * Uses pure NumPy broadcasting (no awkward vectorization).
      * Diagonal entries (i==j) are masked out and set to 0.
    """
    # Slice to selected events only
    Mask = Momenta['Mask'][event_mask]          # (N,J) bool
    pt   = Momenta['pt'][event_mask]
    eta  = Momenta['eta'][event_mask]
    cph  = Momenta['cos_phi'][event_mask]
    sph  = Momenta['sin_phi'][event_mask]
    mass = Momenta['mass'][event_mask]

    N, J = pt.shape

    # Pair mask (exclude diagonal)
    pair_mask = (Mask[:, :, None] & Mask[:, None, :])
    diag = np.eye(J, dtype=bool)[None, :, :]
    pair_mask &= ~diag

    # Δη and Δφ using trig identities (robust wrapping)
    deta = eta[:, :, None] - eta[:, None, :]
    cos_dphi = cph[:, :, None] * cph[:, None, :] + sph[:, :, None] * sph[:, None, :]
    sin_dphi = sph[:, :, None] * cph[:, None, :] - cph[:, :, None] * sph[:, None, :]
    dphi = np.arctan2(sin_dphi, cos_dphi)
    deltaR = np.sqrt(deta * deta + dphi * dphi)

    # Four-vectors
    px = pt * cph
    py = pt * sph
    pz = pt * np.sinh(eta)
    E  = np.sqrt((pt * np.cosh(eta)) ** 2 + mass ** 2)

    # Pair sums
    px_sum = px[:, :, None] + px[:, None, :]
    py_sum = py[:, :, None] + py[:, None, :]
    pz_sum = pz[:, :, None] + pz[:, None, :]
    E_sum  = E[:,  :, None] + E[:,  None, :]

    m2 = E_sum * E_sum - (px_sum * px_sum + py_sum * py_sum + pz_sum * pz_sum)
    invM = np.sqrt(np.clip(m2, 0.0, None))

    pti = pt[:, :, None]
    ptj = pt[:, None, :]
    denom = pti + ptj
    z = np.where(denom > 0, np.minimum(pti, ptj) / denom, 0.0)
    kT = np.minimum(pti, ptj) * deltaR

    # Apply mask and cast
    deltaR = (deltaR * pair_mask).astype(np.float32, copy=False)
    invM   = (invM   * pair_mask).astype(np.float32, copy=False)
    z      = (z      * pair_mask).astype(np.float32, copy=False)
    kT     = (kT     * pair_mask).astype(np.float32, copy=False)

    return {
        'mask': pair_mask,
        'deltaR': deltaR,
        'invM': invM,
        'z': z,
        'kT': kT,
    }

# -----------------------------------------------------------------------------
# Target tensors
# -----------------------------------------------------------------------------

def _clip_idx(arr: np.ndarray, max_jet: int) -> np.ndarray:
    out = arr.copy()
    bad = (out < 0) | (out >= max_jet)
    out[bad] = -1
    return out.astype(np.int32, copy=False)


def prepare_Target(tbl: ak.Array, max_jet: int = 9) -> Dict[str, np.ndarray]:
    N = len(tbl["genTtbarId"])  # type: ignore[index]

    hp = ak.fill_none(ak.pad_none(tbl["Index_Hard_Process"], 4, clip=True), -1)
    hp_np = ak.to_numpy(hp).astype(np.int32, copy=False)
    decay_mode = tbl["decay_mode"]

    hb_, w1_, w2_, lb_ = hp_np[:, 0], hp_np[:, 1], hp_np[:, 2], hp_np[:, 3]
    hb, w1, w2, lb = (_clip_idx(x, max_jet) for x in (hb_, w1_, w2_, lb_))

    dm = ak.to_numpy(decay_mode)
    m_45 = (dm == 45)
    m_4x = np.isin(dm, [43, 41])
    m_2x = np.isin(dm, [23, 21])

    def _take_or_neg1(mask: np.ndarray, vals: np.ndarray) -> np.ndarray:
        out = np.full(vals.shape, -1, dtype=np.int32)
        out[mask] = vals[mask]
        return out

    w_45_1 = _take_or_neg1(m_45, w1); w_45_2 = _take_or_neg1(m_45, w2)
    w_4_1  = _take_or_neg1(m_4x, w1); w_4_2  = _take_or_neg1(m_4x, w2)
    w_2_1  = _take_or_neg1(m_2x, w1); w_2_2  = _take_or_neg1(m_2x, w2)

    try:
        origin  = tbl["Gen_HF_Origin"]
        flavour = tbl["Gen_HF_Flavour"]
    except:
        origin  = ak.full_like(tbl["Jet_Pt"], -1, dtype=np.int32)
        flavour = ak.full_like(tbl["Jet_Pt"], -1, dtype=np.int32)

    abs_origin  = ak.where(origin  < 0, -origin,  origin)
    abs_flavour = ak.where(flavour < 0, -flavour, flavour)

    mask_common = (abs_origin == 21)
    mask_b = (abs_flavour == 5) & mask_common
    mask_c = (abs_flavour == 4) & mask_common

    loc_idx = ak.local_index(origin, axis=1)

    def first_two_to_np(idx_list: ak.Array) -> np.ndarray:
        pad2 = ak.pad_none(idx_list, 2, clip=True)
        arr2 = ak.to_numpy(ak.fill_none(pad2, -1)).astype(np.int32, copy=False)
        return arr2

    b12 = first_two_to_np(loc_idx[mask_b])
    c12 = first_two_to_np(loc_idx[mask_c])

    b12 = np.where((b12 >= 0) & (b12 < max_jet), b12, -1).astype(np.int32, copy=False)
    c12 = np.where((c12 >= 0) & (c12 < max_jet), c12, -1).astype(np.int32, copy=False)

    gent = (ak.to_numpy(tbl["genTtbarId"]).astype(np.int32)) % 100

    B_ONE = np.isin(gent, [51, 52])
    B_TWO = np.isin(gent, [53, 54, 55])
    C_ONE = np.isin(gent, [41, 42])
    C_TWO = np.isin(gent, [43, 44, 45])

    gbb1 = np.full(N, -1, dtype=np.int32); gbb2 = np.full(N, -1, dtype=np.int32)
    gcc1 = np.full(N, -1, dtype=np.int32); gcc2 = np.full(N, -1, dtype=np.int32)
    gqq1 = np.full(N, -1, dtype=np.int32); gqq2 = np.full(N, -1, dtype=np.int32)

    sel_b = B_ONE | B_TWO
    gbb1[sel_b] = b12[sel_b, 0]
    gbb2[B_TWO] = b12[B_TWO, 1]
    gqq1[B_ONE] = b12[B_ONE, 0]; gqq2[B_ONE] = -1
    gqq1[B_TWO] = b12[B_TWO, 0]; gqq2[B_TWO] = b12[B_TWO, 1]

    sel_c = C_ONE | C_TWO
    gcc1[sel_c] = c12[sel_c, 0]
    gcc2[C_TWO] = c12[C_TWO, 1]
    gqq1[C_ONE] = c12[C_ONE, 0]; gqq2[C_ONE] = -1
    gqq1[C_TWO] = c12[C_TWO, 0]; gqq2[C_TWO] = c12[C_TWO, 1]

    # Optional duplicate check (commented to keep stdout clean)
    # stack6 = np.stack([lb, hb, w1, w2, gqq1, gqq2], axis=1)
    # dup = np.any(stack6 >= 0, axis=1) & (
    #       (stack6[:, :4] >= 0).sum(axis=1) != np.unique(stack6[:, :4], axis=1).shape[1])

    return {
        "lb":  lb, "hb": hb, "w1": w1, "w2": w2,
        "w_45_1": w_45_1, "w_45_2": w_45_2,
        "w_4_1":  w_4_1,  "w_4_2":  w_4_2,
        "w_2_1":  w_2_1,  "w_2_2":  w_2_2,
        "gqq1": gqq1, "gqq2": gqq2,
        "gcc1": gcc1, "gcc2": gcc2,
        "gbb1": gbb1, "gbb2": gbb2,
    }

# -----------------------------------------------------------------------------
# HDF5 saving
# -----------------------------------------------------------------------------

def save_dataset(path: Path,
                 mask: np.ndarray,
                 fold: np.ndarray,
                 run: np.ndarray,
                 lumi: np.ndarray,
                 event: np.ndarray,
                 Momenta: Mapping[str, np.ndarray],
                 edges: Mapping[str, np.ndarray],
                 Met: Mapping[str, np.ndarray],
                 Lepton: Mapping[str, np.ndarray],
                 TARGETS: Mapping[str, np.ndarray],
                 CLASSIFICATIONS: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as h5:
        h5.create_group('DATAINFO')
        h5.create_dataset('DATAINFO/fold',  data=fold[mask].astype(np.uint32))
        h5.create_dataset('DATAINFO/run',   data=run[mask].astype(np.uint32))
        h5.create_dataset('DATAINFO/event', data=event[mask].astype(np.uint32))
        h5.create_dataset('DATAINFO/lumi',  data=lumi[mask].astype(np.uint32))

        h5.create_group('INPUTS'); h5.create_group('TARGETS')
        h5.create_group('INPUTS/Momenta')
        h5.create_dataset('INPUTS/Momenta/MASK',     data=Momenta['Mask'][mask].astype(bool))
        h5.create_dataset('INPUTS/Momenta/pt',       data=Momenta['pt'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Momenta/eta',      data=Momenta['eta'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Momenta/cos_phi',  data=Momenta['cos_phi'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Momenta/sin_phi',  data=Momenta['sin_phi'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Momenta/mass',     data=Momenta['mass'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Momenta/qtag',     data=Momenta['qtag'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Momenta/btag',     data=Momenta['btag'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Momenta/cvsl',     data=Momenta['cvsl'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Momenta/cvsb',     data=Momenta['cvsb'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Momenta/Era',      data=Momenta['Era'][mask].astype(np.float32))

        h5.create_group('INPUTS/EDGE')
        h5.create_dataset('INPUTS/EDGE/MASK',   data=edges['mask'][mask].astype(bool))
        h5.create_dataset('INPUTS/EDGE/deltaR', data=edges['deltaR'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/EDGE/invM',   data=edges['invM'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/EDGE/z',      data=edges['z'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/EDGE/kT',     data=edges['kT'][mask].astype(np.float32))

        h5.create_group('INPUTS/Met')
        h5.create_dataset('INPUTS/Met/MASK',    data=Met['MASK'][mask].astype(bool))
        h5.create_dataset('INPUTS/Met/met',     data=Met['met'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Met/cos_phi', data=Met['cos_phi'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Met/sin_phi', data=Met['sin_phi'][mask].astype(np.float32))

        h5.create_group('INPUTS/Lepton')
        h5.create_dataset('INPUTS/Lepton/pt',      data=Lepton['pt'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Lepton/eta',     data=Lepton['eta'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Lepton/cos_phi', data=Lepton['cos_phi'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Lepton/sin_phi', data=Lepton['sin_phi'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Lepton/mass',    data=Lepton['mass'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Lepton/utag',    data=Lepton['utag'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Lepton/etag',    data=Lepton['etag'][mask].astype(np.float32))
        h5.create_dataset('INPUTS/Lepton/Era',     data=Lepton['Era'][mask].astype(np.float32))

        h5.create_group('TARGETS/lt'); h5.create_group('TARGETS/ht'); h5.create_group('TARGETS/hw')
        h5.create_group('TARGETS/hw_45'); h5.create_group('TARGETS/hw_4'); h5.create_group('TARGETS/hw_2')
        h5.create_group('TARGETS/g1'); h5.create_group('TARGETS/g2')

        h5.create_dataset('TARGETS/lt/lb',    data=TARGETS['lb'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/ht/hb',    data=TARGETS['hb'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/ht/w1',    data=TARGETS['w1'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/ht/w2',    data=TARGETS['w2'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/ht/w_45_1',data=TARGETS['w_45_1'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/ht/w_45_2',data=TARGETS['w_45_2'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/ht/w_4_1', data=TARGETS['w_4_1'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/ht/w_4_2', data=TARGETS['w_4_2'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/ht/w_2_1', data=TARGETS['w_2_1'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/ht/w_2_2', data=TARGETS['w_2_2'][mask].astype(np.int64))

        h5.create_dataset('TARGETS/hw/w1',       data=TARGETS['w1'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/hw/w2',       data=TARGETS['w2'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/hw_45/w_45_1',data=TARGETS['w_45_1'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/hw_45/w_45_2',data=TARGETS['w_45_2'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/hw_4/w_4_1',  data=TARGETS['w_4_1'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/hw_4/w_4_2',  data=TARGETS['w_4_2'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/hw_2/w_2_1',  data=TARGETS['w_2_1'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/hw_2/w_2_2',  data=TARGETS['w_2_2'][mask].astype(np.int64))

        h5.create_dataset('TARGETS/g1/gqq', data=TARGETS['gqq1'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/g2/gqq', data=TARGETS['gqq2'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/g1/gcc', data=TARGETS['gcc1'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/g2/gcc', data=TARGETS['gcc2'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/g1/gbb', data=TARGETS['gbb1'][mask].astype(np.int64))
        h5.create_dataset('TARGETS/g2/gbb', data=TARGETS['gbb2'][mask].astype(np.int64))

        h5.create_group('CLASSIFICATIONS'); h5.create_group('CLASSIFICATIONS/EVENT')
        h5.create_dataset('CLASSIFICATIONS/EVENT/signal', data=CLASSIFICATIONS['signal'][mask].astype(np.int64))

# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class BuildConfig:
    run: int = 2
    eras: Tuple[str, ...] = field(default_factory=tuple)   # 비워두고 __post_init__에서 채움
    base_template: str = ""                                # 마찬가지
    nbuckets: int = 3
    max_jet: int = 9
    workers: int = 8
    lumi_map: Optional[Mapping[str, float]] = None
    outdir: Path = Path(".")
    validation_set: bool = False
    validation_fraction: float = 0.1
    

    def __post_init__(self):
        # run에 따라 eras 기본값 설정 (사용자가 직접 넘기면 그 값을 유지)
        if not self.eras:
            if self.run == 2:
                self.eras = ("2016preVFP", "2016postVFP", "2017", "2018")
                self.lumi_map = {
                    "2016preVFP": 19.5,
                    "2016postVFP": 16.8,
                    "2017": 41.5,
                    "2018": 59.7,
                }
            elif self.run == 3:
                self.eras = ("2022", "2022EE", "2023", "2023BPix")
                self.lumi_map = {
                    "2022": 7.98,
                    "2022EE": 26.67,
                    "2023": 17.79,
                    "2023BPix": 9.45,
                }
            else:
                raise ValueError(f"Unsupported run={self.run}")

        # run에 따라 base_template 기본값 설정 (사용자가 직접 넘기면 그 값을 유지)
        if not self.base_template:
            if self.run == 2:
                self.base_template = "/gv0/Users/isyoon/SKFlatOutput/Run2UltraLegacy_v3/Vcb/{era}/RunNewTrainingTree__/"
            elif self.run == 3:
                self.base_template = "/data9/Users/yeonjoon/SKNanoOutput/Vcb_SL/{ch}_Training/{era}/"
                
        if self.validation_fraction < 0.0 or self.validation_fraction > 1.0:
            raise ValueError("validation_fraction must be in [0.0, 1.0]")
                


def build_tasks(cfg: BuildConfig) -> List[Tuple[str,str,float,int,int,int,int]]:
    tasks = []
    if cfg.run == 2:
        for era in cfg.eras:
            basedir = cfg.base_template.format(era=era)
            sig = basedir + 'Vcb_TTLJ_WtoCB_powheg.root'
            bkg = basedir + 'Vcb_TTLJ_powheg.root'
            tasks += [
                (sig, "El/Training_Tree", 0.00051, 1, 0, 0, cfg.nbuckets),
                (sig, "Mu/Training_Tree", 0.10566, 0, 1, 0, cfg.nbuckets),
                (bkg, "El/Training_Tree", 0.00051, 1, 0, 1, cfg.nbuckets),
                (bkg, "Mu/Training_Tree", 0.10566, 0, 1, 1, cfg.nbuckets),
            ]
    elif cfg.run == 3:
        for era in cfg.eras:
            basedir = cfg.base_template
            sig_mu = basedir.format(era=era,ch="Mu") + 'TTLJ_Vcb_powheg.root'
            bkg_mu = basedir.format(era=era,ch="Mu") + 'TTLJ_powheg.root'
            sig_el = basedir.format(era=era,ch="El") + 'TTLJ_Vcb_powheg.root'
            bkg_el = basedir.format(era=era,ch="El") + 'TTLJ_powheg.root'
            tasks += [
                (sig_el, "Training_Tree", 0.00051, 1, 0, 0, cfg.nbuckets),
                (sig_mu, "Training_Tree", 0.10566, 0, 1, 0, cfg.nbuckets),
                (bkg_el, "Training_Tree", 0.00051, 1, 0, 1, cfg.nbuckets),
                (bkg_mu, "Training_Tree", 0.10566, 0, 1, 1, cfg.nbuckets),
            ]
    return tasks
    

def run_pipeline(cfg: BuildConfig) -> None:
    import logging, time
    from contextlib import contextmanager

    # --- logging setup (idempotent) ---
    logger = logging.getLogger("dataset_builder")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    logger.setLevel(logging.INFO)

    @contextmanager
    def step(name: str):
        t0 = time.time()
        logger.info(f"▶ {name} ...")
        try:
            yield
        except Exception:
            logger.exception(f"✖ {name} failed after {time.time() - t0:.2f}s")
            raise
        else:
            logger.info(f"✔ {name} done in {time.time() - t0:.2f}s")

    with step("compile C++ fold kernel"):
        _compile_cpp_once()

    with step("build task list"):
        tasks = build_tasks(cfg)
        logger.info(f"num tasks: {len(tasks)}")

    # load in parallel
    with step("load ROOT trees in parallel"):
        el_vcb, mu_vcb, el, mu = [], [], [], []
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            for arr, args in zip(ex.map(lambda a: load_tree_minimal(*a), tasks), tasks):
                _, tree, lep_m, is_el, is_mu, is_bkg, *_ = args
                if is_el and is_bkg == 0:
                    el_vcb.append(arr)
                elif is_mu and is_bkg == 0:
                    mu_vcb.append(arr)
                elif is_el:
                    el.append(arr)
                else:
                    mu.append(arr)

    with step("concatenate per-channel datasets"):
        rdf_Vcb_el = ak.concatenate(el_vcb, axis=0)
        rdf_Vcb_mu = ak.concatenate(mu_vcb, axis=0)
        rdf_el     = ak.concatenate(el, axis=0)
        rdf_mu     = ak.concatenate(mu, axis=0)
        logger.info(
            "counts | Vcb_el=%d, Vcb_mu=%d, el=%d, mu=%d",
            len(rdf_Vcb_el), len(rdf_Vcb_mu), len(rdf_el), len(rdf_mu)
        )

    with step("renew_y & coverage stats"):
            dfs = [rdf_mu, rdf_el, rdf_Vcb_mu, rdf_Vcb_el]
            names = ["rdf_mu", "rdf_el", "rdf_Vcb_mu", "rdf_Vcb_el"]
            for i in range(len(dfs)):
                stats = coverage_stats(dfs[i])
                if not stats["all_covered"]:
                    logger.warning("%s coverage: %s", names[i], stats)
                else:
                    logger.info("%s coverage OK: total=%d, uncovered=%d", names[i], stats["total"], stats["num_uncovered"])
                dfs[i] = renew_y(dfs[i])
            rdf_mu, rdf_el, rdf_Vcb_mu, rdf_Vcb_el = dfs
            
    if cfg.validation_set:
        with step("validation-set flag was given -> not rebalancing, only use 2018 or 2022EE"):
            selected_era_idx = 1 if cfg.run == 3 else 3
            rdf_Vcb_el = rdf_Vcb_el[rdf_Vcb_el["era_index"] == selected_era_idx]
            rdf_Vcb_mu = rdf_Vcb_mu[rdf_Vcb_mu["era_index"] == selected_era_idx]
            rdf_el     = rdf_el[rdf_el["era_index"] == selected_era_idx]
            rdf_mu     = rdf_mu[rdf_mu["era_index"] == selected_era_idx]
            
            frac = cfg.validation_fraction
            nbucks_sel = int(1/frac)
            bkg = ak.to_packed(ak.concatenate([rdf_el, rdf_mu], axis=0))
            sig = ak.to_packed(ak.concatenate([rdf_Vcb_el, rdf_Vcb_mu], axis=0))
            sel_mask = np.random.default_rng(42).integers(0, nbucks_sel, size=len(bkg)) == 0
            n_before = len(bkg)
            bkg = ak.to_packed(bkg[sel_mask])
            logger.info("selected bkg %d -> %d", n_before, len(bkg))
            data = ak.to_packed(ak.concatenate([sig, bkg], axis=0))
            logger.info("final selected | sig=%d, bkg=%d, total=%d", len(sig), len(bkg), len(data))
            
            
            
            
    else:
        with step("ratio-balanced downsample inside each (vcb/non-vcb, mu/el)"):
            lumi_arr = [cfg.lumi_map[e] for e in cfg.eras]

            sel_vcb_mu, plan_vcb_mu = balanced_equal_class_counts_by_era(
                rdf_Vcb_mu["era_index"], rdf_Vcb_mu["y"], ratio=lumi_arr, seed=42, strict=True, return_plan=True)
            sel_vcb_el, plan_vcb_el = balanced_equal_class_counts_by_era(
                rdf_Vcb_el["era_index"], rdf_Vcb_el["y"], ratio=lumi_arr, seed=42, strict=True, return_plan=True)
            sel_mu, plan_mu = balanced_equal_class_counts_by_era(
                rdf_mu["era_index"], rdf_mu["y"], ratio=lumi_arr, seed=42, strict=True, return_plan=True)
            sel_el, plan_el = balanced_equal_class_counts_by_era(
                rdf_el["era_index"], rdf_el["y"], ratio=lumi_arr, seed=42, strict=True, return_plan=True)

            logger.info(
                "selected (vcb_mu=%d, vcb_el=%d, mu=%d, el=%d), T_targets=(%d,%d,%d,%d)",
                len(sel_vcb_mu), len(sel_vcb_el), len(sel_mu), len(sel_el),
                plan_vcb_mu.get("T_target", -1), plan_vcb_el.get("T_target", -1),
                plan_mu.get("T_target", -1), plan_el.get("T_target", -1)
            )

            rdf_Vcb_mu = ak.to_packed(rdf_Vcb_mu[sel_vcb_mu])
            rdf_Vcb_el = ak.to_packed(rdf_Vcb_el[sel_vcb_el])
            rdf_mu     = ak.to_packed(rdf_mu[sel_mu])
            rdf_el     = ak.to_packed(rdf_el[sel_el])

        with step("merge mu/el and second-stage balancing"):
            mu = ak.to_packed(ak.concatenate([rdf_Vcb_mu, rdf_mu], axis=0))
            el = ak.to_packed(ak.concatenate([rdf_Vcb_el, rdf_el], axis=0))

            sel_mu, plan_mu2 = balanced_equal_class_counts_by_era(
                mu["era_index"], mu["y"], ratio=lumi_arr, seed=42, strict=True, return_plan=True)
            sel_el, plan_el2 = balanced_equal_class_counts_by_era(
                el["era_index"], el["y"], ratio=lumi_arr, seed=42, strict=True, return_plan=True)

            mu = ak.to_packed(mu[sel_mu])
            el = ak.to_packed(el[sel_el])
            data = ak.to_packed(ak.concatenate([mu, el]))
            logger.info(
                "final selected | mu=%d (T=%d), el=%d (T=%d), total=%d",
                len(mu), plan_mu2.get("T_target", -1), len(el), plan_el2.get("T_target", -1), len(data)
            )

    with step("tensor materialization"):
        fold  = ak.to_numpy(data['fold'])
        run   = ak.to_numpy(data['run'])
        event = ak.to_numpy(data['event'])
        lumi  = ak.to_numpy(data['lumi'])
        
    
        with step("Classifications"):
            CLASSIFICATIONS = {'signal': ak.to_numpy(data['y'])}
        with step("Lepton"):
            Lepton  = prepare_Lepton(data)
        with step("Met"):
            Met     = prepare_MET(data)
        with step("Jets"):
            Momenta = prepare_Jets(data, max_jet=cfg.max_jet)
        with step("Targets"):
            TARGETS = prepare_Target(data, max_jet=cfg.max_jet)
        with step("Edges"):
            Edges   = prepare_Edges(Momenta, event_mask=np.ones(len(data), dtype=bool))
        logger.info("tensors: fold uniq=%s", np.unique(fold))
    with step("write HDF5 per fold"):
        fold_values = np.unique(fold)
        for k in fold_values:
            mask = (fold == k)
            if cfg.run == 2:
                out = cfg.outdir / f"inclusive_RunII_fold{k}.h5"
            else:
                out = cfg.outdir / f"inclusive_RunIII_fold{k}.h5"
            
            if cfg.validation_set:
                out = out.with_name(out.stem + "_valset" + out.suffix)
                
            save_dataset(out, mask, fold, run, lumi, event, Momenta, Edges, Met, Lepton, TARGETS, CLASSIFICATIONS)
            logger.info("[WRITE] %s :: %d events", out, int(mask.sum()))

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> BuildConfig:
    ap = argparse.ArgumentParser(description="Refactored Run-II dataset builder")
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nbuckets", type=int, default=3)
    ap.add_argument("--max-jet", type=int, default=9)
    ap.add_argument("--run", type=int, default=2, help="Run number (2 or 3)")
    ap.add_argument("--validation-set", action="store_true", help="Create validation set instead of training set")
    ap.add_argument("--validation-fraction", type=float, default=0.1, help="Fraction of validation set (if --validation-set is given)")
    args = ap.parse_args()

    return BuildConfig(workers=args.workers,
                       nbuckets=args.nbuckets,
                       max_jet=args.max_jet,
                       outdir=args.outdir,
                       run=args.run,
                       validation_set=args.validation_set,
                       validation_fraction=args.validation_fraction
                       )


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
