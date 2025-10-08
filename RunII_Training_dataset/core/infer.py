"""
infer.py — Reusable ONNXRuntime inference utilities for SPANet HDF5 datasets.

Public API
----------
- load_session(onnx_path, providers=None) -> ort.InferenceSession
- infer_h5(file_path, session, batch_size=32768, use_tqdm=True) -> dict[str, np.ndarray]
- split_by_class(infer_dict, class_key='real_class', signal_value=1) -> dict[str, dict[str, np.ndarray]]
- count_n_b_tagged_jets(btag, era, jet_mask=None) -> np.ndarray[int32]

Notes
-----
- Outputs are returned with keys identical to the ONNX model's output names.
- Common side info (real_class, real_target, n_jet, n_b_tagged, Era, predict_class if present) are included.
- Presence-optional heads(e.g., g1_*)는 모델의 출력 유무에 따라 자동 처리됩니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple, Optional, Dict, List
import logging
import numpy as np
import h5py
import onnxruntime as ort
import correctionlib
import os

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

__all__ = [
    "load_session",
    "infer_h5",
    "split_by_class",
    "count_n_b_tagged_jets",
]

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# ONNX Session
# -----------------------------------------------------------------------------
def load_session(
    onnx_path: str,
    providers: Optional[Sequence[str]] = None,
    intra_op_num_threads: Optional[int] = None,
    inter_op_num_threads: Optional[int] = None,
) -> ort.InferenceSession:
    """
    Create an ONNX Runtime InferenceSession.

    Parameters
    ----------
    onnx_path : str
        Path to .onnx model.
    providers : list[str] | None
        e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider'].
    intra_op_num_threads, inter_op_num_threads : int | None
        Threading knobs (optional).

    Returns
    -------
    onnxruntime.InferenceSession
    """
    sess_opts = ort.SessionOptions()
    if intra_op_num_threads is not None:
        sess_opts.intra_op_num_threads = int(intra_op_num_threads)
    if inter_op_num_threads is not None:
        sess_opts.inter_op_num_threads = int(inter_op_num_threads)

    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=list(providers))
    logger.info("Loaded ONNX model: %s", onnx_path)
    logger.debug("Inputs:  %s", [i.name for i in session.get_inputs()])
    logger.debug("Outputs: %s", [o.name for o in session.get_outputs()])
    return session

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _ensure_1d_era(era: np.ndarray) -> np.ndarray:
    if era.ndim == 2:
        era = era[:, 0]
    elif era.ndim != 1:
        raise ValueError(f"era must be (N,) or (N,J); got shape {era.shape}")
    return era.astype(np.int32, copy=False)

def _era_index_to_ul_dir(idx: int) -> str:
    """0..3 index -> jsonpog-integration UL 디렉토리명"""
    mapping = {
        0: "2016preVFP_UL",
        1: "2016postVFP_UL",
        2: "2017_UL",
        3: "2018_UL",
    }
    try:
        return mapping[idx]
    except KeyError:
        raise ValueError(f"Unsupported era index '{idx}'. Expected one of {list(mapping.keys())}.")

def _load_wp_threshold(repo_path: str, era_dir: str, corr_name: str, wp_label: str) -> float:
    """
    해당 era의 btagging.json.gz에서 corr_name('deepJet_wp_values')을 열어
    wp_label('L'|'M'|'T') 임계값(float)을 반환.
    """
    jpath = os.path.join(repo_path, era_dir, "btagging.json.gz")
    if not os.path.isfile(jpath):
        raise FileNotFoundError(f"Missing correction file: {jpath}")
    cset = correctionlib.CorrectionSet.from_file(jpath)
    if corr_name not in cset:
        raise KeyError(f"Correction '{corr_name}' not found in {jpath}. "
                       f"Available: {list(cset.keys())}")
    # jsonpog의 BTV deepJet_wp_values는 보통 입력이 ('L'|'M'|'T') 하나.
    thr = float(cset[corr_name].evaluate(wp_label))
    return thr

def count_n_b_tagged_jets(
    btag: np.ndarray,   # (N, J) float
    era: np.ndarray,    # (N,) or (N, J) int (0..3)
    era_map: Mapping[str, int] = {'2016preVFP': 0, '2016postVFP': 1, '2017': 2, '2018': 3},
    repo_path: Optional[str] = '/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV',
    jet_mask: Optional[np.ndarray] = None,  # (N, J) bool; True for valid jets only
    wp: str = "M",                          # 'L' | 'M' | 'T'
    corr_name: str = "deepJet_wp_values",   # 보통 이 이름을 사용
) -> np.ndarray:
    """
    Return (N,) int32 — era-dependent deepJet WP('L'|'M'|'T')로 b-tag된 jet 수.
    correctionlib로 /cvmfs 아래 JSON을 읽어 실제 공식 임계값을 사용.

    Parameters
    ----------
    btag : (N, J) float
        per-jet deepJet discriminator (e.g., 'btagDeepFlavB').
    era : (N,) or (N, J) int in {0,1,2,3}
        0: 2016preVFP, 1: 2016postVFP, 2: 2017, 3: 2018
        (N,J)인 경우 첫 열을 사용.
    repo_path : str
        POG/BTV 루트 경로. 기본값은 /cvmfs jsonpog-integration.
    jet_mask : (N, J) bool or None
        True = 유효 jet만 집계.
    wp : str
        'L'|'M'|'T'
    corr_name : str
        보통 'deepJet_wp_values'.

    Returns
    -------
    (N,) np.int32
    """
    if repo_path is None:
        raise ValueError("repo_path must point to POG/BTV root (jsonpog-integration).")
    wp = wp.upper()
    if wp not in ("L", "M", "T"):
        raise ValueError(f"wp must be one of 'L','M','T', got '{wp}'")

    btag = np.asarray(btag)
    if btag.ndim != 2:
        raise ValueError(f"btag must be (N,J), got shape={btag.shape}")
    N, J = btag.shape

    era1d = _ensure_1d_era(era)
    if era1d.shape[0] != N:
        raise ValueError(f"era length {era1d.shape[0]} != N {N}")

    if jet_mask is not None:
        jet_mask = np.asarray(jet_mask, dtype=bool)
        if jet_mask.shape != (N, J):
            raise ValueError(f"jet_mask must be (N,J) = {(N,J)}, got {jet_mask.shape}")

    # era별 임계값을 한 번씩만 읽어서 캐시
    unique_eras = np.unique(era1d)
    era_thr: Dict[int, float] = {}
    for e in unique_eras.tolist():
        era_dir = _era_index_to_ul_dir(int(e))
        thr = _load_wp_threshold(repo_path, era_dir, corr_name, wp)
        era_thr[int(e)] = thr

    # 각 이벤트에 맞는 threshold 벡터 (N,1)
    th = np.array([era_thr[int(e)] for e in era1d], dtype=np.float32)[:, None]

    comp = (btag > th)  # (N,J) bool
    if jet_mask is not None:
        comp &= jet_mask

    return comp.sum(axis=1).astype(np.int32, copy=False)

@dataclass(frozen=True)
class _H5Paths:
    # input branch layout (필요 시 수정)
    mom_mask: str = "INPUTS/Momenta/MASK"
    mom: Mapping[str, str] = None
    met: Mapping[str, str] = None
    lep: Mapping[str, str] = None
    cls_signal: str = "CLASSIFICATIONS/EVENT/signal"
    targets: Mapping[str, str] = None

    def __init__(self):
        object.__setattr__(self, "mom", {
            "pt": "INPUTS/Momenta/pt",
            "eta": "INPUTS/Momenta/eta",
            "sin_phi": "INPUTS/Momenta/sin_phi",
            "cos_phi": "INPUTS/Momenta/cos_phi",
            "mass": "INPUTS/Momenta/mass",
            "btag": "INPUTS/Momenta/btag",
            "cvsl": "INPUTS/Momenta/cvsl",
            "cvsb": "INPUTS/Momenta/cvsb",
            "Era": "INPUTS/Momenta/Era",
        })
        object.__setattr__(self, "met", {
            "met": "INPUTS/Met/met",
            "sin_phi": "INPUTS/Met/sin_phi",
            "cos_phi": "INPUTS/Met/cos_phi",
        })
        object.__setattr__(self, "lep", {
            "pt": "INPUTS/Lepton/pt",
            "eta": "INPUTS/Lepton/eta",
            "sin_phi": "INPUTS/Lepton/sin_phi",
            "cos_phi": "INPUTS/Lepton/cos_phi",
            "mass": "INPUTS/Lepton/mass",
            "utag": "INPUTS/Lepton/utag",
            "etag": "INPUTS/Lepton/etag",
            "Era": "INPUTS/Lepton/Era",
        })
        object.__setattr__(self, "targets", {
            "lt_lb": "TARGETS/lt/lb",
            "ht_hb": "TARGETS/ht/hb",
            "hw_w1": "TARGETS/hw/w1",
            "hw_w2": "TARGETS/hw/w2",
            "hw_45_w_1": "TARGETS/hw_45/w_45_1",
            "hw_45_w_2": "TARGETS/hw_45/w_45_2",
            "hw_4_w_1": "TARGETS/hw_4/w_4_1",
            "hw_4_w_2": "TARGETS/hw_4/w_4_2",
            "hw_2_w_1": "TARGETS/hw_2/w_2_1",
            "hw_2_w_2": "TARGETS/hw_2/w_2_2"
        })
        


def _stack_last_axis(dset_map: Mapping[str, h5py.Dataset], idx: Sequence[int]) -> np.ndarray:
    arrs = [np.asarray(dset_map[k][idx]) for k in dset_map]
    x = np.stack(arrs, axis=-1).astype(np.float32, copy=False)
    return x

def _to_bool(x: np.ndarray) -> np.ndarray:
    return x.astype(bool, copy=False)

def _iter_batches(n: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        yield s, e

# -----------------------------------------------------------------------------
# Core inference
# -----------------------------------------------------------------------------
def infer_h5(
    file_path: str,
    session: ort.InferenceSession,
    batch_size: int = 32768,
    use_tqdm: bool = True,
    include_ctag: bool = False,
    reduction_factor: Optional[int] = None,
    reduction_mode: str = "head",   # 'head' | 'stride' | 'random'
    random_seed: int = 123,
) -> Dict[str, np.ndarray]:
    P = _H5Paths()
    with h5py.File(file_path, "r") as f:
        nentry = len(f[P.mom_mask])

        # ----- decide which indices to process -----
        if reduction_factor is None or reduction_factor <= 1:
            sel_idx = np.arange(nentry, dtype=np.int64)
        else:
            if reduction_mode == "head":
                nproc = max(0, nentry // reduction_factor)
                sel_idx = np.arange(nproc, dtype=np.int64)
            elif reduction_mode == "stride":
                sel_idx = np.arange(0, nentry, reduction_factor, dtype=np.int64)
            elif reduction_mode == "random":
                rng = np.random.default_rng(random_seed)
                nproc = max(0, nentry // reduction_factor)
                sel_idx = rng.choice(nentry, size=nproc, replace=False)
                sel_idx.sort()  # I/O locality
            else:
                raise ValueError("reduction_mode must be 'head'|'stride'|'random'")

        nproc = int(sel_idx.size)
        if nproc == 0:
            logger.warning("No entries selected (nentry=%d, reduction_factor=%s). Returning empty result.",
                           nentry, str(reduction_factor))
            return {}

        logger.info("Processing %d/%d entries from %s (mode=%s, factor=%s)",
                    nproc, nentry, file_path, reduction_mode, str(reduction_factor))

        output_names = [o.name for o in session.get_outputs()]
        out_collect: Dict[str, List[np.ndarray]] = {name: [] for name in output_names}

        # allocate side-info with EXACT processed length
        real_class_all = np.empty((nproc,), dtype=f[P.cls_signal].dtype)
        real_target_all = np.empty((nproc, 8), dtype=np.int64)
        n_jet_all = np.empty((nproc,), dtype=np.int32)
        n_btag_all = np.empty((nproc,), dtype=np.int32)
        era_all = np.empty((nproc,), dtype=np.int32)

        # datasets
        mom_mask_ds = f[P.mom_mask]
        mom_ds = {k: f[v] for k, v in P.mom.items()}
        met_ds = {k: f[v] for k, v in P.met.items()}
        lep_ds = {k: f[v] for k, v in P.lep.items()}
        tgt_ds = {k: f[v] for k, v in P.targets.items()}
        cls_signal_ds = f[P.cls_signal]

        # batching over selected indices (use integer indexing for fancy selection)
        iterator = _iter_batches(nproc, int(batch_size))
        if use_tqdm and _tqdm is not None:
            iterator = _tqdm(iterator, total=(nproc + batch_size - 1) // batch_size, desc="Infer")

        for s, e in iterator:
            idx = sel_idx[s:e]  # (B,)

            # Inputs (advanced indexing)
            mom_mask = _to_bool(np.asarray(mom_mask_ds[idx]))
            mom_fields = [
                np.asarray(mom_ds["pt"][idx]),
                np.asarray(mom_ds["eta"][idx]),
                np.asarray(mom_ds["sin_phi"][idx]),
                np.asarray(mom_ds["cos_phi"][idx]),
                np.asarray(mom_ds["mass"][idx]),
                np.asarray(mom_ds["btag"][idx]),
            ]
            if include_ctag:
                mom_fields += [
                    np.asarray(mom_ds["cvsl"][idx]),
                    np.asarray(mom_ds["cvsb"][idx]),
                ]
            mom_fields += [np.asarray(mom_ds["Era"][idx])]
            mom_data = np.stack(mom_fields, axis=-1).astype(np.float32, copy=False)

            n_jet = mom_mask.sum(axis=1).astype(np.int32, copy=False)

            met_data = np.stack(
                [
                    np.asarray(met_ds["met"][idx]),
                    np.asarray(met_ds["sin_phi"][idx]),
                    np.asarray(met_ds["cos_phi"][idx]),
                ],
                axis=-1,
            ).astype(np.float32, copy=False).reshape(len(idx), 1, 3)
            met_mask = np.ones((len(idx), 1), dtype=bool)

            lep_data = np.stack(
                [
                    np.asarray(lep_ds["pt"][idx]),
                    np.asarray(lep_ds["eta"][idx]),
                    np.asarray(lep_ds["sin_phi"][idx]),
                    np.asarray(lep_ds["cos_phi"][idx]),
                    np.asarray(lep_ds["mass"][idx]),
                    np.asarray(lep_ds["utag"][idx]),
                    np.asarray(lep_ds["etag"][idx]),
                    np.asarray(lep_ds["Era"][idx]),
                ],
                axis=-1,
            ).astype(np.float32, copy=False).reshape(len(idx), 1, 8)
            lep_mask = np.ones((len(idx), 1), dtype=bool)

            # Derived
            n_btag = count_n_b_tagged_jets(
                btag=np.asarray(mom_ds["btag"][idx]),
                era=np.asarray(mom_ds["Era"][idx]),
                jet_mask=mom_mask,
            )
            era = _ensure_1d_era(np.asarray(lep_ds["Era"][idx]))

            # Run model
            feeds = {
                "Momenta_data": mom_data,
                "Momenta_mask": mom_mask,
                "Met_data": met_data,
                "Met_mask": met_mask,
                "Lepton_data": lep_data,
                "Lepton_mask": lep_mask,
            }
            outs = session.run(output_names, feeds)
            for name, val in zip(output_names, outs):
                out_collect[name].append(val)

            # Side info
            real_class = np.asarray(cls_signal_ds[idx])
            real_target = np.stack(
                [
                    np.asarray(tgt_ds["lt_lb"][idx]),
                    np.asarray(tgt_ds["ht_hb"][idx]),
                    np.asarray(tgt_ds["hw_2_w_1"][idx]),
                    np.asarray(tgt_ds["hw_2_w_2"][idx]),
                    np.asarray(tgt_ds["hw_4_w_1"][idx]),
                    np.asarray(tgt_ds["hw_4_w_2"][idx]),
                    np.asarray(tgt_ds["hw_45_w_1"][idx]),
                    np.asarray(tgt_ds["hw_45_w_2"][idx]),
                ],
                axis=-1,
            ).astype(np.int64, copy=False)

            real_class_all[s:e] = real_class
            real_target_all[s:e] = real_target
            n_jet_all[s:e] = n_jet
            n_btag_all[s:e] = n_btag
            era_all[s:e] = era

        # Concatenate model outputs
        result: Dict[str, np.ndarray] = {}
        for name, chunks in out_collect.items():
            result[name] = np.concatenate(chunks, axis=0) if chunks else np.empty((0,))

        # Predict class alias if present
        if "EVENT/signal" in result:
            result["predict_class"] = result["EVENT/signal"]

        # Side info (length == nproc)
        result["real_class"] = real_class_all
        result["real_target"] = real_target_all
        result["n_jet"] = n_jet_all
        result["n_b_tagged"] = n_btag_all
        result["Era"] = era_all

        # 최종 길이 체크(안전망)
        for k, v in result.items():
            if isinstance(v, np.ndarray) and v.shape[0] != nproc:
                raise RuntimeError(f"Length mismatch for key '{k}': {v.shape[0]} vs {nproc}")

        return result

# -----------------------------------------------------------------------------
# Post-processing helpers
# -----------------------------------------------------------------------------
def split_by_class(
    infer_dict: Dict[str, np.ndarray],
    class_key: str = "real_class",
    signal_value: int | float = 1,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Split arrays into 'Vcb' (signal_value) and 'TTHF' (~signal_value) partitions.
    """
    if class_key not in infer_dict:
        raise KeyError(f"'{class_key}' not found in infer_dict keys: {list(infer_dict.keys())}")

    mask = np.asarray(infer_dict[class_key]) == signal_value
    out = {"Vcb": {}, "TTHF": {}}
    for k, v in infer_dict.items():
        if not isinstance(v, np.ndarray):
            continue
        out["Vcb"][k] = v[mask]
        out["TTHF"][k] = v[~mask]
    return out

# -----------------------------------------------------------------------------
# CLI / example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Run SPANet ONNX inference on HDF5.")
    parser.add_argument("--onnx", required=True, help="Path to .onnx model")
    parser.add_argument("--h5", required=True, help="Path to input .h5")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar")
    args = parser.parse_args()

    sess = load_session(args.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    res = infer_h5(args.h5, sess, batch_size=args.batch_size, use_tqdm=not args.no_tqdm)

    # Optional split
    parts = split_by_class(res, class_key="real_class", signal_value=1)

    # Minimal summary print (safe for CLI; silent when imported)
    print(f"[OK] Inferred: {args.h5}")
    print("Keys:", list(res.keys()))
    print("Vcb/TTHF sizes:", len(parts["Vcb"]["real_class"]), len(parts["TTHF"]["real_class"]))