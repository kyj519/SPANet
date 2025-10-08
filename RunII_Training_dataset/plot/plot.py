from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Sequence, Mapping, Any, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os

# ---------------------------
# Stats helpers
# ---------------------------
def _wilson_interval(success, total, z: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wilson score interval. total=0 → (nan, nan).
    z=1.0 ≈ 68% 구간, z=1.96 ≈ 95% 구간.
    """
    success = np.asarray(success, dtype=float)
    total   = np.asarray(total, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.divide(success, total, out=np.full_like(total, np.nan), where=total > 0)

        denom  = 1.0 + (z**2)/total
        center = (p + (z**2)/(2*total)) / denom
        term   = np.divide(p*(1.0 - p), total, out=np.zeros_like(total), where=total > 0)
        half   = z * np.sqrt(term + (z**2)/(4*total**2)) / denom

        lo = center - half
        hi = center + half
    return lo, hi


# ---------------------------
# Core
# ---------------------------
import numpy as np
from typing import Any, Mapping, Sequence, Dict

# NOTE: assumes _wilson_interval(success, total, z) is defined elsewhere.

def get_assignment_accuracy(
    n_jet: np.ndarray,
    n_b_tagged: np.ndarray,
    infer_assignments: np.ndarray,
    real_target: np.ndarray,
    real_target_idx: Sequence[int],
    *,
    z: float = 1.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Vcb 이벤트에서 (n_jet, n_b_tagged) 셀별 W 페어(w1,w2) 배정 정확도(순서 무시)와 통계 오차 계산.

    Parameters
    ----------
    n_jet : (N,) int array
    n_b_tagged : (N,) int array
    infer_assignments : (N, K>=2) int array
        마지막 2컬럼이 (w1, w2) 추론 결과라고 가정.
    real_target : (N, M) int array
        실측 타겟 (여러 컬럼 중 real_target_idx가 (w1, w2) 위치를 가리킴).
    real_target_idx : Sequence[int] 길이 2
        real_target에서 (w1, w2)에 해당하는 2개 컬럼 인덱스 (예: (2,3)).
    z : float, default=1.0
        Wilson score interval의 z-score (1.0 ≈ 68% 신뢰구간).
    verbose : bool, default=False
        셀/전체 로그 출력 여부.

    Returns
    -------
    dict
        keys:
          accuracy, stderr, wilson_lo, wilson_hi, counts,
          n_jets_unique, n_b_jets_unique,
          overall={accuracy, stderr, wilson_lo, wilson_hi, N}
    """
    # ---- shape & key checks
    n_jet   = np.asarray(n_jet)
    n_bjet  = np.asarray(n_b_tagged)
    ia      = np.asarray(infer_assignments)
    rt_full = np.asarray(real_target)

    if n_jet.ndim != 1 or n_bjet.ndim != 1 or n_jet.shape[0] != n_bjet.shape[0]:
        raise ValueError("n_jet and n_b_tagged must be 1D with same length N.")
    N = n_jet.shape[0]
    if ia.ndim != 2 or ia.shape[0] != N or ia.shape[1] < 2:
        raise ValueError("infer_assignments must be shape (N, K>=2).")
    if rt_full.ndim != 2 or rt_full.shape[0] != N:
        raise ValueError("real_target must be shape (N, M).")

    if len(real_target_idx) != 2:
        raise ValueError("real_target_idx must have length 2 (indices of (w1,w2)).")
    rt_cols = (int(real_target_idx[0]), int(real_target_idx[1]))

    # baseline: real_target의 (w1,w2) 두 인덱스가 음수가 아닌 이벤트만 사용
    rt_sel = rt_full[:, rt_cols]  # (N,2)
    mask_pass_baseline = np.all(rt_sel >= 0, axis=1)

    if verbose:
        frac = 100.0 * np.sum(mask_pass_baseline) / N if N > 0 else 0.0
        print(f"{frac:.2f}% of events have all 4 jets that pass baseline cuts")

    # 고유 (njet, nb) 목록
    nj_u   = np.unique(n_jet)
    nb_u   = np.unique(n_bjet)
    J, B   = len(nj_u), len(nb_u)
    nj_idx = {int(v): i for i, v in enumerate(nj_u)}
    nb_idx = {int(v): i for i, v in enumerate(nb_u)}

    # 결과 그리드
    acc = np.full((J, B), np.nan, dtype=np.float32)
    se  = np.full((J, B), np.nan, dtype=np.float32)
    cnt = np.zeros((J, B), dtype=np.int32)
    wlo = np.full((J, B), np.nan, dtype=np.float32)
    whi = np.full((J, B), np.nan, dtype=np.float32)

    # 예측 (w1,w2): infer_assignments의 마지막 2컬럼
    ia_w = ia[:, -2:]  # (N,2)

    # 셀 단위 계산
    for nj in nj_u:
        for nb in nb_u:
            m = (mask_pass_baseline & (n_jet == nj) & (n_bjet == nb))
            N_cell = int(np.sum(m))
            if N_cell == 0:
                continue

            ia_cell = ia_w[m]   # (Nc,2)
            rt_cell = rt_sel[m] # (Nc,2)  ← baseline에서 음수 없음

            # 순서 무시 정확도: (w1,w2) == (t1,t2) or (t2,t1)
            exact   = np.all(ia_cell == rt_cell, axis=1)
            swapped = np.all(ia_cell == rt_cell[:, ::-1], axis=1)
            s = int(np.sum(exact | swapped))

            p = s / N_cell
            se_ = float(np.sqrt(max(p * (1.0 - p) / N_cell, 0.0)))
            lo, hi = _wilson_interval(s, N_cell, z=z)

            i, j = nj_idx[int(nj)], nb_idx[int(nb)]
            acc[i, j] = float(p)
            se [i, j] = se_
            cnt[i, j] = N_cell
            wlo[i, j] = float(lo)
            whi[i, j] = float(hi)

            if verbose:
                print(
                    f"n_jet={int(nj)}, n_b_jet={int(nb)} → "
                    f"acc={p:.4f} ± {se_:.4f} (N={N_cell}, Wilson[{lo:.4f}, {hi:.4f}])"
                )

    # Overall (baseline 통과 전체)
    m_all = mask_pass_baseline
    N_all = int(np.sum(m_all))
    if N_all > 0:
        ia_a = ia_w[m_all]
        rt_a = rt_sel[m_all]
        exact   = np.all(ia_a == rt_a, axis=1)
        swapped = np.all(ia_a == rt_a[:, ::-1], axis=1)
        s_all = int(np.sum(exact | swapped))
        p_all = s_all / N_all
        stderr_all = float(np.sqrt(max(p_all * (1.0 - p_all) / N_all, 0.0)))
        wlo_all, whi_all = _wilson_interval(s_all, N_all, z=z)
        if verbose:
            print(
                f"Overall accuracy (Vcb): {p_all:.4f} ± {stderr_all:.4f} "
                f"(N={N_all}, Wilson[{wlo_all:.4f}, {whi_all:.4f}])"
            )
    else:
        p_all = np.nan
        stderr_all = np.nan
        wlo_all = np.nan
        whi_all = np.nan

    return {
        'accuracy'        : acc,
        'stderr'          : se,
        'wilson_lo'       : wlo,
        'wilson_hi'       : whi,
        'counts'          : cnt,
        'n_jets_unique'   : nj_u,
        'n_b_jets_unique' : nb_u,
        'overall': {
            'accuracy' : float(p_all),
            'stderr'   : float(stderr_all),
            'wilson_lo': float(wlo_all),
            'wilson_hi': float(whi_all),
            'N'        : int(N_all),
        }
    }

def _extract_series(
    res: Mapping[str, Any],
    jets: Iterable[int] = (4, 5, 6, 7),
    *,
    min_nb: int = 2,
    include_overall: bool = True,
) -> Dict[Tuple[int, int], Tuple[float, float, int]]:
    """
    get_assignment_accuracy() 반환값에서
    (njet in jets, nb in [min_nb .. njet]) 셀만 추려 (p, stderr, N)을 dict로 구성.

    Returns
    -------
    dict:
      key=(njet, nb) → value=(p, stderr, N)
      (선택) key=(-1, -1) → overall
    """
    nj_u = np.asarray(res['n_jets_unique']).astype(int)
    nb_u = np.asarray(res['n_b_jets_unique']).astype(int)
    nj_index = {int(v): i for i, v in enumerate(nj_u)}
    nb_index = {int(v): i for i, v in enumerate(nb_u)}

    acc = np.asarray(res['accuracy'])
    se  = np.asarray(res['stderr'])
    cnt = np.asarray(res['counts'])

    series: Dict[Tuple[int,int], Tuple[float,float,int]] = {}
    for nj in jets:
        nj = int(nj)
        if nj not in nj_index:
            continue
        i = nj_index[nj]
        for nb in range(max(min_nb, 0), nj + 1):
            if nb not in nb_index:
                continue
            j = nb_index[nb]
            N = int(cnt[i, j])
            p = float(acc[i, j])
            s = float(se[i, j])
            if N <= 0 or not np.isfinite(p):
                continue
            series[(nj, nb)] = (p, s, N)

    if include_overall and 'overall' in res:
        ov = res['overall']
        series[(-1, -1)] = (float(ov['accuracy']), float(ov['stderr']), int(ov['N']))

    return series

def _series_by_njet(res, njet, min_nb=2, use_wilson=True):
    nj_u = np.asarray(res['n_jets_unique']).astype(int)
    nb_u = np.asarray(res['n_b_jets_unique']).astype(int)
    if njet not in nj_u:
        return np.array([]), np.array([]), (np.array([]), np.array([])), np.array([])
    i = {int(v): k for k, v in enumerate(nj_u)}[njet]
    jmap = {int(v): k for k, v in enumerate(nb_u)}

    acc = np.asarray(res['accuracy'], dtype=float)
    se  = np.asarray(res['stderr'], dtype=float)
    wlo = np.asarray(res['wilson_lo'], dtype=float)
    whi = np.asarray(res['wilson_hi'], dtype=float)
    cnt = np.asarray(res['counts'], dtype=int)

    xs, ys, ylo, yhi, Ns = [], [], [], [], []
    for nb in range(max(min_nb, 0), njet + 1):
        if nb not in jmap: 
            continue
        j = jmap[nb]
        if cnt[i, j] <= 0 or not np.isfinite(acc[i, j]):
            continue
        xs.append(nb)
        ys.append(acc[i, j])
        if use_wilson and np.isfinite(wlo[i, j]) and np.isfinite(whi[i, j]):
            ylo.append(acc[i, j] - wlo[i, j])
            yhi.append(whi[i, j] - acc[i, j])
        else:
            ylo.append(se[i, j]); yhi.append(se[i, j])
        Ns.append(cnt[i, j])

    xs  = np.asarray(xs,  int)
    ys  = np.asarray(ys,  float)
    ylo = np.asarray(ylo, float)
    yhi = np.asarray(yhi, float)
    Ns  = np.asarray(Ns,  int)
    return xs, ys, (ylo, yhi), Ns

def plot_assignment_accuracy(
    res_45: dict, res_4x: dict, res_2x: dict,
    *,
    jets=(4,5,6,7),
    min_nb=2,
    use_wilson=True,
    labels=("45", "4x", "2x"),
    figsize=(6.5, 3.2),
    ylim=(0.0, 1.05),
    annotate=True,
    annotate_percent=True,
    annotate_show_N=True,
    annotate_unc=False,          # 텍스트에 ±오차 포함 여부
    annotate_min_N=3,            # 이 값 미만이면 주석 생략
    fontsize=9,
    save_dir: str | None = None, # 경로 주면 자동 저장
    file_prefix="acc_by_nb_njet",
):
    """
    n_jet별로 개별 그림 생성. (fig, ax) 리스트 반환.
    """
    try: plt.style.use(hep.style.CMS)
    except Exception: pass

    figs_axes = []
    modes = [
        (res_45, labels[0], dict(marker="o", linestyle="-",  linewidth=1.8)),
        (res_4x, labels[1], dict(marker="s", linestyle="-.", linewidth=1.8)),
        (res_2x, labels[2], dict(marker="^", linestyle="--", linewidth=1.8)),
    ]
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for nj in jets:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        handles, labs = [], []

        for res, lab, style_kw in modes:
            x, y, (ylo, yhi), Ns = _series_by_njet(res, int(nj), min_nb=min_nb, use_wilson=use_wilson)
            if x.size == 0:
                continue
            eb = ax.errorbar(x, y, yerr=np.vstack([ylo, yhi]), capsize=3, label=lab, **style_kw)
            # ErrorbarContainer → 대표 선 핸들
            line = eb.lines[0] if hasattr(eb, "lines") else eb[0]
            handles.append(line); labs.append(lab)

            if annotate:
                for xi, yi, lo_i, hi_i, Ni in zip(x, y, ylo, yhi, Ns):
                    if annotate_min_N and Ni < annotate_min_N:
                        continue
                    if annotate_percent:
                        s_val = f"{yi*100:.1f}%"
                        s_unc = f" ± {((lo_i+hi_i)/2.0)*100:.1f}%" if annotate_unc else ""
                    else:
                        s_val = f"{yi:.3f}"
                        s_unc = f" ± {((lo_i+hi_i)/2.0):.3f}" if annotate_unc else ""
                    s_N   = f" (N={Ni})" if annotate_show_N else ""
                    ax.annotate(f"{s_val}{s_unc}{s_N}", (xi, yi),
                                textcoords="offset points", xytext=(0, 6),
                                ha="center", va="bottom", fontsize=fontsize)

        ax.set_ylim(*ylim)
        ax.set_xlabel("n_b_tagged (≥ WP)")
        ax.set_ylabel("Assignment accuracy")
        ax.grid(True, alpha=0.3)
        fig.legend(handles, labs, ncol=3, frameon=False)
        try:
            hep.cms.label("Preliminary", data=True, loc=0, ax=ax)   # (13 TeV) 필요하면 label_args로 추가
        except Exception:
            pass
        fig.tight_layout(rect=(0, 0.02, 1, 0.88))

        if save_dir:
            out = os.path.join(save_dir, f"{file_prefix}{nj}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")

        figs_axes.append((nj, fig, ax))

    return figs_axes

def ROC_AUC(score, y, plot_path, weight=None, fname="ROC.png", style="CMS",
            scale="linear", labels=None,
            title=None, subtitle=None,
            extra_text=None, extra_loc="upper left", extra_kwargs=None,
            legend_loc="lower right"):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    try:
        import mplhep as hep
        hep.style.use(style)
    except Exception:
        pass

    # normalize inputs to list-of-series
    is_multi = isinstance(score, (list, tuple))
    if not is_multi:
        score_list  = [score]
        y_list      = [y]
        weight_list = [weight]
    else:
        score_list  = list(score)
        y_list      = list(y) if isinstance(y, (list, tuple)) else [y]*len(score_list)
        weight_list = list(weight) if isinstance(weight, (list, tuple)) else [weight]*len(score_list)

    # fallback colors if user constants are undefined
    try:
        BASELINE
    except NameError:
        BASELINE = "0.5"
    try:
        ROC_COLOR
    except NameError:
        ROC_COLOR = None  # let matplotlib choose

    fig, ax = plt.subplots(figsize=(12.0, 9.0), dpi=150)

    # random baseline
    if scale == "log":
        y = np.logspace(-2, 0, 300)
        ax.plot(y, y, ls="--", lw=1.5, label="Random")
    else:
        ax.plot([0, 1], [0, 1], ls="--", lw=1.5, label="Random")

    aucs = []
    for i, (s, yy, w) in enumerate(zip(score_list, y_list, weight_list)):
        s  = np.asarray(s, dtype=np.float32)
        yy = np.asarray(yy, dtype=np.int8)
        if w is not None:
            w = np.asarray(w, dtype=np.float64)
            mask = np.isfinite(s) & np.isfinite(yy) & np.isfinite(w) & (w > 0)
            s, yy, w = s[mask], yy[mask], w[mask]
        else:
            mask = np.isfinite(s) & np.isfinite(yy)
            s, yy = s[mask], yy[mask]
            w = None

        fpr, tpr, _ = roc_curve(yy, s, sample_weight=w, drop_intermediate=True)
        auc = roc_auc_score(yy, s, sample_weight=w)
        aucs.append(auc)

        lab = labels[i] if (labels and i < len(labels)) else (f"ROC (AUC = {auc:.3f})" if not is_multi else f"Fold {i} (AUC = {auc:.3f})")
        ax.plot(fpr, tpr, lw=2.2, label=lab)

    # axes cosmetics
    if scale == "log":
        ax.set_xscale("log")
        ax.set_xlim(1e-2, 1.0)
    else:
        ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc=legend_loc)

    # CMS-like label (safe to skip if mplhep missing)
    try:
        hep.cms.label(llabel="Preliminary", data=False, com=13, ax=ax)
    except Exception:
        pass

    # Title & Subtitle
    if title:
        ax.set_title(title, loc="left", fontsize=18, pad=10)
        if subtitle:
            # subtitle를 제목 바로 아래에 살짝 작게
            ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)

    # Extra text (multi-line supported)
    if extra_text:
        if isinstance(extra_text, (list, tuple)):
            txt = "\n".join(map(str, extra_text))
        else:
            txt = str(extra_text)

        # 위치 해석
        loc_map = {
            "upper left":  (0.02, 0.98, "left",  "top"),
            "upper right": (0.98, 0.98, "right", "top"),
            "lower left":  (0.02, 0.02, "left",  "bottom"),
            "lower right": (0.98, 0.02, "right", "bottom"),
        }
        x, y, ha, va = loc_map.get(extra_loc, loc_map["upper left"])

        kw = dict(
            transform=ax.transAxes,
            ha=ha, va=va,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6)
        )
        if extra_kwargs:
            kw.update(extra_kwargs)

        ax.text(x, y, txt, **kw)

    # save
    os.makedirs(plot_path, exist_ok=True)
    out_path = os.path.join(plot_path, fname)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return aucs if is_multi else aucs[0]