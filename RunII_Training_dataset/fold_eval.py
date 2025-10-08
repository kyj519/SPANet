#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from matplotlib import pyplot as plt
import numpy as np

import core.infer
import eval.assign
import plot.plot as plotmod  # plot.plot 모듈

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model on a fold")
    p.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    p.add_argument('--data_path',  type=str, required=True, help='Path to the data file')
    p.add_argument('--include_ctag', action='store_true', help='Include ctag in evaluation')
    p.add_argument('--outdir', type=str, default="plots", help='Output directory for plots')
    p.add_argument('--reduction_factor', type=int, default=1, help='Speed-up factor when reading h5')
    p.add_argument('--roc_title', type=str, default="Class 0 vs each class", help='Title for ROC plot')
    return p.parse_args()

def _maybe(results, key, default=None):
    return results[key] if key in results else default

def compute_assignments(results):
    """기존 assignment 픽 함수들 실행 (원한다면 결과를 리턴/저장)"""
    lt = results['lt_assignment_log_probability']
    ht = results['ht_assignment_log_probability']
    a4  = eval.assign.pick_assignments_numpy(lt, ht, results['hw_4_assignment_log_probability'])
    a2  = eval.assign.pick_assignments_numpy(lt, ht, results['hw_2_assignment_log_probability'])
    a45 = eval.assign.pick_assignments_numpy(lt, ht, results['hw_45_assignment_log_probability'])
    return {"assign_4": a4, "assign_2": a2, "assign_45": a45}

def plot_roc_zero_vs_each(pred, y, outdir, weights=None, class_names=None,
                          title="Class 0 vs each class", fname="roc_zero_vs_each.png",
                          scale="log", legend_loc="lower right"):
    """
    pred: (N, K) — class probability or score (0~1 권장)
    y:    (N,)   — int class labels in [0..K-1]
    weights: (N,) optional
    class 0을 양성(positive)으로 두고, 각 k!=0에 대해 (0 vs k) ROC를 한 그림에 그림.
    """
    os.makedirs(outdir, exist_ok=True)

    pred = np.asarray(pred)
    y    = np.asarray(y).astype(np.int64)
    K    = pred.shape[1]
    assert K >= 2, f"K={K} < 2"
    assert y.ndim == 1 and y.shape[0] == pred.shape[0], "y shape mismatch"
    if weights is not None:
        weights = np.asarray(weights)

    # 점수는 "클래스 0의 점수"로 통일 (0 vs k 이므로)
    score_all = pred[:, 0]

    score_list, y_list, w_list, labels = [], [], [], []
    for k in range(1, K):
        mask = (y == 0) | (y == k)
        if not np.any(mask):  # 해당 pair 데이터 없음
            continue
        yy = (y[mask] == 0).astype(np.int8)  # 0이면 1, k면 0 → "class 0 is positive"
        ss = score_all[mask]
        ww = weights[mask] if weights is not None else None

        score_list.append(ss)
        y_list.append(yy)
        w_list.append(ww)

        cname0 = "0" if class_names is None else class_names[0]
        cnamek = str(k) if class_names is None else class_names[k]
        labels.append(f"{cname0} vs {cnamek}")

    # plot.plot의 ROC_AUC를 그대로 사용 (멀티 인풋 지원)
    aucs = plotmod.ROC_AUC(
        score=score_list,
        y=y_list,
        weight=w_list,
        plot_path=outdir,
        fname=fname,
        scale=scale,
        labels=labels,
        title=title,
        legend_loc=legend_loc,
    )
    return dict(zip(labels, aucs))

def main():
    args = parse_args()

    print(f"[INFO] Evaluating model from {args.model_path} on data {args.data_path}")
    session = core.infer.load_session(args.model_path)
    result = core.infer.infer_h5(
        args.data_path, session,
        include_ctag=args.include_ctag,
        reduction_factor=args.reduction_factor
    )

    # 선택: assignment 계산 (필요 시 사용)
    assigns = compute_assignments(result)
    a4 = assigns['assign_4']
    a2 = assigns['assign_2']
    a45 = assigns['assign_45']
    res_4x = plotmod.get_assignment_accuracy(result['n_jet'], result['n_b_tagged'], a4, result['real_target'], [4,5])
    res_2x = plotmod.get_assignment_accuracy(result['n_jet'], result['n_b_tagged'], a2, result['real_target'], [2,3])
    res_45 = plotmod.get_assignment_accuracy(result['n_jet'], result['n_b_tagged'], a45, result['real_target'], [6,7])

    figs_axes = plotmod.plot_assignment_accuracy(
        res_45, res_4x, res_2x,
        annotate=True,
        annotate_percent=True,
        annotate_unc=True,      # 텍스트에 ±오차도 같이
        annotate_show_N=True,
        figsize=(13, 9)
    )
    os.makedirs(args.outdir, exist_ok=True)
    for nj, fig, ax in figs_axes:
        fig.savefig(os.path.join(args.outdir, f"assignment_accuracy_{nj}.png"))
        plt.close(fig)
    print(f"[INFO] Assignment accuracy plots saved to {args.outdir}/")

    # 분류 결과/레이블/가중치
    classif_pred = result['EVENT/signal']       # (N, K)
    classif_y    = result['real_class']         # (N,)
    weight       = None

    # 일관성 체크 (원 코드의 assert 버그 수정: "!=" → "==")
    num_class = np.unique(classif_y).size
    assert num_class == classif_pred.shape[1], \
        f"num_class {num_class} != pred.shape[1] {classif_pred.shape[1]}"


    # 0 vs 각 클래스 ROC
    auc_map = plot_roc_zero_vs_each(
        pred=classif_pred,
        y=classif_y,
        weights=weight,
        class_names=None,
        outdir=args.outdir,
        title=args.roc_title,
        fname="roc_0_vs_each.png",
        legend_loc="lower right",
    )

    # 결과 요약 출력
    print("[ROC_AUC] 0 vs each class")
    for k, v in auc_map.items():
        print(f"  {k}: AUC = {v:.4f}")

if __name__ == "__main__":
    main()