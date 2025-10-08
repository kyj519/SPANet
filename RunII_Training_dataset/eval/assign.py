from __future__ import annotations
import numpy as np
from typing import Optional, Sequence, Literal, Tuple

Role = Literal['hw', 'ht', 'lt']

# -------------------------
# numeric helpers
# -------------------------
def logsumexp_nd(a: np.ndarray, axis=None, keepdims=False):
    """
    다축 logsumexp. -inf만 있는 슬라이스도 안전하게 처리.
    """
    if axis is None:
        axis = tuple(range(a.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    a_max = np.max(a, axis=axis, keepdims=True)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        exp_terms = np.exp(np.where(np.isfinite(a), a - a_max, -np.inf))
        sum_exp = np.sum(exp_terms, axis=axis, keepdims=True)
        out = a_max + np.log(sum_exp)  # sum_exp=0이면 -inf
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out

def apply_temperature_to_logsoftmax_nd(
    log_probs: np.ndarray,
    T: float,
    axes: Optional[Tuple[int, ...]] = None,
    mask: Optional[np.ndarray] = None,
    return_probs: bool = False,
):
    """
    log_probs: (N, d1, d2, ...), 이미 log-softmax(로그확률) 상태
    T: float (>0), 또는 broadcast 가능한 배열
    axes: 정규화할 축(tuple). None이면 배치축(0)을 제외한 모든 축.
    mask: True=유효, False=무효(정규화에서 제외)
    """
    if axes is None:
        axes = tuple(range(1, log_probs.ndim))
    if T <= 0:
        raise ValueError("temperature must be > 0")

    scaled = log_probs / T
    if mask is not None:
        scaled = np.where(mask, scaled, -np.inf)
    lse = logsumexp_nd(scaled, axis=axes, keepdims=True)
    log_probs_T = scaled - lse
    if return_probs:
        return log_probs_T, np.exp(log_probs_T)
    return log_probs_T

# -------------------------
# assignment (greedy only)
# -------------------------
def _check_shapes(lt_logp: np.ndarray, ht_logp: np.ndarray, hw_logp: np.ndarray):
    if lt_logp.ndim != 2 or lt_logp.shape[1] != 9:
        raise ValueError(f"lt_logp shape must be (N,9), got {lt_logp.shape}")
    if ht_logp.ndim != 2 or ht_logp.shape[1] != 9 or ht_logp.shape[0] != lt_logp.shape[0]:
        raise ValueError(f"ht_logp shape must be (N,9) with same N, got {ht_logp.shape}")
    if hw_logp.ndim != 3 or hw_logp.shape[1:] != (9, 9) or hw_logp.shape[0] != lt_logp.shape[0]:
        raise ValueError(f"hw_logp shape must be (N,9,9) with same N, got {hw_logp.shape}")

def _argmax_with_mask_vec(v: np.ndarray, used: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    v: (N,9) logp
    used: (N,9) bool (True=이미 사용됨→배제)
    return: (best_idx(N,), best_val(N,))
    """
    masked = np.where(~used, v, -np.inf)
    best_idx = np.argmax(masked, axis=1)
    rows = np.arange(v.shape[0])
    best_val = masked[rows, best_idx]
    return best_idx, best_val

def _argmax_with_mask_mat(m: np.ndarray, used: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    m: (N,9,9) logp (hw)
    used: (N,9) bool — 행/열 모두에서 True=사용됨→배제, 또한 대각선(i==j)도 배제
    return: (i_best(N,), j_best(N,), best_val(N,))
    """
    N = m.shape[0]
    rows = np.arange(N)

    # 기본 배제: 대각선
    base_mask = ~np.eye(9, dtype=bool)  # True=유효
    # 행/열 사용 배제
    # mask_hw[n, i, j] = base_mask[i,j] & (~used[n,i]) & (~used[n,j])
    valid = base_mask[None, :, :] & (~used[:, :, None]) & (~used[:, None, :])

    masked = np.where(valid, m, -np.inf)
    # argmax over 9*9
    flat = masked.reshape(N, -1)
    flat_idx = np.argmax(flat, axis=1)
    best_val = flat[rows, flat_idx]
    i_best, j_best = np.unravel_index(flat_idx, (9, 9))
    return i_best, j_best, best_val

def _finalize_with_fallback(
    lt_logp: np.ndarray, ht_logp: np.ndarray, hw_logp: np.ndarray,
    lb: np.ndarray, hb: np.ndarray, w1: np.ndarray, w2: np.ndarray
):
    """
    일부 이벤트가 -inf로 실패했을 때 폴백 시도(느리지만 건수 제한적으로 처리).
    """
    N = lt_logp.shape[0]
    rows = np.arange(N)

    # 어떤 값이든 -inf면 bad
    bad = np.where(
        ~np.isfinite(lb.astype(float)) |
        ~np.isfinite(hb.astype(float)) |
        ~np.isfinite(w1.astype(float)) |
        ~np.isfinite(w2.astype(float))
    )[0]

    if bad.size == 0:
        return

    diag = np.arange(9)
    for n in bad:
        # hw에서 가능한 페어를 점수순으로 훑어보며 유효 조합 찾기
        hw_row = hw_logp[n].copy()
        hw_row[diag, diag] = -np.inf
        order = np.argsort(hw_row, axis=None)[::-1]
        found = False
        for p in order:
            i, j = divmod(int(p), 9)
            if i == j:
                continue
            # ht: i, j 제외
            ht_row = ht_logp[n].copy()
            ht_row[i] = -np.inf; ht_row[j] = -np.inf
            hb_n = int(np.argmax(ht_row))
            if not np.isfinite(ht_row[hb_n]):
                continue
            # lt: i, j, hb 제외
            lt_row = lt_logp[n].copy()
            lt_row[i] = lt_row[j] = lt_row[hb_n] = -np.inf
            lb_n = int(np.argmax(lt_row))
            if not np.isfinite(lt_row[lb_n]):
                continue
            w1[n], w2[n], hb[n], lb[n] = i, j, hb_n, lb_n
            found = True
            break
        if not found:
            # 그대로 두되, 이후 상위에서 에러를 내도록 함
            continue

    still_bad = np.where(
        ~np.isfinite(lb.astype(float)) |
        ~np.isfinite(hb.astype(float)) |
        ~np.isfinite(w1.astype(float)) |
        ~np.isfinite(w2.astype(float))
    )[0][:10]
    if still_bad.size > 0:
        raise RuntimeError(f"No valid assignment for events: {still_bad.tolist()} (up to 10)")

def pick_assignments_numpy(
    lt_logp: np.ndarray,      # (N,9) log-softmax
    ht_logp: np.ndarray,      # (N,9) log-softmax
    hw_logp: np.ndarray,      # (N,9,9) log-softmax
    return_probs: bool = False,
    order: Sequence[Role] = ('hw','ht','lt'),  # 임의 순서 허용
    temperature: float = 1.0,                  # hw 전용 T
):
    """
    Greedy only. order는 ('hw','ht','lt')의 임의 순서 가능.
    - 제약: hw는 i!=j, 그리고 세 인덱스(lb,hb,w1,w2)가 모두 달라야 함(자원 충돌 방지).
    - 입력은 이미 log-softmax(로그확률)라고 가정.
    """
    _check_shapes(lt_logp, ht_logp, hw_logp)
    if sorted(order) != ['ht', 'hw', 'lt']:
        raise ValueError("order must be a permutation of ('lt','ht','hw')")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    N = lt_logp.shape[0]
    rows = np.arange(N)

    # 온도 스케일링: hw에만 적용(원하면 lt/ht에도 쉽게 확장 가능)
    if temperature != 1.0:
        hw_logp = apply_temperature_to_logsoftmax_nd(hw_logp, temperature)

    # 사용된 슬롯 마스크 (행=인덱스 사용 여부)
    used = np.zeros((N, 9), dtype=bool)

    # 결과 컨테이너 (초기 -1)
    lb = np.full(N, -1, dtype=np.int64)
    hb = np.full(N, -1, dtype=np.int64)
    w1 = np.full(N, -1, dtype=np.int64)
    w2 = np.full(N, -1, dtype=np.int64)

    # 각 단계별 점수 저장(요청 시 반환)
    lt_best = np.full(N, -np.inf, dtype=np.float32)
    ht_best = np.full(N, -np.inf, dtype=np.float32)
    hw_best = np.full(N, -np.inf, dtype=np.float32)

    # 역할별 선택 함수
    for step, role in enumerate(order):
        if role == 'hw':
            i_best, j_best, val = _argmax_with_mask_mat(hw_logp, used)
            # 반영
            w1[:] = np.where(w1 < 0, i_best, w1)
            w2[:] = np.where(w2 < 0, j_best, w2)
            hw_best[:] = val.astype(np.float32, copy=False)
            # 사용 처리 (유효한 것만)
            ok = np.isfinite(val)
            used[rows[ok], i_best[ok]] = True
            used[rows[ok], j_best[ok]] = True

        elif role == 'ht':
            idx, val = _argmax_with_mask_vec(ht_logp, used)
            hb[:] = np.where(hb < 0, idx, hb)
            ht_best[:] = val.astype(np.float32, copy=False)
            ok = np.isfinite(val)
            used[rows[ok], idx[ok]] = True

        elif role == 'lt':
            idx, val = _argmax_with_mask_vec(lt_logp, used)
            lb[:] = np.where(lb < 0, idx, lb)
            lt_best[:] = val.astype(np.float32, copy=False)
            ok = np.isfinite(val)
            used[rows[ok], idx[ok]] = True

        else:
            raise AssertionError("unreachable")

    # 폴백: 여전히 -1인 경우(혹은 -inf 점수) 보정 시도
    need_fallback = (
        (lb < 0).any() or (hb < 0).any() or (w1 < 0).any() or (w2 < 0).any() or
        (~np.isfinite(lt_best)).any() or (~np.isfinite(ht_best)).any() or (~np.isfinite(hw_best)).any()
    )
    if need_fallback:
        _finalize_with_fallback(lt_logp, ht_logp, hw_logp, lb, hb, w1, w2)

        # 폴백 이후 점수 재계산(보장용)
        lt_best = lt_logp[rows, lb]
        ht_best = ht_logp[rows, hb]
        hw_best = hw_logp[rows, w1, w2]

    assignments = np.stack([lb, hb, w1, w2], axis=1).astype(np.uint8, copy=False)

    if return_probs:
        return assignments, hw_best, lt_best, ht_best
    return assignments