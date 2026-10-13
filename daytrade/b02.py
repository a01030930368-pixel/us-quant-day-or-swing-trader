"""규칙서 b02: 그날 거래가 몰린 종목(상대거래량 상위)의 시가 범위 고가 돌파, 매수만, 장 마감 청산.

한 종목-하루의 정규장 1분봉(mi = 9:30부터 몇 분째)과 시간 창 w(분)를 받아 거래 하나를 낸다.
범위 = 9:30부터 w분 동안의 고가·저가. 범위가 양봉(마지막 종가 > 첫 시가)일 때만, w분 이후 범위 고가에 닿으면
그 값에 매수(갭으로 넘으면 그 1분봉 시가). 매수한 1분봉 안에서 손절가에도 닿았으면 손절로 친다(불리한 쪽).
"""
import numpy as np

CLOSE_MIN = 385   # 15:55 청산
LAST_ENTRY = 330  # 15:00 이후에는 새로 사지 않음


def trade(day, w, stop_mode="range", atr=None, atr_frac=0.10):
    mi, o, h, l, c = day["mi"], day["o"], day["h"], day["l"], day["c"]
    n_rng = np.searchsorted(mi, w)
    if n_rng == 0 or n_rng >= len(mi):
        return None
    if c[n_rng - 1] <= o[0]:
        return None  # 범위가 음봉이거나 보합 -> 건너뜀 (논문은 공매도)
    rh, rl = h[:n_rng].max(), l[:n_rng].min()
    n_last = np.searchsorted(mi, LAST_ENTRY)
    hit = np.flatnonzero(h[n_rng:n_last] >= rh)
    if len(hit) == 0:
        return None
    i = n_rng + hit[0]
    px = max(o[i], rh)
    stop = px - atr_frac * atr if stop_mode == "atr" else rl - 0.01
    if stop <= 0 or stop >= px:
        return None
    n_close = np.searchsorted(mi, CLOSE_MIN)
    if i >= n_close:
        return None
    s = np.flatnonzero(l[i:n_close] <= stop)
    if len(s):
        k = i + s[0]
        xp, why = (stop if k == i else min(o[k], stop)), "손절"
    elif n_close < len(mi):
        k, xp, why = n_close, o[n_close], "마감"
    else:
        k, xp, why = len(mi) - 1, c[-1], "마감"
    return dict(entry_mi=int(mi[i]), exit_mi=int(mi[k]), entry_px=px, gross=xp / px - 1,
                stop_dist=1 - stop / px, reason=why, range_pct=rh / rl - 1)
