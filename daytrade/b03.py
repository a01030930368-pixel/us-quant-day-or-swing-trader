"""규칙서 b03: 뉴스로 크게 갭 상승한 종목의 갭 앤 고 (매수만, 15:55 청산).

한 종목-하루의 정규장 1분봉(mi = 9:30부터 몇 분째)과 전날 종가 pc, 장전 고가·저가로 거래 하나를 낸다.
매수한 1분봉 안에서 손절가에도 닿았으면 손절로 친다(불리한 쪽).
"""
import numpy as np

CLOSE_MIN = 385   # 15:55 청산
DEADLINE = 90     # 11:00까지만 매수
MAX_STOP = 0.08   # 손절폭이 8%보다 깊으면 건너뜀


def _exit(day, i, px, stop, half_r=None):
    mi, o, h, l, c = day["mi"], day["o"], day["h"], day["l"], day["c"]
    n_close = np.searchsorted(mi, CLOSE_MIN)
    if i >= n_close or stop >= px or stop <= 0:
        return None

    def run(start, stop_px):
        s = np.flatnonzero(l[start:n_close] <= stop_px)
        if len(s):
            k = start + s[0]
            return (stop_px if k == i else min(o[k], stop_px)), k, "손절"
        if n_close < len(mi):
            return o[n_close], n_close, "마감"
        return c[-1], len(mi) - 1, "마감"

    xp, k, why = run(i, stop)
    gross = xp / px - 1
    if half_r is not None:
        target = px + half_r * (px - stop)
        t = np.flatnonzero(h[i:n_close] >= target)
        if len(t) and i + t[0] < k:  # 손절보다 먼저 목표가 -> 절반 익절, 나머지는 원래 손절·마감
            kt = i + t[0]
            tp = target if kt == i else max(o[kt], target)
            gross = 0.5 * (tp / px - 1) + 0.5 * gross
            why = "절반 익절 후 " + why
    return dict(entry_mi=int(mi[i]), exit_mi=int(mi[k]), entry_px=px, gross=gross, stop_dist=1 - stop / px, reason=why)


def _gap_held(l, upto, pc):
    """9:30부터 upto 직전까지 저가가 전날 종가 위에 있었나 (갭이 안 메워졌나)."""
    return upto == 0 or l[:upto].min() > pc


def range_breakout(day, w, pc, half_r=None):
    mi, o, h, l = day["mi"], day["o"], day["h"], day["l"]
    n_rng = np.searchsorted(mi, w)
    if n_rng == 0 or n_rng >= len(mi):
        return None
    rh, rl = h[:n_rng].max(), l[:n_rng].min()
    if rl / rh - 1 < -MAX_STOP:
        return None
    n_last = np.searchsorted(mi, DEADLINE)
    hit = np.flatnonzero(h[n_rng:n_last] >= rh)
    if len(hit) == 0:
        return None
    i = n_rng + hit[0]
    if not _gap_held(l, i, pc):
        return None
    return _exit(day, i, max(o[i], rh), rl - 0.01, half_r)


def premarket_breakout(day, pc, pm_high, pm_low):
    mi, o, h, l = day["mi"], day["o"], day["h"], day["l"]
    if not np.isfinite(pm_high):
        return None
    n_last = np.searchsorted(mi, DEADLINE)
    hit = np.flatnonzero(h[:n_last] >= pm_high)
    if len(hit) == 0:
        return None
    i = hit[0]
    if not _gap_held(l, i, pc):
        return None
    px = max(o[i], pm_high)
    stop = (l[:i].min() if i > 0 else pm_low) - 0.01
    if 1 - stop / px > MAX_STOP:
        return None
    return _exit(day, i, px, stop)


def open_buy(day, pm_low):
    """9:31 시가에 바로 매수, 손절은 장전 저가."""
    mi, o = day["mi"], day["o"]
    i = np.searchsorted(mi, 1)
    if i >= len(mi) or not np.isfinite(pm_low):
        return None
    px = o[i]
    stop = pm_low - 0.01
    if 1 - stop / px > MAX_STOP:
        return None
    return _exit(day, i, px, stop)
