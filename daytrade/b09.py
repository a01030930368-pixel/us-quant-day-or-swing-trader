"""규칙서 b09: 장중 급락(전날 종가 대비 -10% 이하) 종목의 반등 확인 매수, 그날 저가 손절, 장 마감 매도.

day = 그날 정규장 1분봉 dict(mi, o, h, l, c, v). mi = 9:30부터 몇 분째. 5분봉은 9:30부터 5분 단위(mi // 5).
"""
import numpy as np

CLOSE = 389


def hit_index(day, pc, thr):
    """처음으로 저가가 전날 종가 x (1 - thr) 이하가 된 1분봉 번호 (없으면 None)."""
    k = np.nonzero(day["l"] <= pc * (1 - thr))[0]
    return int(k[0]) if len(k) else None


def exit_from(day, e, px, stop, until=CLOSE):
    """e번 1분봉 시가에 px로 산 뒤: 저가가 stop 이하면 그 값(시가가 더 낮으면 시가)에 손절, 아니면 until 이하 마지막 봉 종가."""
    mi, o, l, c = day["mi"], day["o"], day["l"], day["c"]
    j_end = int(np.searchsorted(mi, until, side="right")) - 1
    if stop is not None:
        hit = np.nonzero(l[e:j_end + 1] <= stop)[0]
        if len(hit):
            j = e + hit[0]
            out = min(o[j], stop)
            return dict(exit_mi=int(mi[j]), exit_px=out, gross=out / px - 1, reason="손절")
    return dict(exit_mi=int(mi[j_end]), exit_px=c[j_end], gross=c[j_end] / px - 1, reason="마감")


def signals_5min(day, start):
    """start번 1분봉이 속한 5분봉부터, 5분봉 종가가 직전 3개 5분봉 고가 최고값을 넘은 5분봉들의 '다음 1분봉 시각'."""
    mi, h, c = day["mi"], day["h"], day["c"]
    k = mi // 5
    ks, first = np.unique(k, return_index=True)
    last = np.r_[first[1:], len(mi)] - 1
    hi = np.maximum.reduceat(h, first)
    out = []
    k0 = k[start]
    for n in range(3, len(ks)):
        if ks[n] < k0 or ks[n - 3] != ks[n] - 3:
            continue
        if c[last[n]] > hi[n - 3:n].max():
            out.append(int(ks[n] * 5 + 5))
    return out


def signals_vwap(day, start):
    """C3: start 이후 1분봉 종가가 그날 VWAP(9:30부터) 위로 올라선 봉들의 '다음 1분봉 시각'."""
    mi, c, v = day["mi"], day["c"], day["v"]
    vwap = np.cumsum(c * v) / np.maximum(np.cumsum(v), 1e-9)
    above = c > vwap
    return [int(mi[j] + 1) for j in range(max(start, 1), len(mi)) if above[j] and not above[j - 1]]


def confirmed(day, pc, thr=0.10, how="5min", t_lo=15, t_hi=330, max_stop=0.08, min_low_age=10, until=None):
    """C1(how='5min')/C3(how='vwap'): 급락 뒤 반등 확인 신호 중 첫 번째로 조건(9:45~15:00, 저가가 10분 이상 전, 손절폭 8% 이내)을
    맞는 것에 매수. 손절 = 매수 전까지 그날 저가. until: 청산 시각(None이면 15:59, 숫자면 매수 후 그 분 뒤)."""
    s = hit_index(day, pc, thr)
    if s is None:
        return None
    mi, o, l, v = day["mi"], day["o"], day["l"], day["v"]
    sigs = signals_5min(day, s) if how == "5min" else signals_vwap(day, s)
    for t in sigs:
        if t < t_lo or t > t_hi:
            continue
        e = int(np.searchsorted(mi, t))
        if e >= len(mi) or mi[e] > t + 3:
            continue
        lo_i = int(np.argmin(l[:e]))
        low, px = l[lo_i], o[e]
        if mi[lo_i] > mi[e] - min_low_age or px <= low or (px - low) / px > max_stop:
            continue
        r = exit_from(day, e, px, low, CLOSE if until is None else min(CLOSE, int(mi[e]) + until))
        return dict(entry_mi=int(mi[e]), entry_px=px, stop_dist=(px - low) / px, drop_at_entry=1 - low / pc,
                    vol_to_entry=float(v[:e].sum()), **r)
    return None


def at_touch(day, pc, thr=0.10):
    """C2: -10%에 닿는 순간 그 가격에 매수 (갭으로 그 아래서 시작하면 첫 시가), 손절 없이 15:59 종가."""
    s = hit_index(day, pc, thr)
    if s is None or day["mi"][s] > 330:
        return None
    px = min(day["o"][s], pc * (1 - thr))
    r = exit_from(day, s, px, None)
    return dict(entry_mi=int(day["mi"][s]), entry_px=px, stop_dist=np.nan, drop_at_entry=1 - px / pc,
                vol_to_entry=float(day["v"][:s].sum()), **r)
