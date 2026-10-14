"""규칙서 b21~b23(3배 ETF)과 b25(주말 비트코인 → 월요일 관련주)의 날짜 계산과 체결.

day = 그날 정규장 1분봉 dict(mi, o, h, l, c, v). mi = 9:30부터 몇 분째.
체결: 넘는 순간 매수는 기준가에 (그 1분봉 시가가 이미 위면 시가), 손절은 1분봉 저가가 닿으면 손절가,
매수한 1분봉 안에서 손절가에도 닿았으면 손절 (불리한 쪽). 청산은 15:59 1분봉 종가 (반나절 장은 12:59).
"""
import datetime as dt

import numpy as np
from dateutil.easter import easter

from daytrade.mom import simulate

TICK = 0.01
CLOSE, HALF_CLOSE = 389, 209
SPLIT_RATIOS = (1.5, 2, 3, 4, 5, 8, 10, 20)


def _nth_weekday(y, m, wd, n):
    d = dt.date(y, m, 1)
    d += dt.timedelta(days=(wd - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(y, m, wd):
    d = dt.date(y, m + 1, 1) - dt.timedelta(days=1) if m < 12 else dt.date(y, 12, 31)
    return d - dt.timedelta(days=(d.weekday() - wd) % 7)


def _observed(d, sat_to_fri=True):
    if d.weekday() == 5:
        return d - dt.timedelta(days=1) if sat_to_fri else None
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def nyse_holidays(y0, y1):
    """NYSE 정규 휴장일 {날짜: 이름}. 임시 휴장(9·11, 허리케인, 대통령 장례)은 넣지 않는다."""
    out = {}
    for y in range(y0, y1 + 1):
        cand = [("새해", _observed(dt.date(y, 1, 1), sat_to_fri=False)),
                ("마틴 루터 킹", _nth_weekday(y, 1, 0, 3) if y >= 1998 else None),
                ("대통령의 날", _nth_weekday(y, 2, 0, 3)),
                ("성금요일", easter(y) - dt.timedelta(days=2)),
                ("메모리얼 데이", _last_weekday(y, 5, 0)),
                ("준틴스", _observed(dt.date(y, 6, 19)) if y >= 2022 else None),
                ("독립기념일", _observed(dt.date(y, 7, 4))),
                ("노동절", _nth_weekday(y, 9, 0, 1)),
                ("추수감사절", _nth_weekday(y, 11, 3, 4)),
                ("크리스마스", _observed(dt.date(y, 12, 25)))]
        out.update({d: n for n, d in cand if d is not None})
    return out


def split_days(daily, window=10):
    """daily: DataFrame(ticker, d, o, c) 정렬됨. 분할로 보이는 갭(시가/전날 종가가 분할 비율 ±3%) 앞뒤 window거래일 집합."""
    bad = set()
    for tk, g in daily.groupby("ticker", sort=False):
        r = (g.o / g.c.shift()).to_numpy()
        hit = np.zeros(len(g), bool)
        for k in SPLIT_RATIOS:
            hit |= (np.abs(r / k - 1) < 0.03) | (np.abs(r * k - 1) < 0.03)
        ds = g.d.to_numpy()
        for i in np.nonzero(hit)[0]:
            bad.update((tk, x) for x in ds[max(0, i - window):i + window + 1])
    return bad


def narrow_flags(h, l):
    """일봉 고가·저가 배열 -> (NR7, ID/NR4) 불리언 배열 (그날 기준)."""
    rng = h - l
    n = len(rng)
    nr7, idnr4 = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(n):
        if i >= 6:
            nr7[i] = rng[i] < rng[i - 6:i].min()
        if i >= 3:
            inside = h[i] < h[i - 1] and l[i] > l[i - 1]
            idnr4[i] = inside and rng[i] < rng[i - 3:i].min()
    return nr7, idnr4


def stretch(o, h, l, n=10):
    """직전 n일 min(|시가-고가|, |시가-저가|) 평균 (그날 값은 안 씀)."""
    m = np.minimum(np.abs(o - h), np.abs(o - l))
    out = np.full(len(o), np.nan)
    for i in range(n, len(o)):
        out[i] = m[i - n:i].mean()
    return out


def exit_index(day, half=False):
    mi = day["mi"]
    return int(np.searchsorted(mi, HALF_CLOSE if half else CLOSE, side="right")) - 1


def open_to_close(day, half=False):
    """9:31 1분봉 시가 매수 -> 15:59 (반나절이면 12:59) 종가."""
    mi = day["mi"]
    e = int(np.searchsorted(mi, 1))
    x = exit_index(day, half)
    if e >= len(mi) or mi[e] > 5 or x <= e:
        return None
    return dict(entry_mi=int(mi[e]), entry_px=day["o"][e], exit_mi=int(mi[x]), exit_px=day["c"][x],
                gross=day["c"][x] / day["o"][e] - 1, reason="마감")


def crabel(day, st, stop_mode="open", t_hi=330, half=False):
    """b21: 오늘 시가 + 스트레치를 넘는 순간 매수. stop_mode 'open' = 시가 아래, 'stretch' = 시가 - 스트레치."""
    mi, o, h = day["mi"], day["o"], day["h"]
    if not len(mi) or mi[0] > 5 or not st > 0:
        return None
    o0 = o[0]
    level = o0 + st
    stop = o0 - TICK if stop_mode == "open" else o0 - st
    until = HALF_CLOSE if half else CLOSE
    for j in range(len(mi)):
        if mi[j] > t_hi or mi[j] >= until:
            return None
        if h[j] > level:
            px = max(o[j], level)
            r = simulate(day, j, px, stop, None, until=until, entry_in_bar=True)
            if r is None:
                return None
            return dict(entry_mi=int(mi[j]), entry_px=px, stop_dist=(px - stop) / px, **r)
    return None


def first15_break(day, max_stop=0.08, t_lo=15, t_hi=90):
    """b25: 9:45~11:00 첫 15분 고가 + 0.01 돌파 매수, 손절 첫 15분 저가 - 0.01, 15:59 청산."""
    mi, o, h, l = day["mi"], day["o"], day["h"], day["l"]
    k = int(np.searchsorted(mi, t_lo))
    if k == 0 or k >= len(mi):
        return None
    hi, lo = h[:k].max(), l[:k].min()
    level, stop = hi + TICK, lo - TICK
    for j in range(k, len(mi)):
        if mi[j] > t_hi:
            return None
        if h[j] >= level:
            px = max(o[j], level)
            if (px - stop) / px > max_stop:
                return None
            r = simulate(day, j, px, stop, None, until=CLOSE, entry_in_bar=True)
            if r is None:
                return None
            return dict(entry_mi=int(mi[j]), entry_px=px, stop_dist=(px - stop) / px, **r)
    return None
