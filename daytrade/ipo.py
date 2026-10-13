"""규칙서 b16(큰 IPO 첫날)·b17(IPO 둘째 날)·b18(IPO 베이스 돌파)의 매수·매도.

day = 그날 정규장 1분봉 dict(mi, o, h, l, c, v). mi = 9:30부터 몇 분째. 청산은 15:59 1분봉 종가.
"""
import numpy as np
import pandas as pd

from daytrade.mom import simulate

CLOSE = 389
TICK = 0.01


def is_despac(row, d0):
    """b16 U2: SEC 회사 정보(row)로 본 스팩 합병·이미 공시하던 회사. row가 없거나 그 회사가 그때 없었으면 None(확인 안 됨)."""
    if row is None or pd.isna(row.first_filing) or row.first_filing > d0 + pd.Timedelta(days=30):
        return None
    if row.sic == 6770 or row.acq_name:
        return True
    if pd.notna(row.first_425) and row.first_425 < d0:
        return True
    return bool(pd.notna(row.first_periodic) and row.first_periodic < d0 - pd.Timedelta(days=10))


def first_close_to_close(day, wait=0):
    """b16: 첫 1분봉 종가 매수 (wait=5면 첫 거래 5분째 1분봉 종가가 첫 1분봉 종가보다 높을 때만 그 값에 매수) -> 15:59 종가."""
    mi, c = day["mi"], day["c"]
    x = int(np.searchsorted(mi, CLOSE, side="right")) - 1
    e = 0
    if wait:
        e = int(np.searchsorted(mi, mi[0] + wait)) - 1
        if e <= 0 or not c[e] > c[0]:
            return None
    if x <= e:
        return None
    return dict(entry_mi=int(mi[e]), entry_px=c[e], exit_mi=int(mi[x]), exit_px=c[x], gross=c[x] / c[e] - 1, reason="마감")


def day1_summary(day):
    """상장일 첫 시가, 15:59 종가, 고가, 저가, 거래대금."""
    mi = day["mi"]
    x = int(np.searchsorted(mi, CLOSE, side="right")) - 1
    sl = slice(0, x + 1)
    return dict(o0=day["o"][0], c0=day["c"][x], h0=day["h"][sl].max(), l0=day["l"][sl].min(),
                dv0=float((day["c"][sl] * day["v"][sl]).sum()))


def open_to_close(day):
    """9:31 1분봉 시가(없으면 9:35 전 첫 1분봉) 매수 -> 15:59 종가."""
    mi = day["mi"]
    e = int(np.searchsorted(mi, 1))
    x = int(np.searchsorted(mi, CLOSE, side="right")) - 1
    if e >= len(mi) or mi[e] > 5 or x <= e:
        return None
    return dict(entry_mi=int(mi[e]), entry_px=day["o"][e], exit_mi=int(mi[x]), exit_px=day["c"][x], gross=day["c"][x] / day["o"][e] - 1,
                reason="마감", stop_dist=np.nan)


def breakout(day, level, t_lo, t_hi, stop_fn, max_stop=None):
    """t_lo~t_hi 사이 1분봉 고가가 level을 넘는 순간 max(시가, level)에 매수. stop_fn(j) -> 손절가(None이면 손절 없음)."""
    mi, o, h = day["mi"], day["o"], day["h"]
    j0 = int(np.searchsorted(mi, t_lo))
    for j in range(j0, len(mi)):
        if mi[j] > t_hi:
            return None
        if h[j] > level:
            px = max(o[j], level)
            stop = stop_fn(j, px)
            if stop is not None and (stop >= px or (max_stop is not None and (px - stop) / px > max_stop)):
                return None
            r = simulate(day, j, px, stop, None, until=CLOSE, entry_in_bar=True)
            if r is None:
                return None
            return dict(entry_mi=int(mi[j]), entry_px=px, stop_dist=(px - stop) / px if stop is not None else np.nan, **r)
    return None


def b17(day, h0, stop=True):
    """2일차 9:35~11:30에 1일차 고가 + 0.01을 넘으면 매수. 손절: 매수 전까지 그날 저가 - 0.01 (8% 넘으면 건너뜀)."""
    lo = day["l"]

    def stop_fn(j, px):
        if not stop:
            return None
        return (lo[:j].min() if j > 0 else lo[0]) - TICK
    return breakout(day, h0 + TICK, 5, 120, stop_fn, max_stop=0.08 if stop else None)


def cum_profile(prev_days):
    """직전 날들의 1분봉 -> 0~389분 각 시각까지 누적 거래량의 평균 (390칸)."""
    t = np.arange(390)
    rows = []
    for d in prev_days:
        cv = np.cumsum(d["v"])
        k = np.searchsorted(d["mi"], t, side="right") - 1
        rows.append(np.where(k >= 0, cv[np.maximum(k, 0)], 0.0))
    return np.mean(rows, axis=0)


def b18(day, H, prof, mult=1.5, stop=0.07, t_lo=5, t_hi=330):
    """9:35~15:00에 고가가 H + 0.01을 넘고 그때까지 누적 거래량 ≥ 직전 5일 같은 시각 평균 × mult면 다음 1분봉 시가에 매수.
    매수가 > H × 1.05면 그날은 건너뜀. 손절 매수가 -7% (stop=None이면 없음). 15:59 청산."""
    mi, o, h, v = day["mi"], day["o"], day["h"], day["v"]
    cv = np.cumsum(v)
    level = H + TICK
    j0 = int(np.searchsorted(mi, t_lo))
    for j in range(j0, len(mi) - 1):
        if mi[j] > t_hi:
            return None
        if h[j] > level and cv[j] >= mult * prof[mi[j]] and prof[mi[j]] > 0:
            e = j + 1
            if mi[e] > mi[j] + 3:
                continue
            px = o[e]
            if px > H * 1.05:
                return None
            sp = px * (1 - stop) if stop else None
            r = simulate(day, e, px, sp, None, until=CLOSE)
            if r is None:
                return None
            return dict(entry_mi=int(mi[e]), entry_px=px, stop_dist=stop if stop else np.nan, **r)
    return None


def hold_days(trade, daily, k_entry, n, stop_px=None):
    """스윙 비교: 매수일(k_entry) 당일 결과 trade가 손절이 아니면 n거래일째(매수일 = 1일째) 종가까지. 그 사이 저가가 stop_px면 손절.
    daily: 그 IPO의 하루 표 (k 순, 열 o, l, c)."""
    if trade is None:
        return None
    if trade["reason"] == "손절":
        return trade["gross"]
    px = trade["entry_px"]
    k_end = k_entry + n - 1
    if k_end >= len(daily):
        return None
    for k in range(k_entry + 1, k_end + 1):
        if stop_px is not None and daily.l.iloc[k] <= stop_px:
            return min(daily.o.iloc[k], stop_px) / px - 1
    return daily.c.iloc[k_end] / px - 1
