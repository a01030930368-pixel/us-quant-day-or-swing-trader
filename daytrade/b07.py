"""규칙서 b07: FOMC 발표 직후 "안도 랠리".

mi = 9:30부터 몇 분째 (14:00 = 270). 시각 t_in 1분봉 시가에 사서 t_out 1분봉 종가에 판다.
"""
import numpy as np


def to_mi(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m - 570


def window(day, t_in, t_out):
    """t_in 1분봉 시가 매수 (3분 안에 거래 없으면 건너뜀), t_out 이하 마지막 1분봉 종가 매도."""
    mi, o, c = day["mi"], day["o"], day["c"]
    i = np.searchsorted(mi, t_in)
    if i >= len(mi) or mi[i] > t_in + 3:
        return None
    j = np.searchsorted(mi, t_out, side="right") - 1
    if j < i:
        return None
    return dict(entry_mi=int(mi[i]), exit_mi=int(mi[j]), entry_px=o[i], exit_px=c[j], gross=c[j] / o[i] - 1)


def price_before(day, t):
    """t 직전에 끝난 1분봉 종가 (5분 안에 거래 없으면 None)."""
    mi, c = day["mi"], day["c"]
    k = np.searchsorted(mi, t) - 1
    return None if k < 0 or mi[k] < t - 5 else c[k]


def busiest_minute(day, lo=120, hi=330):
    """11:30~15:00 사이 거래량이 가장 많은 1분 (발표 시각이 맞는지 확인용)."""
    mi, v = day["mi"], day["v"]
    m = (mi >= lo) & (mi < hi)
    return int(mi[m][np.argmax(v[m])]) if m.any() else None


def spike(day, t):
    """t ~ t+2분 최대 거래량 / t-60 ~ t-10분 거래량 중앙값. 발표가 있었던 시각이면 크게 튄다."""
    mi, v = day["mi"], day["v"]
    now, before = v[(mi >= t) & (mi <= t + 2)], v[(mi >= t - 60) & (mi < t - 10)]
    if not len(now) or not len(before) or np.median(before) <= 0:
        return None
    return float(now.max() / np.median(before))
