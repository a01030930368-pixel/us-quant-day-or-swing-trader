"""규칙서 b04: 쿨라매기 에피소딕 피벗(EP) 당일판 (매수만).

한 종목-하루의 정규장 1분봉(mi = 9:30부터 몇 분째, 거래량 v 포함)으로 매수 시점을 찾고,
당일판은 15:55 청산(b03._exit), E7은 원작처럼 며칠 들고 간다(swing).
"""
import numpy as np

from daytrade.b03 import _exit

DEADLINE = 90    # 11:00까지만 매수
ADR_MULT = 1.5   # 손절폭이 평균 하루 변동폭의 1.5배를 넘으면 건너뜀
VOL_WIN = 20     # 거래량 폭발 확인: 9:30~9:50


def find_entry(day, w, adv_sh, vol_frac, adr, deadline=DEADLINE):
    """첫 w분 고가를 넘는 첫 1분봉에서 매수. 거래량 폭발(U3/B4)이 확인되지 않은 동안은 기다린다.

    9:50 전에는 지난 분들의 누적 거래량이 '20분에 평소 하루의 vol_frac' 속도 이상이어야 하고,
    9:50 이후에는 9:30~9:50 거래량으로 판단한다. 반환: (i, 매수가, 손절가) 또는 None.
    """
    mi, o, h, l, v = day["mi"], day["o"], day["h"], day["l"], day["v"]
    n_rng = np.searchsorted(mi, w)
    if n_rng == 0 or n_rng >= len(mi) or not (adv_sh > 0) or not (adr > 0):
        return None
    rh = h[:n_rng].max()
    need = vol_frac * adv_sh
    cumv = np.cumsum(v)
    v20 = v[mi < VOL_WIN].sum()
    for i in range(n_rng, np.searchsorted(mi, deadline)):
        if h[i] < rh:
            continue
        t = mi[i]
        if t < VOL_WIN:
            if cumv[i - 1] < need * t / VOL_WIN:
                continue
        elif v20 < need:
            return None
        px = max(o[i], rh)
        stop = l[:i].min() - 0.01  # 손절은 그날 저가 (매수 전까지)
        if 1 - stop / px > ADR_MULT * adr:
            return None
        return i, px, stop
    return None


def intraday(day, w, adv_sh, vol_frac, adr, deadline=DEADLINE):
    e = find_entry(day, w, adv_sh, vol_frac, adr, deadline)
    return None if e is None else _exit(day, *e)


def swing(day, i, px, stop, closes_before, nxt, max_days=60):
    """E7: 원작 매도. 첫날은 1분봉으로 손절 확인, 이후는 일봉.

    3일째 종가에 절반 매도하고 손절을 본전으로, 나머지는 종가가 10일 평균 아래로 마감하면 매도, 최대 60거래일.
    closes_before: 매수일 직전 종가들(오래된 것부터, 9개면 충분). nxt: 매수일 다음 날부터의 일봉 dict(o, h, l, c).
    주식분할로 보이는 날(시가/전날 종가가 0.6 미만 또는 1.7 초과)이 오면 전날 종가로 끝낸다 (자료가 분할 보정 안 됨).
    """
    mi, o, l, c = day["mi"], day["o"], day["l"], day["c"]
    s = np.flatnonzero(l[i:] <= stop)
    if len(s):
        k = i + s[0]
        xp = stop if k == i else min(o[k], stop)
        return dict(entry_mi=int(mi[i]), entry_px=px, gross=xp / px - 1, stop_dist=1 - stop / px, days=1, reason="손절")
    out = dict(entry_mi=int(mi[i]), entry_px=px, stop_dist=1 - stop / px)
    closes = list(closes_before) + [c[-1]]
    pos, realized, prev = 1.0, 0.0, c[-1]
    for n in range(len(nxt["o"])):
        dn = n + 2
        O, L, C = nxt["o"][n], nxt["l"][n], nxt["c"][n]
        if not (0.6 <= O / prev <= 1.7):
            return dict(out, gross=realized + pos * (prev / px - 1), days=dn - 1, reason="분할 의심")
        if O <= stop or L <= stop:
            xp = O if O <= stop else stop
            return dict(out, gross=realized + pos * (xp / px - 1), days=dn, reason="손절" if pos == 1 else "절반 후 본전 손절")
        closes.append(C)
        if dn == 3:
            realized += 0.5 * (C / px - 1)
            pos, stop = 0.5, max(stop, px)
        if len(closes) >= 10 and C < np.mean(closes[-10:]):
            return dict(out, gross=realized + pos * (C / px - 1), days=dn, reason="10일선 이탈")
        if dn >= max_days:
            return dict(out, gross=realized + pos * (C / px - 1), days=dn, reason="60일")
        prev = C
    return dict(out, gross=realized + pos * (prev / px - 1), days=len(nxt["o"]) + 1, reason="자료 끝")
