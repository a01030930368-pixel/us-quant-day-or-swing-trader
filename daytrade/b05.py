"""규칙서 b05: 래리 윌리엄스 "웁스(Oops!)" 갭 하락 반등 (매수만).

시가가 전날 저가보다 아래에서 시작한 날, 장중 1분봉 고가가 전날 저가에 닿으면 전날 저가에 산다.
당일판은 15:55 청산(b03._exit), F7은 "처음 이익 난 시가"에 판다(swing).
"""
import numpy as np

from daytrade.b03 import _exit

DEADLINE = 330  # 15:00까지만 매수


def find_entry(day, pl, deadline=DEADLINE):
    """반환: (i, 매수가, 손절가 = 매수 전까지의 그날 저가 바로 아래) 또는 None."""
    mi, o, h, l = day["mi"], day["o"], day["h"], day["l"]
    hit = np.flatnonzero(h[:np.searchsorted(mi, deadline)] >= pl)
    if len(hit) == 0:
        return None
    i = hit[0]
    low = l[:i].min() if i > 0 else l[0]  # 첫 1분봉에서 바로 닿으면 그 봉의 저가
    return i, max(o[i], pl), low - 0.01


def intraday(day, pl, deadline=DEADLINE, use_stop=True):
    e = find_entry(day, pl, deadline)
    if e is None:
        return None
    i, px, stop = e
    t = _exit(day, i, px, stop if use_stop else 0.01)
    if t and not use_stop:
        t["stop_dist"] = 1 - stop / px  # 손절은 없지만 비중은 그날 저가까지 거리로 정함
    return t


def swing(day, i, px, stop, nxt, max_days=10):
    """F7: 매수 후 처음으로 시가가 매수가보다 높게 시작한 날 그 시가에 매도. 그날 저가 손절은 유지, 최대 10거래일.

    nxt: 매수일 다음 날부터의 일봉 dict(o, l, c). 주식분할로 보이는 날이 오면 전날 종가로 끝낸다.
    """
    mi, o, l, c = day["mi"], day["o"], day["l"], day["c"]
    out = dict(entry_mi=int(mi[i]), entry_px=px, stop_dist=1 - stop / px)
    s = np.flatnonzero(l[i:] <= stop)
    if len(s):
        k = i + s[0]
        return dict(out, gross=(stop if k == i else min(o[k], stop)) / px - 1, days=1, reason="손절")
    prev = c[-1]
    for n in range(min(len(nxt["o"]), max_days - 1)):
        dn = n + 2
        O, L, C = nxt["o"][n], nxt["l"][n], nxt["c"][n]
        if not (0.6 <= O / prev <= 1.7):
            return dict(out, gross=prev / px - 1, days=dn - 1, reason="분할 의심")
        if O > px:
            return dict(out, gross=O / px - 1, days=dn, reason="이익 난 시가")
        if O <= stop or L <= stop:
            return dict(out, gross=min(O, stop) / px - 1, days=dn, reason="손절")
        prev = C
    return dict(out, gross=prev / px - 1, days=min(len(nxt["o"]), max_days - 1) + 1,
                reason="10일" if len(nxt["o"]) >= max_days - 1 else "자료 끝")
