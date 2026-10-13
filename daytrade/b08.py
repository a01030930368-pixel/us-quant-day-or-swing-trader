"""규칙서 b08: IPO 첫날 매수 -> 15:59 종가 매도.

day = 그날 1분봉 dict(mi, o, h, l, c, v). mi = 9:30부터 몇 분째. 첫 거래 시각은 날마다 다르다(큰 IPO는 11~13시).
"""
import re

import numpy as np

CLOSE = 389


def is_spac_like(ticker, day):
    """U3: 유닛·워런트·권리, 또는 10달러 근처에서 거의 안 움직인 스팩."""
    if re.search(r"[.\-/ ]", ticker) or (len(ticker) == 5 and ticker[-1] in "UWR"):
        return True
    o0 = day["o"][0]
    return 9.8 <= o0 <= 10.2 and (day["h"].max() - day["l"].min()) / o0 < 0.02


def bad_prints(day, k=3.0, w=7):
    """자료 오류: 어느 1분봉의 시가·고가·저가·종가가 앞뒤 w분 종가 중앙값의 k배 위나 1/k 아래.
    (99999.99 같은 자리채움 값, 한 봉만 튄 잘못된 체결. 진짜 급등은 여러 분에 걸쳐 오르므로 걸리지 않는다)"""
    c = day["c"]
    pad = np.pad(c, w, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1), axis=1)
    px = np.vstack([day["o"], day["h"], day["l"], c])
    if ((px > med * k) | (px < med / k)).any():
        return True
    # 값 자체가 한순간 3배 넘게 바뀌어 그대로 이어지는 경우 (예: 139999.99로 시작해 나중에 정상 가격)
    lv = np.r_[day["o"][0], c]
    return bool((np.abs(np.diff(np.log(lv))) > np.log(k)).any())


def dv60(day):
    """첫 거래부터 60분 거래대금."""
    m = day["mi"] < day["mi"][0] + 60
    return float((day["c"][m] * day["v"][m]).sum())


def trade(day, mode="I1", stop=None, delay=0, cond=None):
    """mode I1: 첫 1분봉 종가 매수, I2: 첫 1분봉 시가 매수,
    delay>0: 첫 거래 delay분 뒤 1분봉 시가 매수. cond='up'/'down': 그때 가격(직전 봉 종가)이 첫 시가보다 높/낮을 때만.
    매도는 15:59 이하 마지막 1분봉 종가. stop: 저가가 매수가*(1-stop) 이하가 되면 그 값(시가가 더 낮으면 시가)에 매도.
    시가 매수면 매수한 봉부터, 종가 매수(I1)면 다음 봉부터 본다."""
    mi, o, l, c = day["mi"], day["o"], day["l"], day["c"]
    m0 = mi[0]
    if m0 > 330:  # B2: 15:00 이후 첫 거래면 건너뜀
        return None
    if delay:
        e = int(np.searchsorted(mi, m0 + delay))
        if e >= len(mi) or mi[e] > CLOSE:
            return None
        if (cond == "up" and not c[e - 1] > o[0]) or (cond == "down" and not c[e - 1] < o[0]):
            return None
        px, first_check = o[e], e
    elif mode == "I2":
        e, px, first_check = 0, o[0], 0
    else:
        e, px, first_check = 0, c[0], 1
    j_end = int(np.searchsorted(mi, CLOSE, side="right")) - 1
    reason, out = "마감", c[j_end]
    if stop is not None:
        sp = px * (1 - stop)
        hit = np.nonzero(l[first_check:j_end + 1] <= sp)[0]
        if len(hit):
            j = first_check + hit[0]
            reason, out = "손절", min(o[j], sp)
    return dict(entry_mi=int(mi[e]), entry_px=px, exit_px=out, gross=out / px - 1, reason=reason)
