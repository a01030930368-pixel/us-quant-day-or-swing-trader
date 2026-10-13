"""규칙서 b06: 레버리지 ETF 재조정 "장 막판 쏠림".

기준 지수 ETF가 판단 시각까지 전날 종가 대비 ±thr 이상 움직였으면, 그 방향의 레버리지 ETF를
판단 시각 다음 분 시가에 사서 장 마감(15:59 1분봉 종가)에 판다.
mi = 9:30부터 몇 분째 (14:45 = 315).
"""
import numpy as np

CLOSE_BAR = 389  # 15:59 1분봉


def signal(idx_day, pc, thr, t):
    """판단 시각 t 직전 1분봉 종가로 본 하루 움직임과 방향(+1 / -1 / 0)."""
    mi, c = idx_day["mi"], idx_day["c"]
    k = np.searchsorted(mi, t) - 1  # t 직전에 끝난 1분봉 (14:45 판단 -> 14:44 봉)
    if k < 0 or mi[k] < t - 5:  # 판단 시각 근처 거래가 없으면 판단 안 함
        return 0.0, 0
    move = c[k] / pc - 1
    return move, (1 if move >= thr else -1 if move <= -thr else 0)


def hold(etf_day, t, exit_at=None, stop=None):
    """t+1분 1분봉 시가에 매수. exit_at=None이면 15:59 종가, 숫자면 그 분 1분봉 시가에 매도. stop: 매수가 대비 손절 비율."""
    mi, o, l, c = etf_day["mi"], etf_day["o"], etf_day["l"], etf_day["c"]
    i = np.searchsorted(mi, t + 1)
    if i >= len(mi) or mi[i] > t + 6 or mi[i] >= CLOSE_BAR:
        return None
    px = o[i]
    if exit_at is None:
        j = np.searchsorted(mi, CLOSE_BAR, side="right") - 1
        xp, why = c[j], "마감"
    else:
        j = np.searchsorted(mi, exit_at)
        if j >= len(mi):
            j = len(mi) - 1
            xp = c[j]
        else:
            xp = o[j]
            j -= 1  # 손절 확인은 매도 직전 봉까지
        why = "마감"
    if stop is not None:
        sp = px * (1 - stop)
        s = np.flatnonzero(l[i:j + 1] <= sp)
        if len(s):
            k = i + s[0]
            xp, why = (sp if k == i else min(o[k], sp)), "손절"
    return dict(entry_mi=int(mi[i]), entry_px=px, exit_px=xp, gross=xp / px - 1, reason=why)
