"""워뇨띠 장부에서 뽑은 규칙: '급하게 움직인 뒤 반대로 지정가, 조금 되돌리면 익절'.

장부에서 확인한 특징 (crypto/analysis 참고)
- 직전 15분~4시간 동안 내린 뒤 사고, 오른 뒤 판다 (역추세)
- 거래량이 평소보다 많고 (15분 거래량 1.3배 vs 평소 0.8배) 변동성이 큰 때 매매
- 지정가 주문이 대부분, 손절 주문은 거의 없음, 더 밀리면 추가 매수(물타기)
- 익절은 직전 움직임의 20% 정도 되돌림 (평균 +0.3~0.4%), 이긴 비율 약 79%

시뮬레이션 (1분봉, 전 봉 종가까지만 보고 판단)
- 신호: L분 수익률이 최근 하루 변동성 기준 Z배 이상 + 15분 거래량 V배 이상
- 진입: 신호 종가에서 off 만큼 불리한 곳에 지정가, W분 안에 '가격이 뚫고 지나가야' 체결
- 추가: ladder=1 이면 진입가에서 step 더 밀린 곳에 같은 크기 한 번 더 (물타기)
- 청산: 평균단가 대비 tp 익절 지정가 / sl 손절(시장가) / H분 지나면 시장가
- 같은 1분봉 안에서 익절·손절 둘 다 닿으면 손절로 처리, 진입한 봉에서는 익절 안 함 (보수적)
- 수수료: 지정가 maker, 시장가 taker + 슬리피지, 펀딩비 8시간마다 보유 방향에 부과
"""
import numpy as np
from numba import njit


@njit(cache=True)
def run(o, h, l, c, sig, off, W, ladder, step, tp, sl, H,
        maker, taker, slip, fund8h, lev):
    """sig: +1 매수신호, -1 매도신호, 0 없음 (해당 봉 종가 기준).
    반환: 1분 단위 계좌 가치(1에서 시작), 거래 기록 (진입봉, 청산봉, 방향, 수익률)."""
    n = len(c)
    eq = np.ones(n)
    tr = np.zeros((n // 10 + 10, 4))
    nt = 0
    equity = 1.0
    state = 0  # 0 대기, 1 지정가 대기, 2 보유
    side = 0
    lim = 0.0; lim_until = 0
    avg = 0.0; units = 0.0; add_px = 0.0; added = False
    t_in = 0
    for i in range(1, n):
        fresh = False  # 이번 봉에 진입/추가 체결 -> 같은 봉 익절 금지(봉 안 순서를 모르므로)
        if state == 1:
            # 지정가 체결 확인 (가격이 지정가를 뚫고 지나가야 체결)
            filled = (side > 0 and l[i] < lim) or (side < 0 and h[i] > lim)
            if filled:
                state = 2; avg = lim; t_in = i; fresh = True
                units = 0.5 if ladder == 1 else 1.0
                add_px = lim * (1 - side * step); added = False
                equity -= units * lev * maker * equity
            elif i >= lim_until:
                state = 0
        if state == 2:
            # 추가 매수(물타기)
            if ladder == 1 and not added:
                if (side > 0 and l[i] < add_px) or (side < 0 and h[i] > add_px):
                    avg = (avg * units + add_px * 0.5) / (units + 0.5)
                    units += 0.5; added = True; fresh = True
                    equity -= 0.5 * lev * maker * equity
            exit_px = 0.0; fee = 0.0
            tp_px = avg * (1 + side * tp); sl_px = avg * (1 - side * sl)
            hit_sl = (side > 0 and l[i] <= sl_px) or (side < 0 and h[i] >= sl_px)
            hit_tp = (not fresh) and ((side > 0 and h[i] > tp_px) or (side < 0 and l[i] < tp_px))
            if hit_sl:
                # 갭으로 손절가보다 더 나쁘게 시작하면 시가에 체결
                base = sl_px
                if (side > 0 and o[i] < sl_px) or (side < 0 and o[i] > sl_px):
                    base = o[i]
                exit_px = base * (1 - side * slip); fee = taker
            elif hit_tp:
                exit_px = tp_px; fee = maker
            elif i - t_in >= H and not fresh:
                exit_px = c[i] * (1 - side * slip); fee = taker
            if i % 480 == 0:
                equity -= side * fund8h * units * lev * equity
            if exit_px > 0:
                r = side * (exit_px / avg - 1)
                equity *= (1 + units * lev * r)
                equity -= units * lev * fee * equity
                if nt < tr.shape[0]:
                    tr[nt, 0] = t_in; tr[nt, 1] = i; tr[nt, 2] = side; tr[nt, 3] = units * lev * r
                    nt += 1
                state = 0
                if equity <= 0:
                    equity = 0.0
        if state == 0 and sig[i] != 0 and equity > 0:
            side = sig[i]
            lim = c[i] * (1 - side * off)
            lim_until = i + W
            state = 1
        # 평가금액 (보유 중이면 종가로 평가)
        if state == 2:
            eq[i] = equity * (1 + units * lev * side * (c[i] / avg - 1))
        else:
            eq[i] = equity
    return eq, tr[:nt]


def signals(c, v, L, Z, V):
    """L분 수익률 z점수 + 15분 거래량 배수로 역추세 신호."""
    import pandas as pd
    cs = pd.Series(c); vs = pd.Series(v)
    lr = np.log(cs).diff()
    sd = lr.rolling(1440, min_periods=720).std()
    z = (np.log(cs) - np.log(cs.shift(L))) / (sd * np.sqrt(L))
    vr = vs.rolling(15).sum() / (vs.rolling(1440, min_periods=720).sum() / 96)
    s = np.zeros(len(c), dtype=np.int64)
    ok = (vr >= V).to_numpy()
    zz = z.to_numpy()
    s[(zz <= -Z) & ok] = 1
    s[(zz >= Z) & ok] = -1
    return s
