"""불타기 vs 물타기 vs 한 번에: 같은 진입 신호·같은 청산 규칙에서 '크기 늘리는 방법'만 바꿔 비교.

크기 방식 (mode)
- 0 한 번에: 신호에 전부
- 1 불타기: 1/3 진입, 첫 진입가보다 step 유리해지면 1/3, 2·step 이면 마지막 1/3 (역지정가 = 시장가 비용)
- 2 물타기: 1/3 진입, step 불리해지면 1/3, 2·step 이면 마지막 1/3 (지정가 비용)
청산 (exit_kind)
- 0 목표·손절: 평균단가 대비 +tp 익절(지정가), 첫 진입가 대비 -sl 손절, H분 지나면 정리
- 1 따라가는 손절: 가장 유리했던 가격에서 trail 되돌리면 정리, 첫 진입가 대비 -sl 손절, H분
진입은 신호 다음 봉 시가에 시장가. 같은 봉에서 추가와 청산이 겹치면 청산이 먼저(보수적), 진입·추가한 봉에선 익절 안 함.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def run_scale(o, h, l, c, sig, mode, exit_kind, step, tp, sl, trail, H,
              maker, taker, slip, fund8h):
    n = len(c)
    eq = np.ones(n)
    tr = np.zeros((n // 30 + 10, 4))
    nt = 0
    equity = 1.0
    hold = False
    side = 0; p0 = 0.0; avg = 0.0; units = 0.0; nadd = 0; t_in = 0; best = 0.0
    pending = 0
    for i in range(1, n):
        fresh = False
        if pending != 0 and not hold:
            side = pending; pending = 0
            p0 = o[i] * (1 + side * slip)
            units = 1.0 if mode == 0 else 1.0 / 3
            avg = p0; nadd = 0; t_in = i; best = p0; hold = True; fresh = True
            equity -= units * taker * equity
        if hold:
            # 청산 먼저 확인
            sl_px = p0 * (1 - side * sl)
            exit_px = 0.0; fee = 0.0
            if (side > 0 and l[i] <= sl_px) or (side < 0 and h[i] >= sl_px):
                base = sl_px
                if (side > 0 and o[i] < sl_px) or (side < 0 and o[i] > sl_px):
                    base = o[i]
                exit_px = base * (1 - side * slip); fee = taker
            elif exit_kind == 1:
                tr_px = best * (1 - side * trail)
                if (not fresh) and ((side > 0 and l[i] <= tr_px) or (side < 0 and h[i] >= tr_px)):
                    base = tr_px
                    if (side > 0 and o[i] < tr_px) or (side < 0 and o[i] > tr_px):
                        base = o[i]
                    exit_px = base * (1 - side * slip); fee = taker
            else:
                tp_px = avg * (1 + side * tp)
                if (not fresh) and ((side > 0 and h[i] > tp_px) or (side < 0 and l[i] < tp_px)):
                    exit_px = tp_px; fee = maker
            if exit_px == 0.0 and i - t_in >= H:
                exit_px = c[i] * (1 - side * slip); fee = taker
            if exit_px == 0.0 and mode != 0 and nadd < 2:
                k = nadd + 1
                if mode == 1:
                    add_px = p0 * (1 + side * step * k)
                    if (side > 0 and h[i] >= add_px) or (side < 0 and l[i] <= add_px):
                        fill = add_px * (1 + side * slip)
                        avg = (avg * units + fill / 3) / (units + 1.0 / 3); units += 1.0 / 3; nadd += 1
                        equity -= taker / 3 * equity
                else:
                    add_px = p0 * (1 - side * step * k)
                    if (side > 0 and l[i] < add_px) or (side < 0 and h[i] > add_px):
                        avg = (avg * units + add_px / 3) / (units + 1.0 / 3); units += 1.0 / 3; nadd += 1
                        equity -= maker / 3 * equity
            if side > 0:
                best = max(best, h[i])
            else:
                best = min(best, l[i])
            if i % 480 == 0:
                equity -= side * fund8h * units * equity
            if exit_px > 0:
                r = side * (exit_px / avg - 1)
                equity *= (1 + units * r)
                equity -= units * fee * equity
                if nt < tr.shape[0]:
                    tr[nt, 0] = t_in; tr[nt, 1] = i; tr[nt, 2] = side; tr[nt, 3] = units * r
                    nt += 1
                hold = False
                if equity <= 0:
                    equity = 0.0
        if not hold and pending == 0 and sig[i] != 0 and equity > 0:
            pending = sig[i]
        eq[i] = equity * (1 + units * side * (c[i] / avg - 1)) if hold else equity
    return eq, tr[:nt]
