"""'눌림목 매매' from a Korean YouTube video (진짜 트레이딩), rulebook 01.

A stock breaks out of a long sideways range on a big, high-volume up day (the
"기준봉"). It then rests: volume dries up and the closes hold the top of the
old range (the support line). Buy the second dip after the support has held
once. Stop just under the support, sell half at 1R and the rest at 2R.

All numbers are from /mnt/project-files/rulebooks/01-눌림목-진짜트레이딩.md
(tags there: [원전] said in the video, [해석] our numbers, [추가] added for
automation), fixed before the first run. The rulebook's five test variants:

P1  second dip (B), tight stop, half at 1R + rest at 2R (default)
P2  B, wide stop at the bottom of the range
P3  B, tight stop, sell at the high of the first rise
P4  limit buy at the support line on the first dip (A), tight stop, 1R/2R
P5  P1, but rank by long shallow rests first

Engine limits [근사]: daily bars, so the second-dip buy is at the next open
(the video buys during the day), and the 0.05% slippage in the rulebook is
inside the usual 0.25% per-side cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import sma
from .strategies import Strategy

BASE_DAYS = 40          # sideways range before the breakout
BASE_WIDTH = 0.20       # (high - low) / low of that range
BODY_MIN = 0.04         # breakout candle body
VOL_MULT = 2.0          # breakout volume vs 50-day average
SETUP_DAYS = 20         # the breakout must be within the last 20 trading days
DRY_VOL = 0.5           # rest volume <= half the breakout volume
DOWN_VOL = 1.5          # no down day in the rest above 1.5x the 50-day average volume
BREAK = 0.02            # a close 2% under the support kills the setup
TOUCH_BAND = 0.02       # low within 2% of the support counts as a touch
MAX_TOUCHES = 3         # 3 or more touches: everyone is watching it, skip
NEAR = 0.03             # first support: low within 3% above the support
BOUNCE = 0.03           # bounce: close 3% above the first support low
DIP = 0.03              # second dip: low 3% under the bounce high
PIVOT = 5               # swing high/low: higher than 5 days on each side
TREND_DAYS = 120
MIN_DOLLAR_VOL = 20e6
MIN_PRICE = 10.0
MAX_RISK = 0.08         # skip when the stop is more than 8% away
RISK_PER_TRADE = 0.01   # lose at most 1% of the account per trade
HARD_STOP = 0.10
TIME_EXIT = 20


def _pivot_confirmed(x: pd.DataFrame, highs: bool) -> pd.DataFrame:
    """Pivot value on the day it is confirmed (PIVOT days after it), NaN elsewhere."""
    w = 2 * PIVOT + 1
    ext = x.rolling(w, center=True, min_periods=w).max() if highs else \
        x.rolling(w, center=True, min_periods=w).min()
    piv = x.where(x == ext)
    return piv.shift(PIVOT)


def pullback_setups(o, h, l, c, v, method: str = "B", rank: str = "volume", body: float = BODY_MIN,
                    vol_mult: float = VOL_MULT, base_days: int = BASE_DAYS) -> dict[str, pd.DataFrame]:
    """Signal days with their support, range bottom and first-rise high.

    method "B": the second dip after a bounce off the support (buy next open).
    method "A": any resting day above the support before it was first touched
    (a limit order at the support line for the next day).
    rank "volume": breakout volume / 50-day average; "rest": long, shallow rests first.
    Returns frames rank (NaN on non-signal days), support, floor, peak.
    """
    support = c.rolling(base_days, min_periods=base_days).max().shift(1)
    floor = l.rolling(base_days, min_periods=base_days).min().shift(1)
    top = h.rolling(base_days, min_periods=base_days).max().shift(1)
    avg50 = v.rolling(50, min_periods=50).mean().shift(1)
    breakout = ((c > support) & ((c - o) / o >= body) & (v >= vol_mult * avg50)
                & ((top - floor) / floor <= BASE_WIDTH))
    rising = pd.DataFrame(True, index=c.index, columns=c.columns)
    for n in (20, 60, 120):
        m = sma(c, n)
        rising &= m > m.shift(10)
    liquid = ((c * v).rolling(20, min_periods=20).mean() >= MIN_DOLLAR_VOL) & (c >= MIN_PRICE)
    ph, pl = _pivot_confirmed(h, True), _pivot_confirmed(l, False)

    shape = c.shape
    out = {k: np.full(shape, np.nan) for k in ("rank", "support", "floor", "peak")}
    for j, t in enumerate(c.columns):
        O, H, L, C, V = (x[t].to_numpy() for x in (o, h, l, c, v))
        BO, S0, F0, A50 = breakout[t].to_numpy(), support[t].to_numpy(), floor[t].to_numpy(), avg50[t].to_numpy()
        RI, LQ, PH, PL = rising[t].to_numpy(), liquid[t].to_numpy(), ph[t].to_numpy(), pl[t].to_numpy()
        highs, lows = [], []                       # (day, value) of confirmed pivots
        alive = False
        for i in range(len(C)):
            if not np.isnan(PH[i]):
                highs = (highs + [(i - PIVOT, PH[i])])[-2:]
            if not np.isnan(PL[i]):
                lows = (lows + [(i - PIVOT, PL[i])])[-2:]
            if BO[i]:
                alive, b, S, F = True, i, S0[i], F0[i]
                bvol, vmult = V[i], V[i] / A50[i]
                peak, n, vsum, bad, touches, phase = H[i], 0, 0.0, False, 0, 0
                continue
            if not alive:
                continue
            if i - b > SETUP_DAYS or C[i] < S * (1 - BREAK):
                alive = False
                continue
            if H[i] > peak:                        # still in the first rise: the rest starts over
                peak, n, vsum, bad, touches, phase = H[i], 0, 0.0, False, 0, 0
                continue
            n += 1
            vsum += V[i]
            if C[i] < C[i - 1] and V[i] > DOWN_VOL * A50[i]:
                bad = True
            if abs(L[i] - S) <= TOUCH_BAND * S:
                touches += 1
            trend = (len(highs) == 2 and len(lows) == 2 and highs[1][1] > highs[0][1]
                     and lows[1][1] > lows[0][1] and min(highs[0][0], lows[0][0]) >= i - TREND_DAYS)
            ok = (vsum / n <= DRY_VOL * bvol and not bad and touches < MAX_TOUCHES
                  and RI[i] and LQ[i] and trend)
            fire = False
            if method == "A":
                fire = phase == 0 and C[i] > S and ok
            if phase == 0 and L[i] <= S * (1 + NEAR) and C[i] > S:
                phase, low1 = 1, L[i]
            elif phase == 1:
                if C[i] >= low1 * (1 + BOUNCE):
                    phase, bounce_high = 2, H[i]
                elif L[i] < low1 and C[i] > S:
                    low1 = L[i]
            elif phase == 2:
                bounce_high = max(bounce_high, H[i])
                if L[i] <= bounce_high * (1 - DIP) and L[i] > low1 and method == "B":
                    fire, alive = ok, False        # one try per setup
            if fire:
                depth = max((peak - C[i]) / peak, 0.01)
                out["rank"][i, j] = vmult if rank == "volume" else n / depth
                out["support"][i, j], out["floor"][i, j], out["peak"][i, j] = S, F, peak
    return {k: pd.DataFrame(a, index=c.index, columns=c.columns) for k, a in out.items()}


class PullbackP1(Strategy):
    name = "pullback_p1"
    label = "P1 두 번째 눌림 + 좁은 손절 + 1R·2R"
    description = ("40일 박스를 거래량 2배·몸통 4% 장대양봉으로 돌파한 뒤, 거래량이 줄며 지지선(박스 최고 "
                   "종가)을 지키고, 한 번 지지·반등한 뒤 두 번째로 눌린 날 종가를 보고 다음 날 시가에 삽니다. "
                   "종가가 지지선 아래이고 몸통 절반 이상이 지지선 아래면 손절, 1R에 절반·2R에 나머지 익절, "
                   "20거래일 시간 청산. 한 번에 계좌 1%만 잃도록 비중을 정하고 최대 5종목.")
    max_positions = 5
    method = "B"
    wide = False
    target = "2R"
    rank_by = "volume"
    body, vol_mult, base_days = BODY_MIN, VOL_MULT, BASE_DAYS

    def _prepare(self) -> None:
        cols = self.tradable
        s = pullback_setups(*(x[cols] for x in (self.o, self.h, self.l, self.c, self.v)),
                            method=self.method, rank=self.rank_by, body=self.body,
                            vol_mult=self.vol_mult, base_days=self.base_days)
        self.rank, self.sup, self.flo, self.pk = s["rank"], s["support"], s["floor"], s["peak"]
        self.info: dict[str, tuple[float, float, float]] = {}
        self.trade: dict[str, dict] = {}

    def candidates(self, i: int) -> list[str]:
        r = self.rank.iloc[i].dropna()
        self.info = {t: (self.sup[t].iloc[i], self.flo[t].iloc[i], self.pk[t].iloc[i]) for t in r.index}
        return list(r.sort_values(ascending=False).index)

    def _stop_level(self, t: str) -> float:
        S, F, _ = self.info[t]
        return F if self.wide else S * (1 - BREAK)

    def weight(self, t: str, i: int) -> float:
        guess = self.info[t][0] if self.method == "A" else self.c[t].iloc[i]
        risk = max((guess - self._stop_level(t)) / guess, 0.005)
        return min(RISK_PER_TRADE / risk, 1 / self.max_positions)

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        S, F, peak = self.info[t]
        R = price - self._stop_level(t)
        if R <= 0 or R / price > MAX_RISK:
            return False
        if self.target == "peak" and peak - price < R:
            return False
        self.trade[t] = {"S": S, "F": F, "peak": peak, "R": R}
        return True

    def initial_stop(self, t: str, i: int, price: float) -> float:
        return self.trade[t]["F"] if self.wide else price * (1 - HARD_STOP)

    def update_stop(self, pos: dict, i: int) -> float | None:
        if pos.get("half_sold"):
            return max(pos["stop"], pos["entry_price"])
        return pos["stop"]

    def take_profit(self, pos: dict, i: int):
        tr = self.trade[pos["ticker"]]
        if self.target == "peak":
            return tr["peak"], 1.0, "직전 고점 익절"
        e, R = pos["entry_price"], tr["R"]
        if not pos.get("half_sold"):
            if pos["high_max"] >= e + R:
                pos["half_sold"] = True
            return e + R, 0.5, "1R 절반 익절"
        return e + 2 * R, 1.0, "2R 익절"

    def exit_reason(self, pos: dict, i: int) -> str | None:
        t = pos["ticker"]
        if not self.wide:
            S = self.trade[t]["S"]
            o, c = self.o[t].iloc[i], self.c[t].iloc[i]
            if c < S and (o + c) / 2 < S:
                return "지지선 이탈 (몸통 절반 아래)"
        if i - pos["entry_i"] + 1 >= TIME_EXIT:
            return "20거래일 시간 청산"
        return None


class PullbackP2(PullbackP1):
    name = "pullback_p2"
    label = "P2 두 번째 눌림 + 넓은 손절(박스 하단) + 1R·2R"
    description = "P1과 같고, 손절만 박스 하단(40일 최저가)에 걸어 둡니다."
    wide = True


class PullbackP3(PullbackP1):
    name = "pullback_p3"
    label = "P3 두 번째 눌림 + 좁은 손절 + 직전 고점 익절"
    description = "P1과 같고, 익절을 돌파 뒤 1차 상승 고점에서 한 번에 합니다(고점까지 남은 폭이 손절폭보다 작으면 안 삼)."
    target = "peak"


class PullbackP4(PullbackP1):
    name = "pullback_p4"
    label = "P4 첫 눌림에 지지선 지정가 매수 + 좁은 손절 + 1R·2R"
    description = "지지선에 처음 닿기 전, 다음 날 지지선 가격에 지정가 매수를 겁니다. 나머지는 P1과 같습니다."
    method = "A"
    order_all = True

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        S = self.info[t][0]
        if open_ <= S:
            return open_
        return S if self.l[t].iloc[i] <= S else None


class PullbackP5(PullbackP1):
    name = "pullback_p5"
    label = "P5 P1 + 오래·얕게 쉰 종목 먼저"
    description = "P1과 같고, 여러 종목이 동시에 나오면 쉰 기간이 길고 눌림이 얕은 종목부터 삽니다."
    rank_by = "rest"


def _sweep(param: str, value, text: str) -> type:
    """P1 with one breakout number changed: the rulebook's sensitivity check."""
    tag = str(value).replace(".", "")
    return type(f"PullbackP1_{param}_{tag}", (PullbackP1,), {
        "name": f"pullback_p1_{param}_{tag}", "label": f"P1 민감도: {text}",
        "description": f"P1과 같고 {text}만 바꿉니다.", param: value})


SWEEP = [_sweep("body", 0.03, "기준봉 몸통 3%"), _sweep("body", 0.06, "기준봉 몸통 6%"),
         _sweep("vol_mult", 1.5, "거래량 1.5배"), _sweep("vol_mult", 3.0, "거래량 3배"),
         _sweep("base_days", 20, "박스 20일"), _sweep("base_days", 60, "박스 60일")]
PULLBACK_YT = [PullbackP1, PullbackP2, PullbackP3, PullbackP4, PullbackP5, *SWEEP]
