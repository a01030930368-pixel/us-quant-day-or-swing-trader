"""Short-term mean reversion: buy a sharp dip in a stock that is still in an uptrend.

The most-tested public swing results on survivorship-free data (delisted
stocks included) come from this family. Our first pullback test (strategies.
Pullback) bought at the next open and lost money; the published versions
below differ mainly in buying only if the stock falls further the next day,
with a limit order below the close. Rules are copied from the sources and
fixed before running (doc chapter 20). Values marked [해석] are where the
source tested a range and we take the middle.

A. Alvarez "3 lower lows" (alvarezquanttrading.com, Simple Ideas for a Mean
   Reversion Strategy with Good Results): Russell 1000, 2004-2014, 22% a year,
   max drawdown -21%.
B. Alvarez RSI(2) with an intraday pullback entry (same site, Mean Reversion
   Entry: At Open vs. Intraday Pullback vs Confirmation).
C. Quantitativo "The Holy Grail still works": RSI(2) below 5, buy at the
   open, sell after a close above the prior high; 10-position version.
"""
from __future__ import annotations

import pandas as pd

from .indicators import atr, rsi, sma
from .strategies import Strategy


class LimitEntry:
    """Mix-in: buy only if the next day trades down to a limit price set from today's close."""
    order_all = True                 # place a limit order for every signal, fill until slots run out

    def limit_price(self, t: str, i: int) -> float:
        raise NotImplementedError

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        limit = self.limit_price(t, i - 1)
        if open_ <= limit:
            return open_                                   # opened below the limit: filled at the open
        return limit if self.l[t].iloc[i] <= limit else None


class ThreeLowerLows(LimitEntry, Strategy):
    name = "three_lower_lows"
    label = "3일 연속 저점 하락 + 지정가"
    description = ("100일 평균 위에 있는 종목이 5일 평균 아래로 내려오고 저가가 3일 연속 낮아지면, "
                   "다음 날 '종가 − 0.5×ATR(10)'에 지정가로 삽니다. 종가가 전날보다 오르면 다음 날 "
                   "시가에 팝니다. 손절 없음. 최대 10종목, 한 종목 10%. 하루 거래대금 1,000만 달러 이상.")
    max_positions = 10                                     # [원전] tested version: 10 x 10%

    def _prepare(self) -> None:
        c, l = self.c, self.l
        self.atr10 = atr(self.h, l, c, 10)
        dollar = (c * self.v).rolling(21, min_periods=21).mean()
        lower = (l < l.shift(1)) & (l.shift(1) < l.shift(2)) & (l.shift(2) < l.shift(3))
        self.signal = ((c > sma(c, 100)) & (c < sma(c, 5)) & lower
                       & (dollar > 10e6) & (c > 1))                        # [원전]
        self.vol100 = c.pct_change().rolling(100, min_periods=100).std()

    def candidates(self, i: int) -> list[str]:
        hit = self.signal.iloc[i].reindex(self.tradable).fillna(False).astype(bool)
        # [해석] the source picks at random among signals; we take the most volatile first.
        return list(self.vol100.iloc[i][hit[hit].index].sort_values(ascending=False).index)

    def limit_price(self, t: str, i: int) -> float:
        return self.c[t].iloc[i] - 0.5 * self.atr10[t].iloc[i]          # [원전]

    def exit_reason(self, pos: dict, i: int) -> str | None:
        t = pos["ticker"]
        if self.c[t].iloc[i] > self.c[t].iloc[i - 1]:
            return "반등 (종가 상승)"                                       # [원전]
        return None


class RSI2Limit(LimitEntry, Strategy):
    name = "rsi2_limit"
    label = "RSI(2) 과매도 + 지정가"
    description = ("종목과 S&P500이 모두 200일 평균 위일 때, 종목의 2일 RSI가 5 아래로 떨어지면 다음 날 "
                   "종가보다 1% 낮은 값에 지정가로 삽니다. 2일 RSI가 70을 넘으면 다음 날 시가에 팝니다. "
                   "손절 없음. 최대 10종목, 한 종목 10%. 변동성이 큰 종목부터.")
    max_positions = 10                                     # [원전]
    rsi_max = 5.0                                          # [해석] tested 2.5-10
    pullback = 0.01                                        # [해석] tested 0.5-2%

    def _prepare(self) -> None:
        c = self.c
        self.rsi2 = rsi(c, 2)
        dollar = (c * self.v).rolling(20, min_periods=20).mean()
        self.signal = (c > self.sma200) & (self.rsi2 < self.rsi_max) & (dollar > 0.5e6) & (c > 1)
        self.vol100 = c.pct_change().rolling(100, min_periods=100).std()

    def candidates(self, i: int) -> list[str]:
        if not self.market_ok.iloc[i]:                     # [원전] S&P 500 above its 200-day average
            return []
        hit = self.signal.iloc[i].reindex(self.tradable).fillna(False).astype(bool)
        return list(self.vol100.iloc[i][hit[hit].index].sort_values(ascending=False).index)  # [원전]

    def limit_price(self, t: str, i: int) -> float:
        return self.c[t].iloc[i] * (1 - self.pullback)

    def exit_reason(self, pos: dict, i: int) -> str | None:
        if self.rsi2[pos["ticker"]].iloc[i] > 70:
            return "반등 (RSI(2) 70 초과)"                                  # [원전]
        return None


class HolyGrailRSI2(Strategy):
    name = "rsi2_prior_high"
    label = "RSI(2) 5 미만 + 전날 고가 돌파 매도"
    description = ("2일 RSI가 5 아래면 다음 날 시가에 삽니다. 종가가 전날 고가를 넘으면 다음 날 시가에 "
                   "팔고, 200일 평균 아래로 마감해도 팝니다. 최대 10종목, 한 종목 10%.")
    max_positions = 10                                     # [원전] the 10-position version

    def _prepare(self) -> None:
        self.rsi2 = rsi(self.c, 2)

    def candidates(self, i: int) -> list[str]:
        r = self.rsi2.iloc[i].reindex(self.tradable)
        r = r[self.c.iloc[i].reindex(r.index) > self.sma200.iloc[i].reindex(r.index)]  # [해석] else it sells at once
        # [근사] the source prefers the smaller of its large caps; we have no market caps,
        # so the most oversold come first.
        return list(r[r < 5].sort_values().index)

    def exit_reason(self, pos: dict, i: int) -> str | None:
        t = pos["ticker"]
        c = self.c[t].iloc[i]
        if c > self.h[t].iloc[i - 1]:
            return "반등 (전날 고가 돌파)"                                  # [원전]
        if c < self.sma200[t].iloc[i]:
            return "200일 평균 이탈"                                        # [원전]
        return None


REVERSION = [ThreeLowerLows, RSI2Limit, HolyGrailRSI2]
