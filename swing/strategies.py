"""Swing strategy candidates.

Each strategy reads indicators computed once up front and answers four
questions for the engine, always using data up to the close of day i only:

    candidates(i)      stocks to buy at the next open, best first
    initial_stop(t, i, price)  stop price for a stock bought at day i's open
    update_stop(pos, i)        stop for tomorrow after day i's close (or None)
    exit_reason(pos, i)        why to sell at the next open (or None to keep)

Parameters are the commonly published defaults, not tuned on this data, so the
backtest is a fair test rather than a fitted curve.
"""
from __future__ import annotations

import pandas as pd

from .indicators import atr, rsi, sma


class Strategy:
    name = "base"
    label = ""
    description = ""
    max_positions = 5

    def prepare(self, d: dict[str, pd.DataFrame], market: str = "SPY") -> "Strategy":
        self.o, self.h, self.l, self.c = d["Open"], d["High"], d["Low"], d["Close"]
        self.market = market
        self.sma200 = sma(self.c, 200)
        # Market filter: only open new positions while the S&P 500 is above
        # its 200-day average. Most big swing losses happen in bear markets.
        self.market_ok = (self.c[market] > self.sma200[market]).fillna(False)
        self.tradable = [t for t in self.c.columns if t != market]
        self._prepare()
        return self

    def _prepare(self) -> None:
        pass

    def candidates(self, i: int) -> list[str]:
        raise NotImplementedError

    def initial_stop(self, ticker: str, i: int, price: float) -> float | None:
        return None

    def update_stop(self, pos: dict, i: int) -> float | None:
        return pos.get("stop")

    def exit_reason(self, pos: dict, i: int) -> str | None:
        return None


class Pullback(Strategy):
    """Buy a short, sharp dip in a stock that is in a long uptrend; sell on the bounce.

    Larry Connors' RSI(2) pullback: close above its 200-day average, 2-day RSI
    below 10. Sell when the close gets back above the 5-day average, after 10
    trading days at most, or at an 8% stop loss.
    """
    name = "pullback"
    label = "눌림목 매수"
    description = ("200일 평균 위에 있는 상승 종목이 2~3일 급하게 빠지면(RSI(2)<10) 사고, "
                   "5일 평균 위로 반등하면 팝니다. 최대 10거래일 보유, 손절 -8%.")
    stop_pct = 0.08
    max_days = 10

    def _prepare(self) -> None:
        self.rsi2 = rsi(self.c, 2)
        self.sma5 = sma(self.c, 5)

    def candidates(self, i: int) -> list[str]:
        if not self.market_ok.iloc[i]:
            return []
        c, r = self.c.iloc[i], self.rsi2.iloc[i]
        ok = (c > self.sma200.iloc[i]) & (r < 10)
        picks = r[ok].reindex(self.tradable).dropna()
        return list(picks.sort_values().index)

    def initial_stop(self, ticker: str, i: int, price: float) -> float:
        return price * (1 - self.stop_pct)

    def exit_reason(self, pos: dict, i: int) -> str | None:
        t = pos["ticker"]
        if self.c[t].iloc[i] > self.sma5[t].iloc[i]:
            return "반등 (5일 평균 회복)"
        if i - pos["entry_i"] + 1 >= self.max_days:
            return "보유기간 만료"
        return None


class Breakout(Strategy):
    """Buy a stock making a new 55-day high in an uptrend; ride it with a trailing stop.

    Classic Donchian / turtle style breakout. Ranked by 6-month momentum when
    more stocks break out than there are free slots. The stop starts 2 ATR
    below the entry and trails 3 ATR below the highest close since entry.
    """
    name = "breakout"
    label = "신고가 돌파"
    description = ("200일 평균 위에서 55일 신고가를 낸 종목을 사고, 고점 대비 3 ATR(평소 하루 "
                   "변동폭의 3배) 밀리면 팝니다. 처음 손절선은 매수가 - 2 ATR.")

    def _prepare(self) -> None:
        self.high55 = self.c.rolling(55, min_periods=55).max()
        self.atr20 = atr(self.h, self.l, self.c, 20)
        self.mom126 = self.c / self.c.shift(126) - 1

    def candidates(self, i: int) -> list[str]:
        if not self.market_ok.iloc[i]:
            return []
        c = self.c.iloc[i]
        ok = (c >= self.high55.iloc[i]) & (c > self.sma200.iloc[i])
        picks = self.mom126.iloc[i][ok].reindex(self.tradable).dropna()
        return list(picks.sort_values(ascending=False).index)

    def initial_stop(self, ticker: str, i: int, price: float) -> float | None:
        a = self.atr20[ticker].iloc[i - 1]
        return price - 2 * a if pd.notna(a) else price * 0.9

    def update_stop(self, pos: dict, i: int) -> float | None:
        t = pos["ticker"]
        a = self.atr20[t].iloc[i]
        if pd.isna(a):
            return pos["stop"]
        return max(pos["stop"], pos["high_close"] - 3 * a)


class MomentumSwing(Strategy):
    """Each Friday hold the strongest stocks of the last 6 months; drop the ones that fade.

    Buy the top 5 by 6-month return (skipping the last week to avoid chasing a
    spike) among stocks above their 50-day average. Sell when a stock falls out
    of the top 15, closes below its 50-day average, or hits a 10% stop loss.
    """
    name = "momentum"
    label = "주간 모멘텀"
    description = ("매주 금요일, 최근 6개월 가장 강한 종목 5개를 보유합니다. 순위가 15위 밖으로 "
                   "밀리거나 50일 평균 아래로 내려가면 팔고, 손절 -10%.")
    stop_pct = 0.10
    keep_rank = 15

    def _prepare(self) -> None:
        self.sma50 = sma(self.c, 50)
        self.score = self.c.shift(5) / self.c.shift(126) - 1
        idx = self.c.index
        week = pd.Series(idx.to_period("W-FRI"), index=idx)
        self.week_end = (week != week.shift(-1)).to_numpy().copy()
        self.week_end[-1] = idx[-1].dayofweek == 4

    def _ranked(self, i: int) -> pd.Series:
        c = self.c.iloc[i]
        ok = c > self.sma50.iloc[i]
        return self.score.iloc[i][ok].reindex(self.tradable).dropna().sort_values(ascending=False)

    def candidates(self, i: int) -> list[str]:
        if not self.week_end[i] or not self.market_ok.iloc[i]:
            return []
        return list(self._ranked(i).index)

    def initial_stop(self, ticker: str, i: int, price: float) -> float:
        return price * (1 - self.stop_pct)

    def exit_reason(self, pos: dict, i: int) -> str | None:
        t = pos["ticker"]
        if self.c[t].iloc[i] < self.sma50[t].iloc[i]:
            return "50일 평균 이탈"
        if self.week_end[i] and t not in self._ranked(i).index[: self.keep_rank]:
            return "모멘텀 순위 하락"
        return None


ALL = [Pullback, Breakout, MomentumSwing]
