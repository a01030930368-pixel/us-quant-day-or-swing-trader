"""'하수구 돌파' (sewer breakout): the second break above the 200-day line.

From a Korean YouTube video (주식탐구소, 2026). The idea: a stock that sat
under its 200-day average for a long time has many trapped holders, so its
first break above the line usually fails while that supply is absorbed. The
buy is the next break above the line, on more volume than the first one.
The video gives no formula text and no sell rule, so the numbers below are
our reading of it, fixed before the first run (doc chapter 26).

Rules [원전 = said in the video, 해석 = our reading, 근사 = engine limit]:
- Long base: in the 250 trading days before the first break, the close was
  under the 200-day average on at least 90% of days [해석: "1년 이동평균을
  1년 6개월 동안 돌파 못 함"].
- First break: close crosses above the 200-day average [원전]. It must then
  close back under the line at least once ("가는 척하다 매물 소화") [원전].
- Signal: within 250 trading days of the first break, a later close back
  above the line on more volume than the first break day [원전: 거래량이 첫
  돌파 거래량을 넘음]. One signal per base.
- Buy: the video buys at the close of the day after the signal [원전]; we
  decide at that close (still above the line) and buy at the next open [근사].
  Most volume growth first when more stocks signal than there are slots.
- 10 positions of 10%, no market filter (the video has none).
A. Sell when a close falls under the 200-day average [해석: the video has no
   sell rule; the line is the whole method].
B. Swing version: sell at +25% (the video's examples show 28-81% open
   upside), at a 10% stop, or after 40 trading days [해석].
C. A, but only stocks that fell at least 50% from their 2-year high before
   the first break [해석: the video's examples fell 70-80%].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategies import Strategy

BASE_DAYS = 250
BASE_SHARE = 0.9
WINDOW = 250
DEEP_DROP = 0.5


def sewer_signals(close: pd.DataFrame, line: pd.DataFrame, volume: pd.DataFrame,
                  deep: bool = False) -> pd.DataFrame:
    """Volume growth (second break volume / first break volume) on signal days, NaN elsewhere."""
    below = (close < line).where(line.notna())
    share = below.rolling(BASE_DAYS, min_periods=BASE_DAYS).mean().shift(1)
    above = (close > line).fillna(False)
    cross = above & ~above.shift(1, fill_value=False)
    drop_ok = close.rolling(BASE_DAYS, min_periods=1).min() <= (
        DEEP_DROP * close.rolling(2 * BASE_DAYS, min_periods=1).max())
    out = np.full(close.shape, np.nan)
    for j, t in enumerate(close.columns):
        c, s, v = close[t].to_numpy(), line[t].to_numpy(), volume[t].to_numpy()
        x, base, deep_ok = cross[t].to_numpy(), share[t].to_numpy(), drop_ok[t].to_numpy()
        start, first_vol, fell = -1, 0.0, False
        for i in range(len(c)):
            if start >= 0 and i - start > WINDOW:
                start = -1
            if x[i]:
                if start < 0:
                    if base[i] >= BASE_SHARE and (deep_ok[i] or not deep):
                        start, first_vol, fell = i, v[i], False
                elif fell and v[i] > first_vol > 0:
                    out[i, j] = v[i] / first_vol
                    start = -1
            elif start >= 0 and c[i] < s[i]:
                fell = True
    return pd.DataFrame(out, index=close.index, columns=close.columns)


class SewerBreakout(Strategy):
    name = "sewer"
    label = "하수구 돌파 (200일선 이탈 시 매도)"
    description = ("1년 가까이 200일선 아래에 있던 종목이 200일선을 한 번 넘었다가 다시 밀린 뒤, "
                   "첫 돌파보다 큰 거래량으로 다시 넘으면 다음 날 종가를 보고 그다음 날 시가에 삽니다. "
                   "종가가 200일선 아래로 내려가면 팝니다. 최대 10종목.")
    max_positions = 10
    deep = False

    def _prepare(self) -> None:
        sig = sewer_signals(self.c[self.tradable], self.sma200[self.tradable],
                            self.v[self.tradable], self.deep)
        self.signal = sig.shift(1)                     # decide at the close of the day after

    def candidates(self, i: int) -> list[str]:
        s = self.signal.iloc[i].dropna()
        if s.empty:
            return []
        still = self.c.iloc[i][s.index] > self.sma200.iloc[i][s.index]
        return list(s[still].sort_values(ascending=False).index)

    def exit_reason(self, pos: dict, i: int) -> str | None:
        t = pos["ticker"]
        if self.c[t].iloc[i] < self.sma200[t].iloc[i]:
            return "200일선 이탈"
        return None


class SewerSwing(SewerBreakout):
    name = "sewer_swing"
    label = "하수구 돌파 스윙 (+25% 익절, -10% 손절, 40일)"
    description = "매수는 위와 같고, +25%에 팔거나 -10%에 손절하거나 40거래일이 지나면 팝니다."
    target = 0.25
    stop_pct = 0.10
    max_days = 40

    def initial_stop(self, ticker: str, i: int, price: float) -> float:
        return price * (1 - self.stop_pct)

    def take_profit(self, pos: dict, i: int):
        return pos["entry_price"] * (1 + self.target), 1.0, "+25% 익절"

    def exit_reason(self, pos: dict, i: int) -> str | None:
        if i - pos["entry_i"] + 1 >= self.max_days:
            return "40거래일 보유 완료"
        return None


class SewerDeep(SewerBreakout):
    name = "sewer_deep"
    label = "하수구 돌파 + 고점 대비 -50% 이상 하락 종목만"
    description = "A 규칙에, 첫 돌파 전에 2년 고점 대비 50% 넘게 빠졌던 종목만 고릅니다."
    deep = True


SEWER = [SewerBreakout, SewerSwing, SewerDeep]
