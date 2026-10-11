"""The second group of swing legends: Darvas, Qullamaggie, Raschke, Kell.

Rules follow chapter 14 of the project doc ("미국 스윙매매 전설 10인의 전략"),
tagged the same way as `legends.py`:

    [원전]  the trader stated the number himself
    [해석]  a third party turned his words into the number
    [근사]  no number exists anywhere; we chose it

Raschke's setups use her buy-stop entries (filled during the day only if the
price reaches the trigger); the others buy at the next open like the first group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import adx, ema, sma
from .strategies import Strategy

TICK = 0.125        # Darvas's 1/8 point


class Darvas(Strategy):
    """Nicolas Darvas: buy the breakout of a box built from a new 52-week high (rulebook 14-1)."""
    name = "darvas"
    label = "다바스 박스 돌파"
    description = ("52주 신고가 뒤 3일간 넘지 못한 고가를 박스 상단, 그 뒤 3일간 깨지지 않은 저가를 "
                   "박스 하단으로 잡고, 종가가 상단을 거래량 2배로 넘으면 다음 날 시가에 삽니다. "
                   "손절은 박스 하단 0.125달러 아래(최대 10%), 새 박스가 생기면 손절선을 새 하단으로 "
                   "올리고, 3주 동안 이익이 없으면 팝니다. 5종목 같은 금액.")
    max_positions = 5                # [근사]
    vol_mult = 2.0                   # [원전] "at least double" / base = 50-day avg [근사]
    max_stop = 0.10                  # [원전]
    time_stop = 15                   # [원전] 3 weeks (a guideline)

    def _prepare(self) -> None:
        c, h, l, v = self.c, self.h, self.l, self.v
        hi252 = h.rolling(252, min_periods=252).max()                    # [해석] 52-week lookback
        lo252 = l.rolling(252, min_periods=252).min()
        avg_vol = v.shift(1).rolling(50, min_periods=50).mean()
        n, m = c.shape
        top = np.full((n, m), np.nan)
        bottom = np.full((n, m), np.nan)
        hv, lv, newhi = h.to_numpy(), l.to_numpy(), (h >= hi252).to_numpy()
        for j in range(m):
            # 0 wait for a new high, 1 forming the top, 2 forming the bottom, 3 box done
            state, cand_top, top_cnt, top_lows = 0, np.nan, 0, []
            cand_bot, bot_cnt, box_t, box_b = np.nan, 0, np.nan, np.nan
            for i in range(n):
                hi, lo = hv[i, j], lv[i, j]
                if np.isnan(hi) or np.isnan(lo):
                    continue
                if newhi[i, j] and (state in (0, 3) and (np.isnan(box_t) or hi > box_t)
                                    or state in (1, 2) and hi > cand_top):
                    state, cand_top, top_cnt, top_lows = 1, hi, 0, []     # a new high starts a top
                elif state == 1:
                    if hi >= cand_top:                                    # a tie restarts [해석]
                        cand_top, top_cnt, top_lows = hi, 0, []
                    else:
                        top_cnt += 1
                        top_lows.append(lo)
                        if top_cnt == 3:                                  # 3 days not exceeded [원전]
                            k = int(np.argmin(top_lows))
                            state, cand_bot, bot_cnt = 2, top_lows[k], 2 - k
                elif state == 2:
                    if lo < cand_bot:
                        cand_bot, bot_cnt = lo, 0
                    else:
                        bot_cnt += 1
                if state == 2 and bot_cnt >= 3:                           # 3 days not undercut [원전]
                    state, box_t, box_b = 3, cand_top, cand_bot
                elif state == 3 and lo < box_b:                           # box broken downward
                    state, box_t, box_b = 0, np.nan, np.nan
                top[i, j], bottom[i, j] = box_t, box_b
        self.top = pd.DataFrame(top, index=c.index, columns=c.columns)
        self.bottom = pd.DataFrame(bottom, index=c.index, columns=c.columns)
        doubled = hi252 >= 2 * lo252                                      # [원전 per secondary]
        signal = ((c > self.top + TICK) & (c.shift(1) <= self.top.shift(1) + TICK)
                  & (v >= self.vol_mult * avg_vol) & doubled)
        self.signal = signal.mul(self.market_ok, axis=0).astype(bool)    # [근사] S&P 200-day
        self.vol_ratio = v / avg_vol

    def candidates(self, i: int) -> list[str]:
        hit = self.signal.iloc[i].reindex(self.tradable).fillna(False).astype(bool)
        return list(self.vol_ratio.iloc[i][hit[hit].index].sort_values(ascending=False).index)

    def initial_stop(self, t: str, i: int, price: float) -> float:
        return max(self.bottom[t].iloc[i - 1] - TICK, price * (1 - self.max_stop))

    def update_stop(self, pos: dict, i: int) -> float | None:
        t = pos["ticker"]
        pos.setdefault("box_top", self.top[t].iloc[pos["entry_i"] - 1])
        top, bot = self.top[t].iloc[i], self.bottom[t].iloc[i]
        if pd.notna(top) and top > pos["box_top"]:                        # a new, higher box
            pos["box_top"] = top
            return max(pos["stop"], bot - TICK)                           # [원전]
        return pos["stop"]

    def exit_reason(self, pos: dict, i: int):
        days = i - pos["entry_i"] + 1
        if days >= self.time_stop and self.c[pos["ticker"]].iloc[i] <= pos["entry_price"]:
            return "3주 무수익"
        return None


class Qullamaggie(Strategy):
    """Kristjan Kullamägi: breakout of a tight base after a big move (rulebook 14-2)."""
    name = "qullamaggie"
    label = "쿨라매기 돌파"
    description = ("최근 1·3·6개월 수익률 상위 10% 안에 들고 1~3개월 안에 30% 이상 오른 종목이, "
                   "2주~2개월 쉬며 저점을 높이다 그 고점을 종가로 넘으면 다음 날 시가에 삽니다. "
                   "손절은 돌파일 저가(하루 평균 변동폭보다 멀면 안 삼), 3일째 이익이면 1/3 매도 후 "
                   "본전, 나머지는 10일선 아래 마감 시 매도. 손절 시 계좌 0.5% 손실 비중(최대 15%, 10종목).")
    max_positions = 10               # [근사]
    risk_per_trade = 0.005           # [원전] 0.25-1%, mostly 0.3-0.5%
    max_weight = 0.15                # [원전] 10-15%
    top_pct = 0.10                   # [근사] (his: top 1-2% of all US stocks)
    prior_move = 0.30                # [원전]
    adr_min = 0.03                   # [근사] (fans: 5%)

    def _prepare(self) -> None:
        c, h, l = self.c, self.h, self.l
        t = self.tradable
        ranks = [(c[t] / c[t].shift(n) - 1).rank(axis=1, pct=True) for n in (21, 63, 126)]
        leader = (ranks[0] >= 1 - self.top_pct) | (ranks[1] >= 1 - self.top_pct) \
            | (ranks[2] >= 1 - self.top_pct)                              # [원전] 1/3/6 months
        leader = leader.reindex(columns=c.columns).fillna(False)
        self.adr = (h / l).rolling(20, min_periods=20).mean() - 1         # [원전] formula
        self.sma10, self.sma20 = sma(c, 10), sma(c, 20)
        hv, lv, cv = h.to_numpy(), l.to_numpy(), c.to_numpy()
        piv = np.full(c.shape, np.nan)
        pre = (leader & (self.adr >= self.adr_min) & (self.sma10 > self.sma20)
               & (c > h.shift(1).rolling(10, min_periods=10).max()))
        pre = pre.mul(self.market_filter(), axis=0).astype(bool)
        for i, j in zip(*np.nonzero(pre.to_numpy())):
            if i < 110:
                continue
            win = hv[i - 40:i - 9, j]                                     # base 10-40 days [원전 2w-2m]
            if np.isnan(win).all():
                continue
            p = i - 40 + int(np.nanargmax(win))
            peak = hv[p, j]
            if np.nanmax(hv[p:i, j]) > peak or cv[i - 1, j] > peak:
                continue
            start_low = np.nanmin(lv[max(0, p - 63):p + 1, j])            # move within 3 months
            if peak < (1 + self.prior_move) * start_low:
                continue
            base_low = np.nanmin(lv[p:i, j])
            if 1 - base_low / peak > 0.25:                                # [근사] base depth
                continue
            mid = p + (i - p) // 2                                        # higher lows [근사]
            if np.nanmin(lv[mid:i, j]) <= np.nanmin(lv[p:mid, j]):
                continue
            if cv[i, j] > peak:
                piv[i, j] = peak
        self.pivot = pd.DataFrame(piv, index=c.index, columns=c.columns)

    def market_filter(self) -> pd.Series:
        """S&P 500 above its 20-day average and 10-day above 20-day [근사; his: Nasdaq/QQQ]."""
        m = self.market
        s10, s20 = sma(self.c[[m]], 10)[m], sma(self.c[[m]], 20)[m]
        return (self.c[m] > s20) & (s10 > s20)

    def candidates(self, i: int) -> list[str]:
        row = self.pivot.iloc[i].reindex(self.tradable).dropna()
        score = (self.c.iloc[i] / self.c.iloc[max(0, i - 63)] - 1)[row.index]
        return list(score.sort_values(ascending=False).index)

    def weight(self, t: str, i: int) -> float:
        c, lo = self.c[t].iloc[i], self.l[t].iloc[i]
        dist = (c - lo) / c
        return self.max_weight if dist <= 0 else min(self.max_weight, self.risk_per_trade / dist)

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        stop = self.l[t].iloc[i - 1]
        return stop < price and (price - stop) / price <= self.adr[t].iloc[i - 1]   # [원전] <= 1 ADR

    def initial_stop(self, t: str, i: int, price: float) -> float:
        return self.l[t].iloc[i - 1]                                      # [근사] breakout-day low

    def exit_reason(self, pos: dict, i: int):
        t = pos["ticker"]
        c = self.c[t].iloc[i]
        day = i - pos["entry_i"] + 1
        if c < self.sma10[t].iloc[i]:
            return "10일선 아래 마감"                                       # [원전]
        if day == 3 and c > pos["entry_price"] and not pos.get("third_sold"):
            pos["third_sold"] = True
            pos["stop"] = max(pos["stop"], pos["entry_price"])            # [원전] then breakeven
            return ("3일째 1/3 매도", 1 / 3)                                # [원전] 1/3-1/2 after 3-5 days
        return None


class HolyGrail(Strategy):
    """Linda Raschke's Holy Grail: first pullback to the 20 EMA in a strong trend (rulebook 14-3)."""
    name = "holy_grail"
    label = "래쉬키 홀리 그레일"
    description = ("ADX(14)가 30 넘는 상승 추세에서 주가가 처음 20일 지수이동평균까지 눌리면, 다음 날 "
                   "그 날 고가를 넘을 때 삽니다(예약 매수). 손절은 눌린 날 저가, 목표는 직전 20일 "
                   "고점, 10일 지나면 매도. 7종목 같은 금액.")
    max_positions = 7                # [원전] at most 7-8 positions
    adx_min = 30                     # [원전]
    max_days = 10                    # [근사]

    def _prepare(self) -> None:
        c, h, l = self.c, self.h, self.l
        a, pdi, mdi = adx(h, l, c, 14)
        self.ema20 = ema(c, 20)
        touch = l <= self.ema20                                           # [해석]
        first = touch & ~(touch.shift(1, fill_value=False).rolling(10, min_periods=1).max()
                          .astype(bool))                                  # first in 10 days [근사]
        strong = (a.shift(1) > self.adx_min) & (pdi.shift(1) > mdi.shift(1))   # [원전] / up-trend [근사]
        self.signal = (first & strong & (c > 0)).astype(bool)
        self.prior_high = h.shift(1).rolling(20, min_periods=20).max()
        self.adx = a

    def candidates(self, i: int) -> list[str]:
        hit = self.signal.iloc[i].reindex(self.tradable).fillna(False).astype(bool)
        return list(self.adx.iloc[i][hit[hit].index].sort_values(ascending=False).index)

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        trigger = self.h[t].iloc[i - 1]                                   # [원전] above prior bar high
        if open_ >= trigger:
            return open_
        return trigger if high >= trigger else None

    def initial_stop(self, t: str, i: int, price: float) -> float:
        return self.l[t].iloc[i - 1]                                      # [해석] swing low

    def take_profit(self, pos: dict, i: int):
        t = pos["ticker"]
        pos.setdefault("target", self.prior_high[t].iloc[pos["entry_i"] - 1])
        if pos["target"] <= pos["entry_price"]:
            return None
        return pos["target"], 1.0, "직전 고점 도달"                         # [원전] retest of the high

    def exit_reason(self, pos: dict, i: int):
        if i - pos["entry_i"] + 1 >= self.max_days:
            return "10일 보유 만료"
        return None


class TurtleSoupPlusOne(Strategy):
    """Raschke & Connors: a failed new 20-day low, bought back the next day (rulebook 14-3)."""
    name = "turtle_soup"
    label = "래쉬키 터틀 수프 플러스 원"
    description = ("20일 신저가를 내고 이전 20일 저가(3일 이상 전) 아래에서 끝난 다음 날, 주가가 이전 "
                   "20일 저가 위로 돌아오면 삽니다(예약 매수). 손절은 전날 저가, 이후 전날 저가로 "
                   "따라 올리고 5일 지나면 매도. 7종목 같은 금액.")
    max_positions = 7                # [원전]
    max_days = 5                     # [근사] "a few hours to a few days"

    def _prepare(self) -> None:
        c, l = self.c, self.l
        prev_low = l.shift(1).rolling(20, min_periods=20).min()
        age = l.shift(1).rolling(20, min_periods=20).apply(lambda x: 19 - int(np.nanargmin(x)), raw=True)
        self.signal = ((l < prev_low) & (c <= prev_low) & (age >= 3)).astype(bool)   # [원전]
        self.trigger = prev_low
        self.depth = prev_low / l - 1

    def candidates(self, i: int) -> list[str]:
        hit = self.signal.iloc[i].reindex(self.tradable).fillna(False).astype(bool)
        return list(self.depth.iloc[i][hit[hit].index].sort_values(ascending=False).index)

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        trigger = self.trigger[t].iloc[i - 1]                             # [원전]
        if open_ >= trigger:
            return open_
        return trigger if high >= trigger else None

    def initial_stop(self, t: str, i: int, price: float) -> float:
        return self.l[t].iloc[i - 1]                                      # [근사] day-1 low

    def update_stop(self, pos: dict, i: int) -> float | None:
        if i > pos["entry_i"]:
            return max(pos["stop"], self.l[pos["ticker"]].iloc[i])        # [해석] prior bar low
        return pos["stop"]

    def exit_reason(self, pos: dict, i: int):
        if i - pos["entry_i"] + 1 >= self.max_days:
            return "5일 보유 만료"
        return None


class Kell(Strategy):
    """Oliver Kell: Wedge Pop and EMA Crossback in his cycle of price action (rulebook 14-4)."""
    name = "kell"
    label = "켈 가격 사이클"
    description = ("50일선 위 종목이 10·20일 지수이동평균 아래에서 쉬다 다시 올라서거나(웨지 팝), "
                   "20일선을 잠깐 깼다 되찾으면(크로스백) 다음 날 시가에 삽니다. 10일선보다 4% 넘게 "
                   "떠 있으면 안 삽니다. 계좌 15%씩, 손절은 신호일 저가(최대 2%), 10일선 아래 마감 시 "
                   "절반, 20일선 아래 마감 시 전량, 많이 오른 뒤 장대 음봉이면 전량 매도. "
                   "S&P500이 20일선 아래면 새로 안 삽니다. 6종목.")
    max_positions = 6                # [원전] 6-7 in a good market
    size = 0.15                      # [원전]
    max_stop = 0.02                  # [원전] 1-2%
    chase = 0.04                     # [원전] "3-4% above the MAs was too much"
    extended = 0.10                  # [근사]

    def _prepare(self) -> None:
        c, h, l, o = self.c, self.h, self.l, self.o
        self.e10, self.e20 = ema(c, 10), ema(c, 20)
        sma50 = sma(c, 50)
        above = (c > self.e10) & (c > self.e20)
        below = (c < self.e10) & (c < self.e20)
        wedge_pop = (above & below.shift(1, fill_value=False).astype(float).rolling(5).sum().eq(5)
                     & (c > h.shift(1).rolling(5, min_periods=5).max()))  # [해석]
        lost20 = (l <= self.e20).astype(float).shift(1).rolling(3, min_periods=1).max().astype(bool)
        crossback = above & lost20 & (c > h.shift(1)) & (self.e10 > self.e20)   # [해석]
        trend = c > sma50                                                 # [근사]
        not_ext = c <= self.e10 * (1 + self.chase)
        m = self.market
        mkt = self.c[m] > ema(self.c[[m]], 20)[m]                         # [원전] index > 20 EMA
        sig = (wedge_pop | crossback) & trend & not_ext
        self.signal = sig.mul(mkt, axis=0).astype(bool)
        self.ext = c / self.e10 - 1
        self.engulf = ((c < o) & (o >= c.shift(1)) & (c <= o.shift(1))
                       & (c.shift(1) > o.shift(1)))                       # bearish engulfing

    def candidates(self, i: int) -> list[str]:
        hit = self.signal.iloc[i].reindex(self.tradable).fillna(False).astype(bool)
        return list(self.ext.iloc[i][hit[hit].index].sort_values().index)

    def weight(self, t: str, i: int) -> float:
        return self.size

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        return price > self.initial_stop(t, i, price)

    def initial_stop(self, t: str, i: int, price: float) -> float:
        return max(self.l[t].iloc[i - 1], price * (1 - self.max_stop))   # [원전] day low / 1-2%

    def exit_reason(self, pos: dict, i: int):
        t = pos["ticker"]
        c = self.c[t].iloc[i]
        if c < self.e20[t].iloc[i]:
            return "20일선 아래 마감"                                       # [원전] staggered
        if self.ext[t].iloc[i - 1] >= self.extended and self.engulf[t].iloc[i]:
            return "과열 뒤 장대 음봉"                                       # [원전] / 10% [근사]
        if c < self.e10[t].iloc[i] and not pos.get("half_sold"):
            pos["half_sold"] = True
            return ("10일선 아래 마감 (절반)", 0.5)                          # [원전]
        return None


LEGENDS2 = [Darvas, Qullamaggie, HolyGrail, TurtleSoupPlusOne, Kell]
