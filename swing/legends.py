"""Rules of famous US swing traders, written as exact numbers.

Each class follows the rulebook in the project doc "미국 스윙매매 전설 10인의
전략", chapter 12. Every number there is tagged:

    [원전]  the trader stated the number himself (book, interview, blog)
    [해석]  a third party turned his words into the number
    [근사]  no number exists anywhere; we chose it

The same tags appear in the comments below so code and doc can be checked
against each other. All four buy at the next open after the signal close; the
traders themselves mostly bought intraday at the breakout price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import sma
from .strategies import Strategy


def rs_score(close: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """IBD-style relative strength, 1-99 within our own universe [해석].

    0.4 x 3-month + 0.2 x 6-month + 0.2 x 9-month + 0.2 x 12-month return,
    ranked across stocks each day.
    """
    c = close[tickers]
    raw = (0.4 * (c / c.shift(63) - 1) + 0.2 * (c / c.shift(126) - 1)
           + 0.2 * (c / c.shift(189) - 1) + 0.2 * (c / c.shift(252) - 1))
    return raw.rank(axis=1, pct=True) * 98 + 1


def week_end_flags(idx: pd.DatetimeIndex) -> np.ndarray:
    """True on the last trading day of each week (the data's last day only if a Friday)."""
    week = pd.Series(idx.to_period("W-FRI"), index=idx)
    out = (week != week.shift(-1)).to_numpy().copy()
    out[-1] = idx[-1].dayofweek == 4
    return out


def weekly(d: dict[str, pd.DataFrame], idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Weekly bars, each labelled with the last trading day of its week."""
    key = pd.Series(idx.to_period("W-FRI"), index=idx)
    last_day = pd.Series(idx, index=idx).groupby(key.values).max()

    def agg(frame: pd.DataFrame, how: str) -> pd.DataFrame:
        out = getattr(frame.groupby(key.values), how)()
        out.index = pd.DatetimeIndex(last_day.loc[out.index].values)
        return out

    return {"Open": agg(d["Open"], "first"), "High": agg(d["High"], "max"),
            "Low": agg(d["Low"], "min"), "Close": agg(d["Close"], "last"),
            "Volume": agg(d["Volume"], "sum")}


class Weinstein(Strategy):
    """Stan Weinstein: buy the start of Stage 2 on weekly charts (rulebook 12-1)."""
    name = "weinstein"
    label = "와인스타인 2단계 돌파"
    description = ("주봉 기준. 12주 바닥(폭 30% 이하)의 고점을 30주선 위에서 거래량 2배로 돌파하고 "
                   "맨스필드 상대강도가 0보다 크면 다음 주 월요일 시가에 삽니다. 손절은 8주 최저가와 "
                   "30주선 중 낮은 쪽 1% 아래(15% 넘게 멀면 안 삼), 매주 올리고, 주간 종가가 30주선 "
                   "아래면 팝니다. 최대 8종목.")
    max_positions = 8                # [근사] (book: 5-6 under $25K, 10-20 over $100K [원전])
    base_weeks = 12                  # [근사]
    base_width = 0.30                # [근사]
    vol_mult = 2.0                   # [원전]
    vol_weeks = 4                    # [원전]
    stop_weeks = 8                   # [근사]
    trail_weeks = 4                  # [근사]
    max_risk = 0.15                  # [원전]

    def _prepare(self) -> None:
        idx = self.c.index
        w = weekly({"Open": self.o, "High": self.h, "Low": self.l, "Close": self.c,
                    "Volume": self.v}, idx)
        wc, wh, wl, wv = w["Close"], w["High"], w["Low"], w["Volume"]
        ma30 = wc.rolling(30, min_periods=30).mean()                       # [원전] simple [해석]
        rising = ma30 >= ma30.shift(1)                                     # [해석]
        resist = wh.shift(1).rolling(self.base_weeks, min_periods=self.base_weeks).max()
        floor = wl.shift(1).rolling(self.base_weeks, min_periods=self.base_weeks).min()
        avg_vol = wv.shift(1).rolling(self.vol_weeks, min_periods=self.vol_weeks).mean()
        rp = wc.div(wc[self.market], axis=0)
        mansfield = rp / rp.rolling(52, min_periods=52).mean() - 1         # [해석]
        mkt = (wc[self.market] > ma30[self.market])                        # [근사]
        signal = ((wc > resist) & (wc > ma30) & rising & (resist / floor - 1 <= self.base_width)
                  & (wv >= self.vol_mult * avg_vol) & (mansfield > 0))
        signal = signal.mul(mkt, axis=0).astype(bool)

        def daily(frame):
            return frame.reindex(idx)

        self.week_end = week_end_flags(idx)
        self.signal = daily(signal).fillna(False).astype(bool)
        self.rank = daily(mansfield)
        self.ma30 = daily(ma30)
        self.ma30_prev = daily(ma30.shift(1))
        self.low_stop = daily(wl.rolling(self.stop_weeks, min_periods=1).min())
        self.low_trail = daily(wl.rolling(self.trail_weeks, min_periods=1).min())
        self.wclose = daily(wc)

    def candidates(self, i: int) -> list[str]:
        if not self.week_end[i]:
            return []
        hit = self.signal.iloc[i].reindex(self.tradable).fillna(False).astype(bool)
        return list(self.rank.iloc[i][hit[hit].index].sort_values(ascending=False).index)

    def _stop_at_signal(self, t: str, i: int) -> float:
        return min(self.low_stop[t].iloc[i], self.ma30[t].iloc[i]) * 0.99   # [근사] 1% below

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        stop = self._stop_at_signal(t, i - 1)
        return stop < price and price - stop <= self.max_risk * price        # [원전] 15%

    def initial_stop(self, t: str, i: int, price: float) -> float:
        return self._stop_at_signal(t, i - 1)

    def update_stop(self, pos: dict, i: int) -> float | None:
        if not self.week_end[i]:
            return pos["stop"]
        t = pos["ticker"]
        ma, ma_prev, low4 = self.ma30[t].iloc[i], self.ma30_prev[t].iloc[i], self.low_trail[t].iloc[i]
        if pd.isna(ma) or pd.isna(low4):
            return pos["stop"]
        new = low4 if ma <= ma_prev else min(ma, low4)   # flat MA = Stage 3: tighten [해석]
        return max(pos["stop"], new)                     # only raise [원전]

    def exit_reason(self, pos: dict, i: int) -> str | None:
        t = pos["ticker"]
        if self.week_end[i] and self.wclose[t].iloc[i] < self.ma30[t].iloc[i]:
            return "주간 종가 30주선 이탈"                  # [원전]
        return None


class Minervini(Strategy):
    """Mark Minervini: Trend Template + volatility contraction breakout (rulebook 12-2)."""
    name = "minervini"
    label = "미너비니 트렌드 템플릿 + VCP"
    description = ("트렌드 템플릿 8개 조건(이동평균 정배열, 52주 저가 +30% 이상, 고가 -25% 이내, "
                   "상대강도 70 이상)을 통과한 종목이 3~13주 바닥에서 변동성이 줄고 거래량이 마른 뒤 "
                   "최근 10일 고점을 거래량 1.4배로 돌파하면 삽니다. 손절은 최근 10일 저가(최대 8%), "
                   "위험의 3배 오르면 본전, 15일 안에 +25%면 절반 매도, 이후 50일선 이탈 시 매도. "
                   "손절 시 계좌 1.25% 손실이 되게 비중을 정합니다(최대 25%, 6종목).")
    max_positions = 6                # [근사] (he says at most 10-12 [원전])
    risk_per_trade = 0.0125          # [원전] 1.25-2.5%
    max_weight = 0.25                # [원전]
    ma200_rise_days = 21             # [해석] 1 month; 100 is a variant
    rs_min = 70                      # [원전]
    base_min, base_max = 15, 65      # [원전] 3-65 weeks / [근사] 13-week cap
    depth_min, depth_max = 0.10, 0.35   # [원전]
    last_n = 10                      # [근사] final contraction = last 10 days
    last_range = 0.10                # [근사]
    vol_mult = 1.4                   # [근사]
    chase = 0.05                     # [근사]
    max_stop = 0.08                  # [근사] (his hard wall is 10% [원전])
    breakeven_r = 3.0                # [원전] 2-3x risk
    climax_gain, climax_days = 0.25, 15   # [원전] 25-50% in 1-3 weeks / [근사] 15 days, half

    def _prepare(self) -> None:
        c, h, l, v = self.c, self.h, self.l, self.v
        self.sma50, self.sma150 = sma(c, 50), sma(c, 150)
        ma200 = self.sma200
        lo52 = l.rolling(252, min_periods=252).min()
        hi52 = h.rolling(252, min_periods=252).max()
        rs = rs_score(c, self.tradable).reindex(columns=c.columns)
        template = ((c > self.sma150) & (c > ma200) & (self.sma150 > ma200)
                    & (ma200 > ma200.shift(self.ma200_rise_days))
                    & (self.sma50 > self.sma150) & (self.sma50 > ma200) & (c > self.sma50)
                    & (c >= 1.30 * lo52) & (c >= 0.75 * hi52) & (rs >= self.rs_min))
        avg_vol = v.shift(1).rolling(50, min_periods=50).mean()
        pivot = h.shift(1).rolling(self.last_n, min_periods=self.last_n).max()
        last_low = l.shift(1).rolling(self.last_n, min_periods=self.last_n).min()
        dry = v.shift(1).rolling(self.last_n, min_periods=self.last_n).mean() < avg_vol
        pre = (template & (c > pivot) & (c.shift(1) <= pivot.shift(0)) & dry
               & (v >= self.vol_mult * avg_vol) & (pivot / last_low - 1 <= self.last_range))
        pre = pre.mul(self.market_ok, axis=0).astype(bool)
        self.pivot, self.last_low, self.rs = pivot, last_low, rs

        # Base shape, checked only where the cheap conditions already passed.
        hv, lv = h.to_numpy(), l.to_numpy()
        sig = np.zeros(c.shape, dtype=bool)
        for i, j in zip(*np.nonzero(pre.to_numpy())):
            if i < self.base_max + 1:
                continue
            win = hv[i - self.base_max:i - self.base_min + 1, j]
            if np.isnan(win).all():
                continue
            s = i - self.base_max + int(np.nanargmax(win))
            top = hv[s, j]
            if np.nanmax(hv[s + 1:i, j]) > top:          # no higher high after the base start
                continue
            base_low = np.nanmin(lv[s:i, j])
            depth = 1 - base_low / top
            if not (self.depth_min <= depth <= self.depth_max):
                continue
            mid = s + (i - s) // 2                       # contraction: back half <= half of front
            front = np.nanmax(hv[s:mid, j]) / np.nanmin(lv[s:mid, j]) - 1
            back = np.nanmax(hv[mid:i, j]) / np.nanmin(lv[mid:i, j]) - 1
            if back <= 0.5 * front:
                sig[i, j] = True
        self.signal = pd.DataFrame(sig, index=c.index, columns=c.columns)

    def candidates(self, i: int) -> list[str]:
        hit = self.signal.iloc[i].reindex(self.tradable).fillna(False).astype(bool)
        return list(self.rs.iloc[i][hit[hit].index].sort_values(ascending=False).index)

    def _stop_from(self, t: str, i_sig: int, price: float) -> float:
        return max(self.last_low[t].iloc[i_sig], price * (1 - self.max_stop))

    def weight(self, t: str, i: int) -> float:
        c = self.c[t].iloc[i]
        dist = (c - self._stop_from(t, i, c)) / c
        return self.max_weight if dist <= 0 else min(self.max_weight, self.risk_per_trade / dist)

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        return price <= self.pivot[t].iloc[i - 1] * (1 + self.chase) and \
            price > self._stop_from(t, i - 1, price)

    def initial_stop(self, t: str, i: int, price: float) -> float:
        stop = self._stop_from(t, i - 1, price)
        return stop

    def update_stop(self, pos: dict, i: int) -> float | None:
        if "risk" not in pos:
            pos["risk"] = pos["entry_price"] - pos["stop"]
        if pos["high_max"] >= pos["entry_price"] + self.breakeven_r * pos["risk"]:
            pos["breakeven"] = True
            return max(pos["stop"], pos["entry_price"])
        return pos["stop"]

    def take_profit(self, pos: dict, i: int):
        if pos.get("half_sold") or i - pos["entry_i"] + 1 > self.climax_days:
            return None
        target = pos["entry_price"] * (1 + self.climax_gain)
        if pos["high_max"] >= target:
            pos["half_sold"] = True
        return target, 0.5, "급등 절반 매도 (+25%)"

    def exit_reason(self, pos: dict, i: int) -> str | None:
        t = pos["ticker"]
        if pos.get("breakeven") and self.c[t].iloc[i] < self.sma50[t].iloc[i]:
            return "50일선 이탈"
        return None


class Bonde(Strategy):
    """Pradeep Bonde (Stockbee): 4% momentum burst, held 3-5 days (rulebook 12-3)."""
    name = "bonde"
    label = "본드 4% 모멘텀 버스트"
    description = ("횡보하던 종목이 거래량 늘며 하루 4% 이상 오르고 고가 근처에서 끝나면 다음 날 "
                   "시가에 삽니다(시장 폭 필터 통과 시). 손절은 돌파일 저가, +5%면 본전, +10%에서 "
                   "절반 매도, 3일째 종가가 매수가 이하면 매도, 나머지는 5일째 장 마감 후 매도. "
                   "손절 시 계좌 1% 손실이 되게 비중을 정합니다(최대 20%, 8종목).")
    max_positions = 8                # [근사]
    risk_per_trade = 0.01            # [원전]
    max_weight = 0.20                # [근사]
    chase = 0.03                     # [근사]
    use_breadth = True

    def _prepare(self) -> None:
        c, h, l, v = self.c, self.h, self.l, self.v
        c1 = c.shift(1)
        up1 = c > c1
        up_two = up1.shift(1, fill_value=False) & up1.shift(2, fill_value=False)
        rng = h - l
        cons_hi = h.shift(1).rolling(10, min_periods=10).max()
        cons_lo = l.shift(1).rolling(10, min_periods=10).min()
        down4 = c / c1 <= 0.96
        signal = ((c >= 1.04 * c1)                                        # [원전]
                  & (v > v.shift(1)) & (v >= 100_000)                     # [원전]
                  & ~up_two                                               # [원전] not up 2 days
                  & ((c1 < c.shift(2)) | (rng.shift(1) < rng.shift(2).rolling(7).mean()))  # [근사]
                  & ((c - l) >= 0.7 * rng)                                # [해석]
                  & (cons_hi / cons_lo - 1 <= 0.15)                       # [근사]
                  & (down4.astype(float).shift(1).rolling(10, min_periods=10).sum() <= 1))  # [원전]
        if self.use_breadth:
            ok = self._breadth()
        else:
            ok = self.market_ok
        self.signal = signal.mul(ok, axis=0).astype(bool)
        self.vol_ratio = v / v.shift(1)

    def _breadth(self) -> pd.Series:
        """Stockbee Market Monitor on our universe: 10-day 4%-up / 4%-down ratio.

        Turn on at >= 2, off at <= 0.5 [원전 numbers, 근사 universe]. Starts on.
        """
        t = self.tradable
        c, v = self.c[t], self.v[t]
        r = c / c.shift(1)
        up = ((r >= 1.04) & (v > v.shift(1))).sum(axis=1).rolling(10).sum()
        dn = (r <= 0.96).sum(axis=1).rolling(10).sum()
        ratio = up / dn.replace(0, np.nan)
        state, out = True, []
        for x in ratio.to_numpy():
            if x >= 2:
                state = True
            elif x <= 0.5:
                state = False
            out.append(state)
        self.breadth_ratio = ratio
        return pd.Series(out, index=c.index)

    def candidates(self, i: int) -> list[str]:
        hit = self.signal.iloc[i].reindex(self.tradable).fillna(False).astype(bool)
        return list(self.vol_ratio.iloc[i][hit[hit].index].sort_values(ascending=False).index)

    def weight(self, t: str, i: int) -> float:
        c, lo = self.c[t].iloc[i], self.l[t].iloc[i]
        dist = (c - lo) / c
        return self.max_weight if dist <= 0 else min(self.max_weight, self.risk_per_trade / dist)

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        return self.l[t].iloc[i - 1] < price <= self.c[t].iloc[i - 1] * (1 + self.chase)

    def initial_stop(self, t: str, i: int, price: float) -> float:
        return self.l[t].iloc[i - 1]                                       # [원전]

    def update_stop(self, pos: dict, i: int) -> float | None:
        if pos["high_max"] >= pos["entry_price"] * 1.05:                   # [원전] 4-5%
            return max(pos["stop"], pos["entry_price"])
        return pos["stop"]

    def take_profit(self, pos: dict, i: int):
        if pos.get("half_sold"):
            return None
        target = pos["entry_price"] * 1.10                                 # [원전]
        if pos["high_max"] >= target:
            pos["half_sold"] = True
        return target, 0.5, "+10% 절반 매도"

    def exit_reason(self, pos: dict, i: int) -> str | None:
        day = i - pos["entry_i"] + 1
        if day == 3 and self.c[pos["ticker"]].iloc[i] <= pos["entry_price"]:
            return "3일째 힘 없음"                                           # [원전] / [근사]
        if day >= 5:
            return "5일 보유 만료"                                          # [원전]
        return None


class BondeSPY(Bonde):
    """Same as Bonde but with the S&P 500 200-day filter instead of breadth."""
    name = "bonde_spy"
    label = "본드 4% 버스트 (S&P500 200일선 필터)"
    description = "본드 4% 버스트와 같고, 시장 폭 대신 S&P500이 200일선 위일 때만 삽니다."
    use_breadth = False


class ONeil(Strategy):
    """William O'Neil: cup-with-handle / flat-base breakout with his sell rules (rulebook 12-4)."""
    name = "oneil"
    label = "오닐 바닥 돌파 + 매도 규칙"
    description = ("컵앤핸들(7~65주, 깊이 12~33%, 손잡이 12% 이내) 또는 평평한 바닥(25일 폭 15% 이내)"
                   "의 고점+0.10달러를 거래량 1.4배로 돌파하고 상대강도 80 이상이면 삽니다(피벗보다 "
                   "5% 넘게 높으면 안 삼). 손절 -7.5%, +20% 익절, 단 3주 안에 +20%면 8주 보유. "
                   "50일선을 거래량 실어 깨면 매도. 시장 분산일 5회 이상이면 신규 매수 중단. 5종목.")
    max_positions = 5                # [원전] $100K -> 5 stocks
    stop_pct = 0.075                 # [원전] 7-8%
    target = 0.20                    # [원전] 20-25%
    fast_days, hold_days = 15, 40    # [원전] 3 weeks, 8 weeks
    chase = 0.05                     # [해석]
    vol_mult = 1.4                   # [원전]
    rs_min = 80                      # [원전]

    def _prepare(self) -> None:
        c, h, l, v = self.c, self.h, self.l, self.v
        self.sma50 = sma(c, 50)
        self.avg_vol = v.shift(1).rolling(50, min_periods=50).mean()
        self.rs = rs_score(c, self.tradable).reindex(columns=c.columns)
        mkt = self._market_direction()
        hi25 = h.shift(1).rolling(25, min_periods=25).max()
        pre = ((c > hi25 * 0.95) & (v >= self.vol_mult * self.avg_vol) & (self.rs >= self.rs_min))
        pre = pre.mul(mkt, axis=0).astype(bool)

        hv, lv, cv = h.to_numpy(), l.to_numpy(), c.to_numpy()
        ma50 = self.sma50.to_numpy()
        piv = np.full(c.shape, np.nan)
        for i, j in zip(*np.nonzero(pre.to_numpy())):
            p = self._cup_pivot(hv[:, j], lv[:, j], ma50[:, j], i)
            if p is None:
                p = self._flat_pivot(hv[:, j], lv[:, j], i)
            if p is not None and cv[i, j] > p and cv[i - 1, j] <= p:
                piv[i, j] = p
        self.pivot = pd.DataFrame(piv, index=c.index, columns=c.columns)

    @staticmethod
    def _cup_pivot(hv, lv, ma50, i):
        if i < 400:
            return None
        hw = hv[i - 25:i - 4]                          # right-side peak 5-25 days back [근사]
        if np.isnan(hw).all():
            return None
        r = i - 25 + int(np.nanargmax(hw))
        right = hv[r]
        if np.nanmax(hv[r:i]) > right:                 # handle high must be the right-side peak
            return None
        left_win = hv[r - 325:r]                       # cup 7-65 weeks [원전]
        if np.isnan(left_win).all():
            return None
        lft = r - 325 + int(np.nanargmax(left_win))
        top = hv[lft]
        if r - lft < 35 or not (0.85 * top <= right <= top * 1.0):   # right side near the left top [근사]
            return None
        cup_low = np.nanmin(lv[lft:r + 1])
        if not (0.12 <= 1 - cup_low / top <= 0.33):    # [원전]
            return None
        pre_low = np.nanmin(lv[max(0, lft - 126):lft + 1])
        if top < 1.30 * pre_low:                       # prior uptrend 30% [원전] in 6 months [근사]
            return None
        handle_low = np.nanmin(lv[r + 1:i]) if i > r + 1 else lv[r]
        if 1 - handle_low / right > 0.12:              # [원전]
            return None
        if handle_low <= (top + cup_low) / 2 or handle_low < ma50[i]:   # upper half, above 50-day [원전]
            return None
        return right + 0.10                            # [원전]

    @staticmethod
    def _flat_pivot(hv, lv, i):
        if i < 160:
            return None
        top, bot = np.nanmax(hv[i - 25:i]), np.nanmin(lv[i - 25:i])
        if top / bot - 1 > 0.15:                       # [해석]
            return None
        if top < 1.30 * np.nanmin(lv[i - 25 - 126:i - 25]):   # prior uptrend 30% [해석]
            return None
        return top + 0.10                              # [원전]

    def _market_direction(self) -> pd.Series:
        """Stop new buys at >= 5 distribution days in 25 sessions [해석]; resume when < 5
        and SPY above its 50-day average [근사]."""
        m = self.market
        spy, vol = self.c[m], self.v[m]
        dist = ((spy / spy.shift(1) - 1 <= -0.002) & (vol > vol.shift(1))).astype(int)
        count = dist.rolling(25, min_periods=1).sum()
        above50 = spy > sma(self.c[[m]], 50)[m]
        state, out = True, []
        for n, a in zip(count.to_numpy(), above50.to_numpy()):
            if n >= 5:
                state = False
            elif not state and a:
                state = True
            out.append(state)
        return pd.Series(out, index=spy.index)

    def candidates(self, i: int) -> list[str]:
        row = self.pivot.iloc[i].reindex(self.tradable).dropna()
        return list(self.rs.iloc[i][row.index].sort_values(ascending=False).index)

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        return price <= self.pivot[t].iloc[i - 1] * (1 + self.chase)

    def initial_stop(self, t: str, i: int, price: float) -> float:
        return price * (1 - self.stop_pct)

    def take_profit(self, pos: dict, i: int):
        if pos.get("hold8"):
            return None
        target = pos["entry_price"] * (1 + self.target)
        if pos["high_max"] >= target and i - pos["entry_i"] + 1 <= self.fast_days:
            pos["hold8"] = True                        # 8-week rule: keep it [원전]
            pos["stop"] = max(pos["stop"], pos["entry_price"])   # never let it turn into a loss [원전]
            return None
        return target, 1.0, "+20% 익절"

    def exit_reason(self, pos: dict, i: int) -> str | None:
        t = pos["ticker"]
        c, ma, v = self.c[t].iloc[i], self.sma50[t].iloc[i], self.v[t].iloc[i]
        days = i - pos["entry_i"] + 1
        if c < ma and v >= self.avg_vol[t].iloc[i]:
            return "50일선 거래량 이탈"                   # [해석]
        if pos.get("hold8") and days >= self.hold_days and c < ma:
            return "8주 보유 후 50일선 이탈"               # [근사]
        return None


LEGENDS = [Weinstein, Minervini, Bonde, BondeSPY, ONeil]
