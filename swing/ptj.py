"""'변곡점 돌파 + 5일선 추격매수' from a Korean YouTube video (부자의 선), rulebook 02.

The channel's reading of Paul Tudor Jones (not his own published rules) mixed
with a Korean trader's (남석관) 5-day line add-on buying. A stock that just
made a higher low leaves a tight 20-day box on a big, high-volume up day and
closes back above its 200-day average. Buy a third, add a third on the first
touch of the 5-day average and a third on a later up day while the trade is
winning. Stop under the turning-point low, sell if the breakout does not
follow through within 7 days, and ride the rest until the 10-day line breaks.

All numbers are from /mnt/project-files/rulebooks/02-폴튜더존스-변곡점돌파.md
([원전] said in the video, [해석] our numbers, [근사] automated stand-in,
[추가] added), fixed before the first run. Test list:

J1  no earnings filter, 1/3 + 1/3 + 1/3, exit on 3 closes under the 10-day line
J2  J1 with latest-quarter EPS and sales both +20% year over year
J3  J1 but the full size on the first buy (no pyramiding)
J4  J1 but sell on the first close under the 10-day line
J5  J1 without the 7-day time stop
plus J1 with one number changed: big day 4/7%, volume 2/3x, box 10/20%,
time stop 5/10 days.

Not automated [미구현]: the daily -2% and monthly -10% stop-buying limits and
the smaller size after losing streaks (the engine does not tell a strategy
its account value). With 1% risk per trade and 5 positions they rarely bind.
"""
from __future__ import annotations

import pandas as pd

from .indicators import sma
from .pullback_yt import _pivot_confirmed
from .strategies import Strategy

BOX_DAYS = 20
BOX_WIDTH = 0.15
BIG_DAY = 0.05
VOL_MULT = 2.5
FAR_FROM_LOW = 1.0         # close no more than 100% above the 252-day low
TREND_DAYS = 120
MIN_DOLLAR_VOL = 20e6
MIN_PRICE = 10.0
RISK_PER_TRADE = 0.01
MAX_RISK = 0.08
TIME_STOP = 7
ADD_WINDOW = 15
UNDER_10 = 3
CLIMAX_GAIN = 0.20
CLIMAX_VOL = 3.0
CLIMAX_BODY = 0.04
MAX_HOLD = 60


def last_two_lows(low: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per day, the latest confirmed swing low, the one before it, and the day number of the earlier one."""
    piv = _pivot_confirmed(low, highs=False)
    pos = pd.DataFrame(range(len(low)), index=low.index).iloc[:, 0]
    last, prev, prev_day = {}, {}, {}
    for t in low.columns:
        s = piv[t].dropna()
        days = pos.reindex(s.index) - 5              # the pivot itself, 5 days before confirmation
        last[t] = s.reindex(low.index).ffill()
        prev[t] = s.shift(1).reindex(low.index).ffill()
        prev_day[t] = days.shift(1).reindex(low.index).ffill()
    return pd.DataFrame(last), pd.DataFrame(prev), pd.DataFrame(prev_day)


class PTJ1(Strategy):
    name = "ptj_j1"
    label = "J1 변곡점 돌파 + 3번 나눠 사기 + 10일선 3일 이탈"
    description = ("저점을 높인 종목이 20일 박스(폭 15% 이하)를 하루 +5%·거래량 2.5배로 뚫고 200일선 위로 막 "
                   "올라서면 1/3을 삽니다. 처음 5일선에 닿고 지킨 날 1/3, 그 뒤 수익 중이고 전날보다 오른 날 "
                   "1/3을 더 삽니다(15거래일 안). 손절은 변곡점 저점 1% 아래, 7거래일 안에 신호일 고가를 못 넘으면 "
                   "매도, 10일선 아래 3일이면 매도, 5·10일선 데드크로스면 매도, 최대 60거래일.")
    max_positions = 5
    box_width, big_day, vol_mult, time_stop = BOX_WIDTH, BIG_DAY, VOL_MULT, TIME_STOP
    pyramid = True
    under_days = UNDER_10
    use_time_stop = True

    def _prepare(self) -> None:
        c, h, l, o, v = (x[self.tradable] for x in (self.c, self.h, self.l, self.o, self.v))
        s200 = self.sma200[self.tradable]
        self.sma5, self.sma10 = sma(c, 5), sma(c, 10)
        self.avg50 = v.rolling(50, min_periods=50).mean().shift(1)
        top = h.rolling(BOX_DAYS, min_periods=BOX_DAYS).max().shift(1)
        bottom = l.rolling(BOX_DAYS, min_periods=BOX_DAYS).min().shift(1)
        last, prev, prev_day = last_two_lows(l)
        day = pd.DataFrame({t: range(len(c)) for t in c.columns}, index=c.index)
        higher_low = (last > prev) & (day - prev_day <= TREND_DAYS)
        tight = (top - bottom) / bottom <= self.box_width
        big = (c / c.shift(1) - 1 >= self.big_day) & (c > top)
        loud = v >= self.vol_mult * self.avg50
        was_below = (c < s200).astype(float).shift(1).rolling(BOX_DAYS, min_periods=1).max() > 0
        above = (c > s200) & was_below
        not_extended = c <= (1 + FAR_FROM_LOW) * c.rolling(252, min_periods=200).min()
        liquid = ((c * v).rolling(20, min_periods=20).mean() >= MIN_DOLLAR_VOL) & (c >= MIN_PRICE)
        sig = higher_low & tight & big & loud & above & not_extended & liquid
        self.rank = (v / self.avg50).where(sig)
        self.stop_level = pd.concat([l, last]).groupby(level=0).max().reindex(c.index) * 0.99
        self.pending: dict[str, dict] = {}
        self.trade: dict[str, dict] = {}

    def candidates(self, i: int) -> list[str]:
        r = self.rank.iloc[i].dropna()
        self.pending = {t: {"stop": self.stop_level[t].iloc[i], "sig_high": self.h[t].iloc[i]}
                        for t in r.index}
        return list(r.sort_values(ascending=False).index)

    def _planned(self, t: str, i: int) -> float:
        px = self.c[t].iloc[i]
        risk = max((px - self.pending[t]["stop"]) / px, 0.005)
        return min(RISK_PER_TRADE / risk, 1 / self.max_positions)

    def weight(self, t: str, i: int) -> float:
        w = self._planned(t, i)
        if t in self.pending:
            self.pending[t]["planned"] = w
        return w / 3 if self.pyramid else w

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        p = self.pending[t]
        if price <= p["stop"] or (price - p["stop"]) / price > MAX_RISK:
            return False
        self.trade[t] = {**p, "R": price - p["stop"], "first": price, "touched5": False,
                         "under": 0, "half": False}
        return True

    def initial_stop(self, t: str, i: int, price: float) -> float:
        return self.trade[t]["stop"]

    def update_stop(self, pos: dict, i: int) -> float | None:
        tr = self.trade[pos["ticker"]]
        if self.c[pos["ticker"]].iloc[i] >= tr["first"] + tr["R"]:
            return max(pos["stop"], pos["entry_price"])          # breakeven stop [원전 본절 컷]
        return pos["stop"]

    def exit_reason(self, pos: dict, i: int):
        t = pos["ticker"]
        tr = self.trade[t]
        c, o, v = self.c[t].iloc[i], self.o[t].iloc[i], self.v[t].iloc[i]
        s5, s10 = self.sma5[t].iloc[i], self.sma10[t].iloc[i]
        days = i - pos["entry_i"] + 1
        tr["under"] = tr["under"] + 1 if c < s10 else 0
        if self.use_time_stop and days >= self.time_stop and pos["high_max"] <= tr["sig_high"]:
            return f"{self.time_stop}거래일 안에 돌파 실패"
        if tr["under"] >= self.under_days:
            return f"10일선 아래 {self.under_days}일"
        if s5 < s10 and c < s10:
            return "5·10일선 데드크로스"
        if days >= MAX_HOLD:
            return "60거래일 보유 완료"
        if (not tr["half"] and c >= pos["entry_price"] * (1 + CLIMAX_GAIN) and c < o
                and (o - c) / o >= CLIMAX_BODY and v >= CLIMAX_VOL * self.avg50[t].iloc[i]):
            tr["half"] = True
            return "과열 대량 음봉 절반 매도", 0.5
        return None

    def add_weight(self, pos: dict, i: int) -> float | None:
        if not self.pyramid:
            return None
        t = pos["ticker"]
        tr = self.trade[t]
        adds = pos.get("adds", 0)
        if adds >= 2 or i - pos["entry_i"] + 1 > ADD_WINDOW:
            return None
        c, lo = self.c[t].iloc[i], self.l[t].iloc[i]
        s5, s10 = self.sma5[t].iloc[i], self.sma10[t].iloc[i]
        winning = c > pos["entry_price"] and c > s5 > s10
        if adds == 0 and not tr["touched5"]:
            if lo <= s5 <= c:
                tr["touched5"] = True                            # first touch of the 5-day line
                return tr["planned"] / 3 if winning else None
            return None
        if c > self.c[t].iloc[i - 1] and winning:
            return tr["planned"] / 3
        return None


class PTJ2(PTJ1):
    name = "ptj_j2"
    label = "J2 J1 + 실적(EPS·매출 +20%)"
    description = "J1과 같고, 최근 분기 EPS와 매출이 1년 전보다 모두 20% 이상 늘어난 종목만 삽니다."

    def prepare(self, d, market: str = "SPY"):
        self.earn_ok = d["EarnOK"].reindex(index=d["Close"].index).fillna(False).astype(bool)
        return super().prepare(d, market)

    def candidates(self, i: int) -> list[str]:
        row = self.earn_ok.iloc[i]
        return [t for t in super().candidates(i) if bool(row.get(t, False))]


class PTJ3(PTJ1):
    name = "ptj_j3"
    label = "J3 J1 + 한 번에 전량 매수"
    description = "J1과 같고, 신호 다음 날 계획한 비중을 한 번에 다 삽니다(추가 매수 없음)."
    pyramid = False


class PTJ4(PTJ1):
    name = "ptj_j4"
    label = "J4 J1 + 10일선 1일 이탈 즉시 매도"
    description = "J1과 같고, 종가가 10일선 아래로 한 번만 내려가도 팝니다."
    under_days = 1


class PTJ5(PTJ1):
    name = "ptj_j5"
    label = "J5 J1 + 시간 손절 없음"
    description = "J1과 같고, 7거래일 시간 손절을 쓰지 않습니다."
    use_time_stop = False


def _sweep(param: str, value, text: str) -> type:
    """J1 with one number changed: the rulebook's sensitivity check."""
    tag = str(value).replace(".", "")
    return type(f"PTJ1_{param}_{tag}", (PTJ1,), {
        "name": f"ptj_j1_{param}_{tag}", "label": f"J1 민감도: {text}",
        "description": f"J1과 같고 {text}만 바꿉니다.", param: value})


SWEEP = [_sweep("big_day", 0.04, "장대양봉 4%"), _sweep("big_day", 0.07, "장대양봉 7%"),
         _sweep("vol_mult", 2.0, "거래량 2배"), _sweep("vol_mult", 3.0, "거래량 3배"),
         _sweep("box_width", 0.10, "박스 폭 10%"), _sweep("box_width", 0.20, "박스 폭 20%"),
         _sweep("time_stop", 5, "시간 손절 5일"), _sweep("time_stop", 10, "시간 손절 10일")]
PTJ = [PTJ1, PTJ2, PTJ3, PTJ4, PTJ5, *SWEEP]
