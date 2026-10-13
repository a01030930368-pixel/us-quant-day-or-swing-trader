"""'3연속 음봉 눌림 후 전봉 고가 돌파' from a Korean YouTube video (부자회사원), rulebook 04.

A review of Oliver Velez and Greg Capra's swing setup: after a strong run to a
new high, wait for three or more red candles in a row, each with a lower high.
Place a buy-stop one cent above the last candle's high. The stop goes one cent
under that candle's low; sell half at 2R (R = buy - stop), the rest at 3R, and
the rest at 2R if the price comes back there. The channel tested it on each
month's top two gainers of days 1-15.

All numbers are from /mnt/project-files/rulebooks/04-벨레즈-3음봉눌림-고가돌파.md
([원전] said in the video, [해석] our numbers, [근사] automated stand-in,
[추가] added), fixed before the first run. Test list:

V1  book setup: +15% in 20 days to a 20-day closing high, then 3 red candles
    in a row with lower highs; 2R half, 3R rest, 2R protection
V2  the channel's stock list: each month's top 2 gainers of days 1-15 (S&P 1500,
    a year of history), traded from the 16th to the month end; V1 entries
V3  V2 stocks within 120 days of being listed: the same setup near the
    150- or 200-day line; sell on a close 3% under the 200-day line
V4  V1, but the second half sells on a close under the 20-day line
V5  V1 entries, a fixed -7% stop, sell on a close under the 20-day line
V6  V2 with the channel's looser count: 3 red candles since the high, not in a row
plus V1 with one number changed (section 11 sensitivity list) and V1 in bull
and bear markets only (S&P 500 above / below its 200-day line).

Not automated [미구현]: the base-low stop (no picture in the video) and the
daily -2% stop-buying limit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import sma
from .strategies import Strategy

TICK = 0.01
RUN_DAYS = 20
RISE = 0.15
N_RED = 3
VALID = 20
RETRACE = 0.618
MAX_RISK = 0.10
WIDE_RISK = 0.08
RISK_PER_TRADE = 0.01
MAX_HOLD = 30
LONG_HOLD = 60
MIN_DOLLAR_VOL = 20e6
MIN_PRICE = 10.0
TOP_N = 2
LIST_DAY = 15
MA_NEAR = 0.03
MA_TOUCH_DAYS = 5
MA_TRACK = 120
MA_BREAK = 0.03
MA_HOLD = 120


def monthly_top(c: pd.DataFrame, ok: pd.DataFrame, n: int = TOP_N) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(listed, picked): which stocks are tradable each day, and the day each month's list was made.

    The list is each month's top n by the return from the previous month's
    last close to the last close on or before the 15th, among stocks passing
    `ok` that day. It is known at that close and used until the month ends.
    """
    idx = c.index
    month = idx.to_period("M")
    listed = pd.DataFrame(False, index=idx, columns=c.columns)
    picked = pd.DataFrame(False, index=idx, columns=c.columns)
    pos = pd.Series(range(len(idx)), index=idx)
    for m in month.unique():
        days = pos[(month == m)]
        first_half = days[days.index.day <= LIST_DAY]
        if len(first_half) == 0 or days.iloc[0] == 0:
            continue
        a, b = days.iloc[0] - 1, first_half.iloc[-1]
        ret = (c.iloc[b] / c.iloc[a] - 1).where(ok.iloc[b]).dropna()
        top = ret.sort_values(ascending=False).index[:n]
        if len(top) == 0:
            continue
        listed.iloc[b:days.iloc[-1] + 1, listed.columns.get_indexer(top)] = True
        picked.iloc[b, picked.columns.get_indexer(top)] = True
    return listed, picked


def setups(o, h, l, c, rise, n_red, loose, use_peak):
    """One ticker's buy-stop days: the trigger price for the next day (NaN = no order).

    use_peak: count only after a 20-day closing high that ran up `rise` in 20
    days (rise None = any 20-day closing high), within 20 days of it and
    before the pullback gives back 61.8% of the run. Otherwise (V3) count
    anywhere. Strict count: red candles in a row, each with a lower high; one
    small 'pause' candle keeps the order alive for one more day [원전 E4].
    Loose count (V6): red candles since the high.
    """
    o, h, l, c = (x.tolist() for x in (o, h, l, c))
    n = len(c)
    out = np.full(n, np.nan)
    peak, peak_c, floor = -1, 0.0, 0.0
    run, paused, reds = 0, False, 0
    for i in range(RUN_DAYS, n):
        if use_peak:
            window = c[i - RUN_DAYS:i + 1]
            if c[i] >= max(window) and (rise is None or c[i] / c[i - RUN_DAYS] - 1 >= rise):
                peak, peak_c, floor = i, c[i], min(window)
                run, paused, reds = 0, False, 0
                continue
            if peak < 0 or i - peak > VALID or l[i] < peak_c - RETRACE * (peak_c - floor):
                peak = -1
                continue
        red = c[i] < o[i]
        if loose:
            reds += red
            if reds >= n_red and c[i] < peak_c:
                out[i] = h[i] + TICK
            continue
        if red and h[i] < h[i - 1]:
            run, paused = run + 1, False
        elif run >= n_red and not paused and h[i] < h[i - 1]:
            paused = True                                   # a pause candle: try once more
        else:
            run, paused = (1, False) if red else (0, False)
        if run >= n_red:
            out[i] = h[i] + TICK
    return out


class VelezV1(Strategy):
    name = "velez_v1"
    label = "V1 3연속 음봉 눌림 + 전봉 고가 돌파 (책)"
    description = ("20거래일 +15% 이상 올라 20일 종가 신고가를 낸 종목이, 20거래일 안에 고가가 낮아지는 음봉 "
                   "3개를 연속으로 만들면(상승폭 61.8% 넘게 되돌리면 취소) 다음 날 전봉 고가 + 1센트에 매수 "
                   "예약합니다. 손절은 전봉 저가 - 1센트(8% 넘으면 매수 캔들 저가, 10% 넘으면 안 삼), 2R에서 "
                   "절반, 3R에서 나머지, 2R 위로 간 뒤 2R로 돌아오면 나머지 매도. 종가 +1R이면 손절을 매수가로, "
                   "최대 30거래일.")
    max_positions = 5
    order_all = True
    rise, n_red, loose = RISE, N_RED, False
    tp1, tp2 = 2.0, 3.0
    protect = True                        # after tp1, stop moves up to the tp1 price
    ma_exit = None                        # sell on a close under this average (20) instead of tp2
    fixed_stop = None                     # V5: -7%
    max_hold = MAX_HOLD
    monthly = False                       # V2/V3/V6: only each month's top gainers
    top_n = TOP_N
    ma_reentry = False                    # V3
    regime = None                         # "bull" / "bear": buy only in that market
    entry_day_stop = True                 # False: no stop on the buy day (the optimistic bound)

    def _prepare(self) -> None:
        t = self.tradable
        c, h, l, o, v = (x[t] for x in (self.c, self.h, self.l, self.o, self.v))
        liquid = ((c * v).rolling(20, min_periods=20).mean() >= MIN_DOLLAR_VOL) & (c >= MIN_PRICE)
        self.sma20, self.sma150, self.sma200t = sma(c, 20), sma(c, 150), sma(c, 200)
        rise = None if self.monthly else self.rise
        trig = pd.DataFrame({k: setups(o[k], h[k], l[k], c[k], rise, self.n_red, self.loose,
                                       not self.ma_reentry) for k in t}, index=c.index)
        ok = liquid
        if self.monthly:
            aged = c.notna().cumsum() >= 252                # at least a year of prices [원전: 신규 상장 제외]
            listed, picked = monthly_top(c, liquid & aged, self.top_n)
            month = c.index.to_period("M")
            same_month = pd.Series(month[1:].tolist() + [None], index=c.index) == pd.Series(month, index=c.index)
            if self.ma_reentry:
                recent = picked.astype(float).rolling(MA_TRACK, min_periods=1).max() > 0
                near = pd.concat([(l / self.sma150 - 1).abs(), (l / self.sma200t - 1).abs()]
                                 ).groupby(level=0).min().reindex(c.index) <= MA_NEAR
                touched = near.astype(float).rolling(MA_TOUCH_DAYS, min_periods=1).max() > 0
                ok = ok & recent & touched
            else:
                ok = ok & listed.mul(same_month, axis=0).astype(bool)
        self.trigger = trig.where(ok)
        ret20 = c / c.shift(RUN_DAYS) - 1
        self.rank = ret20.where(self.trigger.notna())
        self.pending: dict[str, dict] = {}
        self.trade: dict[str, dict] = {}

    def candidates(self, i: int) -> list[str]:
        if self.regime == "bull" and not self.market_ok.iloc[i]:
            return []
        if self.regime == "bear" and self.market_ok.iloc[i]:
            return []
        r = self.rank.iloc[i].dropna()
        self.pending = {t: {"trigger": self.trigger[t].iloc[i], "stop": self.l[t].iloc[i] - TICK}
                        for t in r.index}
        return list(r.sort_values(ascending=False).index)

    def weight(self, t: str, i: int) -> float:
        p = self.pending[t]
        risk = self.fixed_stop or max((p["trigger"] - p["stop"]) / p["trigger"], 0.005)
        return min(RISK_PER_TRADE / risk, 1 / self.max_positions)

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        trig = self.pending[t]["trigger"]
        if open_ >= trig:
            return open_                                    # gap above the order: filled at the open
        return trig if high >= trig else None

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        stop = price * (1 - self.fixed_stop) if self.fixed_stop else self.pending[t]["stop"]
        if price <= stop or (price - stop) / price > MAX_RISK:
            return False
        self.trade[t] = {"entry": price, "stop": stop, "R": price - stop, "half": False,
                         "wide": (price - stop) / price > WIDE_RISK}
        return True

    def initial_stop(self, t: str, i: int, price: float) -> float | None:
        # The engine checks the stop on the buy day too. For a buy-stop that is cautious: a day that
        # dips under the prior low and then breaks the prior high counts as stopped out.
        return self.trade[t]["stop"] if self.entry_day_stop else None

    def update_stop(self, pos: dict, i: int) -> float | None:
        t = pos["ticker"]
        tr = self.trade[t]
        stop = tr["stop"] if pos["stop"] is None else pos["stop"]
        if tr["wide"] and i == pos["entry_i"]:
            stop = max(stop, self.l[t].iloc[i] - TICK)       # too far: under the buy candle's low
            tr["R"] = tr["entry"] - stop
        if not self.fixed_stop and self.c[t].iloc[i] >= tr["entry"] + tr["R"]:
            stop = max(stop, tr["entry"])                     # +1R: stop to the buy price [원전]
        return stop

    def take_profit(self, pos: dict, i: int):
        if self.fixed_stop:
            return None
        t = pos["ticker"]
        tr = self.trade[t]
        hi = self.h[t].iloc[i]
        if not tr["half"]:
            target = tr["entry"] + self.tp1 * tr["R"]
            if hi >= target:
                tr["half"] = True
                pos["stop"] = max(pos["stop"] or 0.0, target if self.protect else tr["entry"])
            return target, 0.5, f"{self.tp1:g}R 절반 익절"
        if self.ma_exit:
            return None
        return tr["entry"] + self.tp2 * tr["R"], 1.0, f"{self.tp2:g}R 익절"

    def exit_reason(self, pos: dict, i: int):
        t = pos["ticker"]
        if i - pos["entry_i"] + 1 >= self.max_hold:
            return f"{self.max_hold}거래일 보유 완료"
        c = self.c[t].iloc[i]
        if self.ma_reentry and c < self.sma200t[t].iloc[i] * (1 - MA_BREAK):
            return "200일선 3% 이탈"
        if self.ma_exit and (self.fixed_stop or self.trade[t]["half"]) and c < self.sma20[t].iloc[i]:
            return "종가 20일선 이탈"
        return None


class VelezV2(VelezV1):
    name = "velez_v2"
    label = "V2 V1 + 매달 1~15일 상승률 상위 2종목만 (채널 방식)"
    description = ("매달 1~15일(전달 말 종가 대비) 상승률 상위 2종목(상장 1년 이상)을 골라 16일부터 말일까지 "
                   "V1 타점을 기다립니다. 고점은 20일 종가 신고가(상승률 조건 없음), 청산은 V1과 같습니다.")
    monthly = True


class VelezV3(VelezV1):
    name = "velez_v3"
    label = "V3 상위 종목의 150·200일선 눌림 타점"
    description = ("V2 목록에 오른 지 120거래일 안의 종목이 최근 5거래일 안에 150일선이나 200일선 ±3%에 닿고 "
                   "3연속 음봉 눌림이 나오면 전봉 고가 돌파에 삽니다. 손절은 V1과 같고, 종가가 200일선보다 3% "
                   "넘게 아래면 매도, 최대 120거래일.")
    monthly = True
    ma_reentry = True
    tp1 = tp2 = None
    max_hold = MA_HOLD

    def take_profit(self, pos, i):
        return None


class VelezV4(VelezV1):
    name = "velez_v4"
    label = "V4 V1 + 2R 절반, 나머지 20일선 이탈"
    description = "V1과 같고, 2R에서 절반을 판 뒤 손절을 매수가로 두고 나머지는 종가 20일선 이탈에 팝니다. 최대 60거래일."
    ma_exit = 20
    protect = False
    max_hold = LONG_HOLD


class VelezV5(VelezV1):
    name = "velez_v5"
    label = "V5 V1 + 손절 -7% 고정, 20일선 이탈"
    description = "V1과 같이 사고, 손절은 -7% 고정, 익절 없이 종가 20일선 이탈에 팝니다. 최대 60거래일."
    fixed_stop = 0.07
    ma_exit = 20
    max_hold = LONG_HOLD


class VelezV6(VelezV2):
    name = "velez_v6"
    label = "V6 V2 + 고점 이후 음봉 3개 (연속 아님)"
    description = "V2와 같고, 음봉 3개를 연속이 아니라 고점 이후 누적으로 셉니다(신호 전날 종가 < 고점 종가)."
    loose = True


def _sweep(base: type, param: str, value, text: str, **extra) -> type:
    """One number changed: the rulebook's sensitivity check."""
    tag = str(value).replace(".", "")
    return type(f"{base.__name__}_{param}_{tag}", (base,), {
        "name": f"{base.name}_{param}_{tag}", "label": f"{base.label[:2]} 민감도: {text}",
        "description": f"{base.label[:2]}과 같고 {text}만 바꿉니다.", param: value, **extra})


SWEEP = [_sweep(VelezV1, "rise", 0.10, "상승 기준 10%"), _sweep(VelezV1, "rise", 0.25, "상승 기준 25%"),
         _sweep(VelezV1, "n_red", 4, "음봉 4개"),
         _sweep(VelezV1, "tp1", 1.5, "익절 1.5R 절반·2R 나머지", tp2=2.0),
         _sweep(VelezV1, "tp1", 1.5, "익절 1.5R 절반·3R 나머지"),
         _sweep(VelezV2, "top_n", 5, "상위 5종목"), _sweep(VelezV2, "top_n", 10, "상위 10종목"),
         _sweep(VelezV1, "regime", "bull", "강세장(S&P500 200일선 위)에서만 매수"),
         _sweep(VelezV1, "regime", "bear", "약세장(S&P500 200일선 아래)에서만 매수"),
         _sweep(VelezV1, "entry_day_stop", False, "매수 당일 손절 안 봄(낙관적 상한)"),
         _sweep(VelezV2, "entry_day_stop", False, "매수 당일 손절 안 봄(낙관적 상한)")]
SWEEP[3].name += "_2r"
VELEZ = [VelezV1, VelezV2, VelezV3, VelezV4, VelezV5, VelezV6, *SWEEP]
