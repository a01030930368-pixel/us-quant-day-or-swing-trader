"""'신고가 + 20일선 눌림 + 5일선 돌파' from a Korean YouTube video (부자회사원), rulebook 03.

A review of 차트박사's book: put a stock on the watch list when it makes a
60-day closing high with a sign of strength (an all-time-ish high on big
volume, record volume, or a touch of the upper envelope on big volume). Buy
when it has drifted back to a rising 20-day average on light volume, stayed
above the old high it broke, and then a candle opens and closes above the
5-day average. Sell when the close drops under the 5-day average, at -5%, or
on a big-volume down candle.

All numbers are from /mnt/project-files/rulebooks/03-차트박사-신고가-20일눌림-5일선돌파.md
([원전] said in the video, [해석] our numbers, [근사] automated stand-in,
[추가] added), fixed before the first run. Test list:

S1  the rules above, buy at the next open, -5% stop
S2  S1 with a -7% stop
S3  S1, watch list only from a 1-year high on big volume
S4  S1, but sell half at +10% and the rest on a close under the 20-day line
S5  S1, buy at the signal day's close
S6  김형준 'quiet candle' entry: a small, low-volume candle near the 5-day
    line within 5 days of the new high
S7  the 7x trader's bottom breakout (a 120-day box broken on 2x volume)
    instead of the 60-day high, then the S1 entry and exits
plus S1 with one number changed (section 11 sensitivity list).

Not automated [미구현]: the daily -2% stop-buying limit (the engine does not
tell a strategy its account value) and the weekly-chart bonus (2-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import sma
from .strategies import Strategy

HIGH_DAYS = 60
YEAR = 252
BIG_VOL = 3.0             # "대량 거래량" = 3x the 50-day average
ENVELOPE = 0.40
ENV_LOOKBACK = 10
WATCH = 30
SLOPE_DAYS = 5
NEAR_20 = 0.02
BREAK_20 = 0.02
PULL_VOL = 0.40
CRASH = -0.07
NEAR_5 = 0.01
STOP = 0.05
BIG_DOWN = 1.5
BIG_DOWN_ANY = 2.0
MAX_HOLD = 40
RISK_PER_TRADE = 0.01
MIN_DOLLAR_VOL = 20e6
MIN_PRICE = 10.0
QUIET_DAYS = 5
QUIET_BODY = 0.015
QUIET_NEAR = 0.02
BOX_DAYS = 120
BOX_WIDTH = 0.30
BOX_VOL = 2.0
HALF_AT = 0.10


def watch_starts(c, h, v, sma20, avg50, high_days=HIGH_DAYS, big_vol=BIG_VOL,
                 envelope=ENVELOPE, only_year_high=False, bottom=False, l=None):
    """(new_high, start, old_high): new-high days, days that open a watch, the old high each one broke."""
    prior = c.rolling(high_days, min_periods=high_days).max().shift(1)
    new_high = c > prior
    if bottom:                                    # S7: break out of a tight 120-day base on 2x volume
        top = h.rolling(BOX_DAYS, min_periods=BOX_DAYS).max().shift(1)
        low = l.rolling(BOX_DAYS, min_periods=BOX_DAYS).min().shift(1)
        start = (c > top) & ((top - low) / low <= BOX_WIDTH) & (v >= BOX_VOL * avg50)
        return new_high | start, start, top.where(start, prior)
    loud = v >= big_vol * avg50
    year_high = (c > c.rolling(YEAR, min_periods=YEAR).max().shift(1)) & loud
    record_vol = v > v.rolling(YEAR, min_periods=YEAR).max().shift(1)
    env_hit = ((h >= sma20 * (1 + envelope)) & loud).astype(float)
    env = env_hit.rolling(ENV_LOOKBACK + 1, min_periods=1).max() > 0
    strong = year_high if only_year_high else (year_high | record_vol | env)
    return new_high, new_high & strong, prior


def _signals_one(o, h, l, c, v, s5, s20, avg50, vol5, nh, start, old, liquid, p):
    """One ticker's pullback-and-5-day-line buy days, scored by the watch day's volume ratio."""
    n = len(c)
    out = np.full(n, np.nan)
    o, h, l, c, v, s5, s20, avg50, vol5, nh, start, old, liquid = (
        x.tolist() for x in (o, h, l, c, v, s5, s20, avg50, vol5, nh, start, old, liquid))
    anchor, score = -1, np.nan
    for i in range(1, n):
        if nh[i] and (start[i] or (anchor >= 0 and i - anchor <= p["watch"])):
            if start[i]:
                score = v[i] / avg50[i] if avg50[i] > 0 else np.nan
            anchor, old_high, a_vol = i, old[i], v[i]
            touched, min_c, sum_v, n_v, crash, quiet_done = False, np.inf, 0.0, 0, False, False
            continue
        if anchor < 0 or i - anchor > p["watch"]:
            continue
        if not s20[i] == s20[i] or c[i] < s20[i] * (1 - BREAK_20):
            anchor = -1                                   # broke the 20-day line: off the list
            continue
        if p["quiet"]:
            if not quiet_done and i - anchor <= QUIET_DAYS and liquid[i] and vol5[i] == vol5[i]:
                if (abs(c[i] - o[i]) / o[i] <= QUIET_BODY and abs(c[i] - s5[i]) / s5[i] <= QUIET_NEAR
                        and v[i] < vol5[i]):
                    out[i], quiet_done = score, True
        else:
            touched_now = touched or l[i] <= s20[i] * (1 + NEAR_20)
            if p["b6"] == "close":
                above5 = c[i] > s5[i]
            elif p["b6"] == "low":
                above5 = l[i] > s5[i] and c[i] > o[i]
            else:
                above5 = o[i] > s5[i] and c[i] > o[i]
            ok = (n_v >= 1 and touched_now and liquid[i]
                  and s20[i] > s20[i - SLOPE_DAYS]
                  and min_c > old_high and c[i] > old_high
                  and sum_v / n_v <= p["pull_vol"] * a_vol and not crash
                  and c[i - 1] <= s5[i - 1] * (1 + NEAR_5) and above5
                  and (not p["b7"] or (v[i] > v[i - 1] and v[i] >= vol5[i])))
            if ok:
                out[i] = score
        touched = touched or l[i] <= s20[i] * (1 + NEAR_20)
        min_c = min(min_c, c[i])
        sum_v += v[i]
        n_v += 1
        crash = crash or c[i] / c[i - 1] - 1 <= CRASH
    return out


class ChartDocS1(Strategy):
    name = "chartdoc_s1"
    label = "S1 신고가 + 20일선 눌림 + 5일선 돌파"
    description = ("60일 종가 신고가를 1년 고점·대량 거래량, 1년 최대 거래량, 엔벨로프(20, 40%) 상단 터치 중 "
                   "하나와 함께 낸 종목을 30거래일 지켜봅니다. 우상향 20일선 2% 안까지 거래량이 줄며 눌리고 "
                   "(신고가 거래량의 40% 이하, 20일선 2% 넘게 깨지 않음, 바닥이 전고점 위) 양봉 시가가 5일선 위, "
                   "거래량이 늘면 다음 날 시가에 삽니다. 종가 5일선 이탈, -5%, 큰 거래량 음봉이면 팝니다. "
                   "S&P500이 200일선 위일 때만, 최대 40거래일.")
    max_positions = 5
    high_days, big_vol, envelope = HIGH_DAYS, BIG_VOL, ENVELOPE
    pull_vol, b6, b7 = PULL_VOL, "body", True
    stop_pct, big_down = STOP, BIG_DOWN
    only_year_high = bottom = quiet = False
    use_market = True
    half_at = None                       # S4: sell half here, then exit under the 20-day line
    at_close = False                     # S5: fill at the signal day's close

    def _prepare(self) -> None:
        t = self.tradable
        c, h, l, o, v = (x[t] for x in (self.c, self.h, self.l, self.o, self.v))
        self.sma5, self.sma20 = sma(c, 5), sma(c, 20)
        self.avg50 = v.rolling(50, min_periods=50).mean().shift(1)
        self.vol5 = v.rolling(5, min_periods=5).mean().shift(1)
        nh, start, old = watch_starts(c, h, v, self.sma20, self.avg50, self.high_days, self.big_vol,
                                      self.envelope, self.only_year_high, self.bottom, l)
        liquid = ((c * v).rolling(20, min_periods=20).mean() >= MIN_DOLLAR_VOL) & (c >= MIN_PRICE)
        p = {"watch": WATCH, "quiet": self.quiet, "b6": self.b6, "b7": self.b7, "pull_vol": self.pull_vol}
        arr = lambda x, k: x[k].to_numpy(dtype=float)
        self.rank = pd.DataFrame({k: _signals_one(
            arr(o, k), arr(h, k), arr(l, k), arr(c, k), arr(v, k), arr(self.sma5, k), arr(self.sma20, k),
            arr(self.avg50, k), arr(self.vol5, k), nh[k].to_numpy(), start[k].to_numpy(),
            arr(old, k), liquid[k].to_numpy(), p) for k in t}, index=c.index)

    def candidates(self, i: int) -> list[str]:
        if self.use_market and not self.market_ok.iloc[i]:
            return []
        r = self.rank.iloc[i].dropna()
        return list(r.sort_values(ascending=False).index)

    def weight(self, t: str, i: int) -> float:
        return min(RISK_PER_TRADE / self.stop_pct, 1 / self.max_positions)

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        return self.c[t].iloc[i - 1] if self.at_close else open_

    def initial_stop(self, t: str, i: int, price: float) -> float:
        return price * (1 - self.stop_pct)

    def take_profit(self, pos: dict, i: int):
        if self.half_at is None or pos.get("half"):
            return None
        if self.h[pos["ticker"]].iloc[i] >= pos["entry_price"] * (1 + self.half_at):
            pos["half"] = True
            pos["stop"] = max(pos["stop"], pos["entry_price"])     # the rest: stop up to the buy price
        return pos["entry_price"] * (1 + self.half_at), 0.5, f"+{self.half_at:.0%} 절반 매도"

    def exit_reason(self, pos: dict, i: int):
        t = pos["ticker"]
        c, o, v = self.c[t].iloc[i], self.o[t].iloc[i], self.v[t].iloc[i]
        v5 = self.vol5[t].iloc[i]
        if i - pos["entry_i"] + 1 >= MAX_HOLD:
            return f"{MAX_HOLD}거래일 보유 완료"
        if pd.notna(v5) and ((c < o and v >= self.big_down * v5)
                             or (c < self.c[t].iloc[i - 1] and v >= BIG_DOWN_ANY * v5)):
            return "큰 거래량 음봉"
        if self.half_at is not None:
            if c < self.sma20[t].iloc[i]:
                return "종가 20일선 이탈"
        elif c < self.sma5[t].iloc[i]:
            return "종가 5일선 이탈"
        return None


class ChartDocS2(ChartDocS1):
    name = "chartdoc_s2"
    label = "S2 S1 + 손절 -7%"
    description = "S1과 같고, 손절만 -7%입니다(비중은 손실 1%에 맞춰 약 14%)."
    stop_pct = 0.07


class ChartDocS3(ChartDocS1):
    name = "chartdoc_s3"
    label = "S3 S1 + 1년 고점·대량 거래량 종목만"
    description = "S1과 같고, 관심 목록을 '1년 최고 종가 + 거래량 3배'로 신고가를 낸 종목으로만 좁힙니다."
    only_year_high = True


class ChartDocS4(ChartDocS1):
    name = "chartdoc_s4"
    label = "S4 S1 + 10% 절반 매도, 나머지 20일선 이탈"
    description = ("S1과 같이 사고, +10%에서 절반을 팔고 손절을 매수가로 올립니다. 나머지는 종가가 20일선 "
                   "아래로 내려가면 팝니다(5일선 이탈 매도 없음).")
    half_at = HALF_AT


class ChartDocS5(ChartDocS1):
    name = "chartdoc_s5"
    label = "S5 S1 + 신호일 종가 매수"
    description = "S1과 같고, 다음 날 시가가 아니라 신호일 종가에 삽니다(장 마감 직전 매수)."
    at_close = True


class ChartDocS6(ChartDocS1):
    name = "chartdoc_s6"
    label = "S6 신고가 직후 수급 음봉 매수"
    description = ("S1과 같은 관심 목록에서, 신고가 뒤 5거래일 안에 몸통 1.5% 이하·5일선 2% 안·거래량이 "
                   "5일 평균보다 적은 작은 캔들이 나오면 다음 날 시가에 삽니다. 매도는 S1과 같습니다.")
    quiet = True


class ChartDocS7(ChartDocS1):
    name = "chartdoc_s7"
    label = "S7 바닥권 탈출 + 20일선 눌림 + 5일선 돌파"
    description = ("120거래일 박스(폭 30% 이하)를 거래량 2배로 뚫은 종목을 지켜보고, 매수·매도는 S1과 "
                   "같습니다(7배 투자자 방식).")
    bottom = True


def _sweep(param: str, value, text: str) -> type:
    """S1 with one number changed: the rulebook's sensitivity check."""
    tag = str(value).replace(".", "")
    return type(f"ChartDocS1_{param}_{tag}", (ChartDocS1,), {
        "name": f"chartdoc_s1_{param}_{tag}", "label": f"S1 민감도: {text}",
        "description": f"S1과 같고 {text}만 바꿉니다.", param: value})


SWEEP = [_sweep("high_days", 120, "신고가 120일"), _sweep("high_days", 250, "신고가 250일"),
         _sweep("big_vol", 2.0, "대량 거래량 2배"), _sweep("big_vol", 5.0, "대량 거래량 5배"),
         _sweep("pull_vol", 0.30, "눌림 거래량 30%"), _sweep("pull_vol", 0.60, "눌림 거래량 60%"),
         _sweep("b6", "close", "5일선 돌파를 종가만으로"), _sweep("b6", "low", "5일선 돌파를 저가까지"),
         _sweep("envelope", 0.20, "엔벨로프 20%"), _sweep("envelope", 0.30, "엔벨로프 30%"),
         _sweep("big_down", 2.0, "큰 거래량 음봉 2배"), _sweep("use_market", False, "상승장 필터 없음"),
         _sweep("b7", False, "신호일 거래량 증가 조건 없음")]
CHARTDOC = [ChartDocS1, ChartDocS2, ChartDocS3, ChartDocS4, ChartDocS5, ChartDocS6, ChartDocS7, *SWEEP]
