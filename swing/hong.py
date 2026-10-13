"""홍인기 '주도주 첫 장대양봉' from a Korean YouTube video (부자회사원), rulebook 06.

Find today's leading sector: among the 30 stocks with the most dollar volume,
three or more stocks of one industry (GICS sub-industry) are up 5% or more.
Trade only that sector's top gainer, and only if today is its first big
up-candle (+10%) through its 6-month high on 3x its usual dollar volume. Buy
at the close and get out within three days: sell at the next open on a gap
down of more than 1%, take +5% on day 2 or day 3, stop out at -3% on day 3,
and sell what is left at the day-3 close.

All numbers are from /mnt/project-files/rulebooks/06-홍인기-주도주-첫장대양봉.md
([원전] said in the video, [해석] our numbers, [근사] automated stand-in,
[추가] added), fixed before the first run. Test list:

H1  the book: leader of the leading sector, first big candle, buy at the close
H2  H1 ranking dollar volume against its own 50-day average instead of in dollars
H3  H1 with a +10% profit target
H4  H1 sold at the day-2 close no matter what
H5  day-trading stand-in: buy back at day 1's close on day 2, sell at day 2's close
H6  H1 with a 1-year (252-day) high instead of 6 months
H7  H1 for stocks that have had 2+ explosive days in the last year ('끼')
H8  H1 held until the close drops below the 5-day line (at most 20 days)
H9  H1 only while the S&P 500 is above its 200-day line
H10 H1 on the S&P 1500 (small caps added)
H11 the first big candle alone, no leading-sector condition
H12 H1 bought at the prior high during the day (uses the day's close to pick the
    leader, so it peeks at the future)
plus H1 with one number changed (section 10 sensitivity list).

Not automated [미구현]: after-hours prices, intraday program buying, broker
flows, switching to a new leader during the day, the 30-minute wait.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import sma
from .strategies import Strategy

TOP_N = 30                 # dollar-volume rank checked [원전 15위, 없으면 30위]
MOVER = 0.05               # 'up 5% or more' [원전]
SAME = 3                   # stocks of one industry that make a leading sector [원전]
BIG = 0.10                 # big up-candle [원전 10%]
HIGH_DAYS = 126            # 6-month high [원전 책]
VALUE_X = 3.0              # dollar volume vs its 50-day average [해석]
KKI_DAYS, KKI_MIN = 250, 2 # '끼': explosive days in the last year [해석]
GAP = 0.01                 # day-2 gap down [원전]
TARGET = 0.05              # profit target vs the previous close [원전 5~10%]
DAY3_STOP = 0.03           # day-3 stop vs day 2's close [원전 3~5%]
HOLD_DAYS = 20             # H8 longest hold [해석]
PER_DAY = 2                # H11 new buys a day [해석]
WEIGHT = 0.25              # [해석]
MAX_POS = 4                # [해석]
MIN_DOLLAR_VOL = 20e6      # [추가]
MIN_PRICE = 10.0           # [추가]
MEMBERS = (500, 400)       # S&P 500 + 400 [해석]; H10 adds 600


def leader_signals(c: np.ndarray, o: np.ndarray, h: np.ndarray, value: np.ndarray, avg50: np.ndarray,
                   liquid: np.ndarray, group: np.ndarray, *, by_ratio: bool = False, top_n: int = TOP_N,
                   mover: float = MOVER, same: int = SAME, big: float = BIG, high_days: int = HIGH_DAYS,
                   value_x: float = VALUE_X, sector: bool = True, kki: np.ndarray | None = None,
                   per_day: int = PER_DAY) -> np.ndarray:
    """Return a days x stocks array: the day-1 gain where a stock is a buy at that close, else NaN.

    Arrays are days x stocks. group holds an integer industry code per stock
    (-1 = unknown). Everything on day i uses data up to day i's close.
    """
    n, k = c.shape
    out = np.full((n, k), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.full((n, k), np.nan)
        gain[1:] = c[1:] / c[:-1] - 1
        ratio = value / avg50
    prior_high = pd.DataFrame(h).rolling(high_days, min_periods=high_days).max().shift(1).to_numpy()
    candle = (gain >= big) & (c > o) & (c > prior_high)                  # B2 + B3
    loud = ratio >= value_x                                              # B5
    last_block = np.full(k, -10 ** 9)                                    # last breakout that was not a buy
    for i in range(n):
        ok = liquid[i]
        first = i - last_block > high_days                               # B4 (a traded breakout does not block)
        ready = candle[i] & loud[i] & first & ok
        if kki is not None:
            ready &= kki[i]
        picks: list[int] = []
        if sector:
            score = np.where(ok, ratio[i] if by_ratio else value[i], np.nan)
            valid = np.flatnonzero(~np.isnan(score))
            if len(valid):
                top = valid[np.argsort(-score[valid])[:top_n]]          # L1
                movers = top[(gain[i, top] >= mover) & (group[top] >= 0)]
                for g in np.unique(group[movers]):
                    members = movers[group[movers] == g]
                    if len(members) >= same:                             # L2
                        lead = members[np.argmax(gain[i, members])]      # L3: only the top gainer
                        if ready[lead]:
                            picks.append(lead)
        else:
            idx = np.flatnonzero(ready)
            picks = list(idx[np.argsort(-gain[i, idx])[:per_day]])
        for j in picks:
            out[i, j] = gain[i, j]
        blocked = candle[i] & ~np.isin(np.arange(k), picks)
        last_block[blocked] = i
    return out


class HongH1(Strategy):
    name = "hong_h1"
    label = "H1 주도 업종 1등주 첫 장대양봉 종가 매수, 3일 안 정리"
    description = ("거래대금 상위 30위 안에서 +5% 넘게 오른 종목이 같은 세부 업종에 3개 이상이면 주도 업종, 그 "
                   "업종 상승률 1등이 +10% 양봉으로 6개월(126일) 고가를 처음 뚫고 거래대금이 50일 평균의 3배 "
                   "이상이면 종가에 삽니다(25%, 최대 4종목). 다음 날 시가가 1% 넘게 갭 하락이면 시가 매도, "
                   "전날 종가 +5%면 익절. 셋째 날은 둘째 날 종가 +5% 익절, -3% 손절, 아니면 종가에 정리합니다.")
    max_positions = MAX_POS
    close_fill = True
    members = MEMBERS
    by_ratio = False
    top_n, mover, same = TOP_N, MOVER, SAME
    big, high_days, value_x = BIG, HIGH_DAYS, VALUE_X
    sector = True
    use_kki = False
    market_filter = False
    gap, target, day3_stop = GAP, TARGET, DAY3_STOP
    day2_stop = None            # sensitivity: stop at day 1's close x (1 - day2_stop) on day 2
    exit_mode = "three_days"    # "day2_close" (H4), "sma5" (H8), "day_trade" (H5)
    prior_high_fill = False     # H12

    def prepare(self, d: dict[str, pd.DataFrame], market: str = "SPY") -> "HongH1":
        self.info = d.get("INFO")
        return super().prepare(d, market)

    def _prepare(self) -> None:
        t = self.tradable
        c, o, h, v = (x[t] for x in (self.c, self.o, self.h, self.v))
        value = c * v
        avg50 = value.rolling(50, min_periods=50).mean().shift(1)
        liquid = (value.rolling(20, min_periods=20).mean() >= MIN_DOLLAR_VOL) & (c >= MIN_PRICE)
        info = self.info if self.info is not None else pd.DataFrame(columns=["index", "sub_industry"])
        info = info.reindex(t)
        member = info["index"].isin(self.members).to_numpy()
        liquid = liquid & member[None, :]
        codes, _ = pd.factorize(info["sub_industry"].replace("", np.nan))
        kki = None
        if self.use_kki:
            hot = ((c / c.shift(1) - 1 > self.big) & (value >= self.value_x * avg50)).astype(float)
            kki = (hot.rolling(KKI_DAYS, min_periods=1).sum().shift(1) >= KKI_MIN).to_numpy()
        sig = leader_signals(c.to_numpy(float), o.to_numpy(float), h.to_numpy(float), value.to_numpy(float),
                             avg50.to_numpy(float), liquid.fillna(False).to_numpy(bool), np.asarray(codes),
                             by_ratio=self.by_ratio, top_n=self.top_n, mover=self.mover, same=self.same,
                             big=self.big, high_days=self.high_days, value_x=self.value_x, sector=self.sector,
                             kki=kki)
        self.rank = pd.DataFrame(sig, index=c.index, columns=t)
        if self.market_filter:
            self.rank.loc[~self.market_ok.reindex(c.index).to_numpy(bool)] = np.nan
        self.prior_high = h.rolling(self.high_days, min_periods=self.high_days).max().shift(1)
        self.sma5 = sma(c, 5)

    def candidates(self, i: int) -> list[str]:
        r = self.rank.iloc[i].dropna()
        return list(r.sort_values(ascending=False).index)

    def weight(self, t: str, i: int) -> float:
        return WEIGHT

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        if self.prior_high_fill:                                         # H12 [근사, 미래 정보]
            lvl, o1 = self.prior_high[t].iloc[i - 1], self.o[t].iloc[i - 1]
            return o1 if o1 >= lvl else lvl
        return self.c[t].iloc[i - 1]                                     # B6: the signal day's close

    def _day1(self, pos: dict) -> float:
        return self.c[pos["ticker"]].iloc[pos["entry_i"] - 1]

    def initial_stop(self, t: str, i: int, price: float) -> float | None:
        if self.exit_mode == "three_days" and self.day2_stop is not None:
            return self.c[t].iloc[i - 1] * (1 - self.day2_stop)
        return None

    def open_exit(self, pos: dict, i: int) -> str | None:
        if self.exit_mode == "three_days" and i == pos["entry_i"] \
                and self.o[pos["ticker"]].iloc[i] < self._day1(pos) * (1 - self.gap):
            return "2일차 갭 하락 (시가 매도)"
        return None

    def take_profit(self, pos: dict, i: int):
        if self.exit_mode != "three_days":
            return None
        e, t = pos["entry_i"], pos["ticker"]
        if i == e:
            return self._day1(pos) * (1 + self.target), 1.0, "2일차 익절"
        if i == e + 1:
            return self.c[t].iloc[e] * (1 + self.target), 1.0, "3일차 익절"
        return None

    def update_stop(self, pos: dict, i: int) -> float | None:
        if self.exit_mode == "three_days" and i == pos["entry_i"]:
            return self.c[pos["ticker"]].iloc[i] * (1 - self.day3_stop)  # for day 3
        return pos.get("stop")

    def close_exit(self, pos: dict, i: int) -> str | None:
        e = pos["entry_i"]
        if self.exit_mode == "three_days" and i == e + 1:
            return "3일차 종가 정리"
        if self.exit_mode == "day2_close" and i == e:
            return "2일차 종가 매도"
        if self.exit_mode == "day_trade" and i == e:
            return "당일 종가 정리"
        return None

    def exit_reason(self, pos: dict, i: int) -> str | None:
        if self.exit_mode != "sma5":
            return None
        t = pos["ticker"]
        if self.c[t].iloc[i] < self.sma5[t].iloc[i]:
            return "종가 5일선 이탈"
        if i - pos["entry_i"] + 2 >= HOLD_DAYS:                          # counted from the buy day's close
            return f"{HOLD_DAYS}거래일 보유 완료"
        return None


class HongH2(HongH1):
    name = "hong_h2"
    label = "H2 H1 + 거래대금 순위를 평소 대비 배수로"
    description = "H1과 같고, 거래대금 상위 30위를 금액이 아니라 50일 평균 대비 배수로 고릅니다."
    by_ratio = True


class HongH3(HongH1):
    name = "hong_h3"
    label = "H3 H1 + 익절 +10%"
    description = "H1과 같고, 둘째·셋째 날 익절을 +10%로 올립니다."
    target = 0.10


class HongH4(HongH1):
    name = "hong_h4"
    label = "H4 H1 + 다음 날 종가에 무조건 매도"
    description = "H1과 같은 종목을 종가에 사서 다음 날 종가에 팝니다(갭·익절 규칙 없음)."
    exit_mode = "day2_close"


class HongH5(HongH1):
    name = "hong_h5"
    label = "H5 데이트레이딩 근사 (다음 날 전일 종가에 매수, 당일 정리)"
    description = ("H1 신호 다음 날, 주가가 신호일 종가까지 내려오면 그 가격에 삽니다(시가가 1% 넘게 갭 하락이면 "
                   "안 삼). 신호일 종가 -1% 손절(먼저 처리), 아니면 그날 종가에 팝니다.")
    close_fill = False
    exit_mode = "day_trade"

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        d1 = self.c[t].iloc[i - 1]
        if open_ < d1 * (1 - GAP):
            return None
        if open_ <= d1:
            return open_
        return d1 if self.l[t].iloc[i] <= d1 else None

    def initial_stop(self, t: str, i: int, price: float) -> float | None:
        return self.c[t].iloc[i - 1] * (1 - GAP)


class HongH6(HongH1):
    name = "hong_h6"
    label = "H6 H1 + 1년(252일) 신고가"
    description = "H1과 같고, 6개월 대신 252거래일 고가를 처음 뚫을 때만 삽니다."
    high_days = 252


class HongH7(HongH1):
    name = "hong_h7"
    label = "H7 H1 + 끼 있는 종목"
    description = "H1과 같고, 지난 250거래일에 +10% 넘게 오르며 거래대금 3배 이상인 날이 2일 이상인 종목만 삽니다."
    use_kki = True


class HongH8(HongH1):
    name = "hong_h8"
    label = "H8 H1 + 길게 보유 (종가 5일선 이탈 매도)"
    description = "H1과 같은 종목을 사서, 종가가 5일선 아래면 다음 날 시가에 팝니다(최대 20거래일)."
    exit_mode = "sma5"


class HongH9(HongH1):
    name = "hong_h9"
    label = "H9 H1 + S&P500 200일선 위에서만"
    description = "H1과 같고, S&P500 종가가 200일선 위일 때만 삽니다."
    market_filter = True


class HongH10(HongH1):
    name = "hong_h10"
    label = "H10 H1 + 소형주(S&P 600)까지"
    description = "H1과 같고, 대상에 S&P 600 소형주를 더합니다."
    members = (500, 400, 600)


class HongH11(HongH1):
    name = "hong_h11"
    label = "H11 주도 업종 조건 없이 첫 장대양봉만"
    description = "주도 업종·1등주 조건 없이 첫 장대양봉(+10%, 6개월 고가, 거래대금 3배)만 보고 하루 2종목까지 삽니다."
    sector = False


class HongH12(HongH1):
    name = "hong_h12"
    label = "H12 H1 + 전고점 가격에 장중 매수 (미래 정보 섞임)"
    description = ("H1과 같은 종목을 그날 장중 6개월 고가를 넘는 순간(시가가 이미 위면 시가)에 산 것으로 칩니다. "
                   "1등주는 종가로 정하므로 미래 정보가 섞여 있습니다.")
    prior_high_fill = True


def _sweep(base: type, param: str, value, text: str) -> type:
    """One number changed: the rulebook's sensitivity check."""
    tag = str(value).replace(".", "")
    return type(f"{base.__name__}_{param}_{tag}", (base,), {
        "name": f"{base.name}_{param}_{tag}", "label": f"{base.label[:2]} 민감도: {text}",
        "description": f"{base.label[:2]}과 같고 {text}만 바꿉니다.", param: value})


SWEEP = [_sweep(HongH1, "big", 0.07, "장대양봉 7%"), _sweep(HongH1, "big", 0.15, "장대양봉 15%"),
         _sweep(HongH1, "value_x", 2.0, "거래대금 2배"), _sweep(HongH1, "value_x", 5.0, "거래대금 5배"),
         _sweep(HongH1, "high_days", 63, "신고가 63일"),
         _sweep(HongH1, "top_n", 15, "거래대금 15위"), _sweep(HongH1, "top_n", 50, "거래대금 50위"),
         _sweep(HongH1, "same", 2, "같은 업종 2종목"), _sweep(HongH1, "same", 4, "같은 업종 4종목"),
         _sweep(HongH1, "gap", 0.02, "갭 하락 2%"), _sweep(HongH1, "day3_stop", 0.05, "3일차 손절 5%"),
         _sweep(HongH1, "day2_stop", 0.03, "2일차 손절 3%")]
HONG = [HongH1, HongH2, HongH3, HongH4, HongH5, HongH6, HongH7, HongH8, HongH9, HongH10, HongH11, HongH12,
        *SWEEP]
