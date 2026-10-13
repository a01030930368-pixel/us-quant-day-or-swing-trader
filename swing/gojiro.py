"""'이동평균선 대순환' (5·20·40일선) + 쿨라매기 돌파 from a Korean YouTube video (부자회사원), rulebook 05.

Koshiro's moving-average cycle: the order of the 5-, 20- and 40-day averages
splits the market into six stages. Buy at the close on the day a long-falling
stock's 5-day line, which has already crossed the 20-day line, now crosses the
40-day line (stage 6: 5 > 40 > 20), with a rising 20-day line and a flat
40-day line. Buy a full slice when the day's dollar volume explodes, half a
slice when it does not. Add when the 20-day line crosses the 40-day line and
again on a pullback to the 5-day line. Hold while the 20-day line rises.

All numbers are from /mnt/project-files/rulebooks/05-고지로-이평선대순환-쿨라매기.md
([원전] said in the video, [해석] our numbers, [근사] automated stand-in,
[추가] added), fixed before the first run. Test list:

K1  stage-6 buy at the close, sized by volume, two adds, chapter-4 exits
K2  'perfect order': the first day of stage 1 with all three lines rising
K3  K1 with the whole planned size on the first buy, no adds
K4  K1 + a MACD golden cross from below zero in the last 10 days
K5  Qullamaggie breakout of a 2-week to 2-month rest after a 30%+ run
K6  K1 on 10/20/50-day exponential averages (Qullamaggie's lines)
K7  K1 and K5 together in one account
K8  K1 with 'the three lines were bunched within 3% for 20 days' instead of
    '20 of the last 60 days in stage 4'
plus K1 or K5 with one number changed (section 10 sensitivity list).

Not automated [미구현]: the daily -2% stop-buying limit and the 30% overnight
limit (a day-trader's rule; the swing test allows up to 100% invested).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import ema, sma
from .strategies import Strategy

DOWN_LOOK, DOWN_DAYS = 60, 20
BUNCH_DAYS, BUNCH = 20, 0.03
CROSS_WINDOW = 20
SLOPE = 5
FLAT = 0.01
BIG_VALUE, BIG_GAIN = 2.0, 0.04
EXIT_SLOPE = 5
MAX_HOLD = 60
MAX_POS = 6
MIN_DOLLAR_VOL = 20e6
MIN_PRICE = 10.0
MACD_LOOK = 10
# Qullamaggie (chapter 5)
Q_RUN, Q_TOP, Q_MIN_RUN, Q_PRIOR = 63, 0.98, 0.30, 1.0
Q_REST_MIN, Q_REST_MAX = 10, 40
Q_UNDER_EMA = 0.05
Q_DRY, Q_LOUD = 0.70, 1.5
Q_SELL_DAY, Q_SELL_PART = 5, 1 / 3
RISK_PER_TRADE = 0.01
MAX_WEIGHT = 0.20


def lines(c: pd.DataFrame, kind: str):
    """Short, middle and long averages: 5/20/40 simple [원전 고지로] or 10/20/50 exponential [원전 쿨라매기]."""
    if kind == "ema":
        return ema(c, 10), ema(c, 20), ema(c, 50)
    return sma(c, 5), sma(c, 20), sma(c, 40)


def stage6_signals(o, c, s, m, L, down_days=DOWN_DAYS, bunched=False, bunch=BUNCH, flat=FLAT):
    """Days the short line crosses the long line to make stage 6 after a long fall (3-1 A1-A4)."""
    stage4 = (L > m) & (m > s)
    if bunched:
        width = (pd.concat([s, m, L]).groupby(level=0).max() - pd.concat([s, m, L]).groupby(level=0).min()
                 ).reindex(c.index) / c
        base = (width <= bunch).astype(float).rolling(BUNCH_DAYS, min_periods=BUNCH_DAYS).min().shift(1) > 0
    else:
        base = stage4.astype(float).rolling(DOWN_LOOK, min_periods=DOWN_LOOK).sum().shift(1) >= down_days
    cross_sm = (s > m) & (s.shift(1) <= m.shift(1))
    recent = cross_sm.astype(float).rolling(CROSS_WINDOW, min_periods=1).max() > 0
    cross_sl = (s > L) & (s.shift(1) <= L.shift(1))
    stage6 = (s > L) & (L > m)
    rising = (m > m.shift(SLOPE)) & ((L / L.shift(SLOPE) - 1).abs() <= flat)
    above = (c > s) & (c > o)
    return base & recent & cross_sl & stage6 & rising & above


def perfect_order(c, s, m, L):
    """First day of stage 1 (the middle line crosses the long one) with all three lines rising (3-4)."""
    first = (s > m) & (m > L) & (m.shift(1) <= L.shift(1))
    up = (s > s.shift(SLOPE)) & (m > m.shift(SLOPE)) & (L > L.shift(SLOPE))
    return first & up


def macd_ok(c: pd.DataFrame) -> pd.DataFrame:
    """A MACD(12, 26, 9) golden cross from a negative histogram in the last 10 days (3-5)."""
    line = ema(c, 12) - ema(c, 26)
    sig = line.ewm(span=9, min_periods=9, adjust=False).mean()
    hist = line - sig
    cross = (hist > 0) & (hist.shift(1) < 0)
    return cross.astype(float).rolling(MACD_LOOK + 1, min_periods=1).max() > 0


def qulla_setups(h, l, c, v, ema20, avg50, leader):
    """Armed days for the Qullamaggie breakout and the buy-stop price for the next day (chapter 5).

    At day j's close: a leader (top 2% of 63-day returns, +30% or more, not
    already +100% in the three months before) whose 63-day high is 10-40 days
    back, no low since then more than 5% under the 20-day EMA, last 5 days'
    volume at most 70% of the 50-day average, and still under the rest's top.
    The buy-stop is one cent over the highest high since the peak.
    """
    hv, lv, cv, vv, ev, av = (x.to_numpy(dtype=float) for x in (h, l, c, v, ema20, avg50))
    out = np.full(c.shape, np.nan)
    for j, k in zip(*np.nonzero(leader.to_numpy())):
        if j < Q_RUN:
            continue
        win = hv[j - Q_RUN + 1:j + 1, k]
        if np.isnan(win).all():
            continue
        p = j - Q_RUN + 1 + int(np.nanargmax(win))
        if not Q_REST_MIN <= j - p <= Q_REST_MAX:
            continue
        top = np.nanmax(hv[p + 1:j + 1, k])
        if cv[j, k] > top or np.nanmin(lv[p + 1:j + 1, k] / ev[p + 1:j + 1, k]) < 1 - Q_UNDER_EMA:
            continue
        if not np.nanmean(vv[j - 4:j + 1, k]) <= Q_DRY * av[j, k]:
            continue
        out[j, k] = top + 0.01
    return pd.DataFrame(out, index=c.index, columns=c.columns)


class GojiroK1(Strategy):
    name = "gojiro_k1"
    label = "K1 대순환 6단계 급소 종가 매수 + 2·3차 추가"
    description = ("60거래일 중 20일 이상 역배열(40>20>5일선)이던 종목의 5일선이 최근 20일 안에 20일선을 뚫고, "
                   "오늘 40일선까지 뚫어 5>40>20일선이 되면(20일선 상승, 40일선 5일 기울기 ±1% 안, 양봉, 종가 > "
                   "5일선) 종가에 삽니다. 거래대금 2배·+4% 이상이면 1/3, 아니면 1/6(한 종목 계획 비중 1/6 기준). "
                   "20일선이 40일선을 뚫으면 2차, 정배열 눌림에서 5일선 회복이면 3차(매수가 위에서만). 2차 전이나 "
                   "약한 진입은 종가 5일선 이탈, 그 뒤는 5<20일선이면서 20일선이 평평해지면, 20<40일선이면 매도. "
                   "최대 60거래일.")
    max_positions = MAX_POS
    kind = "sma"
    down_days, bunched, bunch, flat = DOWN_DAYS, False, BUNCH, FLAT
    big_value = BIG_VALUE
    exit_slope = EXIT_SLOPE
    adds = True
    split = True                          # False (K3): the whole planned size at once
    use_macd = False
    perfect = False                       # K2
    at_close = True                       # buy at the signal day's close [원전 종가 배팅]
    stage = True                          # trade the moving-average cycle
    qulla = False                         # trade the Qullamaggie breakout
    q_exit = 20

    def _prepare(self) -> None:
        t = self.tradable
        c, h, l, o, v = (x[t] for x in (self.c, self.h, self.l, self.o, self.v))
        self.s, self.m, self.L = lines(c, self.kind)
        self.sma5 = sma(c, 5)
        self.vol5 = v.rolling(5, min_periods=5).mean().shift(1)
        self.avg50 = v.rolling(50, min_periods=50).mean().shift(1)
        value = c * v
        liquid = (value.rolling(20, min_periods=20).mean() >= MIN_DOLLAR_VOL) & (c >= MIN_PRICE)
        self.big = (value >= self.big_value * value.rolling(50, min_periods=50).mean().shift(1)) \
            & (c / c.shift(1) - 1 >= BIG_GAIN)
        self.planned = 1 / self.max_positions
        self.kind_of: dict[str, str] = {}
        self.stage_rank = self.q_trigger = None
        if self.stage:
            if self.perfect:
                sig = perfect_order(c, self.s, self.m, self.L)
            else:
                sig = stage6_signals(o, c, self.s, self.m, self.L, self.down_days, self.bunched,
                                     self.bunch, self.flat)
            if self.use_macd:
                sig = sig & macd_ok(c)
            self.stage_rank = (value / value.rolling(50, min_periods=50).mean().shift(1)).where(sig & liquid)
        if self.qulla:
            self.ema20, self.ema10 = ema(c, 20), ema(c, 10)
            self.adr = (h / l).rolling(20, min_periods=20).mean() - 1
            ret = c / c.shift(Q_RUN) - 1
            prior = c.shift(Q_RUN) / c.shift(2 * Q_RUN) - 1
            leader = (ret.where(liquid).rank(axis=1, pct=True) >= Q_TOP) & (ret >= Q_MIN_RUN) \
                & ~(prior >= Q_PRIOR) & liquid
            self.q_trigger = qulla_setups(h, l, c, v, self.ema20, self.avg50, leader)
        self.pending: dict[str, dict] = {}
        self.trade: dict[str, dict] = {}

    # ----- entries
    def candidates(self, i: int) -> list[str]:
        picks, self.pending = [], {}
        if self.stage_rank is not None:
            r = self.stage_rank.iloc[i].dropna().sort_values(ascending=False)
            for t in r.index:
                self.pending[t] = {"kind": "stage", "big": bool(self.big[t].iloc[i])}
                picks.append(t)
        if self.q_trigger is not None:
            q = self.q_trigger.iloc[i].dropna()
            ret = (self.c.iloc[i] / self.c.iloc[i - Q_RUN] - 1)[q.index].sort_values(ascending=False)
            for t in ret.index:
                if t not in self.pending:
                    self.pending[t] = {"kind": "qulla", "trigger": q[t], "adr": self.adr[t].iloc[i]}
                    picks.append(t)
        return picks

    @property
    def order_all(self) -> bool:            # buy-stops: send every order, most never fill
        return self.qulla

    def weight(self, t: str, i: int) -> float:
        p = self.pending[t]
        if p["kind"] == "qulla":
            return min(RISK_PER_TRADE / max(p["adr"], 0.005), MAX_WEIGHT)
        if not self.split:
            return self.planned * (1.0 if p["big"] else 0.5)
        return self.planned / 3 * (1.0 if p["big"] else 0.5)

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        p = self.pending[t]
        if p["kind"] == "qulla":
            if open_ >= p["trigger"]:
                return open_
            return p["trigger"] if high >= p["trigger"] else None
        return self.c[t].iloc[i - 1] if self.at_close else open_

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        p = self.pending[t]
        self.trade[t] = {**p, "first": price, "last_buy": price, "switched": self.perfect, "sold_part": False}
        if self.perfect:
            self.trade[t]["big"] = True                     # K2 uses only the chapter-4 base exit
        return True

    # ----- adds (moving-average cycle only)
    def add_weight(self, pos: dict, i: int) -> float | None:
        t = pos["ticker"]
        tr = self.trade[t]
        if tr["kind"] != "stage" or not self.adds or not self.split or pos.get("adds", 0) >= 2:
            return None
        c = self.c[t].iloc[i]
        if c <= tr["last_buy"]:
            return None                                     # only above the last buy [추가]
        s, m, L = self.s[t], self.m[t], self.L[t]
        if pos.get("adds", 0) == 0:
            if m.iloc[i] > L.iloc[i] and m.iloc[i - 1] <= L.iloc[i - 1]:
                tr["last_buy"] = c
                return self.planned / 3                     # 2nd: middle line crosses the long one
            return None
        back = s.iloc[i] > m.iloc[i] and s.iloc[i - 1] <= m.iloc[i - 1] and m.iloc[i] > L.iloc[i]
        touch = (self.l[t].iloc[i] <= self.sma5[t].iloc[i] <= c
                 and self.v[t].iloc[i] >= self.vol5[t].iloc[i])
        if back or touch:
            tr["last_buy"] = c
            return self.planned / 3                         # 3rd: back from a stage-1 pullback
        return None

    # ----- exits
    def exit_reason(self, pos: dict, i: int):
        t = pos["ticker"]
        tr = self.trade[t]
        days = i - pos["entry_i"] + 1
        if days >= MAX_HOLD:
            return f"{MAX_HOLD}거래일 보유 완료"
        c = self.c[t].iloc[i]
        if tr["kind"] == "qulla":
            return self._qulla_exit(pos, tr, i, c, days)
        s, m, L = self.s[t].iloc[i], self.m[t].iloc[i], self.L[t].iloc[i]
        if not tr["switched"] and m > L:
            tr["switched"] = True                           # the 2nd-buy moment, bought or not
        if m < L and tr["switched"]:
            return "20일선 < 40일선 (단계 붕괴)"
        if not tr["switched"] or not tr["big"]:            # weak entries keep the 5-day stop [원전]
            if c < s:
                return "종가 5일선 이탈 (첫 손절)"
            return None
        if s < m and m <= self.m[t].iloc[i - self.exit_slope]:
            return "5일선 < 20일선, 20일선 평평"
        return None

    def _qulla_exit(self, pos, tr, i, c, days):
        t = pos["ticker"]
        if days == 1:
            lo = self.l[t].iloc[i]
            tr["stop_close"] = lo
            loud = self.v[t].iloc[i] >= Q_LOUD * self.avg50[t].iloc[i]
            if not loud or (tr["first"] - lo) / tr["first"] > tr["adr"] or c <= tr["trigger"]:
                return "돌파 불인정 (거래량·ADR·종가)"         # [근사] checked at the close
        if c < tr["stop_close"]:
            return "종가 손절"
        if days == Q_SELL_DAY and not tr["sold_part"]:
            tr["sold_part"] = True
            tr["stop_close"] = max(tr["stop_close"], tr["first"])
            return "5일째 1/3 매도", Q_SELL_PART
        line = self.ema10 if self.q_exit == 10 else self.ema20
        if c < line[t].iloc[i]:
            return f"종가 {self.q_exit}일 지수 이동평균 이탈"
        return None


class GojiroK2(GojiroK1):
    name = "gojiro_k2"
    label = "K2 퍼펙트 오더 (정배열 첫날, 세 선 우상향)"
    description = ("5>20>40일선 정배열이 처음 된 날(20일선이 40일선 돌파), 세 선이 모두 5일 전보다 높으면 종가에 "
                   "계획 비중을 다 삽니다. 매도는 K1의 2차 이후 규칙과 같습니다.")
    perfect = True
    split = False
    adds = False

    def weight(self, t, i):
        return self.planned


class GojiroK3(GojiroK1):
    name = "gojiro_k3"
    label = "K3 K1 + 1차에 전부 매수 (추가 없음)"
    description = "K1과 같고, 1차에 계획 비중을 모두 삽니다(거래대금이 약하면 절반). 추가 매수는 없습니다."
    split = False
    adds = False


class GojiroK4(GojiroK1):
    name = "gojiro_k4"
    label = "K4 K1 + MACD 필터"
    description = "K1과 같고, 최근 10거래일 안에 MACD가 음수 히스토그램에서 골든크로스한 종목만 삽니다."
    use_macd = True


class GojiroK5(GojiroK1):
    name = "gojiro_k5"
    label = "K5 쿨라매기 돌파"
    description = ("63거래일 수익률 상위 2%이면서 +30% 이상(그 전 3개월 +100% 이상이면 제외) 종목이 고점 뒤 "
                   "10~40거래일 쉬며(저가가 20일 지수 이동평균 5% 아래로 안 감, 마지막 5일 거래량 50일 평균의 70% "
                   "이하) 쉬는 구간 최고가 + 1센트를 넘으면 삽니다. 그날 거래량 1.5배 미만, 손절폭 > ADR, 종가가 "
                   "돌파선 아래면 다음 날 시가에 팝니다. 종가가 돌파일 저가 아래면 손절, 5일째 1/3 매도 후 손절을 "
                   "매수가로, 나머지는 종가 20일 지수 이동평균 이탈. 손실 1% 비중(최대 20%).")
    stage = False
    qulla = True


class GojiroK6(GojiroK1):
    name = "gojiro_k6"
    label = "K6 K1을 10·20·50일 지수 이동평균으로"
    description = "K1과 같고, 5·20·40일 단순 이동평균 대신 10·20·50일 지수 이동평균을 씁니다."
    kind = "ema"


class GojiroK7(GojiroK1):
    name = "gojiro_k7"
    label = "K7 K1 + K5 (대순환 + 쿨라매기)"
    description = "K1과 K5 신호를 한 계좌에서 함께 매매합니다(같은 날 겹치면 K1 우선). 각자 매도 규칙을 씁니다."
    qulla = True


class GojiroK8(GojiroK1):
    name = "gojiro_k8"
    label = "K8 K1 + 세 선이 20일 이상 3% 안에 뭉침"
    description = "K1과 같고, '역배열 기간' 대신 '신호 전 20거래일 동안 세 선 간격이 종가의 3% 이내'를 씁니다."
    bunched = True


def _sweep(base: type, param: str, value, text: str) -> type:
    """One number changed: the rulebook's sensitivity check."""
    tag = str(value).replace(".", "")
    return type(f"{base.__name__}_{param}_{tag}", (base,), {
        "name": f"{base.name}_{param}_{tag}", "label": f"{base.label[:2]} 민감도: {text}",
        "description": f"{base.label[:2]}과 같고 {text}만 바꿉니다.", param: value})


SWEEP = [_sweep(GojiroK1, "down_days", 10, "역배열 10일"), _sweep(GojiroK1, "down_days", 40, "역배열 40일"),
         _sweep(GojiroK8, "bunch", 0.02, "뭉침 폭 2%"), _sweep(GojiroK8, "bunch", 0.05, "뭉침 폭 5%"),
         _sweep(GojiroK1, "flat", 0.005, "40일선 평평 0.5%"), _sweep(GojiroK1, "flat", 0.02, "40일선 평평 2%"),
         _sweep(GojiroK1, "big_value", 1.5, "거래대금 1.5배"), _sweep(GojiroK1, "big_value", 3.0, "거래대금 3배"),
         _sweep(GojiroK1, "exit_slope", 3, "20일선 기울기 3일"), _sweep(GojiroK1, "exit_slope", 10, "20일선 기울기 10일"),
         _sweep(GojiroK1, "at_close", False, "다음 날 시가 매수"),
         _sweep(GojiroK5, "q_exit", 10, "10일 지수 이동평균 청산")]
GOJIRO = [GojiroK1, GojiroK2, GojiroK3, GojiroK4, GojiroK5, GojiroK6, GojiroK7, GojiroK8, *SWEEP]
