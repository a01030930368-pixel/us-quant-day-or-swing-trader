"""남석관 '장대양봉 후 첫 5일선' + 수급단타왕 '수급 1·2 음봉' from a Korean YouTube video (부자회사원), rulebook 07.

The day a whole industry jumps, its top gainer makes its first +10% candle
after a long sideways stretch, through its 6-month high on 3x volume. Buy 10%
of the account at that close, 20% more the first time the price comes back to
the 5-day line (within 10 days), and 10% more twice while it keeps making new
highs above the 5- and 10-day lines. Sell when the close drops below the
10-day line or below the big candle's low.

All numbers are from /mnt/project-files/rulebooks/07-남석관-장대양봉후-첫5일선-분할매수.md
([원전] said in the video, [해석] our numbers, [근사] automated stand-in,
[추가] added), fixed before the first run. The lines are exponential moving
averages (the video's chart setting). Test list:

G1  signal-day close 10% + first 5-day-line touch 20% + two 10% adds
G2  G1 without the 'whole industry jumps / top gainer' condition
G3  G1, the touch counts only on a dry-volume day (bought at the next open)
G4  '수급 음봉': a quiet down day within 2 days after the big candle, bought at its close
G5  G1 without the two adds
G6  G1 selling half below the 5-day line, the rest below the 10-day line
G7  no signal-day buy: 20% at the first 5-day-line touch, then the adds
G8  G1 for companies whose latest quarter grew sales and EPS year over year
G9  20% at the close of a day that comes back with volume above the 5-day line
plus G1 with one number changed (section 9 sensitivity list).

Not automated [미구현]: buyer flows (foreign, institutional, program),
news, the market-phase reading, the 'big-volume red candle' warning,
choosing the industry's biggest company instead of its top gainer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import ema, sma
from .strategies import Strategy

MOVER, SAME = 0.05, 3       # G-1 [해석]
BIG = 0.10                  # G-4 [화면]
FLAT_DAYS, FLAT = 60, 0.30  # G-3 [해석]
HIGH_DAYS = 126             # G-5 [해석]
VOL_X = 3.0                 # G-5 [해석]
CUM = 0.30                  # sensitivity: 30% above the sideways low [해석]
WAIT = 10                   # B1/B5 [해석]
SIGNAL_W, TOUCH_W, ADD_W, MAX_W = 0.10, 0.20, 0.10, 0.50   # 3-0, B1 [원전 20%], P1, P4
ADD_GAIN = 0.05             # P2 [해석]
BREAKEVEN = 0.10            # 본절 컷 [해석]
MAX_HOLD = 60               # [추가]
MAX_POS = 5                 # [해석]
DRY_DAYS = 2                # G4: within 1-2 days [원전]
DRY_DROP, DRY_VOL = 0.03, 0.5   # G4 [해석]
LOUD_FROM, LOUD_TO, LOUD_X = 2, 10, 1.5   # G9 [해석]
MIN_DOLLAR_VOL = 20e6       # [추가]
MIN_PRICE = 10.0            # [추가]
MEMBERS = (500, 400)        # S&P 500 + 400 [해석]


def industry_leaders(gain: np.ndarray, liquid: np.ndarray, group: np.ndarray, mover: float = MOVER,
                     same: int = SAME) -> np.ndarray:
    """days x stocks: True where the stock is the top gainer of an industry with `same`+ stocks up `mover`+."""
    n, k = gain.shape
    out = np.zeros((n, k), bool)
    for i in range(n):
        up = np.flatnonzero(liquid[i] & (gain[i] >= mover) & (group >= 0))
        if len(up) < same:
            continue
        for g in np.unique(group[up]):
            members = up[group[up] == g]
            if len(members) >= same:
                out[i, members[np.argmax(gain[i, members])]] = True
    return out


class NamG1(Strategy):
    name = "namsg_g1"
    label = "G1 단체 상승 1등주 첫 장대양봉 10% + 첫 5일선 20% + 추가"
    description = ("같은 세부 업종에서 +5% 넘게 오른 종목이 3개 이상인 날, 그 업종 상승률 1등이 60거래일 동안 30% "
                   "안에서 옆으로 기다가 처음으로 +10%(6개월 고가 돌파, 거래량 50일 평균 3배)를 만들면 종가에 계좌 10%. "
                   "다음 10거래일 안에 처음 5일 지수 이동평균에 닿으면 그 가격에 20%, 이후 종가 신고가·종가 > 5일선 > "
                   "10일선·평균 단가 +5% 이상이면 다음 날 시가에 10%씩(한 종목 최대 50%). 종가가 10일선 아래거나 "
                   "장대양봉 저가 아래면 다음 날 시가 매도, +10% 갔다가 평균 단가 아래면 매도, 최대 60거래일.")
    max_positions = MAX_POS
    close_fill = True
    kind = "ema"
    group_rule = True
    mover, same = MOVER, SAME
    big, flat, vol_x = BIG, FLAT, VOL_X
    flat_days, high_days = FLAT_DAYS, HIGH_DAYS
    cum = False
    wait = WAIT
    exit_ma = 10
    entry = "signal"            # "signal" (G1), "touch" (G7), "dry" (G4), "loud" (G9)
    signal_buy = True
    touch_buy = True
    dry_touch = False           # G3
    adds = True                 # G5: False
    half_at_5 = False           # G6
    earnings = False            # G8

    def _prepare(self) -> None:
        t = self.tradable
        c, o, h, l, v = (x[t] for x in (self.c, self.o, self.h, self.l, self.v))
        info = getattr(self, "info", None)
        info = (info if info is not None else pd.DataFrame(columns=["index", "sub_industry"])).reindex(t)
        member = info["index"].isin(MEMBERS).to_numpy()
        value = c * v
        liquid = ((value.rolling(20, min_periods=20).mean() >= MIN_DOLLAR_VOL) & (c >= MIN_PRICE)).to_numpy()
        liquid = liquid & member[None, :]
        gain = (c / c.shift(1) - 1)
        avg50 = v.rolling(50, min_periods=50).mean().shift(1)
        fd = self.flat_days
        hi60 = h.rolling(fd, min_periods=fd).max().shift(1)
        lo60 = l.rolling(fd, min_periods=fd).min().shift(1)
        prior_high = h.rolling(self.high_days, min_periods=self.high_days).max().shift(1)
        earlier = (gain >= self.big).astype(float).rolling(fd, min_periods=1).sum().shift(1)
        candle = (gain >= self.big) & (c > prior_high) & (v >= self.vol_x * avg50) \
            & (hi60 / lo60 - 1 <= self.flat) & (earlier == 0)
        if self.cum:
            candle &= c / lo60 - 1 >= CUM
        sig = candle.to_numpy(bool) & liquid
        if self.group_rule:
            codes, _ = pd.factorize(info["sub_industry"].replace("", np.nan))
            sig &= industry_leaders(gain.to_numpy(float), liquid, np.asarray(codes), self.mover, self.same)
        if self.earnings:
            sig &= self.earn.reindex(index=c.index, columns=t).fillna(False).astype(bool).to_numpy()
        self.signal = pd.DataFrame(sig, index=c.index, columns=t)
        self.gain = gain
        if self.kind == "ema":
            self.s5, self.s10 = ema(c, 5), ema(c, 10)
            self.exit_line = ema(c, self.exit_ma)
            self.limit = self.s5                                         # intraday EMA = price where price = yesterday's EMA
        else:
            self.s5, self.s10 = sma(c, 5), sma(c, 10)
            self.exit_line = sma(c, self.exit_ma)
            self.limit = c.rolling(4, min_periods=4).mean()              # price = SMA5 incl. today's price
        self.vol20 = v.rolling(20, min_periods=20).mean().shift(1)
        self.events = self._events(c.to_numpy(float), o.to_numpy(float), l.to_numpy(float), v.to_numpy(float),
                                   gain.to_numpy(float), sig)
        self.col = {tk: k for k, tk in enumerate(t)}
        self.pending: dict[str, dict] = {}
        self.meta: dict[str, dict] = {}

    def prepare(self, d: dict[str, pd.DataFrame], market: str = "SPY") -> "NamG1":
        self.info = d.get("INFO")
        self.earn = d.get("EarnOK") if self.earnings else None
        return super().prepare(d, market)

    def _events(self, c, o, l, v, gain, sig) -> dict:
        """For each signal (day s, stock k): its first touch day and the G4/G9 buy days."""
        lim = self.limit.to_numpy(float)
        s5 = self.s5.to_numpy(float)
        vol20 = self.vol20.to_numpy(float)
        n = len(c)
        out = {"touch": {}, "dry": {}, "loud": {}}
        for s, k in zip(*np.nonzero(sig)):
            end = min(s + self.wait, n - 1)
            touch = end + 1                                              # 'no touch in the window'
            for j in range(s + 1, end + 1):
                if l[j, k] <= lim[j - 1, k]:
                    touch = j
                    break
            out["touch"][(s, k)] = touch
            ref = s + 1 if s + 1 < n and gain[s + 1, k] >= self.big else s   # two big days: the second
            for j in range(ref + 1, min(ref + DRY_DAYS, n - 1) + 1):
                if c[j, k] < o[j, k] and gain[j, k] >= -DRY_DROP and v[j, k] <= DRY_VOL * v[ref, k]:
                    out["dry"][(s, k)] = j
                    break
            for j in range(s + LOUD_FROM, min(s + LOUD_TO, n - 1) + 1):
                if c[j, k] > o[j, k] and v[j, k] >= LOUD_X * v[j - 1, k] and v[j, k] >= vol20[j, k] \
                        and c[j, k] > s5[j, k]:
                    out["loud"][(s, k)] = j
                    break
        return out

    # --- entries -------------------------------------------------------------------------
    def candidates(self, i: int) -> list[str]:
        t = self.tradable
        picks: list[tuple[float, str, dict]] = []
        if self.entry == "signal":
            for k in np.flatnonzero(self.signal.iloc[i].to_numpy()):
                picks.append((self.gain.iat[i, k], t[k], {"s": i}))
        elif self.entry == "touch":
            for s in range(max(0, i - self.wait + 1), i + 1):
                for k in np.flatnonzero(self.signal.iloc[s].to_numpy()):
                    if i + 1 <= min(self.events["touch"][(s, k)], s + self.wait):
                        picks.append((self.gain.iat[s, k], t[k], {"s": s}))
        else:
            ev = self.events[self.entry]
            lo = DRY_DAYS + 1 if self.entry == "dry" else LOUD_TO
            for s in range(max(0, i - lo), i):
                for k in np.flatnonzero(self.signal.iloc[s].to_numpy()):
                    if ev.get((s, k)) == i:
                        picks.append((self.gain.iat[s, k], t[k], {"s": s}))
        picks.sort(key=lambda x: -x[0])
        out = []
        for _, tk, m in picks:
            if tk not in out:
                self.pending[tk] = m
                out.append(tk)
        return out

    def weight(self, t: str, i: int) -> float:
        return SIGNAL_W if self.entry == "signal" else TOUCH_W

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        if self.entry != "touch":
            return self.c[t].iloc[i - 1]                                 # bought at the signal day's close
        lim = self.limit[t].iloc[i - 1]
        if open_ <= lim:
            return open_
        return lim if self.l[t].iloc[i] <= lim else None

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        self.meta[t] = dict(self.pending[t])
        return True

    # --- adds ----------------------------------------------------------------------------
    def add_weight(self, pos: dict, i: int):
        t = pos["ticker"]
        m = self.meta[t]
        s = m["s"]
        if self.entry == "signal" and self.touch_buy and not m.get("touched"):
            touch = self.events["touch"][(s, self.col[t])]
            if self.dry_touch:                                           # G3: decided at the touch day's close
                if i == touch and touch <= s + self.wait:
                    m["touched"] = True
                    if self.v[t].iloc[i] <= self.vol20[t].iloc[i]:
                        return TOUCH_W
                return None
            if i + 1 <= min(touch, s + self.wait):
                if i + 1 == touch:
                    m["touched"] = True                                  # this limit is the first touch
                return TOUCH_W, self.limit[t].iloc[i]
            return None
        if not self.adds or pos["weight"] >= MAX_W - 1e-9:
            return None
        if self.entry == "signal" and self.touch_buy and pos.get("adds", 0) == 0:
            return None                                                  # adds only after the touch buy
        c = self.c[t]
        first = pos["entry_i"] - 1 if self.close_fill else pos["entry_i"]
        prev = c.iloc[first:i].max()
        if c.iloc[i] > prev and c.iloc[i] > self.s5[t].iloc[i] > self.s10[t].iloc[i] \
                and c.iloc[i] >= pos["entry_price"] * (1 + ADD_GAIN):
            return min(ADD_W, MAX_W - pos["weight"])
        return None

    # --- exits ---------------------------------------------------------------------------
    def exit_reason(self, pos: dict, i: int):
        t = pos["ticker"]
        m = self.meta[t]
        c = self.c[t].iloc[i]
        if c < self.l[t].iloc[m["s"]]:
            return "장대양봉 저가 이탈 (손절)"
        if c < self.exit_line[t].iloc[i]:
            return f"종가 {self.exit_ma}일선 이탈"
        if pos["high_close"] >= pos["entry_price"] * (1 + BREAKEVEN) and c < pos["entry_price"]:
            return "본절 컷"
        if i - pos["entry_i"] + 1 >= MAX_HOLD:
            return f"{MAX_HOLD}거래일 보유 완료"
        if self.half_at_5 and not m.get("half") and c < self.s5[t].iloc[i]:
            m["half"] = True
            return "종가 5일선 이탈 (절반)", 0.5
        return None


class NamG2(NamG1):
    name = "namsg_g2"
    label = "G2 G1 + 단체 상승 조건 없음"
    description = "G1과 같고, 같은 업종 단체 상승·1등주 조건을 뺍니다."
    group_rule = False


class NamG3(NamG1):
    name = "namsg_g3"
    label = "G3 G1 + 터치일 거래량 마름"
    description = ("G1과 같고, 첫 5일선 터치일 거래량이 20일 평균 이하일 때만 20%를 삽니다. 거래량은 장 마감에 "
                   "알 수 있어서 다음 날 시가에 삽니다.")
    dry_touch = True


class NamG4(NamG1):
    name = "namsg_g4"
    label = "G4 수급 1·2 음봉 종가 매수"
    description = ("장대양봉(이틀 연속이면 둘째 날) 다음 1~2거래일 안에 하락률 3% 이하·거래량 장대양봉 날의 50% 이하인 "
                   "음봉이 나오면 그날 종가에 20%. 이후 추가 매수와 매도는 G1과 같습니다.")
    entry = "dry"


class NamG5(NamG1):
    name = "namsg_g5"
    label = "G5 G1 + 추가 매수 없음"
    description = "G1과 같고, 첫 5일선 터치 뒤 추가 매수를 하지 않습니다."
    adds = False


class NamG6(NamG1):
    name = "namsg_g6"
    label = "G6 G1 + 5일선 이탈 절반 매도"
    description = "G1과 같고, 종가가 5일선 아래면 절반을 팔고 10일선 이탈 때 나머지를 팝니다."
    half_at_5 = True


class NamG7(NamG1):
    name = "namsg_g7"
    label = "G7 장대양봉 날 매수 없이 첫 5일선 터치 20% + 추가"
    description = "G1과 같은 신호 뒤 10거래일 안에 처음 5일선에 닿으면 그 가격에 20%를 삽니다. 장대양봉 날에는 사지 않습니다."
    entry = "touch"


class NamG8(NamG1):
    name = "namsg_g8"
    label = "G8 G1 + 직전 분기 매출·EPS 증가"
    description = "G1과 같고, 신호일에 공시된 최근 분기 매출과 EPS가 모두 1년 전보다 늘어난 회사만 삽니다."
    earnings = True


class NamG9(NamG1):
    name = "namsg_g9"
    label = "G9 다시 거래량 붙는 양봉 종가 20% + 추가"
    description = ("신호 뒤 2~10거래일 안에 처음 나온 '양봉, 거래량이 전날의 1.5배 이상이고 20일 평균 이상, 종가 > "
                   "5일선'인 날 종가에 20%. 이후 추가 매수와 매도는 G1과 같습니다.")
    entry = "loud"


for _cls in (NamG1, NamG2, NamG3, NamG4, NamG5, NamG6, NamG7, NamG8, NamG9):
    _cls.close_fill = _cls.entry != "touch"


def _sweep(base: type, param: str, value, text: str) -> type:
    """One number changed: the rulebook's sensitivity check."""
    tag = str(value).replace(".", "")
    return type(f"{base.__name__}_{param}_{tag}", (base,), {
        "name": f"{base.name}_{param}_{tag}", "label": f"{base.label[:2]} 민감도: {text}",
        "description": f"{base.label[:2]}과 같고 {text}만 바꿉니다.", param: value})


SWEEP = [_sweep(NamG1, "kind", "sma", "단순 이동평균"),
         _sweep(NamG1, "big", 0.07, "장대양봉 7%"), _sweep(NamG1, "big", 0.15, "장대양봉 15%"),
         _sweep(NamG1, "cum", True, "횡보 저점 대비 30% 이상"),
         _sweep(NamG1, "flat", 0.20, "횡보 폭 20%"), _sweep(NamG1, "flat", 0.40, "횡보 폭 40%"),
         _sweep(NamG1, "vol_x", 2.0, "거래량 2배"), _sweep(NamG1, "vol_x", 5.0, "거래량 5배"),
         _sweep(NamG1, "wait", 5, "터치 대기 5일"), _sweep(NamG1, "wait", 20, "터치 대기 20일"),
         _sweep(NamG1, "exit_ma", 5, "5일선 청산"), _sweep(NamG1, "exit_ma", 20, "20일선 청산"),
         _sweep(NamG1, "same", 2, "단체 상승 2종목"), _sweep(NamG1, "same", 4, "단체 상승 4종목")]
NAMSG = [NamG1, NamG2, NamG3, NamG4, NamG5, NamG6, NamG7, NamG8, NamG9, *SWEEP]
