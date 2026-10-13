"""부자회사원 '추세 종배' from a Korean YouTube video, rulebook 09.

A stock that moved sideways makes its first clean +10% candle just above the
200-day line. A few days later, a short red candle on drying volume that holds
just above a rising 5-day line is bought at the close. The next morning a gap
down or a dip to the buy price sells a third at a time; once in profit the
stop moves to the buy price. Profits run until the close breaks the 5-day line
(half) and the 10-day line (the rest). Every signal stock gets 3% of the
account.

All numbers are from /mnt/project-files/rulebooks/09-부자회사원-추세종배-200일선첫장대양봉.md
([원전] said in the video, [해석] our numbers, [근사] daily stand-in for
minute-bar handling, [추가] added), fixed before the first run. Moving
averages are exponential (the video's chart), simple as a variant.

Not automated [미구현]: buyer flows (foreign/institution), news, the first
3-minute bar (T11 needs minute data), the 'pick the opportunity' judgment in
the sizing rule (approximated by the month's first signal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import ema, sma
from .strategies import Strategy

MIN_DOLLAR_VOL = 20e6       # [해석]
MIN_PRICE = 10.0            # [해석]
MEMBERS = (500, 400)        # S&P 500 + 400 [해석]
FLAT, FLAT_DAYS = 0.30, 60  # S1 [해석]
NEAR50, NEAR50_DAYS = 0.05, 20  # T5 [해석]
BIG = 0.10                  # S2 [해석]
NEAR200 = 0.15              # S2 'just above' the 200-day line [해석]
VOL_X = 3.0                 # S3 [해석]
WAIT = (2, 10)              # B1 [해석]
BODY = 0.03                 # B1 [해석]
ABOVE, BELOW = 0.99, 1.03   # B1 low >= MA5 x 0.99, close <= MA5 x 1.03 [해석]
DRY = 0.50                  # B1 [해석]
BAD_UP = 0.03               # B4 [해석]
DIP = 0.03                  # L3 screen '-3%' [원전]
HOLD = 120                  # [추가]
W = 0.03                    # W1 [원전 2~3%]
RE_RANGE, RE_DAYS, RE_LOOK = 0.15, 10, 120  # T12 [해석]
BIG_CAP, BIG_DONE = 1.0, 0.33   # W2 cap, W4 goal [해석]
DEFAULT_LOSS = 0.05         # W2 average loss before any loss happened [원전 예시 -5%]

_SHARED = None


class Trend(Strategy):
    """T1: the video's way."""
    name = "trend_t1"
    label = "T1 추세 종배 (200일선 위 첫 장대양봉 뒤 마른 음봉 종가)"
    description = ""
    max_positions = 33
    close_fill = True
    kind = "ema"
    big, vol_x, near200, flat, dry = BIG, VOL_X, NEAR200, FLAT, DRY
    buy_ma, half_ma, final_ma = 5, 5, 10
    buy_at_signal = False       # T2
    near50 = False              # T5
    earnings = False            # T6
    sizing = False              # T7
    reentry = True              # T8: False
    morning = "both"            # L2/L3: "both", "even" (buy price only), "dip" (-3% only), "" (T9)
    breakeven = True            # L4 (T9: False)
    open_even = False           # T13
    red_vs_prev = False         # T10
    second_signals = False      # T12
    w = W

    def prepare(self, d: dict[str, pd.DataFrame], market: str = "SPY") -> "Trend":
        self.info = d.get("INFO")
        self.earn = d.get("EarnOK") if self.earnings else None
        return super().prepare(d, market)

    # ------------------------------------------------------------------ setup
    def _base(self) -> dict:
        """Arrays shared by every variant of one data set (the variants would not fit in memory)."""
        global _SHARED
        key = (self.kind,)
        if _SHARED is not None and _SHARED[0] is self.c and _SHARED[1] is self.info and _SHARED[2] == key:
            return _SHARED[3]
        t = self.tradable
        c, o, h, l, v = (x[t] for x in (self.c, self.o, self.h, self.l, self.v))
        info = self.info if self.info is not None else pd.DataFrame(columns=["index"])
        member = info.reindex(t)["index"].isin(MEMBERS).to_numpy()
        value = c * v
        liquid = ((value.rolling(20, min_periods=20).mean() >= MIN_DOLLAR_VOL) & (c >= MIN_PRICE)).to_numpy()
        avg = ema if self.kind == "ema" else sma
        b = {"C": c.to_numpy(float), "O": o.to_numpy(float), "H": h.to_numpy(float), "L": l.to_numpy(float),
             "V": v.to_numpy(float), "liquid": liquid & member[None, :],
             "M200": avg(c, 200).shift(1).to_numpy(float),
             "avg50": v.rolling(50, min_periods=50).mean().shift(1).to_numpy(float),
             "hi": h.rolling(FLAT_DAYS, min_periods=FLAT_DAYS).max().shift(1).to_numpy(float),
             "lo": l.rolling(FLAT_DAYS, min_periods=FLAT_DAYS).min().shift(1).to_numpy(float),
             "lines": {n: avg(c, n).to_numpy(float) for n in (3, 5, 7, 10, 20, 50)}}
        g = np.full_like(b["C"], np.nan)
        g[1:] = b["C"][1:] / b["C"][:-1] - 1
        b["G"] = g
        ma50 = avg(c, 50)
        near = ((c / ma50 - 1).abs() <= NEAR50).astype(float)
        b["near50_ok"] = (near.rolling(NEAR50_DAYS, min_periods=NEAR50_DAYS).sum().shift(1) == NEAR50_DAYS).to_numpy()
        _SHARED = (self.c, self.info, key, b)
        return b

    def _prepare(self) -> None:
        b = self._base()
        self.__dict__.update({k: v for k, v in b.items() if k != "lines"})
        self.lines = b["lines"]
        self.col = {tk: k for k, tk in enumerate(self.tradable)}
        G, C, O, V = self.G, self.C, self.O, self.V
        earlier = pd.DataFrame(G >= self.big).astype(float).rolling(FLAT_DAYS, min_periods=1).sum().shift(1).to_numpy()
        big = (G >= self.big) & (V >= self.vol_x * self.avg50) & self.liquid
        first = big & (earlier == 0) & (self.hi / self.lo - 1 <= self.flat) \
            & (O >= self.M200) & (O <= self.M200 * (1 + self.near200))
        if self.near50:
            first &= self.near50_ok
        if self.earnings:
            first &= self.earn.reindex(index=self.c.index, columns=self.tradable).fillna(False) \
                .astype(bool).to_numpy()
        sig = first
        if self.second_signals:
            hi = pd.DataFrame(self.H).rolling(RE_DAYS, min_periods=RE_DAYS).max().shift(1).to_numpy()
            lo = pd.DataFrame(self.L).rolling(RE_DAYS, min_periods=RE_DAYS).min().shift(1).to_numpy()
            recent = pd.DataFrame(first).astype(float).rolling(RE_LOOK, min_periods=1).sum().shift(1).to_numpy() > 0
            sig = first | (big & recent & (C > self.M200) & (hi / lo - 1 <= RE_RANGE))
        self.orders: dict[int, list] = {}
        n = len(C)
        for s, k in zip(*np.nonzero(sig)):
            spec = {"s": s, "score": G[s, k]}
            if self.buy_at_signal:
                self.orders.setdefault(s, []).append((k, spec))
                continue
            for j in self._buy_days(s, k, n):
                self.orders.setdefault(j, []).append((k, spec))
        self.begin()

    def _buy_days(self, s: int, k: int, n: int):
        C, O, L, V = self.C[:, k], self.O[:, k], self.L[:, k], self.V[:, k]
        m = self.lines[self.buy_ma][:, k]
        for j in range(s + 1, min(s + WAIT[1], n - 1) + 1):
            if V[j] > V[s] and C[j] / C[j - 1] - 1 < BAD_UP:
                return                                               # B4: a bigger-volume day that failed
            if j < s + WAIT[0]:
                continue
            red = C[j] < C[j - 1] if self.red_vs_prev else C[j] < O[j]
            if not (red and abs(O[j] - C[j]) / O[j] <= BODY and L[j] >= m[j] * ABOVE and C[j] <= m[j] * BELOW
                    and V[j] <= self.dry * V[s] and V[j] < V[j - 1] and m[j] > m[j - 1] and self.liquid[j, k]):
                continue
            ch = C[s:j + 1] / C[s - 1:j] - 1
            vol = V[s + 1:j + 1]
            up, down = vol[ch[1:] > 0], vol[ch[1:] <= 0]
            if len(up) and len(down) and up.mean() <= down.mean():
                continue                                             # B3: volume rises on down days
            yield j

    def begin(self) -> None:
        self.pending: dict[str, dict] = {}
        self.meta: dict[str, dict] = {}
        self.used: set = set()
        self.month_pnl: dict = {}
        self.losses: list[float] = []
        self.big_month, self.big_w, self.big_gain = None, 0.0, None

    # ------------------------------------------------------------------ entries
    def candidates(self, i: int) -> list[str]:
        picks = sorted(self.orders.get(i, []), key=lambda x: -x[1]["score"])
        out = []
        for k, spec in picks:
            t = self.tradable[k]
            if (not self.reentry and (spec["s"], k) in self.used) or t in out:
                continue
            self.pending[t] = spec
            out.append(t)
        return out

    @property
    def partial_ok(self) -> bool:
        return self.sizing

    def weight(self, t: str, i: int) -> float:
        if not self.sizing:
            return self.w
        month = self.c.index[i].to_period("M")
        if self.big_month != month:                                  # W2: last month's profit / average loss
            self.big_month = month
            prev = self.month_pnl.get(month - 1, 0.0)
            loss = np.mean(self.losses) if self.losses else DEFAULT_LOSS
            self.big_w = min(BIG_CAP, prev / loss) if prev > 0 else 0.0
        return max(self.w, self.big_w)

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        return self.c[t].iloc[i - 1]                                 # the signal day's close

    def bought_before_open(self, t: str) -> bool:
        return True

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        spec = self.pending[t]
        self.used.add((spec["s"], self.col[t]))
        big = self.sizing and self.big_w > self.w
        self.meta[t] = {**spec, "big": big}
        if big:
            self.big_w = 0.0                                         # one big bet a month
        return True

    def on_close(self, trade: dict, i: int) -> None:
        if not self.sizing:
            return
        month = self.c.index[i].to_period("M")
        self.month_pnl[month] = self.month_pnl.get(month, 0.0) + trade["return"] * trade["weight"] * trade["part"]
        if trade["return"] < 0:
            self.losses.append(-trade["return"])

    # ------------------------------------------------------------------ exits
    def _k(self, pos: dict) -> int:
        return self.col[pos["ticker"]]

    def initial_stop(self, t: str, i: int, price: float) -> float | None:
        return None

    def update_stop(self, pos: dict, i: int) -> float | None:
        if self.breakeven and i >= pos["entry_i"] and self.C[i, self._k(pos)] > pos["entry_price"]:
            return pos["entry_price"]                                # L4: in profit -> stop at the buy price
        return pos.get("stop")

    def open_exit(self, pos: dict, i: int):
        if not self.morning or self.open_even or i != pos["entry_i"]:
            return None
        if self.O[i, self._k(pos)] < pos["entry_price"]:
            pos["gap_sold"] = True
            return "다음 날 갭 하락 (1/3)", 1 / 3                       # L2
        return None

    def partial_stops(self, pos: dict, i: int) -> list:
        if i != pos["entry_i"]:
            return []
        buy = pos["entry_price"]
        if self.open_even:                                           # T13
            if self.O[i, self._k(pos)] >= buy:
                return [(buy, 1.0, "본절 (다음 날)")]
            return []
        left = pos["shares"] / pos["initial_shares"]
        out = []
        if self.morning in ("both", "even") and not pos.get("gap_sold"):
            out.append((buy, (1 / 3) / left, "다음 날 본절 (1/3)"))
            left -= 1 / 3
        if self.morning in ("both", "dip"):
            out.append((buy * (1 - DIP), (1 / 3) / left, "다음 날 -3% (1/3)"))
        return out

    def exit_reason(self, pos: dict, i: int):
        k = self._k(pos)
        c, buy = self.C[i, k], pos["entry_price"]
        half, final = self.lines[self.half_ma][i, k], self.lines[max(self.final_ma, self.half_ma)][i, k]
        if c < buy and c < half:
            return "손실 중 종가 5일선 이탈"                                # L1
        if c < final:
            return f"종가 {max(self.final_ma, self.half_ma)}일선 이탈"     # P2
        if c < half and not pos.get("halved"):
            pos["halved"] = True
            return f"수익 중 종가 {self.half_ma}일선 이탈 (절반)", 0.5
        if i - pos["entry_i"] + 2 >= HOLD:
            return f"{HOLD}거래일 보유 완료"
        return None


def _make(name: str, label: str, description: str, **kw) -> type:
    return type(name, (Trend,), {"name": name.lower(), "label": label, "description": description, **kw})


T1 = _make("Trend_T1", "T1 추세 종배 (200일선 위 첫 장대양봉 → 마른 음봉 종가)",
           "60거래일 30% 안 횡보, 직전 60일 +10% 날 없음, 시가가 200일 지수이동평균 이상·1.15배 이하에서 +10%·거래량 "
           "50일 평균 3배. 그 뒤 2~10거래일 안에 5일선 우상향 속 몸통 3% 이하 음봉(저가 ≥ 5일선×0.99, 종가 ≤ 5일선×1.03, "
           "거래량 ≤ 장대양봉 날 50%·전날보다 적음, 오른 날 거래량 > 내린 날)이면 종가에 3%. 그사이 장대양봉보다 큰 거래량에 "
           "+3% 못 간 날이 있으면 취소. 다음 날 갭 하락이면 시가 1/3, 매수가·-3%에 1/3씩, 종가가 매수가 위면 본절 손절. "
           "손실 중 종가 5일선 이탈 전량, 수익 중 5일선 이탈 절반·10일선 이탈 나머지. 청산 뒤 같은 신호로 재진입. 최대 120일.")
T2 = _make("Trend_T2", "T2 장대양봉 날 종가 매수", "T1과 같고, 마른 음봉을 기다리지 않고 장대양봉 날 종가에 삽니다.",
           buy_at_signal=True)
T3 = _make("Trend_T3", "T3 익절 기준 20일선", "T1과 같고, 나머지를 10일선 대신 종가 20일선 이탈에 팝니다.", final_ma=20)
T4 = _make("Trend_T4", "T4 익절 기준 50일선", "T1과 같고, 나머지를 종가 50일선 이탈에 팝니다.", final_ma=50)
T5 = _make("Trend_T5", "T5 + 50일선 근처 횡보", "T1과 같고, 직전 20거래일 종가가 모두 50일선 ±5% 안이어야 합니다.",
           near50=True)
T6 = _make("Trend_T6", "T6 + 직전 분기 매출·EPS 증가",
           "T1과 같고, 장대양봉 날 공시된 최근 분기 매출과 EPS가 모두 1년 전보다 늘었어야 합니다(SEC 공시, 영업이익 대신 EPS).",
           earnings=True)
T7 = _make("Trend_T7", "T7 + 시장이 정하는 비중 (지난달 수익 ÷ 평균 손실률)",
           "T1과 같고, 지난달 실현 수익이 플러스면 이번 달 첫 신호에 (지난달 수익 ÷ 평균 손실률)만큼(최대 계좌 100%, 남은 "
           "현금까지) 겁니다. 나머지는 3%.", sizing=True)
T8 = _make("Trend_T8", "T8 재진입 없음", "T1과 같고, 한 신호로 한 번만 삽니다.", reentry=False)
T9 = _make("Trend_T9", "T9 아침 분할 매도·본절 없이 5·10일선만",
           "T1과 같고, 다음 날 아침 분할 매도와 본절 손절 없이 5·10일선 규칙만 씁니다.", morning="", breakeven=False)
T10 = _make("Trend_T10", "T10 음봉 = 전날보다 낮은 종가", "T1과 같고, 음봉을 '종가 < 전날 종가'로 봅니다(시가 무관).",
            red_vs_prev=True)
T12 = _make("Trend_T12", "T12 + 두 번째 장대양봉 재진입 신호",
            "T1과 같고, 최근 120거래일 안에 신호가 있던 종목은 10거래일 15% 안에서 쉬다 나온 +10%·거래량 3배 날을 "
            "200일선 거리·'첫' 조건 없이(종가 > 200일선만) 새 신호로 봅니다.", second_signals=True)
T13 = _make("Trend_T13", "T13 다음 날 시가가 매수가 이상이면 그날부터 본절",
            "T1과 같고, 다음 날 아침 1/3씩 파는 대신 시가가 매수가 이상이면 그날부터 매수가에 전량 손절을 겁니다.",
            open_even=True)


def _sweep(base: type, text: str, **kw) -> type:
    tag = "_".join(f"{k}{str(v).replace('.', '')}" for k, v in kw.items())
    return type(f"{base.__name__}_{tag}", (base,), {
        "name": f"{base.name}_{tag}", "label": f"{base.label.split()[0]} 민감도: {text}",
        "description": f"{base.label.split()[0]}과 같고 {text}만 바꿉니다.", **kw})


SWEEP = [_sweep(T1, "단순 이동평균", kind="sma"),
         _sweep(T1, "아침 본절만", morning="even"), _sweep(T1, "아침 -3%만", morning="dip"),
         _sweep(T1, "장대양봉 7%", big=0.07), _sweep(T1, "장대양봉 15%", big=0.15), _sweep(T1, "장대양봉 20%", big=0.20),
         _sweep(T1, "거래량 5배", vol_x=5.0), _sweep(T1, "거래량 10배", vol_x=10.0),
         _sweep(T1, "200일선 범위 25%", near200=0.25), _sweep(T1, "200일선 범위 40%", near200=0.40),
         _sweep(T1, "횡보 폭 20%", flat=0.20), _sweep(T1, "횡보 폭 40%", flat=0.40),
         _sweep(T1, "매수·절반 이평 7일", buy_ma=7, half_ma=7), _sweep(T1, "매수·절반 이평 10일", buy_ma=10, half_ma=10),
         _sweep(T1, "절반 매도 3일선", half_ma=3),
         _sweep(T1, "마른 거래량 30%", dry=0.30), _sweep(T1, "마른 거래량 70%", dry=0.70),
         _sweep(T1, "종목당 1.5%", w=0.015, max_positions=66), _sweep(T1, "종목당 5%", w=0.05, max_positions=20)]
TREND = [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T12, T13, *SWEEP]
