"""세븐 스플릿 + 엔벨로프 하단 분할 매수 from a Korean YouTube video (부자회사원), rulebook 10.

E  성경호 envelope bottom: a large-cap stock touching the lower envelope line
   (20-day average x 0.80) is bought with 40% of its budget, then 30% and 30%
   more at -7% steps. Half is sold at +7% over the average cost, the rest at
   +14%; the stop is 5% below the third buy.
F  박성현 Seven Split on top: the same first touch starts seven separate lots
   of 2% of the account each. Lot 1 is the long-term lot (+10% or a year);
   lots 2-7 are bought at -3/-5/-7/-10/-20/-40% from the first price and each
   is sold the morning after it closes in profit, then bought again when the
   price comes back to its level. No stop loss.

Lots are kept apart (never averaged), so this module runs its own day loop
instead of swing/engine.py; the timeline is the same: limit orders fill when
the day's low reaches them (at the open if it opens below), close decisions
fill at the next open, nothing bought on a day is sold the same day.

All numbers are from /mnt/project-files/rulebooks/10-세븐스플릿-엔벨로프하단-분할매수.md
([원전] said in the video, [해석] our numbers, [근사] stand-in, [추가] added),
fixed before the first run.

Not automated [미구현]: news / default rumours (B7), the value screen F6
(PBR, ROE), the W-pattern judgment (E2 uses break-even instead), the small-cap
method (recorded only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Result
from .indicators import ema, sma
from .strategies import Strategy

MIN_DOLLAR_VOL = 20e6       # [해석]
MIN_PRICE = 10.0            # [해석]
MEMBERS = (500, 400)        # S&P 500 + 400 [근사]
ENV_N, ENV_PCT = 20, 0.20   # B2 [화면]
NEAR = 0.03                 # B12 [화면]
BULL_NEAR = 0.02            # B11 [원전 1~2%]
PASS_DROP = 0.10            # E6 [해석]
E_BUDGET, E_MAX = 0.10, 10  # [해석]
E_PARTS = (0.4, 0.3, 0.3)   # B3-B5 [원전]
E_STEP = 0.07               # B4 [원전 7~10%]
E_STOP = 0.05               # B6 [원전]
BIG_RED_VOL, BIG_RED_BODY = 3.0, 0.05   # E2 [해석]
LOT = 0.02                  # A3 [원전 예시]
LOT1_MAX, LOT_MAX = 20, 5   # A2/A3 [해석]
STEPS_CHANNEL = (0.03, 0.05, 0.07, 0.10, 0.20, 0.40)   # A6 [원전 21:37]
STEPS_AUTHOR = (0.03, 0.05, 0.10, 0.20, 0.40, 0.70)    # A6 [원전 25:26]
LONG_TP, LONG_DAYS = 0.10, 252  # A9 [원전]
F_STOP = 0.05               # F8 [해석]
GONE = 10                   # trading days without a price -> treated as delisted [추가]
_SHARED = None


class Split(Strategy):
    """E1: 성경호 original."""
    name = "split_e1"
    mode = "E"
    env_pct, env_kind = ENV_PCT, "sma"
    entry = "touch"             # "touch", "close" (E3), "bull" (E4)
    pass_drop = False           # E6
    # E
    e_step, tp1, tp2 = E_STEP, 0.07, 0.14
    e_rest = "target"           # "target" (E1) or "even" (E2: break-even stop + big red candle)
    stop = True                 # E5: False
    # F
    steps = STEPS_CHANNEL
    short_tp = 0.0              # F3: 0.03 limit
    lot1_short = False          # F4
    flexible = False            # F5
    bear_start = False          # F7
    f_stop = ""                 # F8 "always", F9 "bear"
    lot = LOT
    long_tp = LONG_TP
    near = NEAR

    def prepare(self, d: dict[str, pd.DataFrame], market: str = "SPY") -> "Split":
        self.info = d.get("INFO")
        self.member_daily = d.get("MEMBER")
        return super().prepare(d, market)

    def _prepare(self) -> None:
        """Arrays are shared by every variant of one data set (30 variants would not fit in memory)."""
        global _SHARED
        if _SHARED is None or _SHARED[0] is not self.c or _SHARED[1] is not self.info \
                or _SHARED[2] is not self.member_daily:
            _SHARED = (self.c, self.info, self.member_daily, self._base(), {})
        base, lines = _SHARED[3], _SHARED[4]
        self.__dict__.update(base)
        key = (self.env_pct, self.env_kind)
        if key not in lines:
            avg = (sma if self.env_kind == "sma" else ema)(self.c[self.tradable], ENV_N) * (1 - self.env_pct)
            lines[key] = (avg.shift(1).to_numpy(float), avg.to_numpy(float))   # yesterday's line [근사], today's
        self.lower_prev, self.lower_now = lines[key]

    def _base(self) -> dict:
        t = self.tradable
        c, o, h, l, v = (x[t] for x in (self.c, self.o, self.h, self.l, self.v))
        b = {k: x.to_numpy(float) for k, x in zip("COHLV", (c, o, h, l, v))}
        value = (c * v).rolling(20, min_periods=20).mean().shift(1)
        ok = ((value >= MIN_DOLLAR_VOL) & (c.shift(1) >= MIN_PRICE)).to_numpy()
        if self.member_daily is not None:                      # point-in-time index membership
            member = self.member_daily.reindex(index=c.index, columns=t).fillna(False).astype(bool).to_numpy()
        else:
            info = self.info if self.info is not None else pd.DataFrame(columns=["index"])
            member = np.broadcast_to(info.reindex(t)["index"].isin(MEMBERS).to_numpy()[None, :], ok.shape)
        b["ok"] = ok & member
        b["avg50"] = v.rolling(50, min_periods=50).mean().shift(1).to_numpy(float)
        b["bull"] = self.market_ok.reindex(c.index).fillna(False).to_numpy(bool)   # S&P 500 above its 200-day line
        prev = np.full_like(b["C"], np.nan)
        prev[1:] = b["C"][:-1]
        b["P"] = prev
        return b

    # ------------------------------------------------------------------ the day loop
    def simulate(self, cost_bps: float, start: str | None = None) -> Result:
        dates = self.c.index
        first = 0 if start is None else int(dates.searchsorted(pd.Timestamp(start)))
        cost = cost_bps / 10_000
        C, O, H, L = self.C, self.O, self.H, self.L
        n, nk = C.shape
        last = pd.DataFrame(C).ffill().to_numpy()
        seen = np.where(~np.isnan(C), np.arange(n)[:, None], -1)
        seen = np.maximum.accumulate(seen, axis=0)                 # last day with a price
        self.cash, eq_prev = 1.0, 1.0
        self.trades, equity, exposure = [], [], []
        self.book: dict[int, dict] = {}                            # k -> holding
        pending: dict[int, str] = {}                               # k -> reason to sell at the next open
        for i in range(first, n):
            self.i, self.cost, self.sold_today = i, cost, set()
            # 1. open: sells decided yesterday
            for k, (which, reason) in list(pending.items()):
                if k not in self.book:
                    continue
                px = O[i, k] if not np.isnan(O[i, k]) else last[i, k]
                if self.mode == "E":
                    self._sell_e(k, 1.0, px, reason)
                else:
                    b = self.book[k]
                    lots = list(b["lots"]) if which == "all" else [lt for lt in b["lots"] if any(lt is w for w in which)]
                    self._sell_lots(k, lots, px, reason)
            pending = {}
            # delisted or halted: sell at the last price
            for k in list(self.book):
                if i - seen[i, k] > GONE:
                    if self.mode == "F":
                        self._sell_lots(k, list(self.book[k]["lots"]), last[i, k], "거래 중단 (상장폐지 추정)")
                    else:
                        self._sell_e(k, 1.0, last[i, k], "거래 중단 (상장폐지 추정)")
            # 2. during the day: profit targets, then limit buys
            for k in list(self.book):
                if np.isnan(H[i, k]):
                    continue
                self._targets(k, i)
            for k in list(self.book):
                if not np.isnan(L[i, k]):
                    self._adds(k, i, eq_prev)
            self._entries(i, eq_prev)
            # 3. close: value the account, decide tomorrow's sells
            for k in list(self.book):
                if not np.isnan(C[i, k]):
                    r = self._close_rules(k, i)
                    if r:
                        pending[k] = r
            value = self.cash + sum(sum(lt["shares"] for lt in b["lots"]) * last[i, k] for k, b in self.book.items())
            equity.append(value)
            exposure.append(1 - self.cash / value if value > 0 else 0.0)
            eq_prev = value
        eq = pd.Series(equity, index=dates[first:first + len(equity)], name=self.name)
        cols = ["ticker", "entry_date", "entry_price", "exit_date", "exit_price", "return", "days",
                "reason", "part", "weight"]
        open_pos = [{"ticker": self.tradable[k], "entry_date": dates[min(lt["i"] for lt in b["lots"])].date(),
                     "entry_price": b["lots"][0]["price"], "stop": None} for k, b in self.book.items()]
        return Result(self.name, eq, pd.DataFrame(self.trades, columns=cols), pd.Series(exposure, index=eq.index),
                      open_pos)

    # ------------------------------------------------------------------ helpers
    def _buy(self, k: int, i: int, px: float, w: float, eq_prev: float, n_lot) -> bool:
        spend = eq_prev * w
        if spend <= 0 or self.cash < spend * (1 + self.cost) or not px > 0:
            return False                                          # no cash left: the money is tied up
        self.cash -= spend * (1 + self.cost)
        b = self.book.setdefault(k, {"lots": [], "first": px, "start": i, "halved": False})
        b["lots"].append({"n": n_lot, "price": px, "shares": spend / px, "i": i, "w": spend / eq_prev})
        return True

    def _record(self, k: int, entry: float, i_in: int, px: float, reason: str, part: float, w: float) -> None:
        dates = self.c.index
        self.trades.append({"ticker": self.tradable[k], "entry_date": dates[i_in].date(), "entry_price": entry,
                            "exit_date": dates[self.i].date(), "exit_price": px,
                            "return": px * (1 - self.cost) / (entry * (1 + self.cost)) - 1,
                            "days": self.i - i_in + 1, "reason": reason, "part": part, "weight": w})

    def _sell_lots(self, k: int, lots: list, px: float, reason: str) -> None:
        b = self.book[k]
        for lt in lots:
            self.cash += lt["shares"] * px * (1 - self.cost)
            self._record(k, lt["price"], lt["i"], px, reason, 1.0, lt["w"])
            b["lots"].remove(lt)
        if not b["lots"]:
            del self.book[k]
            self.sold_today.add(k)

    def _avg(self, b: dict) -> tuple[float, float]:
        sh = sum(lt["shares"] for lt in b["lots"])
        return sum(lt["shares"] * lt["price"] for lt in b["lots"]) / sh, sh

    def _sell_e(self, k: int, frac: float, px: float, reason: str) -> None:
        """E keeps one averaged position: sell `frac` of what is left."""
        b = self.book[k]
        avg, sh = self._avg(b)
        start = min(lt["i"] for lt in b["lots"])
        total = b.setdefault("bought", sh)
        w = sum(lt["w"] for lt in b["lots"]) if "w_all" not in b else b["w_all"]
        b["w_all"] = w
        sold = sh * frac
        self.cash += sold * px * (1 - self.cost)
        self._record(k, avg, start, px, reason, sold / total, w)
        for lt in b["lots"]:
            lt["shares"] *= 1 - frac
        if frac >= 0.999:
            del self.book[k]
            self.sold_today.add(k)
        else:
            b["halved"] = True

    def _fill_below(self, k: int, i: int, level: float) -> float | None:
        """Price a buy limit at `level` gets today, or None."""
        if np.isnan(level) or np.isnan(self.L[i, k]) or self.L[i, k] > level:
            return None
        return min(self.O[i, k], level)

    def _fill_above(self, k: int, i: int, level: float) -> float | None:
        if self.H[i, k] < level:
            return None
        return max(self.O[i, k], level)

    # ------------------------------------------------------------------ entries
    def _entries(self, i: int, eq_prev: float) -> None:
        if self.mode == "F" and self.bear_start and self.bull[i - 1 if i else 0]:
            return
        held = len(self.book)
        if self.mode == "E" and held >= E_MAX:
            return
        if self.mode == "F" and sum(1 for b in self.book.values() if any(lt["n"] == 1 for lt in b["lots"])) >= LOT1_MAX:
            return
        if self.entry == "close":
            near = np.abs(self.C[i] / self.lower_now[i] - 1) <= self.near
            ks = np.flatnonzero(near & self.ok[i])
            fills = {k: self.C[i, k] for k in ks}
        else:
            line = self.lower_prev[i] * ((1 + BULL_NEAR) if self.entry == "bull" and self.bull[max(i - 1, 0)] else 1)
            touch = (self.L[i] <= line) & self.ok[i]
            fills = {k: min(self.O[i, k], line[k]) for k in np.flatnonzero(touch)}
            if self.pass_drop:
                fills = {k: p for k, p in fills.items() if p > self.P[i, k] * (1 - PASS_DROP)}
        order = sorted((k for k in fills if k not in self.book and k not in self.sold_today),   # [추가]
                       key=lambda k: np.nan_to_num(fills[k] / self.P[i, k], nan=1.0))   # deepest drop first
        for k in order:
            if self.mode == "E":
                if len(self.book) >= E_MAX:
                    break
                if self._buy(k, i, fills[k], E_BUDGET * E_PARTS[0], eq_prev, 1):
                    self.book[k]["e_last"] = fills[k]
            else:
                n1 = sum(1 for b in self.book.values() if any(lt["n"] == 1 for lt in b["lots"]))
                if n1 >= LOT1_MAX:
                    break
                if self._buy(k, i, fills[k], self.lot, eq_prev, 1):
                    self.book[k]["steps"] = self._steps(i)

    def _steps(self, i: int) -> tuple:
        if self.flexible and not self.bull[max(i - 1, 0)]:
            return tuple(s for s in self.steps if s >= 0.10)       # F5: start the adds at -10% in a bear market
        return self.steps

    def _adds(self, k: int, i: int, eq_prev: float) -> None:
        b = self.book[k]
        if self.mode == "E":
            n = len(b["lots"])
            if n >= len(E_PARTS) or b["halved"]:
                return
            px = self._fill_below(k, i, b["e_last"] * (1 - self.e_step))
            if px is not None and self._buy(k, i, px, E_BUDGET * E_PARTS[n], eq_prev, n + 1):
                b["e_last"] = px
            return
        have = {lt["n"] for lt in b["lots"]}
        for j, step in enumerate(b["steps"], start=2):
            if j in have:
                continue
            count = sum(1 for bb in self.book.values() for lt in bb["lots"] if lt["n"] == j)
            if count >= LOT_MAX:
                continue
            px = self._fill_below(k, i, b["first"] * (1 - step))
            if px is not None:
                self._buy(k, i, px, self.lot, eq_prev, j)

    # ------------------------------------------------------------------ exits
    def _targets(self, k: int, i: int) -> None:
        b = self.book[k]
        if self.mode == "E":
            if all(lt["i"] == i for lt in b["lots"]):
                return
            avg, _ = self._avg(b)
            if not b["halved"]:
                px = self._fill_above(k, i, avg * (1 + self.tp1))
                if px is not None:
                    self._sell_e(k, 0.5, px, f"+{self.tp1 * 100:.0f}% 절반 익절")
                    return
            elif self.e_rest == "target":
                px = self._fill_above(k, i, b.get("avg_half", avg) * (1 + self.tp2))
                if px is not None:
                    self._sell_e(k, 1.0, px, f"+{self.tp2 * 100:.0f}% 나머지 익절")
            elif self.e_rest == "even":
                px = self._fill_below(k, i, b.get("avg_half", avg))
                if px is not None:
                    self._sell_e(k, 1.0, px, "나머지 본절")
            return
        sell = []
        for lt in b["lots"]:
            if lt["i"] == i:
                continue
            if lt["n"] == 1 and not self.lot1_short:
                px = self._fill_above(k, i, lt["price"] * (1 + self.long_tp))
                if px is not None:
                    sell.append((lt, px, f"1번 묶음 +{self.long_tp * 100:.0f}%"))
            elif self.short_tp > 0:
                px = self._fill_above(k, i, lt["price"] * (1 + self.short_tp))
                if px is not None:
                    sell.append((lt, px, f"{lt['n']}번 묶음 +{self.short_tp * 100:.0f}%"))
        for lt, px, reason in sell:
            if k in self.book:
                self._sell_lots(k, [lt], px, reason)

    def _close_rules(self, k: int, i: int):
        """(which lots, reason) to sell at tomorrow's open; sells at today's close happen here."""
        b, c = self.book[k], self.C[i, k]
        if self.mode == "E":
            avg, _ = self._avg(b)
            if b["halved"] and "avg_half" not in b:
                b["avg_half"] = avg
            if self.stop and len(b["lots"]) == len(E_PARTS) and c <= b["e_last"] * (1 - E_STOP):
                return ("all", "3차 매수가 -5% 손절")
            if self.e_rest == "even" and b["halved"] and self.V[i, k] >= BIG_RED_VOL * self.avg50[i, k] \
                    and (c - self.O[i, k]) / self.O[i, k] <= -BIG_RED_BODY:
                return ("all", "대량 거래량 장대음봉")
            return None
        if self.f_stop and any(lt["n"] == 7 for lt in b["lots"]):
            lt7 = next(lt for lt in b["lots"] if lt["n"] == 7)
            if c <= lt7["price"] * (1 - F_STOP) and (self.f_stop == "always" or not self.bull[i]):
                return ("all", "7번 묶음 -5% 손절")
        for lt in list(b["lots"]):
            if lt["n"] == 1 and not self.lot1_short:
                if i - lt["i"] >= LONG_DAYS and c > lt["price"]:
                    self._sell_lots(k, [lt], c, "1번 묶음 1년 보유 뒤 수익 (종가)")
                    if k not in self.book:
                        return None
        if self.short_tp > 0:
            return None
        lots = [lt for lt in b["lots"] if (lt["n"] > 1 or self.lot1_short) and c > lt["price"]]
        if not lots:
            return None
        return (lots, "단기 묶음 수익 (다음 날 시가)")


def _make(name: str, label: str, description: str, **kw) -> type:
    return type(name, (Split,), {"name": name.lower(), "label": label, "description": description, **kw})


E1 = _make("Split_E1", "E1 성경호 엔벨로프 하단 40·30·30% (+7% 절반, +14%, 3차 -5% 손절)",
           "S&P 500·400 종목의 저가가 엔벨로프(20일 단순 이동평균 × 0.8, 전날 값) 하단선에 닿으면 종목 예산(계좌 10%)의 40%, "
           "직전 매수가 -7%마다 30%·30% 지정가. 평균 단가 +7%에 절반, +14%에 나머지. 종가가 3차 매수가 -5% 이하면 다음 날 "
           "시가 전량. 최대 10종목.")
E2 = _make("Split_E2", "E2 채널 익절 (+10% 절반, 나머지 본절·대량 장대음봉)",
           "E1과 같고, +10%에 절반, 나머지는 평균 단가(본절)에 닿거나 거래량 50일 평균 3배·-5% 이상 음봉이 나오면 다음 날 시가에 팝니다.",
           tp1=0.10, e_rest="even")
E3 = _make("Split_E3", "E3 조건검색: 종가가 하단선 ±3% 안이면 종가 매수",
           "E1과 같고, 첫 매수를 '종가가 그날 하단선의 ±3% 안'일 때 종가에 합니다.", entry="close")
E4 = _make("Split_E4", "E4 상승장엔 하단선 2% 위에서도 매수",
           "E1과 같고, S&P500이 200일선 위인 날에는 하단선 × 1.02에 닿으면 삽니다.", entry="bull")
E5 = _make("Split_E5", "E5 손절 없음", "E1과 같고, 3차 매수가 -5% 손절을 하지 않습니다.", stop=False)
E6 = _make("Split_E6", "E6 급락 패스 (하단선 터치 가격이 전날 종가 -10% 이하면 안 삼)",
           "E1과 같고, 하단선에 닿은 가격이 전날 종가보다 10% 넘게 낮으면 사지 않습니다.", pass_drop=True)
F1 = _make("Split_F1", "F1 세븐 스플릿 + 엔벨로프 (채널 단계 -3/-5/-7/-10/-20/-40%)",
           "하단선에 닿으면 1번 묶음(계좌 2%), 첫 매수가 대비 -3/-5/-7/-10/-20/-40%에 2~7번 묶음(각 2%) 지정가. "
           "묶음은 평균 내지 않고 따로 팝니다. 2~7번은 종가가 그 묶음 매수가보다 높으면 다음 날 시가, 다시 그 자리에 오면 또 삼. "
           "1번 묶음은 +10% 지정가, 또는 1년 뒤 수익이면 종가 매도(손실이면 계속 보유). 손절 없음. 1번 묶음 최대 20종목, "
           "2~7번은 묶음마다 최대 5종목.", mode="F")
F2 = _make("Split_F2", "F2 저자 단계 -3/-5/-10/-20/-40/-70%", "F1과 같고, 추가 매수 단계를 저자 방식으로 합니다.",
           mode="F", steps=STEPS_AUTHOR)
F3 = _make("Split_F3", "F3 단기 묶음 +3% 지정가 익절", "F1과 같고, 2~7번 묶음을 매수가 +3% 지정가에 팝니다.",
           mode="F", short_tp=0.03)
F4 = _make("Split_F4", "F4 스윙만 (1번 묶음도 수익 나면 다음 날 매도)",
           "F1과 같고, 1번 묶음도 단기 묶음처럼 종가 수익이면 다음 날 시가에 팝니다.", mode="F", lot1_short=True)
F5 = _make("Split_F5", "F5 유연 추가 (하락장엔 -10%부터)",
           "F1과 같고, S&P500이 200일선 아래일 때 시작한 종목은 추가 매수를 -10/-20/-40%에서만 합니다.",
           mode="F", flexible=True)
F7 = _make("Split_F7", "F7 하락장 전용 (S&P500 200일선 아래서만 시작)", "F1과 같고, S&P500이 200일선 아래일 때만 새로 시작합니다.",
           mode="F", bear_start=True)
F8 = _make("Split_F8", "F8 손절 추가 (7번 묶음 매수가 -5%)", "F1과 같고, 7번 묶음까지 산 뒤 종가가 7번 매수가 -5% 이하면 전부 팝니다.",
           mode="F", f_stop="always")
F9 = _make("Split_F9", "F9 나쁜 장에서만 손절", "F8과 같고, S&P500이 200일선 아래일 때만 손절합니다.", mode="F", f_stop="bear")


def _sweep(base: type, text: str, **kw) -> type:
    tag = "_".join(f"{k}{str(v).replace('.', '')}" for k, v in kw.items() if k != "steps") or "steps"
    return type(f"{base.__name__}_{tag}", (base,), {
        "name": f"{base.name}_{tag}", "label": f"{base.label.split()[0]} 민감도: {text}",
        "description": f"{base.label.split()[0]}과 같고 {text}만 바꿉니다.", **kw})


SWEEP = [_sweep(E1, "엔벨로프 10%", env_pct=0.10), _sweep(E1, "엔벨로프 15%", env_pct=0.15),
         _sweep(E1, "엔벨로프 25%", env_pct=0.25), _sweep(E1, "지수 이동평균", env_kind="ema"),
         _sweep(E1, "추가 매수 간격 8.5%", e_step=0.085), _sweep(E1, "추가 매수 간격 10%", e_step=0.10),
         _sweep(E1, "분할 익절 +5%", tp1=0.05, tp2=0.10), _sweep(E1, "분할 익절 +10%", tp1=0.10, tp2=0.20),
         _sweep(E3, "근접 1%", near=0.01),
         _sweep(F1, "엔벨로프 10%", env_pct=0.10), _sweep(F1, "엔벨로프 15%", env_pct=0.15),
         _sweep(F1, "5%마다 균등 단계", steps=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30)),
         _sweep(F1, "단기 묶음 +5%", short_tp=0.05), _sweep(F1, "1번 묶음 목표 20%", long_tp=0.20),
         _sweep(F1, "묶음 1%", lot=0.01), _sweep(F1, "묶음 3%", lot=0.03)]
SPLIT = [E1, E2, E3, E4, E5, E6, F1, F2, F3, F4, F5, F7, F8, F9, *SWEEP]
