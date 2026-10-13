"""직장인 종가 배팅 모음 from a Korean YouTube video (부자회사원), rulebook 08.

Five independent closing-price techniques, each tested on its own and then
together in one account:

A  서희파더: the leader of an industry that jumps together (+15%, Korea's
   'limit up') is bought on the pullback with limit orders at -5% and -10%
   from its close; half is sold at +3% over the average cost, the rest at +8%;
   stop at -15%.
B  강창권: after a +10% candle and a 4-10 day quiet pause, buy the close of
   the up day where volume comes back above the 5-day line; sell half below
   the 5-day line, the rest below the 10-day line.
C  남석관 '상한가 5일선': after a +15% day following a flat 60/120-day line,
   buy a clean up close near the 5-day line 3-5 days later; out the next day.
D  남석관 'N형': a 20% jump after a flat stretch, buy the close 3-7 days later
   when the 10-day line holds; out within 5 days.
E  남석관 '양음양': a +15% day, then a big red candle that closes above the
   5-day line: buy that close, sell the next day.

All numbers are from /mnt/project-files/rulebooks/08-종가배팅-모음-서희파더-강창권-남석관.md
([원전] said in the video, [해석] our numbers, [근사] automated stand-in,
[추가] added), fixed before the first run. Moving averages are simple.

Not automated [미구현]: news, 'the issue is over', 'the leader changed',
buyer flows, the after-hours price, the Korean 30% daily limit itself
(stood in for by +15%), technique M's long-term swing (recorded only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import sma
from .strategies import Strategy

MIN_DOLLAR_VOL = 20e6       # [추가]
MIN_PRICE = 10.0            # [추가]
MEMBERS = (500, 400)        # S&P 500 + 400 [해석]
VOL_X = 3.0                 # volume vs its 50-day average on the big day [해석]
MOVER, SAME = 0.05, 3       # A1 industry jumps together [해석]
LIMIT_UP = 0.15             # stand-in for the Korean +30% limit [해석]
A_RUNNER = 0.70             # A2: the runner-up gains at most 70% of the leader [해석]
A_RUN_AWAY = 0.09           # A3 [원전]
A_FIRST, A_STEP = 0.05, 0.05   # 1.2 [원전 -5%, 4~6% steps]
A_TP1, A_TP2 = 0.03, 0.08   # 1.3 [원전]
A_WAIT, A_HOLD = 10, 10     # [해석], [추가]
A_W = 0.10                  # each buy [원전]
B_BIG = 0.10                # B1 [해석]
B_PAUSE = (4, 10)           # B2 [원전 4~5일 이상, 해석]
B_DRY = 0.35                # B2 [해석]
B_LOUD = 1.5                # B3 [해석]
B_REACH = 30                # how long after the big day a pause can end [해석]
B_BIG_WIN = 0.50            # 'big gain' half [원전 50~60%]
B_STOP = 0.05               # [추가]
B_HOLD = 60                 # [추가]
C_FLAT, C_FLAT_DAYS = 0.03, 20  # C1 [해석]
C_FROM, C_TO = 3, 5         # C2 [원전 3~5일]
C_NEAR, C_WICK = 0.01, 0.10 # C2, C3 [해석]
DAY_TP, DAY_STOP = 0.03, 0.03   # C [원전 -3%, 해석 +3%]
D_RANGE, D_RANGE_DAYS = 0.30, 60  # D1 [해석]
D_JUMP = 0.20               # D1 [원전]
D_FROM, D_TO = 3, 7         # D2 [원전]
D_NEAR = 0.02               # D2 [해석]
D_VALID = 10                # D4 [근사]
D3_TO = 5                   # D-3 [해석]
SHORT_STOP = 0.05           # D/E [원전 -3~5%]
GAP_HALF = 0.03             # 'gap up early: take half' [해석]
SHORT_HOLD = 5              # [원전 1주일]
E_RED = 0.05                # E2 [해석]
E_PAUSE = 0.03              # E-2 short candle [해석]
E_GAP, E_SECTOR = 0.02, 0.03   # E-3, E-4 [해석]
E4_HOLD = 20                # [추가]
STREAK, BREAK = 3, 10       # 6장: 3 losses in a row -> 10 days off [해석]
W = 0.20                    # B-E single-technique weight [해석]


def _roll_max(x: np.ndarray, n: int) -> np.ndarray:
    return pd.DataFrame(x).rolling(n, min_periods=n).max().to_numpy()


def leaders(gain: np.ndarray, liquid: np.ndarray, group: np.ndarray, mover: float = MOVER,
            same: int = SAME) -> tuple[np.ndarray, np.ndarray]:
    """Per day: True for the top gainer of an industry with `same`+ stocks up `mover`+,
    and the runner-up's gain in that industry (NaN elsewhere)."""
    n, k = gain.shape
    lead = np.zeros((n, k), bool)
    second = np.full((n, k), np.nan)
    for i in range(n):
        up = np.flatnonzero(liquid[i] & (gain[i] >= mover) & (group >= 0))
        if len(up) < same:
            continue
        for g in np.unique(group[up]):
            m = up[group[up] == g]
            if len(m) >= same:
                order = m[np.argsort(-gain[i, m])]
                lead[i, order[0]] = True
                second[i, order[0]] = gain[i, order[1]]
    return lead, second


class Jongga(Strategy):
    """One account trading any of the techniques in `techs`."""
    name = "jongga"
    techs: tuple[str, ...] = ()
    max_positions = 5
    close_fill = True
    same_day_adds = True
    # A
    a_group, a_levels, a_stop_mode, a_space_by_runner, a_above20 = True, 2, "level", False, False
    a_first, a_tp2 = A_FIRST, A_TP2
    limit_up, vol_x = LIMIT_UP, VOL_X
    # B
    b_pause, b_dry = B_PAUSE, B_DRY
    # D
    d_mode, d_jump, short_hold, short_stop = "book", D_JUMP, SHORT_HOLD, SHORT_STOP
    # E
    e_mode = "e1"
    # M
    streak_rule = False
    bear_only = ()              # techniques traded only while the S&P 500 is below its 200-day line
    w_single = W

    # ------------------------------------------------------------------ setup
    def prepare(self, d: dict[str, pd.DataFrame], market: str = "SPY") -> "Jongga":
        self.info = d.get("INFO")
        return super().prepare(d, market)

    def _prepare(self) -> None:
        t = self.tradable
        self.col = {tk: k for k, tk in enumerate(t)}
        c, o, h, l, v = (x[t] for x in (self.c, self.o, self.h, self.l, self.v))
        self.C, self.O, self.H, self.L, self.V = (x.to_numpy(float) for x in (c, o, h, l, v))
        info = self.info if self.info is not None else pd.DataFrame(columns=["index", "sub_industry"])
        info = info.reindex(t)
        member = info["index"].isin(MEMBERS).to_numpy()
        value = c * v
        liquid = ((value.rolling(20, min_periods=20).mean() >= MIN_DOLLAR_VOL) & (c >= MIN_PRICE)).to_numpy()
        self.liquid = liquid & member[None, :]
        self.group = np.asarray(pd.factorize(info["sub_industry"].replace("", np.nan))[0])
        g = np.full_like(self.C, np.nan)
        g[1:] = self.C[1:] / self.C[:-1] - 1
        self.G = g
        self.avg50 = v.rolling(50, min_periods=50).mean().shift(1).to_numpy()
        self.vol5 = v.rolling(5, min_periods=5).mean().shift(1).to_numpy()
        self.vol20 = v.rolling(20, min_periods=20).mean().shift(1).to_numpy()
        self.S5, self.S10, self.S20 = (sma(c, n).to_numpy() for n in (5, 10, 20))
        self.loud = (self.V >= self.vol_x * self.avg50) & self.liquid
        self.bull = self.market_ok.reindex(c.index).to_numpy(bool)
        self.orders: dict[int, list] = {}
        for tech in self.techs:
            getattr(self, f"_plan_{tech}")()
        self.pending: dict[str, dict] = {}
        self.meta: dict[str, dict] = {}
        self.used: set = set()
        self.streak, self.pause_until = 0, -1

    def _add(self, day: int, k: int, spec: dict) -> None:
        """An order placed at day `day`'s close for the next day."""
        self.orders.setdefault(day, []).append((k, spec))

    # A: 서희파더 ----------------------------------------------------------------
    def _plan_a(self) -> None:
        G, n = self.G, len(self.C)
        big = (G >= self.limit_up) & self.loud
        if self.a_group:
            lead, second = leaders(G, self.liquid, self.group, MOVER, SAME)
            sig = big & lead & ~(second > A_RUNNER * G)
        else:
            sig, second = big, np.full_like(G, np.nan)
        for s, k in zip(*np.nonzero(sig)):
            if self.a_above20 and sig[s].sum() > 1 and not self.C[s, k] > self.S20[s, k]:
                continue
            L = self.C[s, k]
            step = A_STEP
            if self.a_space_by_runner:
                step = 0.04 if second[s, k] >= 0.10 else 0.06
            levels = [L * (1 - self.a_first - j * step) for j in range(3)]
            spec = {"tech": "a", "s": s, "L": L, "levels": levels[:self.a_levels] if self.a_stop_mode == "level"
                    else levels, "stop": levels[2], "score": G[s, k]}
            for day in range(s, min(s + A_WAIT, n - 1)):
                if day > s and self.H[s + 1:day + 1, k].max() >= L * (1 + A_RUN_AWAY):
                    break                                           # A3: ran away before the pullback
                self._add(day, k, spec)

    # B: 강창권 -----------------------------------------------------------------
    def _plan_b(self) -> None:
        C, O, L, V, n = self.C, self.O, self.L, self.V, len(self.C)
        lo, hi = self.b_pause
        sig = (self.G >= B_BIG) & self.loud
        for s, k in zip(*np.nonzero(sig)):
            for j in range(s + lo + 1, min(s + B_REACH, n - 1) + 1):
                p = s + int(np.argmax(C[s:j, k]))                    # the high close so far
                quiet = j - p - 1
                if not lo <= quiet <= hi:
                    continue
                if V[p + 1:j, k].mean() > self.b_dry * V[s, k]:
                    continue
                if not (C[j, k] > O[j, k] and C[j, k] > self.S5[j, k]
                        and (C[j - 1, k] <= self.S5[j - 1, k] or L[j, k] <= self.S5[j, k])
                        and V[j, k] >= B_LOUD * V[j - 1, k] and V[j, k] >= self.vol5[j, k]):
                    continue
                if self.liquid[j, k]:
                    self._add(j, k, {"tech": "b", "s": s, "score": self.G[s, k]})

    # C: 상한가 5일선 -------------------------------------------------------------
    def _plan_c(self) -> None:
        c = self.c[self.tradable]
        flat = ((sma(c, 60) / sma(c, 120) - 1).abs() <= C_FLAT).astype(float)
        flat = (flat.rolling(C_FLAT_DAYS, min_periods=C_FLAT_DAYS).sum().shift(1) == C_FLAT_DAYS).to_numpy()
        sig = (self.G >= self.limit_up) & self.loud & flat
        C, O, H, L, n = self.C, self.O, self.H, self.L, len(self.C)
        for s, k in zip(*np.nonzero(sig)):
            for j in range(s + C_FROM, min(s + C_TO, n - 1) + 1):
                rng = H[j, k] - L[j, k]
                if L[j, k] <= self.S5[j, k] * (1 + C_NEAR) and C[j, k] > O[j, k] \
                        and rng > 0 and H[j, k] - C[j, k] <= C_WICK * rng and self.liquid[j, k]:
                    self._add(j, k, {"tech": "c", "s": s, "score": self.G[s, k]})
                    break

    # D: N형 -------------------------------------------------------------------
    def _plan_d(self) -> None:
        C, O, H, L, V, n = self.C, self.O, self.H, self.L, self.V, len(self.C)
        rng = _roll_max(H, D_RANGE_DAYS) / -_roll_max(-L, D_RANGE_DAYS) - 1        # ending each day
        done = np.full(C.shape[1], -1)
        for r in range(D_RANGE_DAYS + 2, n):
            for b in (r, r - 1, r - 2):                               # a 1-3 day jump ending on day r
                ok = (C[r] / C[b - 1] - 1 >= self.d_jump) & (rng[b - 1] <= D_RANGE) \
                    & (V[b:r + 1].max(axis=0) >= VOL_X * self.avg50[b]) & self.liquid[r] & (done < b)
                for k in np.flatnonzero(ok):
                    done[k] = r
                    self._plan_d_one(r, k)

    def _plan_d_one(self, r: int, k: int) -> None:
        C, O, L, V, n = self.C, self.O, self.L, self.V, len(self.C)
        spec = {"tech": "d", "s": r, "score": C[r, k] / C[r - 1, k] - 1}
        if self.d_mode == "chart":                                   # D-3
            for j in range(r + 1, min(r + D3_TO, n - 1) + 1):
                if C[j, k] > O[j, k] and V[j, k] >= B_LOUD * V[j - 1, k] and C[j, k] > self.S5[j, k]:
                    self._add(j, k, spec)
                    return
            return
        for j in range(r + 1, min(r + D_VALID, n - 1) + 1):
            p = r + int(np.argmax(C[r:j, k]))
            if not D_FROM <= j - p <= D_TO:
                continue
            if (C[p:j, k] < self.S10[p:j, k]).any():
                return                                               # D3: the 10-day line broke
            if L[j, k] <= self.S10[j, k] * (1 + D_NEAR) and C[j, k] >= self.S10[j, k]:
                if self.d_mode == "loud" and not (C[j, k] > O[j, k] and V[j, k] >= B_LOUD * V[j - 1, k]):
                    continue
                self._add(j, k, spec)
                return

    # E: 양음양 ----------------------------------------------------------------
    def _plan_e(self) -> None:
        C, O, n = self.C, self.O, len(self.C)
        sig = (self.G >= self.limit_up) & self.loud
        for s, k in zip(*np.nonzero(sig)):
            j = s + 1
            if j < n - 1 and C[j, k] < O[j, k] and (O[j, k] - C[j, k]) / O[j, k] >= E_RED \
                    and C[j, k] > self.S5[j, k]:
                self._add(j, k, {"tech": "e", "s": s, "score": self.G[s, k]})
        if self.e_mode == "e3":
            hot = (self.G >= E_SECTOR) & self.liquid
            self.sector_hot = np.zeros_like(hot)
            for i in range(len(C)):
                idx = np.flatnonzero(hot[i] & (self.group >= 0))
                if len(idx):
                    g, cnt = np.unique(self.group[idx], return_counts=True)
                    busy = set(g[cnt >= 2])
                    self.sector_hot[i] = np.isin(self.group, list(busy))

    # ------------------------------------------------------------------ entries
    def candidates(self, i: int) -> list[str]:
        if self.streak_rule and i < self.pause_until:
            return []
        picks = []
        for k, spec in self.orders.get(i, []):
            if spec["tech"] in self.bear_only and self.bull[i]:
                continue
            key = (spec["tech"], spec["s"], k)
            if key in self.used:
                continue
            picks.append((spec["score"], k, spec))
        picks.sort(key=lambda x: -x[0])
        out = []
        for _, k, spec in picks:
            tk = self.tradable[k]
            if tk not in out:
                self.pending[tk] = spec
                out.append(tk)
        return out

    @property
    def order_all(self) -> bool:
        return "a" in self.techs                                    # limit orders: most never fill

    def weight(self, t: str, i: int) -> float:
        return A_W if self.pending[t]["tech"] == "a" else self.w_single

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        spec = self.pending[t]
        if spec["tech"] != "a":
            return self.c[t].iloc[i - 1]                              # bought at the signal day's close
        lim = spec["levels"][0]
        if self.a_stop_mode == "level" and open_ <= spec["stop"]:
            return None                                             # opened below the stop: skip
        if open_ <= lim:
            return open_
        return lim if self.l[t].iloc[i] <= lim else None

    def bought_before_open(self, t: str) -> bool:
        return self.pending[t]["tech"] != "a"

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        spec = self.pending[t]
        self.used.add((spec["tech"], spec["s"], self.col[t]))
        self.meta[t] = {**spec, "buy_i": i}
        return True

    def on_close(self, trade: dict, i: int) -> None:
        if not self.streak_rule or trade["part"] < 0.999 and trade["return"] >= 0:
            return
        if trade["return"] < 0:
            self.streak += 1
            if self.streak >= STREAK:
                self.pause_until, self.streak = i + 1 + BREAK, 0
        else:
            self.streak = 0

    # ------------------------------------------------------------------ helpers
    def _m(self, pos: dict) -> dict:
        return self.meta[pos["ticker"]]

    @staticmethod
    def _halved(pos: dict) -> bool:
        return pos["shares"] < pos["initial_shares"] * 0.999

    def _k(self, pos: dict) -> int:
        return self.col[pos["ticker"]]

    # ------------------------------------------------------------------ adds
    def add_weight(self, pos: dict, i: int):
        m = self._m(pos)
        if m["tech"] != "a" or self._halved(pos):
            return None
        n_done = pos.get("adds", 0) + 1
        if n_done >= len(m["levels"]) or i + 1 > m["s"] + A_WAIT:
            return None
        return A_W, m["levels"][n_done]

    def close_add(self, pos: dict, i: int) -> float | None:
        m, k = self._m(pos), self._k(pos)
        if m["tech"] == "e" and self.e_mode in ("e2", "e3", "e4") and i == m["s"] + 2 \
                and abs(self.C[i, k] - self.O[i, k]) / self.O[i, k] <= E_PAUSE and self.C[i, k] > self.S5[i, k]:
            m["paused"] = True
            return self.w_single
        return None

    # ------------------------------------------------------------------ exits
    def initial_stop(self, t: str, i: int, price: float) -> float | None:
        spec = self.pending[t]
        if spec["tech"] == "a" and self.a_stop_mode == "level":
            return spec["stop"]
        if spec["tech"] == "c":
            return self.c[t].iloc[i - 1] * (1 - DAY_STOP)
        if spec["tech"] == "e":
            return self.c[t].iloc[i - 1] * (1 - SHORT_STOP)
        return None

    def update_stop(self, pos: dict, i: int) -> float | None:
        if self._m(pos)["tech"] == "e":
            return pos["entry_price"] * (1 - SHORT_STOP)              # follows the average cost
        return pos.get("stop")

    def take_profit(self, pos: dict, i: int):
        m, k, e = self._m(pos), self._k(pos), pos["entry_i"]
        avg, op = pos["entry_price"], self.O[i, k]
        tech = m["tech"]
        if tech == "a":
            if i == e and not pos["before_open"]:
                return None                                         # bought during the day: order unknown
            if not self._halved(pos):
                return avg * (1 + A_TP1), 0.5, "1차 익절 (평균 단가 +3%)"
            return avg * (1 + self.a_tp2), 1.0, f"최종 익절 (+{self.a_tp2 * 100:.0f}%)"
        if tech == "b":
            if not m.get("big_half") and self.H[i, k] >= avg * (1 + B_BIG_WIN):
                m["big_half"] = True
                return avg * (1 + B_BIG_WIN), 0.5, "+50% 절반 익절"
            return None
        if tech == "c" and i == e:
            if op >= avg * (1 + GAP_HALF):
                return avg * (1 + GAP_HALF), 0.5, "갭 상승 절반 익절"
            return avg * (1 + DAY_TP), 1.0, "다음 날 +3% 익절"
        if tech not in ("d", "e") or self._halved(pos):
            return None
        if tech == "e" and self.e_mode == "e4":                      # +2% over the previous close
            return (op, 0.5, "갭 상승 절반 익절") if op >= self.C[i - 1, k] * (1 + E_GAP) else None
        return (op, 0.5, "갭 상승 절반 익절") if op >= avg * (1 + GAP_HALF) else None

    def open_exit(self, pos: dict, i: int) -> str | None:
        m, k = self._m(pos), self._k(pos)
        if m["tech"] == "e" and self.e_mode == "e3" and m.get("paused") and i == m["s"] + 3:
            gap = self.O[i, k] >= self.C[i - 1, k] * (1 + E_GAP) and self.sector_hot[i - 1, k]
            return None if gap else "멈춤 다음 날 시가 매도"
        return None

    def close_exit(self, pos: dict, i: int) -> str | None:
        m, k, e = self._m(pos), self._k(pos), pos["entry_i"]
        tech = m["tech"]
        if tech == "c" and i == e:
            return "다음 날 종가 매도"
        if tech == "d" and i - e + 2 >= self.short_hold:
            return f"{self.short_hold}거래일째 종가 매도"
        if tech == "e":
            s = m["s"]
            if self.e_mode == "e1" and i == s + 2:
                return "다음 날 종가 매도"
            if self.e_mode in ("e2", "e3") and i == s + 2:
                body = abs(self.C[i, k] - self.O[i, k]) / self.O[i, k]
                if not (body <= E_PAUSE and self.C[i, k] > self.S5[i, k]):
                    return "다음 날 종가 매도"
            if self.e_mode == "e3" and m.get("paused") and i == s + 3:
                return "갭 상승 날 종가 매도"
            if self.e_mode == "e4" and i >= s + 2:
                body = abs(self.C[i, k] - self.O[i, k])
                wick = self.H[i, k] - max(self.C[i, k], self.O[i, k])
                if self.C[i, k] < self.S5[i, k]:
                    return "종가 5일선 이탈"
                if self.V[i, k] >= VOL_X * self.vol20[i, k] and wick >= 2 * body:
                    return "대량 거래량 윗꼬리"
                if i - e + 2 >= E4_HOLD:
                    return f"{E4_HOLD}거래일 보유 완료"
        return None

    def exit_reason(self, pos: dict, i: int):
        m, k, e = self._m(pos), self._k(pos), pos["entry_i"]
        c, avg = self.C[i, k], pos["entry_price"]
        tech = m["tech"]
        if tech == "a":
            if self.a_stop_mode == "ma" and (c < self.S20[i, k] or c < self.O[m["s"], k]):
                return "종가 20일선·장대양봉 시가 이탈"
            if i - e + 1 >= A_HOLD:
                return f"{A_HOLD}거래일 보유 완료"
            return None
        if tech == "b":
            if c < avg * (1 - B_STOP):
                return "종가 -5% 손절"
            if m.get("big_half"):
                return "+50% 뒤 종가 20일선 이탈" if c < self.S20[i, k] else None
            if c < self.S10[i, k]:
                return "종가 10일선 이탈"
            if c < self.S5[i, k] and not self._halved(pos):
                return "종가 5일선 이탈 (절반)", 0.5
            if i - e + 2 >= B_HOLD:
                return f"{B_HOLD}거래일 보유 완료"
            return None
        if tech == "d":
            if c < avg * (1 - self.short_stop):
                return f"종가 -{self.short_stop * 100:.0f}% 손절"
            return None
        if tech == "e" and self.e_mode == "e2" and m.get("paused") and i == m["s"] + 2:
            return "멈춤 다음 날 시가 매도"
        return None


def _make(name: str, label: str, description: str, **kw) -> type:
    return type(name, (Jongga,), {"name": name.lower(), "label": label, "description": description, **kw})


A1 = _make("Jongga_A1", "A-1 서희파더 단체 상승 대장주 -5%·-10% 지정가, +3% 절반·+8%",
           "같은 세부 업종 3종목 이상 +5%인 날 그 업종 1등(+15% 이상, 거래량 3배, 2등은 1등 상승률의 70% 이하)을 "
           "종가 L 기준 L×0.95, L×0.90 지정가로 10%씩(10거래일 대기, 그 전에 L×1.09를 넘으면 취소). 평균 단가 "
           "+3%에 절반, +8%에 나머지, 저가가 L×0.85면 손절, 최대 10거래일. 최대 3종목.",
           techs=("a",), max_positions=3)
A2 = _make("Jongga_A2", "A-2 A-1 + 단체 상승 조건 없음", "A-1과 같고, 단체 상승·대장주 조건 없이 +15%·거래량 3배면 됩니다.",
           techs=("a",), max_positions=3, a_group=False)
A3 = _make("Jongga_A3", "A-3 A-1 + 3분할(L×0.85까지), 20일선·시가 손절",
           "A-1과 같고 L×0.85에 3차까지 사며, 손절은 종가 < 20일선 또는 < 장대양봉 시가(다음 날 시가 매도)입니다.",
           techs=("a",), max_positions=3, a_stop_mode="ma")
A4 = _make("Jongga_A4", "A-4 A-1 + 2등주 강도로 간격 4%/6%",
           "A-1과 같고, 업종 2등이 +10% 이상이면 지정가 간격 4%, 아니면 6%입니다(손절은 3차 자리).",
           techs=("a",), max_positions=3, a_space_by_runner=True)
A5 = _make("Jongga_A5", "A-5 A-1 + 후보가 여럿이면 20일선 위만",
           "A-1과 같고, 같은 날 후보가 둘 이상이면 종가 > 20일선인 종목만 삽니다.",
           techs=("a",), max_positions=3, a_above20=True)
B1 = _make("Jongga_B1", "B-1 강창권 기간 조정 후 거래량 붙는 양봉 종가",
           "+10%·거래량 3배 장대양봉 뒤 고점 이후 4~10거래일 쉬며 평균 거래량이 장대양봉 날의 35% 이하, 그 뒤 "
           "거래량이 전날 1.5배·5일 평균 이상인 양봉으로 5일선 위에 올라서면 종가 20%. 종가 -5% 손절, 5일선 이탈 "
           "절반·10일선 이탈 나머지, +50%면 절반 익절 후 20일선 이탈까지.", techs=("b",))
B2 = _make("Jongga_B2", "B-2 B-1 + 조정 2~3일", "B-1과 같고, 쉬는 기간을 2~3거래일로 봅니다.",
           techs=("b",), b_pause=(2, 3))
C1 = _make("Jongga_C1", "C-1 남석관 상한가 5일선 (다음 날 정리)",
           "60일선이 20일 내내 120일선 ±3% 안이던 종목의 +15%·거래량 3배 뒤 3~5거래일째, 저가가 5일선 +1% 안에 "
           "닿고 윗꼬리 없는(10% 이하) 양봉이면 종가 20%. 다음 날 +3% 익절(갭 +3%면 시가 절반), -3% 손절, 아니면 종가 매도.",
           techs=("c",))
D1 = _make("Jongga_D1", "D-1 남석관 N형 (10일선 지지 종가)",
           "60거래일 30% 안 횡보 뒤 1~3일에 +20%·거래량 3배, 고점 3~7거래일 뒤 저가가 10일선 +2% 안에 닿고 종가가 "
           "10일선 위면 종가 20%(그사이 종가 10일선 이탈 없음, 10거래일 안). 종가 -5% 손절, 시가 +3% 갭이면 절반, "
           "5거래일째 종가 매도.", techs=("d",))
D2 = _make("Jongga_D2", "D-2 D-1 + 거래량 붙는 양봉일 때만", "D-1과 같고, 그날이 양봉이고 거래량이 전날의 1.5배 이상일 때만 삽니다.",
           techs=("d",), d_mode="loud")
D3 = _make("Jongga_D3", "D-3 N형 차트 모양 (급등 뒤 1~5일 거래량 양봉)",
           "D-1과 같은 급등 뒤 1~5거래일 안에 양봉·거래량 전날 1.5배·종가 > 5일선인 날 종가에 삽니다. 매도는 D-1과 같습니다.",
           techs=("d",), d_mode="chart")
E1 = _make("Jongga_E1", "E-1 남석관 양음양 (음봉 종가, 다음 날 정리)",
           "+15%·거래량 3배 다음 날 -5% 이상 음봉이 5일선 위에서 끝나면 종가 20%. 다음 날 시가 +3% 갭이면 절반, "
           "저가 -5% 손절, 아니면 그날 종가 매도.", techs=("e",))
E2 = _make("Jongga_E2", "E-2 E-1 + 멈춤 캔들 2차 매수",
           "E-1과 같고, 다음 날이 몸통 3% 이하·5일선 위 짧은 캔들이면 그 종가에 20% 더 사고 그다음 날 시가에 팝니다.",
           techs=("e",), e_mode="e2")
E3 = _make("Jongga_E3", "E-3 E-2 + 갭 상승·업종 강세면 그날 종가까지",
           "E-2와 같고, 멈춤 다음 날 시가가 +2% 갭이고 전날 같은 업종 2종목 이상 +3%면 시가에 팔지 않고 그날 종가에 "
           "팝니다(3차 매수는 하지 않음).", techs=("e",), e_mode="e3")
E4 = _make("Jongga_E4", "E-4 양음양 YG플러스 방식 (5일선 이탈까지)",
           "E-2처럼 사고, 다음 날 시가가 전날 종가 +2% 이상이면 절반, 나머지는 종가 < 5일선 또는 거래량 20일 평균 3배 + "
           "윗꼬리가 몸통 2배인 날 종가에 팝니다. 저가 -5% 손절, 최대 20거래일.", techs=("e",), e_mode="e4")
M1 = _make("Jongga_M1", "M-1 A-1·B-1·C-1·D-1·E-1 한 계좌",
           "A-1, B-1, C-1, D-1, E-1을 한 계좌(최대 5종목)에서 함께 하고, 손실 거래가 3번 연속이면 10거래일 쉽니다.",
           techs=("a", "b", "c", "d", "e"), streak_rule=True)
M2 = _make("Jongga_M2", "M-2 M-1 + C·D·E는 S&P500 200일선 아래서만",
           "M-1과 같고, 하락장 기법 C·D·E는 S&P500이 200일선 아래일 때만 삽니다.",
           techs=("a", "b", "c", "d", "e"), streak_rule=True, bear_only=("c", "d", "e"))


def _sweep(base: type, text: str, **kw) -> type:
    tag = "_".join(f"{k}{str(v).replace('.', '')}" for k, v in kw.items())
    return type(f"{base.__name__}_{tag}", (base,), {
        "name": f"{base.name}_{tag}", "label": f"{base.label.split()[0]} 민감도: {text}",
        "description": f"{base.label.split()[0]}과 같고 {text}만 바꿉니다.", **kw})


SWEEP = [_sweep(A1, "상한가 대용 10%", limit_up=0.10), _sweep(A1, "상한가 대용 20%", limit_up=0.20),
         _sweep(A1, "거래량 2배", vol_x=2.0), _sweep(A1, "거래량 5배", vol_x=5.0),
         _sweep(A1, "1차 매수 -3%", a_first=0.03), _sweep(A1, "1차 매수 -7%", a_first=0.07),
         _sweep(A1, "목표 +5%", a_tp2=0.05), _sweep(A1, "목표 +12%", a_tp2=0.12),
         _sweep(B1, "조정 거래량 25%", b_dry=0.25), _sweep(B1, "조정 거래량 50%", b_dry=0.50),
         _sweep(D1, "N형 급등 15%", d_jump=0.15), _sweep(D1, "N형 급등 30%", d_jump=0.30),
         _sweep(D1, "손절 -3%", short_stop=0.03),
         _sweep(D1, "최대 보유 3일", short_hold=3), _sweep(D1, "최대 보유 10일", short_hold=10),
         _sweep(E1, "상한가 대용 10%", limit_up=0.10), _sweep(E1, "상한가 대용 20%", limit_up=0.20)]
JONGGA = [A1, A2, A3, A4, A5, B1, B2, C1, D1, D2, D3, E1, E2, E3, E4, M1, M2, *SWEEP]
