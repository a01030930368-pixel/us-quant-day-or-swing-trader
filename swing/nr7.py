"""Rulebook 20: Toby Crabel's NR7 / inside day, bought on a break above the signal day's high (swing version).

NR7: today's range is smaller than each of the 6 days before. Inside day:
today's high and low within yesterday's. In an uptrend, place a buy-stop a
cent above the signal day's high for the next day only; stop a cent below
the signal day's low; target 2x the risk; 5 trading days at most.
Numbers from /mnt/project-files/rulebooks/20-크래벨-NR7-인사이드데이-돌파-스윙.md.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .events import Panel, Trade, buy_stop_fill, one_per_stock, walk

TITLE = "크래벨 NR7·인사이드 데이 돌파 (스윙)"
RULEBOOK = "20"
START = "2010-01-01"
PERIODS = ["2010", "2018"]
SLOTS = 10
NEEDS = ()
PIT_OK = True

MAX_DAYS, R_MULT, STRETCH = 5, 2.0, 0.5


def rule(p: Panel, ext: dict, *, nr: int = 7, inside: bool = True, stretch: bool = False, ma_exit: bool = False,
         trend: bool = True, hv: bool = False) -> list[Trade]:
    O, H, L, C = p.O, p.H, p.L, p.C
    n = len(C)
    rng = H - L
    narrow = np.ones_like(C, bool)
    for j in range(1, nr):
        prev = np.full_like(rng, np.nan)
        prev[j:] = rng[:-j]
        narrow &= rng < prev
    sig = narrow & p.elig
    if inside:
        ins = np.zeros_like(sig)
        ins[1:] = (H[1:] <= H[:-1]) & (L[1:] >= L[:-1])
        sig &= ins
    if trend:
        sig &= (C > p.sma(50)) & (p.sma(50) > p.sma(200))
    if hv:
        lr = pd.DataFrame(np.log(C)).diff()
        ratio = (lr.rolling(10, min_periods=10).std() / lr.rolling(100, min_periods=100).std()).to_numpy()
        sig &= ratio < 0.5
    sig[-1] = False
    avg_rng = pd.DataFrame(rng).rolling(10, min_periods=10).mean().to_numpy()
    ma10 = p.sma(10)
    st = ext.setdefault("_nr7", {})
    key = (nr, inside, stretch, ma_exit, trend, hv)
    st[key] = [0, 0, 0]                                         # signals, fills, stopped on day one

    def make(s, k):
        q = s + 1
        st[key][0] += 1
        if stretch:
            if not np.isfinite(O[q, k]):
                return None
            level = O[q, k] + STRETCH * avg_rng[s, k]
            stop = O[q, k] - STRETCH * avg_rng[s, k]
            fill = level if H[q, k] >= level else None
        else:
            level = H[s, k] + 0.01
            stop = L[s, k] - 0.01
            fill = buy_stop_fill(O[q, k], H[q, k], level)
        if fill is None or not np.isfinite(fill):
            return None
        st[key][1] += 1
        kw = {"close_rule": lambda j: "종가 < 10일 평균" if C[j, k] < ma10[j, k] else None} if ma_exit else \
            {"target": fill + R_MULT * (fill - stop)}
        legs = walk(p, k, q, fill, at_close=False, stop=stop, max_day=q + (20 if ma_exit else MAX_DAYS) - 1, **kw)
        if legs and legs[0][1] == q and legs[0][3].startswith("손절"):
            st[key][2] += 1
        return Trade(k, q, fill, legs, score=-rng[s, k] / C[s, k])
    return one_per_stock(sig, make)


RULES = [
    ("C1", "C1 NR7 + 인사이드 데이, 상승 추세, 고가 돌파 대기 주문, 2R 익절, 5일", partial(rule)),
    ("C2", "C2 인사이드 데이 조건 없음 (NR7만)", partial(rule, inside=False)),
    ("C3", "C3 NR4 + 인사이드 데이", partial(rule, nr=4)),
    ("C4", "C4 다음 날 시가 + 0.5 × 10일 평균 폭 (크래벨 stretch), 손절 시가 − 같은 폭", partial(rule, stretch=True)),
    ("C5", "C5 익절 없이 종가 < 10일 평균 매도 (최대 20일)", partial(rule, ma_exit=True)),
    ("C6", "C6 추세 조건 없음", partial(rule, trend=False)),
    ("C7", "C7 10일/100일 변동성 < 0.5 추가", partial(rule, hv=True)),
]


def extra(p: Panel, ext: dict, frames: dict[str, pd.DataFrame], split: pd.Timestamp) -> list[str]:
    lines = ["## 대기 주문 체결률과 첫날 손절", ""]
    for (nr, ins, stretch, ma, trend, hv), (s, f, d1) in ext.get("_nr7", {}).items():
        name = f"NR{nr}{'+인사이드' if ins else ''}{', stretch' if stretch else ''}{', 10일선 매도' if ma else ''}" \
               f"{'' if trend else ', 추세 없음'}{', 변동성 수축' if hv else ''}"
        lines.append(f"- {name}: 신호 {s}번, 체결 {f}번({f / max(s, 1) * 100:.0f}%), 체결 당일 손절 {d1}번"
                     f"({d1 / max(f, 1) * 100:.0f}%)")
    return lines
