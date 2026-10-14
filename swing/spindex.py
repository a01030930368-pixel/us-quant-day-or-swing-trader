"""Rulebooks 18 and 19: S&P 500 additions (buy before the index funds) and deletions (buy after they sold).

Member lists by date come from github.com/fja05680/sp500 (pit.py). The lists
give the day a change took effect, not the day S&P announced it, so:
- 18: buy 2 trading days before the first day in the index (almost always
  after the announcement) [근사]; X3 buys 5 days before and may be before the
  announcement (look-ahead warning).
- 19: the deletion took effect on the first day out; index funds sold at the
  close before it. Buy at that first day's close [해석].
Rulebooks: /mnt/project-files/rulebooks/18-SP500-편입-발표직후-매수.md, 19-SP500-퇴출종목-반등.md.
S&P 400 has no member history, so the "moved from/to the S&P 400" variants
(X5, X6, Q6) cannot be run [미구현].
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .events import Panel, Trade, close_out, stopped_at, walk

PERIODS = ["1996", "2004", "2010", "2018"]
SLOTS = 10
NEEDS = ("sp_member",)
PIT_OK = True
START = "1996-01-01"
HISTORY = 60         # an addition needs 60 days of prices before it (not a fresh spin-off or merger) [추가]


def changes(member: np.ndarray, added: bool) -> np.ndarray:
    """(day, column) where a stock entered (added=True) or left the index; the first day is skipped."""
    m = member.astype(bool)
    flip = np.zeros_like(m)
    flip[1:] = (m[1:] & ~m[:-1]) if added else (~m[1:] & m[:-1])
    return np.argwhere(flip)


def add_rule(p: Panel, ext: dict, *, before: int = 2, sell: int = -1, after_buy: bool = False) -> list[Trade]:
    """Buy at the close `before` days ahead of the first day in the index, sell at day `sell` (relative)."""
    C = p.C
    n = len(C)
    out = []
    for x, k in changes(ext["sp_member"], True):
        b = x - 1 if after_buy else x - before
        xi = x + sell
        if b < HISTORY or xi >= n or not p.liquid[b, k] or not np.isfinite(C[x - HISTORY, k]):
            continue
        out.append(Trade(int(k), int(b), C[b, k], [close_out(C[:, k], b, xi, "정해진 날 종가")],
                         tag=str(p.dates[x].date())))
    return out


def del_rule(p: Panel, ext: dict, *, hold: int = 20, drop: bool = True, stop: float | None = 0.15) -> list[Trade]:
    C = p.C
    n = len(C)
    value = pd.DataFrame(C * p.V).rolling(20, min_periods=20).mean().to_numpy()
    out = []
    stats = ext.setdefault("_del", {"mna": 0, "events": 0})
    stats["mna"] = stats["events"] = 0
    for x, k in changes(ext["sp_member"], False):
        if x + 1 >= n or not np.isfinite(C[x, k]):
            continue
        stats["events"] += 1
        if stopped_at(C[:, k], x, min(x + 30, n - 1)) is not None:
            stats["mna"] += 1                 # trading stopped within 30 days: taken over [해석, 미래 정보]
            continue
        if not (C[x, k] >= 5 and value[x, k] >= 10e6):
            continue
        if drop and not (x - 11 >= 0 and C[x - 1, k] / C[x - 11, k] - 1 <= -0.05):
            continue
        ep = C[x, k]
        legs = walk(p, k, int(x), ep, at_close=True, stop=None if stop is None else ep * (1 - stop),
                    max_day=int(x) + hold)
        out.append(Trade(int(k), int(x), ep, legs, tag=str(p.dates[x].date())))
    return out


ADD_RULES = [
    ("X1", "X1 편입일 2거래일 전 종가 매수 → 편입 전날 종가", partial(add_rule)),
    ("X2", "X2 매도 = 편입일 뒤 5거래일째 종가", partial(add_rule, sell=5)),
    ("X3", "X3 매수 = 편입일 5거래일 전 (미래 보기 위험)", partial(add_rule, before=5)),
    ("X4", "X4 반대: 편입 전날 종가 매수 → 20거래일 뒤", partial(add_rule, after_buy=True, sell=19)),
]
DEL_RULES = [
    ("Q1", "Q1 퇴출 뒤 첫날 종가 매수, 퇴출 전 10일 −5% 이하, 20일 보유, 손절 −15%", partial(del_rule)),
    ("Q2", "Q2 10일 보유", partial(del_rule, hold=10)),
    ("Q3", "Q3 40일 보유", partial(del_rule, hold=40)),
    ("Q4", "Q4 하락 조건 없음", partial(del_rule, drop=False)),
    ("Q5", "Q5 손절 없음", partial(del_rule, stop=None)),
    ("Q7", "Q7 기록용 장기: 250일 보유", partial(del_rule, hold=250, stop=None)),
]


def extra_add(p: Panel, ext: dict, frames: dict[str, pd.DataFrame], split: pd.Timestamp) -> list[str]:
    ev = changes(ext["sp_member"], True)
    years = pd.Series([p.dates[x].year for x, _ in ev]).value_counts().sort_index()
    have = sum(1 for x, k in ev if np.isfinite(p.C[x - 1, k]))
    return ["## 편입 사건 수", "",
            f"- 구성 목록에서 찾은 편입 {len(ev)}건, 그중 가격이 있는 것 {have}건. 해마다: "
            + ", ".join(f"{y} {c}" for y, c in years.items())]


def extra_del(p: Panel, ext: dict, frames: dict[str, pd.DataFrame], split: pd.Timestamp) -> list[str]:
    st = ext.get("_del", {})
    lines = ["## 퇴출 사건", "",
             f"- 퇴출 {st.get('events', 0)}건(가격 있는 것), 그중 30거래일 안에 거래가 멈춰 인수합병으로 보고 뺀 것 "
             f"{st.get('mna', 0)}건."]
    for name, df in frames.items():
        if not df.empty:
            gone = df["reason"].str.contains("거래 중단").sum()
            lines.append(f"- {name}: 보유 중 거래가 멈춘(상장폐지 등) 거래 {gone}건")
    return lines
