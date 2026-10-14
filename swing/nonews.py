"""Rulebook 14: buy a sharp one-day drop that came with no company news; sell on the bounce.

Chan (2003) and others: drops without news tend to reverse, drops with news
keep going. Free news data does not exist, so a company SEC filing (any 8-K,
8-K/A, 10-Q, 10-K) stands in for news [근사]: an analyst downgrade with no
filing counts as "no news". The no-news vs with-news comparison (N1 vs N8)
is the point. Numbers from /mnt/project-files/rulebooks/14-뉴스없는-급락-반등.md.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .events import Panel, Trade, one_per_stock, walk

TITLE = "뉴스 없는 급락 반등"
RULEBOOK = "14"
START = "2010-01-01"
PERIODS = ["2010", "2018"]
SLOTS = 10
NEEDS = ("news",)
PIT_OK = False

DROP, VS_SPY, MAX_DAYS = -0.07, -0.05, 10


def _news_count(p: Panel, ext: dict, back: int, ahead: int) -> np.ndarray:
    """Filings known by the open of days s-back .. s+ahead, per (day s, stock)."""
    n, m = p.C.shape
    hit = np.zeros((n + ahead + back + 1, m))
    for k, days in ext["news"].items():
        np.add.at(hit[:, k], days[days < n], 1)
    cs = np.vstack([np.zeros((1, m)), np.cumsum(hit, axis=0)])
    s = np.arange(n)
    lo = np.clip(s - back, 0, None)
    hi = np.clip(s + ahead + 1, 0, len(hit))
    return cs[hi] - cs[lo]


def rule(p: Panel, ext: dict, *, drop: float = DROP, trend: bool = True, stop: float | None = None,
         days: int = MAX_DAYS, sma5_exit: bool = True, quiet_vol: bool = False, news: bool = False) -> list[Trade]:
    C, O = p.C, p.O
    n = len(C)
    r = np.full_like(C, np.nan)
    r[1:] = C[1:] / C[:-1] - 1
    spy = np.r_[np.nan, p.spy[1:] / p.spy[:-1] - 1]
    sig = (r <= drop) & (r - spy[:, None] <= VS_SPY) & p.elig
    if trend:
        sig &= C > p.sma(200)
    if quiet_vol:
        avg50 = pd.DataFrame(p.V).rolling(50, min_periods=50).mean().shift(1).to_numpy()
        sig &= p.V <= 3 * avg50
    cnt = _news_count(p, ext, 3, 1)
    sig &= (cnt > 0) if news else (cnt == 0)
    sig[-1] = False
    ma5 = p.sma(5)

    def make(s, k):
        ei = s + 1
        ep = O[ei, k]
        if not np.isfinite(ep):
            return None
        return Trade(k, ei, ep, walk(p, k, ei, ep, at_close=False, stop=None if stop is None else ep * (1 - stop),
                                     max_day=s + days,
                                     close_rule=(lambda j: "종가 > 5일 평균" if C[j, k] > ma5[j, k] else None)
                                     if sma5_exit else None), score=-r[s, k])
    return one_per_stock(sig, make)


def weekly(p: Panel, ext: dict) -> list[Trade]:
    """N9: a week down 12%+ with no filing that week (or before Monday's open); Monday open to Friday close."""
    C, O = p.C, p.O
    wk = pd.Series(np.arange(len(p.dates)), index=p.dates).groupby(p.dates.to_period("W-FRI"))
    ends = [int(g.iloc[-1]) for _, g in wk]
    starts = [int(g.iloc[0]) for _, g in wk]
    cnt = _news_count(p, ext, 0, 0)
    out = []
    for w in range(1, len(ends) - 1):
        a, b = ends[w - 1], ends[w]
        nb, ne = starts[w + 1], ends[w + 1]
        r = C[b] / C[a] - 1
        filings = cnt[a + 1:nb + 1].sum(axis=0)                 # this week's opens through next Monday's open
        ok = (r <= -0.12) & (filings == 0) & p.elig[b] & np.isfinite(O[nb])
        for k in np.flatnonzero(ok):
            out.append(Trade(int(k), nb, O[nb, k], [walk_close(C[:, k], nb, ne)], score=-r[k]))
    return out


def walk_close(c: np.ndarray, a: int, b: int) -> tuple:
    from .events import close_out
    return close_out(c, a, b, "금요일 종가")


RULES = [
    ("N1", "N1 뉴스(공시) 없는 −7% 급락, 200일선 위, 다음 날 시가 매수 → 5일선 위 종가 또는 10일", partial(rule)),
    ("N2", "N2 급락 −5%", partial(rule, drop=-0.05)),
    ("N3", "N3 급락 −10%", partial(rule, drop=-0.10)),
    ("N4", "N4 200일 조건 없음", partial(rule, trend=False)),
    ("N5", "N5 손절 −8%", partial(rule, stop=0.08)),
    ("N6", "N6 정확히 5거래일 보유", partial(rule, days=5, sma5_exit=False)),
    ("N7", "N7 조용한 급락 (거래량 50일 평균 3배 이하)", partial(rule, quiet_vol=True)),
    ("N8", "N8 대조군: 같은 급락인데 공시 있음", partial(rule, news=True)),
    ("N9", "N9 주간: 한 주 −12%, 공시 없음, 월요일 시가 → 금요일 종가", weekly),
]


def extra(p: Panel, ext: dict, frames: dict[str, pd.DataFrame], split: pd.Timestamp) -> list[str]:
    lines = ["## 거래대금 크기별 (시가총액 대신, N1)", "",
             "시가총액 자료가 없어 매수 전 20일 평균 거래대금으로 나눴습니다 [근사]. 논문: 작은 종목에서만 효과.", ""]
    df = frames.get("N1")
    if df is None or df.empty:
        return lines + ["(거래 없음)"]
    value = pd.DataFrame(p.C * p.V, index=p.dates, columns=p.tickers).rolling(20).mean()
    dv = [value.at[pd.Timestamp(d), t] if t in value.columns else np.nan
          for d, t in zip(df["entry_date"], df["ticker"])]
    d = df.assign(dv=dv).dropna(subset=["dv"])
    if len(d) < 9:
        return lines + ["(거래가 적어 생략)"]
    d["third"] = pd.qcut(d["dv"].rank(method="first"), 3, labels=["작은 1/3", "중간 1/3", "큰 1/3"])
    lines += ["| 크기 | 거래수 | 비용 전 평균 | 비용 후 평균 | 승률 |", "|---|---|---|---|---|"]
    for name, x in d.groupby("third", observed=True):
        lines.append(f"| {name} | {len(x)} | {x['gross'].mean() * 100:+.2f}% | {x['net'].mean() * 100:+.2f}% | "
                     f"{(x['net'] > 0).mean() * 100:.0f}% |")
    return lines
