"""Rulebook 12: buy a few days before a scheduled earnings release, sell the day before it.

Papers: Frazzini & Lamont (stocks rise into scheduled announcements), Aboody
et al. (past 12-month winners +1.58% over the market in the 5 days before,
-1.86% in the 5 days after, 1971-2005), Heitz et al. 2020 (the US premium is
gone after 2004). All numbers from /mnt/project-files/rulebooks/12-실적발표직전-매수-발표프리미엄.md.

Expected date [원전 방식]: last year's same-quarter release (8-K item 2.02)
+ 364 days, the next trading day if that is a holiday. Only the stock's own
past releases are used, so nothing is known before it happened.
- No buy if this quarter's release already came out (a release in the
  30 trading days before the buy day) [추가].
- Sell at the close the day before the actual release reacts; if no release
  comes by the expected day + 5 trading days, sell at that close [추가].
- A release right after the buy (reaction the next day) leaves no day to hold;
  those signals are skipped and counted.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .events import Panel, Trade, close_out

TITLE = "실적 발표 직전 매수 (발표 프리미엄)"
RULEBOOK = "12"
START = "2004-01-01"
PERIODS = ["2004", "2010", "2018"]
SLOTS = 10
NEEDS = ("earnings",)
PIT_OK = False

LEAD = 5            # A1: buy 5 trading days before [원전]
LATE = 5            # sell if no release by the expected day + 5 [추가]
ALREADY = 30        # a release in the last 30 trading days = this quarter's is out [추가]
WINDOW = (330, 400)  # last year's same quarter: a release 330~400 days before the expected day


def _expected(react: np.ndarray, dates: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """(past release day, expected day of next year's release) for each past release."""
    out = []
    for r in react:
        e = int(dates.searchsorted(dates[r] + pd.Timedelta(days=364)))
        if e < len(dates):
            out.append((int(r), e))
    return out


def rule(p: Panel, ext: dict, *, lead: int = LEAD, through: bool = False, top: bool = False,
         habit: bool = False, market: bool = False, volume: bool = False) -> list[Trade]:
    react_all = ext["earnings"]
    C, O, V = p.C, p.O, p.V
    n = len(p.dates)
    ret12 = np.full_like(C, np.nan)
    ret12[252:] = C[252:] / C[:-252] - 1
    elig_ret = np.where(p.elig, ret12, np.nan)
    cut = np.nanquantile(np.where(np.isfinite(elig_ret), elig_ret, np.nan), 0.9, axis=1) if top else None
    spy_ok = p.spy > p.spy_sma(200) if market else None
    avg50 = p.sma(50, "V") if volume else None
    trades, skipped = [], 0
    for k, react in react_all.items():
        if len(react) == 0:
            continue
        busy = -1
        for prev, e in _expected(react, p.dates):
            b = e - lead
            if b <= busy or b < 1 or b >= n or not p.elig[b, k]:
                continue
            past = react[react <= b]
            if len(past) and past[-1] > b - ALREADY:
                continue                                    # this quarter's release is already out
            nxt = react[react > b]
            if top and not (ret12[b, k] >= cut[b]):
                continue
            if market and not spy_ok[b]:
                continue
            if habit:
                four = past[-4:]
                runs = [C[r - 1, k] / C[r - 1 - lead, k] - 1 for r in four if r - 1 - lead >= 0]
                if len(runs) < 4 or not np.nanmean(runs) > 0:
                    continue
            if volume:
                week = V[max(prev - 2, 0):prev + 3, k]
                base = avg50[max(prev - 5, 0), k]
                if not (np.isfinite(base) and np.nanmean(week) >= 2 * base):
                    continue
            ep = C[b, k]
            if len(nxt) and nxt[0] <= e + LATE:
                r = int(nxt[0])
                if r - 1 <= b:
                    skipped += 1
                    continue
                if through:
                    xi = r + 1
                    if xi >= n or not np.isfinite(O[xi, k]):
                        leg = close_out(C[:, k], b, xi, "발표 다음 날 시가")
                    else:
                        leg = (1.0, xi, O[xi, k], "발표 다음 날 시가")
                else:
                    leg = close_out(C[:, k], b, r - 1, "발표 전날 종가")
            else:
                leg = close_out(C[:, k], b, e + LATE, "발표 없음 (예정일 +5일)")
            tr = Trade(k, b, ep, [leg], score=ret12[b, k] if np.isfinite(ret12[b, k]) else 0.0)
            trades.append(tr)
            busy = tr.last()
    ext.setdefault("_skipped", {})[(lead, through, top, habit, market, volume)] = skipped
    return trades


RULES = [
    ("A1", "A1 예정일 5일 전 종가 매수 → 발표 전날 종가 매도", partial(rule)),
    ("A2", "A2 발표까지 들고 감 (발표 다음 날 시가 매도)", partial(rule, through=True)),
    ("A3", "A3 예정일 10일 전 매수", partial(rule, lead=10)),
    ("A4", "A4 지난 12개월 상승률 상위 10%만", partial(rule, top=True)),
    ("A5", "A5 A4 + 발표까지 들고 감", partial(rule, top=True, through=True)),
    ("A6", "A6 지난 4번 발표 전 5일이 평균 플러스인 종목만", partial(rule, habit=True)),
    ("A7", "A7 SPY 200일 평균 위일 때만", partial(rule, market=True)),
    ("A8", "A8 작년 같은 분기 발표 주간 거래량 2배 이상", partial(rule, volume=True)),
]


def extra(p: Panel, ext: dict, frames: dict[str, pd.DataFrame], split: pd.Timestamp) -> list[str]:
    """Market-adjusted return over the holding window (the paper's +1.58% compares to this)."""
    lines = ["## 보유 기간 시장 대비 수익 (논문 비교)", "",
             "SPY 같은 기간 수익을 뺀 값입니다(비용 전). Aboody 외: 강세주 발표 전 5일 +1.58%, 발표 뒤 5일 −1.86%.", "",
             "| 규칙 | 2004~2009 | 2010~2017 | 2018~ |", "|---|---|---|---|"]
    spy = pd.Series(p.spy, index=p.dates)
    for name, df in frames.items():
        if df.empty:
            continue
        d = df.copy()
        a = spy.reindex(pd.to_datetime(d["entry_date"])).to_numpy()
        b = spy.reindex(pd.to_datetime(d["exit_date"])).to_numpy()
        d["excess"] = d["gross"] - (b / a - 1)
        y = pd.to_datetime(d["entry_date"]).dt.year
        cells = []
        for lo, hi in ((2004, 2010), (2010, 2018), (2018, 2100)):
            x = d[(y >= lo) & (y < hi)]["excess"]
            cells.append(f"{x.mean() * 100:+.2f}% ({len(x)})" if len(x) else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    sk = ext.get("_skipped", {})
    if sk:
        lines += ["", f"- 매수 다음 날 바로 발표가 나와 들고 있을 날이 없어 건너뛴 신호: A1 기준 {list(sk.values())[0]}번."]
    return lines
