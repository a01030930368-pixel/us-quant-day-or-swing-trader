"""Rulebook 17: buy the year's big losers in late December, sell in mid-January (tax-loss selling).

The classic January effect lives in small stocks owned by individuals; large
caps may even show December winners going on (J6, the reverse control). One
trade cluster a year, so the year is the sample (date-level t, b25 lesson).
Numbers from /mnt/project-files/rulebooks/17-연말손실주-1월반등.md.

Pick day: the first trading day after December's 15th trading day; the
screen uses the close before it, the buy is that day's close [해석].
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .events import MEMBERS, Panel, Trade, close_out

TITLE = "연말 손실주 1월 반등 (절세 매도)"
RULEBOOK = "17"
START = "1998-01-01"
PERIODS = ["1998", "2010", "2018"]
SLOTS = 20
NEEDS = ()
PIT_OK = True
GATE_YEARS = 15

PICK, SELL, MAX_N = 15, 10, 20


def calendar(dates: pd.DatetimeIndex) -> list[dict]:
    """Per year: first day, December's first day, the pick day, the last December day, next January days."""
    out = []
    pos = pd.Series(np.arange(len(dates)), index=dates)
    for y in sorted(set(dates.year)):
        yr = pos[pos.index.year == y]
        dec = yr[yr.index.month == 12]
        jan = pos[(pos.index.year == y + 1) & (pos.index.month == 1)]
        if len(dec) <= PICK or len(jan) < SELL or yr.index[0].month != 1:
            continue
        out.append({"year": y, "first": int(yr.iloc[0]), "dec1": int(dec.iloc[0]), "pick": int(dec.iloc[PICK]),
                    "dec_last": int(dec.iloc[-1]), "jan": [int(x) for x in jan.iloc[:SELL]]})
    return out


def rule(p: Panel, ext: dict, *, loss: float = -0.30, buy_last: bool = False, sell_day: int = SELL,
         bottom: bool = False, top: bool = False, small: bool = False) -> list[Trade]:
    C = p.C
    elig = p.elig
    if small and "INFO" in ext:
        info = ext["INFO"].reindex(p.tickers)["index"]
        m = info.isin(MEMBERS + (600,)).to_numpy()
        elig = p.liquid & m[None, :]
    out = []
    for cal in calendar(p.dates):
        b = cal["dec_last"] if buy_last else cal["pick"]
        s = b - 1                                               # screen on the close before the buy
        ytd = C[s] / C[cal["first"]] - 1
        dec = C[s] / C[cal["dec1"]] - 1
        ok = elig[s] & np.isfinite(ytd) & np.isfinite(C[b])
        if top or bottom:
            vals = np.where(ok, ytd, np.nan)
            if not np.isfinite(vals).any():
                continue
            q = np.nanquantile(vals, 0.9 if top else 0.1)
            pick = ok & ((ytd >= q) if top else (ytd <= q))
        else:
            pick = ok & (ytd <= loss) & (dec < 0)
        ks = np.flatnonzero(pick)
        order = ks[np.argsort(-ytd[ks] if top else ytd[ks])][:MAX_N]
        xi = cal["jan"][sell_day - 1]
        for k in order:
            out.append(Trade(int(k), b, C[b, k], [close_out(C[:, k], b, xi, f"1월 {sell_day}번째 거래일 종가")],
                             score=-ytd[k] if not top else ytd[k], tag=str(cal["year"])))
    return out


RULES = [
    ("J1", "J1 올해 −30% 이하·12월도 하락, 12월 16번째 거래일 종가 매수 → 1월 10번째 거래일 종가", partial(rule)),
    ("J2", "J2 12월 마지막 거래일 종가 매수", partial(rule, buy_last=True)),
    ("J3", "J3 1월 5번째 거래일 매도", partial(rule, sell_day=5)),
    ("J4", "J4 올해 −20% 이하", partial(rule, loss=-0.20)),
    ("J5", "J5 올해 하락률 하위 10%", partial(rule, bottom=True)),
    ("J6", "J6 반대 대조: 올해 상승률 상위 10%", partial(rule, top=True)),
    ("J7", "J7 S&P 600 소형주 포함 (지금 구성 종목)", partial(rule, small=True)),
]


def extra(p: Panel, ext: dict, frames: dict[str, pd.DataFrame], split: pd.Timestamp) -> list[str]:
    """One row per year (the year is the sample) and the date-level t."""
    spy = pd.Series(p.spy, index=p.dates)
    lines = []
    for name in ("J1", "J2", "J6", "J7"):
        df = frames.get(name)
        if df is None or df.empty:
            continue
        lines += [f"## 해마다 ({name})", "",
                  "| 해 | 종목 수 | 평균(비용 전) | 중앙값 | SPY 같은 기간 | SPY 대비 |", "|---|---|---|---|---|---|"]
        rows = []
        for tag, x in df.groupby("tag"):
            a, b = pd.Timestamp(x["entry_date"].iloc[0]), pd.Timestamp(x["exit_date"].max())
            sp = spy.asof(b) / spy.asof(a) - 1
            rows.append((tag, len(x), x["gross"].mean(), x["gross"].median(), sp))
            lines.append(f"| {tag} | {len(x)} | {x['gross'].mean() * 100:+.1f}% | {x['gross'].median() * 100:+.1f}% | "
                         f"{sp * 100:+.1f}% | {(x['gross'].mean() - sp) * 100:+.1f}%p |")
        for lo, hi, lab in ((0, 2018, "2017년까지"), (2018, 2100, "2018년부터")):
            g = np.array([r[2] for r in rows if lo <= int(r[0]) < hi])
            ex = np.array([r[2] - r[4] for r in rows if lo <= int(r[0]) < hi])
            if len(g) >= 3:
                t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if g.std(ddof=1) > 0 else np.nan
                te = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))) if ex.std(ddof=1) > 0 else np.nan
                lines.append(f"\n- {lab}: {len(g)}해, 해 평균 {g.mean() * 100:+.2f}% (날짜 기준 t {t:.2f}), "
                             f"SPY 대비 {ex.mean() * 100:+.2f}%p (t {te:.2f}). 관문: 비용 전 +0.9% 이상, 해 수 {GATE_YEARS} 이상.")
        lines.append("")
    return lines
