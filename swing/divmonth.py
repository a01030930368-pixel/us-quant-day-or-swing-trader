"""Rulebook 16: buy before the expected ex-dividend day, sell the day before it (dividend month premium).

Hartzmark & Solomon: stocks earn more in months they are expected to pay a
dividend (about 0.5% a month, less than our round-trip cost), so the gain
would have to sit in the last days before the ex-date. Numbers from
/mnt/project-files/rulebooks/16-배당락전-매수-배당월프리미엄.md.

D0 [해석]: expected ex-date = last ex-date + the usual gap, only for stocks
whose last 4 gaps are within 10 days of their average. Buy N trading days
before it, sell at the close before the actual ex-date (no dividend), or at
the expected day + 10 if none came.
Prices are Yahoo's dividend-adjusted closes; between a buy and a sale both
before the ex-date that makes no difference. D6 keeps the dividend and pays
15% US withholding tax on it.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .events import Panel, Trade, close_out

TITLE = "배당락 전 매수 (배당 달 프리미엄)"
RULEBOOK = "16"
START = "2010-01-01"
PERIODS = ["2010", "2018"]
SLOTS = 10
NEEDS = ("dividends",)
PIT_OK = False

LEAD, LATE, REG_TOL, TAX = 10, 10, 10, 0.15


def _yield(p: Panel, ext: dict) -> np.ndarray:
    if "div_yield" not in p._cache:
        div = pd.DataFrame(ext["div_amount"], index=p.dates)
        trailing = div.rolling("365D").sum().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            p._cache["div_yield"] = trailing / ext["raw_close"]
    return p._cache["div_yield"]


def regular(dates: pd.DatetimeIndex, xs: np.ndarray, j: int) -> float | None:
    """The usual gap in days if the 4 gaps up to ex-date j are regular, else None."""
    if j < 4:
        return None
    gaps = np.diff(dates[xs[j - 4:j + 1]].values).astype("timedelta64[D]").astype(float)
    if np.all(np.abs(gaps - gaps.mean()) <= REG_TOL):
        return float(np.median(gaps))
    return None


def rule(p: Panel, ext: dict, *, lead: int = LEAD, min_yield: float = 0.02, keep: bool = False,
         actual: bool = False, market: bool = False) -> list[Trade]:
    C, O = p.C, p.O
    n = len(C)
    yld = _yield(p, ext)
    spy_ok = p.spy > p.spy_sma(200)
    out = []
    for k, xs in ext["ex_days"].items():
        busy = -1
        for j in range(len(xs) - 1):
            gap = regular(p.dates, xs, j)
            if gap is None:
                continue
            nxt = int(xs[j + 1])
            if actual:
                pe = nxt
            else:
                pe = int(p.dates.searchsorted(p.dates[xs[j]] + pd.Timedelta(days=gap)))
            b = pe - lead
            if b <= busy or b <= xs[j] or b >= n or nxt <= b or not p.elig[b, k]:
                continue
            if min_yield and not (yld[b, k] >= min_yield):
                continue
            if market and not spy_ok[b]:
                continue
            if nxt <= pe + LATE:
                if nxt - 1 <= b:
                    continue
                if keep:
                    if not np.isfinite(O[nxt, k]):
                        continue
                    amount = ext["div_amount"][nxt, k]
                    raw_prev = ext["raw_close"][nxt - 1, k]
                    tax = TAX * amount / raw_prev * C[nxt - 1, k] if raw_prev > 0 else 0.0
                    leg = (1.0, nxt, O[nxt, k] - tax, "배당락일 시가 (배당 받음, 세금 15%)")
                else:
                    leg = close_out(C[:, k], b, nxt - 1, "배당락 전날 종가")
            else:
                leg = close_out(C[:, k], b, pe + LATE, "배당락 없음 (예정일 +10일)")
            tr = Trade(k, b, C[b, k], [leg], score=yld[b, k] if np.isfinite(yld[b, k]) else 0.0)
            out.append(tr)
            busy = tr.last()
    return out


RULES = [
    ("D1", "D1 예정 배당락 10일 전 종가 매수 → 배당락 전날 종가, 배당수익률 2% 이상", partial(rule)),
    ("D2", "D2 5일 전 매수", partial(rule, lead=5)),
    ("D3", "D3 20일 전 매수", partial(rule, lead=20)),
    ("D4", "D4 배당수익률 조건 없음", partial(rule, min_yield=0.0)),
    ("D5", "D5 배당수익률 4% 이상", partial(rule, min_yield=0.04)),
    ("D6", "D6 배당락일 시가 매도 (배당 받고 세금 15% 뺌)", partial(rule, keep=True)),
    ("D7", "D7 실제 배당락일 기준 10일 전 (발표 뒤라는 가정)", partial(rule, actual=True)),
    ("D8", "D8 SPY 200일 평균 위일 때만", partial(rule, market=True)),
]


def extra(p: Panel, ext: dict, frames: dict[str, pd.DataFrame], split: pd.Timestamp) -> list[str]:
    """Average excess return over SPY by day around the actual ex-date (-20 .. +5)."""
    C = p.C
    r = np.full_like(C, np.nan)
    r[1:] = C[1:] / C[:-1] - 1
    spy = np.r_[np.nan, p.spy[1:] / p.spy[:-1] - 1]
    ex = r - spy[:, None]
    yld = _yield(p, ext)
    sums = {"2010~2017": np.zeros(26), "2018~": np.zeros(26)}
    cnts = {"2010~2017": np.zeros(26), "2018~": np.zeros(26)}
    for k, xs in ext["ex_days"].items():
        for x in xs:
            if x - 21 < 0 or x + 5 >= len(C) or not p.elig[x - 21, k] or not (yld[x - 21, k] >= 0.02):
                continue
            if p.dates[x] < pd.Timestamp(START):
                continue
            key = "2018~" if p.dates[x] >= split else "2010~2017"
            seg = ex[x - 20:x + 6, k]
            ok = np.isfinite(seg)
            sums[key][ok] += seg[ok]
            cnts[key][ok] += 1
    lines = ["## 배당락일 앞뒤 날짜별 평균 수익 (SPY 대비, 비용 전)", "",
             "배당수익률 2% 이상, 조건을 만족한 종목의 실제 배당락일 기준. 0일 = 배당락일(배당 보정 가격이라 배당만큼의 하락은 빠져 있음).", "",
             "| 날 | 2010~2017 | 2018~ | 2018~ 막대 |", "|---|---|---|---|"]
    for i, d in enumerate(range(-20, 6)):
        a = sums["2010~2017"][i] / max(cnts["2010~2017"][i], 1)
        b = sums["2018~"][i] / max(cnts["2018~"][i], 1)
        bar = ("+" * int(round(b * 1e4 / 2)) if b > 0 else "-" * int(round(-b * 1e4 / 2)))[:30]
        lines.append(f"| {d:+d} | {a * 100:+.3f}% | {b * 100:+.3f}% | {bar} |")
    lines.append(f"\n- 사건 수: 2010~2017 {int(cnts['2010~2017'][0])}번, 2018~ {int(cnts['2018~'][0])}번. 막대 + 하나 = 0.02%.")
    pre10 = sums["2018~"][10:20].sum() / max(cnts["2018~"][10], 1)
    lines.append(f"- 2018~ 배당락 전 10일(−10 ~ −1일) 합: {pre10 * 100:+.2f}% (왕복 비용 0.6%와 비교)")
    return lines
