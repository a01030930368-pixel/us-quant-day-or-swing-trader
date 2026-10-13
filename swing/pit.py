"""Point-in-time S&P 500 members and prices that include delisted stocks.

Used to measure survivorship bias for rulebook 10 (buying stocks that fell,
with no stop loss), where today's member list hides exactly the stocks that
kept falling.

- Members: today's S&P 500 list rolled back through Wikipedia's "Selected
  changes to the list of S&P 500 components" table. Wikipedia's change list
  is known to miss a few early changes; from 2018 it is close to complete.
- Prices: daily regular-session bars aggregated from the free 1-minute data
  set mito0o852/OHLCV-1m (it keeps delisted tickers), built by the day-trading
  workflow (daily_YYYY.parquet: ticker, d, o, h, l, c, v). These prices are not
  adjusted for splits or dividends; splits are found from overnight gaps that
  match a split ratio and taken out [근사]. Dividends stay in (small effect).
- A ticker's series stops at the first gap of more than 10 trading days after
  it left the index, so a ticker reused later by another company is not
  joined to the old one.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SPLITS = (2.0, 3.0, 4.0, 5.0, 10.0, 1.5, 20.0, 8.0)    # forward split ratios (price falls to 1/r)
REVERSE = (2.0, 3.0, 4.0, 5.0, 10.0, 8.0, 20.0, 15.0, 25.0, 30.0)
TOL = 0.03
GAP = 10


def norm(t: str) -> str:
    return str(t).upper().replace(".", "").replace("-", "").replace("/", "").strip()


def sp500_tables(cache: Path = Path("data_cache") / "sp500_pit.csv") -> tuple[list[str], pd.DataFrame]:
    """Today's members and the change table (date, added, removed), normalized tickers."""
    cur_path = cache.with_name("sp500_now.txt")
    if cache.exists() and cur_path.exists():
        ch = pd.read_csv(cache, parse_dates=["date"], keep_default_na=False)
        return cur_path.read_text().split(), ch
    import requests

    html = requests.get(URL, headers={"User-Agent": "us-quant-day-or-swing-trader backtest"}, timeout=30).text
    tables = pd.read_html(StringIO(html))
    now = next(t for t in tables if "Symbol" in [str(c) for c in t.columns] and len(t) > 400)
    current = [norm(s) for s in now["Symbol"].dropna()]
    changes = None
    for t in tables:
        flat = [" ".join(str(x) for x in (c if isinstance(c, tuple) else (c,))).lower() for c in t.columns]
        if any("added" in f for f in flat) and any("removed" in f for f in flat):
            t = t.copy()
            t.columns = flat
            date_c = next(c for c in flat if "date" in c)
            add_c = next(c for c in flat if "added" in c and ("ticker" in c or "symbol" in c))
            rem_c = next(c for c in flat if "removed" in c and ("ticker" in c or "symbol" in c))
            changes = pd.DataFrame({"date": pd.to_datetime(t[date_c], errors="coerce"),
                                    "added": t[add_c].fillna("").map(lambda x: norm(x) if x else ""),
                                    "removed": t[rem_c].fillna("").map(lambda x: norm(x) if x else "")})
            break
    if changes is None:
        raise RuntimeError("S&P 500 변경 표를 찾지 못했습니다")
    changes = changes.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    changes["added"] = changes["added"].replace("NAN", "")
    changes["removed"] = changes["removed"].replace("NAN", "")
    cache.parent.mkdir(exist_ok=True)
    changes.to_csv(cache, index=False)
    cur_path.write_text("\n".join(current))
    return current, changes


def membership(dates: pd.DatetimeIndex, current: list[str], changes: pd.DataFrame) -> pd.DataFrame:
    """True where the ticker was in the S&P 500 at that day's open (normalized tickers as columns)."""
    names = sorted(set(current) | set(changes["added"]) | set(changes["removed"]) - {""})
    col = {t: j for j, t in enumerate(names)}
    out = np.zeros((len(dates), len(names)), bool)
    state = set(current)
    ch = changes.sort_values("date", ascending=False)
    pos = len(dates)
    for d, grp in ch.groupby("date", sort=False):
        k = int(dates.searchsorted(d))                       # rows from k on are on/after the change
        idx = [col[t] for t in state]
        out[k:pos, idx] = True
        pos = min(pos, k)
        for _, r in grp.iterrows():                          # step back to before the change
            if r["added"]:
                state.discard(r["added"])
            if r["removed"]:
                state.add(r["removed"])
    if pos > 0:
        out[:pos, [col[t] for t in state]] = True
    return pd.DataFrame(out, index=dates, columns=names)


def adjust_splits(g: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Back-adjust one ticker's daily bars for splits found from overnight gaps."""
    g = g.sort_values("d").reset_index(drop=True)
    r = (g["o"] / g["c"].shift(1)).to_numpy()
    factor = np.ones(len(g))
    found = 0
    for i in range(1, len(g)):
        x = r[i]
        if not np.isfinite(x) or 0.7 < x < 1.4:
            continue
        f = None
        for s in SPLITS:
            if abs(x * s - 1) < TOL:
                f = 1 / s
        for s in REVERSE:
            if abs(x / s - 1) < TOL:
                f = s
        if f is not None:
            factor[:i] *= f
            found += 1
    out = g.copy()
    for c in ("o", "h", "l", "c"):
        out[c] = g[c] * factor
    out["v"] = g["v"] / factor
    return out, found


def load_daily(paths: list[Path], keep: set[str], left: dict[str, pd.Timestamp]) -> tuple[dict, dict]:
    """Wide OHLCV frames for the normalized tickers in `keep`, split-adjusted.

    `left` maps a ticker to the last day it was in the index; its series is cut
    at the first gap of more than GAP trading days after that.
    """
    parts = []
    for p in paths:                                          # filter file by file: all tickers would not fit
        x = pd.read_parquet(p, columns=["ticker", "d", "o", "h", "l", "c", "v"])
        x["t"] = x["ticker"].map(norm)
        parts.append(x[x["t"].isin(keep)])
    raw = pd.concat(parts, ignore_index=True)
    raw["d"] = pd.to_datetime(raw["d"])
    raw = raw.drop_duplicates(["t", "d"]).sort_values(["t", "d"])
    days = pd.DatetimeIndex(sorted(raw["d"].unique()))
    pieces, info = [], {"splits": 0, "cut": 0}
    for t, g in raw.groupby("t"):
        if t in left:
            k = g["d"].map(lambda x: days.searchsorted(x)).to_numpy()
            after = g["d"].to_numpy() > np.datetime64(left[t])
            jump = np.r_[False, np.diff(k) > GAP] & after
            if jump.any():
                g = g.iloc[:int(np.argmax(jump))]
                info["cut"] += 1
        g, n = adjust_splits(g)
        info["splits"] += n
        pieces.append(g)
    df = pd.concat(pieces)
    wide = {name: df.pivot(index="d", columns="t", values=c).reindex(days)
            for name, c in (("Open", "o"), ("High", "h"), ("Low", "l"), ("Close", "c"), ("Volume", "v"))}
    return wide, info
