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

import re
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
        head = " ".join(flat + [str(x).lower() for x in t.head(2).to_numpy().ravel()])
        if not ("added" in head and "removed" in head and t.shape[1] >= 5):
            continue
        cols = {}
        for key, words in (("date", ("date",)), ("added", ("added",)), ("removed", ("removed",))):
            hits = [j for j, f in enumerate(flat) if any(w in f for w in words)]
            tick = [j for j in hits if "ticker" in flat[j] or "symbol" in flat[j]]
            cols[key] = (tick or hits or [None])[0]
        body = t
        if None in cols.values() or cols["added"] == cols["removed"]:
            # header came in as data rows: Wikipedia's layout is date, added (ticker, name), removed (ticker, name), reason
            cols = {"date": 0, "added": 1, "removed": 3}
            body = t[pd.to_datetime(t.iloc[:, 0], errors="coerce").notna()]
        changes = pd.DataFrame({"date": pd.to_datetime(body.iloc[:, cols["date"]], errors="coerce"),
                                "added": body.iloc[:, cols["added"]].fillna("").map(lambda x: norm(x) if x else ""),
                                "removed": body.iloc[:, cols["removed"]].fillna("").map(lambda x: norm(x) if x else "")})
        break
    if changes is None:
        shapes = [(t.shape, [str(c) for c in t.columns][:6]) for t in tables]
        raise RuntimeError(f"S&P 500 변경 표를 찾지 못했습니다: {shapes}")
    changes = changes.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    changes["added"] = changes["added"].replace("NAN", "")
    changes["removed"] = changes["removed"].replace("NAN", "")
    cache.parent.mkdir(exist_ok=True)
    changes.to_csv(cache, index=False)
    cur_path.write_text("\n".join(current))
    return current, changes


FJA = "https://api.github.com/repos/fja05680/sp500/contents/"


def sp500_snapshots(cache: Path = Path("data_cache") / "sp500_snapshots.csv") -> pd.DataFrame:
    """Daily S&P 500 member lists (date, tickers) from github.com/fja05680/sp500, a public data set
    built from S&P change notices. Used first; Wikipedia's change table is the fallback."""
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])
    import requests

    files = requests.get(FJA, headers={"User-Agent": "us-quant-day-or-swing-trader backtest"}, timeout=30).json()
    pick = sorted(f for f in files if f["name"].startswith("S&P 500 Historical Components")
                  and f["name"].endswith(".csv"))
    if not pick:
        raise RuntimeError("fja05680/sp500 파일을 찾지 못했습니다")
    raw = requests.get(pick[-1]["download_url"], timeout=60).text
    df = pd.read_csv(StringIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "tickers"})
    df["date"] = pd.to_datetime(df["date"])
    cache.parent.mkdir(exist_ok=True)
    df[["date", "tickers"]].to_csv(cache, index=False)
    print("S&P 500 snapshots:", pick[-1]["name"], len(df), "rows", df["date"].min().date(), "~", df["date"].max().date())
    return df[["date", "tickers"]]


def snapshot_members(dates: pd.DatetimeIndex, snaps: pd.DataFrame) -> pd.DataFrame:
    """Membership from dated member lists: each day uses the latest list on or before it."""
    snaps = snaps.sort_values("date")
    lists = [{norm(re.sub(r"-\d{6}$", "", x)) for x in str(t).split(",") if x.strip()} for t in snaps["tickers"]]
    names = sorted(set().union(*lists))
    col = {t: j for j, t in enumerate(names)}
    out = np.zeros((len(dates), len(names)), bool)
    starts = [int(dates.searchsorted(d)) for d in snaps["date"]] + [len(dates)]
    for j, members in enumerate(lists):
        a, b = starts[j], starts[j + 1]
        if j == 0:
            a = 0                                            # before the first list: use it
        if b > a:
            out[a:b, [col[t] for t in members]] = True
    return pd.DataFrame(out, index=dates, columns=names)


def snapshot_exits(snaps: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Ticker -> the date of the first list it is missing from after its last appearance."""
    snaps = snaps.sort_values("date").reset_index(drop=True)
    last: dict[str, int] = {}
    for j, t in enumerate(snaps["tickers"]):
        for x in str(t).split(","):
            if x.strip():
                last[norm(re.sub(r"-\d{6}$", "", x))] = j
    end = len(snaps) - 1
    return {t: snaps["date"][j + 1] for t, j in last.items() if j < end}


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
