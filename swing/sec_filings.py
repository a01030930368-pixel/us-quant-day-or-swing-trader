"""Every SEC filing of a company with its acceptance time (EDGAR submissions data), cached on disk.

Rulebooks 12 and 13 need earnings release days (8-K item 2.02, or item 12
before August 2004 when the item was renumbered). Rulebook 14 needs every
8-K, 8-K/A, 10-Q and 10-K as a stand-in for "company news".

Times: the SEC's acceptanceDateTime is New York time although it is labelled
UTC (see announce.py). A filing accepted before 9:30 is known at that day's
open; one accepted later is known at the next day's open.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .announce import MORE_URL, SUBMISSIONS_URL
from .fundamentals import CACHE_DIR, _get, cik_of

NEWS_FORMS = ("8-K", "8-K/A", "10-Q", "10-K", "10-Q/A", "10-K/A")
ITEM12_UNTIL = "2004-08-23"      # item 12 "Results of Operations" became item 2.02 on this day


def _rows(block: dict) -> list[list[str]]:
    forms, items = block.get("form", []), block.get("items", [])
    acc = block.get("acceptanceDateTime", [])
    return [[f, it or "", a] for f, it, a in zip(forms, items, acc) if f in NEWS_FORMS]


def load_submissions(tickers: list[str], since: str = "2003-01-01",
                     use_cache: bool = True) -> dict[str, list[list[str]]]:
    """Ticker -> [[form, items, acceptance time], ...] for the forms in NEWS_FORMS since `since`."""
    cik = cik_of(use_cache)
    folder = CACHE_DIR / "filings"
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in tickers:
        c = cik(t)
        if c is None:
            out[t] = []
            continue
        path = folder / f"{c}_{since}.json"
        if use_cache and path.exists():
            out[t] = json.loads(path.read_text())
            continue
        sub = _get(SUBMISSIONS_URL.format(cik=c))
        rows = _rows(sub.get("filings", {}).get("recent", {}))
        for f in sub.get("filings", {}).get("files", []):
            if f.get("filingTo", "9999") >= since:
                time.sleep(0.12)
                rows += _rows(_get(MORE_URL.format(name=f["name"])))
        rows = sorted({tuple(r) for r in rows if r[2][:10] >= since}, key=lambda r: r[2])
        out[t] = [list(r) for r in rows]
        path.write_text(json.dumps(out[t]))
        time.sleep(0.12)                     # SEC limit: at most 10 requests a second
    return out


def is_earnings(form: str, items: str, accepted: str) -> bool:
    if form != "8-K":
        return False
    parts = [x.strip() for x in items.split(",")]
    return "2.02" in parts or (accepted[:10] < ITEM12_UNTIL and "12" in parts)


def _times(accepted: list[str]) -> pd.DatetimeIndex:
    return pd.to_datetime([a[:19] for a in accepted])


def reaction_index(accepted: list[str], index: pd.DatetimeIndex) -> np.ndarray:
    """Position of the first close after each release (after 16:00 = the next day), unique and sorted."""
    if not accepted:
        return np.array([], int)
    ts = _times(accepted)
    after = (ts.hour * 60 + ts.minute) >= 16 * 60
    day = ts.normalize()
    pos = np.where(after, index.searchsorted(day, side="right"), index.searchsorted(day, side="left"))
    return np.unique(pos[pos < len(index)])


def known_index(accepted: list[str], index: pd.DatetimeIndex) -> np.ndarray:
    """Position of the first open after each filing (before 9:30 = that day, else the next day)."""
    if not accepted:
        return np.array([], int)
    ts = _times(accepted)
    pre = (ts.hour * 60 + ts.minute) < 9 * 60 + 30
    day = ts.normalize()
    pos = np.where(pre, index.searchsorted(day, side="left"), index.searchsorted(day, side="right"))
    return np.unique(pos[pos < len(index)])


def earnings_days(subs: dict[str, list], columns: list[str], index: pd.DatetimeIndex) -> dict[int, np.ndarray]:
    """Column -> reaction days of its earnings releases."""
    out = {}
    for k, t in enumerate(columns):
        acc = [a for f, it, a in subs.get(t, []) if is_earnings(f, it, a)]
        out[k] = reaction_index(acc, index)
    return out


def news_days(subs: dict[str, list], columns: list[str], index: pd.DatetimeIndex) -> dict[int, np.ndarray]:
    """Column -> first open after each 8-K/10-Q/10-K filing."""
    return {k: known_index([a for _, _, a in subs.get(t, [])], index) for k, t in enumerate(columns)}
