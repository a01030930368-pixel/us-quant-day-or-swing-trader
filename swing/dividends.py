"""Ex-dividend dates and amounts from Yahoo Finance, with closes not adjusted for dividends.

Yahoo's adjusted prices (used everywhere else) shrink old prices by past
dividends, which would overstate an old dividend yield; the yield here uses
the split-adjusted close without the dividend adjustment.
Only stocks that still trade are covered (Yahoo drops delisted tickers).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data_cache")


def load_dividends(tickers: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(dividends per share on each ex-date, else 0; split-adjusted close without dividend adjustment)."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.sha1(",".join(sorted(tickers)).encode()).hexdigest()[:10]
    pd_path, pc_path = CACHE_DIR / f"div_{key}_{start}.csv", CACHE_DIR / f"rawclose_{key}_{start}.csv"
    if pd_path.exists() and pc_path.exists():
        return (pd.read_csv(pd_path, index_col=0, parse_dates=True),
                pd.read_csv(pc_path, index_col=0, parse_dates=True))
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=False, actions=True, progress=False,
                      group_by="column", threads=True)
    div = raw["Dividends"].reindex(columns=tickers).fillna(0.0)
    close = raw["Close"].reindex(columns=tickers)
    div.to_csv(pd_path)
    close.to_csv(pc_path)
    return div, close
