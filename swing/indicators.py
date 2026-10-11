"""Technical indicators on wide frames (index=date, columns=tickers)."""
from __future__ import annotations

import pandas as pd


def sma(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close.rolling(n, min_periods=n).mean()


def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder's RSI. 0 = only falls lately, 100 = only rises."""
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = up / down
    out = 100 - 100 / (1 + rs)
    return out.where(down > 0, 100.0).where(up.notna())


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Average true range: the typical daily move in dollars."""
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()
    tr = tr.reindex(close.index)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
