import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_ohlc(n_days: int = 700, tickers=("AAA", "BBB", "CCC", "DDD", "EEE", "SPY"),
              seed: int = 0, drift: float = 0.0006) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    rets = rng.normal(drift, 0.015, size=(n_days, len(tickers)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=list(tickers))
    gap = pd.DataFrame(rng.normal(0, 0.004, size=close.shape), index=idx, columns=close.columns)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + gap)
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, size=close.shape))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, size=close.shape))
    volume = pd.DataFrame(rng.lognormal(13, 0.5, size=close.shape), index=idx, columns=close.columns)
    return {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}


@pytest.fixture
def ohlc():
    return make_ohlc()
