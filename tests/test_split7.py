import numpy as np
import pandas as pd
import pytest

from swing import engine, split7
from swing.split7 import E1, F1, SPLIT

from conftest import make_ohlc

TICKERS = tuple(f"T{i}" for i in range(40)) + ("SPY",)
INFO = pd.DataFrame({"index": [500] * 30 + [400] * 10, "sector": "", "sub_industry": ""},
                    index=list(TICKERS[:-1]))
LOOSE = {"env_pct": 0.02, "near": 0.02}


def _data(seed=7):
    d = make_ohlc(n_days=1200, tickers=TICKERS, seed=seed, drift=0.0003)
    d["INFO"] = INFO
    return d


@pytest.mark.parametrize("cls", SPLIT[:14])
def test_no_lookahead(cls, monkeypatch):
    monkeypatch.setattr(split7, "MIN_DOLLAR_VOL", 0.0)
    strat = type(cls.__name__ + "Loose", (cls,), LOOSE)
    d1 = _data()
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 900
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    r1 = engine.run(strat().prepare(d1), cost_bps=30)
    assert len(r1.trades) > 0
    e2 = engine.run(strat().prepare(d2), cost_bps=30).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])


def test_point_in_time_members_only():
    d = _data()
    d["MEMBER"] = pd.DataFrame(False, index=d["Close"].index, columns=list(TICKERS[:-1]))
    d["MEMBER"]["T3"] = True
    strat = type("F1Loose", (F1,), LOOSE)().prepare(d)
    tr = engine.run(strat, cost_bps=30).trades
    assert len(tr) > 0 and set(tr.ticker) == {"T3"}


def _story(path):
    idx = pd.bdate_range("2020-01-01", periods=len(path))
    close = pd.DataFrame({"A0": path, "SPY": 100.0}, index=idx)
    open_ = close.shift(1).fillna(close)
    high = np.maximum(open_, close) + 0.01
    low = np.minimum(open_, close) - 0.01
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    return {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol,
            "INFO": pd.DataFrame({"index": [500], "sector": "", "sub_industry": ""}, index=["A0"])}


def test_seven_split_lots_are_sold_one_by_one():
    # 60 flat days, then a slide to -25% from the first touch, then a recovery
    path = np.r_[np.full(60, 100.0), np.linspace(100, 79, 5), np.linspace(78, 60, 10), np.linspace(61, 95, 25)]
    d = _story(path)
    r = engine.run(F1().prepare(d), cost_bps=0)
    tr = r.trades
    assert len(tr) >= 3
    first = tr.entry_price.max()
    # each lot keeps its own price and is sold with a profit
    assert (tr["return"] > 0).all()
    assert tr.entry_price.min() <= first * 0.80 + 1e-9
    assert tr.reason.str.contains("단기 묶음").any()


def test_envelope_e1_averages_and_takes_half_at_7pct():
    path = np.r_[np.full(60, 100.0), np.linspace(100, 74, 5), np.full(3, 74.0), np.linspace(74, 88, 6), np.full(5, 88.0)]
    d = _story(path)
    tr = engine.run(E1().prepare(d), cost_bps=0).trades
    assert tr.reason.iloc[0] == "+7% 절반 익절"
    assert tr.part.iloc[0] == pytest.approx(0.5)
    assert tr.exit_price.iloc[0] == pytest.approx(tr.entry_price.iloc[0] * 1.07, rel=1e-6)
