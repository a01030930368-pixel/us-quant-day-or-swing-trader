import numpy as np
import pandas as pd
import pytest

from swing import engine
from swing.chartdoc import CHARTDOC, ChartDocS1, _signals_one

from conftest import make_ohlc

TICKERS = tuple(f"T{i}" for i in range(40)) + ("SPY",)


def _loose(cls):
    """Random prices rarely pass every rule; loosen a few numbers so the test makes trades."""
    return type(cls.__name__, (cls,), {"big_vol": 1.0, "pull_vol": 5.0, "b7": False,
                                        "envelope": 0.02, "use_market": False})


@pytest.mark.parametrize("cls", [_loose(c) for c in CHARTDOC[:7]])
def test_no_lookahead(cls):
    d1 = make_ohlc(n_days=1500, tickers=TICKERS, seed=7, drift=0.0006)
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 1100
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    r1 = engine.run(cls().prepare(d1), cost_bps=25)
    e1 = r1.equity
    assert len(r1.trades) > 0
    e2 = engine.run(cls().prepare(d2), cost_bps=25).equity
    pd.testing.assert_series_equal(e1.iloc[:k + 1], e2.iloc[:k + 1])


def _path():
    """New high on day 1, a quiet drift to the 20-day line, then a candle above the 5-day line."""
    n = 6
    c = np.array([100, 110, 108, 106, 105, 108.0])
    o = np.array([100, 101, 109, 107, 106, 106.0])
    h, l = c + 1, np.array([99, 100, 107, 104, 103.5, 105.5])
    v = np.array([1, 10, 3, 2, 2, 3.0])
    s5 = np.array([100, 101, 108, 107, 106, 105.5])
    s20 = np.array([90, 95, 100, 101, 103, 104.0])
    s20[0] = 89
    avg50, vol5 = np.full(n, 2.0), np.full(n, 2.5)
    nh = np.array([False, True, False, False, False, False])
    start = nh.copy()
    old = np.full(n, 104.0)
    liquid = np.ones(n, bool)
    return o, h, l, c, v, s5, s20, avg50, vol5, nh, start, old, liquid


P = {"watch": 30, "quiet": False, "b6": "body", "b7": True, "pull_vol": 0.4}


def test_signal_after_quiet_pullback():
    out = _signals_one(*_path(), P)
    assert np.isnan(out[:5]).all() and out[5] == pytest.approx(5.0)


def test_no_signal_when_pullback_falls_under_the_old_high():
    args = list(_path())
    args[11] = np.full(6, 105.5)                 # old high above the pullback's lowest close
    assert np.isnan(_signals_one(*args, P)).all()


def test_no_signal_on_heavy_pullback_volume():
    args = list(_path())
    args[4] = np.array([1, 10, 6, 5, 5, 6.0])  # pullback averages 53% of the new-high day
    assert np.isnan(_signals_one(*args, P)).all()


def test_half_sale_moves_stop_to_entry():
    s = ChartDocS1()
    s.half_at = 0.10
    s.h = pd.DataFrame({"T": [100.0, 112.0]})
    pos = {"ticker": "T", "entry_price": 100.0, "stop": 95.0}
    assert s.take_profit(pos, 1)[:2] == (pytest.approx(110.0), 0.5)
    assert pos["stop"] == 100.0 and pos["half"] and s.take_profit(pos, 1) is None
