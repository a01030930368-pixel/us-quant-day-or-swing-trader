import numpy as np
import pandas as pd
import pytest

from swing import engine
from swing.sewer import SEWER, sewer_signals

from conftest import make_ohlc


def _path(second_vol: float, fall: bool = True):
    idx = pd.bdate_range("2015-01-01", periods=320)
    c = np.full(len(idx), 90.0)
    c[300] = 101.0                                   # first break
    c[301:310] = 95.0 if fall else 103.0             # falls back (or not)
    c[310:] = 102.0                                  # second break
    v = np.full(len(idx), 1000.0)
    v[310] = second_vol
    close = pd.DataFrame({"AAA": c}, index=idx)
    line = pd.DataFrame({"AAA": 100.0}, index=idx)
    return close, line, pd.DataFrame({"AAA": v}, index=idx)


def test_second_break_on_bigger_volume_signals():
    s = sewer_signals(*_path(2500.0))["AAA"]
    assert s.dropna().index.tolist() == [s.index[310]]
    assert s.iloc[310] == pytest.approx(2.5)


def test_no_signal_on_smaller_volume_or_without_failed_first_break():
    assert sewer_signals(*_path(900.0))["AAA"].isna().all()
    assert sewer_signals(*_path(2500.0, fall=False))["AAA"].isna().all()


def test_deep_variant_needs_a_big_drop():
    assert sewer_signals(*_path(2500.0), deep=True)["AAA"].isna().all()   # flat at 90: no 50% drop


@pytest.mark.parametrize("cls", SEWER)
def test_no_lookahead(cls):
    d1 = make_ohlc(n_days=900, seed=11, drift=-0.0002)
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 600
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    e1 = engine.run(cls().prepare(d1), cost_bps=20).equity
    e2 = engine.run(cls().prepare(d2), cost_bps=20).equity
    pd.testing.assert_series_equal(e1.iloc[:k + 1], e2.iloc[:k + 1])
