import pandas as pd
import pytest

from swing import engine
from swing.reversion import REVERSION, ThreeLowerLows

from conftest import make_ohlc


@pytest.mark.parametrize("cls", REVERSION)
def test_no_lookahead(cls):
    d1 = make_ohlc(n_days=900, seed=11, drift=0.0008)
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 600
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    e1 = engine.run(cls().prepare(d1), cost_bps=20).equity
    e2 = engine.run(cls().prepare(d2), cost_bps=20).equity
    pd.testing.assert_series_equal(e1.iloc[:k + 1], e2.iloc[:k + 1])


def test_limit_entry_fills_only_when_low_reaches_it():
    d = make_ohlc(n_days=300, seed=2)
    s = ThreeLowerLows().prepare(d)
    i, t = 250, "AAA"
    limit = s.limit_price(t, i - 1)
    s.l.loc[s.l.index[i], t] = limit * 1.01
    assert s.entry_fill(t, i, limit * 1.02, limit * 1.05) is None        # never traded down to it
    s.l.loc[s.l.index[i], t] = limit * 0.99
    assert s.entry_fill(t, i, limit * 1.02, limit * 1.05) == pytest.approx(limit)
    assert s.entry_fill(t, i, limit * 0.97, limit * 1.05) == pytest.approx(limit * 0.97)  # gap below


def test_reversion_strategies_trade():
    d = make_ohlc(n_days=900, seed=5, drift=0.0008)
    for cls in REVERSION:
        r = engine.run(cls().prepare(d), cost_bps=20)
        assert len(r.trades) > 0, cls.name
