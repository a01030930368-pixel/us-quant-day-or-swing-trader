import pandas as pd
import pytest

from swing import engine, gojiro
from swing.gojiro import GOJIRO, GojiroK5, GojiroK7, stage6_signals

from conftest import make_ohlc

TICKERS = tuple(f"T{i}" for i in range(40)) + ("SPY",)


def _check(cls):
    d1 = make_ohlc(n_days=1500, tickers=TICKERS, seed=7, drift=0.0006)
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 1100
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    r1 = engine.run(cls().prepare(d1), cost_bps=25)
    assert len(r1.trades) > 0
    e2 = engine.run(cls().prepare(d2), cost_bps=25).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])


@pytest.mark.parametrize("cls", [c for c in GOJIRO[:8] if not c.qulla])
def test_no_lookahead(cls):
    _check(cls)


@pytest.mark.parametrize("cls", [GojiroK5, GojiroK7])
def test_no_lookahead_qullamaggie(cls, monkeypatch):
    # random prices rarely make a leader's quiet base; loosen it so the test trades
    monkeypatch.setattr(gojiro, "Q_TOP", 0.7)
    monkeypatch.setattr(gojiro, "Q_MIN_RUN", 0.0)
    monkeypatch.setattr(gojiro, "Q_DRY", 2.0)
    monkeypatch.setattr(gojiro, "Q_LOUD", 0.0)
    _check(cls)


def test_stage6_needs_the_cross_order_and_a_long_fall():
    idx = pd.RangeIndex(70)
    one = lambda vals: pd.DataFrame({"X": vals}, index=idx, dtype=float)
    s = one([90.0] * 60 + [93, 95, 96, 97, 99, 100, 101, 102, 103, 104])
    m = one([95.0] * 55 + [94.0] * 15)                      # 20-day line, rising at the end
    m.iloc[64:, 0] = [95.0, 95.5, 96, 96.5, 97, 97.5]
    L = one([98.0] * 70)                                     # flat 40-day line
    c = s + 1
    o = c - 1
    sig = stage6_signals(o, c, s, m, L)["X"]
    # stage 4 (98 > 95 > 90) for 60 days; 5-day line crosses 20 at day 61... then 40 on day 64
    assert sig.iloc[64] and sig.sum() == 1
