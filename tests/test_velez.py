import numpy as np
import pandas as pd
import pytest

from swing import engine
from swing.velez import VELEZ, monthly_top, setups

from conftest import make_ohlc

TICKERS = tuple(f"T{i}" for i in range(40)) + ("SPY",)


@pytest.mark.parametrize("cls", VELEZ[:6])
def test_no_lookahead(cls):
    d1 = make_ohlc(n_days=1500, tickers=TICKERS, seed=7, drift=0.0006)
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 1100
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    r1 = engine.run(cls().prepare(d1), cost_bps=25)
    assert len(r1.trades) > 0
    e2 = engine.run(cls().prepare(d2), cost_bps=25).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])


def _bars(rows):
    """rows of (open, high, low, close) after 20 flat days at 100."""
    flat = [(100, 100.5, 99.5, 100)] * 20
    o, h, l, c = (pd.Series([r[j] for r in flat + rows], dtype=float) for j in range(4))
    return o, h, l, c


RUN = [(110, 120, 109, 119)]                                  # +19% to a 20-day high
REDS = [(118, 118.5, 115, 116), (116, 117, 113, 114), (114, 115, 111, 112)]


def test_three_red_lower_highs_arm_a_buy_stop_above_the_last_high():
    out = setups(*_bars(RUN + REDS), rise=0.15, n_red=3, loose=False, use_peak=True)
    assert np.isnan(out[:23]).all() and out[23] == pytest.approx(115.01)


def test_one_pause_candle_keeps_the_order_then_the_count_restarts():
    pause = [(112.5, 113.5, 112, 113)]                       # green, lower high: one more try
    out = setups(*_bars(RUN + REDS + pause + pause), rise=0.15, n_red=3, loose=False, use_peak=True)
    assert out[24] == pytest.approx(113.51) and np.isnan(out[25])


def test_no_setup_without_the_run_up():
    o, h, l, c = _bars([(100, 101, 99.5, 100.8)] + REDS)
    out = setups(o, h - 15, l - 15, c - 15, rise=0.15, n_red=3, loose=False, use_peak=True)
    assert np.isnan(out).all()


def test_monthly_top_lists_the_first_half_gainers_until_month_end():
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    c = pd.DataFrame({"A": 100.0, "B": 100.0, "C": 100.0}, index=idx)
    feb = (idx.month == 2)
    c.loc[feb & (idx.day >= 5), "B"] = 130.0                 # B leads Feb 1-15
    c.loc[feb & (idx.day >= 5), "C"] = 110.0
    ok = pd.DataFrame(True, index=idx, columns=c.columns)
    listed, picked = monthly_top(c, ok, n=1)
    assert picked.loc["2024-02-15", "B"] and listed.loc["2024-02-29", "B"]
    assert not listed.loc["2024-02-14", "B"] and not listed.loc["2024-03-01", "B"]
