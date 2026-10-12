import pandas as pd
import pytest

from swing import engine
from swing.pullback_yt import PULLBACK_YT, PullbackP1, PullbackP3

from conftest import make_ohlc

TICKERS = tuple(f"T{i}" for i in range(40)) + ("SPY",)


@pytest.mark.parametrize("cls", PULLBACK_YT[:5])
def test_no_lookahead(cls):
    d1 = make_ohlc(n_days=1500, tickers=TICKERS, seed=7, drift=0.0006)
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 1100
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    e1 = engine.run(cls().prepare(d1), cost_bps=25).equity
    e2 = engine.run(cls().prepare(d2), cost_bps=25).equity
    pd.testing.assert_series_equal(e1.iloc[:k + 1], e2.iloc[:k + 1])


def test_half_at_1r_then_rest_at_2r_and_stop_to_breakeven():
    s = PullbackP1().prepare(make_ohlc(n_days=300, tickers=TICKERS, seed=1))
    s.trade["T1"] = {"S": 95.0, "F": 90.0, "peak": 120.0, "R": 5.0}
    pos = {"ticker": "T1", "entry_price": 100.0, "high_max": 103.0, "stop": 90.0}
    assert s.take_profit(pos, 10) == (105.0, 0.5, "1R 절반 익절")
    assert not pos.get("half_sold")
    pos["high_max"] = 106.0
    s.take_profit(pos, 11)
    assert pos["half_sold"] and s.update_stop(pos, 11) == 100.0
    assert s.take_profit(pos, 12) == (110.0, 1.0, "2R 익절")


def test_entry_rules_skip_wide_stops_and_small_room_to_the_high():
    s = PullbackP3().prepare(make_ohlc(n_days=300, tickers=TICKERS, seed=1))
    s.info["T1"] = (100.0, 90.0, 106.0)             # support, range bottom, first-rise high
    assert not s.accept_entry("T1", 10, 104.0)       # stop 98, R 6, room to high 2 < R -> no
    s.info["T1"] = (100.0, 90.0, 115.0)
    assert s.accept_entry("T1", 10, 104.0)
    assert not s.accept_entry("T1", 10, 108.0)       # stop 98 is 9% away (> 8%)
