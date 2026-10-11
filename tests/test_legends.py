import numpy as np
import pandas as pd
import pytest

from swing import engine
from swing.legends import LEGENDS, Bonde, ONeil, Strategy, weekly

from conftest import make_ohlc


@pytest.mark.parametrize("cls", LEGENDS)
def test_no_lookahead(cls):
    """Changing prices after day k must not change anything up to day k."""
    d1 = make_ohlc(n_days=1100, seed=7, drift=0.0012)
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 800
    for f in d2:
        d2[f].iloc[k + 1:] *= 1.5
    e1 = engine.run(cls().prepare(d1), cost_bps=20).equity
    e2 = engine.run(cls().prepare(d2), cost_bps=20).equity
    pd.testing.assert_series_equal(e1.iloc[:k + 1], e2.iloc[:k + 1])


@pytest.mark.parametrize("cls", LEGENDS)
def test_limits(cls):
    d = make_ohlc(n_days=1100, seed=8, drift=0.0012)
    s = cls().prepare(d)
    res = engine.run(s, cost_bps=20)
    assert (res.exposure <= 1 + 1e-9).all()
    assert (res.equity > 0).all()
    for t in s.tradable:
        for i in range(300, 1000, 97):
            assert 0 < s.weight(t, i) <= 0.25 + 1e-12


class HalfAt10(Strategy):
    """Buys AAA once; sells half at +10%."""
    max_positions = 1

    def __init__(self):
        self.done = False

    def candidates(self, i):
        if self.done:
            return []
        self.done = True
        return ["AAA"]

    def take_profit(self, pos, i):
        if pos.get("half"):
            return None
        target = pos["entry_price"] * 1.10
        if pos["high_max"] >= target:
            pos["half"] = True
        return target, 0.5, "half"


def test_partial_take_profit():
    idx = pd.bdate_range("2020-01-01", periods=6)
    c = pd.DataFrame({"AAA": [100, 100, 105, 112, 112, 112.0], "SPY": 100.0}, index=idx)
    d = {"Open": c.copy(), "High": c + 0.5, "Low": c - 0.5, "Close": c, "Volume": c * 0 + 1e6}
    res = engine.run(HalfAt10().prepare(d), cost_bps=0)
    t = res.trades.iloc[0]
    # Bought at day 1 open (100); day 3 opens at 112 above the 110 target -> fill at the open.
    assert t["exit_price"] == 112 and t["part"] == 0.5 and t["reason"] == "half"
    assert len(res.trades) == 1                    # the other half is still held, sold once only
    assert abs(res.equity.iloc[-1] - (0.5 * 112 / 100 + 0.5 * 112 / 100)) < 1e-9


def test_weekly_bars_use_only_their_week():
    d = make_ohlc(n_days=30)
    w = weekly(d, d["Close"].index)
    assert all(day.dayofweek == 4 or day == d["Close"].index[-1] for day in w["Close"].index)
    first = w["Close"].index[0]
    assert w["High"]["AAA"].iloc[0] == d["High"]["AAA"].loc[:first].max()


def test_bonde_signal():
    d = make_ohlc(n_days=400, seed=3)
    t, k = "AAA", 300
    base = d["Close"][t].iloc[k - 12]
    for f in ("Open", "High", "Low", "Close"):           # quiet, flat 12 days
        d[f].iloc[k - 12:k, 0] = base
    d["High"].iloc[k - 12:k, 0] = base * 1.01
    d["Low"].iloc[k - 12:k, 0] = base * 0.99
    d["Close"].iloc[k - 1, 0] = base * 0.995            # small down day before
    d["Open"].iloc[k, 0] = base
    d["Close"].iloc[k, 0] = base * 1.06
    d["High"].iloc[k, 0] = base * 1.065
    d["Low"].iloc[k, 0] = base * 0.995
    d["Volume"].iloc[k, 0] = d["Volume"][t].iloc[k - 1] * 3
    s = Bonde().prepare(d)
    assert bool(s.signal[t].iloc[k]) or not s._breadth().iloc[k]
    s2 = Bonde()
    s2.use_breadth = False
    s2.prepare(d)
    assert bool(s2.signal[t].iloc[k]) == bool(s2.market_ok.iloc[k])
    assert s2.initial_stop(t, k + 1, base * 1.06) == base * 0.995


def test_oneil_eight_week_rule():
    s = ONeil()
    pos = {"entry_price": 100.0, "entry_i": 10, "stop": 92.5, "high_max": 121.0}
    assert s.take_profit(pos, 15) is None and pos["hold8"] and pos["stop"] == 100.0
    slow = {"entry_price": 100.0, "entry_i": 10, "stop": 92.5, "high_max": 121.0}
    assert s.take_profit(slow, 40)[0] == pytest.approx(120.0)


from swing.legends2 import LEGENDS2  # noqa: E402


@pytest.mark.parametrize("cls", LEGENDS2)
def test_group2_no_lookahead_and_limits(cls):
    d1 = make_ohlc(n_days=1100, seed=7, drift=0.0012)
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 800
    for f in d2:
        d2[f].iloc[k + 1:] *= 1.5
    r1 = engine.run(cls().prepare(d1), cost_bps=20)
    e2 = engine.run(cls().prepare(d2), cost_bps=20).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])
    assert (r1.exposure <= 1 + 1e-9).all() and (r1.equity > 0).all()


class BuyStopOnce(Strategy):
    """Wants AAA with a buy-stop at 103; sells a third at the next open after day 2."""
    max_positions = 1

    def __init__(self):
        self.done = False

    def candidates(self, i):
        if self.done:
            return []
        self.done = True
        return ["AAA"]

    def entry_fill(self, t, i, open_, high):
        return max(open_, 103.0) if high >= 103 else None

    def exit_reason(self, pos, i):
        if i - pos["entry_i"] == 1 and not pos.get("sold"):
            pos["sold"] = True
            return ("third", 1 / 3)
        return None


def test_buy_stop_fill_and_partial_at_open():
    idx = pd.bdate_range("2020-01-01", periods=6)
    c = pd.DataFrame({"AAA": [100, 101, 104, 106, 106, 106.0], "SPY": 100.0}, index=idx)
    d = {"Open": c.shift(1).fillna(100), "High": c + 0.5, "Low": c.shift(1).fillna(100) - 0.5,
         "Close": c, "Volume": c * 0 + 1e6}
    res = engine.run(BuyStopOnce().prepare(d), cost_bps=0)
    # Day 1 (open 100, high 101.5) never reaches 103: no trade, and the order is not repeated.
    assert len(res.trades) == 0 and res.open_positions == []
    d["High"].iloc[1, 0] = 104
    res = engine.run(BuyStopOnce().prepare(d), cost_bps=0)
    t = res.trades.iloc[0]
    assert t["entry_price"] == 103 and t["exit_price"] == d["Open"]["AAA"].iloc[3]
    assert abs(t["part"] - 1 / 3) < 1e-12 and len(res.open_positions) == 1
