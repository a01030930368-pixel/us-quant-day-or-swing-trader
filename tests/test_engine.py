import numpy as np
import pandas as pd

from swing import engine
from swing.strategies import ALL, Strategy

from conftest import make_ohlc


class BuyFirstDay(Strategy):
    """Buys AAA once on the first signal day and never sells unless stopped."""
    max_positions = 1

    def __init__(self, stop=None):
        self.stop = stop
        self.done = False

    def candidates(self, i):
        if self.done:
            return []
        self.done = True
        return ["AAA"]

    def initial_stop(self, t, i, price):
        return self.stop


def flat_data(n=10):
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = pd.DataFrame({"AAA": np.linspace(100, 109, n), "SPY": 100.0}, index=idx)
    return {"Open": c.copy(), "High": c + 0.5, "Low": c - 0.5, "Close": c}


def test_buys_next_open_and_pays_costs():
    d = flat_data()
    s = BuyFirstDay().prepare(d)
    res = engine.run(s, cost_bps=0)
    # Signal at day 0 close, bought at day 1 open (101), worth 109 at the last close.
    assert abs(res.equity.iloc[-1] - 109 / 101) < 1e-9
    res_cost = engine.run(BuyFirstDay().prepare(d), cost_bps=50)
    assert res_cost.equity.iloc[-1] < res.equity.iloc[-1]


def test_stop_fills_at_stop_or_gap_open():
    d = flat_data()
    d["Low"].iloc[3, 0] = 90          # touches a 95 stop intraday on day 3
    res = engine.run(BuyFirstDay(stop=95).prepare(d), cost_bps=0)
    t = res.trades.iloc[0]
    assert t["exit_price"] == 95 and t["reason"] == "손절"

    d = flat_data()
    d["Open"].iloc[3, 0] = 80          # gaps below the stop: fill at the open
    d["Low"].iloc[3, 0] = 79
    res = engine.run(BuyFirstDay(stop=95).prepare(d), cost_bps=0)
    assert res.trades.iloc[0]["exit_price"] == 80


def test_no_lookahead():
    """Changing prices after day k must not change anything up to day k."""
    d1 = make_ohlc(seed=1)
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 500
    for f in d2:
        d2[f].iloc[k + 1:] *= 1.5
    for cls in ALL:
        e1 = engine.run(cls().prepare(d1), cost_bps=20).equity
        e2 = engine.run(cls().prepare(d2), cost_bps=20).equity
        pd.testing.assert_series_equal(e1.iloc[:k + 1], e2.iloc[:k + 1])


def test_position_limits_and_cash(ohlc):
    for cls in ALL:
        s = cls().prepare(ohlc)
        res = engine.run(s, cost_bps=20)
        assert (res.exposure <= 1 + 1e-9).all()
        assert (res.equity > 0).all()
        if len(res.trades):
            # never more than max_positions open at once
            tr = res.trades
            days = pd.to_datetime(pd.concat([tr["entry_date"], tr["exit_date"]])).sort_values().unique()
            for day in days[::20]:
                day = pd.Timestamp(day).date()
                open_n = ((tr["entry_date"] <= day) & (tr["exit_date"] > day)).sum()
                assert open_n <= s.max_positions


def test_bear_exit_sells_everything():
    from swing.strategies import MomentumBearExit
    d = make_ohlc(seed=2)
    s = MomentumBearExit().prepare(d)
    res = engine.run(s, cost_bps=20)
    bear_days = s.market_ok[~s.market_ok].index
    # The day after any bear close, nothing is held at the close.
    for day in bear_days[:-1]:
        nxt = d["Close"].index[d["Close"].index.get_loc(day) + 1]
        assert res.exposure[nxt] < 1e-9


def test_vol_weight_capped():
    from swing.strategies import MomentumBearExitVol
    s = MomentumBearExitVol().prepare(make_ohlc())
    for i in range(30, 300, 37):
        for t in s.tradable:
            assert 0 < s.weight(t, i) <= 0.2 + 1e-12
