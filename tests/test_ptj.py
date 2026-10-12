import pandas as pd
import pytest

from swing import engine
from swing.ptj import PTJ, PTJ1

from conftest import make_ohlc

TICKERS = tuple(f"T{i}" for i in range(40)) + ("SPY",)
NO_EARNINGS = [c for c in PTJ if c.name != "ptj_j2"]


@pytest.mark.parametrize("cls", NO_EARNINGS[:4])
def test_no_lookahead(cls):
    d1 = make_ohlc(n_days=1500, tickers=TICKERS, seed=7, drift=0.0006)
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 1100
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    e1 = engine.run(cls().prepare(d1), cost_bps=25).equity
    e2 = engine.run(cls().prepare(d2), cost_bps=25).equity
    pd.testing.assert_series_equal(e1.iloc[:k + 1], e2.iloc[:k + 1])


class _AddOnce(PTJ1):
    """Buys T1 on day 300 and adds 10% of the account the next day."""
    max_positions = 1

    def candidates(self, i):
        self.pending = {"T1": {"stop": 1.0, "sig_high": 1e9}}
        return ["T1"] if i == 299 else []

    def weight(self, t, i):
        return 0.1

    def accept_entry(self, t, i, price):
        self.trade[t] = {"stop": 1.0, "sig_high": 1e9, "R": price - 1, "first": price,
                         "touched5": False, "under": 0, "half": False, "planned": 0.3}
        return True

    def exit_reason(self, pos, i):
        return "끝" if i == 320 else None

    def add_weight(self, pos, i):
        return 0.1 if pos.get("adds", 0) == 0 else None


def test_engine_adds_at_next_open_and_averages_the_cost():
    d = make_ohlc(n_days=400, tickers=TICKERS, seed=3)
    r = engine.run(_AddOnce().prepare(d), cost_bps=0)
    o = d["Open"]["T1"]
    tr = r.trades.iloc[0]
    a, b = 0.1, 0.1 * r.equity.iloc[300]                    # dollars of the buy and of the add
    avg = (a + b) / (a / o.iloc[300] + b / o.iloc[301])
    assert tr["entry_price"] == pytest.approx(avg)
    assert tr["weight"] == pytest.approx(0.2, rel=0.05)
