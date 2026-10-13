import numpy as np
import pandas as pd
import pytest

from swing import engine
from swing.hong import HONG, HongH1, leader_signals

from conftest import make_ohlc

TICKERS = tuple(f"T{i}" for i in range(40)) + ("SPY",)
INFO = pd.DataFrame({"index": [500] * 30 + [600] * 10,
                     "sector": "", "sub_industry": [f"G{i % 4}" for i in range(40)]},
                    index=list(TICKERS[:-1]))
# random prices never make a +10% candle; loosen the numbers so the tests trade
LOOSE = {"big": 0.01, "mover": 0.005, "value_x": 0.8, "same": 2, "top_n": 40, "high_days": 20}


def _data(seed=7):
    d = make_ohlc(n_days=1200, tickers=TICKERS, seed=seed, drift=0.0006)
    d["INFO"] = INFO
    return d


def _loose(cls):
    return type(cls.__name__ + "Loose", (cls,), LOOSE)


@pytest.mark.parametrize("cls", HONG[:12])
def test_no_lookahead(cls):
    d1 = _data()
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 900
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    strat = _loose(cls)
    r1 = engine.run(strat().prepare(d1), cost_bps=30)
    assert len(r1.trades) > 0
    e2 = engine.run(strat().prepare(d2), cost_bps=30).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])


def test_leader_is_the_top_gainer_of_a_crowded_industry():
    n, k = 140, 6
    c = np.full((n, k), 100.0)
    c[-1] = [112, 108, 106, 120, 101, 100]       # stocks 0-2 share industry 0; 3 is alone
    o = c.copy()
    o[-1] = 100.0
    h = c.copy()
    value = np.full((n, k), 1e8)
    value[-1] = [9e8, 8e8, 7e8, 9.5e8, 1e8, 1e8]
    avg50 = np.full((n, k), 1e8)
    liquid = np.ones((n, k), bool)
    group = np.array([0, 0, 0, 1, 2, 2])
    sig = leader_signals(c, o, h, value, avg50, liquid, group)
    # stock 3 gained most but its industry has one mover; stock 0 leads industry 0
    assert np.flatnonzero(~np.isnan(sig[-1])).tolist() == [0]
    # the second stock is never bought, even if the leader is not a fresh high
    h2 = h.copy()
    h2[100, 0] = 200.0
    sig2 = leader_signals(c, o, h2, value, avg50, liquid, group)
    assert np.isnan(sig2[-1]).all()


def test_three_day_exits():
    idx = pd.bdate_range("2020-01-01", periods=200)
    tick = [f"A{i}" for i in range(4)] + ["SPY"]
    close = pd.DataFrame(100.0, index=idx, columns=tick)
    s = 180                                       # signal day: A0..A2 jump, A0 the most
    close.iloc[s, :3] = [115.0, 108.0, 107.0]
    close.iloc[s + 1:, :3] = [115.0, 108.0, 107.0]
    open_ = close.shift(1).fillna(100.0)
    open_.iloc[s, :3] = 100.0
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    vol = pd.DataFrame(1e6, index=idx, columns=tick)
    vol.iloc[s, :3] = 5e6
    # day 2: no gap, the high reaches +5% of the day-1 close
    high.iloc[s + 1, 0] = 115 * 1.06
    d = {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol,
         "INFO": pd.DataFrame({"index": 500, "sector": "", "sub_industry": ["X", "X", "X", "Y"]},
                              index=tick[:4])}
    r = engine.run(HongH1().prepare(d), cost_bps=0)
    tr = r.trades
    assert tr.ticker.tolist() == ["A0"]
    assert tr.entry_price.iloc[0] == 115.0 and tr.reason.iloc[0] == "2일차 익절"
    assert tr.exit_price.iloc[0] == pytest.approx(115 * 1.05)
    # no target, no gap: sold at the day-3 close
    high.iloc[s + 1, 0] = 115.1
    r = engine.run(HongH1().prepare(d), cost_bps=0)
    assert r.trades.reason.iloc[0] == "3일차 종가 정리" and r.trades.exit_price.iloc[0] == 115.0
    # gap down more than 1% on day 2: sold at that open
    open_.iloc[s + 1, 0] = 113.0
    low.iloc[s + 1, 0] = 112.9
    r = engine.run(HongH1().prepare(d), cost_bps=0)
    assert r.trades.reason.iloc[0] == "2일차 갭 하락 (시가 매도)" and r.trades.exit_price.iloc[0] == 113.0
