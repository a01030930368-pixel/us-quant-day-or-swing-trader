import numpy as np
import pandas as pd
import pytest

from swing import engine
from swing.namsg import NAMSG, NamG1, NamG7, industry_leaders

from conftest import make_ohlc

TICKERS = tuple(f"T{i}" for i in range(40)) + ("SPY",)
INFO = pd.DataFrame({"index": [500] * 30 + [400] * 10,
                     "sector": "", "sub_industry": [f"G{i % 4}" for i in range(40)]},
                    index=list(TICKERS[:-1]))
# random prices never make a +10% candle after a quiet stretch; loosen the numbers so the tests trade
LOOSE = {"big": 0.012, "mover": 0.004, "same": 2, "vol_x": 0.5, "flat": 1.0, "flat_days": 3, "high_days": 5}


def _data(seed=7):
    d = make_ohlc(n_days=1200, tickers=TICKERS, seed=seed, drift=0.0006)
    d["INFO"] = INFO
    d["EarnOK"] = pd.DataFrame(np.random.default_rng(1).random((1200, 40)) > 0.3,
                               index=d["Close"].index, columns=list(TICKERS[:-1]))
    return d


@pytest.mark.parametrize("cls", NAMSG[:9] + [NAMSG[9]])
def test_no_lookahead(cls):
    d1 = _data()
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 900
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    strat = type(cls.__name__ + "Loose", (cls,), LOOSE)
    r1 = engine.run(strat().prepare(d1), cost_bps=30)
    assert len(r1.trades) > 0
    e2 = engine.run(strat().prepare(d2), cost_bps=30).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])


def test_industry_leader_needs_three_movers():
    gain = np.array([[0.12, 0.06, 0.05, 0.20, 0.0]])
    group = np.array([0, 0, 0, 1, 1])
    out = industry_leaders(gain, np.ones((1, 5), bool), group)
    assert out[0].tolist() == [True, False, False, False, False]


def _story():
    """A0..A2 jump together after a flat year; A0 leads and later dips to its 5-day line."""
    idx = pd.bdate_range("2020-01-01", periods=220)
    tick = ["A0", "A1", "A2", "A3", "SPY"]
    close = pd.DataFrame(100.0, index=idx, columns=tick)
    s = 200
    path = [115, 112, 113, 114, 113.5, 112, 111, 103, 103, 103]
    close.iloc[s:s + len(path), 0] = path
    close.iloc[s + len(path):, 0] = 103
    close.iloc[s:, 1] = 107
    close.iloc[s:, 2] = 106
    open_ = close.shift(1).fillna(100.0)
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    low.iloc[s + 1, 0] = 104.0                       # day after the candle dips to the 5-day line
    vol = pd.DataFrame(1e6, index=idx, columns=tick)
    vol.iloc[s, :3] = 5e6
    info = pd.DataFrame({"index": 500, "sector": "", "sub_industry": ["X", "X", "X", "Y"]}, index=tick[:4])
    return {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol, "INFO": info}, s


def test_signal_buy_touch_add_and_ten_day_exit():
    d, s = _story()
    strat = NamG1().prepare(d)
    lim = strat.limit["A0"].iloc[s]                  # yesterday's 5-day EMA = today's touch price
    r = engine.run(strat, cost_bps=0)
    tr = r.trades
    assert tr.ticker.tolist() == ["A0"]
    # 10% at 115, then 20% at the 5-day line the next day
    expect = (0.10 + 0.20) / (0.10 / 115 + 0.20 / lim)
    assert tr.entry_price.iloc[0] == pytest.approx(expect)
    assert tr.weight.iloc[0] == pytest.approx(0.30, rel=1e-3)
    assert tr.reason.iloc[0] == "종가 10일선 이탈"


def test_touch_only_entry():
    d, s = _story()
    strat = NamG7().prepare(d)
    lim = strat.limit["A0"].iloc[s]
    r = engine.run(strat, cost_bps=0)
    assert r.trades.ticker.tolist() == ["A0"]
    # 20% at the 5-day line, then two 10% adds on new closing highs (113, 114) at the next opens
    assert r.trades.weight.iloc[0] == pytest.approx(0.40, rel=1e-2)
    assert lim < r.trades.entry_price.iloc[0] < 110
