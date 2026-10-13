import numpy as np
import pandas as pd
import pytest

from swing import engine, jongga
from swing.jongga import A1, E2, JONGGA, leaders

from conftest import make_ohlc

TICKERS = tuple(f"T{i}" for i in range(40)) + ("SPY",)
INFO = pd.DataFrame({"index": [500] * 30 + [400] * 10,
                     "sector": "", "sub_industry": [f"G{i % 4}" for i in range(40)]},
                    index=list(TICKERS[:-1]))
# random prices never jump +15%; loosen the numbers so the tests trade
LOOSE = {"limit_up": 0.012, "vol_x": 0.3, "d_jump": 0.02}
MODULE = {"MOVER": 0.002, "SAME": 2, "B_BIG": 0.012, "B_DRY": 5.0, "B_LOUD": 0.5, "VOL_X": 0.3,
          "C_FLAT": 1.0, "C_NEAR": 0.05, "C_WICK": 0.6, "D_RANGE": 5.0, "D_NEAR": 0.05,
          "E_RED": 0.003, "E_PAUSE": 0.02, "MIN_DOLLAR_VOL": 0.0}


def _data(seed=7):
    d = make_ohlc(n_days=1200, tickers=TICKERS, seed=seed, drift=0.0006)
    d["INFO"] = INFO
    return d


@pytest.mark.parametrize("cls", JONGGA[:17])
def test_no_lookahead(cls, monkeypatch):
    for k, v in MODULE.items():
        monkeypatch.setattr(jongga, k, v)
    strat = type(cls.__name__ + "Loose", (cls,), {**LOOSE, "b_dry": 5.0})
    d1 = _data()
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 900
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    r1 = engine.run(strat().prepare(d1), cost_bps=30)
    assert len(r1.trades) > 0
    e2 = engine.run(strat().prepare(d2), cost_bps=30).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])


def test_leaders_need_a_crowd():
    gain = np.array([[0.20, 0.10, 0.06, 0.30, 0.0]])
    liquid = np.ones((1, 5), bool)
    group = np.array([0, 0, 0, 1, 1])
    lead, second = leaders(gain, liquid, group)
    assert lead[0].tolist() == [True, False, False, False, False]
    assert second[0, 0] == pytest.approx(0.10)


def _story(tick=("A0", "A1", "A2", "SPY"), n=200):
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame(100.0, index=idx, columns=list(tick))
    open_ = close.copy()
    high, low = close + 0.5, close - 0.5
    vol = pd.DataFrame(1e6, index=idx, columns=list(tick))
    return idx, open_, high, low, close, vol


def _dict(o, h, l, c, v):
    tick = [t for t in c.columns if t != "SPY"]
    return {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v,
            "INFO": pd.DataFrame({"index": 500, "sector": "", "sub_industry": "X"}, index=tick)}


def test_a_limit_fills_and_takes_profit():
    idx, o, h, l, c, v = _story()
    s = 150                                    # all three jump; A0 leads with +20%
    c.iloc[s:, :3] = [120.0, 110.0, 106.0]
    h.iloc[s, :3] = c.iloc[s, :3] + 0.1
    v.iloc[s, :3] = 5e6
    o.iloc[s + 1:, 0] = h.iloc[s + 1:, 0] = l.iloc[s + 1:, 0] = c.iloc[s + 1:, 0] = 120.0
    # day s+1: dips to 113 (L*0.95 = 114 fills), day s+2: high 118 (+3% over 114 = 117.42)
    o.iloc[s + 1, 0], l.iloc[s + 1, 0], h.iloc[s + 1, 0], c.iloc[s + 1, 0] = 118.0, 113.0, 118.5, 115.0
    o.iloc[s + 2, 0], l.iloc[s + 2, 0], h.iloc[s + 2, 0], c.iloc[s + 2, 0] = 115.0, 114.5, 118.0, 116.0
    o.iloc[s + 3:, 0] = l.iloc[s + 3:, 0] = h.iloc[s + 3:, 0] = c.iloc[s + 3:, 0] = 116.0
    r = engine.run(A1().prepare(_dict(o, h, l, c, v)), cost_bps=0)
    tr = r.trades
    assert tr.ticker.tolist()[0] == "A0"
    assert tr.entry_price.iloc[0] == pytest.approx(114.0)
    assert tr.reason.iloc[0].startswith("1차 익절")
    assert tr.exit_price.iloc[0] == pytest.approx(114.0 * 1.03)


def test_e2_buys_the_red_close_adds_on_the_pause_and_sells_next_open():
    idx, o, h, l, c, v = _story()
    s = 150
    c.iloc[s:, 0] = 120.0
    o.iloc[s, 0], h.iloc[s, 0], v.iloc[s, 0] = 100.0, 120.5, 5e6
    o.iloc[s + 1, 0], h.iloc[s + 1, 0], l.iloc[s + 1, 0], c.iloc[s + 1, 0] = 120.0, 120.5, 112.0, 113.0
    o.iloc[s + 2, 0], h.iloc[s + 2, 0], l.iloc[s + 2, 0], c.iloc[s + 2, 0] = 113.0, 115.0, 112.5, 114.0
    o.iloc[s + 3:, 0] = h.iloc[s + 3:, 0] = l.iloc[s + 3:, 0] = c.iloc[s + 3:, 0] = 115.0
    r = engine.run(E2().prepare(_dict(o, h, l, c, v)), cost_bps=0)
    tr = r.trades
    assert len(tr) == 1 and tr.ticker.iloc[0] == "A0"
    assert tr.entry_price.iloc[0] == pytest.approx(2 / (1 / 113.0 + 1 / 114.0))   # equal money, share-weighted
    assert tr.exit_price.iloc[0] == 115.0 and tr.reason.iloc[0] == "멈춤 다음 날 시가 매도"


def test_same_object_runs_twice_the_same(monkeypatch):
    for k, v in MODULE.items():
        monkeypatch.setattr(jongga, k, v)
    strat = type("M1Loose", (JONGGA[15],), {**LOOSE, "b_dry": 5.0})().prepare(_data())
    r1 = engine.run(strat, cost_bps=30)
    r2 = engine.run(strat, cost_bps=30)
    assert len(r1.trades) > 0
    pd.testing.assert_series_equal(r1.equity, r2.equity)
