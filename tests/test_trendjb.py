import numpy as np
import pandas as pd
import pytest

from swing import engine, trendjb
from swing.trendjb import T1, TREND

from conftest import make_ohlc

TICKERS = tuple(f"T{i}" for i in range(40)) + ("SPY",)
INFO = pd.DataFrame({"index": [500] * 30 + [400] * 10, "sector": "", "sub_industry": ""},
                    index=list(TICKERS[:-1]))
# random prices never jump +10% after a quiet stretch; loosen the numbers so the tests trade
LOOSE = {"big": 0.03, "vol_x": 0.3, "near200": 5.0, "flat": 5.0, "dry": 5.0}
MODULE = {"BODY": 0.05, "ABOVE": 0.9, "BELOW": 1.2, "BAD_UP": -1.0, "NEAR50": 1.0, "RE_RANGE": 5.0}


def _data(seed=7):
    d = make_ohlc(n_days=1200, tickers=TICKERS, seed=seed, drift=0.0006)
    d["INFO"] = INFO
    d["EarnOK"] = pd.DataFrame(np.random.default_rng(1).random((1200, 40)) > 0.3,
                               index=d["Close"].index, columns=list(TICKERS[:-1]))
    return d


@pytest.mark.parametrize("cls", TREND[:12] + [TREND[12]])
def test_no_lookahead(cls, monkeypatch):
    for k, v in MODULE.items():
        monkeypatch.setattr(trendjb, k, v)
    strat = type(cls.__name__ + "Loose", (cls,), LOOSE)
    d1 = _data()
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 900
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    r1 = engine.run(strat().prepare(d1), cost_bps=30)
    assert len(r1.trades) > 0
    e2 = engine.run(strat().prepare(d2), cost_bps=30).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])


def test_same_object_runs_twice_the_same(monkeypatch):
    for k, v in MODULE.items():
        monkeypatch.setattr(trendjb, k, v)
    strat = type("T7Loose", (TREND[6],), LOOSE)().prepare(_data())
    r1, r2 = engine.run(strat, cost_bps=30), engine.run(strat, cost_bps=30)
    assert len(r1.trades) > 0
    pd.testing.assert_series_equal(r1.equity, r2.equity)


def _story(n=400):
    idx = pd.bdate_range("2019-01-01", periods=n)
    tick = ["A0", "SPY"]
    close = pd.DataFrame(100.0, index=idx, columns=tick)
    # slow rise so the 200-day line sits just below; flat last 60 days
    close["A0"] = np.r_[np.linspace(80, 100, 300), np.full(n - 300, 100.0)]
    open_ = close.copy()
    high, low = close + 0.3, close - 0.3
    vol = pd.DataFrame(1e6, index=idx, columns=tick)
    return idx, open_, high, low, close, vol


def _set(df, i, val):
    df.iloc[i, 0] = val


def test_buy_on_dry_red_candle_then_morning_thirds():
    idx, o, h, l, c, v = _story()
    s = 370
    o.iloc[s, 0], c.iloc[s:, 0] = 100.5, 112.0          # +12% candle, open just above the 200-day line
    h.iloc[s, 0], l.iloc[s, 0], v.iloc[s, 0] = 112.5, 100.3, 5e6
    for n, px in enumerate((112.5, 113.0, 113.5, 114.0), start=1):   # quiet rise, the 5-day line catches up
        o.iloc[s + n, 0], c.iloc[s + n, 0], h.iloc[s + n, 0], l.iloc[s + n, 0] = px - 0.3, px, px + 0.2, px - 0.5
        v.iloc[s + n, 0] = 2e6 - n * 1e5
    b = s + 5                                            # short red candle on drying volume: buy the close
    o.iloc[b, 0], c.iloc[b, 0], h.iloc[b, 0], l.iloc[b, 0], v.iloc[b, 0] = 114.6, 114.2, 114.7, 113.9, 1e6
    # next day: gaps down to 113.5, dips to 110 (below -3%), closes 114
    o.iloc[b + 1, 0], h.iloc[b + 1, 0], l.iloc[b + 1, 0], c.iloc[b + 1, 0] = 113.5, 114.5, 110.0, 114.0
    o.iloc[b + 2:, 0], h.iloc[b + 2:, 0], l.iloc[b + 2:, 0], c.iloc[b + 2:, 0] = 114.0, 114.2, 113.8, 114.0
    d = {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v,
         "INFO": pd.DataFrame({"index": [500], "sector": "", "sub_industry": ""}, index=["A0"])}
    r = engine.run(T1().prepare(d), cost_bps=0)
    tr = r.trades
    assert tr.entry_price.iloc[0] == 114.2
    assert tr.reason.iloc[0] == "다음 날 갭 하락 (1/3)" and tr.exit_price.iloc[0] == 113.5
    assert tr.part.iloc[0] == pytest.approx(1 / 3)
    assert tr.reason.iloc[1] == "다음 날 -3% (1/3)" and tr.exit_price.iloc[1] == pytest.approx(114.2 * 0.97)
    assert tr.part.iloc[1] == pytest.approx(1 / 3)
