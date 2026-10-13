import numpy as np
import pandas as pd
import pytest

from swing import canslim, engine
from swing.canslim import CANSLIM, R1, market_5day, market_oneil, quality_flags

from conftest import make_ohlc

TICKERS = tuple(f"T{i}" for i in range(40)) + ("SPY",)
INFO = pd.DataFrame({"index": [500] * 30 + [400] * 10, "sector": "",
                     "sub_industry": [f"ind{i % 4}" for i in range(40)]}, index=list(TICKERS[:-1]))
# random prices rarely break a tight box on double volume; loosen the numbers so the tests trade
LOOSE = {"box": 5.0, "vol_x": 0.5}
MODULE = {"BIG_UP": -1.0, "TOP": 0.0, "SECTOR": -1.0, "NEAR_HIGH": 0.0, "RS_TOP": 0.3, "BREAK_VOL": 0.5,
          "BREAK_DROP": 0.0, "NEAR50": 0.05}


def _data(seed=7):
    d = make_ohlc(n_days=1200, tickers=TICKERS, seed=seed, drift=0.0006)
    d["INFO"] = INFO
    rng = np.random.default_rng(1)
    d["EarnOK"] = pd.DataFrame(rng.random((1200, 40)) > 0.2, index=d["Close"].index, columns=list(TICKERS[:-1]))
    d["QualOK"] = pd.DataFrame(rng.random((1200, 40)) > 0.2, index=d["Close"].index, columns=list(TICKERS[:-1]))
    d["VIX"] = pd.Series(rng.uniform(10, 40, 1200), index=d["Close"].index)
    return d


@pytest.mark.parametrize("cls", CANSLIM)
def test_no_lookahead(cls, monkeypatch):
    for k, v in MODULE.items():
        monkeypatch.setattr(canslim, k, v)
    monkeypatch.setattr(canslim, "_SHARED", None)
    strat = type(cls.__name__ + "Loose", (cls,), LOOSE)
    d1 = _data()
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 900
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    r1 = engine.run(strat().prepare(d1), cost_bps=30)
    assert len(r1.trades) > 0
    monkeypatch.setattr(canslim, "_SHARED", None)
    e2 = engine.run(strat().prepare(d2), cost_bps=30).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])


def test_same_object_runs_twice_the_same(monkeypatch):
    for k, v in MODULE.items():
        monkeypatch.setattr(canslim, k, v)
    strat = type("R9Loose", (CANSLIM[8],), LOOSE)().prepare(_data())
    r1, r2 = engine.run(strat, cost_bps=30), engine.run(strat, cost_bps=30)
    assert len(r1.trades) > 0
    pd.testing.assert_series_equal(r1.equity, r2.equity)


def _bars(close, opens=None):
    c = np.asarray(close, float)
    o = c.copy() if opens is None else np.asarray(opens, float)
    return o, np.maximum(o, c) + 0.1, np.minimum(o, c) - 0.1, c, np.full(len(c), 1e6)


def test_five_day_turns_up_after_three_days_with_white_candles():
    # fall, then three closes above the 5-day line, each a white candle
    close = [100 - i for i in range(15)] + [92, 93, 94, 95]
    opens = [c + 0.5 for c in close[:15]] + [91, 92, 93, 94]
    up, turns, downs = market_5day(*_bars(close, opens))
    first = 15                                   # first close above the 5-day line
    assert not up[first] and not up[first + 1]
    assert up[first + 2] and turns == [first + 2]


def test_five_day_needs_two_white_candles():
    close = [100 - i for i in range(15)] + [92, 93, 94]
    opens = [c + 0.5 for c in close[:15]] + [93, 94, 95]           # all three red
    up, turns, _ = market_5day(*_bars(close, opens))
    assert turns == []
    up, turns, _ = market_5day(*_bars(close, opens), quick=True)   # M1d: one close is enough
    assert turns == [15]


def test_five_day_gap_down_turns_down_next_day():
    close = [80 + i for i in range(20)] + [96, 95]
    opens = [c - 0.5 for c in close[:20]] + [99, 95.5]            # break below the line, then gap down
    up, turns, downs = market_5day(*_bars(close, opens))
    assert up[19] and downs == [21]


def test_five_day_quick_recovery_is_not_a_turn_down():
    close = [80 + i for i in range(20)] + [96, 100, 101]
    opens = [c - 0.5 for c in close[:20]] + [99, 97, 100]          # break, then back above the next day
    up, _, downs = market_5day(*_bars(close, opens))
    assert downs == [] and up[-1]


def test_five_day_three_days_below_turns_down():
    close = [80 + i for i in range(20)] + [96, 96.5, 96.2, 96.0]
    opens = [c - 0.5 for c in close[:20]] + [99, 96, 96, 96]           # no gap, but never back above
    _, _, downs = market_5day(*_bars(close, opens))
    assert downs == [23]                                          # three days after the break


def test_oneil_follow_through_and_distribution():
    # a fall, a low, rally day 1, then day 4 up 1.5% on higher volume = follow-through
    close = [100 - i for i in range(10)] + [91.5, 92, 92.2, 93.6]
    o, h, l, c, v = _bars(close)
    v = np.arange(len(c), dtype=float) + 1e6
    up, turns, _ = market_oneil(o, h, l, c, v)
    assert turns == [13]
    # then five down days on rising volume inside 25 days = market down
    c2 = np.r_[c, [93.4, 93.2, 93.0, 92.8, 92.6]]
    o2, h2, l2, _, _ = _bars(c2)
    v2 = np.arange(len(c2), dtype=float) + 1e6
    _, _, downs = market_oneil(o2, h2, l2, c2, v2)
    assert downs == [18]


def test_quality_flags_from_annual_report():
    idx = pd.bdate_range("2020-01-01", periods=400)
    facts = {"facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [{"start": "2019-01-01", "end": "2019-12-31", "val": 20,
                                             "filed": "2020-02-20", "form": "10-K"}]}},
        "StockholdersEquity": {"units": {"USD": [{"end": "2019-12-31", "val": 100,
                                                  "filed": "2020-02-20", "form": "10-K"}]}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            {"start": "2019-01-01", "end": "2019-12-31", "val": 25, "filed": "2020-02-20", "form": "10-K"}]}}}}}
    weak = {"facts": {"us-gaap": {**facts["facts"]["us-gaap"],
                                  "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                                      {"start": "2019-01-01", "end": "2019-12-31", "val": 21,
                                       "filed": "2020-02-20", "form": "10-K"}]}}}}}
    flags, covered = quality_flags({"A": facts, "B": weak}, idx)
    assert covered == 2
    assert not flags["A"].loc["2020-02-20"] and flags["A"].loc["2020-02-21"]   # usable the next day
    assert not flags["B"].any()                                                 # cash flow 21 < 1.2 x 20


def _story(n=320):
    """One stock in a tight box breaks out on huge volume the day the market turns up."""
    idx = pd.bdate_range("2019-01-01", periods=n)
    tick = ["A0", "A1", "SPY"]
    close = pd.DataFrame(100.0, index=idx, columns=tick)
    close["A0"] = np.r_[np.linspace(60, 100, 260), np.full(n - 260, 100.0)]
    close["A1"] = close["A0"]
    spy = np.r_[np.linspace(200, 300, 280), np.linspace(300, 270, 20), np.full(n - 300, 270.0)]
    spy[300:303] = [280, 282, 284]                            # back above the 5-day line, three white days
    close["SPY"] = spy
    open_ = close.copy()
    open_.iloc[300:303, 2] = [275, 281, 283]
    vol = pd.DataFrame(1e6, index=idx, columns=tick)
    s = 302                                                    # breakout on the turn day
    close.iloc[s, 0] = close.iloc[s, 1] = 106.0
    open_.iloc[s, 0] = open_.iloc[s, 1] = 101.0
    vol.iloc[s, :2] = 5e6
    close.iloc[s + 1:, 0] = 108.0
    open_.iloc[s + 1:, 0] = 108.0
    close.iloc[s + 1:, 1] = 106.0
    high, low = np.maximum(open_, close) + 0.3, np.minimum(open_, close) - 0.3
    high.iloc[s + 5, 0] = 130.0                                # +20% target hit five days later
    d = {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}
    d["INFO"] = pd.DataFrame({"index": [500, 500], "sector": "", "sub_industry": ["x", "x"]}, index=["A0", "A1"])
    d["EarnOK"] = pd.DataFrame(True, index=idx, columns=["A0", "A1"])
    d["QualOK"] = d["EarnOK"]
    return d, s


def test_story_buys_breakout_at_close_and_takes_20_percent(monkeypatch):
    monkeypatch.setattr(canslim, "_SHARED", None)
    monkeypatch.setattr(canslim, "MIN_DOLLAR_VOL", 0)
    monkeypatch.setattr(canslim, "RS_TOP", 0.0)
    d, s = _story()
    res = engine.run(R1().prepare(d), cost_bps=0)
    tr = res.trades[res.trades["ticker"] == "A0"]
    assert len(tr) == 1
    row = tr.iloc[0]
    assert row["entry_price"] == pytest.approx(106.0)                      # the breakout day's close
    assert str(row["entry_date"]) == str(d["Close"].index[s + 1].date())    # held from the next open
    assert row["exit_price"] == pytest.approx(106.0 * 1.2)
    assert row["reason"].startswith("익절")
