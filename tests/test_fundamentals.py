import numpy as np
import pandas as pd
import pytest

from swing import engine
from swing.earnings_legends import EARNINGS
from swing.fundamentals import daily_flags, quarterly, yoy_growth

from conftest import make_ohlc


def fact(start, end, val, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form}


def facts(eps_items):
    return {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": eps_items}}}}}


FY2020 = [fact("2020-01-01", "2020-03-31", 1.0, "2020-05-01"),
          fact("2020-04-01", "2020-06-30", 1.0, "2020-08-01"),
          fact("2020-07-01", "2020-09-30", 1.0, "2020-11-01"),
          fact("2020-01-01", "2020-12-31", 4.0, "2021-02-15", "10-K")]
FY2021 = [fact("2021-01-01", "2021-03-31", 1.5, "2021-05-01"),
          # restated a year later in the next 10-Q: the first value must win
          fact("2021-01-01", "2021-03-31", 0.5, "2022-05-01"),
          fact("2021-04-01", "2021-06-30", 1.1, "2021-08-01"),
          fact("2021-07-01", "2021-09-30", 1.0, "2021-11-01"),
          fact("2021-01-01", "2021-12-31", 5.6, "2022-02-15", "10-K")]


def test_quarters_first_filed_and_q4_derived():
    q = quarterly(["EarningsPerShareDiluted"], facts(FY2020 + FY2021), "USD/shares")
    assert len(q) == 8
    assert q.iloc[4]["val"] == 1.5                                        # not the restatement
    q4 = q[q["end"] == pd.Timestamp("2021-12-31")].iloc[0]
    assert q4["val"] == pytest.approx(5.6 - 3.6) and q4["filed"] == pd.Timestamp("2022-02-15")
    g = yoy_growth(q).set_index("end")["growth"]
    assert g[pd.Timestamp("2021-03-31")] == pytest.approx(0.5)
    assert g[pd.Timestamp("2021-12-31")] == pytest.approx(1.0)
    assert np.isnan(g[pd.Timestamp("2020-03-31")])                       # no year-earlier quarter


def test_flags_start_the_day_after_filing():
    g = pd.DataFrame({"end": pd.to_datetime(["2021-03-31", "2021-06-30"]),
                      "filed": pd.to_datetime(["2021-05-03", "2021-08-02"]),
                      "growth": [0.5, 0.1]})
    idx = pd.bdate_range("2021-04-01", "2022-03-31")
    f = daily_flags(g, idx, 0.25)
    assert not f[:"2021-05-03"].any()                       # filing day itself: not yet known
    assert f["2021-05-04":"2021-08-02"].all()
    assert not f["2021-08-03":].any()                       # next quarter too weak


def test_growth_from_a_loss_does_not_count():
    q = pd.DataFrame({"end": pd.to_datetime(["2020-03-31", "2021-03-31"]),
                      "filed": pd.to_datetime(["2020-05-01", "2021-05-01"]), "val": [-1.0, 1.0]})
    assert np.isnan(yoy_growth(q)["growth"].iloc[1])


@pytest.mark.parametrize("cls", EARNINGS)
def test_earnings_sets_no_lookahead_and_gate(cls):
    d1 = make_ohlc(n_days=1100, seed=7, drift=0.0012)
    ok = pd.DataFrame(True, index=d1["Close"].index, columns=d1["Close"].columns)
    ok["AAA"] = False
    d1["EarnOK"] = ok
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 800
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    r1 = engine.run(cls().prepare(d1), cost_bps=20)
    e2 = engine.run(cls().prepare(d2), cost_bps=20).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])
    if cls.name.endswith("_earn"):
        assert "AAA" not in set(r1.trades["ticker"])


def test_sue_uses_only_past_changes():
    from swing.pead import sue_events
    ends = pd.date_range("2015-03-31", periods=16, freq="QE")
    vals = [1.0, 1.0, 1.0, 1.0] + [1.0 + 0.1 * (k % 2) for k in range(11)] + [3.0]
    q = pd.DataFrame({"end": ends, "filed": ends + pd.Timedelta(days=40), "val": vals})
    ev = sue_events(q).set_index("end")["sue"]
    assert ev.iloc[:8].isna().all()                  # needs 4 past year-over-year changes
    assert ev.iloc[-1] > 10                          # a jump far beyond the usual change


@pytest.mark.parametrize("cls", __import__("swing.pead", fromlist=["PEAD_SET"]).PEAD_SET)
def test_pead_no_lookahead_and_holds_60_days(cls):
    d1 = make_ohlc(n_days=1100, seed=3, drift=0.0012)
    sue = pd.DataFrame(np.nan, index=d1["Close"].index, columns=d1["Close"].columns)
    sue.iloc[300::70, 0] = 3.0
    sue.iloc[320::70, 1] = 1.0                       # below the threshold: never bought
    d1["SUE"] = sue
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 800
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    r1 = engine.run(cls().prepare(d1), cost_bps=20)
    e2 = engine.run(cls().prepare(d2), cost_bps=20).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])
    names = d1["Close"].columns
    assert names[1] not in set(r1.trades["ticker"])
    if cls.name == "pead":
        assert len(r1.trades) and (r1.trades.query("reason != '손절'")["days"] == 61).all()   # open to open: 60 days, counted inclusive
