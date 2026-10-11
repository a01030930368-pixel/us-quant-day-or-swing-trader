import numpy as np
import pandas as pd
import pytest

from swing import engine
from swing.announce import ANNOUNCE, announcement_tables, reaction_days, top_fifth_cut

from conftest import make_ohlc


def test_reaction_day_moves_after_the_close():
    idx = pd.bdate_range("2021-03-01", periods=10)
    pos = reaction_days(["2021-03-02T07:30:00.000Z", "2021-03-02T16:05:00.000Z",
                         "2021-03-06T10:00:00.000Z"], idx)            # pre-open, after close, Saturday
    assert list(pos) == [1, 2, 5]


def test_quarter_matched_to_its_release_not_the_10q():
    q = lambda s, e, v, f: {"start": s, "end": e, "val": v, "filed": f, "form": "10-Q"}
    ends = pd.date_range("2018-03-31", periods=14, freq="QE")
    vals = [1.0 + 0.1 * (k % 3) for k in range(13)] + [3.0]
    eps = [q(str((e - pd.Timedelta(days=89)).date()), str(e.date()), v, str((e + pd.Timedelta(days=40)).date()))
           for e, v in zip(ends, vals)]
    facts = {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": eps}}}}}
    last = ends[-1]
    release = (last + pd.Timedelta(days=20)).strftime("%Y-%m-%dT08:00:00.000Z")
    idx = pd.bdate_range("2018-01-01", "2022-06-30")
    ann, sue, cover = announcement_tables({"AAA": facts}, {"AAA": [release]}, idx)
    day = idx[idx.searchsorted(pd.Timestamp(release[:10])) + 1]       # the day after the reaction day
    assert cover == {"announce": 1, "matched": 1}
    assert ann["AAA"].dropna().index.tolist() == [day]
    assert sue.loc[day, "AAA"] > 10 and sue["AAA"].notna().sum() == 1


def test_top_fifth_uses_only_past_days():
    idx = pd.bdate_range("2020-01-01", periods=200)
    ear = pd.DataFrame(np.nan, index=idx, columns=["A"])
    ear.iloc[::2, 0] = np.arange(100, dtype=float)
    cut = top_fifth_cut(ear)
    ear2 = ear.copy()
    ear2.iloc[150:, 0] = 1e6
    pd.testing.assert_series_equal(cut.iloc[:151], top_fifth_cut(ear2).iloc[:151])


@pytest.mark.parametrize("cls", ANNOUNCE)
def test_announce_no_lookahead(cls):
    d1 = make_ohlc(n_days=1100, seed=5, drift=0.0012)
    names = [c for c in d1["Close"].columns if c != "SPY"]
    ann = pd.DataFrame(np.nan, index=d1["Close"].index, columns=names)
    sue = ann.copy()
    rng = np.random.default_rng(1)
    for j, t in enumerate(names):
        days = np.arange(250 + 7 * j, 1100, 13)
        ann.iloc[days, j] = 1.0
        sue.iloc[days, j] = rng.normal(1.5, 1.5, size=len(days))
    d1["ANN"], d1["SUE_ANN"] = ann, sue
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 800
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    r1 = engine.run(cls().prepare(d1), cost_bps=25)
    e2 = engine.run(cls().prepare(d2), cost_bps=25).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])
    assert len(r1.trades) > 0
