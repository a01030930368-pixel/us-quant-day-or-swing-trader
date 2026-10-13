import numpy as np
import pandas as pd

from swing.pit import adjust_splits, load_daily, membership


def test_membership_rolls_back_changes():
    dates = pd.bdate_range("2020-01-01", periods=10)
    changes = pd.DataFrame({"date": [dates[3], dates[7]], "added": ["B", "C"], "removed": ["X", "B"]})
    m = membership(dates, ["A", "C"], changes)
    assert m["A"].all()
    assert m["X"].tolist() == [True] * 3 + [False] * 7
    assert m["B"].tolist() == [False] * 3 + [True] * 4 + [False] * 3
    assert m["C"].tolist() == [False] * 7 + [True] * 3


def test_split_is_taken_out():
    d = pd.bdate_range("2020-01-01", periods=6)
    g = pd.DataFrame({"d": d, "o": [100, 101, 102, 51, 52, 53.0], "h": 0.0, "l": 0.0,
                      "c": [101, 102, 102, 52, 53, 54.0], "v": 1e6})
    g["h"], g["l"] = g[["o", "c"]].max(axis=1), g[["o", "c"]].min(axis=1)
    out, n = adjust_splits(g)
    assert n == 1
    assert np.allclose(out["c"].iloc[:3], [50.5, 51, 51])
    assert out["v"].iloc[0] == 2e6 and out["c"].iloc[-1] == 54


def test_reused_ticker_is_cut(tmp_path):
    d = pd.bdate_range("2020-01-01", periods=60)
    rows = [{"ticker": "OLD", "d": x, "o": 10.0, "h": 10.0, "l": 10.0, "c": 10.0, "v": 1.0} for x in d[:20]]
    rows += [{"ticker": "OLD", "d": x, "o": 99.0, "h": 99.0, "l": 99.0, "c": 99.0, "v": 1.0} for x in d[45:]]
    rows += [{"ticker": "SPY", "d": x, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0} for x in d]
    pd.DataFrame(rows).to_parquet(tmp_path / "daily_2020.parquet")
    wide, info = load_daily([tmp_path / "daily_2020.parquet"], {"OLD", "SPY"}, {"OLD": d[10]})
    assert info["cut"] == 1
    assert wide["Close"]["OLD"].dropna().max() == 10.0
