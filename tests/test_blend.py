import pandas as pd
import pytest

from swing.blend import blend


def test_blend_resets_each_month_and_charges_moves():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    up, down = pd.Series(0.01, index=idx), pd.Series(-0.01, index=idx)
    free = blend({"a": up, "b": down}, {"a": 1, "b": 1}, cost_bps=0)
    paid = blend({"a": up, "b": down}, {"a": 1, "b": 1}, cost_bps=25)
    assert free.iloc[0] == pytest.approx(0.0)                 # half up 1%, half down 1%
    first_feb = idx.get_loc(idx[idx.month == 2][0])
    assert free.iloc[first_feb] == pytest.approx(0.0)         # back to 50/50 on the first day of a month
    assert paid.iloc[first_feb] < free.iloc[first_feb]        # moving money between parts costs
    assert (paid.iloc[1:first_feb] == free.iloc[1:first_feb]).all()


def test_blend_of_one_strategy_is_that_strategy():
    idx = pd.bdate_range("2021-01-01", periods=80)
    r = pd.Series([0.01, -0.02, 0.005, 0.0] * 20, index=idx)
    pd.testing.assert_series_equal(blend({"a": r}, {"a": 1}, cost_bps=25), r, check_names=False)
