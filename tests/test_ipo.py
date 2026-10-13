import numpy as np
import pandas as pd

from daytrade import ipo


def _day(c, start=0, v=None):
    c = np.array(c, float)
    o = np.r_[c[0], c[:-1]]
    n = len(c)
    return dict(mi=np.arange(start, start + n), o=o, h=np.maximum(o, c) + 0.05, l=np.minimum(o, c) - 0.05, c=c,
                v=np.ones(n) if v is None else np.array(v, float))


def test_first_close_and_wait():
    day = _day(np.linspace(10, 12, 300), start=90)
    t = ipo.first_close_to_close(day)
    assert t["entry_mi"] == 90 and t["exit_mi"] == 389
    assert ipo.first_close_to_close(_day(np.linspace(12, 10, 300), start=90), wait=5) is None


def test_b17_breakout_and_stop():
    c = [10.0] * 10 + [10.6] * 380
    t = ipo.b17(_day(c), h0=10.5)
    assert t["entry_mi"] == 10 and abs(t["entry_px"] - 10.51) < 1e-9
    assert abs(t["entry_px"] * (1 - t["stop_dist"]) - (10.0 - 0.05 - 0.01)) < 1e-9


def test_despac():
    d0 = pd.Timestamp("2021-06-01")
    r = pd.Series(dict(sic=7372, acq_name=False, first_filing=pd.Timestamp("2019-01-01"), first_425=pd.NaT,
                       first_periodic=pd.Timestamp("2021-08-01")))
    assert ipo.is_despac(r, d0) is False
    assert ipo.is_despac(r.copy().set_axis(r.index).replace({7372: 6770}), d0) is True
    r2 = r.copy()
    r2["first_periodic"] = pd.Timestamp("2020-03-01")
    assert ipo.is_despac(r2, d0) is True
    r3 = r.copy()
    r3["first_filing"] = pd.Timestamp("2024-01-01")  # 나중에 같은 코드를 쓴 다른 회사
    assert ipo.is_despac(r3, d0) is None


def test_cum_profile():
    p = ipo.cum_profile([_day([1, 1, 1], v=[1, 2, 3]), _day([1, 1, 1], v=[3, 2, 1])])
    assert p[0] == 2 and p[2] == 6 and p[389] == 6
