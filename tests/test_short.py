import numpy as np
import pandas as pd

from daytrade import short


def test_decision_rsi_matches_wilder_when_decide_price_is_close():
    rng = np.random.default_rng(0)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    g = pd.DataFrame({"c": c, "cd": c, "hd": c * 1.01, "ld": c * 0.99})
    x = short.decision_series(g)
    ref = short.rsi_wilder(c, 2)
    ok = ~np.isnan(ref) & ~np.isnan(x.rsi2.to_numpy())
    assert ok.sum() > 290
    assert np.allclose(x.rsi2.to_numpy()[ok], ref[ok])
    assert np.isclose(x.sma200.iloc[-1], c[-200:].mean())
    assert np.isclose(x.sma5.iloc[-1], c[-5:].mean())
    assert np.isnan(x.sma200.iloc[198]) and not np.isnan(x.sma200.iloc[199])


def test_decision_uses_decide_price_not_close():
    c = np.array([10, 11, 12, 13, 14, 15.0])
    cd = c.copy()
    cd[-1] = 9.0  # 판단 시각엔 크게 떨어져 있었음
    g = pd.DataFrame({"c": c, "cd": cd, "hd": c + 1, "ld": np.minimum(c, cd) - 0.0})
    x = short.decision_series(g)
    assert x.rsi2.iloc[-1] < 30
    assert np.isclose(x.sma5.iloc[-1], (11 + 12 + 13 + 14 + 9) / 5)
    assert np.isclose(x.ibs.iloc[-1], 0.0)


def test_synthetic_and_exits():
    c = np.array([100, 101, 99, 102.0])
    r = short.synthetic_return(c, 0, 2)
    assert np.isclose(r, (1 + 0.03) * (1 + 3 * (99 / 101 - 1)) - 1 - 0.0004)
    assert np.isclose(short.hold_return(c, 0, 3), 0.02)
    cd = np.array([10, 9, 9.5, 11, 12.0])
    sma5 = np.array([np.nan, 10, 10, 10, 10])
    assert short.exit_rsi(cd, sma5, 0) == 3
    assert short.exit_rsi(cd, sma5, 0, max_days=2) == 2
    assert short.exit_ibs(np.array([0.1, 0.5, 0.8, 0.9]), 0) == 2
    assert short.exit_ibs(np.array([0.1, 0.5, 0.6]), 0) == 2  # 자료 끝


def test_daily_table_decide_vs_close():
    rows = []
    for mi in range(0, 390):
        rows.append(("SPY", "2020-01-02", mi, 100, 101, 99, 100 + (5 if mi > 385 else 0), 1))
    b = pd.DataFrame(rows, columns=["ticker", "d", "mi", "o", "h", "l", "c", "v"])
    t = short.daily_table(b, set())
    assert t.c.iloc[0] == 105 and t.cd.iloc[0] == 100
