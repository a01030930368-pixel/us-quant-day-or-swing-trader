import pandas as pd

from swing.indicators import atr, rsi, sma


def test_sma():
    s = pd.DataFrame({"A": [1.0, 2, 3, 4]})
    assert sma(s, 2)["A"].tolist()[1:] == [1.5, 2.5, 3.5]


def test_rsi_extremes():
    up = pd.DataFrame({"A": [float(i) for i in range(1, 30)]})
    down = pd.DataFrame({"A": [float(i) for i in range(30, 1, -1)]})
    assert rsi(up, 2)["A"].iloc[-1] == 100
    assert rsi(down, 2)["A"].iloc[-1] < 1


def test_atr_constant_range():
    n = 40
    c = pd.DataFrame({"A": [100.0] * n})
    h, l = c + 1, c - 1
    assert abs(atr(h, l, c, 14)["A"].iloc[-1] - 2.0) < 1e-9
