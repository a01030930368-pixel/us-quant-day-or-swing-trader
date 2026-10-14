import numpy as np
import pandas as pd
import pytest

from swing import bulkowski, divmonth, elder, epgap, events, nonews, nr7, preearn, spindex, taxloss
from swing.events import Panel, Trade, simulate, walk

N, M = 1500, 30
TICK = [f"T{i}" for i in range(M)] + ["SPY"]


def _data(seed=3, vol=0.03):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=N)
    r = rng.normal(0.0004, vol, size=(N, M + 1))
    r[:, -1] = rng.normal(0.0004, 0.01, size=N)
    close = pd.DataFrame(50 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=TICK)
    jump = np.where(rng.random(close.shape) < 0.03, rng.choice([-0.10, 0.08, 0.12], size=close.shape), 0.0)
    jump[:, -1] = 0.0
    close = close * np.exp(np.cumsum(jump, axis=0))                 # rare big gaps and drops
    gap = pd.DataFrame(rng.normal(0, vol / 2, size=close.shape) + jump, index=idx, columns=TICK)
    o = close.shift(1).fillna(close.iloc[0]) * np.exp(gap)
    h = np.maximum(o, close) * (1 + rng.uniform(0, vol / 2, size=close.shape))
    lo = np.minimum(o, close) * (1 - rng.uniform(0, vol / 2, size=close.shape))
    v = pd.DataFrame(rng.lognormal(13, 0.6, size=close.shape), index=idx, columns=TICK)
    return {"Open": o, "High": h, "Low": lo, "Close": close, "Volume": v}


def _ext(p: Panel, seed=5):
    rng = np.random.default_rng(seed)
    ext = {}
    # earnings every ~63 days with jitter
    ext["earnings"] = {k: np.unique(np.clip(np.arange(30 + k, N, 63) + rng.integers(-3, 4, size=len(range(30 + k, N, 63))),
                                            0, N - 1)) for k in range(M)}
    ext["news"] = {k: np.unique(rng.integers(0, N, size=60)) for k in range(M)}
    amount = np.zeros((N, M))
    for k in range(M):
        amount[np.arange(20 + k, N, 63), k] = 0.6
    ext["div_amount"] = amount
    ext["raw_close"] = p.C.copy()
    ext["ex_days"] = {k: np.flatnonzero(amount[:, k] > 0) for k in range(M)}
    mem = np.zeros((N, M), bool)
    for k in range(M):
        a = rng.integers(100, N - 200)
        b = rng.integers(a + 50, N)
        mem[a:b, k] = True
    ext["sp_member"] = mem
    ext["INFO"] = pd.DataFrame({"index": [500] * 20 + [400] * 5 + [600] * 5}, index=TICK[:-1])
    return ext


def _panel(d):
    return Panel(d, "SPY", min_price=0.0, min_dollar_vol=0.0)


LOOSE = {"epgap": {"VOL_X": 1.0}}
MODULES = [(preearn, preearn.RULES), (epgap, epgap.RULES), (nonews, nonews.RULES), (elder, elder.RULES),
           (divmonth, divmonth.RULES), (taxloss, taxloss.RULES), (nr7, nr7.RULES), (bulkowski, bulkowski.RULES),
           (spindex, spindex.ADD_RULES), (spindex, spindex.DEL_RULES)]
CASES = [(m, name, fn) for m, rules in MODULES for name, _, fn in rules]


@pytest.mark.parametrize("mod,name,fn", CASES, ids=[f"{m.__name__.split('.')[-1]}-{n}" for m, n, _ in CASES])
def test_no_lookahead(mod, name, fn, monkeypatch):
    for k, v in LOOSE.get(mod.__name__.split(".")[-1], {}).items():
        monkeypatch.setattr(mod, k, v)
    d1 = _data()
    p1 = _panel(d1)
    ext1 = _ext(p1)
    t1 = fn(p1, ext1)
    k = 1100
    d2 = {f: x.copy() for f, x in d1.items()}
    for f in d2:
        d2[f].iloc[k + 1:] *= 1.5
    p2 = _panel(d2)
    ext2 = _ext(p2)
    ext2["raw_close"] = p2.C.copy()
    t2 = fn(p2, ext2)
    before = lambda ts: sorted((t.k, t.ei, round(t.ep, 8)) for t in ts if t.ei <= k)
    assert before(t1) == before(t2)
    legs = lambda ts: sorted((t.k, t.ei, x, round(px, 8), r) for t in ts if t.ei <= k for _, x, px, r in t.legs if x <= k)
    assert legs(t1) == legs(t2)
    a1 = simulate(t1, p1, 30, 10)
    a2 = simulate(t2, p2, 30, 10)
    pd.testing.assert_series_equal(a1.equity.iloc[:k + 1], a2.equity.iloc[:k + 1])


def test_rules_trade_on_random_data(monkeypatch):
    monkeypatch.setattr(epgap, "VOL_X", 1.0)
    p = _panel(_data())
    ext = _ext(p)
    for mod, rules in MODULES:
        if mod is bulkowski:
            continue                     # random walks rarely draw these patterns; see the story tests
        name, _, fn = rules[0]
        assert len(fn(p, ext)) > 0, (mod.__name__, name)


def _one(close, opens=None, highs=None, lows=None):
    c = np.asarray(close, float)
    o = c.copy() if opens is None else np.asarray(opens, float)
    h = np.maximum(o, c) + 0.1 if highs is None else np.asarray(highs, float)
    lo = np.minimum(o, c) - 0.1 if lows is None else np.asarray(lows, float)
    idx = pd.bdate_range("2020-01-01", periods=len(c))
    f = lambda x, spy: pd.DataFrame({"A": x, "SPY": spy}, index=idx)
    spy = np.full(len(c), 100.0)
    return Panel({"Open": f(o, spy), "High": f(h, spy), "Low": f(lo, spy), "Close": f(c, spy),
                  "Volume": f(np.full(len(c), 1e6), np.full(len(c), 1e6))}, "SPY", min_price=0, min_dollar_vol=0)


def test_walk_stop_target_and_gap():
    p = _one([10, 10, 10, 9, 12], opens=[10, 10, 10, 9.4, 12], lows=[9.9, 9.9, 9.9, 9.0, 11.9],
             highs=[10.1, 10.1, 10.1, 9.5, 12.1])
    legs = walk(p, 0, 0, 10.0, at_close=True, stop=9.5)
    assert legs == [(1.0, 3, 9.4, "손절 (갭 하락)")]                # opened below the stop
    legs = walk(p, 0, 0, 10.0, at_close=True, stop=8.0, target=11.0)
    assert legs == [(1.0, 4, 12.0, "익절")]                          # gap above the target fills at the open
    legs = walk(p, 0, 0, 10.0, at_close=True, max_day=2)
    assert legs == [(1.0, 2, 10.0, "보유 기간 끝")]


def test_walk_partial_moves_stop():
    p = _one([10, 10.5, 11, 11.5, 10.2], lows=[9.9, 10.4, 10.9, 11.4, 9.95])

    def part(j, st):
        if j == 3:
            st["stop"] = st["ep"]
            return (1 / 3, "3일째")
        return None
    legs = walk(p, 0, 0, 10.0, at_close=True, stop=9.0, partial=part)
    assert legs[0] == (1 / 3, 3, 11.5, "3일째")
    assert legs[1][1:] == (4, 10.0, "손절") and legs[1][0] == pytest.approx(2 / 3)


def test_simulate_books_cost_and_slots():
    p = _one([10, 10, 11, 11, 11])
    trades = [Trade(0, 1, 10.0, [(1.0, 3, 11.0, "x")])]
    acc = simulate(trades, p, 100, slots=2)                          # 1% each way, half the account
    spend = 0.5
    expect = 1 - spend * 1.01 + spend / 10 * 11 * 0.99
    assert acc.equity.iloc[-1] == pytest.approx(expect)
    assert acc.equity.iloc[2] == pytest.approx(1 - spend * 1.01 + spend / 10 * 11)


def test_preearn_buys_five_days_before_expected_and_sells_day_before():
    d = _data()
    p = _panel(d)
    r0 = 100
    e = int(p.dates.searchsorted(p.dates[r0] + pd.Timedelta(days=364)))
    actual = e + 2
    ext = {"earnings": {**{k: np.array([], int) for k in range(M)}, 0: np.array([r0, actual])}}
    tr = [t for t in preearn.rule(p, ext) if t.k == 0]
    assert len(tr) == 2 and tr[0].ei == e - 5 and tr[0].legs[0][1] == actual - 1
    late = {"earnings": {**{k: np.array([], int) for k in range(M)}, 0: np.array([r0, e + 20])}}
    tr = [t for t in preearn.rule(p, late) if t.k == 0]
    assert tr[0].legs[0][1] == e + 5 and "발표 없음" in tr[0].legs[0][3]


def test_double_bottom_story():
    # decline, bottom 1, +15% bounce, bottom 2 (same price) after 30 days, breakout above the bounce high
    down = np.linspace(130, 100, 61)
    b1 = np.r_[np.linspace(100, 100.5, 3), np.linspace(101, 115, 12), np.linspace(115, 101, 13)]
    b2 = np.r_[np.linspace(100.4, 100.8, 3), np.linspace(101, 114, 10), [116, 117, 118, 118, 118]]
    close = np.r_[np.full(30, 130), down, b1, b2]
    lows = close - 0.2
    p = _one(close, lows=lows, highs=close + 0.2)
    sig = bulkowski.dbl_signals(p, 0, shape=False)
    assert len(sig) == 1
    i, lower, peak, j1, j2 = sig[0]
    assert close[i] > peak and close[i - 1] <= peak
    assert 20 <= j2 - j1 <= 65 and lower == pytest.approx(min(lows[j1], lows[j2]))


def test_high_tight_flag_story():
    pole = np.linspace(20, 40, 30)                                   # doubles in 30 days
    flag = np.r_[np.linspace(39.5, 36, 8), np.linspace(36, 39, 5)]
    close = np.r_[np.full(40, 20), pole, flag, [41]]
    p = _one(close, highs=close + 0.1, lows=close - 0.1)
    sig = bulkowski.htf_signals(p, 0, 1.9)
    assert [s[0] for s in sig] == [len(close) - 1]


def test_taxloss_calendar_and_pick():
    cal = taxloss.calendar(pd.bdate_range("2015-01-01", "2017-02-28"))
    c = cal[0]
    idx = pd.bdate_range("2015-01-01", "2017-02-28")
    assert idx[c["pick"]].month == 12 and idx[c["jan"][9]].year == 2016
    assert sum(1 for d in idx[:c["pick"]] if d.year == 2015 and d.month == 12) == taxloss.PICK


def test_index_changes():
    m = np.zeros((10, 2), bool)
    m[4:8, 0] = True
    assert spindex.changes(m, True).tolist() == [[4, 0]]
    assert spindex.changes(m, False).tolist() == [[8, 0]]


def test_dividend_regular_gap():
    dates = pd.bdate_range("2015-01-01", periods=600)
    xs = np.array([0, 63, 126, 189, 252])
    assert divmonth.regular(dates, xs, 4) == pytest.approx(91, abs=3)
    xs2 = np.array([0, 40, 126, 189, 252])
    assert divmonth.regular(dates, xs2, 4) is None
