import numpy as np

from daytrade import b01, portfolio


def make_day(bars3):
    """3분봉 목록 [(o,h,l,c,v)] -> 1분봉 (3분 모두 같은 값, 거래량은 3등분)."""
    mi, o, h, l, c, v = [], [], [], [], [], []
    for k, (bo, bh, bl, bc, bv) in enumerate(bars3):
        for j in range(3):
            mi.append(k * 3 + j)
            o.append(bo if j == 0 else (bl if j == 1 else bh))
            h.append(bh)
            l.append(bl)
            c.append(bc if j == 2 else (bl if j == 0 else bh))
            v.append(bv / 3)
    return {k: np.array(x, float) for k, x in zip(("mi", "o", "h", "l", "c", "v"), (mi, o, h, l, c, v))} | {
        "mi": np.array(mi)}


def textbook(rest=None, base_v=600_000):
    # 첫 3분봉 10 -> 11 양봉 (거래대금 약 660만$), 조정 10.6 저점, 거래량 감소, 재돌파 11.2, 이후 +3% 도달
    bars = [(10, 11.05, 9.95, 11.0, base_v),
            (11.0, 11.0, 10.7, 10.8, 50_000),
            (10.8, 10.85, 10.6, 10.7, 40_000),
            (10.7, 11.3, 10.7, 11.2, 300_000),   # 돌파봉: 종가 11.2 > 11.0
            (11.25, 11.4, 11.2, 11.35, 100_000)]
    bars += rest if rest is not None else [(11.35, 11.7, 11.3, 11.6, 100_000)] * 3
    return make_day(bars + [(11.6, 11.6, 11.6, 11.6, 10_000)] * (130 - len(bars)))


def test_three_min_aggregation():
    day = textbook()
    b3 = b01.three_min(day["mi"], day["o"], day["h"], day["l"], day["c"], day["v"])
    assert b3["k"][0] == 0 and b3["o"][0] == 10 and b3["c"][0] == 11.0 and b3["h"][0] == 11.05
    assert abs(b3["v"][0] - 600_000) < 1e-6


def test_d1_textbook_trade_hits_target():
    out = b01.run_day(textbook(), adv=50e6, variants=[b01.D1])
    assert len(out) == 1
    t = out[0]
    assert t["entry_mi"] == 12 and t["entry_px"] == 11.25  # 돌파봉 다음 3분봉 시가
    assert t["reason"] == "익절" and abs(t["gross"] - 0.03) < 1e-9


def test_vol_filter_and_relative_threshold():
    d2 = b01.VARIANTS[1]
    assert b01.run_day(textbook(), adv=50e6, variants=[d2])
    # 기준봉이 20일 평균의 5%에 못 미치면 신호 없음
    assert not b01.run_day(textbook(), adv=500e6, variants=[b01.D1])


def test_no_trade_when_pullback_breaks_base_open():
    day = textbook()
    day["l"][4] = 9.9  # 조정 중 첫봉 시가(10) 아래
    assert not b01.run_day(day, adv=50e6, variants=[b01.D1])


def test_stop_and_close_exit():
    down = [(11.25, 11.25, 10.9, 10.95, 100_000)]
    out = b01.run_day(textbook(rest=down), adv=50e6, variants=[b01.D1])
    t = out[0]
    assert t["reason"] == "손절" and abs(t["gross"] - (-0.02)) < 1e-9  # -2%가 조정 저점보다 가까움
    flat = b01.run_day(textbook(rest=[]), adv=50e6, variants=[b01.VARIANTS[5]])  # D5: 마감까지
    assert flat[0]["reason"] == "마감" and flat[0]["exit_mi"] == 385


def test_touch_entry():
    out = b01.run_day(textbook(), adv=50e6, variants=[b01.VARIANTS[2]])
    assert out and out[0]["entry_px"] == 11.0


def test_portfolio_limits():
    import pandas as pd
    days = pd.Index(["2020-01-02", "2020-01-03"])
    tr = pd.DataFrame(dict(d=["2020-01-02"] * 5, entry_mi=[10, 11, 12, 13, 14], exit_mi=[300] * 5,
                           gross=[0.03] * 5, stop_dist=[0.02] * 5, rank=[0] * 5))
    ret, k = portfolio.daily_returns(tr, days, 0.0)
    assert len(k) == 3  # 동시 보유 3종목
    assert abs(ret.iloc[0] - 3 * 0.25 * 0.03) < 1e-12 and ret.iloc[1] == 0


def test_b02_orb_entry_stop_and_close():
    from daytrade import b02
    n = 390
    mi = np.arange(n)
    o = np.full(n, 10.5); h = np.full(n, 10.6); l = np.full(n, 10.4); c = np.full(n, 10.5)
    o[0], c[4], h[:5], l[:5] = 10.0, 10.4, 10.45, 9.9   # 첫 5분: 10.0 -> 10.4 양봉, 고가 10.45 저가 9.9
    day = dict(mi=mi, o=o, h=h, l=l, c=c)
    t = b02.trade(day, 5, "range")
    assert t["entry_mi"] == 5 and t["entry_px"] == 10.5  # 6번째 1분봉이 시가 10.5로 갭 -> 시가 체결
    assert t["reason"] == "마감" and t["exit_mi"] == 385 and abs(t["gross"]) < 1e-12
    t = b02.trade(day, 5, "atr", atr=1.0)  # 손절 10.5 - 0.1 = 10.4 -> 첫 1분봉 저가 10.4에 닿음
    assert t["reason"] == "손절" and abs(t["gross"] - (-0.1 / 10.5)) < 1e-12
    c[4] = 9.95  # 음봉 범위
    assert b02.trade(day, 5, "range") is None
