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


def test_b03_gap_and_go():
    from daytrade import b03
    n = 390
    mi = np.arange(n)
    o = np.full(n, 10.8); h = np.full(n, 10.9); l = np.full(n, 10.7); c = np.full(n, 10.8)
    h[:15], l[:15] = 10.6, 10.3   # 첫 15분 범위 10.3~10.6, 전날 종가 10.0 (갭 유지)
    o[15], h[15] = 10.5, 10.9     # 9:45 1분봉이 10.6을 넘음 -> 10.6에 매수
    day = dict(mi=mi, o=o, h=h, l=l, c=c)
    t = b03.range_breakout(day, 15, pc=10.0)
    assert t["entry_mi"] == 15 and t["entry_px"] == 10.6 and t["reason"] == "마감"
    assert abs(t["gross"] - (10.8 / 10.6 - 1)) < 1e-12
    assert b03.range_breakout(day, 15, pc=10.35) is None  # 범위 저가 10.3이 전날 종가 아래 -> 갭 메워짐
    h2 = h.copy(); h2[100] = 11.5  # 2R = 10.6 + 2 x 0.31 = 11.22 도달
    t = b03.range_breakout(dict(day, h=h2), 15, pc=10.0, half_r=2.0)
    assert t["reason"].startswith("절반 익절") and abs(t["gross"] - (0.5 * (11.22 / 10.6 - 1) + 0.5 * (10.8 / 10.6 - 1))) < 1e-9
    t = b03.open_buy(day, pm_low=10.2)
    assert t["entry_px"] == 10.8 and t["entry_mi"] == 1


def test_b04_episodic_pivot_entry_volume_and_swing():
    from daytrade import b04
    n = 390
    mi = np.arange(n)
    o = np.full(n, 11.0); h = np.full(n, 11.1); l = np.full(n, 10.9); c = np.full(n, 11.0); v = np.full(n, 1000.0)
    o[0], h[0], l[0], c[0] = 10.5, 10.8, 10.4, 10.7   # 첫 1분봉 고가 10.8, 저가 10.4
    o[1], h[1], l[1] = 10.7, 10.9, 10.6               # 9:31에 10.8 돌파
    v[0] = 20000.0
    day = dict(mi=mi, o=o, h=h, l=l, c=c, v=v)
    # 평소 하루 20만 주, 기준 50% -> 20분에 10만 주 속도: 1분 지났으면 5천 주 필요, 첫 1분봉 2만 주라 통과
    i, px, stop = b04.find_entry(day, 1, 200000, 0.5, adr=0.05)
    assert (i, px, round(stop, 2)) == (1, 10.8, 10.39)
    # 기준 100%로 올리고 첫 봉 거래량을 줄이면 9:50 전엔 속도 미달 -> 9:50 이후 9:30~9:50 합계(19만 9천+)도 미달 -> 없음
    v2 = v.copy(); v2[0] = 1000.0
    assert b04.find_entry(dict(day, v=v2), 1, 200000, 1.0, adr=0.05) is None
    assert b04.find_entry(day, 1, 200000, 0.5, adr=0.01) is None  # 손절폭 3.8% > 1.5 x 1%
    t = b04.intraday(day, 1, 200000, 0.5, 0.05)
    assert t["reason"] == "마감" and abs(t["gross"] - (11.0 / 10.8 - 1)) < 1e-12
    # E7: 2일째 12, 3일째 13에 절반 매도·손절 본전, 4일째 10.5로 시가 하락 -> 시가가 본전(10.8) 아래라 시가에 나머지 매도
    nxt = dict(o=np.array([11.5, 12.5, 10.5]), h=np.array([12.1, 13.1, 10.6]), l=np.array([11.4, 12.4, 10.4]),
               c=np.array([12.0, 13.0, 10.5]))
    t = b04.swing(day, i, px, stop, np.full(9, 10.0), nxt)
    assert t["days"] == 4 and t["reason"] == "절반 후 본전 손절"
    assert abs(t["gross"] - (0.5 * (13 / 10.8 - 1) + 0.5 * (10.5 / 10.8 - 1))) < 1e-12


def test_b05_oops_entry_stop_and_swing():
    from daytrade import b05
    n = 390
    mi = np.arange(n)
    o = np.full(n, 10.2); h = np.full(n, 10.3); l = np.full(n, 10.1); c = np.full(n, 10.2)
    o[:10], h[:10], l[:10], c[:10] = 9.6, 9.8, 9.5, 9.7   # 전날 저가 10.0 아래에서 시작, 저가 9.5
    h[10] = 10.05                                          # 9:40에 전날 저가 10.0 회복
    o[10] = 9.8
    day = dict(mi=mi, o=o, h=h, l=l, c=c)
    i, px, stop = b05.find_entry(day, 10.0)
    assert (i, px, round(stop, 2)) == (10, 10.0, 9.49)
    t = b05.intraday(day, 10.0)
    assert t["reason"] == "마감" and abs(t["gross"] - 0.02) < 1e-12
    assert b05.intraday(day, 10.0, deadline=10) is None    # 9:40 전 마감이면 거래 없음
    # F7: 2일째 시가 9.9(매수가 아래) -> 보유, 3일째 시가 10.4 -> 그 시가에 매도
    t = b05.swing(day, i, px, stop, dict(o=np.array([9.9, 10.4]), l=np.array([9.8, 10.3]), c=np.array([10.1, 10.5])))
    assert t["days"] == 3 and t["reason"] == "이익 난 시가" and abs(t["gross"] - 0.04) < 1e-12


def test_b06_leveraged_etf_close_momentum():
    from daytrade import b06
    mi = np.arange(390)
    idx = dict(mi=mi, c=np.full(390, 102.5))           # 14:44 종가 102.5, 전날 종가 100 -> +2.5%
    move, sgn = b06.signal(idx, 100.0, 0.02, 315)
    assert sgn == 1 and abs(move - 0.025) < 1e-12
    assert b06.signal(idx, 102.0, 0.02, 315)[1] == 0
    o = np.full(390, 50.0); h = np.full(390, 50.5); l = np.full(390, 49.5); c = np.full(390, 50.0)
    o[316], c[389] = 40.0, 42.0                        # 14:46 시가 40에 매수, 15:59 종가 42
    etf = dict(mi=mi, o=o, h=h, l=l, c=c)
    t = b06.hold(etf, 315)
    assert t["entry_mi"] == 316 and abs(t["gross"] - 0.05) < 1e-12
    t = b06.hold(etf, 315, exit_at=385)                 # 15:55 시가 50
    assert abs(t["gross"] - 0.25) < 1e-12
    l2 = l.copy(); l2[316] = 39.0                       # 매수한 봉에서 -2% (39.2) 아래로
    t = b06.hold(dict(etf, l=l2), 315, stop=0.02)
    assert t["reason"] == "손절" and abs(t["gross"] + 0.02) < 1e-12


def test_b07_fomc_window():
    from daytrade import b07
    assert b07.to_mi("14:00") == 270 and b07.to_mi("12:30") == 180
    mi = np.arange(390)
    o = np.full(390, 50.0); c = np.full(390, 50.0); v = np.full(390, 100.0)
    o[265], c[325] = 40.0, 42.0                        # 13:55 시가 40 매수, 14:55 종가 42 매도
    v[271] = 9999.0                                    # 14:01 거래량 최대
    day = dict(mi=mi, o=o, c=c, v=v)
    t = b07.window(day, 265, 325)
    assert t["entry_mi"] == 265 and t["exit_mi"] == 325 and abs(t["gross"] - 0.05) < 1e-12
    assert b07.busiest_minute(day) == 271
    assert b07.price_before(day, 271) == c[270]
    assert b07.window(dict(mi=mi[:200], o=o[:200], c=c[:200], v=v[:200]), 265, 325) is None
