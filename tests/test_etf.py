import datetime as dt

import numpy as np

from daytrade import etf


def _day(o, h, l, c, start=0):
    n = len(o)
    return dict(mi=np.arange(start, start + n), o=np.array(o, float), h=np.array(h, float),
                l=np.array(l, float), c=np.array(c, float), v=np.ones(n))


def test_holidays():
    h = etf.nyse_holidays(2021, 2023)
    assert dt.date(2021, 12, 24) in h          # 크리스마스 토요일 -> 금요일
    assert dt.date(2021, 12, 31) not in h      # 새해 토요일 -> 휴장 없음
    assert dt.date(2022, 6, 20) in h           # 준틴스 일요일 -> 월요일
    assert dt.date(2023, 4, 7) in h            # 성금요일
    assert dt.date(2023, 11, 23) in h          # 추수감사절
    assert not any(d.weekday() >= 5 for d in h)
    assert dt.date(1997, 1, 20) not in etf.nyse_holidays(1997, 1997)  # MLK는 1998년부터


def test_narrow_flags():
    h = np.array([10, 10, 10, 10, 10, 10, 10, 9.5])
    l = np.array([8, 8, 8, 8, 8, 8, 8, 9.0])
    nr7, idnr4 = etf.narrow_flags(h, l)
    assert nr7[-1] and idnr4[-1] and not nr7[:-1].any()


def test_open_to_close_half():
    n = 390
    c = np.linspace(100, 110, n)
    day = _day(c, c, c, c)
    t = etf.open_to_close(day)
    assert t["entry_mi"] == 1 and t["exit_mi"] == 389
    t = etf.open_to_close(day, half=True)
    assert t["exit_mi"] == 209


def test_crabel_entry_and_stop():
    # 시가 100, 스트레치 1 -> 101 넘으면 매수, 손절 99.99
    o = [100, 100.5, 101.5, 101.5] + [101.5] * 386
    h = [100.2, 101.2, 101.6, 101.6] + [101.6] * 386
    l = [99.9, 100.4, 101.4, 101.4] + [101.4] * 386
    c = [100.1, 101.1, 101.5, 101.5] + [101.5] * 386
    t = etf.crabel(_day(o, h, l, c), 1.0)
    assert t["entry_mi"] == 1 and abs(t["entry_px"] - 101) < 1e-9 and t["reason"] != "손절"
    # 매수 봉 안에서 손절가까지 내려가면 손절
    l2 = list(l)
    l2[1] = 99.0
    t = etf.crabel(_day(o, h, l2, c), 1.0)
    assert t["reason"] == "손절" and abs(t["exit_px"] - 99.99) < 1e-9


def test_first15_break_skips_wide_stop():
    o = [100] * 390
    h = [101] * 15 + [102] * 375
    l = [90] * 15 + [100] * 375
    assert etf.first15_break(_day(o, h, l, o)) is None   # 손절까지 11% -> 건너뜀
    l = [99] * 15 + [100] * 375
    t = etf.first15_break(_day(o, h, l, o))
    assert t["entry_mi"] == 15 and abs(t["entry_px"] - 101.01) < 1e-9
