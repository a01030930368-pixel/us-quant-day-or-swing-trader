"""b21~b23(3배 ETF)·b25(주말 비트코인 → 월요일 관련주) 백테스트: prep_etf 폴더 -> 결과표(마크다운).

    python scripts/run_etf.py --prep prep_etf --rule b21 --out reports/b21
"""
import argparse
import datetime as dt
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daytrade import etf, portfolio  # noqa: E402

FEE, SLIP, SLIP_ETF = 0.0025, 0.001, 0.0002
GATE = 0.009
ETF3 = ["TQQQ", "SPXL", "TNA", "SOXL", "FAS", "LABU", "TECL"]
BASE = ["SPY", "QQQ", "IWM", "SMH"]
CRYPTO = ["MSTR", "COIN", "MARA", "RIOT", "CLSK", "HUT", "BITF", "GBTC", "BITO", "IBIT"]
HALF_MAX_MI = 215  # SPY 마지막 1분봉이 이보다 이르면 반나절 장


# ---------------------------------------------------------------- 자료
# 상장일 (그 전 같은 코드는 다른 종목이라 버림)
INCEPTION = {"SPXL": "2008-11-05", "BGU": "2008-11-05", "UPRO": "2009-06-25", "TQQQ": "2010-02-11", "TNA": "2008-11-05",
             "SOXL": "2010-03-11", "FAS": "2008-11-06", "LABU": "2015-05-28", "TECL": "2008-12-17", "COIN": "2021-04-14",
             "BITO": "2021-10-19", "IBIT": "2024-01-11", "HUT": "2021-06-14", "BITF": "2021-06-24", "GBTC": "2015-05-04",
             "SMH": "2000-06-05", "QQQ": "1999-03-10", "IWM": "2000-05-22"}
RULE_TICKERS = {"b21": ETF3 + BASE + ["BGU"], "b22": ["TNA", "IWM", "SPXL", "BGU", "TQQQ", "SPY"],
                "b23": ["SPXL", "BGU", "TNA", "TQQQ", "SPY"], "b25": CRYPTO + ["SPY"]}


def load(prep, tickers):
    """정규장 1분봉만 (GBTC는 b25 대체용 장전 가격 때문에 장전도)."""
    fs = sorted(glob.glob(os.path.join(prep, "bars_*.parquet")))
    parts = []
    for f in fs:
        x = pq.read_table(f, filters=[("ticker", "in", list(tickers))]).to_pandas()
        parts.append(x[(x.mi >= 0) | (x.ticker == "GBTC")])
    df = pd.concat(parts, ignore_index=True)
    df["d"] = pd.to_datetime(df.d).dt.date
    first = df.ticker.map({k: dt.date.fromisoformat(v) for k, v in INCEPTION.items()}).fillna(dt.date(1900, 1, 1))
    df = df[df.d >= first]
    # SPXL = 2012년 이름이 바뀌기 전 BGU (같은 펀드). 둘 다 있는 날은 SPXL
    bgu = df[df.ticker == "BGU"]
    have = set(df.loc[df.ticker == "SPXL", "d"])
    df = pd.concat([df[df.ticker != "BGU"], bgu[~bgu.d.isin(have)].assign(ticker="SPXL")], ignore_index=True)
    return df.sort_values(["ticker", "d", "mi"]).reset_index(drop=True)


def build(df):
    reg = df[df.mi >= 0]
    days = {(tk, d): {k: g[k].to_numpy() for k in ("mi", "o", "h", "l", "c", "v")}
            for (tk, d), g in reg.groupby(["ticker", "d"], sort=False)}
    daily = (reg[reg.mi <= etf.CLOSE].groupby(["ticker", "d"], sort=False)
             .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"),
                  last_mi=("mi", "max")).reset_index())
    dv = (reg.c * reg.v).groupby([reg.ticker, reg.d]).sum()
    daily["dv"] = dv.reindex(pd.MultiIndex.from_frame(daily[["ticker", "d"]])).to_numpy()
    spy = daily[daily.ticker == "SPY"].set_index("d")
    cal = list(spy.index)
    half = set(spy.index[spy.last_mi <= HALF_MAX_MI])
    return days, daily, cal, half, spy


def tstat(g):
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g.std() > 0 else np.nan


def pct(x, k=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{k}f}%"


ETF_HEAD = ("| 변형 | 거래 수 | 비용 전 평균 | 중앙값 | 승률 | t값 | 비용 후 (왕복 0.7%) | 비용 후 (왕복 0.54%) | 2010~ 평균 | 2018~ 평균 | 관문 |",
            "|---|---|---|---|---|---|---|---|---|---|---|")


def etf_row(name, s, judge=True, min_n=30):
    if s is None or len(s) == 0:
        return f"| {name} | 0 |" + " |" * 9
    g = s.gross
    yr = pd.to_datetime(s.d).dt.year
    n7 = portfolio.net_return(g, FEE + SLIP).mean()
    n5 = portfolio.net_return(g, FEE + SLIP_ETF).mean()
    if not judge:
        v = "(판정 안 함)"
    elif len(g) < min_n:
        v = "표본 부족"
    else:
        v = "통과" if g.mean() >= GATE else "탈락"
    return (f"| {name} | {len(g)} | {pct(g.mean())} | {pct(g.median())} | {(g > 0).mean():.0%} | {tstat(g):.2f} | "
            f"{pct(n7)} | {pct(n5)} | {pct(g[yr >= 2010].mean())} | {pct(g[yr >= 2018].mean())} | {v} |")


def yearly_table(sel, names, codes):
    lines = ["| 해 | " + " | ".join(names[c] for c in codes) + " |", "|---" * (len(codes) + 1) + "|"]
    years = sorted({y for c in codes if len(sel[c]) for y in pd.to_datetime(sel[c].d).dt.year})
    for y in years:
        cells = []
        for c in codes:
            s = sel[c]
            g = s.gross[pd.to_datetime(s.d).dt.year == y] if len(s) else pd.Series(dtype=float)
            cells.append(f"{pct(g.mean())} ({len(g)})" if len(g) else "")
        lines.append(f"| {y} | " + " | ".join(cells) + " |")
    return lines


def account_lines(sel, names, codes, cal, spy, weight=0.5, slip=SLIP):
    """하루 계좌 weight(50%)를 그날 거래들에 똑같이 나눔."""
    days = pd.DatetimeIndex(pd.to_datetime(cal))
    s0 = portfolio.stats(spy.c.pct_change().fillna(0.0).set_axis(days))
    out = ["| 변형 | 연수익률 | 최대낙폭 | 샤프 | 거래 수 |", "|---|---|---|---|---|",
           f"| SPY 보유 (배당 제외) | {pct(s0['cagr'], 1)} | {pct(s0['mdd'], 1)} | {s0['sharpe']:.2f} | |"]
    for c in codes:
        s = sel[c]
        if not len(s):
            continue
        s = s.assign(d=pd.to_datetime(s.d))
        start = s.d.min()
        net = portfolio.net_return(s.gross, FEE + slip)
        w = weight / s.groupby("d").d.transform("size")
        r = (w * net).groupby(s.d).sum().reindex(days[days >= start], fill_value=0.0)
        st = portfolio.stats(r)
        out.append(f"| {names[c]} ({start.year}~) | {pct(st['cagr'], 1)} | {pct(st['mdd'], 1)} | {st['sharpe']:.2f} | {len(s)} |")
    return out


def daily_arrays(daily, tk):
    g = daily[daily.ticker == tk].reset_index(drop=True)
    return g


# ---------------------------------------------------------------- b21
def rule_b21(days, daily, cal, half, spy):
    bad = etf.split_days(daily[["ticker", "d", "o", "c"]])
    rows = []
    for tk in ETF3 + BASE:
        g = daily_arrays(daily, tk)
        if len(g) < 30:
            continue
        nr7, idnr4 = etf.narrow_flags(g.h.to_numpy(), g.l.to_numpy())
        st = etf.stretch(g.o.to_numpy(), g.h.to_numpy(), g.l.to_numpy())
        for i in range(11, len(g)):
            d = g.d[i]
            if (tk, d) in bad or (tk, g.d[i - 1]) in bad:
                continue
            day = days.get((tk, d))
            if day is None:
                continue
            hf = d in half
            base = dict(ticker=tk, d=d, nr7=bool(nr7[i - 1]), idnr4=bool(idnr4[i - 1]), base=tk in BASE)
            for code, kw in (("open", {}), ("stretch", dict(stop_mode="stretch")), ("early", dict(t_hi=60))):
                t = etf.crabel(day, st[i], half=hf, **kw)
                if t:
                    rows.append(dict(base, mode=code, **t))
    t = pd.DataFrame(rows)
    narrow = t.nr7 | t.idnr4
    e3 = ~t.base
    sel = {"N1": t[e3 & narrow & (t["mode"] == "open")],
           "N2": t[e3 & (t["mode"] == "open")],
           "N3-nr7": t[e3 & t.nr7 & (t["mode"] == "open")],
           "N3-id": t[e3 & t.idnr4 & (t["mode"] == "open")],
           "N4": t[e3 & narrow & (t["mode"] == "stretch")],
           "N5": t[t.base & narrow & (t["mode"] == "open")],
           "N6": t[e3 & narrow & (t["mode"] == "early")]}
    names = {"N1": "N1 기본 (전날 NR7 또는 ID/NR4, 시가+스트레치 돌파, 손절 시가 아래)",
             "N2": "N2 좁은 날 조건 없이 (모든 날)", "N3-nr7": "N3 전날 NR7만", "N3-id": "N3 전날 ID/NR4만",
             "N4": "N4 손절 시가 - 스트레치", "N5": "N5 기초지수 ETF(SPY·QQQ·IWM·SMH), N1 규칙 (판정 안 함)",
             "N6": "N6 매수 10:30까지만"}
    lines = ["# b21 백테스트 결과: 크래블 좁은 날 다음 시가 범위 돌파 (3배 ETF)", "",
             "대상: TQQQ, SPXL(2012년 전은 같은 펀드의 옛 이름 BGU), TNA, SOXL, FAS, LABU, TECL (자료에 있는 기간). "
             "주식분할로 보이는 날 앞뒤 10거래일 제외. 스트레치 = 직전 10일 min(|시가-고가|, |시가-저가|) 평균. "
             "비용 왕복 0.7% (참고 0.54%). 관문: 비용 전 거래당 ≥ +0.9%.", "",
             "## 1차 관문", "", *ETF_HEAD]
    passed = []
    for c in sel:
        lines.append(etf_row(names[c], sel[c], judge=c != "N5"))
        s = sel[c]
        if c != "N5" and len(s) >= 30 and s.gross.mean() >= GATE:
            passed.append(c)
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else '없음 → b21 탈락'}**", ""]
    for c in ("N1", "N2"):
        lines += [f"## ETF별 ({names[c]})", "", "| ETF | 기간 | 거래 수 | 비용 전 평균 | 중앙값 | 승률 | t값 | 손절 비율 |", "|---|---|---|---|---|---|---|---|"]
        for tk, s in sel[c].groupby("ticker"):
            g = s.gross
            lines.append(f"| {tk} | {s.d.min()}~ | {len(g)} | {pct(g.mean())} | {pct(g.median())} | {(g > 0).mean():.0%} | "
                         f"{tstat(g):.2f} | {(s.reason == '손절').mean():.0%} |")
        lines.append("")
    s = sel["N1"]
    if len(s):
        lines.append(f"N1 청산 이유: {', '.join(f'{k} {v}' for k, v in s.reason.value_counts().items())}. "
                     f"매수 시각 중앙값 9:30 + {int(s.entry_mi.median())}분, 평균 손절폭 {pct(s.stop_dist.mean())}.")
    lines += ["", "## 해마다 (비용 전 거래당 평균, 괄호는 거래 수)", "", *yearly_table(sel, names, ["N1", "N2", "N5"])]
    lines += ["", "## 계좌 기준 (하루 계좌 50%를 그날 거래에 나눔, 비용 왕복 0.7%)", "",
              *account_lines(sel, names, ["N1", "N2", "N4", "N6"], cal, spy)]
    return t, lines


# ---------------------------------------------------------------- b22
def holiday_days(cal):
    cal_set = set(cal)
    hol = etf.nyse_holidays(cal[0].year, cal[-1].year)
    hol = {d: n for d, n in hol.items() if cal[0] < d < cal[-1]}
    traded = sorted(d for d in hol if d in cal_set)
    pre = {}
    idx = {d: i for i, d in enumerate(cal)}
    for d, name in hol.items():
        if d in cal_set:
            continue
        k = int(np.searchsorted(np.array(cal), d)) - 1
        p = cal[k]
        pre.setdefault(p, name)
    # 규칙 밖의 휴장(자료에 거래가 없는 평일)도 세어 둔다
    wk = [cal[0] + dt.timedelta(days=i) for i in range((cal[-1] - cal[0]).days)]
    other = [d for d in wk if d.weekday() < 5 and d not in cal_set and d not in hol]
    return pre, idx, traded, other


def rule_b22(days, daily, cal, half, spy):
    pre, idx, traded, other = holiday_days(cal)
    closes = {tk: daily[daily.ticker == tk].set_index("d").c for tk in ("SPY", "IWM", "TNA", "SPXL", "TQQQ")}
    rows = []
    for p, name in pre.items():
        i = idx[p]
        holy_thu = name == "성금요일"
        for k, tag in ((0, "pre1"), (1, "pre2")):
            if i - k < 1:
                continue
            d = cal[i - k]
            for tk in ("TNA", "IWM", "SPXL", "TQQQ", "SPY"):
                day = days.get((tk, d))
                if day is None:
                    continue
                t = etf.open_to_close(day, half=d in half)
                if t:
                    rows.append(dict(ticker=tk, d=d, tag=tag, holiday=name, holy=holy_thu, kind="oc", **t))
            if k == 0:
                prev = cal[i - 1]
                for tk in ("SPY", "IWM"):
                    c = closes[tk]
                    if d in c.index and prev in c.index:
                        rows.append(dict(ticker=tk, d=d, tag=tag, holiday=name, holy=holy_thu, kind="cc",
                                         gross=c[d] / c[prev] - 1, entry_mi=-1, exit_mi=389, reason="마감"))
    t = pd.DataFrame(rows)
    oc, p1 = t.kind == "oc", t.tag == "pre1"
    sel = {"V1": t[oc & p1 & (t.ticker == "TNA")],
           "V1-iwm": t[oc & p1 & (t.ticker == "IWM")],
           "V2-spxl": t[oc & p1 & (t.ticker == "SPXL")],
           "V2-tqqq": t[oc & p1 & (t.ticker == "TQQQ")],
           "V3-iwm": t[(t.kind == "cc") & (t.ticker == "IWM")],
           "V3-spy": t[(t.kind == "cc") & (t.ticker == "SPY")],
           "V4": t[oc & (t.tag == "pre2") & (t.ticker == "TNA")],
           "V6": t[oc & p1 & (t.ticker == "TNA") & t.holy]}
    names = {"V1": "V1 기본: TNA 휴장 전날 시가 → 종가", "V1-iwm": "(참고) IWM 휴장 전날 시가 → 종가 (판정 안 함)",
             "V2-spxl": "V2 SPXL", "V2-tqqq": "V2 TQQQ",
             "V3-iwm": "V3 논문 재현: IWM 전날 종가 → 휴장 전날 종가 (판정 안 함)",
             "V3-spy": "V3 논문 재현: SPY 전날 종가 → 휴장 전날 종가 (판정 안 함)",
             "V4": "V4 휴장 2일 전 TNA 시가 → 종가", "V6": "V6 성목요일(성금요일 전날)만, TNA"}
    judge = {"V1", "V2-spxl", "V2-tqqq", "V4", "V6"}
    lines = ["# b22 백테스트 결과: 휴장 전날 효과 (3배 ETF, 시가 → 종가)", "",
             f"휴장일: NYSE 정규 휴장 규칙으로 계산한 날 중 자료에 거래가 없는 날 (자료에 거래가 있는 '휴장일' {len(traded)}개는 뺌: "
             f"{', '.join(map(str, traded[:8]))}{' …' if len(traded) > 8 else ''}). 규칙 밖에 거래가 없는 평일 {len(other)}개 "
             f"(임시 휴장 등: {', '.join(map(str, other[:12]))}{' …' if len(other) > 12 else ''})는 휴장으로 보지 않음. "
             "반나절 장은 12:59 종가로 청산. 비용 왕복 0.7% (참고 0.54%). 관문 ≥ +0.9%.", "",
             "## 1차 관문", "", *ETF_HEAD]
    passed = []
    for c in sel:
        lines.append(etf_row(names[c], sel[c], judge=c in judge, min_n=15))
        if c in judge and len(sel[c]) >= 15 and sel[c].gross.mean() >= GATE:
            passed.append(c)
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else '없음 → b22 탈락'}**", ""]
    lines += ["## 기간 나눠 보기 (비용 전 평균, 괄호는 횟수)", "", "| 변형 | 1993~2009 | 2010~2026 |", "|---|---|---|"]
    for c in ("V1", "V1-iwm", "V3-iwm", "V3-spy", "V2-spxl", "V2-tqqq"):
        s = sel[c]
        y = pd.to_datetime(s.d).dt.year
        a, b = s.gross[y <= 2009], s.gross[y >= 2010]
        lines.append(f"| {names[c]} | {pct(a.mean())} ({len(a)}) | {pct(b.mean())} ({len(b)}) |")
    lines += ["", "## V5 휴장일별 (TNA 시가 → 종가 / IWM 시가 → 종가 / SPY 종가 → 종가, 비용 전)", "",
              "| 휴장 | TNA 횟수 | TNA 평균 | TNA 승률 | IWM 평균 (횟수) | SPY 종가→종가 평균 (횟수) |", "|---|---|---|---|---|---|"]
    for name in ("새해", "마틴 루터 킹", "대통령의 날", "성금요일", "메모리얼 데이", "준틴스", "독립기념일", "노동절", "추수감사절", "크리스마스"):
        a = sel["V1"][sel["V1"].holiday == name].gross
        b = sel["V1-iwm"][sel["V1-iwm"].holiday == name].gross
        c = sel["V3-spy"][sel["V3-spy"].holiday == name].gross
        lines.append(f"| {name} 전날 | {len(a)} | {pct(a.mean())} | {(a > 0).mean() if len(a) else 0:.0%} | {pct(b.mean())} ({len(b)}) | {pct(c.mean())} ({len(c)}) |")
    lines += ["", "## 해마다 (비용 전 평균, 괄호는 횟수)", "", *yearly_table(sel, names, ["V1", "V2-spxl", "V2-tqqq", "V3-spy"])]
    lines += ["", "## 계좌 기준 (휴장 전날만 계좌 50%, 비용 왕복 0.7%)", "", *account_lines(sel, names, ["V1", "V2-spxl", "V2-tqqq", "V4"], cal, spy)]
    return t, lines


# ---------------------------------------------------------------- b23
def month_numbers(cal):
    """거래일 -> (번호): 그달 마지막 거래일 -1, -2 …, 그달 첫 거래일 +1, +2 …"""
    s = pd.Series(cal, index=pd.to_datetime(cal))
    out = {}
    for _, g in s.groupby([s.index.year, s.index.month]):
        ds = list(g)
        for k, d in enumerate(ds):
            out.setdefault(d, set()).update({k + 1, -(len(ds) - k)})
    return out


def rule_b23(days, daily, cal, half, spy):
    num = month_numbers(cal)
    closes = spy.c
    spy_m = closes.groupby(pd.to_datetime(closes.index).to_period("M")).last()
    prev_m_ret = (spy_m / spy_m.shift() - 1).shift()  # 그달 기준, 전달 수익률
    rows = []
    cal_arr = cal
    for i, d in enumerate(cal_arr[1:], start=1):
        ks = [k for k in num[d] if -4 <= k <= 3]
        if not ks:
            continue
        pm = prev_m_ret.get(pd.Period(d, "M"), np.nan)
        for tk in ("SPXL", "TNA", "TQQQ", "SPY"):
            day = days.get((tk, d))
            if day is None:
                continue
            t = etf.open_to_close(day, half=d in half)
            if t:
                for k in ks:
                    rows.append(dict(ticker=tk, d=d, k=k, kind="oc", prev_neg=pm < 0, **t))
        p = cal_arr[i - 1]
        for k in ks:
            rows.append(dict(ticker="SPY", d=d, k=k, kind="cc", prev_neg=pm < 0, gross=closes[d] / closes[p] - 1,
                             entry_mi=-1, exit_mi=389, reason="마감"))
    t = pd.DataFrame(rows)
    oc = t.kind == "oc"
    sel, names = {}, {}
    for tk in ("SPXL", "TNA", "TQQQ"):
        for k in (-4, -3, -2, -1, 1, 2, 3):
            if tk != "SPXL" and k < -1:
                continue
            c = f"{tk}{k:+d}"
            sel[c] = t[oc & (t.ticker == tk) & (t.k == k)]
            lab = "M1" if tk == "SPXL" and k >= -1 else ("M4" if tk == "SPXL" else "M2")
            names[c] = f"{lab} {tk} 거래일 {k:+d}"
    sel["M6"] = t[oc & (t.ticker == "SPXL") & (t.k == 1) & t.prev_neg]
    names["M6"] = "M6 SPXL 거래일 +1, 전달 SPY 마이너스였던 달만"
    for k in (-4, -3, -2, -1, 1, 2, 3):
        sel[f"M3{k:+d}"] = t[(t.kind == "cc") & (t.k == k)]
        names[f"M3{k:+d}"] = f"M3 논문 재현: SPY 종가→종가 거래일 {k:+d} (판정 안 함)"
    sel["other"] = None
    judge = {c for c in sel if c[:2] != "M3" and c != "other"}
    order = ["SPXL-1", "SPXL+1", "SPXL+2", "SPXL+3", "TNA-1", "TNA+1", "TNA+2", "TNA+3", "TQQQ-1", "TQQQ+1", "TQQQ+2",
             "TQQQ+3", "SPXL-4", "SPXL-3", "SPXL-2", "M6"] + [f"M3{k:+d}" for k in (-4, -3, -2, -1, 1, 2, 3)]
    # 대조: SPY 모든 날 종가→종가 평균
    allcc = closes.pct_change().dropna()
    lines = ["# b23 백테스트 결과: 월말·월초 효과 (3배 ETF, 시가 → 종가)", "",
             "거래일 번호: -1 = 그달 마지막 거래일, +1 = 다음 달 첫 거래일. SPXL은 2012년 전 옛 이름 BGU 포함. "
             f"비용 왕복 0.7% (참고 0.54%). 관문 ≥ +0.9%, 하루씩 따로 판정. 참고로 SPY 모든 날 종가→종가 평균 {pct(allcc.mean(), 3)}.", "",
             "## 1차 관문", "", *ETF_HEAD]
    passed = []
    for c in order:
        lines.append(etf_row(names[c], sel[c], judge=c in judge))
        if c in judge and len(sel[c]) >= 30 and sel[c].gross.mean() >= GATE:
            passed.append(c)
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else '없음 → b23 탈락'}**", ""]
    lines += ["## M5 기간 나눠 보기 (비용 전 평균, 괄호는 횟수)", "", "| 거래일 | SPY 시가→종가 1993~2009 | SPY 시가→종가 2010~2026 | SPY 종가→종가 1993~2009 | SPY 종가→종가 2010~2026 | SPXL 시가→종가 2010~2026 |", "|---|---|---|---|---|---|"]
    for k in (-4, -3, -2, -1, 1, 2, 3):
        a = t[oc & (t.ticker == "SPY") & (t.k == k)]
        b = t[(t.kind == "cc") & (t.k == k)]
        x = t[oc & (t.ticker == "SPXL") & (t.k == k)]
        ya, yb, yx = (pd.to_datetime(z.d).dt.year for z in (a, b, x))
        cells = [a.gross[ya <= 2009], a.gross[ya >= 2010], b.gross[yb <= 2009], b.gross[yb >= 2010], x.gross[yx >= 2010]]
        lines.append(f"| {k:+d} | " + " | ".join(f"{pct(z.mean(), 3)} ({len(z)})" for z in cells) + " |")
    lines += ["", "## 해마다 (SPXL, 비용 전 평균, 괄호는 횟수)", "", *yearly_table(sel, names, ["SPXL-1", "SPXL+1", "SPXL+2", "SPXL+3"])]
    four = t[oc & (t.ticker == "SPXL") & t.k.isin([-1, 1, 2, 3])].drop_duplicates(["d"])
    sel["4d"], names["4d"] = four, "SPXL 4일 모두 (-1, +1, +2, +3)"
    lines += ["", "## 계좌 기준 (해당일 계좌 50%, 비용 왕복 0.7%)", "", *account_lines(sel, names, ["4d", "SPXL-1", "SPXL+1", "SPXL-3"], cal, spy)]
    return t, lines


# ---------------------------------------------------------------- b25
def btc_series(prep):
    p = os.path.join(prep, "btc.parquet")
    if not os.path.exists(p):
        return None, None
    b = pd.read_parquet(p)
    step = pd.Timedelta(minutes=1) if b.src.iloc[0] == "binance-1m" else pd.Timedelta(hours=1)
    s = b.assign(end=b.t + step).set_index("end").c.sort_index()
    return s, b.src.iloc[0]


def price_at(s, ts):
    k = s.index.searchsorted(ts, side="right") - 1
    return float(s.iloc[k]) if k >= 0 and ts - s.index[k] <= pd.Timedelta(hours=2) else np.nan


def rule_b25(days, daily, cal, half, spy, df, prep):
    btc, src = btc_series(prep)
    gb = None
    if btc is None:
        src = "GBTC 대체 (금요일 종가 → 월요일 9:00 장전 가격)"
        pm = df[(df.ticker == "GBTC") & (df.mi < -30)].groupby("d").c.last()
        gb = (pm, daily[daily.ticker == "GBTC"].set_index("d").c)
    # 관련주 크기 조건: 직전 20일 평균 거래대금, 전날 종가
    feats = {}
    for tk in CRYPTO:
        g = daily[daily.ticker == tk].set_index("d")
        if not len(g):
            continue
        feats[tk] = pd.DataFrame({"adv": g.dv.rolling(20, min_periods=10).mean().shift(), "pc": g.c.shift()})
    rows, events = [], []
    cal_d = [d for d in cal if d >= dt.date(2017, 8, 1)]
    for i in range(1, len(cal_d)):
        d, p = cal_d[i], cal_d[i - 1]
        weekend = (d - p).days >= 3
        if btc is not None:
            b0 = price_at(btc, pd.Timestamp(dt.datetime.combine(p, dt.time(16, 0))))
            b9 = price_at(btc, pd.Timestamp(dt.datetime.combine(d, dt.time(9, 0))))
        else:
            b0, b9 = gb[1].get(p, np.nan), gb[0].get(d, np.nan)
        ret = b9 / b0 - 1 if b0 and b0 > 0 else np.nan
        if np.isnan(ret):
            continue
        events.append(dict(d=d, weekend=weekend, ret=ret))
        if abs(ret) < 0.03:
            continue
        for tk, f in feats.items():
            if d not in f.index or not (f.adv[d] >= 10e6 and f.pc[d] >= 2):
                continue
            day = days.get((tk, d))
            if day is None:
                continue
            base = dict(ticker=tk, d=d, weekend=weekend, ret=ret)
            t = etf.first15_break(day)
            if t:
                up = np.nan
                if btc is not None:
                    te = pd.Timestamp(dt.datetime.combine(d, dt.time(9, 30))) + pd.Timedelta(minutes=t["entry_mi"])
                    up = price_at(btc, te) > b9
                rows.append(dict(base, mode="break", btc_up=up, **t))
            t = etf.open_to_close(day)
            if t:
                rows.append(dict(base, mode="open", btc_up=np.nan, stop_dist=np.nan, **t))
    t = pd.DataFrame(rows)
    ev = pd.DataFrame(events)
    wk, br, op = t.weekend, t["mode"] == "break", t["mode"] == "open"

    def top5(s):
        return s.sort_values(["d", "entry_mi"]).groupby("d").head(5)
    sel = {"C1": top5(t[wk & br & (t.ret >= 0.05)]), "C2-3": top5(t[wk & br & (t.ret >= 0.03)]),
           "C2-10": top5(t[wk & br & (t.ret >= 0.10)]), "C3": top5(t[wk & op & (t.ret >= 0.05)]),
           "C4": top5(t[wk & op & (t.ret <= -0.05)]), "C5": top5(t[br & (t.ret >= 0.05)]),
           "C6": top5(t[wk & br & (t.ret >= 0.05) & (t.btc_up == True)])}  # noqa: E712
    names = {"C1": "C1 기본 (주말 비트코인 +5%, 첫 15분 고가 돌파, 손절 첫 15분 저가)", "C2-3": "C2 주말 +3% 이상",
             "C2-10": "C2 주말 +10% 이상", "C3": "C3 돌파 기다리지 않고 9:31 시가 매수", "C4": "C4 주말 급락(-5% 이하) 뒤 9:31 매수",
             "C5": "C5 모든 거래일 (전날 16:00 → 9:00 +5%)", "C6": "C6 C1 + 매수 때 비트코인이 9:00보다 높음"}
    for s in sel.values():
        s["low_px"] = s.entry_px < 5
        s["hurdle"] = np.where(s.low_px, 0.013, GATE)
        s["cost"] = FEE + np.where(s.low_px, 0.003, SLIP)
    nwk = ev.weekend.sum()
    lines = ["# b25 백테스트 결과: 주말 비트코인 급등 → 월요일 비트코인 관련주", "",
             f"비트코인 가격: {src}. 주말 = 직전 거래일과 3일 이상 떨어진 거래일 (월요일, 월요일 휴장이면 화요일). "
             f"주말 {nwk}번 중 비트코인 +5% 이상 {int((ev.weekend & (ev.ret >= 0.05)).sum())}번, -5% 이하 {int((ev.weekend & (ev.ret <= -0.05)).sum())}번 "
             f"(모든 거래일 +5% 이상 {int((ev.ret >= 0.05).sum())}번). 관련주: {', '.join(CRYPTO)} (2017-08 ~, 직전 20일 평균 거래대금 ≥ 1,000만 달러, 전날 종가 ≥ 2달러). "
             "비용: 수수료 0.25% + 미끄러짐 0.1% (매수가 5달러 미만이면 0.3%). 관문: 비용 전 거래당 ≥ +0.9% (5달러 미만 +1.3%) → 관문 초과분 ≥ 0. 하루 최대 5종목.", "",
             "## 1차 관문", "",
             "| 변형 | 거래 수 | 날 수 | 비용 전 평균 | 중앙값 | 승률 | t값 | 2025~26 평균 (수) | 관문 초과분 | 비용 후 평균 | 평균 손절폭 | 청산 이유 | 관문 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    passed = []
    for c, s in sel.items():
        if not len(s):
            lines.append(f"| {names[c]} | 0 |" + " |" * 11)
            continue
        g = s.gross
        y = pd.to_datetime(s.d).dt.year
        ex = (g - s.hurdle).mean()
        net = portfolio.net_return(g, s.cost)
        v = "표본 부족" if len(g) < 30 else ("통과" if ex >= 0 else "탈락")
        if v == "통과":
            passed.append(c)
        lines.append(f"| {names[c]} | {len(g)} | {s.d.nunique()} | {pct(g.mean())} | {pct(g.median())} | {(g > 0).mean():.0%} | {tstat(g):.2f} | "
                     f"{pct(g[y >= 2025].mean())} ({int((y >= 2025).sum())}) | {ex * 100:+.2f}%p | {pct(net.mean())} | {pct(s.stop_dist.mean())} | "
                     f"{', '.join(f'{k} {v}' for k, v in s.reason.value_counts().items())} | {v} |")
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else '없음 → b25 탈락'}**", ""]
    lines += ["## 믿을 만한가 (같은 날 거래는 서로 비슷하게 움직이므로 날 단위로도 봄)", "",
              "| 변형 | 날 수 | 날 평균 | 날 단위 t값 | 상위 10개 거래 뺀 평균 | 상위 3일 뺀 평균 | 2021년 뺀 평균 | 2024년 뺀 평균 |", "|---|---|---|---|---|---|---|---|"]
    for c, s in sel.items():
        if not len(s):
            continue
        dm = s.groupby("d").gross.mean()
        y = pd.to_datetime(s.d).dt.year
        top3 = dm.nlargest(3).index
        lines.append(f"| {names[c]} | {len(dm)} | {pct(dm.mean())} | {tstat(dm):.2f} | {pct(s.gross.sort_values(ascending=False).iloc[10:].mean())} | "
                     f"{pct(s.gross[~s.d.isin(top3)].mean())} | {pct(s.gross[y != 2021].mean())} | {pct(s.gross[y != 2024].mean())} |")
    th = "| 날짜 | 종목 | 비트코인 주말 | 매수가 | 비용 전 |"
    for c in ("C1", "C4"):
        s = sel[c]
        lines += ["", f"{names[c]} 가장 좋은 10개 / 나쁜 5개:", "", th, "|---|---|---|---|---|"]
        lines += [f"| {r.d} | {r.ticker} | {r.ret * 100:+.1f}% | {r.entry_px:.2f} | {r.gross * 100:+.1f}% |" for r in pd.concat([s.nlargest(10, "gross"), s.nsmallest(5, "gross")]).itertuples()]
    lines += ["", "## 종목별 (C1 / C3, 비용 전)", "", "| 종목 | C1 수 | C1 평균 | C3 수 | C3 평균 |", "|---|---|---|---|---|"]
    for tk in CRYPTO:
        a, b = sel["C1"][sel["C1"].ticker == tk].gross, sel["C3"][sel["C3"].ticker == tk].gross
        if len(a) or len(b):
            lines.append(f"| {tk} | {len(a)} | {pct(a.mean())} | {len(b)} | {pct(b.mean())} |")
    lines += ["", "## 기간 나눠 보기 (비용 전 평균, 괄호는 수)", "", "| 변형 | 2017~2020 | 2021~2026 |", "|---|---|---|"]
    for c, s in sel.items():
        y = pd.to_datetime(s.d).dt.year if len(s) else pd.Series(dtype=int)
        a, b = (s.gross[y <= 2020], s.gross[y >= 2021]) if len(s) else (pd.Series(dtype=float),) * 2
        lines.append(f"| {names[c]} | {pct(a.mean())} ({len(a)}) | {pct(b.mean())} ({len(b)}) |")
    lines += ["", "## 매수가 구간별 (C1, 비용 전)", "", "| 매수가 | 수 | 평균 | 중앙값 |", "|---|---|---|---|"]
    for lo, hi in ((1, 5), (5, 20), (20, 1e9)):
        g = sel["C1"].gross[(sel["C1"].entry_px >= lo) & (sel["C1"].entry_px < hi)]
        lines.append(f"| {lo}~{'' if hi > 1e8 else hi}달러 | {len(g)} | {pct(g.mean())} | {pct(g.median())} |")
    lines += ["", "## 해마다 (비용 전 평균, 괄호는 수)", "", *yearly_table(sel, names, ["C1", "C3", "C4", "C5"])]
    days_idx = pd.DatetimeIndex(pd.to_datetime([d for d in cal if d >= dt.date(2017, 8, 1)]))
    s0 = portfolio.stats(spy.c.pct_change().fillna(0.0).set_axis(pd.DatetimeIndex(pd.to_datetime(cal)))[days_idx])
    lines += ["", "## 계좌 기준 (손절 시 계좌 0.5% 위험, 한 종목 최대 20%, 동시 5종목, 하루 -1.5% 중단, 2017-08~)", "",
              "| 변형 | 연수익률 | 최대낙폭 | 샤프 | 실제 거래 수 |", "|---|---|---|---|---|",
              f"| SPY 보유 (배당 제외) | {pct(s0['cagr'], 1)} | {pct(s0['mdd'], 1)} | {s0['sharpe']:.2f} | |"]
    for c, s in sel.items():
        if not len(s):
            continue
        k = s.assign(d=pd.to_datetime(s.d), rank=range(len(s)),
                     stop_dist=s.stop_dist.fillna(portfolio.RISK / 0.2))
        r, taken = portfolio.daily_returns(k, days_idx, k.cost, max_w=0.2, max_pos=5)
        st = portfolio.stats(r)
        lines.append(f"| {names[c]} | {pct(st['cagr'], 1)} | {pct(st['mdd'], 1)} | {st['sharpe']:.2f} | {len(taken)} |")
    up5 = ev[ev.weekend & (ev.ret >= 0.05)]
    lines += ["", "주말 비트코인 +5% 이상 횟수 (해마다): " + ", ".join(f"{y} {n}" for y, n in pd.to_datetime(up5.d).dt.year.value_counts().sort_index().items())]
    top = ev[ev.weekend].sort_values("ret")
    lines += ["", "## 눈으로 확인: 주말 비트코인 움직임 큰 날", "",
              "가장 많이 오른 5번: " + ", ".join(f"{r.d} {r.ret * 100:+.1f}%" for r in top.tail(5).iloc[::-1].itertuples()),
              "", "가장 많이 내린 5번: " + ", ".join(f"{r.d} {r.ret * 100:+.1f}%" for r in top.head(5).itertuples())]
    return t, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_etf")
    ap.add_argument("--rule", required=True, choices=["b21", "b22", "b23", "b25"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"reports/{a.rule}"
    os.makedirs(out, exist_ok=True)
    df = load(a.prep, RULE_TICKERS[a.rule])
    days, daily, cal, half, spy = build(df)
    print("tickers:", daily.groupby("ticker").d.agg(["min", "max", "size"]).to_string(), flush=True)
    if a.rule == "b25":
        t, lines = rule_b25(days, daily, cal, half, spy, df, a.prep)
    else:
        t, lines = {"b21": rule_b21, "b22": rule_b22, "b23": rule_b23}[a.rule](days, daily, cal, half, spy)
    t.to_csv(os.path.join(out, "trades.csv.gz"), index=False)
    text = "\n".join(lines) + "\n"
    open(os.path.join(out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
