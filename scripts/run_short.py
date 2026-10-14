"""b26~b30(1~3일 짧은 보유) 백테스트: prep_etf 폴더 -> 결과표(마크다운).

    python scripts/run_short.py --prep prep_etf --rule b26 --out reports/b26

종목 짝: SPY -> SPXL(2012년 전 BGU), QQQ -> TQQQ, IWM -> TNA. 신호는 기초 ETF로 판단하고
기초 ETF(1배, 참고), 합성 3배(기초 ETF로 만든 것, 긴 기간), 실제 3배 ETF(2009년 무렵부터) 세 가지로 수익을 잰다.
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from daytrade import etf, portfolio, short  # noqa: E402
from run_etf import INCEPTION, load  # noqa: E402

FEE, SLIP, SLIP_ETF = 0.0025, 0.001, 0.0002
COST, COST_REF = FEE + SLIP, FEE + SLIP_ETF
GATE = 0.009
PAIRS = {"SPY": "SPXL", "QQQ": "TQQQ", "IWM": "TNA"}
KINDS = ("1x", "syn3", "real3")
KIND_NAME = {"1x": "1배", "syn3": "합성 3배", "real3": "실제 3배"}
HALF_MAX_MI = 215


# ---------------------------------------------------------------- 자료
class Book:
    """기초 ETF와 짝 3배 ETF의 하루 표, 날짜 -> 줄 번호."""

    def __init__(self, bars):
        spy_last = bars[bars.ticker == "SPY"].groupby("d").mi.max()
        self.half = set(spy_last.index[spy_last <= HALF_MAX_MI])
        t, self.splits = short.split_adjust(short.daily_table(bars, self.half))
        self.tab = {tk: g.reset_index(drop=True) for tk, g in t.groupby("ticker", sort=False)}
        self.idx = {tk: {d: i for i, d in enumerate(g.d)} for tk, g in self.tab.items()}
        self.cal = list(self.tab["SPY"].d)
        self._base = {}
        # 종목 줄 번호 -> SPY 거래일 번호 (QQQ는 2004~2011년 이름이 QQQQ라 자료가 비어 있음: 빈 곳을 건너는 거래는 버림)
        spy_pos = {d: i for i, d in enumerate(self.cal)}
        self.pos = {tk: np.array([spy_pos.get(d, -10 ** 6) for d in g.d]) for tk, g in self.tab.items()}
        # 1분봉은 FOMC 발표 직전 가격에만 씀: 기초 ETF와 3배 ETF만 남김 (메모리 절약)
        keep = set(PAIRS) | set(PAIRS.values())
        bb = bars[bars.ticker.isin(keep)]
        self.by_day = {k: (g.mi.to_numpy(), g.c.to_numpy()) for k, g in bb.groupby(["ticker", "d"], sort=False)}

    def continuous(self, tk, i, j):
        p = self.pos[tk]
        return p[i] >= 0 and p[j] - p[i] == j - i

    def legs(self, base, i, j):
        """기초 ETF i일 종가 매수 -> j일 종가 매도의 세 가지 수익과 날마다 수익(계좌 계산용)."""
        g = self.tab[base]
        if not self.continuous(base, i, j):
            return {}
        c = g.c.to_numpy()
        ds = list(g.d[i:j + 1])
        out = {"1x": (short.hold_return(c, i, j), ds, c[i + 1:j + 1] / c[i:j] - 1)}
        r1 = c[i + 1:j + 1] / c[i:j] - 1
        out["syn3"] = (short.synthetic_return(c, i, j), ds, 3 * r1 - short.SYN_DAILY_COST)
        lev = PAIRS.get(base)
        if lev in self.idx:
            a, b = self.idx[lev].get(ds[0]), self.idx[lev].get(ds[-1])
            if a is not None and b is not None and b - a == j - i and self.continuous(lev, a, b):
                cl = self.tab[lev].c.to_numpy()
                out["real3"] = (short.hold_return(cl, a, b), ds, cl[a + 1:b + 1] / cl[a:b] - 1)
        return out

    def baseline(self, base, kind, n, since=None):
        """아무 날에나 사서 n거래일 들고 있었을 때의 평균 (같은 종목, 같은 기간). 신호가 시장 평균보다 나은지 보려고."""
        key = (base, kind, n, since)
        if key not in self._base:
            if kind == "real3":
                tk = PAIRS.get(base)
                if tk not in self.tab:
                    self._base[key] = np.nan
                    return np.nan
                g = self.tab[tk]
            else:
                g = self.tab[base]
            c = g.c.to_numpy()
            ok = np.ones(len(c) - n, bool) if since is None else (pd.to_datetime(g.d[:len(c) - n]).dt.year >= since).to_numpy()
            p = self.pos[g.ticker.iloc[0]]
            ok = ok & (p[:-n] >= 0) & (p[n:] - p[:-n] == n)
            if kind == "syn3":
                r = c[1:] / c[:-1] - 1
                lg = np.r_[0.0, np.cumsum(np.log1p(3 * r))]
                v = np.exp(lg[n:] - lg[:-n]) - 1 - short.SYN_DAILY_COST * n
            else:
                v = c[n:] / c[:-n] - 1
            self._base[key] = float(v[ok].mean())
        return self._base[key]

    def price_before(self, tk, d, mi):
        """그날 mi분 전 마지막 1분봉 종가."""
        x = self.by_day.get((tk, d))
        if x is None:
            return None
        k = np.searchsorted(x[0], mi) - 1
        f = self.tab[tk].f.iloc[self.idx[tk][d]] if d in self.idx[tk] else 1.0
        return float(x[1][k]) * f if k >= 0 else None


def make_rows(book, base, entries, variant):
    """entries: [(i, j, extra dict)] 기초 ETF 줄 번호 -> 거래 줄들."""
    rows = []
    g = book.tab[base]
    for i, j, extra in entries:
        if i < 0 or j >= len(g) or j <= i:
            continue
        for kind, (gross, ds, path) in book.legs(base, i, j).items():
            rows.append(dict(variant=variant, base=base, kind=kind, d=g.d[i], exit_d=g.d[j], days=j - i,
                             gross=gross, path=path, path_d=ds[1:], **extra))
    return rows


# ---------------------------------------------------------------- 표
def tstat(g):
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g.std() > 0 else np.nan


def pct(x, k=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{k}f}%"


HEAD = ("| 변형 | 종목 | 기간 | 거래 수 | 평균 보유일 | 비용 전 평균 | **아무 날이나 같은 일수 보유** | 중앙값 | 승률 | t값 | 상위 10개 뺀 평균 "
        "| 비용 후 (왕복 0.7%) | 비용 후 (왕복 0.54%) | 2010~ 평균 | 2018~ 평균 | 관문 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def label(base, kind):
    return base if kind == "1x" else (f"{base}×3 합성" if kind == "syn3" else PAIRS.get(base, f"{base}×3"))


BOOK = None


def row(name, s, min_n=30):
    if not len(s):
        return f"| {name} | | | 0 |" + " |" * 12, False
    g = s.gross.reset_index(drop=True)
    yr = pd.to_datetime(s.d).dt.year.reset_index(drop=True)
    top = g.sort_values().iloc[:-10].mean() if len(g) > 20 else np.nan
    ok = len(g) >= min_n and g.mean() >= GATE
    verdict = "표본 부족" if len(g) < min_n else ("통과" if ok else "탈락")
    base, kind = s.base.iloc[0], s.kind.iloc[0]
    span = f"{yr.min()}~{yr.max()}"
    bl = BOOK.baseline(base, kind, max(1, int(round(s.days.mean()))), since=int(yr.min())) if BOOK is not None else np.nan
    return (f"| {name} | {label(base, kind)} | {span} | {len(g)} | {s.days.mean():.1f} | {pct(g.mean())} | {pct(bl)} | {pct(g.median())} | "
            f"{(g > 0).mean():.0%} | {tstat(g):.2f} | {pct(top)} | {pct(portfolio.net_return(g, COST).mean())} | "
            f"{pct(portfolio.net_return(g, COST_REF).mean())} | {pct(g[yr >= 2010].mean())} | {pct(g[yr >= 2018].mean())} | {verdict} |"), ok


def gate_table(t, names, bases=("SPY", "QQQ", "IWM")):
    lines = ["## 1차 관문 (비용 전 거래당 ≥ +0.9%, 30번 이상)", "", *HEAD]
    passed = []
    for v in names:
        for base in bases:
            for kind in KINDS:
                s = t[(t.variant == v) & (t.base == base) & (t.kind == kind)]
                if not len(s):
                    continue
                line, ok = row(names[v], s)
                lines.append(line)
                if ok:
                    passed.append(f"{v} {label(base, kind)}")
    lines += ["", f"**관문 통과: {', '.join(passed) if passed else '없음 → 탈락'}**", ""]
    return lines, passed


def period_table(t, names, kind="syn3"):
    cuts = [(1993, 2003), (2004, 2009), (2010, 2017), (2018, 2026)]
    lines = [f"## 기간 나눠 보기 ({KIND_NAME[kind]}, 비용 전 평균, 괄호는 수)", "",
             "| 변형 | 종목 | " + " | ".join(f"{a}~{b}" for a, b in cuts) + " |", "|---|---|" + "---|" * len(cuts)]
    for v in names:
        for base in ("SPY", "QQQ", "IWM"):
            s = t[(t.variant == v) & (t.base == base) & (t.kind == kind)]
            if not len(s):
                continue
            y = pd.to_datetime(s.d).dt.year
            cells = []
            for a, b in cuts:
                g = s.gross[(y >= a) & (y <= b)]
                cells.append(f"{pct(g.mean())} ({len(g)})" if len(g) else "")
            lines.append(f"| {names[v]} | {label(base, kind)} | " + " | ".join(cells) + " |")
    return lines


def yearly(t, codes):
    """codes: [(variant, base, kind, 이름)]"""
    sel = [(t[(t.variant == v) & (t.base == b) & (t.kind == k)], n) for v, b, k, n in codes]
    years = sorted({y for s, _ in sel for y in pd.to_datetime(s.d).dt.year})
    lines = ["| 해 | " + " | ".join(n for _, n in sel) + " |", "|---" * (len(sel) + 1) + "|"]
    for y in years:
        cells = []
        for s, _ in sel:
            g = s.gross[pd.to_datetime(s.d).dt.year == y]
            cells.append(f"{pct(g.mean())} ({len(g)})" if len(g) else "")
        lines.append(f"| {y} | " + " | ".join(cells) + " |")
    return lines


def account(t, codes, book):
    """계좌 100%를 신호 날에만 넣음. 날마다 수익을 쌓고 매수·매도 날 비용을 뺌. 겹치는 거래는 앞 거래가 끝난 뒤만."""
    cal = pd.DatetimeIndex(pd.to_datetime(book.cal))
    spy = book.tab["SPY"].set_index(pd.to_datetime(book.tab["SPY"].d)).c
    lines = ["| 변형 | 종목 | 기간 | 연수익률 | 최대낙폭 | 샤프 | 시장에 있던 날 | 같은 기간 SPY 보유 |", "|---|---|---|---|---|---|---|---|"]
    for v, b, k, n in codes:
        s = t[(t.variant == v) & (t.base == b) & (t.kind == k)].sort_values("d")
        if not len(s):
            continue
        r = pd.Series(0.0, index=cal)
        busy_until = None
        for x in s.itertuples():
            if busy_until is not None and x.d < busy_until:
                continue
            path = 1 + np.array(x.path, float)
            path[0] *= 1 - COST
            path[-1] /= 1 + COST
            path -= 1
            for d, rr in zip(x.path_d, path):
                r[pd.Timestamp(d)] += rr
            busy_until = x.exit_d
        start = pd.Timestamp(s.d.min())
        r = r[r.index >= start]
        st = portfolio.stats(r)
        sp = portfolio.stats(spy[spy.index >= start].pct_change().fillna(0.0))
        inmkt = (r != 0).mean()
        lines.append(f"| {n} | {label(b, k)} | {start.year}~ | {pct(st['cagr'], 1)} | {pct(st['mdd'], 1)} | {st['sharpe']:.2f} | "
                     f"{inmkt:.0%} | {pct(sp['cagr'], 1)} / {pct(sp['mdd'], 1)} |")
    return lines


COMMON_NOTE = ("판단은 15:55 가격(반나절 장 12:55), 매수·매도는 15:59 1분봉 종가(반나절 12:59). 합성 3배 = 기초 ETF 종가→종가 × 3을 날마다 쌓고 "
               "보유 하루 0.02%를 뺌 (3배 ETF 운용·차입 비용 대신). 실제 3배: SPXL(2012년 전 BGU, 2008-11~), TQQQ(2010-02~), TNA(2008-11~). "
               "비용: 수수료 0.25% + 미끄러짐 0.1% (한쪽, 왕복 0.7%), 참고로 ETF 미끄러짐 0.02% (왕복 0.54%). 관문: 비용 전 거래당 ≥ +0.9%.")


# ---------------------------------------------------------------- b26 월말월초
def month_last_idx(cal):
    s = pd.Series(range(len(cal)), index=pd.to_datetime(cal))
    return list(s.groupby([s.index.year, s.index.month]).max())


def rule_b26(book):
    rows = []
    spy = book.tab["SPY"]
    dspy = short.decision_series(spy)
    for base in PAIRS:
        g = book.tab[base]
        last = month_last_idx(list(g.d))
        spy_i = book.idx["SPY"]
        for L in last[:-1]:
            e = {}
            for v, (a, b) in {"T1": (-1, 3), "T2": (0, 3), "T3": (-1, 1), "T4": (-4, 3)}.items():
                e[v] = (L + a, L + b)
            for v, (i, j) in e.items():
                rows += make_rows(book, base, [(i, j, {})], v)
            i, j = e["T1"]
            if 0 <= i < len(g):
                k = spy_i.get(g.d[i])
                if k is not None and spy.cd[k] > dspy.sma200[k]:
                    rows += make_rows(book, base, [(i, j, {})], "T5")
    t = pd.DataFrame(rows)
    names = {"T1": "T1 기본: 월말 2일 전 종가 매수 → 다음 달 3번째 거래일 종가 (4일 보유)",
             "T2": "T2 월말 마지막 날 종가 매수 → 3번째 거래일 종가",
             "T3": "T3 월말 2일 전 종가 → 첫 거래일 종가",
             "T4": "T4 월말 5일 전 종가 → 3번째 거래일 종가",
             "T5": "T5 T1 + SPY가 200일 평균 위일 때만"}
    lines = ["# b26 백테스트 결과: 월말·월초 묶음 보유 (1~4일)", "",
             "월말 거래일 번호는 각 기초 ETF의 거래일로 셈 (-1 = 그달 마지막 거래일, +1 = 다음 달 첫 거래일). " + COMMON_NOTE, ""]
    g_lines, passed = gate_table(t, names)
    lines += g_lines + period_table(t, names, "syn3") + [""] + period_table(t, names, "1x")
    lines += ["", "## 해마다 (T1, 비용 전 평균, 괄호는 수)", "",
              *yearly(t, [("T1", "SPY", "1x", "SPY"), ("T1", "SPY", "syn3", "SPY×3 합성"), ("T1", "SPY", "real3", "SPXL"),
                          ("T1", "QQQ", "real3", "TQQQ"), ("T1", "IWM", "real3", "TNA")])]
    lines += ["", "## 계좌 기준 (신호 때 계좌 100%, 비용 왕복 0.7%)", "",
              *account(t, [("T1", "SPY", "syn3", "T1"), ("T1", "SPY", "real3", "T1"), ("T1", "QQQ", "real3", "T1"),
                           ("T2", "SPY", "real3", "T2"), ("T5", "SPY", "real3", "T5")], book)]
    return t, lines


# ---------------------------------------------------------------- b27 휴장 전후
def pre_holiday_idx(cal):
    cal_set = set(cal)
    hol = etf.nyse_holidays(cal[0].year, cal[-1].year)
    arr = np.array(cal)
    out = {}
    for d, name in hol.items():
        if d in cal_set or not cal[0] < d < cal[-1]:
            continue
        k = int(np.searchsorted(arr, d)) - 1
        out.setdefault(k, name)
    return out


def rule_b27(book):
    rows = []
    for base in PAIRS:
        g = book.tab[base]
        pre = pre_holiday_idx(list(g.d))
        for p, name in pre.items():
            for v, (a, b) in {"H1": (-1, 0), "H2": (-2, 1), "H3": (-1, 1), "H4": (0, 1)}.items():
                rows += make_rows(book, base, [(p + a, p + b, {"holiday": name})], v)
    t = pd.DataFrame(rows)
    names = {"H1": "H1 기본: 휴장 2일 전 종가 매수 → 휴장 전날 종가 (1일)",
             "H2": "H2 휴장 3일 전 종가 → 휴장 다음 날 종가 (3일)",
             "H3": "H3 휴장 2일 전 종가 → 휴장 다음 날 종가 (2일)",
             "H4": "H4 휴장 전날 종가 → 휴장 다음 날 종가 (휴장 넘기기)"}
    lines = ["# b27 백테스트 결과: 휴장 전후 묶음 보유", "",
             "휴장 = NYSE 정규 휴장 규칙으로 계산한 날 중 자료에 거래가 없는 날 (주말과 붙은 휴장 포함). " + COMMON_NOTE, ""]
    g_lines, passed = gate_table(t, names)
    lines += g_lines + period_table(t, names, "syn3")
    lines += ["", "## 휴장별 (H3, 합성 3배 SPY / 실제 SPXL, 비용 전)", "", "| 휴장 | 합성 3배 수 | 합성 3배 평균 | SPXL 수 | SPXL 평균 |", "|---|---|---|---|---|"]
    for name in ("새해", "마틴 루터 킹", "대통령의 날", "성금요일", "메모리얼 데이", "준틴스", "독립기념일", "노동절", "추수감사절", "크리스마스"):
        a = t[(t.variant == "H3") & (t.base == "SPY") & (t.kind == "syn3") & (t.holiday == name)].gross
        b = t[(t.variant == "H3") & (t.base == "SPY") & (t.kind == "real3") & (t.holiday == name)].gross
        lines.append(f"| {name} | {len(a)} | {pct(a.mean())} | {len(b)} | {pct(b.mean())} |")
    lines += ["", "## 해마다 (H3, 비용 전 평균, 괄호는 수)", "",
              *yearly(t, [("H3", "SPY", "syn3", "SPY×3 합성"), ("H3", "SPY", "real3", "SPXL"), ("H3", "QQQ", "real3", "TQQQ"),
                          ("H3", "IWM", "real3", "TNA")])]
    lines += ["", "## 계좌 기준 (신호 때 계좌 100%, 비용 왕복 0.7%)", "",
              *account(t, [("H1", "SPY", "real3", "H1"), ("H3", "SPY", "syn3", "H3"), ("H3", "SPY", "real3", "H3"),
                           ("H3", "IWM", "real3", "H3")], book)]
    return t, lines


# ---------------------------------------------------------------- b28 FOMC 전
def fomc_table():
    p = os.path.join(os.path.dirname(HERE), "daytrade", "fomc_dates.csv")
    f = pd.read_csv(p)
    f["d"] = pd.to_datetime(f.date).dt.date
    f["ann_mi"] = f.announce.map(lambda s: int(s[:2]) * 60 + int(s[3:]) - 570)
    return f


def rule_b28(book):
    f = fomc_table()
    rows = []
    for base in PAIRS:
        g = book.tab[base]
        c = g.c.to_numpy()
        for r in f.itertuples():
            k = book.idx[base].get(r.d)
            if k is None or k < 2:
                continue
            # F3: 전날 종가 → 발표일 종가 (보통 종가→종가)
            rows += make_rows(book, base, [(k - 1, k, {"ann": r.announce})], "F3")
            # F1·F2: 발표 1분 전 가격에 매도
            for v, back in (("F1", 1), ("F2", 2)):
                i = k - back
                if not book.continuous(base, i, k):
                    continue
                px = book.price_before(base, r.d, r.ann_mi)
                if px is None:
                    continue
                # 날마다 수익: 그 전 날들은 종가→종가, 발표일은 전날 종가 → 발표 직전
                prev = c[i + 1:k] / c[i:k - 1] - 1 if k - 1 > i else np.array([])
                part = px / c[k - 1] - 1
                paths = {"1x": np.r_[prev, part]}
                # 합성: 하루씩 3배
                paths["syn3"] = 3 * paths["1x"] - short.SYN_DAILY_COST
                lev = PAIRS[base]
                if lev in book.idx:
                    a, b = book.idx[lev].get(g.d[i]), book.idx[lev].get(r.d)
                    if a is not None and b is not None and b - a == back:
                        pl = book.price_before(lev, r.d, r.ann_mi)
                        if pl is not None:
                            cl = book.tab[lev].c.to_numpy()
                            paths["real3"] = np.r_[cl[a + 1:b] / cl[a:b - 1] - 1, pl / cl[b - 1] - 1]
                for kind, path in paths.items():
                    gross = float(np.prod(1 + path) - 1)
                    rows.append(dict(variant=v, base=base, kind=kind, d=g.d[i], exit_d=r.d, days=back, gross=gross,
                                     path=path, path_d=list(g.d[i + 1:k + 1]), ann=r.announce))
    t = pd.DataFrame(rows)
    names = {"F1": "F1 기본: 발표 전날 종가 매수 → 발표 1분 전 매도",
             "F2": "F2 발표 2일 전 종가 → 발표 1분 전",
             "F3": "F3 발표 전날 종가 → 발표일 종가"}
    lines = ["# b28 백테스트 결과: FOMC 발표 전 드리프트", "",
             f"FOMC 정례 회의 {len(f)}번 (daytrade/fomc_dates.csv, 2011-01 ~ 2026-03, b07과 같은 목록). 발표 시각: 2011~2012년 14:15 "
             "(기자회견 날 12:30), 2013년부터 14:00. 매도는 발표 1분 전 1분봉 종가. " + COMMON_NOTE, ""]
    g_lines, passed = gate_table(t, names)
    lines += g_lines
    lines += ["## 기간 나눠 보기 (비용 전 평균, 괄호는 수)", "", "| 변형 | 종목 | 2011~2014 | 2015~2019 | 2020~2026 |", "|---|---|---|---|---|"]
    for v in names:
        for base in PAIRS:
            for kind in ("1x", "real3"):
                s = t[(t.variant == v) & (t.base == base) & (t.kind == kind)]
                if not len(s):
                    continue
                y = pd.to_datetime(s.d).dt.year
                cells = [s.gross[(y >= a) & (y <= b)] for a, b in ((2010, 2014), (2015, 2019), (2020, 2026))]
                lines.append(f"| {names[v]} | {label(base, kind)} | " + " | ".join(f"{pct(z.mean())} ({len(z)})" for z in cells) + " |")
    lines += ["", "## 해마다 (F1, 비용 전 평균, 괄호는 수)", "",
              *yearly(t, [("F1", "SPY", "1x", "SPY"), ("F1", "SPY", "real3", "SPXL"), ("F1", "QQQ", "real3", "TQQQ")])]
    lines += ["", "## 계좌 기준 (신호 때 계좌 100%, 비용 왕복 0.7%)", "",
              *account(t, [("F1", "SPY", "real3", "F1"), ("F1", "QQQ", "real3", "F1"), ("F3", "SPY", "real3", "F3")], book)]
    return t, lines


# ---------------------------------------------------------------- b29 RSI(2) 과매도
def signal_trades(book, base, enter, exit_fn, variant):
    g = book.tab[base]
    x = short.decision_series(g)
    cd = g.cd.to_numpy()
    ok = enter(x, g)
    entries, busy = [], -1
    for i in np.nonzero(ok)[0]:
        if i <= busy or i >= len(g) - 1:
            continue
        j = exit_fn(x, cd, i)
        entries.append((i, j, {}))
        busy = j - 1  # 매도한 날 종가에 다시 살 수는 있음
    return make_rows(book, base, entries, variant)


def rule_b29(book):
    rows = []
    sma = lambda x, g: g.cd.to_numpy() > x.sma200.to_numpy()  # noqa: E731
    ex_sma5 = lambda x, cd, i: short.exit_rsi(cd, x.sma5.to_numpy(), i, 10)  # noqa: E731
    ex_3 = lambda x, cd, i: min(i + 3, len(cd) - 1)  # noqa: E731
    variants = {
        "R1": (lambda x, g: (x.rsi2.to_numpy() < 10) & sma(x, g), ex_sma5),
        "R2": (lambda x, g: (x.rsi2.to_numpy() < 5) & sma(x, g), ex_sma5),
        "R3": (lambda x, g: x.rsi2.to_numpy() < 10, ex_sma5),
        "R4": (lambda x, g: (x.rsi2.to_numpy() < 10) & sma(x, g), ex_3),
    }
    for base in PAIRS:
        for v, (en, ex) in variants.items():
            rows += signal_trades(book, base, en, ex, v)
    t = pd.DataFrame(rows)
    names = {"R1": "R1 기본: RSI(2) < 10 + 200일 평균 위 → 5일 평균 위로 마감하면 매도 (최대 10일)",
             "R2": "R2 RSI(2) < 5",
             "R3": "R3 200일 평균 조건 없이",
             "R4": "R4 R1 매수, 3일 뒤 종가 매도"}
    lines = ["# b29 백테스트 결과: 단기 과매도 반등 (코너스 RSI(2))", "",
             "신호는 기초 ETF(SPY·QQQ·IWM)로 판단: RSI(2)·200일 평균·5일 평균은 전날까지 종가 + 오늘 15:55 가격으로 계산. "
             "한 번에 한 거래만 (보유 중 신호는 건너뜀). " + COMMON_NOTE, ""]
    g_lines, passed = gate_table(t, names)
    lines += g_lines + period_table(t, names, "syn3") + [""] + period_table(t, names, "1x")
    lines += ["", "## 해마다 (R1, 비용 전 평균, 괄호는 수)", "",
              *yearly(t, [("R1", "SPY", "1x", "SPY"), ("R1", "SPY", "syn3", "SPY×3 합성"), ("R1", "SPY", "real3", "SPXL"),
                          ("R1", "QQQ", "real3", "TQQQ"), ("R1", "IWM", "real3", "TNA")])]
    lines += ["", "## 계좌 기준 (신호 때 계좌 100%, 비용 왕복 0.7%)", "",
              *account(t, [("R1", "SPY", "syn3", "R1"), ("R1", "SPY", "real3", "R1"), ("R1", "QQQ", "real3", "R1"),
                           ("R1", "IWM", "real3", "R1"), ("R2", "SPY", "real3", "R2"), ("R3", "QQQ", "real3", "R3")], book)]
    return t, lines


# ---------------------------------------------------------------- b30 IBS
def rule_b30(book):
    rows = []
    nxt = lambda x, cd, i: i + 1  # noqa: E731
    variants = {
        "I1": (lambda x, g: x.ibs.to_numpy() < 0.2, nxt),
        "I2": (lambda x, g: x.ibs.to_numpy() < 0.1, nxt),
        "I3": (lambda x, g: (x.ibs.to_numpy() < 0.2) & (g.cd.to_numpy() > x.sma200.to_numpy()), nxt),
        "I4": (lambda x, g: x.ibs.to_numpy() < 0.2, lambda x, cd, i: short.exit_ibs(x.ibs.to_numpy(), i, 0.7, 5)),
    }
    for base in PAIRS:
        for v, (en, ex) in variants.items():
            rows += signal_trades(book, base, en, ex, v)
    t = pd.DataFrame(rows)
    names = {"I1": "I1 기본: IBS < 0.2 종가 매수 → 다음 날 종가",
             "I2": "I2 IBS < 0.1",
             "I3": "I3 I1 + 200일 평균 위",
             "I4": "I4 I1 매수, IBS > 0.7로 마감하는 날 매도 (최대 5일)"}
    lines = ["# b30 백테스트 결과: IBS(그날 종가 위치) 반전", "",
             "IBS = (판단가 - 그날 저가) ÷ (그날 고가 - 그날 저가), 15:55까지의 고가·저가 기준. 한 번에 한 거래만. " + COMMON_NOTE, ""]
    g_lines, passed = gate_table(t, names)
    lines += g_lines + period_table(t, names, "syn3") + [""] + period_table(t, names, "1x")
    lines += ["", "## 해마다 (I1, 비용 전 평균, 괄호는 수)", "",
              *yearly(t, [("I1", "SPY", "1x", "SPY"), ("I1", "SPY", "syn3", "SPY×3 합성"), ("I1", "SPY", "real3", "SPXL"),
                          ("I1", "QQQ", "real3", "TQQQ"), ("I1", "IWM", "real3", "TNA")])]
    lines += ["", "## 계좌 기준 (신호 때 계좌 100%, 비용 왕복 0.7%)", "",
              *account(t, [("I1", "SPY", "real3", "I1"), ("I1", "QQQ", "real3", "I1"), ("I2", "QQQ", "real3", "I2"),
                           ("I4", "SPY", "real3", "I4"), ("I3", "QQQ", "syn3", "I3")], book)]
    return t, lines


# ---------------------------------------------------------------- b29 다듬기 + 신호 합치기
COST_LEVELS = (("표준 0.25%+0.1%", 0.0035), ("이벤트 0.07%+0.1%", 0.0017), ("이벤트 0.07%+0.02%", 0.0009))


def inst(base, kind):
    return {"1x": base, "syn2": f"{base}×2", "syn3": f"{base}×3", "real3": PAIRS.get(base, f"{base}×3")}[kind]


def pick(t, v, base, kind):
    if "variant" not in t:
        return pd.DataFrame(columns=["variant", "base", "kind", "d", "exit_d", "days", "gross", "path", "path_d"])
    return t[(t.variant == v) & (t.base == base) & (t.kind == kind)].sort_values("d")


def acct_table(book, combos, start=None):
    """combos: [(이름, [거래표...], SPY 보유 여부)] -> 비용 단계별 연수익·최대낙폭 표."""
    spy = book.tab["SPY"]
    spy_r = dict(zip(spy.d, spy.c.pct_change().fillna(0.0)))
    lines = ["| 조합 | 기간 | " + " | ".join(f"{n}: 연수익 / 최대낙폭 / 샤프" for n, _ in COST_LEVELS) + " | 3배 들고 있던 날 |",
             "|---|---|" + "---|" * len(COST_LEVELS) + "---|"]
    series = {}
    for name, parts, idle in combos:
        trades = [(inst(x.base, x.kind), list(x.path_d), np.asarray(x.path, float)) for s in parts for x in s.itertuples()]
        if not trades:
            continue
        st = start or min(s.d.min() for s in parts if len(s))
        days = [d for d in book.cal if d > st]
        cells, inmkt = [], None
        for cname, c in COST_LEVELS:
            r = short.combine_daily(trades, days, c, spy_r if idle else None)
            series[(name, cname)] = r
            m = portfolio.stats(r)
            cells.append(f"{pct(m['cagr'], 1)} / {pct(m['mdd'], 1)} / {m['sharpe']:.2f}")
            if inmkt is None:
                held = {d for _, ds, _ in trades for d in ds}
                inmkt = np.mean([d in held for d in days])
        lines.append(f"| {name} | {pd.Timestamp(days[0]).year}~ | " + " | ".join(cells) + f" | {inmkt:.0%} |")
    firsts = sorted({next(d for d in book.cal if d > (start or min(s.d.min() for s in parts if len(s)))) for _, parts, _ in combos})
    firsts = [f for k, f in enumerate(firsts) if k == 0 or pd.Timestamp(f).year != pd.Timestamp(firsts[k - 1]).year]
    for f in firsts:
        days = [d for d in book.cal if d >= f]
        m = portfolio.stats(pd.Series([spy_r[d] for d in days]))
        lines.append(f"| SPY 그냥 보유 | {pd.Timestamp(days[0]).year}~ | {pct(m['cagr'], 1)} / {pct(m['mdd'], 1)} / {m['sharpe']:.2f} | | | 0% |")
    return lines, series


def yearly_series(series, keys, book, start):
    spy = book.tab["SPY"]
    sp = pd.Series(spy.c.to_numpy(), index=pd.to_datetime(spy.d)).pct_change().fillna(0.0)
    sp = sp[sp.index > pd.Timestamp(start)]
    cols = {f"{n} ({c})": portfolio.yearly(series[(n, c)]) for n, c in keys}
    cols["SPY"] = portfolio.yearly(sp)
    lines = ["| 해 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for y in cols["SPY"].index:
        lines.append(f"| {y} | " + " | ".join(pct(v.get(y, np.nan), 1) for v in cols.values()) + " |")
    return lines


def rule_b29x(book):
    sma = lambda x, g: g.cd.to_numpy() > x.sma200.to_numpy()  # noqa: E731
    r1 = lambda x, g: (x.rsi2.to_numpy() < 10) & sma(x, g)  # noqa: E731
    r2 = lambda x, g: (x.rsi2.to_numpy() < 5) & sma(x, g)  # noqa: E731

    def ex(stop=None, max_days=10):
        def f(x, cd, i):
            entry = cd[i]
            if stop is None:
                return short.exit_rsi(cd, x.sma5.to_numpy(), i, max_days)
            return short.exit_rsi_stop(cd, x.sma5.to_numpy(), i, entry, stop, max_days)
        return f
    variants = {"R1": (r1, ex()), "R2": (r2, ex()), "R1S3": (r1, ex(0.03)), "R1S5": (r1, ex(0.05)),
                "R2S3": (r2, ex(0.03)), "R1M5": (r1, ex(None, 5))}
    rows = []
    for base in ("SPY", "QQQ"):
        for v, (en, e) in variants.items():
            rows += signal_trades(book, base, en, e, v)
    t = pd.DataFrame(rows)
    names = {"R1": "R1 b29 기본 (RSI(2) < 10, 200일 위, 5일 평균 위 마감에 매도, 최대 10일)",
             "R2": "R2 RSI(2) < 5",
             "R1S3": "R1 + 손절: 기초 ETF 판단가가 매수가보다 3% 아래 (3배로 약 -9%)",
             "R1S5": "R1 + 손절 5% (3배로 약 -15%)",
             "R2S3": "R2 + 손절 3%",
             "R1M5": "R1, 최대 5일"}
    lines = ["# b29 다듬기 + 신호 합치기", "",
             "1부: b29 RSI(2)에 손절·짧은 최대 보유를 붙이고 수수료 단계별로 계좌를 다시 계산. 손절은 15:55 판단가로 보고 15:59 종가에 판다 (장중 손절 아님). "
             "2부: b29·b30 I4·b28 F2 신호를 한 계좌로 합침. 같은 날 여러 신호면 돈을 똑같이 나누고, 신호 없는 날은 현금 또는 SPY 보유. "
             "비용은 종목별 비중이 바뀐 만큼 매김 (SPY → SPXL로 바꾸면 SPY 매도 + SPXL 매수 둘 다). " + COMMON_NOTE, ""]
    g_lines, _ = gate_table(t, names, bases=("SPY", "QQQ"))
    lines += g_lines
    lines += ["## 1부 계좌: 수수료 단계별 (신호 때 계좌 100%, 신호 없으면 현금)", ""]
    combos = []
    for kind, kname in (("real3", "SPXL"), ("syn3", "SPY×3 합성")):
        for v in variants:
            combos.append((f"{v} {kname}", [pick(t, v, "SPY", kind)], False))
    combos.append(("R1 TQQQ", [pick(t, "R1", "QQQ", "real3")], False))
    combos.append(("R1S3 TQQQ", [pick(t, "R1S3", "QQQ", "real3")], False))
    a_lines, _ = acct_table(book, combos)
    lines += a_lines + ["", "합성 3배는 1994년부터라 실제 3배(2009년 무렵부터)와 기간이 달라요. SPY 줄은 각 기간의 SPY 보유.", ""]

    # 2부: 합치기 (실제 3배, FOMC 목록이 2011년부터)
    t28, _ = rule_b28(book)
    t30, _ = rule_b30(book)
    R1, R2 = pick(t, "R1", "SPY", "real3"), pick(t, "R2", "SPY", "real3")
    R1S = pick(t, "R1S5", "SPY", "real3")
    I4 = pick(t30, "I4", "SPY", "real3")
    F2 = pick(t28, "F2", "QQQ", "real3")
    F1 = pick(t28, "F1", "QQQ", "real3")
    start = dt.date(2011, 1, 1)
    clip = lambda s: s[s.d >= start]  # noqa: E731
    R1, R2, R1S, I4, F2, F1 = map(clip, (R1, R2, R1S, I4, F2, F1))
    combos = [("K1 R1만, 나머지 현금", [R1], False),
              ("K2 R1만, 나머지 SPY", [R1], True),
              ("K3 R1+I4+F2, 나머지 현금", [R1, I4, F2], False),
              ("K4 R1+I4+F2, 나머지 SPY", [R1, I4, F2], True),
              ("K5 R1+I4, 나머지 SPY", [R1, I4], True),
              ("K6 R2+I4+F2, 나머지 SPY", [R2, I4, F2], True),
              ("K7 R1S5+I4+F1, 나머지 SPY", [R1S, I4, F1], True),
              ("K8 I4만, 나머지 SPY", [I4], True)]
    lines += ["## 2부: 신호 합치기 (실제 3배 SPXL·TQQQ, 2011~2026-03)", "",
              f"신호 수: R1 {len(R1)}, R2 {len(R2)}, R1S5 {len(R1S)}, I4 {len(I4)}, F2 TQQQ {len(F2)}, F1 TQQQ {len(F1)}.", ""]
    k_lines, series = acct_table(book, combos, start=start)
    lines += k_lines
    lines += ["", "## 해마다 (2부)", "",
              *yearly_series(series, [("K2 R1만, 나머지 SPY", COST_LEVELS[0][0]), ("K4 R1+I4+F2, 나머지 SPY", COST_LEVELS[0][0]),
                                      ("K4 R1+I4+F2, 나머지 SPY", COST_LEVELS[1][0]), ("K3 R1+I4+F2, 나머지 현금", COST_LEVELS[0][0]),
                                      ("K3 R1+I4+F2, 나머지 현금", COST_LEVELS[1][0]), ("K3 R1+I4+F2, 나머지 현금", COST_LEVELS[2][0])],
                             book, start)]
    # 합성 3배로 긴 기간 (FOMC 빼고)
    S1, SI = pick(t, "R1", "SPY", "syn3"), pick(t30, "I4", "SPY", "syn3")
    lines += ["", "## 2부 긴 기간: 합성 3배 (FOMC 없이, 1994~)", ""]
    l_lines, _ = acct_table(book, [("L1 R1 합성, 나머지 SPY", [S1], True), ("L2 R1+I4 합성, 나머지 SPY", [S1, SI], True),
                                    ("L3 R1+I4 합성, 나머지 현금", [S1, SI], False)],
                             start=max(S1.d.min(), SI.d.min()))
    lines += l_lines
    # 2020년(코로나 폭락·반등) 한 해에 기대는지 보려고 그해를 빼고 다시 잼
    lines += ["", "## 2020년 빼고 (2부, 그해 수익을 0으로)", "", "| 조합 | 비용 | 연수익 | 최대낙폭 |", "|---|---|---|---|"]
    for (n, c), r in series.items():
        r2 = r.copy()
        r2[r2.index.year == 2020] = 0.0
        m = portfolio.stats(r2)
        lines.append(f"| {n} | {c} | {pct(m['cagr'], 1)} | {pct(m['mdd'], 1)} |")
    spy = book.tab["SPY"]
    sp = pd.Series(spy.c.to_numpy(), index=pd.to_datetime(spy.d)).pct_change().fillna(0.0)
    sp = sp[sp.index > pd.Timestamp(start)]
    sp[sp.index.year == 2020] = 0.0
    m = portfolio.stats(sp)
    lines.append(f"| SPY 그냥 보유 | | {pct(m['cagr'], 1)} | {pct(m['mdd'], 1)} |")
    return pd.concat([t, t28[t28.variant.isin(["F1", "F2"])], t30[t30.variant == "I4"]], ignore_index=True), lines


# ---------------------------------------------------------------- b31~b40 짧은 보유 모음 2
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
ASSETS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EFA", "EEM"]
INCEPTION2 = {**{t: "1998-12-22" for t in SECTORS}, "TLT": "2002-07-30", "GLD": "2004-11-18", "EFA": "2001-08-27",
              "EEM": "2003-04-14"}
_IND = {}


def ind(book, base):
    """판단용 지표 묶음 (판단가 cd = 15:55 가격을 오늘 종가 자리에)."""
    if base not in _IND:
        g = book.tab[base]
        x = short.decision_series(g)
        c, cd = g.c.to_numpy(float), g.cd.to_numpy(float)
        z = dict(g=g, c=c, cd=cd, o=g.o.to_numpy(float), rsi=x.rsi2.to_numpy(), rsi_c=short.rsi_wilder(c, 2),
                 sma200=x.sma200.to_numpy(), sma5=x.sma5.to_numpy())
        z["prev_c"] = np.r_[np.nan, c[:-1]]
        z["up200"] = cd > z["sma200"]
        for k in (5, 7, 10, 20):
            z[f"m{k}"], z[f"sd{k}"], z[f"lo{k}"], z[f"hi{k}"] = short.decision_window(c, cd, k)
        sma200_c = pd.Series(c).rolling(200).mean().to_numpy()
        z["up200_prev"] = np.r_[False, (c > sma200_c)[:-1]]  # 전날 종가가 200일 평균 위 (시가 매수용)
        _IND[base] = z
    return _IND[base]


def trades_from(book, base, enter, exit_at, variant):
    """enter: 판단일 bool 배열, exit_at(i) -> 매도 줄 번호. 한 번에 한 거래."""
    n = len(book.tab[base])
    entries, busy = [], -1
    for i in np.nonzero(enter)[0]:
        if i <= busy or i >= n - 1:
            continue
        j = exit_at(i)
        entries.append((i, j, {}))
        busy = j - 1
    return make_rows(book, base, entries, variant)


def custom_rows(book, base, i, j, variant, start="close", end="close"):
    """start/end = 'open' 또는 'close'. i일 시가/종가 매수 -> j일 시가/종가 매도."""
    g = book.tab[base]
    if i < 1 or j >= len(g) or j < i or (start == "close" and j == i) or not book.continuous(base, i, j):
        return []

    def path(o, c):
        steps = []
        prev = o[i] if start == "open" else c[i]
        first = i if start == "open" else i + 1
        for k in range(first, j + 1):
            last = o[k] if (k == j and end == "open") else c[k]
            steps.append(last / prev - 1)
            prev = c[k]
        return np.array(steps)
    first = i if start == "open" else i + 1
    ds = list(g.d[first:j + 1])
    out = {"1x": path(g.o.to_numpy(float), g.c.to_numpy(float))}
    out["syn3"] = 3 * out["1x"] - short.SYN_DAILY_COST
    lev = PAIRS.get(base)
    if lev in book.idx:
        a, b = book.idx[lev].get(g.d[i]), book.idx[lev].get(g.d[j])
        if a is not None and b is not None and b - a == j - i and book.continuous(lev, a, b):
            lt = book.tab[lev]
            lo, lc = lt.o.to_numpy(float), lt.c.to_numpy(float)
            steps, prev = [], lo[a] if start == "open" else lc[a]
            for k in range(a if start == "open" else a + 1, b + 1):
                last = lo[k] if (k == b and end == "open") else lc[k]
                steps.append(last / prev - 1)
                prev = lc[k]
            out["real3"] = np.array(steps)
    return [dict(variant=variant, base=base, kind=kind, d=g.d[i], exit_d=g.d[j], days=max(j - i, 1),
                 gross=float(np.prod(1 + pth) - 1), path=pth, path_d=ds) for kind, pth in out.items() if len(pth)]


def report(title, note, t, names, bases=("SPY", "QQQ", "IWM"), acct=None, pooled=False):
    lines = [f"# {title}", "", note + " " + COMMON_NOTE, ""]
    if t.empty:
        return lines + ["**신호가 한 번도 없음**"]
    g_lines, _ = gate_table(t, names, bases)
    lines += g_lines
    if pooled:
        lines += ["## 종목 모두 합쳐서 (비용 전)", "", "| 변형 | 종류 | 거래 수 | 평균 | 중앙값 | 승률 | t값 | 상위 10개 뺀 평균 | 2018~ 평균 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for v in names:
            for kind in ("1x", "syn3", "real3"):
                s_ = t[(t.variant == v) & (t.kind == kind)]
                if not len(s_):
                    continue
                g = s_.gross
                yr = pd.to_datetime(s_.d).dt.year
                lines.append(f"| {names[v]} | {KIND_NAME[kind]} | {len(g)} | {pct(g.mean())} | {pct(g.median())} | {(g > 0).mean():.0%} | "
                             f"{tstat(g):.2f} | {pct(g.sort_values().iloc[:-10].mean()) if len(g) > 20 else ''} | {pct(g[yr >= 2018].mean())} |")
        lines.append("")
    if "syn3" in set(t.kind):
        lines += period_table(t, names, "syn3") + [""]
    lines += period_table(t, names, "1x") + [""]
    if acct:
        lines += ["## 계좌 기준 (신호 때 계좌 100%, 여러 신호면 나눔, 신호 없으면 현금)", ""]
        a_lines, _ = acct_table(BOOK, acct)
        lines += a_lines
    return lines


def std_acct(t, variants, bases=("SPY", "QQQ")):
    out = []
    if t.empty:
        return out
    for v in variants:
        for b in bases:
            for kind in ("real3", "syn3"):
                s_ = pick(t, v, b, kind)
                if len(s_):
                    out.append((f"{v} {label(b, kind)}", [s_], False))
    return out


def rule_b31(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        down = z["cd"] < z["prev_c"]  # 오늘 (판단가)
        cdn = z["c"] < z["prev_c"]  # 지난날들 (종가)
        lag = lambda k: np.r_[np.zeros(k, bool), cdn[:-k]]  # noqa: E731
        streak3 = down & lag(1) & lag(2)
        streak4 = streak3 & lag(3)
        upc = z["cd"] > z["prev_c"]
        ex_up = lambda i: short.first_true(upc, i, 10)  # noqa: E731
        ex_s5 = lambda i: short.first_true(z["cd"] > z["sma5"], i, 10)  # noqa: E731
        for v, en, ex in (("D1", streak3 & z["up200"], ex_up), ("D2", streak4 & z["up200"], ex_up),
                          ("D3", streak3, ex_up), ("D4", streak3 & z["up200"], ex_s5)):
            rows += trades_from(book, base, en, ex, v)
    t = pd.DataFrame(rows)
    names = {"D1": "D1 기본: 3일 연속 하락 마감 + 200일 위 → 첫 상승 마감 (최대 10일)", "D2": "D2 4일 연속",
             "D3": "D3 200일 조건 없이", "D4": "D4 5일 평균 위 마감에 매도"}
    return t, report("b31 백테스트 결과: 연속 하락 반등", "오늘 하락 = 판단가 < 전날 종가, 그 전 날들은 종가 기준.", t, names,
                     acct=std_acct(t, names))


def rule_b32(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        for v, k, mx in (("S1", 7, 15), ("S2", 5, 15), ("S3", 10, 15), ("S4", 7, 7)):
            en = (z["cd"] <= z[f"lo{k}"]) & z["up200"]
            hi = z["cd"] >= z[f"hi{k}"]
            rows += trades_from(book, base, en, lambda i, hi=hi, mx=mx: short.first_true(hi, i, mx), v)
    t = pd.DataFrame(rows)
    names = {"S1": "S1 기본: 7일 최저 종가 + 200일 위 → 7일 최고 종가 (최대 15일)", "S2": "S2 5일", "S3": "S3 10일",
             "S4": "S4 7일, 최대 7일"}
    return t, report("b32 백테스트 결과: 더블 세븐", "최저·최고 = 전날까지 종가 + 오늘 판단가.", t, names, acct=std_acct(t, names))


def rule_b33(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        r, rc = z["rsi"], z["rsi_c"]
        s2 = r + np.r_[np.nan, rc[:-1]]
        s3 = s2 + np.r_[np.nan, np.nan, rc[:-2]]
        ex65 = lambda i: short.first_true(z["rsi"] > 65, i, 10)  # noqa: E731
        ex_s5 = lambda i: short.first_true(z["cd"] > z["sma5"], i, 10)  # noqa: E731
        for v, en, ex in (("C1", (s2 < 35) & z["up200"], ex65), ("C2", (s3 < 45) & z["up200"], ex65),
                          ("C3", (s2 < 20) & z["up200"], ex65), ("C4", (s2 < 35) & z["up200"], ex_s5)):
            rows += trades_from(book, base, en, ex, v)
    t = pd.DataFrame(rows)
    names = {"C1": "C1 기본: 2일 RSI(2) 합 < 35 + 200일 위 → RSI(2) > 65 (최대 10일)", "C2": "C2 3일 합 < 45",
             "C3": "C3 2일 합 < 20", "C4": "C4 5일 평균 위 마감에 매도"}
    return t, report("b33 백테스트 결과: 누적 RSI(2)", "", t, names, acct=std_acct(t, names))


def rsi_pool(book, bases, extra=None):
    """bases 각각 b29 R1 (RSI(2)<10, 200일 위, 5일 평균 위 마감에 매도, 최대 10일). extra: (이름, 조건 함수)."""
    rows = []
    for base in bases:
        if base not in book.tab or len(book.tab[base]) < 250:
            continue
        z = ind(book, base)
        ex = lambda i, z=z: short.first_true(z["cd"] > z["sma5"], i, 10)  # noqa: E731
        rows += trades_from(book, base, (z["rsi"] < 10) & z["up200"], ex, "1")
        rows += trades_from(book, base, (z["rsi"] < 5) & z["up200"], ex, "2")
        if extra:
            for v, f in extra:
                rows += trades_from(book, base, f(base, z), ex, v)
    return rows


def pick_lowest(book, bases, variant):
    """하루에 RSI(2)가 가장 낮은 신호 종목 1개만, 계좌 전체에 한 번에 한 거래."""
    cand = {}
    for base in bases:
        if base not in book.tab or len(book.tab[base]) < 250:
            continue
        z = ind(book, base)
        for i in np.nonzero((z["rsi"] < 10) & z["up200"])[0]:
            d = z["g"].d.iloc[i]
            if d not in cand or z["rsi"][i] < cand[d][2]:
                cand[d] = (base, i, z["rsi"][i])
    rows, busy_until = [], None
    for d in sorted(cand):
        if busy_until is not None and d < busy_until:
            continue
        base, i, _ = cand[d]
        z = ind(book, base)
        if i >= len(z["c"]) - 1:
            continue
        j = short.first_true(z["cd"] > z["sma5"], i, 10)
        r = make_rows(book, base, [(i, j, {})], variant)
        rows += r
        busy_until = z["g"].d.iloc[j]
    return rows


def rule_b34(book):
    spy = ind(book, "SPY")
    spy_up = dict(zip(spy["g"].d, spy["up200"]))
    extra = [("4", lambda b, z: (z["rsi"] < 10) & z["up200"] & np.array([spy_up.get(d, False) for d in z["g"].d]))]
    rows = rsi_pool(book, SECTORS, extra)
    t = pd.DataFrame(rows)
    t["variant"] = "E" + t.variant
    t = pd.concat([t, pd.DataFrame(pick_lowest(book, SECTORS, "E3"))], ignore_index=True)
    names = {"E1": "E1 기본: 섹터 9개 각각 RSI(2) < 10 + 200일 위 → 5일 평균 위 마감", "E2": "E2 RSI(2) < 5",
             "E3": "E3 하루에 RSI 가장 낮은 1개만 (한 번에 한 거래)", "E4": "E4 E1 + SPY도 200일 위"}
    acct = [(f"{v} 1배", [t[(t.variant == v) & (t.kind == "1x")]], False) for v in names] + \
           [(f"{v} 합성 3배", [t[(t.variant == v) & (t.kind == "syn3")]], False) for v in names]
    return t, report("b34 백테스트 결과: 섹터 ETF RSI(2)", "섹터 SPDR 9개 (1998-12~). 3배 ETF는 없음 → 1배와 합성 3배만.",
                     t, names, bases=SECTORS, acct=acct, pooled=True)


def rule_b35(book):
    rows = rsi_pool(book, ASSETS)
    t = pd.DataFrame(rows)
    t["variant"] = "A" + t.variant.map({"1": "1", "2": "3"})
    t = pd.concat([t, t[(t.variant == "A1") & t.base.isin(["TLT", "GLD"])].assign(variant="A2"),
                   t[(t.variant == "A1") & t.base.isin(["SPY", "QQQ", "IWM", "EFA"])].assign(variant="A4")], ignore_index=True)
    names = {"A1": "A1 기본: 7개 자산 각각 RSI(2) < 10 + 200일 위 → 5일 평균 위 마감", "A2": "A2 TLT·GLD만",
             "A3": "A3 RSI(2) < 5", "A4": "A4 주식 4개 (SPY·QQQ·IWM·EFA)"}
    acct = [(f"{v} 1배", [t[(t.variant == v) & (t.kind == "1x")]], False) for v in names] + \
           [(f"{v} 합성 3배", [t[(t.variant == v) & (t.kind == "syn3")]], False) for v in names] + \
           [("A1 실제 3배 있는 것만 (SPY·QQQ·IWM)", [t[(t.variant == "A1") & (t.kind == "real3")]], False)]
    return t, report("b35 백테스트 결과: 여러 자산 RSI(2)", "TLT(채권, 2002~), GLD(금, 2004~), EFA(2001~), EEM(2003~). 3배는 SPY·QQQ·IWM만.",
                     t, names, bases=ASSETS, acct=acct, pooled=True)


def rule_b36(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        m, sd = z["m20"], z["sd20"]
        ex_m = lambda i: short.first_true(z["cd"] > z["m20"], i, 10)  # noqa: E731
        ex_s5 = lambda i: short.first_true(z["cd"] > z["sma5"], i, 10)  # noqa: E731
        for v, en, ex in (("B1", (z["cd"] < m - 2 * sd) & z["up200"], ex_m), ("B2", (z["cd"] < m - 2.5 * sd) & z["up200"], ex_m),
                          ("B3", z["cd"] < m - 2 * sd, ex_m), ("B4", (z["cd"] < m - 2 * sd) & z["up200"], ex_s5)):
            rows += trades_from(book, base, en, ex, v)
    t = pd.DataFrame(rows)
    names = {"B1": "B1 기본: 20일 평균 - 2표준편차 아래 + 200일 위 → 20일 평균 위 마감 (최대 10일)", "B2": "B2 2.5표준편차",
             "B3": "B3 200일 조건 없이", "B4": "B4 5일 평균 위 마감에 매도"}
    return t, report("b36 백테스트 결과: 볼린저 하단 반등", "", t, names, acct=std_acct(t, names))


def rule_b37(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        gap = z["o"] / z["prev_c"] - 1
        for v, th, filt, hold in (("G1", -0.01, True, 0), ("G2", -0.02, True, 0), ("G3", -0.01, False, 0), ("G4", -0.01, True, 1)):
            en = (gap < th) & (z["up200_prev"] if filt else True)
            for i in np.nonzero(en)[0]:
                rows += custom_rows(book, base, i, i + hold, v, start="open", end="close")
    t = pd.DataFrame(rows)
    names = {"G1": "G1 기본: 시가가 전날 종가보다 1% 넘게 낮음 + 전날 200일 위 → 시가 매수, 그날 종가 매도",
             "G2": "G2 갭 -2%", "G3": "G3 200일 조건 없이", "G4": "G4 다음 날 종가까지"}
    return t, report("b37 백테스트 결과: 큰 갭 다운 메우기", "시가 = 9:30 첫 1분봉 시가 (3배 ETF도 같은 방식).", t, names,
                     acct=std_acct(t, names))


def rule_b38(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        day = z["cd"] / z["prev_c"] - 1
        for v, en in (("O1", day < -0.01), ("O2", day < -0.02), ("O3", np.isfinite(day)), ("O4", (day < -0.01) & z["up200"])):
            for i in np.nonzero(en)[0]:
                rows += custom_rows(book, base, i, i + 1, v, start="close", end="open")
    t = pd.DataFrame(rows)
    names = {"O1": "O1 기본: 그날 -1% 넘게 하락 → 종가 매수, 다음 날 시가 매도", "O2": "O2 -2%",
             "O3": "O3 모든 날 밤 (참고)", "O4": "O4 -1% + 200일 위"}
    return t, report("b38 백테스트 결과: 하락한 날 밤 보유", "하락 = 판단가 / 전날 종가 - 1.", t, names,
                     acct=std_acct(t, ["O1", "O2", "O4"]))


def opex_days(cal):
    s = pd.Series(pd.to_datetime(cal))
    out = []
    for (y, m), _ in s.groupby([s.dt.year, s.dt.month]):
        first = pd.Timestamp(y, m, 1)
        fri = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
        out.append((y, m, fri))
    return out


def rule_b39(book):
    cal = pd.DatetimeIndex(pd.to_datetime(book.cal))
    rows = []
    for base in PAIRS:
        g = book.tab[base]
        gi = pd.DatetimeIndex(pd.to_datetime(g.d))
        for y, m, fri in opex_days(book.cal):
            e = np.searchsorted(gi, fri, side="right") - 1  # 만기일 (휴장이면 그 전 거래일)
            if e < 1 or gi[e] < fri - pd.Timedelta(days=3):
                continue
            mon = fri - pd.Timedelta(days=4)
            s_ = np.searchsorted(gi, mon, side="left") - 1  # 만기 주 시작 전 마지막 거래일
            nxt = np.searchsorted(gi, fri + pd.Timedelta(days=7), side="right") - 1
            for v, i, j in (("X1", s_, e), ("X2", e, nxt), ("X3", s_, e)):
                if v == "X3" and m not in (3, 6, 9, 12):
                    continue
                if 0 <= i < j < len(g):
                    rows += make_rows(book, base, [(i, j, {})], v)
    t = pd.DataFrame(rows)
    names = {"X1": "X1 기본: 만기 주 시작 전 종가 → 만기일(셋째 금요일) 종가", "X2": "X2 만기 다음 주 (약세 주장 확인용)",
             "X3": "X3 분기 만기(3·6·9·12월)만"}
    return t, report("b39 백테스트 결과: 옵션 만기 주간", "", t, names, acct=std_acct(t, ["X1", "X3"]))


def rule_b40(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        gi = pd.DatetimeIndex(pd.to_datetime(z["g"].d))
        wk = gi.isocalendar().year.to_numpy() * 100 + gi.isocalendar().week.to_numpy()
        last = np.r_[wk[1:] != wk[:-1], True]  # 그 주 마지막 거래일
        lastidx = np.nonzero(last)[0]
        wr = np.full(len(gi), np.nan)
        for a, b in zip(lastidx[:-1], lastidx[1:]):
            wr[b] = z["cd"][b] / z["c"][a] - 1
        n = len(gi)
        for v, th, filt, hold in (("W1", -0.03, True, 5), ("W2", -0.05, True, 5), ("W3", -0.03, False, 5), ("W4", -0.03, True, 3)):
            en = last & (wr < th) & (z["up200"] if filt else True)
            rows += trades_from(book, base, en, lambda i, hold=hold: min(i + hold, n - 1), v)
    t = pd.DataFrame(rows)
    names = {"W1": "W1 기본: 그 주 -3% 넘게 하락 + 200일 위 → 그 주 마지막 날 종가 매수, 5거래일 뒤 매도", "W2": "W2 -5%",
             "W3": "W3 200일 조건 없이", "W4": "W4 3거래일 뒤 매도"}
    return t, report("b40 백테스트 결과: 나쁜 한 주 뒤 반등", "주간 수익 = 그 주 마지막 날 판단가 / 전주 마지막 날 종가 - 1.", t, names,
                     acct=std_acct(t, names))


def rule_b31x(book):
    """과매도 반등 가족 합치기: 신호 겹침과 한 계좌 성적."""
    t29, _ = rule_b29(book)
    t31, _ = rule_b31(book)
    t32, _ = rule_b32(book)
    t35, _ = rule_b35(book)
    t36, _ = rule_b36(book)
    t30, _ = rule_b30(book)
    t28, _ = rule_b28(book)
    start = dt.date(2011, 1, 1)
    clip = lambda x: x[x.d >= start]  # noqa: E731
    sig = {"R1 (b29) SPXL": pick(t29, "R1", "SPY", "real3"), "R2 (b29) SPXL": pick(t29, "R2", "SPY", "real3"),
           "D3 (b31) TQQQ": pick(t31, "D3", "QQQ", "real3"), "D4 (b31) TQQQ": pick(t31, "D4", "QQQ", "real3"),
           "S2 (b32) SPXL": pick(t32, "S2", "SPY", "real3"), "S2 (b32) TQQQ": pick(t32, "S2", "QQQ", "real3"),
           "A3 (b35) 실제 3배": t35[(t35.variant == "A3") & (t35.kind == "real3")].sort_values("d"),
           "B4 (b36) TQQQ": pick(t36, "B4", "QQQ", "real3"), "I4 (b30) SPXL": pick(t30, "I4", "SPY", "real3"),
           "F2 (b28) TQQQ": pick(t28, "F2", "QQQ", "real3")}
    sig = {k: clip(v) for k, v in sig.items()}
    held = {k: {d for x in v.itertuples() for d in x.path_d} for k, v in sig.items()}
    ks = list(sig)
    lines = ["# 과매도 반등 가족 합치기 (b29·b31·b32·b35·b36 + b30·b28)", "",
             "실제 3배 ETF, 2011~2026-03. 같은 날 여러 신호면 돈을 똑같이 나누고, 신호 없는 날은 현금. 비용은 종목 비중이 바뀐 만큼. "
             + COMMON_NOTE, "", "## 신호가 얼마나 겹치나 (두 신호 중 적은 쪽 보유일 가운데 같이 들고 있던 날 비율)", "",
             "| | 보유일 | " + " | ".join(k.split(" ")[0] + " " + k.split(" ")[-1] for k in ks) + " |", "|---|---|" + "---|" * len(ks)]
    for a in ks:
        cells = [f"{len(held[a] & held[b]) / max(1, min(len(held[a]), len(held[b]))):.0%}" if a != b else "" for b in ks]
        lines.append(f"| {a} | {len(held[a])} | " + " | ".join(cells) + " |")
    S = sig
    combos = [("P1 S2 SPXL만", [S["S2 (b32) SPXL"]], False),
              ("P2 S2 SPXL + D3 TQQQ", [S["S2 (b32) SPXL"], S["D3 (b31) TQQQ"]], False),
              ("P3 S2 SPXL + S2 TQQQ", [S["S2 (b32) SPXL"], S["S2 (b32) TQQQ"]], False),
              ("P4 S2 SPXL + D3 TQQQ + R1 SPXL", [S["S2 (b32) SPXL"], S["D3 (b31) TQQQ"], S["R1 (b29) SPXL"]], False),
              ("P5 S2 SPXL + D3 TQQQ + B4 TQQQ", [S["S2 (b32) SPXL"], S["D3 (b31) TQQQ"], S["B4 (b36) TQQQ"]], False),
              ("P6 A3 실제 3배 (SPY·QQQ·IWM RSI<5)", [S["A3 (b35) 실제 3배"]], False),
              ("P7 S2 SPXL + D3 TQQQ + I4 SPXL + F2 TQQQ", [S["S2 (b32) SPXL"], S["D3 (b31) TQQQ"], S["I4 (b30) SPXL"],
                                                          S["F2 (b28) TQQQ"]], False),
              ("P8 가족 전부 (R1·D3·D4·S2×2·A3·B4)", [S[k] for k in ks if k[:2] in ("R1", "D3", "D4", "S2", "A3", "B4")], False),
              ("참고 K3 (b29x): R1 + I4 + F2", [S["R1 (b29) SPXL"], S["I4 (b30) SPXL"], S["F2 (b28) TQQQ"]], False)]
    lines += ["", "## 한 계좌 성적 (2011~2026-03)", ""]
    a_lines, series = acct_table(book, combos, start=start)
    lines += a_lines
    lines += ["", "## 2020년 빼고 (그해 수익을 0으로)", "", "| 조합 | 비용 | 연수익 | 최대낙폭 |", "|---|---|---|---|"]
    for (n, c), r in series.items():
        r2 = r.copy()
        r2[r2.index.year == 2020] = 0.0
        m = portfolio.stats(r2)
        lines.append(f"| {n} | {c} | {pct(m['cagr'], 1)} | {pct(m['mdd'], 1)} |")
    spy = book.tab["SPY"]
    sp = pd.Series(spy.c.to_numpy(), index=pd.to_datetime(spy.d)).pct_change().fillna(0.0)
    sp = sp[sp.index > pd.Timestamp(start)]
    sp2 = sp.copy()
    sp2[sp2.index.year == 2020] = 0.0
    m = portfolio.stats(sp2)
    lines.append(f"| SPY 그냥 보유 | | {pct(m['cagr'], 1)} | {pct(m['mdd'], 1)} |")
    lines += ["", "## 해마다", "", *yearly_series(series, [("P2 S2 SPXL + D3 TQQQ", COST_LEVELS[0][0]), ("P2 S2 SPXL + D3 TQQQ", COST_LEVELS[1][0]),
                                                         ("P8 가족 전부 (R1·D3·D4·S2×2·A3·B4)", COST_LEVELS[0][0]),
                                                         ("P8 가족 전부 (R1·D3·D4·S2×2·A3·B4)", COST_LEVELS[1][0])], book, start)]
    # 합성 3배로 긴 기간 (1995~, b35 A3는 7개 자산 모두)
    L = {"S2": pick(t32, "S2", "SPY", "syn3"), "D3": pick(t31, "D3", "QQQ", "syn3"), "A3": t35[(t35.variant == "A3") & (t35.kind == "syn3")],
         "R1": pick(t29, "R1", "SPY", "syn3")}
    lines += ["", "## 긴 기간: 합성 3배 (1995~)", ""]
    st2 = dt.date(1995, 1, 1)
    l_lines, _ = acct_table(book, [("L1 S2 SPY×3", [L["S2"][L["S2"].d >= st2]], False),
                                    ("L2 S2 SPY×3 + D3 QQQ×3", [L["S2"][L["S2"].d >= st2], L["D3"][L["D3"].d >= st2]], False),
                                    ("L3 A3 7개 자산×3", [L["A3"][L["A3"].d >= st2]], False),
                                    ("L4 S2 + D3 + A3", [x[x.d >= st2] for x in (L["S2"], L["D3"], L["A3"])], False)], start=st2)
    lines += l_lines
    return pd.concat([t29, t31, t32], ignore_index=True), lines


# ---------------------------------------------------------------- b41~b50 추세·다른 신호
def prior_max(c, n):
    """i일 기준 지난 n일(i-n ~ i-1) 최고 종가."""
    return pd.Series(c).shift(1).rolling(n).max().to_numpy()


def prior_min(c, n):
    return pd.Series(c).shift(1).rolling(n).min().to_numpy()


def atr(g, n=20):
    h, l, c = (g[k].to_numpy(float) for k in ("h", "l", "c"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    return pd.Series(tr).shift(1).rolling(n).mean().to_numpy()  # 어제까지


def donchian_rows(book, base, n_in, n_out, variant, filt=None, atr_stop=None, max_days=60):
    z = ind(book, base)
    hi, lo = prior_max(z["c"], n_in), prior_min(z["c"], n_out)
    en = z["cd"] > hi
    if filt is not None:
        en = en & filt
    a = atr(z["g"]) if atr_stop else None
    lo_d = z["cd"] < prior_min(z["c"], n_out)

    def ex(i):
        if atr_stop:
            stop = z["c"][i] - atr_stop * a[i]
            return short.first_true(z["cd"] < stop, i, max_days)
        return short.first_true(lo_d, i, max_days)
    return trades_from(book, base, en, ex, variant)


def rule_b41(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        rows += donchian_rows(book, base, 20, 10, "T1")
        rows += donchian_rows(book, base, 55, 20, "T2")
        rows += donchian_rows(book, base, 20, 10, "T3", filt=z["up200"])
        rows += donchian_rows(book, base, 20, 10, "T4", atr_stop=2.0)
    t = pd.DataFrame(rows)
    names = {"T1": "T1 기본: 20일 최고 종가 돌파 → 10일 최저 종가 아래 마감 (최대 60일)", "T2": "T2 55일/20일",
             "T3": "T3 + 200일 위", "T4": "T4 20일 돌파 → 매수가 - 2×ATR(20) 아래 마감 (최대 60일)"}
    return t, report("b41 백테스트 결과: 돈치안 돌파", "최고·최저 = 오늘 빼고 지난 n일 종가, 오늘은 판단가로 비교.", t, names,
                     acct=std_acct(t, names))


def rule_b42(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        hi = prior_max(z["c"], 250)
        new = z["cd"] > hi
        first = new & ~np.r_[False, (z["c"] > hi)[:-1]]  # 어제는 신고가가 아니었음
        n = len(z["c"])
        lo20 = z["cd"] < prior_min(z["c"], 20)
        for v, ex in (("H1", lambda i: min(i + 20, n - 1)), ("H2", lambda i: min(i + 10, n - 1)),
                      ("H3", lambda i: min(i + 40, n - 1)), ("H4", lambda i: short.first_true(lo20, i, 60))):
            rows += trades_from(book, base, first, ex, v)
    t = pd.DataFrame(rows)
    names = {"H1": "H1 기본: 52주(250일) 최고 종가 첫 돌파 → 20거래일 뒤", "H2": "H2 10일 뒤", "H3": "H3 40일 뒤",
             "H4": "H4 20일 최저 종가 아래 마감에 매도 (최대 60일)"}
    return t, report("b42 백테스트 결과: 52주 신고가", "", t, names, acct=std_acct(t, names))


def breakout_rows(book, base, i, level, stop_lvl, hold, variant):
    """i+1일 장중 1분봉 종가가 level을 처음 넘는 순간 매수(15:55 전) → hold일 뒤 종가, 그 전 stop_lvl 아래 마감이면 그날 종가."""
    g = book.tab[base]
    k = i + 1
    if k >= len(g) or not book.continuous(base, i, min(k + hold, len(g) - 1)):
        return []
    d = g.d.iloc[k]
    x = book.by_day.get((base, d))
    if x is None:
        return []
    f = g.f.iloc[k]
    mi, c = x[0], x[1] * f
    lim = HALF_DECIDE_MI if d in book.half else DECIDE_MI
    hit = np.nonzero((c > level) & (mi < lim))[0]
    if not len(hit):
        return []
    m, px = int(mi[hit[0]]), float(c[hit[0]])
    cc = g.c.to_numpy(float)
    j = k
    for jj in range(k, min(k + hold, len(g) - 1) + 1):
        j = jj
        if cc[jj] < stop_lvl and jj > k:
            break
    out = {"1x": np.r_[cc[k] / px - 1, cc[k + 1:j + 1] / cc[k:j] - 1]}
    out["syn3"] = 3 * out["1x"] - short.SYN_DAILY_COST
    lev = PAIRS.get(base)
    if lev in book.idx:
        a, b = book.idx[lev].get(d), book.idx[lev].get(g.d.iloc[j])
        lp = book.price_before(lev, d, m + 1)
        if a is not None and b is not None and b - a == j - k and lp:
            lc = book.tab[lev].c.to_numpy(float)
            out["real3"] = np.r_[lc[a] / lp - 1, lc[a + 1:b + 1] / lc[a:b] - 1]
    ds = list(g.d[k:j + 1])
    return [dict(variant=variant, base=base, kind=kd, d=d, exit_d=g.d.iloc[j], days=max(j - k, 1), gross=float(np.prod(1 + pth) - 1),
                 path=pth, path_d=ds) for kd, pth in out.items()]


DECIDE_MI, HALF_DECIDE_MI = 385, 205


def rule_b43(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        g = z["g"]
        h, l = g.h.to_numpy(float), g.l.to_numpy(float)
        rng = h - l
        for v, n, hold, filt in (("N1", 7, 5, False), ("N2", 4, 5, False), ("N3", 7, 10, False), ("N4", 7, 5, True)):
            narrow = rng <= pd.Series(rng).rolling(n).min().to_numpy()
            busy = -1
            for i in np.nonzero(narrow)[0]:
                if i <= busy or (filt and not (z["c"][i] > z["sma200"][i])):
                    continue
                r = breakout_rows(book, base, i, h[i], l[i], hold, v)
                if r:
                    rows += r
                    busy = book.idx[base][r[0]["exit_d"]] - 1
    t = pd.DataFrame(rows)
    names = {"N1": "N1 기본: 어제 NR7 → 오늘 어제 고가 돌파 순간 매수 → 5일 뒤 (어제 저가 아래 마감이면 그날)", "N2": "N2 NR4",
             "N3": "N3 10일 보유", "N4": "N4 200일 위만"}
    return t, report("b43 백테스트 결과: 좁은 날 다음 날 돌파", "돌파는 1분봉 종가 기준, 3배 ETF는 같은 분의 가격으로 매수.", t, names,
                     acct=std_acct(t, names))


def rel_rows(book, other, th, hold, variant, base="SPY"):
    z, o = ind(book, base), ind(book, other)
    od = dict(zip(o["g"].d, range(len(o["g"]))))
    oc = o["c"]
    n = len(z["c"])
    sig = np.zeros(n, bool)
    for i in range(5, n):
        d = z["g"].d.iloc[i]
        k = od.get(d)
        if k is None or k < 5:
            continue
        ro = o["cd"][k] / oc[k - 5] - 1
        rs = z["cd"][i] / z["c"][i - 5] - 1
        sig[i] = ro - rs > th
    return trades_from(book, base, sig, lambda i: min(i + hold, n - 1), variant)


def rule_b44(book):
    rows = []
    for base in PAIRS:
        rows += rel_rows(book, "TLT", 0.05, 5, "B1", base)
        rows += rel_rows(book, "TLT", 0.03, 5, "B2", base)
        rows += rel_rows(book, "TLT", 0.05, 10, "B3", base)
        rows += rel_rows(book, "GLD", 0.05, 5, "B4", base)
    t = pd.DataFrame(rows)
    names = {"B1": "B1 기본: 5일 수익 TLT − 주식 ETF > 5%p → 5일 보유", "B2": "B2 > 3%p", "B3": "B3 10일 보유", "B4": "B4 GLD − 주식 > 5%p"}
    return t, report("b44 백테스트 결과: 주식-채권 벌어짐", "TLT 2002~, GLD 2004~. 5일 수익 = 오늘 판단가 / 5거래일 전 종가 - 1.", t, names,
                     acct=std_acct(t, names))


def rule_b45(book):
    rows = []
    for base in ASSETS:
        if base not in book.tab or len(book.tab[base]) < 300:
            continue
        rows += donchian_rows(book, base, 20, 10, "M1")
        rows += donchian_rows(book, base, 55, 20, "M2")
    t = pd.DataFrame(rows)
    t = pd.concat([t, t[(t.variant == "M1") & t.base.isin(["SPY", "QQQ", "IWM", "EFA"])].assign(variant="M3"),
                   t[(t.variant == "M1") & t.base.isin(["TLT", "GLD"])].assign(variant="M4")], ignore_index=True)
    names = {"M1": "M1 기본: 7개 자산 각각 20일 돌파 → 10일 최저 아래", "M2": "M2 55/20", "M3": "M3 주식 4개", "M4": "M4 TLT·GLD만"}
    acct = [(f"{v} 1배", [t[(t.variant == v) & (t.kind == "1x")]], False) for v in names] + \
           [(f"{v} 합성 3배", [t[(t.variant == v) & (t.kind == "syn3")]], False) for v in names]
    return t, report("b45 백테스트 결과: 여러 자산 돈치안", "", t, names, bases=ASSETS, acct=acct, pooled=True)


def rule_b46(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        g = z["g"]
        hd, ld = g.hd.to_numpy(float), g.ld.to_numpy(float)
        ibs = np.where(hd > ld, (z["cd"] - ld) / np.where(hd > ld, hd - ld, 1), np.nan)
        gap = z["o"] / z["prev_c"] - 1
        n = len(z["c"])
        for v, th, hold, filt in (("U1", 0.005, 3, False), ("U2", 0.01, 3, False), ("U3", 0.005, 1, False), ("U4", 0.005, 3, True)):
            en = (gap > th) & (ibs > 0.8) & (z["up200"] if filt else True)
            rows += trades_from(book, base, en, lambda i, hold=hold: min(i + hold, n - 1), v)
    t = pd.DataFrame(rows)
    names = {"U1": "U1 기본: 갭 +0.5% 넘게 + IBS > 0.8 → 3일 뒤", "U2": "U2 갭 +1%", "U3": "U3 1일 뒤", "U4": "U4 + 200일 위"}
    return t, report("b46 백테스트 결과: 갭 상승 강한 마감", "", t, names, acct=std_acct(t, names))


def rule_b47(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        gi = pd.DatetimeIndex(pd.to_datetime(z["g"].d))
        wk = gi.isocalendar().year.to_numpy() * 100 + gi.isocalendar().week.to_numpy()
        first = np.r_[True, wk[1:] != wk[:-1]]
        day = z["cd"] / z["prev_c"] - 1
        n = len(gi)
        for v, th, hold, filt in (("Y1", -0.005, 1, False), ("Y2", -0.01, 1, False), ("Y3", -0.005, 2, False), ("Y4", -0.005, 1, True)):
            en = first & (day < th) & (z["up200"] if filt else True)
            rows += trades_from(book, base, en, lambda i, hold=hold: min(i + hold, n - 1), v)
    t = pd.DataFrame(rows)
    names = {"Y1": "Y1 기본: 주 첫 거래일 -0.5% 넘게 하락 → 다음 날 종가", "Y2": "Y2 -1%", "Y3": "Y3 이틀 뒤", "Y4": "Y4 + 200일 위"}
    return t, report("b47 백테스트 결과: 반등 화요일", "", t, names, acct=std_acct(t, names))


def rule_b48(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        c, cd = z["c"], z["cd"]
        for v, k, filt in (("A1", 20, True), ("A2", 10, True), ("A3", 50, True), ("A4", 20, False)):
            m = pd.Series(c).rolling(k).mean().to_numpy()
            md = (pd.Series(c).shift(1).rolling(k - 1).sum().to_numpy() + cd) / k  # 오늘 판단가 포함
            cross = (cd > md) & np.r_[False, (c <= m)[:-1]]
            below = cd < md
            en = cross & (z["up200"] if filt else True)
            rows += trades_from(book, base, en, lambda i, below=below: short.first_true(below, i, 60), v)
    t = pd.DataFrame(rows)
    names = {"A1": "A1 기본: 20일 평균 위로 넘음 + 200일 위 → 20일 평균 아래 마감 (최대 60일)", "A2": "A2 10일선", "A3": "A3 50일선",
             "A4": "A4 200일 조건 없이"}
    return t, report("b48 백테스트 결과: 20일선 타기", "", t, names, acct=std_acct(t, names))


def rule_b49(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        c, cd = z["c"], z["cd"]
        r = np.r_[np.nan, cd[1:] / c[:-1] - 1]  # 오늘은 판단가 수익
        rc = pd.Series(np.r_[np.nan, c[1:] / c[:-1] - 1])
        vol = np.sqrt(((rc.shift(1) ** 2).rolling(19).sum().to_numpy() + r ** 2) / 20 * 252)
        for v, ma, vt, mx in (("V1", 50, 0.15, 60), ("V2", 50, 0.20, 60), ("V3", 100, 0.15, 60), ("V4", 50, 0.15, 20)):
            m = (pd.Series(c).shift(1).rolling(ma - 1).sum().to_numpy() + cd) / ma
            ok = (cd > m) & (vol < vt)
            en = ok & ~np.r_[False, ok[:-1]]
            rows += trades_from(book, base, en, lambda i, ok=ok, mx=mx: short.first_true(~ok, i, mx), v)
    t = pd.DataFrame(rows)
    names = {"V1": "V1 기본: 50일 평균 위 + 20일 변동성 < 연 15% 된 날 → 둘 중 하나 깨지면 (최대 60일)", "V2": "V2 변동성 < 20%",
             "V3": "V3 100일 평균", "V4": "V4 최대 20일"}
    return t, report("b49 백테스트 결과: 잔잔한 상승장만 3배", "변동성 = 지난 19일 종가 수익 + 오늘 판단가 수익의 제곱 평균 × 252의 제곱근.",
                     t, names, acct=std_acct(t, names))


def rule_b50(book):
    f = fomc_table()
    rows = []
    for base in PAIRS:
        g = book.tab[base]
        cd = g.cd.to_numpy(float)
        n = len(g)
        for r in f.itertuples():
            k = book.idx[base].get(r.d)
            if k is None:
                continue
            px = book.price_before(base, r.d, r.ann_mi)
            if px is None:
                continue
            react = cd[k] / px - 1
            for v, ok, hold in (("P1", react > 0.005, 5), ("P2", react < -0.005, 5), ("P3", True, 5), ("P4", react > 0.005, 2)):
                if ok and k + hold < n:
                    rows += make_rows(book, base, [(k, k + hold, {"react": react})], v)
    t = pd.DataFrame(rows)
    names = {"P1": "P1 기본: 발표 직전 → 15:55 +0.5% 넘게 오름 → 그날 종가 매수, 5일 뒤", "P2": "P2 -0.5% 넘게 내림",
             "P3": "P3 반응 상관없이", "P4": "P4 +0.5%, 2일 뒤"}
    return t, report("b50 백테스트 결과: FOMC 발표 반응 따라가기", f"FOMC 정례 회의 {len(f)}번 (2011~2026-03).", t, names,
                     acct=std_acct(t, names))


def no2020_table(book, series, start):
    lines = ["| 조합 | 비용 | 연수익 | 최대낙폭 |", "|---|---|---|---|"]
    for (n, c), r in series.items():
        r2 = r.copy()
        r2[r2.index.year == 2020] = 0.0
        m = portfolio.stats(r2)
        lines.append(f"| {n} | {c} | {pct(m['cagr'], 1)} | {pct(m['mdd'], 1)} |")
    spy = book.tab["SPY"]
    sp = pd.Series(spy.c.to_numpy(), index=pd.to_datetime(spy.d)).pct_change().fillna(0.0)
    sp = sp[sp.index > pd.Timestamp(start)]
    sp[sp.index.year == 2020] = 0.0
    m = portfolio.stats(sp)
    lines.append(f"| SPY 그냥 보유 | | {pct(m['cagr'], 1)} | {pct(m['mdd'], 1)} |")
    return lines


def rule_b41x(book):
    """b41~b50 살아남은 신호(b44 B2, b47 Y2·Y3, b41 T4)와 과매도 반등 P5의 겹침, 한 계좌 성적."""
    t44, _ = rule_b44(book)
    t47, _ = rule_b47(book)
    t41, _ = rule_b41(book)
    t31, _ = rule_b31(book)
    t32, _ = rule_b32(book)
    t36, _ = rule_b36(book)
    start = dt.date(2011, 1, 1)
    clip = lambda x: x[x.d >= start]  # noqa: E731
    sig = {"B2 (b44) TQQQ": pick(t44, "B2", "QQQ", "real3"), "B2 (b44) SPXL": pick(t44, "B2", "SPY", "real3"),
           "Y2 (b47) SPXL": pick(t47, "Y2", "SPY", "real3"), "Y3 (b47) TQQQ": pick(t47, "Y3", "QQQ", "real3"),
           "T4 (b41) SPXL": pick(t41, "T4", "SPY", "real3"),
           "S2 (b32) SPXL": pick(t32, "S2", "SPY", "real3"), "D3 (b31) TQQQ": pick(t31, "D3", "QQQ", "real3"),
           "B4 (b36) TQQQ": pick(t36, "B4", "QQQ", "real3")}
    sig = {k: clip(v) for k, v in sig.items()}
    held = {k: {d for x in v.itertuples() for d in x.path_d} for k, v in sig.items()}
    ks = list(sig)
    lines = ["# b41~b50 살아남은 신호 + 과매도 반등 합치기", "",
             "실제 3배 ETF, 2011~2026-03. 같은 날 여러 신호면 돈을 똑같이 나누고, 신호 없는 날은 현금. 비용은 종목 비중이 바뀐 만큼. "
             + COMMON_NOTE, "", "## 신호가 얼마나 겹치나 (두 신호 중 적은 쪽 보유일 가운데 같이 들고 있던 날 비율)", "",
             "| | 거래 수 | 보유일 | " + " | ".join(k.split(" ")[0] + " " + k.split(" ")[-1] for k in ks) + " |",
             "|---|---|---|" + "---|" * len(ks)]
    for a in ks:
        cells = [f"{len(held[a] & held[b]) / max(1, min(len(held[a]), len(held[b]))):.0%}" if a != b else "" for b in ks]
        lines.append(f"| {a} | {len(sig[a])} | {len(held[a])} | " + " | ".join(cells) + " |")
    p5 = set().union(*(held[k] for k in ("S2 (b32) SPXL", "D3 (b31) TQQQ", "B4 (b36) TQQQ")))
    lines += ["", "| 신호 | P5(S2+D3+B4) 보유일과 겹치는 비율 | P5 없는 날에만 있는 보유일 |", "|---|---|---|"]
    for k in ks[:5]:
        lines.append(f"| {k} | {len(held[k] & p5) / max(1, len(held[k])):.0%} | {len(held[k] - p5)} |")
    S = sig
    P5 = [S["S2 (b32) SPXL"], S["D3 (b31) TQQQ"], S["B4 (b36) TQQQ"]]
    combos = [("Q1 B2 TQQQ만", [S["B2 (b44) TQQQ"]], False),
              ("Q2 P5 (S2 SPXL + D3 TQQQ + B4 TQQQ)", P5, False),
              ("Q3 P5 + B2 TQQQ", P5 + [S["B2 (b44) TQQQ"]], False),
              ("Q4 P5 + Y2 SPXL", P5 + [S["Y2 (b47) SPXL"]], False),
              ("Q5 P5 + B2 TQQQ + Y2 SPXL", P5 + [S["B2 (b44) TQQQ"], S["Y2 (b47) SPXL"]], False),
              ("Q6 B2 TQQQ + Y2 SPXL", [S["B2 (b44) TQQQ"], S["Y2 (b47) SPXL"]], False),
              ("Q7 Y2 SPXL + Y3 TQQQ", [S["Y2 (b47) SPXL"], S["Y3 (b47) TQQQ"]], False),
              ("Q8 S2 SPXL + B2 TQQQ", [S["S2 (b32) SPXL"], S["B2 (b44) TQQQ"]], False),
              ("Q9 Q5 + T4 SPXL", P5 + [S["B2 (b44) TQQQ"], S["Y2 (b47) SPXL"], S["T4 (b41) SPXL"]], False),
              ("Q10 B2 SPXL + Y2 SPXL", [S["B2 (b44) SPXL"], S["Y2 (b47) SPXL"]], False)]
    lines += ["", "## 한 계좌 성적 (2011~2026-03)", ""]
    a_lines, series = acct_table(book, combos, start=start)
    lines += a_lines
    lines += ["", "## 2020년 빼고 (그해 수익을 0으로)", "", *no2020_table(book, series, start)]
    lines += ["", "## 해마다", "", *yearly_series(series, [("Q1 B2 TQQQ만", COST_LEVELS[0][0]), ("Q2 P5 (S2 SPXL + D3 TQQQ + B4 TQQQ)", COST_LEVELS[1][0]),
                                                         ("Q5 P5 + B2 TQQQ + Y2 SPXL", COST_LEVELS[0][0]),
                                                         ("Q5 P5 + B2 TQQQ + Y2 SPXL", COST_LEVELS[1][0]),
                                                         ("Q6 B2 TQQQ + Y2 SPXL", COST_LEVELS[0][0])], book, start)]
    # 합성 3배로 긴 기간 (TLT가 2002-07부터라 2003~)
    st2 = dt.date(2003, 1, 1)
    L = {k: pick(t, v, b, "syn3") for k, (t, v, b) in {"B2Q": (t44, "B2", "QQQ"), "B2S": (t44, "B2", "SPY"), "Y2": (t47, "Y2", "SPY"),
                                                       "S2": (t32, "S2", "SPY"), "D3": (t31, "D3", "QQQ")}.items()}
    L = {k: v[v.d >= st2] for k, v in L.items()}
    lines += ["", "## 긴 기간: 합성 3배 (2003~)", ""]
    l_lines, _ = acct_table(book, [("L1 B2 QQQ×3", [L["B2Q"]], False), ("L2 B2 SPY×3", [L["B2S"]], False),
                                    ("L3 Y2 SPY×3", [L["Y2"]], False), ("L4 S2 SPY×3 + D3 QQQ×3", [L["S2"], L["D3"]], False),
                                    ("L5 L4 + B2 QQQ×3 + Y2 SPY×3", [L["S2"], L["D3"], L["B2Q"], L["Y2"]], False)], start=st2)
    lines += l_lines
    return pd.concat([t44, t47], ignore_index=True), lines


def b44_parts(book, base, other="TLT"):
    """i일 기준 5일 수익 (오늘 판단가 / 5거래일 전 종가 - 1): 다른 자산 ro, 주식 ETF rs. 자료 없으면 nan."""
    z, o = ind(book, base), ind(book, other)
    od = dict(zip(o["g"].d, range(len(o["g"]))))
    n = len(z["c"])
    ro, rs = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(5, n):
        k = od.get(z["g"].d.iloc[i])
        if k is None or k < 5:
            continue
        ro[i] = o["cd"][k] / o["c"][k - 5] - 1
        rs[i] = z["cd"][i] / z["c"][i - 5] - 1
    return ro, rs


def with_syn2(t):
    """1배 거래에서 합성 2배(날마다 2배 - 0.013%) 줄을 만들어 붙임."""
    x = t[t.kind == "1x"].copy()
    x["kind"] = "syn2"
    x["path"] = [2 * np.asarray(pth, float) - 0.00013 for pth in x.path]
    x["gross"] = [float(np.prod(1 + pth) - 1) for pth in x.path]
    return pd.concat([t, x], ignore_index=True)


def acct_one(book, parts, start, cost, scale=1.0):
    trades = [(inst(x.base, x.kind), list(x.path_d), np.asarray(x.path, float)) for s in parts for x in s.itertuples()]
    days = [d for d in book.cal if d > start]
    return short.combine_daily(trades, days, cost) * scale


def bh_series(book, tk, start):
    g = book.tab[tk]
    r = pd.Series(g.c.to_numpy(float), index=pd.to_datetime(g.d)).pct_change().fillna(0.0)
    return r[r.index > pd.Timestamp(start)]


def rule_b44x(book):
    """b44 B2 다듬기: 기준값·보유일 민감도, 채권 상승 조건, 손절, 배율·비중."""
    rows = []
    std, low = COST_LEVELS[0][1], COST_LEVELS[1][1]
    for base in ("SPY", "QQQ"):
        z = ind(book, base)
        ro, rs = b44_parts(book, base)
        n = len(z["c"])
        gap = ro - rs
        with np.errstate(invalid="ignore"):
            for th in (0.02, 0.03, 0.04, 0.05):
                for hold in (3, 5, 7, 10):
                    rows += trades_from(book, base, gap > th, lambda i, h=hold: min(i + h, n - 1), f"G{int(th * 100)}-{hold}")
            h5 = lambda i: min(i + 5, n - 1)  # noqa: E731
            b2 = gap > 0.03
            for v, en in (("F1", b2 & (ro > 0)), ("F2", b2 & (ro > 0.01)), ("F3", b2 & z["up200"]), ("F4", b2 & ~z["up200"]),
                          ("C1", rs < -0.03), ("C2", (rs < -0.03) & (ro <= 0))):
                rows += trades_from(book, base, en, h5, v)
            for v, stop in (("S3", 0.03), ("S5", 0.05)):
                c, cd = z["c"], z["cd"]
                rows += trades_from(book, base, b2, lambda i, s_=stop: short.first_true(cd < c[i] * (1 - s_), i, 5), v)
    t = with_syn2(pd.DataFrame(rows))
    st, st2 = dt.date(2011, 1, 1), dt.date(2003, 1, 1)
    lines = ["# b44 다듬기: 주식-채권 벌어짐 (B2)", "",
             "B2 = 5일 수익 TLT − 주식 ETF > 3%p (오늘 판단가 기준) → 그날 종가 매수, 5거래일 뒤 종가 매도. 신호 없는 날 현금. "
             "실제 3배는 2011~2026-03, 합성은 2003~ (TLT 자료가 2002-07부터). QQQ는 2004~2011 자료가 없어서 합성 QQQ 긴 기간은 그 구간이 빠져요 "
             "→ 2008년은 SPY 합성으로만 볼 수 있어요. 합성 2배 = 1배 날마다 × 2 − 0.013%. " + COMMON_NOTE, ""]

    def cell(s_, start, cost=std):
        if not len(s_):
            return "-"
        m = portfolio.stats(acct_one(book, [s_[s_.d >= start]], start, cost))
        return f"{pct(s_.gross.mean())} (t {tstat(s_.gross):.1f}, {len(s_)}번) · {pct(m['cagr'], 1)} / {pct(m['mdd'], 0)}"
    for title, base, kind, start in (("TQQQ 실제 3배, 2011~", "QQQ", "real3", st), ("SPXL 실제 3배, 2011~", "SPY", "real3", st),
                                     ("SPY×3 합성, 2003~ (2008 포함)", "SPY", "syn3", st2)):
        lines += [f"## 민감도: {title}", "", "칸 = 거래당 비용 전 평균 (t값, 거래 수) · 표준 수수료 계좌 연수익 / 최대낙폭", "",
                  "| TLT − 주식 | 3일 보유 | 5일 | 7일 | 10일 |", "|---|---|---|---|---|"]
        for th in (2, 3, 4, 5):
            lines.append(f"| > {th}%p | " + " | ".join(cell(pick(t, f"G{th}-{h}", base, kind), start) for h in (3, 5, 7, 10)) + " |")
        lines.append("")
    names = {"G3-5": "B2 그대로", "F1": "F1 + TLT 5일 수익 > 0 (채권이 실제로 오름)", "F2": "F2 + TLT 5일 수익 > 1%",
             "F3": "F3 + 주식 ETF 200일 평균 위", "F4": "F4 + 200일 평균 아래 (참고)", "S3": "S3 + 손절: 기초 ETF가 매수가보다 3% 아래 마감",
             "S5": "S5 + 손절 5%", "C1": "C1 대조: TLT 없이 주식 5일 -3% 넘게 하락만", "C2": "C2 대조: 주식 -3% + 채권도 안 오름"}
    insts = (("TQQQ", "QQQ", "real3", st), ("SPXL", "SPY", "real3", st), ("SPY×3 합성 2003~", "SPY", "syn3", st2))
    lines += ["## 조건·손절·대조 (기준 3%p, 5일 보유)", "",
              "| 변형 | 종목 | 거래 수 | 비용 전 평균 | t값 | 승률 | 표준: 연수익 / 최대낙폭 | 0.07%+0.1%: 연수익 / 최대낙폭 | 2008 | 2022 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for v, nm in names.items():
        for lab, base, kind, start in insts:
            s_ = pick(t, v, base, kind)
            s_ = s_[s_.d >= start]
            if not len(s_):
                continue
            r1, r2 = acct_one(book, [s_], start, std), acct_one(book, [s_], start, low)
            m1, m2 = portfolio.stats(r1), portfolio.stats(r2)
            y = portfolio.yearly(r1)
            lines.append(f"| {nm} | {lab} | {len(s_)} | {pct(s_.gross.mean())} | {tstat(s_.gross):.2f} | {(s_.gross > 0).mean():.0%} | "
                         f"{pct(m1['cagr'], 1)} / {pct(m1['mdd'], 1)} | {pct(m2['cagr'], 1)} / {pct(m2['mdd'], 1)} | "
                         f"{pct(y.get(2008, np.nan), 1)} | {pct(y.get(2022, np.nan), 1)} |")
    lines += ["", "## 배율과 비중 (B2 그대로)", "",
              "비중 50% = 신호 때 계좌 절반만 사고 나머지는 현금 (날마다 수익 × 0.5로 계산).", "",
              "| 무엇으로 | 기간 | 표준: 연수익 / 최대낙폭 / 샤프 | 0.07%+0.1%: 연수익 / 최대낙폭 / 샤프 | 2008 | 2022 |", "|---|---|---|---|---|---|"]
    lev_rows = [("QQQ 1배", "QQQ", "1x", 1.0, st), ("QQQ 합성 2배", "QQQ", "syn2", 1.0, st), ("TQQQ", "QQQ", "real3", 1.0, st),
                ("TQQQ 비중 50%", "QQQ", "real3", 0.5, st), ("TQQQ 비중 33%", "QQQ", "real3", 1 / 3, st),
                ("SPY 1배", "SPY", "1x", 1.0, st2), ("SPY 합성 2배", "SPY", "syn2", 1.0, st2), ("SPY 합성 3배", "SPY", "syn3", 1.0, st2),
                ("SPY 합성 3배 비중 50%", "SPY", "syn3", 0.5, st2), ("SPXL", "SPY", "real3", 1.0, st), ("SPXL 비중 50%", "SPY", "real3", 0.5, st)]
    series = {}
    for nm, base, kind, sc, start in lev_rows:
        s_ = pick(t, "G3-5", base, kind)
        s_ = s_[s_.d >= start]
        if not len(s_):
            continue
        cells = []
        for cname, c in COST_LEVELS[:2]:
            r = acct_one(book, [s_], start, c, sc)
            series[(nm, cname)] = r
            m = portfolio.stats(r)
            cells.append(f"{pct(m['cagr'], 1)} / {pct(m['mdd'], 1)} / {m['sharpe']:.2f}")
        y = portfolio.yearly(series[(nm, COST_LEVELS[0][0])])
        lines.append(f"| {nm} | {pd.Timestamp(start).year + 0}~ | " + " | ".join(cells) +
                     f" | {pct(y.get(2008, np.nan), 1)} | {pct(y.get(2022, np.nan), 1)} |")
    for tk, start in (("SPY", st), ("QQQ", st), ("SPY", st2)):
        r = bh_series(book, tk, start)
        m = portfolio.stats(r)
        y = portfolio.yearly(r)
        lines.append(f"| {tk} 그냥 보유 | {start.year}~ | {pct(m['cagr'], 1)} / {pct(m['mdd'], 1)} / {m['sharpe']:.2f} | | "
                     f"{pct(y.get(2008, np.nan), 1)} | {pct(y.get(2022, np.nan), 1)} |")
    # 해마다
    keys = [("TQQQ", COST_LEVELS[0][0]), ("TQQQ 비중 50%", COST_LEVELS[0][0]), ("QQQ 합성 2배", COST_LEVELS[0][0]),
            ("SPY 합성 3배", COST_LEVELS[0][0]), ("SPY 합성 3배 비중 50%", COST_LEVELS[0][0])]
    keys = [k for k in keys if k in series]
    cols = {f"{n} (표준)": portfolio.yearly(series[(n, c)]) for n, c in keys}
    f1 = pick(t, "F1", "QQQ", "real3")
    if len(f1):
        cols["F1 TQQQ (표준)"] = portfolio.yearly(acct_one(book, [f1[f1.d >= st]], st, std))
    cols["SPY"] = portfolio.yearly(bh_series(book, "SPY", st2))
    lines += ["", "## 해마다 (표준 수수료)", "", "| 해 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for yv in cols["SPY"].index:
        lines.append(f"| {yv} | " + " | ".join(pct(v.get(yv, np.nan), 1) for v in cols.values()) + " |")
    return t, lines


# ---------------------------------------------------------------- b51~b60 다른 신호 모음 3
def week_last(book, base):
    """그 주 마지막 거래일 여부 (다음 거래일이 다른 주)."""
    gi = pd.DatetimeIndex(pd.to_datetime(book.tab[base].d))
    wk = gi.isocalendar().year.to_numpy() * 100 + gi.isocalendar().week.to_numpy()
    return np.r_[wk[:-1] != wk[1:], False]


def ret_k(z, k):
    """오늘 판단가 / k거래일 전 종가 - 1."""
    out = np.full(len(z["c"]), np.nan)
    out[k:] = z["cd"][k:] / z["c"][:-k] - 1
    return out


def on_dates(book, base, other, arr):
    """other 종목의 날짜별 값 arr를 base 날짜에 맞춤 (없으면 nan)."""
    od = dict(zip(book.tab[other].d, arr))
    return np.array([od.get(d, np.nan) for d in book.tab[base].d], float)


def rule_b51(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        c, cd = z["c"], z["cd"]
        r = pd.Series(np.r_[np.nan, cd[1:] / c[:-1] - 1])  # 오늘은 판단가
        rc = pd.Series(np.r_[np.nan, c[1:] / c[:-1] - 1])
        v10 = np.sqrt(((rc - 0) ** 2).shift(1).rolling(9).sum().to_numpy() / 10 + r.to_numpy() ** 2 / 10)
        v60 = rc.shift(1).rolling(60).std().to_numpy()
        down = cd < z["prev_c"]
        n = len(c)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = v10 / v60
        for v, k, hold in (("V1", 1.8, 5), ("V2", 1.5, 5), ("V3", 2.2, 5), ("V4", 1.8, 3)):
            rows += trades_from(book, base, (ratio > k) & down, lambda i, h=hold: min(i + h, n - 1), v)
    t = pd.DataFrame(rows)
    names = {"V1": "V1 기본: 10일 변동성 > 60일 변동성 × 1.8 + 오늘 하락 → 5일 보유", "V2": "V2 1.5배", "V3": "V3 2.2배", "V4": "V4 3일 보유"}
    return t, report("b51 백테스트 결과: 변동성 급등 반등", "변동성 = 하루 수익 제곱 평균의 제곱근(10일), 60일은 표준편차. 오늘 수익은 판단가.", t, names,
                     acct=std_acct(t, names))


def rule_b52(book):
    spy = ind(book, "SPY")
    cnt = np.zeros(len(spy["c"]))
    for tk in SECTORS:
        if tk not in book.tab:
            continue
        z = ind(book, tk)
        cnt += np.nan_to_num(on_dates(book, "SPY", tk, (z["rsi"] < 10).astype(float)))
    ex = lambda i: short.first_true(spy["cd"] > spy["sma5"], i, 10)  # noqa: E731
    rows = []
    for v, k, filt in (("W1", 7, False), ("W2", 6, False), ("W3", 8, False), ("W4", 7, True)):
        en = (cnt >= k) & (spy["up200"] if filt else True)
        rows += trades_from(book, "SPY", en, ex, v)
    t = pd.DataFrame(rows)
    names = {"W1": "W1 기본: 섹터 9개 중 7개 이상 RSI(2) < 10 → SPY 5일 평균 위 마감 (최대 10일)", "W2": "W2 6개 이상",
             "W3": "W3 8개 이상", "W4": "W4 W1 + SPY 200일 위"}
    return t, report("b52 백테스트 결과: 섹터 폭 바닥", "섹터 SPDR 9개 (1998-12~). SPY·SPXL로 매매.", t, names, bases=("SPY",),
                     acct=std_acct(t, names, bases=("SPY",)))


def rule_b53(book):
    bases = [b for b in PAIRS if b in book.tab]
    spy_d = list(book.tab["SPY"].d)
    last = week_last(book, "SPY")
    r5 = {b: dict(zip(book.tab[b].d, ret_k(ind(book, b), 5))) for b in bases}
    ends = [k for k in np.nonzero(last)[0]]
    rows = []
    for v, mode, hold in (("R1", "worst", "week"), ("R2", "best", "week"), ("R3", "worst2", "week"), ("R4", "worst", 2)):
        for a, k in enumerate(ends):
            d = spy_d[k]
            vals = {b: r5[b].get(d, np.nan) for b in bases}
            vals = {b: x for b, x in vals.items() if not np.isnan(x)}
            if len(vals) < 2:
                continue
            b = max(vals, key=vals.get) if mode == "best" else min(vals, key=vals.get)
            if mode == "worst2" and vals[b] > -0.02:
                continue
            i = book.idx[b].get(d)
            if hold == "week":
                if a + 1 >= len(ends):
                    continue
                j = book.idx[b].get(spy_d[ends[a + 1]])
            else:
                j = i + hold if i is not None and i + hold < len(book.tab[b]) else None
            if i is None or j is None or not book.continuous(b, i, j):
                continue
            rows += make_rows(book, b, [(i, j, {})], v)
    t = pd.DataFrame(rows)
    names = {"R1": "R1 기본: 주 마지막 날 5일 수익 가장 나쁜 지수 3배 → 다음 주 마지막 날", "R2": "R2 가장 좋은 지수 (대조)",
             "R3": "R3 가장 나쁜 지수가 -2% 이하일 때만", "R4": "R4 2일 보유"}
    acct = []
    for v in names:
        for kind in ("real3", "syn3"):
            s_ = t[(t.variant == v) & (t.kind == kind) & (t.base != "ALL")].sort_values("d")
            if len(s_):
                acct.append((f"{v} {KIND_NAME[kind]}", [s_], False))
    return t, report("b53 백테스트 결과: 지수 간 되돌림 (주간)", "아래 '종목 모두 합쳐서'가 세 지수 합친 결과.", t, names,
                     acct=acct, pooled=True)


def rule_b54(book):
    rows = []
    with np.errstate(invalid="ignore"):
        for base, th, v in (("IWM", 0.03, "L1"), ("IWM", 0.02, "L2"), ("IWM", 0.04, "L3"), ("QQQ", 0.02, "L4")):
            if base not in book.tab:
                continue
            ro, rs = b44_parts(book, base, "SPY")
            n = len(book.tab[base])
            rows += trades_from(book, base, ro - rs > th, lambda i: min(i + 5, n - 1), v)
    t = pd.DataFrame(rows)
    names = {"L1": "L1 기본: IWM 5일 수익이 SPY보다 3%p 넘게 나쁨 → 5일 보유", "L2": "L2 2%p", "L3": "L3 4%p",
             "L4": "L4 QQQ가 SPY보다 2%p 넘게 나쁨 → TQQQ 5일"}
    return t, report("b54 백테스트 결과: 소형주·나스닥 뒤처짐", "", t, names, bases=("IWM", "QQQ"), acct=std_acct(t, names, bases=("IWM", "QQQ")))


def rule_b55(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        last = week_last(book, base)
        n = len(z["c"])
        for v, k, filt, hold in (("F1", 20, False, 5), ("F2", 10, False, 5), ("F3", 20, True, 5), ("F4", 20, False, 10)):
            en = last & (z["cd"] <= prior_min(z["c"], k - 1)) & (z["up200"] if filt else True)
            rows += trades_from(book, base, en, lambda i, h=hold: min(i + h, n - 1), v)
    t = pd.DataFrame(rows)
    names = {"F1": "F1 기본: 주 마지막 날 판단가가 20일 최저 종가 이하 → 5일 보유", "F2": "F2 10일 최저", "F3": "F3 + 200일 위",
             "F4": "F4 10일 보유"}
    return t, report("b55 백테스트 결과: 주간 4주 최저 마감", "최저 = 전날까지 19일(9일) 종가와 오늘 판단가.", t, names, acct=std_acct(t, names))


def rule_b56(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        c = z["c"]
        cdn = np.r_[False, c[1:] < c[:-1]]
        lag = lambda k: np.r_[np.zeros(k, bool), cdn[:-k]]  # noqa: E731
        d3 = lag(1) & lag(2) & lag(3)
        d4 = d3 & lag(4)
        up = z["cd"] / z["prev_c"] - 1
        n = len(c)
        for v, en, hold in (("C1", d3 & (up > 0.01), 5), ("C2", d3 & (up > 0.005), 5), ("C3", d4 & (up > 0.01), 5), ("C4", d3 & (up > 0.01), 3)):
            rows += trades_from(book, base, en, lambda i, h=hold: min(i + h, n - 1), v)
    t = pd.DataFrame(rows)
    names = {"C1": "C1 기본: 3일 이상 연속 하락 마감 뒤 오늘 +1% 넘게 오름 → 5일 보유", "C2": "C2 +0.5%", "C3": "C3 4일 이상 하락",
             "C4": "C4 3일 보유"}
    return t, report("b56 백테스트 결과: 확인 후 매수 (반등 첫날)", "", t, names, acct=std_acct(t, names))


def rule_b57(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        gap = z["o"] / z["prev_c"] - 1
        rec = z["cd"] / z["o"] - 1
        n = len(z["c"])
        for v, en, hold in (("H1", (gap <= -0.01) & (rec >= 0.01), 3), ("H2", (gap <= -0.005) & (rec >= 0.01), 3),
                            ("H3", (gap <= -0.01) & (rec >= 0.01), 1), ("H4", (gap <= -0.01) & (rec >= 0.01) & (z["cd"] > z["prev_c"]), 3)):
            rows += trades_from(book, base, en, lambda i, h=hold: min(i + h, n - 1), v)
    t = pd.DataFrame(rows)
    names = {"H1": "H1 기본: 시가 -1% 이상 갭 하락 + 판단가가 시가보다 +1% 이상 → 3일 보유", "H2": "H2 갭 -0.5%", "H3": "H3 1일 보유",
             "H4": "H4 + 판단가가 전날 종가 위"}
    return t, report("b57 백테스트 결과: 장중 반전 (갭 하락 뒤 회복)", "", t, names, acct=std_acct(t, names))


def rule_b58(book):
    rows = []
    for base in PAIRS:
        z = ind(book, base)
        vd = z["g"].vd.to_numpy(float)
        avg = pd.Series(vd).shift(1).rolling(50).mean().to_numpy()
        dn = z["cd"] / z["prev_c"] - 1 < -0.01
        n = len(vd)
        with np.errstate(invalid="ignore"):
            for v, k, hold in (("X1", 2.0, 5), ("X2", 1.5, 5), ("X3", 3.0, 5), ("X4", 2.0, 10)):
                rows += trades_from(book, base, (vd > k * avg) & dn, lambda i, h=hold: min(i + h, n - 1), v)
    t = pd.DataFrame(rows)
    names = {"X1": "X1 기본: 15:55까지 거래량 > 지난 50일 평균 × 2 + 판단가 -1% 넘게 하락 → 5일 보유", "X2": "X2 1.5배", "X3": "X3 3배",
             "X4": "X4 10일 보유"}
    return t, report("b58 백테스트 결과: 거래량 폭발 하락", "거래량은 기초 ETF의 9:30~15:55 합 (분할 전후 주식 수 차이는 SPY·QQQ·IWM에 거의 없음).",
                     t, names, acct=std_acct(t, names))


def rule_b59(book):
    rows = []
    for base in PAIRS:
        g = book.tab[base]
        ds = pd.to_datetime(g.d)
        yrs = sorted(set(ds.dt.year))
        idx = np.arange(len(g))
        for y in yrs:
            dec = idx[(ds.dt.year == y) & (ds.dt.month == 12)]
            jan = idx[(ds.dt.year == y + 1) & (ds.dt.month == 1)]
            nov = idx[(ds.dt.year == y) & (ds.dt.month == 11)]
            ent = []
            if len(dec) >= 10 and len(jan) >= 5:
                ent.append((dec[-6], jan[1], "S1"))
                dec15 = idx[(ds.dt.year == y) & (ds.dt.month == 12) & (ds.dt.day >= 15)]
                ent.append((dec15[0], jan[2], "S3"))
                ent.append((dec[-1], jan[4], "S4"))
            if len(nov):
                thu = [d for d in pd.date_range(f"{y}-11-01", f"{y}-11-30") if d.weekday() == 3][3]
                before = idx[ds < thu]
                after = idx[ds > thu]
                if len(before) >= 2 and len(after):
                    ent.append((before[-2], after[0], "S2"))  # 화요일 종가 → 금요일 종가
            for i, j, v in ent:
                if book.continuous(base, i, j):
                    rows += make_rows(book, base, [(i, j, {})], v)
    t = pd.DataFrame(rows)
    names = {"S1": "S1 산타 랠리: 12월 마지막 5거래일 + 1월 첫 2거래일", "S2": "S2 추수감사절: 전 화요일 종가 → 금요일 종가",
             "S3": "S3 12월 15일 이후 첫날 → 1월 3번째 거래일", "S4": "S4 1월 첫 5거래일"}
    return t, report("b59 백테스트 결과: 연말·추수감사절 계절", "1년에 한 번이라 거래 수가 적어요 (관문 30번은 참고).", t, names,
                     acct=std_acct(t, names))


def rule_b60(book):
    rows = []
    for base in ("QQQ", "SPY"):
        if base not in book.tab:
            continue
        n = len(book.tab[base])
        rk = {}
        for k in (5, 10):
            rk[k] = {tk: on_dates(book, base, tk, ret_k(ind(book, tk), k)) for tk in ("XLU", "XLP", "XLK", "XLY") if tk in book.tab}
        if len(rk[10]) < 4:
            continue
        with np.errstate(invalid="ignore"):
            for v, k, th in (("D1", 10, 0.04), ("D2", 10, 0.03), ("D3", 10, 0.05), ("D4", 5, 0.03)):
                x = rk[k]
                gap = (x["XLU"] + x["XLP"]) / 2 - (x["XLK"] + x["XLY"]) / 2
                rows += trades_from(book, base, gap > th, lambda i: min(i + 5, n - 1), v)
    t = pd.DataFrame(rows)
    names = {"D1": "D1 기본: 10일 수익 (XLU+XLP)/2 − (XLK+XLY)/2 > 4%p → 5일 보유", "D2": "D2 3%p", "D3": "D3 5%p", "D4": "D4 5일 수익, 3%p"}
    return t, report("b60 백테스트 결과: 방어주 앞섬", "섹터 SPDR (1998-12~).", t, names, bases=("QQQ", "SPY"), acct=std_acct(t, names))


def rule_b51x(book):
    """b51~b60 후보(b52 W3, b54 L4, b55 F1·F3, b57 H4, b59 S1)와 지난 최선 Q4(P5 + Y2)의 겹침, 한 계좌 성적."""
    t31, _ = rule_b31(book)
    t32, _ = rule_b32(book)
    t36, _ = rule_b36(book)
    t47, _ = rule_b47(book)
    t52, _ = rule_b52(book)
    t54, _ = rule_b54(book)
    t55, _ = rule_b55(book)
    t57, _ = rule_b57(book)
    t59, _ = rule_b59(book)
    start = dt.date(2011, 1, 1)
    clip = lambda x: x[x.d >= start]  # noqa: E731
    sig = {"W3 (b52) SPXL": pick(t52, "W3", "SPY", "real3"), "L4 (b54) TQQQ": pick(t54, "L4", "QQQ", "real3"),
           "F1 (b55) SPXL": pick(t55, "F1", "SPY", "real3"), "F3 (b55) SPXL": pick(t55, "F3", "SPY", "real3"),
           "H4 (b57) SPXL": pick(t57, "H4", "SPY", "real3"), "S1 (b59) SPXL": pick(t59, "S1", "SPY", "real3"),
           "S2 (b32) SPXL": pick(t32, "S2", "SPY", "real3"), "D3 (b31) TQQQ": pick(t31, "D3", "QQQ", "real3"),
           "B4 (b36) TQQQ": pick(t36, "B4", "QQQ", "real3"), "Y2 (b47) SPXL": pick(t47, "Y2", "SPY", "real3")}
    sig = {k: clip(v) for k, v in sig.items()}
    held = {k: {d for x in v.itertuples() for d in x.path_d} for k, v in sig.items()}
    ks = list(sig)
    lines = ["# b51~b60 후보 + 지난 최선 조합(Q4 = P5 + Y2) 합치기", "",
             "실제 3배 ETF, 2011~2026-03. 같은 날 여러 신호면 돈을 똑같이 나누고, 신호 없는 날은 현금. 비용은 종목 비중이 바뀐 만큼. "
             + COMMON_NOTE, "", "## 신호가 얼마나 겹치나 (두 신호 중 적은 쪽 보유일 가운데 같이 들고 있던 날 비율)", "",
             "| | 거래 수 | 보유일 | " + " | ".join(k.split(" ")[0] + " " + k.split(" ")[-1] for k in ks) + " |",
             "|---|---|---|" + "---|" * len(ks)]
    for a in ks:
        cells = [f"{len(held[a] & held[b]) / max(1, min(len(held[a]), len(held[b]))):.0%}" if a != b else "" for b in ks]
        lines.append(f"| {a} | {len(sig[a])} | {len(held[a])} | " + " | ".join(cells) + " |")
    q4 = set().union(*(held[k] for k in ("S2 (b32) SPXL", "D3 (b31) TQQQ", "B4 (b36) TQQQ", "Y2 (b47) SPXL")))
    lines += ["", "| 신호 | Q4 보유일과 겹치는 비율 | Q4 없는 날에만 있는 보유일 |", "|---|---|---|"]
    for k in ks[:6]:
        lines.append(f"| {k} | {len(held[k] & q4) / max(1, len(held[k])):.0%} | {len(held[k] - q4)} |")
    S = sig
    Q4 = [S["S2 (b32) SPXL"], S["D3 (b31) TQQQ"], S["B4 (b36) TQQQ"], S["Y2 (b47) SPXL"]]
    combos = [("Q4 P5 + Y2 (지난 최선)", Q4, False),
              ("N1 Q4 + S1 산타", Q4 + [S["S1 (b59) SPXL"]], False),
              ("N2 Q4 + L4 TQQQ", Q4 + [S["L4 (b54) TQQQ"]], False),
              ("N3 Q4 + F1 SPXL", Q4 + [S["F1 (b55) SPXL"]], False),
              ("N4 Q4 + F3 SPXL", Q4 + [S["F3 (b55) SPXL"]], False),
              ("N5 Q4 + W3 SPXL", Q4 + [S["W3 (b52) SPXL"]], False),
              ("N6 Q4 + H4 SPXL", Q4 + [S["H4 (b57) SPXL"]], False),
              ("N7 Q4 + S1 + L4", Q4 + [S["S1 (b59) SPXL"], S["L4 (b54) TQQQ"]], False),
              ("N8 Q4 + 후보 전부", Q4 + [S[k] for k in ks[:6]], False),
              ("N9 후보 6개만 (Q4 없이)", [S[k] for k in ks[:6]], False)]
    lines += ["", "## 한 계좌 성적 (2011~2026-03)", ""]
    a_lines, series = acct_table(book, combos, start=start)
    lines += a_lines
    lines += ["", "## 2020년 빼고 (그해 수익을 0으로)", "", *no2020_table(book, series, start)]
    lines += ["", "## 해마다", "", *yearly_series(series, [("Q4 P5 + Y2 (지난 최선)", COST_LEVELS[1][0]), ("N7 Q4 + S1 + L4", COST_LEVELS[1][0]),
                                                         ("N8 Q4 + 후보 전부", COST_LEVELS[1][0]), ("N8 Q4 + 후보 전부", COST_LEVELS[0][0])],
                                         book, start)]
    return pd.concat([t52, t54, t55], ignore_index=True), lines


RULE_TICKERS = {"b34": ["SPY"] + SECTORS, "b35": ASSETS + ["SPXL", "BGU", "TQQQ", "TNA"],
                "b31x": ASSETS + ["SPXL", "BGU", "TQQQ", "TNA"], "b44": ["SPY", "QQQ", "IWM", "TLT", "GLD", "SPXL", "BGU", "TQQQ", "TNA"],
                "b45": ASSETS + ["SPXL", "BGU", "TQQQ", "TNA"],
                "b41x": ["SPY", "QQQ", "IWM", "TLT", "GLD", "SPXL", "BGU", "TQQQ", "TNA"],
                "b44x": ["SPY", "QQQ", "IWM", "TLT", "SPXL", "BGU", "TQQQ", "TNA"],
                "b52": ["SPY", "SPXL", "BGU"] + SECTORS, "b60": ["SPY", "QQQ", "SPXL", "BGU", "TQQQ", "XLU", "XLP", "XLK", "XLY"],
                "b51x": ["SPY", "QQQ", "IWM", "SPXL", "BGU", "TQQQ", "TNA"] + SECTORS}


RULES = {"b26": rule_b26, "b27": rule_b27, "b28": rule_b28, "b29": rule_b29, "b30": rule_b30, "b29x": rule_b29x, "b31": rule_b31, "b32": rule_b32, "b33": rule_b33,
         "b34": rule_b34, "b35": rule_b35, "b36": rule_b36, "b37": rule_b37, "b38": rule_b38, "b39": rule_b39, "b40": rule_b40,
         "b31x": rule_b31x, "b41": rule_b41, "b42": rule_b42, "b43": rule_b43, "b44": rule_b44, "b45": rule_b45,
         "b46": rule_b46, "b47": rule_b47, "b48": rule_b48, "b49": rule_b49, "b50": rule_b50, "b41x": rule_b41x, "b44x": rule_b44x, "b51": rule_b51, "b52": rule_b52, "b53": rule_b53, "b54": rule_b54, "b55": rule_b55,
         "b56": rule_b56, "b57": rule_b57, "b58": rule_b58, "b59": rule_b59, "b60": rule_b60, "b51x": rule_b51x}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_etf")
    ap.add_argument("--rule", required=True, choices=sorted(RULES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--prep2", default=None, help="b34·b35 섹터·채권·금 ETF 자료 폴더 (prep_etf2)")
    a = ap.parse_args()
    out = a.out or f"reports/{a.rule}"
    os.makedirs(out, exist_ok=True)
    tickers = RULE_TICKERS.get(a.rule, ["SPY", "QQQ", "IWM", "SPXL", "BGU", "TQQQ", "TNA"])
    bars = load(a.prep, [t for t in tickers if t not in INCEPTION2])
    if a.prep2 and any(t in INCEPTION2 for t in tickers):
        b2 = load(a.prep2, [t for t in tickers if t in INCEPTION2])
        first = b2.ticker.map({k: dt.date.fromisoformat(v) for k, v in INCEPTION2.items()})
        bars = pd.concat([bars, b2[b2.d >= first]], ignore_index=True)
    bars = bars[bars.mi >= 0]
    book = Book(bars)
    global BOOK
    BOOK = book
    print("tickers:", {tk: (str(g.d.iloc[0]), str(g.d.iloc[-1]), len(g)) for tk, g in book.tab.items()}, flush=True)
    print("splits:", [(tk, str(d), k) for tk, d, k in book.splits], flush=True)
    t, lines = RULES[a.rule](book)
    t.drop(columns=["path", "path_d"], errors="ignore").to_csv(os.path.join(out, "trades.csv.gz"), index=False)
    text = "\n".join(lines) + "\n"
    open(os.path.join(out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
