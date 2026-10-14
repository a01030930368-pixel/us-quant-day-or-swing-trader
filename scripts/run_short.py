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

    def continuous(self, tk, i, j):
        p = self.pos[tk]
        return p[i] >= 0 and p[j] - p[i] == j - i
        self.by_day = {k: (g.mi.to_numpy(), g.c.to_numpy()) for k, g in bars.groupby(["ticker", "d"], sort=False)}

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
        lev = PAIRS[base]
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
                tk = PAIRS[base]
                if tk not in self.tab:
                    self._base[key] = np.nan
                    return np.nan
                g = self.tab[tk]
            else:
                g = self.tab[base]
            c = g.c.to_numpy()
            ok = np.ones(len(c) - n, bool) if since is None else (pd.to_datetime(g.d[:len(c) - n]).dt.year >= since).to_numpy()
            p = self.pos[g.ticker.iloc[0]]
            ok &= (p[:-n] >= 0) & (p[n:] - p[:-n] == n)
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
    return base if kind == "1x" else (f"{base}×3 합성" if kind == "syn3" else PAIRS[base])


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
                r1 = px / c[i] - 1
                legs = {"1x": r1}
                # 합성: 그 전 날들은 하루씩 3배, 발표일은 전날 종가 → 발표 직전 × 3
                prev = c[i + 1:k] / c[i:k - 1] - 1 if k - 1 > i else np.array([])
                part = px / c[k - 1] - 1
                legs["syn3"] = float(np.prod(1 + 3 * prev) * (1 + 3 * part) - 1 - short.SYN_DAILY_COST * back)
                lev = PAIRS[base]
                if lev in book.idx:
                    a, b = book.idx[lev].get(g.d[i]), book.idx[lev].get(r.d)
                    if a is not None and b is not None and b - a == back:
                        pl = book.price_before(lev, r.d, r.ann_mi)
                        if pl is not None:
                            legs["real3"] = pl / book.tab[lev].c.iloc[a] - 1
                for kind, gross in legs.items():
                    rows.append(dict(variant=v, base=base, kind=kind, d=g.d[i], exit_d=r.d, days=back, gross=gross,
                                     path=[gross], path_d=[r.d], ann=r.announce))
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


RULES = {"b26": rule_b26, "b27": rule_b27, "b28": rule_b28, "b29": rule_b29, "b30": rule_b30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_etf")
    ap.add_argument("--rule", required=True, choices=sorted(RULES))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"reports/{a.rule}"
    os.makedirs(out, exist_ok=True)
    bars = load(a.prep, ["SPY", "QQQ", "IWM", "SPXL", "BGU", "TQQQ", "TNA"])
    bars = bars[bars.mi >= 0]
    book = Book(bars)
    global BOOK
    BOOK = book
    print("tickers:", {tk: (str(g.d.iloc[0]), str(g.d.iloc[-1]), len(g)) for tk, g in book.tab.items()}, flush=True)
    print("splits:", [(tk, str(d), k) for tk, d, k in book.splits], flush=True)
    t, lines = RULES[a.rule](book)
    t.drop(columns=["path", "path_d"]).to_csv(os.path.join(out, "trades.csv.gz"), index=False)
    text = "\n".join(lines) + "\n"
    open(os.path.join(out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
