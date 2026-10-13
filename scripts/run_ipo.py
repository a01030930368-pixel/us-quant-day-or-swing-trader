"""b16(큰 IPO 첫날)·b17(IPO 둘째 날)·b18(IPO 베이스 돌파) 백테스트.

    python scripts/run_ipo.py --b08 prep_b08 --ipo prep_ipo --sec sec --rule b16 --out reports/b16
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_b08 import ipo_days, read_all  # noqa: E402
from daytrade import ipo, portfolio  # noqa: E402

FEE, SLIP, SLIP_LOW = 0.0025, 0.002, 0.001  # 왕복 0.9%, 참고 0.7%
GATE, MIN_N = 0.011, 30
WEIGHT, MAX_PER_DAY = 0.2, 5
PERIODS = [("1993~2003", 1993, 2003), ("2004~2021", 2004, 2021), ("2022~2026", 2022, 2026)]


def tstat(g):
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g.std() > 0 else np.nan


def pct(x, k=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{k}f}%"


HEAD = ("| 변형 | 수 | 비용 전 평균 | 중앙값 | 승률 | t값 | 비용 후 (왕복 0.9%) | 비용 후 (왕복 0.7%) | 2004~ 평균 (수) | 2022~ 평균 (수) | 상위 10개 뺀 평균 | 관문 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|")


def verdict(s, judge, both_periods):
    if not judge:
        return "(판정 안 함)", False
    g = s.gross
    if len(g) < MIN_N:
        return "표본 부족", False
    y = s.d.dt.year
    ok = g.mean() >= GATE
    if both_periods:
        a, b = g[y >= 2004], g[y >= 2022]
        ok = len(a) > 0 and len(b) > 0 and a.mean() >= GATE and b.mean() >= GATE
    return ("통과" if ok else "탈락"), ok


def row(name, s, judge=True, both_periods=False):
    if s is None or s.empty:
        return f"| {name} | 0 |" + " |" * 11, False
    g, y = s.gross, s.d.dt.year
    v, ok = verdict(s, judge, both_periods)
    a, b = g[y >= 2004], g[y >= 2022]
    top = g.sort_values(ascending=False).iloc[10:]
    return (f"| {name} | {len(g)} | {pct(g.mean())} | {pct(g.median())} | {(g > 0).mean():.0%} | {tstat(g):.2f} | "
            f"{pct(portfolio.net_return(g, FEE + SLIP).mean())} | {pct(portfolio.net_return(g, FEE + SLIP_LOW).mean())} | "
            f"{pct(a.mean())} ({len(a)}) | {pct(b.mean())} ({len(b)}) | {pct(top.mean())} | {v} |"), ok


def report_common(title, note, sel, names, judge, main_codes, spy, both_periods=False, swing=()):
    lines = [title, "", note, "", "## 1차 관문", "", *HEAD]
    passed = []
    for c, s in sel.items():
        r, ok = row(names[c], s, judge=c in judge, both_periods=both_periods)
        lines.append(r)
        if ok:
            passed.append(c)
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else '없음 → 탈락'}**", ""]
    lines += ["## 기간 나눠 보기 (비용 전 평균 / 중앙값, 괄호는 수)", "",
              "| 변형 | " + " | ".join(p[0] for p in PERIODS) + " |", "|---" * (len(PERIODS) + 1) + "|"]
    for c, s in sel.items():
        if s is None or s.empty:
            continue
        y = s.d.dt.year
        cells = [s.gross[(y >= y0) & (y <= y1)] for _, y0, y1 in PERIODS]
        lines.append(f"| {names[c]} | " + " | ".join(f"{pct(z.mean())} / {pct(z.median())} ({len(z)})" for z in cells) + " |")
    lines += ["", "## 해마다 (비용 전 평균, 괄호는 수)", "", "| 해 | " + " | ".join(names[c].split(" ")[0] for c in main_codes) + " |",
              "|---" * (len(main_codes) + 1) + "|"]
    years = sorted({y for c in main_codes if not sel[c].empty for y in sel[c].d.dt.year})
    for yy in years:
        cells = []
        for c in main_codes:
            g = sel[c].gross[sel[c].d.dt.year == yy] if not sel[c].empty else pd.Series(dtype=float)
            cells.append(f"{pct(g.mean())} ({len(g)})" if len(g) else "")
        lines.append(f"| {yy} | " + " | ".join(cells) + " |")
    base = sel[main_codes[0]]
    if not base.empty:
        def fmt(r):
            return f"| {r.d.date()} | {r.ticker} | {r.entry_px:.2f} | {r.gross * 100:+.1f}% |"
        th = ["| 날짜 | 종목 | 매수가 | 비용 전 |", "|---|---|---|---|"]
        lines += ["", f"## {names[main_codes[0]].split(' ')[0]} 가장 좋은 10개", "", *th] + [fmt(r) for r in base.nlargest(10, "gross").itertuples()]
        lines += ["", f"## {names[main_codes[0]].split(' ')[0]} 가장 나쁜 10개", "", *th] + [fmt(r) for r in base.nsmallest(10, "gross").itertuples()]
        if "reason" in base:
            lines.append(f"\n청산 이유: {', '.join(f'{k} {v}' for k, v in base.reason.value_counts().items())}. "
                         f"매수 시각 중앙값 9:30 + {int(base.entry_mi.median())}분.")
    cal = spy.index[spy.index >= pd.Timestamp("1993-01-01")]
    s0 = portfolio.stats(spy[cal].pct_change().fillna(0.0))
    lines += ["", f"## 계좌 기준 (한 종목에 계좌 {WEIGHT:.0%}, 하루 최대 {MAX_PER_DAY}개, 비용 왕복 0.9%, 1993~)", "",
              "| 변형 | 연수익률 | 최대낙폭 |", "|---|---|---|", f"| SPY 보유 (배당 제외) | {pct(s0['cagr'], 1)} | {pct(s0['mdd'], 1)} |"]
    for c, s in sel.items():
        if s is None or s.empty or c in swing:
            continue
        s = s.sort_values(["d", "entry_mi"]).groupby("d").head(MAX_PER_DAY)
        r = (WEIGHT * portfolio.net_return(s.gross, FEE + SLIP)).groupby(s.d).sum().reindex(cal, fill_value=0.0)
        st = portfolio.stats(r)
        lines.append(f"| {names[c]} | {pct(st['cagr'], 1)} | {pct(st['mdd'], 1)} |")
    return lines, passed


def load_sec(sec):
    p = os.path.join(sec or "", "companies.parquet")
    if not sec or not os.path.exists(p):
        return None
    comp = pd.read_parquet(p)
    return {r.ticker: r for r in comp.itertuples()}


def ipo_base(a):
    tick, bars, spy = read_all(a.b08, "tick"), read_all(a.b08, "bars"), read_all(a.b08, "spy")
    spy = spy.drop_duplicates("d").assign(d=lambda x: pd.to_datetime(x.d)).set_index("d").c.sort_index()
    ipos, drop, cand = ipo_days(tick, bars.sort_values(["ticker", "d", "mi"]), spy)
    comp = load_sec(a.sec)
    info = []
    for tk, d, day, dv in ipos:
        sp = ipo.is_despac(comp.get(tk), d) if comp is not None else None
        info.append(dict(ticker=tk, d=d, despac=sp, dv1=float(day["c"][0] * day["v"][0]), dv60=dv,
                         first_mi=int(day["mi"][0]), **ipo.day1_summary(day)))
    info = pd.DataFrame(info)
    days0 = {(tk, d): day for tk, d, day, _ in ipos}
    sec_note = ("SEC 자료 없음: 스팩 합병 제외를 못 함" if comp is None else
                f"스팩 합병·이미 공시하던 회사로 본 것 {int((info.despac == True).sum())}개, "  # noqa: E712
                f"SEC 자료로 확인 안 된 것(상장폐지 등) {int(info.despac.isna().sum())}개 (그대로 둠)")
    return info, days0, spy, drop, cand, sec_note


def read_ipo_prep(prep, kind):
    fs = sorted(glob.glob(os.path.join(prep, f"{kind}_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["d"] = pd.to_datetime(df.d)
    return df


def minute_days(prep):
    b = read_ipo_prep(prep, "ipob").sort_values(["ticker", "d", "mi"])
    return {(tk, d): {k: g[k].to_numpy() for k in ("mi", "o", "h", "l", "c", "v")} for (tk, d), g in b.groupby(["ticker", "d"], sort=False)}


# ---------------------------------------------------------------- b16
def rule_b16(a):
    info, days0, spy, drop, cand, sec_note = ipo_base(a)
    rows = []
    for r in info.itertuples():
        day = days0[(r.ticker, r.d)]
        base = dict(ticker=r.ticker, d=r.d, despac=r.despac, dv1=r.dv1, dv60=r.dv60, first_mi=r.first_mi)
        t = ipo.first_close_to_close(day)
        if t:
            rows.append(dict(base, mode="c1", **t))
        t = ipo.first_close_to_close(day, wait=5)
        if t:
            rows.append(dict(base, mode="c5", **t))
    t = pd.DataFrame(rows)
    tm = (t.first_mi >= 1) & (t.first_mi <= 270)
    nosp = t.despac != True  # noqa: E712
    c1, c5 = t["mode"] == "c1", t["mode"] == "c5"
    sel = {"Z1": t[c1 & tm & nosp & (t.dv1 >= 10e6)],
           "Z2": t[c1 & tm & nosp & (t.dv1 >= 30e6)],
           "Z3": t[c1 & tm & (t.dv1 >= 10e6)],
           "Z4": t[c1 & tm & nosp & (t.dv60 >= 100e6)],
           "Z5": t[c5 & tm & nosp & (t.dv1 >= 10e6)],
           "Z6": t[c1 & (t.first_mi >= 90) & (t.first_mi <= 270) & nosp & (t.dv1 >= 10e6)],
           "ref": t[c1 & tm & nosp]}
    names = {"Z1": "Z1 기본 (첫 1분봉 거래대금 ≥ 1,000만 달러, 9:31~14:00 첫 거래, 스팩 합병 제외, 첫 1분봉 종가 → 15:59 종가)",
             "Z2": "Z2 첫 1분봉 거래대금 ≥ 3,000만 달러", "Z3": "Z3 스팩 합병 제외 없이",
             "Z4": "Z4 첫 60분 거래대금 ≥ 1억 달러 (미래 정보, 판정 안 함)",
             "Z5": "Z5 첫 5분째 종가가 첫 1분봉 종가보다 높을 때만, 그 값에 매수",
             "Z6": "Z6 첫 거래 11:00 이후만", "ref": "(참고) 크기 조건 없이 9:31~14:00 IPO 모두"}
    judge = {"Z1", "Z2", "Z3", "Z5", "Z6"}
    note = (f"IPO 첫날 찾기는 b08과 같음: 후보 {len(cand)}개 중 IPO {len(info)}개 (빠진 이유: "
            + ", ".join(f"{k} {v}" for k, v in drop.items()) + f"). {sec_note}. "
            "비용 왕복 0.9% (참고 0.7%). 관문: 비용 전 거래당 평균이 **2004~2026과 2022~2026 둘 다** ≥ +1.1% (30개 이상).")
    lines, _ = report_common("# b16 백테스트 결과: 큰 IPO 첫날 (스팩 합병 제외)", note, sel, names, judge,
                             ["Z1", "Z3", "Z6", "ref"], spy, both_periods=True)
    sp = t[c1 & tm & (t.dv1 >= 10e6) & (t.despac == True)]  # noqa: E712
    lines += ["", f"## 스팩 합병 등으로 빠진 것 (Z3에는 있고 Z1에는 없음): {len(sp)}개, 비용 전 평균 {pct(sp.gross.mean())}",
              ", ".join(f"{r.ticker}({r.d.date()}, {r.gross * 100:+.1f}%)" for r in sp.nsmallest(15, "gross").itertuples())]
    lines += ["", "## 첫 1분봉 거래대금 구간별 (9:31~14:00, 스팩 제외, 비용 전)", "", "| 구간 | 수 | 평균 | 중앙값 | 2022~ 평균 (수) |", "|---|---|---|---|---|"]
    ref = sel["ref"]
    for lo, hi in ((0, 3e6), (3e6, 10e6), (10e6, 30e6), (30e6, 100e6), (100e6, 1e15)):
        s = ref[(ref.dv1 >= lo) & (ref.dv1 < hi)]
        g, g2 = s.gross, s.gross[s.d.dt.year >= 2022]
        lab = f"{lo / 1e6:.0f}백만 달러 이상" if hi > 1e14 else f"{lo / 1e6:.0f}~{hi / 1e6:.0f}백만 달러"
        lines.append(f"| {lab} | {len(g)} | {pct(g.mean())} | {pct(g.median())} | {pct(g2.mean())} ({len(g2)}) |")
    return t, lines


# ---------------------------------------------------------------- b17
def rule_b17(a):
    info, days0, spy, drop, cand, sec_note = ipo_base(a)
    daily = read_ipo_prep(a.ipo, "ipod")
    mdays = minute_days(a.ipo)
    dly = {tk: g.sort_values("k").reset_index(drop=True) for tk, g in daily.groupby("ticker")}
    rows = []
    for r in info.itertuples():
        g = dly.get(r.ticker)
        if g is None or len(g) < 2 or g.d.iloc[0] != r.d:
            continue
        d1 = g.d.iloc[1]
        day = mdays.get((r.ticker, d1))
        if day is None or not len(day["mi"]):
            continue
        strong = r.c0 > r.o0
        near_hi = (r.c0 - r.l0) >= 0.7 * (r.h0 - r.l0) if r.h0 > r.l0 else False
        hot = r.c0 > 1.5 * r.o0
        size = r.dv0 >= 20e6 and r.c0 >= 5
        gapok = day["o"][0] >= 0.97 * r.c0
        base = dict(ticker=r.ticker, d=d1, despac=r.despac, strong=strong, near_hi=near_hi, hot=hot, size=size, gapok=gapok)
        t1 = ipo.b17(day, r.h0)
        if t1:
            rows.append(dict(base, mode="T1", **t1))
            sw = ipo.hold_days(dict(t1), g, 1, 5)
            if sw is not None:
                rows.append(dict(base, mode="T6", **dict(t1, gross=sw)))
        t = ipo.b17(day, r.h0, stop=False)
        if t:
            rows.append(dict(base, mode="T5", **t))
        t = ipo.open_to_close(day)
        if t:
            rows.append(dict(base, mode="T2", **t))
    t = pd.DataFrame(rows)
    ok = (t.despac != True) & t["size"] & t.gapok  # noqa: E712
    st, nh, hot = t.strong, t.near_hi, t.hot
    m = t["mode"]
    sel = {"T1": t[(m == "T1") & ok & st & nh & ~hot],
           "T2": t[(m == "T2") & ok & st & nh & ~hot],
           "T3a": t[(m == "T1") & ok & st & nh],
           "T3b": t[(m == "T1") & ok & st & nh & hot],
           "T4": t[(m == "T1") & ok & ~hot],
           "T4b": t[(m == "T2") & (t.despac != True)],  # noqa: E712
           "T5": t[(m == "T5") & ok & st & nh & ~hot],
           "T6": t[(m == "T6") & ok & st & nh & ~hot]}
    names = {"T1": "T1 기본 (첫날 강함·고가 근처 마감·과열 아님, 2일차 1일차 고가 돌파, 손절 그날 저가)",
             "T2": "T2 돌파 기다리지 않고 2일차 9:31 시가 매수", "T3a": "T3 과열(첫 시가 대비 +50% 넘게 마감) 포함",
             "T3b": "T3 과열만", "T4": "T4 첫날 강함·고가 근처 조건 없이 (돌파 매수)",
             "T4b": "(참고) 모든 IPO 2일차 9:31 시가 → 종가 (판정 안 함)", "T5": "T5 손절 없이",
             "T6": "T6 (비교용, 스윙) T1 매수 후 5거래일째 종가"}
    judge = {"T1", "T2", "T3a", "T3b", "T4", "T5"}
    note = (f"IPO {len(info)}개 (b08 방법). {sec_note}. 2일차 1분봉이 있는 IPO만. 비용 왕복 0.9% (참고 0.7%). "
            "관문: 비용 전 거래당 평균 ≥ +1.1% (30개 이상). 2004년 이후·2022년 이후 따로 표시.")
    lines, _ = report_common("# b17 백테스트 결과: IPO 둘째 날 이어가기", note, sel, names, judge, ["T1", "T2", "T4b"], spy, swing=("T6",))
    return t, lines


# ---------------------------------------------------------------- b18
def rule_b18(a):
    info, days0, spy, drop, cand, sec_note = ipo_base(a)
    daily = read_ipo_prep(a.ipo, "ipod")
    mdays = minute_days(a.ipo)
    ok_ipo = {(r.ticker, r.d) for r in info.itertuples() if r.despac != True}  # noqa: E712
    rows, n_cand = [], 0
    for tk, g in daily.groupby("ticker"):
        g = g.sort_values("k").reset_index(drop=True)
        if (tk, g.d.iloc[0]) not in ok_ipo:
            continue
        h, l, c, dv, o0 = g.h.to_numpy(), g.l.to_numpy(), g.c.to_numpy(), g.dv.to_numpy(), g.o.iloc[0]
        for k in range(5, min(41, len(g))):
            i_h = int(np.argmax(h[:k]))
            H = h[i_h]
            depth = 1 - l[i_h:k].min() / H
            if k - i_h < 5 or depth > 0.20 or not c[k - 1] > o0 or c[k - 1] < 5 or dv[k - 5:k].mean() < 10e6 or not h[k] > H:
                continue
            day = mdays.get((tk, g.d[k]))
            prev = [mdays.get((tk, g.d[j])) for j in range(k - 5, k)]
            if day is None or any(p is None for p in prev):
                continue
            n_cand += 1
            prof = ipo.cum_profile(prev)
            base = dict(ticker=tk, d=g.d[k], k=k, depth=depth, H=H)
            for code, kw in (("P1", {}), ("P2", dict(stop=None)), ("P3", dict(mult=2.5))):
                t = ipo.b18(day, H, prof, **kw)
                if t:
                    rows.append(dict(base, mode=code, **t))
                    if code == "P1":
                        sw = ipo.hold_days(dict(t), g, k, 10, stop_px=t["entry_px"] * 0.93)
                        if sw is not None:
                            rows.append(dict(base, mode="P6", **dict(t, gross=sw)))
    t = pd.DataFrame(rows)
    m = t["mode"]
    sel = {"P1": t[m == "P1"], "P2": t[m == "P2"], "P3": t[m == "P3"], "P4": t[(m == "P1") & (t.k <= 25)],
           "P5": t[(m == "P1") & (t.depth <= 0.12)], "P6": t[m == "P6"]}
    names = {"P1": "P1 기본 (상장 5~40일, 베이스 깊이 ≤ 20%, 고점 돌파 + 거래량 1.5배, 손절 -7%)", "P2": "P2 손절 없이",
             "P3": "P3 거래량 2.5배", "P4": "P4 상장 5~25일만", "P5": "P5 베이스 깊이 ≤ 12%",
             "P6": "P6 (비교용, 스윙) -7% 손절 또는 10거래일째 종가"}
    judge = {"P1", "P2", "P3", "P4", "P5"}
    note = (f"IPO {len(info)}개 (b08 방법, 스팩 합병 제외). {sec_note}. 조건 맞는 돌파 후보일 {n_cand}개. "
            "비용 왕복 0.9% (참고 0.7%). 관문: 비용 전 거래당 평균 ≥ +1.1% (30개 이상).")
    lines, _ = report_common("# b18 백테스트 결과: 새내기주 IPO 베이스 돌파", note, sel, names, judge, ["P1", "P2", "P6"], spy, swing=("P6",))
    return t, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b08", default="prep_b08")
    ap.add_argument("--ipo", default="prep_ipo")
    ap.add_argument("--sec", default="sec")
    ap.add_argument("--rule", required=True, choices=["b16", "b17", "b18"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"reports/{a.rule}"
    os.makedirs(out, exist_ok=True)
    t, lines = {"b16": rule_b16, "b17": rule_b17, "b18": rule_b18}[a.rule](a)
    t.to_csv(os.path.join(out, "trades.csv.gz"), index=False)
    text = "\n".join(lines) + "\n"
    open(os.path.join(out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
