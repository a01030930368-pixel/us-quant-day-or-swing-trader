"""b07 데이트레이딩 백테스트 실행 (FOMC 안도 랠리): prep_b06 폴더(1분봉) + daytrade/fomc_dates.csv -> 결과표.

    python scripts/run_b07.py --prep prep_b06 --out reports/b07
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daytrade import b07, portfolio  # noqa: E402

FEE, SLIP, SLIP_LOW = 0.0025, 0.001, 0.0002
GATE = 0.009
WEIGHT = 0.5
LONG3 = ("SPXL", "BGU", "UPRO")
CLOSE = 389
PERIODS = [("논문 전 2011~2017", 2011, 2017), ("논문 후 2018~2026", 2018, 2026)]


def load_bars(prep):
    df = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(os.path.join(prep, "bars_*.parquet")))])
    df = df[pd.DatetimeIndex(df.d).year >= 2010].drop_duplicates(["ticker", "d", "mi"]).sort_values(["ticker", "d", "mi"])
    return {k: {c: g[c].to_numpy() for c in ("mi", "o", "h", "l", "c", "v")} for k, g in df.groupby(["ticker", "d"], sort=False)}


def pick(bars, cands, d):
    return next((tk for tk in cands if (tk, d) in bars), None)


def stats_row(name, s, judge=True):
    if s.empty:
        return f"| {name} | 0 |" + " |" * 9, False
    g = s.gross
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 and g.std() > 0 else np.nan
    n1, n2 = portfolio.net_return(g, FEE + SLIP), portfolio.net_return(g, FEE + SLIP_LOW)
    tn = n1.mean() / (n1.std(ddof=1) / np.sqrt(len(n1))) if len(n1) > 1 and n1.std() > 0 else np.nan
    ok = judge and g.mean() >= GATE
    verdict = ("통과" if ok else "탈락") if judge else "(판정 안 함)"
    return (f"| {name} | {len(g)} | {g.mean() * 100:.2f}% | {g.median() * 100:.2f}% | {(g > 0).mean():.0%} | {t:.2f} | "
            f"{n1.mean() * 100:.2f}% | {n2.mean() * 100:.2f}% | {(n1 > 0).mean():.0%} | {tn:.2f} | {verdict} |"), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_b06")
    ap.add_argument("--fomc", default="daytrade/fomc_dates.csv")
    ap.add_argument("--fed-check", default="fomc_check.txt")
    ap.add_argument("--out", default="reports/b07")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bars = load_bars(a.prep)
    fomc = pd.read_csv(a.fomc, parse_dates=["date"], dtype={"presser_start": str})
    fomc["d"] = fomc.date.dt.date
    spy_days = sorted(d for (tk, d) in bars if tk == "SPY")
    spy = pd.Series([bars[("SPY", d)]["c"][-1] for d in spy_days], index=pd.DatetimeIndex(spy_days))
    vol20 = spy.pct_change().rolling(20).std()
    vol_hi = vol20.shift(1) >= vol20.shift(1).rolling(252, min_periods=120).median()

    # 발표 시각 확인: 후보 시각(12:30, 14:00, 14:15) 중 거래량이 가장 크게 튄 시각이 표의 발표 시각인가
    cands = ("12:30", "14:00", "14:15")
    chk = []
    for r in fomc.itertuples():
        if ("SPY", r.d) not in bars:
            continue
        sp = {c: b07.spike(bars[("SPY", r.d)], b07.to_mi(c)) or 0.0 for c in cands}
        best = max(sp, key=sp.get)
        chk.append(dict(d=r.d, announce=r.announce, best=best, ratio=sp[r.announce], top=best == r.announce, ok=sp[r.announce] >= 3))
    chk = pd.DataFrame(chk)

    rows = []
    for r in fomc.itertuples():
        d = r.d
        if ("SPY", d) not in bars:
            continue
        A = b07.to_mi(r.announce)
        base = dict(d=d, presser=bool(r.presser), sep=bool(r.sep))
        spxl = pick(bars, LONG3, d)
        for code, tk, t_in, t_out in (("V1", "SPY", A - 5, A + 55), ("V2", spxl, A - 5, A + 55), ("V3", "TQQQ", A - 5, A + 55),
                                      ("V4", spxl, A - 5, CLOSE), ("V7", spxl, 1, A - 5)):
            if tk and (tk, d) in bars:
                t = b07.window(bars[(tk, d)], t_in, t_out)
                if t:
                    rows.append(dict(base, code=code, ticker=tk, **t))
        if r.presser and spxl and isinstance(r.presser_start, str):
            P = b07.to_mi(r.presser_start)
            p0, p1 = b07.price_before(bars[("SPY", d)], A), b07.price_before(bars[("SPY", d)], P)
            if p0 and p1 and p1 > p0:
                t = b07.window(bars[(spxl, d)], P + 1, CLOSE)
                if t:
                    rows.append(dict(base, code="V6", ticker=spxl, **t))
    t = pd.DataFrame(rows)
    t["year"] = pd.DatetimeIndex(t.d).year
    t["vol_hi"] = vol_hi.reindex(pd.DatetimeIndex(t.d)).fillna(False).to_numpy()
    t.to_csv(os.path.join(a.out, "trades.csv"), index=False)

    pc = t.presser
    sel = {"V1": t[(t.code == "V1") & pc], "V2": t[(t.code == "V2") & pc], "V3": t[(t.code == "V3") & pc],
           "V4": t[(t.code == "V4") & pc], "V5-all": t[t.code == "V2"], "V5-nopc": t[(t.code == "V2") & ~pc],
           "V6": t[t.code == "V6"], "V7": t[(t.code == "V7") & pc], "V8": t[(t.code == "V2") & pc & t.vol_hi]}
    names = {"V1": "V1 논문 재현: SPY (판정 안 함)", "V2": "V2 기본: SPXL, 발표 -5분 ~ +55분", "V3": "V3 TQQQ",
             "V4": "V4 15:59 종가까지", "V5-all": "V5 기자회견 없는 정례 회의 포함 (전부)",
             "V5-nopc": "V5 기자회견 없는 회의만 (2011~2018)", "V6": "V6 기자회견 전 30분 오르면 기자회견 뒤 매수→종가",
             "V7": "V7 발표 전 상승: 9:31 → 발표 5분 전", "V8": "V8 불확실성 큰 날만 (20일 변동성 ≥ 1년 중앙값)"}

    lines = [f"# b07 백테스트 결과 (FOMC 정례 회의 {len(fomc)}번, {fomc.d.min()} ~ {fomc.d.max()})", "",
             "비용: 한쪽 수수료 0.25% + 미끄러짐 0.1% (왕복 0.7%). 미끄러짐 0.02%(왕복 0.54%)도 함께. t값 2 이상이면 대략 '우연은 아님'.", ""]
    if os.path.exists(a.fed_check):
        lines += ["## 연준 일정표 대조", "", "```", open(a.fed_check).read().strip(), "```", ""]
    bad = chk[~chk.ok]
    lines += ["## 발표 시각 확인 (표의 발표 시각 1~3분 SPY 거래량이 그 전 1시간 평소의 3배 이상 튀었나)", "",
              f"{int(chk.ok.sum())} / {len(chk)}번 3배 이상 (튄 정도 중앙값 {chk.ratio.median():.1f}배). "
              f"12:30·14:00·14:15 중 가장 크게 튄 시각이 표의 시각인 날 {int(chk.top.sum())}번. 3배 미만인 날: "
              + (", ".join(f"{r.d} (표 {r.announce} {r.ratio:.1f}배, 가장 튄 시각 {r.best})" for r in bad.itertuples()) or "없음"), "",
              f"## 1차 관문: 비용 전 거래당 평균 ≥ +{GATE * 100:.1f}% 인가", "",
              "| 변형 | 회의 수 | 비용 전 평균 | 비용 전 중앙값 | 비용 전 승률 | t값 | 비용 후 평균 | 비용 후 (미끄러짐 0.02%) | 비용 후 승률 | 비용 후 t값 | 관문 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    passed = []
    for k, s in sel.items():
        row, ok = stats_row(names[k], s, judge=k != "V1")
        lines.append(row)
        if ok:
            passed.append(k)
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else '없음 → b07 탈락'}**"]

    for lab, y0, y1 in PERIODS:
        lines += ["", f"## {lab}", "", "| 변형 | 회의 수 | 비용 전 평균 | 비용 전 중앙값 | 비용 전 승률 | t값 | 비용 후 평균 | 비용 후 (미끄러짐 0.02%) | 비용 후 승률 | 비용 후 t값 | 관문 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, s in sel.items():
            lines.append(stats_row(names[k], s[(s.year >= y0) & (s.year <= y1)], judge=False)[0])

    years = sorted(t.year.unique())
    keys = list(sel)
    lines += ["", "## 해마다: 비용 전 거래당 평균 (회의 수)", "", "| 해 | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for y in years:
        lines.append(f"| {y} | " + " | ".join(f"{sel[k][sel[k].year == y].gross.mean() * 100:.2f}% ({(sel[k].year == y).sum()})"
                                              if (sel[k].year == y).any() else "" for k in keys) + " |")

    cal = spy.index[spy.index >= pd.Timestamp("2011-04-01")]
    s0 = portfolio.stats(spy[cal].pct_change().fillna(0.0))
    lines += ["", f"## 계좌 성과 (회의 날만 계좌 {WEIGHT:.0%} 투자, 비용 왕복 0.7%, 2011-04 ~)", "",
              "| 변형 | 연수익률 | 최대낙폭 |", "|---|---|---|", f"| SPY 보유 | {s0['cagr'] * 100:.1f}% | {s0['mdd'] * 100:.1f}% |"]
    for k, s in sel.items():
        if s.empty:
            continue
        r = (WEIGHT * portfolio.net_return(s.gross, FEE + SLIP)).groupby(pd.DatetimeIndex(s.d)).sum().reindex(cal, fill_value=0.0)
        st = portfolio.stats(r)
        lines.append(f"| {names[k]} | {st['cagr'] * 100:.1f}% | {st['mdd'] * 100:.1f}% |")
    v2 = sel["V2"]
    if not v2.empty:
        lines.append(f"\nV2 사용 상품: {', '.join(f'{k} {n}' for k, n in v2.ticker.value_counts().items())}")
        lines.append("V2 가장 좋은/나쁜 회의: " + ", ".join(f"{r.d} {r.gross * 100:+.1f}%" for r in v2.nlargest(3, 'gross').itertuples())
                     + " / " + ", ".join(f"{r.d} {r.gross * 100:+.1f}%" for r in v2.nsmallest(3, 'gross').itertuples()))
    text = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
