"""b09 데이트레이딩 백테스트 실행 (장중 급락 반등): prep_b09 폴더 -> 결과표(마크다운).

    python scripts/run_b09.py --prep prep_b09 --out reports/b09
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daytrade import b09, portfolio  # noqa: E402
from run_b03 import earnings_days  # noqa: E402

FEE, SLIP, SLIP2 = 0.0025, 0.001, 0.002
GATE, MIN_N = 0.009, 200
MAX_W, MAX_POS, TOP_DAY = 0.20, 5, 5
RECENT = 2025

# 기본 거래 설정: 키 -> 함수
CFG = {"c1": lambda day, pc: b09.confirmed(day, pc),
       "c2": lambda day, pc: b09.at_touch(day, pc),
       "c3": lambda day, pc: b09.confirmed(day, pc, how="vwap"),
       "c15": lambda day, pc: b09.confirmed(day, pc, thr=0.15),
       "c20": lambda day, pc: b09.confirmed(day, pc, thr=0.20),
       "c8": lambda day, pc: b09.confirmed(day, pc, until=120)}
# 코드, 이름, 기본 설정, 추가 조건
VARIANTS = [("C1", "C1 기본 (-10%, 5분봉 직전 고가 돌파, 저가 손절, 마감 매도)", "c1", None),
            ("C2", "C2 확인 없이 -10% 닿는 순간 매수 → 마감 (손절 없음)", "c2", None),
            ("C3", "C3 확인을 VWAP 회복으로", "c3", None),
            ("C4-15", "C4 급락 기준 -15%", "c15", None),
            ("C4-20", "C4 급락 기준 -20%", "c20", None),
            ("C5", "C5 매수 때까지 거래량 ≥ 평소 하루의 2배", "c1", "vol2"),
            ("C6-e", "C6 실적 발표일만", "c1", "earn"),
            ("C6-x", "C6 실적일 제외", "c1", "noearn"),
            ("C7", "C7 직전 3일 연속 하락한 종목만", "c1", "down3"),
            ("C8", "C8 매수 120분 뒤 청산", "c8", None),
            ("C9-gap", "C9 갭으로 -10% 아래서 시작한 날만", "c1", "gap"),
            ("C9-intra", "C9 장중에 -10%까지 빠진 날만", "c1", "intra")]


def load(prep, kind):
    return pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(os.path.join(prep, f"{kind}_*.parquet")))])


def base_trades(prep, feats):
    f = feats.set_index(["ticker", "d"])
    rows = []
    for fn in sorted(glob.glob(os.path.join(prep, "bars_*.parquet"))):
        df = pq.read_table(fn).to_pandas()
        df = df[df.set_index(["ticker", "d"]).index.isin(f.index)]
        for (tk, d), g in df.groupby(["ticker", "d"], sort=False):
            pc = f.loc[(tk, d)].pc
            day = {k: g[k].to_numpy() for k in ("mi", "o", "h", "l", "c", "v")}
            for key, fn_ in CFG.items():
                t = fn_(day, pc)
                if t:
                    rows.append(dict(ticker=tk, d=d, cfg=key, **t))
        print(f"{os.path.basename(fn)}: base trades so far {len(rows)}", flush=True)
    return pd.DataFrame(rows).merge(feats, on=["ticker", "d"], how="left")


def pick(t, cfg, extra):
    s = t[t.cfg == cfg]
    if extra == "vol2":
        s = s[s.vol_to_entry >= 2 * s.adv_sh]
    elif extra == "earn":
        s = s[s.earn]
    elif extra == "noearn":
        s = s[~s.earn]
    elif extra == "down3":
        s = s[(s.pc < s.pc2) & (s.pc2 < s.pc3) & (s.pc3 < s.pc4)]
    elif extra == "gap":
        s = s[s.o <= s.pc * 0.9]
    elif extra == "intra":
        s = s[s.o > s.pc * 0.9]
    s = s.sort_values(["d", "drop_at_entry"], ascending=[True, False]).groupby("d").head(TOP_DAY)
    return s.assign(rank=-s.drop_at_entry, cost=FEE + SLIP, stop_dist=s.stop_dist.fillna(portfolio.RISK / MAX_W))


def pct(x, nd=1):
    return "" if pd.isna(x) else f"{x * 100:.{nd}f}%"


GATE_HEAD = ["| 변형 | 신호 수 | 비용 전 평균 | 비용 전 중앙값 | 승률(비용 전) | t값 | 2025~26 평균 (수) | 비용 후 평균 | "
             "비용 후 (미끄러짐 0.2%) | 승률(비용 후) | 평균 손절폭 | 청산 이유 | 관문 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def gate_row(name, s):
    if s.empty:
        return f"| {name} | 0 |" + " |" * 11, False
    g = s.gross
    tv = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan
    r = s[pd.DatetimeIndex(s.d).year >= RECENT].gross
    n1, n2 = portfolio.net_return(g, FEE + SLIP), portfolio.net_return(g, FEE + SLIP2)
    ok = g.mean() >= GATE and len(s) >= MIN_N
    rc = ", ".join(f"{k} {v}" for k, v in s.reason.value_counts().items())
    sd = s.stop_dist[s.reason.notna() & (s.cfg != "c2")].mean() if "cfg" in s else np.nan
    return (f"| {name} | {len(s)} | {pct(g.mean(), 2)} | {pct(g.median(), 2)} | {pct((g > 0).mean())} | {tv:.2f} | "
            f"{pct(r.mean(), 2)} ({len(r)}) | {pct(n1.mean(), 2)} | {pct(n2.mean(), 2)} | {pct((n1 > 0).mean())} | "
            f"{pct(sd, 2)} | {rc} | {'통과' if ok else ('탈락' if len(s) >= MIN_N else '신호 부족')} |"), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_b09")
    ap.add_argument("--out", default="reports/b09")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spy = load(a.prep, "spy").drop_duplicates("d").set_index("d").c.sort_index()
    days = spy.index
    spy_ret = spy.pct_change().fillna(0.0)
    earn = earnings_days(a.prep, days)
    if earn is None:
        raise SystemExit("SEC 실적 자료(earnings.parquet)가 없어 b09를 돌릴 수 없음 (회사 주식 구분과 C6에 필요)")
    companies = set(earn.ticker)
    feats = load(a.prep, "feats")
    n_all = len(feats)
    feats = feats[feats.ticker.isin(companies)]
    feats = feats.merge(earn.assign(earn=True), on=["ticker", "d"], how="left").assign(
        earn=lambda x: x.earn.fillna(False).astype(bool))
    t = base_trades(a.prep, feats)
    t.to_csv(os.path.join(a.out, "trades.csv.gz"), index=False)

    sel = {c: pick(t, cfg, extra) for c, _, cfg, extra in VARIANTS}
    lines = [f"# b09 백테스트 결과 ({days[0]} ~ {days[-1]}, {len(days)}거래일)", "",
             "비용: 한쪽 수수료 0.25% + 미끄러짐 0.1% (왕복 0.7%). 미끄러짐 0.2%(왕복 0.9%)도 함께. SPY는 배당 제외 가격 수익률. "
             "t값 2 이상이면 대략 '우연은 아님'.",
             f"후보(장중 저가가 전날 종가보다 10%↓, 전날 종가 5달러↑, 평소 거래대금 2,000만$↑): {n_all}건 중 회사 주식 "
             f"{len(feats)}건 (실적일 {int(feats.earn.sum())}건)", "",
             f"## 1차 관문: 비용 전 거래당 평균 ≥ +{GATE * 100:.1f}% 인가 (하루 하락 큰 순 {TOP_DAY}개, 자금관리 없이)", "",
             *GATE_HEAD]
    passed = []
    for code, name, *_ in VARIANTS:
        row, ok = gate_row(name, sel[code])
        lines.append(row)
        if ok:
            passed.append(code)
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else '없음 → b09 탈락'}**"]

    bins = [0.10, 0.15, 0.25, np.inf]
    labels = ["10~15%", "15~25%", "25% 이상"]
    lines += ["", "## 매수 때까지 하락 폭(전날 종가 → 그날 저가)별 비용 전 거래당 평균 (신호 수)", "",
              "| 변형 | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for code, *_ in VARIANTS:
        s = sel[code]
        if s.empty:
            continue
        g = s.groupby(pd.cut(s.drop_at_entry, bins, labels=labels, right=False), observed=False).gross
        lines.append(f"| {code} | " + " | ".join(f"{m * 100:.2f}% ({n})" if n else "" for m, n in zip(g.mean(), g.size())) + " |")

    s0 = portfolio.stats(spy_ret)
    lines += ["", f"## 계좌 성과 (손절 시 계좌 0.5% 위험, 한 종목 최대 {MAX_W:.0%}, 동시 {MAX_POS}종목, 하루 -1.5% 중단, 비용 왕복 0.7%)", "",
              "| 변형 | 연수익률 | 최대낙폭 | 샤프 | 실제 거래 수 |", "|---|---|---|---|---|",
              f"| SPY 보유 | {pct(s0['cagr'])} | {pct(s0['mdd'])} | {s0['sharpe']:.2f} | |"]
    ytab = {"SPY": portfolio.yearly(spy_ret)}
    for code, name, *_ in VARIANTS:
        s = sel[code]
        if s.empty:
            continue
        ret, k = portfolio.daily_returns(s, days, s.cost, max_w=MAX_W, max_pos=MAX_POS)
        st = portfolio.stats(ret)
        lines.append(f"| {name} | {pct(st['cagr'])} | {pct(st['mdd'])} | {st['sharpe']:.2f} | {len(k)} |")
        if code in ("C1", "C2", "C3"):
            ytab[code] = portfolio.yearly(ret)

    codes = [c for c, *_ in VARIANTS if not sel[c].empty]
    lines += ["", "## 해마다: 비용 전 거래당 평균 (신호 수)", "", "| 해 | " + " | ".join(codes) + " |",
              "|---" * (len(codes) + 1) + "|"]
    for y in sorted(pd.DatetimeIndex(t.d).year.unique()):
        cells = []
        for c in codes:
            g = sel[c][pd.DatetimeIndex(sel[c].d).year == y].gross
            cells.append(f"{g.mean() * 100:.2f}% ({len(g)})" if len(g) else "")
        lines.append(f"| {y} | " + " | ".join(cells) + " |")
    yt = pd.DataFrame(ytab)
    lines += ["", "## 해마다: 계좌 수익률", "", "| 해 | " + " | ".join(yt.columns) + " |", "|---" * (len(yt.columns) + 1) + "|"]
    for y, r in yt.iterrows():
        lines.append(f"| {y} | " + " | ".join(pct(x) for x in r) + " |")

    c1 = sel["C1"]
    if not c1.empty:
        lines.append(f"\nC1 신호가 많은 종목: {', '.join(f'{k} {v}' for k, v in c1.ticker.value_counts().head(10).items())}")
        lines.append(f"C1 매수 시각 중앙값: 9:30 + {int(c1.entry_mi.median())}분, 실적일 비율 {c1.earn.mean():.0%}")
        top = c1.gross.sort_values(ascending=False)
        lines.append(f"C1 가장 좋은 거래 10개를 빼면 비용 전 평균: {pct(top.iloc[10:].mean(), 2)}")
    text = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
