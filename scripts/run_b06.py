"""b06 데이트레이딩 백테스트 실행 (레버리지 ETF 장 막판 쏠림): prep_b06 폴더 -> 결과표(마크다운).

    python scripts/run_b06.py --prep prep_b06 --out reports/b06
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daytrade import b06, portfolio  # noqa: E402

FEE, SLIP, SLIP_LOW = 0.0025, 0.001, 0.0002
GATE, MIN_N = 0.009, 100
WEIGHT = 0.5  # 한 번에 계좌의 50%
LONG3 = ("SPXL", "BGU", "UPRO")  # 3배 S&P 매수: SPXL, 이름 바뀌기 전 BGU, 없으면 UPRO
PERIODS = [("2006~2011", 2006, 2011), ("2012~2019", 2012, 2019), ("2020~2026", 2020, 2026)]

# 코드, 이름, 기준, 기준 크기, 판단 시각(분), 상승 시 매수, 하락 시 매수(None이면 안 함), 매도 시각(None=15:59 종가), 손절, 시작일
VARIANTS = [("R1", "R1 논문 재현: 2배 SSO/SDS (판정 안 함)", "SPY", 0.02, 315, ("SSO",), ("SDS",), None, None, "2006-06-01"),
            ("R2", "R2 기본: 3배 SPXL/SPXU", "SPY", 0.02, 315, LONG3, ("SPXU",), None, None, "2009-01-01"),
            ("R3", "R3 오른 날만 (SPXL)", "SPY", 0.02, 315, LONG3, None, None, None, "2009-01-01"),
            ("R4-1.5", "R4 기준 ±1.5%", "SPY", 0.015, 315, LONG3, ("SPXU",), None, None, "2009-01-01"),
            ("R4-3", "R4 기준 ±3%", "SPY", 0.03, 315, LONG3, ("SPXU",), None, None, "2009-01-01"),
            ("R5-1430", "R5 판단 14:30", "SPY", 0.02, 300, LONG3, ("SPXU",), None, None, "2009-01-01"),
            ("R5-1500", "R5 판단 15:00", "SPY", 0.02, 330, LONG3, ("SPXU",), None, None, "2009-01-01"),
            ("R5-1530", "R5 판단 15:30", "SPY", 0.02, 360, LONG3, ("SPXU",), None, None, "2009-01-01"),
            ("R6", "R6 15:55 매도", "SPY", 0.02, 315, LONG3, ("SPXU",), 385, None, "2009-01-01"),
            ("R7", "R7 손절 -2%", "SPY", 0.02, 315, LONG3, ("SPXU",), None, 0.02, "2009-01-01"),
            ("R8", "R8 나스닥: QQQ → TQQQ/SQQQ", "QQQ", 0.02, 315, ("TQQQ",), ("SQQQ",), None, None, "2010-01-01"),
            ("R9", "R9 반도체: SOXX ±3% → SOXL/SOXS", "SOXX", 0.03, 315, ("SOXL",), ("SOXS",), None, None, "2010-01-01")]


def load_bars(prep):
    df = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(os.path.join(prep, "bars_*.parquet")))])
    df = df.drop_duplicates(["ticker", "d", "mi"]).sort_values(["ticker", "d", "mi"])
    return {k: {c: g[c].to_numpy() for c in ("mi", "o", "h", "l", "c")} for k, g in df.groupby(["ticker", "d"], sort=False)}


def run_variant(bars, idx_days, v):
    code, name, idx, thr, t, longs, shorts, exit_at, stop, start = v
    rows = []
    days, closes = idx_days[idx]
    start = pd.Timestamp(start).date()
    for n in range(1, len(days)):
        d = days[n]
        if d < start:
            continue
        move, sgn = b06.signal(bars[(idx, d)], closes[n - 1], thr, t)
        cands = longs if sgn > 0 else shorts if sgn < 0 else None
        if not cands:
            continue
        tk = next((x for x in cands if (x, d) in bars), None)
        if tk is None:
            rows.append(dict(d=d, side="상승" if sgn > 0 else "하락", move=move, ticker=None, gross=np.nan, reason="상품 없음"))
            continue
        r = b06.hold(bars[(tk, d)], t, exit_at, stop)
        if r:
            rows.append(dict(d=d, side="상승" if sgn > 0 else "하락", move=move, ticker=tk, **r))
    s = pd.DataFrame(rows, columns=["d", "side", "move", "ticker", "gross", "reason", "entry_mi", "entry_px", "exit_px"])
    return s, s[s.gross.notna()]


def pct(x, nd=2):
    return "" if pd.isna(x) else f"{x * 100:.{nd}f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_b06")
    ap.add_argument("--out", default="reports/b06")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bars = load_bars(a.prep)
    idx_days = {}
    for idx in ("SPY", "QQQ", "SOXX"):
        ds = sorted(d for (tk, d) in bars if tk == idx)
        idx_days[idx] = (ds, np.array([bars[(idx, d)]["c"][-1] for d in ds]))
    spy_days, spy_c = idx_days["SPY"]
    spy = pd.Series(spy_c, index=pd.DatetimeIndex(spy_days))
    tickers = sorted({tk for tk, _ in bars})
    first = {tk: min(d for (x, d) in bars if x == tk) for tk in tickers}

    res = {}
    for v in VARIANTS:
        allsig, s = run_variant(bars, idx_days, v)
        res[v[0]] = (v, allsig, s)
        s.assign(code=v[0]).to_csv(os.path.join(a.out, f"trades_{v[0]}.csv"), index=False)

    lines = [f"# b06 백테스트 결과 ({spy.index[0].date()} ~ {spy.index[-1].date()})", "",
             "비용: 한쪽 수수료 0.25% + 미끄러짐 0.1% (왕복 0.7%). 미끄러짐 0.02%(왕복 0.54%)도 함께. SPY는 배당 제외 가격 수익률.",
             "자료 첫날: " + ", ".join(f"{tk} {first[tk]}" for tk in tickers), "",
             f"## 1차 관문: 비용 전 거래당 평균 ≥ +{GATE * 100:.1f}% 인가 (거래 {MIN_N}번 이상, R1은 판정 안 함)", "",
             "| 변형 | 신호 | 거래 수 | 비용 전 평균 | 비용 전 중앙값 | 비용 후 평균 | 비용 후 (미끄러짐 0.02%) | 승률(비용 후) | "
             "오른 날 평균 (수) | 떨어진 날 평균 (수) | 손절 | 관문 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    passed = []
    for code, (v, allsig, s) in res.items():
        if s.empty:
            lines.append(f"| {v[1]} | {len(allsig)} | 0 |" + " |" * 9)
            continue
        n1 = portfolio.net_return(s.gross, FEE + SLIP)
        n2 = portfolio.net_return(s.gross, FEE + SLIP_LOW)
        up, dn = s[s.side == "상승"].gross, s[s.side == "하락"].gross
        ok = code != "R1" and s.gross.mean() >= GATE and len(s) >= MIN_N
        if ok:
            passed.append(code)
        verdict = "(판정 안 함)" if code == "R1" else ("통과" if ok else ("탈락" if len(s) >= MIN_N else "신호 부족"))
        lines.append(f"| {v[1]} | {len(allsig)} | {len(s)} | {pct(s.gross.mean())} | {pct(s.gross.median())} | "
                     f"{pct(n1.mean())} | {pct(n2.mean())} | {pct((n1 > 0).mean(), 0)} | {pct(up.mean())} ({len(up)}) | "
                     f"{pct(dn.mean())} ({len(dn)}) | {(s.reason == '손절').sum()} | {verdict} |")
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else '없음 → b06 탈락'}**"]

    lines += ["", "## 기간별 비용 전 거래당 평균 (거래 수)", "", "| 변형 | " + " | ".join(p for p, *_ in PERIODS) + " |",
              "|---" * (len(PERIODS) + 1) + "|"]
    for code, (v, _, s) in res.items():
        y = pd.DatetimeIndex(s.d).year
        cells = [f"{pct(s.gross[(y >= a0) & (y <= b0)].mean())} ({((y >= a0) & (y <= b0)).sum()})" if ((y >= a0) & (y <= b0)).any()
                 else "" for _, a0, b0 in PERIODS]
        lines.append(f"| {code} | " + " | ".join(cells) + " |")

    years = sorted(pd.DatetimeIndex(spy_days).year.unique())
    lines += ["", "## 해마다: 비용 전 거래당 평균 (거래 수)", "", "| 해 | " + " | ".join(res) + " |", "|---" * (len(res) + 1) + "|"]
    for yr in years:
        cells = []
        for code, (v, _, s) in res.items():
            g = s[pd.DatetimeIndex(s.d).year == yr].gross
            cells.append(f"{pct(g.mean())} ({len(g)})" if len(g) else "")
        lines.append(f"| {yr} | " + " | ".join(cells) + " |")

    lines += ["", f"## 계좌 성과 (한 번에 계좌 {WEIGHT:.0%} 투자, 비용 왕복 0.7%, 변형 시작일부터)", "",
              "| 변형 | 기간 | 연수익률 | 최대낙폭 | 같은 기간 SPY 연수익률 | SPY 최대낙폭 |", "|---|---|---|---|---|---|"]
    for code, (v, _, s) in res.items():
        cal = spy.index[spy.index >= pd.Timestamp(v[-1])]
        if s.empty or len(cal) < 2:
            continue
        r = (WEIGHT * portfolio.net_return(s.gross, FEE + SLIP)).groupby(pd.DatetimeIndex(s.d)).sum().reindex(cal, fill_value=0.0)
        st = portfolio.stats(r)
        s0 = portfolio.stats(spy[cal].pct_change().fillna(0.0))
        lines.append(f"| {v[1]} | {cal[0].date()} ~ | {pct(st['cagr'], 1)} | {pct(st['mdd'], 1)} | {pct(s0['cagr'], 1)} | {pct(s0['mdd'], 1)} |")

    v, allsig, s = res["R2"]
    if not s.empty:
        miss = allsig[allsig.gross.isna()]
        lines.append(f"\nR2 사용 상품: {', '.join(f'{k} {n}' for k, n in s.ticker.value_counts().items())}"
                     f"{'; 상품 자료 없어 못 한 신호 ' + str(len(miss)) + '번' if len(miss) else ''}")
        top = s.gross.sort_values(ascending=False)
        lines.append(f"R2 가장 좋은 거래 5개를 빼면 비용 전 평균: {pct(top.iloc[5:].mean())}; 가장 좋은 거래: "
                     + ", ".join(f"{r.d} {r.ticker} {r.gross * 100:+.1f}%" for r in s.nlargest(5, 'gross').itertuples()))
    text = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
