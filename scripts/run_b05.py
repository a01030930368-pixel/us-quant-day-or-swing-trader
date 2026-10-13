"""b05 데이트레이딩 백테스트 실행 (래리 윌리엄스 웁스 갭 하락 반등): prep_b05 폴더 -> 결과표(마크다운).

    python scripts/run_b05.py --prep prep_b05 --out reports/b05
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daytrade import b05, portfolio  # noqa: E402
from run_b03 import earnings_days  # noqa: E402

FEE, SLIP, SLIP2 = 0.0025, 0.001, 0.002
GATE, MIN_N = 0.009, 200
MAX_W, MAX_POS, TOP_DAY = 0.20, 5, 5
RECENT = 2025
INDEX_ETF = {"SPY", "QQQ", "IWM", "DIA"}

# 매수 설정: 키 -> (매수 마감 분, 손절 사용)
CFG = {"base": (330, True), "nostop": (330, False), "dl90": (90, True), "dl210": (210, True)}
# 코드, 이름, 최소 갭, 대상(all = 회사 주식 + 지수 ETF, etf = 지수 ETF만), 실적일(None / earn / noearn), 매수 설정, 보유
VARIANTS = [("F1", "F1 기본 (갭 1%, 15:00까지, 그날 저가 손절)", 0.01, "all", None, "base", "day"),
            ("F2-3", "F2 갭 3% 이상", 0.03, "all", None, "base", "day"),
            ("F2-6", "F2 갭 6% 이상", 0.06, "all", None, "base", "day"),
            ("F3", "F3 손절 없이 마감까지", 0.01, "all", None, "nostop", "day"),
            ("F4", "F4 매수 마감 11:00", 0.01, "all", None, "dl90", "day"),
            ("F5-e", "F5 실적 발표일 갭 하락만", 0.01, "all", "earn", "base", "day"),
            ("F5-x", "F5 실적일 제외", 0.01, "all", "noearn", "base", "day"),
            ("F6", "F6 지수 ETF만 (SPY·QQQ·IWM·DIA)", 0.01, "etf", None, "base", "day"),
            ("F7", "F7 처음 이익 난 시가에 매도 (최대 10일, 비교용)", 0.01, "all", None, "base", "swing")]
SENS = [("S-dl13", "매수 마감 13:00", 0.01, "all", None, "dl210", "day")]


def load(prep, kind):
    return pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(os.path.join(prep, f"{kind}_*.parquet")))])


def base_trades(prep, feats, daily):
    f = feats.set_index(["ticker", "d"])
    hist = {tk: (g.d.to_numpy(), {k: g[k].to_numpy() for k in ("o", "l", "c")})
            for tk, g in daily[daily.ticker.isin(set(feats.ticker))].groupby("ticker", sort=False)}
    rows = []
    for fn in sorted(glob.glob(os.path.join(prep, "bars_*.parquet"))):
        df = pq.read_table(fn).to_pandas()
        df = df[df.set_index(["ticker", "d"]).index.isin(f.index)]
        for (tk, d), g in df.groupby(["ticker", "d"], sort=False):
            pl = f.loc[(tk, d)].pl
            day = {k: g[k].to_numpy() for k in ("mi", "o", "h", "l", "c")}
            for key, (dl, use_stop) in CFG.items():
                t = b05.intraday(day, pl, dl, use_stop)
                if t:
                    rows.append(dict(ticker=tk, d=d, cfg=key, hold="day", **t))
            e = b05.find_entry(day, pl)
            if e is not None:
                ds, a = hist[tk]
                k = np.searchsorted(ds, d)
                t = b05.swing(day, *e, {n: v[k + 1:k + 10] for n, v in a.items()})
                rows.append(dict(ticker=tk, d=d, cfg="base", hold="swing", exit_mi=385, **t))
        print(f"{os.path.basename(fn)}: base trades so far {len(rows)}", flush=True)
    return pd.DataFrame(rows).merge(feats, on=["ticker", "d"], how="left")


def pick(t, gap, uni, earn, cfg, hold):
    s = t[(t.cfg == cfg) & (t.hold == hold) & (t.gap >= gap)]
    if uni == "etf":
        s = s[s.etf]
    if earn == "earn":
        s = s[s.earn]
    elif earn == "noearn":
        s = s[~s.earn]
    s = s.sort_values(["d", "gap"], ascending=[True, False]).groupby("d").head(TOP_DAY)
    return s.assign(rank=-s.gap, cost=FEE + SLIP)


def pct(x, nd=1):
    return "" if pd.isna(x) else f"{x * 100:.{nd}f}%"


GATE_HEAD = ["| 변형 | 신호 수 | 하루 평균 | 비용 전 평균 | 비용 전 중앙값 | 2025~26 평균 (수) | 2025~26 중앙값 | "
             "비용 후 평균 | 비용 후 (미끄러짐 0.2%) | 승률(비용 후) | 평균 손절폭 | 청산 이유 | 관문 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def gate_row(name, s, days):
    if s.empty:
        return f"| {name} | 0 |" + " |" * 11, False
    r = s[pd.DatetimeIndex(s.d).year >= RECENT].gross
    n1, n2 = portfolio.net_return(s.gross, FEE + SLIP), portfolio.net_return(s.gross, FEE + SLIP2)
    ok = s.gross.mean() >= GATE and len(s) >= MIN_N
    rc = ", ".join(f"{k} {v}" for k, v in s.reason.value_counts().items())
    held = f" (평균 {s.days.mean():.1f}일)" if s.days.notna().any() else ""
    return (f"| {name} | {len(s)} | {len(s) / len(days):.2f} | {pct(s.gross.mean(), 2)} | {pct(s.gross.median(), 2)} | "
            f"{pct(r.mean(), 2)} ({len(r)}) | {pct(r.median(), 2)} | {pct(n1.mean(), 2)} | {pct(n2.mean(), 2)} | "
            f"{pct((n1 > 0).mean())} | {pct(s.stop_dist.mean(), 2)} | {rc}{held} | "
            f"{'통과' if ok else ('탈락' if len(s) >= MIN_N else '신호 부족')} |"), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_b05")
    ap.add_argument("--out", default="reports/b05")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spy = load(a.prep, "spy").drop_duplicates("d").set_index("d").c.sort_index()
    days = spy.index
    spy_ret = spy.pct_change().fillna(0.0)
    earn = earnings_days(a.prep, days)
    if earn is None:
        raise SystemExit("SEC 실적 자료(earnings.parquet)가 없어 b05를 돌릴 수 없음 (회사 주식 구분과 F5에 필요)")
    companies = set(earn.ticker)
    feats = load(a.prep, "feats")
    n_all = len(feats)
    feats = feats[feats.ticker.isin(companies | INDEX_ETF)].assign(etf=lambda x: x.ticker.isin(INDEX_ETF))
    feats = feats.merge(earn.assign(earn=True), on=["ticker", "d"], how="left").assign(
        earn=lambda x: x.earn.fillna(False).astype(bool))
    daily = load(a.prep, "daily").drop_duplicates(["ticker", "d"]).sort_values(["ticker", "d"]).reset_index(drop=True)
    t = base_trades(a.prep, feats, daily)
    if "days" not in t:
        t["days"] = np.nan
    t.to_csv(os.path.join(a.out, "trades.csv.gz"), index=False)

    sel = {c: pick(t, *v) for c, _, *v in VARIANTS + SENS}
    lines = [f"# b05 백테스트 결과 ({days[0]} ~ {days[-1]}, {len(days)}거래일)", "",
             "비용: 한쪽 수수료 0.25% + 미끄러짐 0.1% (미끄러짐 0.2%도 함께). SPY는 배당 제외 가격 수익률.",
             f"후보(시가가 전날 저가보다 1%↑ 아래, 5달러↑, 평소 거래대금 2,000만$↑): {n_all}건 중 회사 주식·지수 ETF "
             f"{len(feats)}건 (지수 ETF {int(feats.etf.sum())}건, 실적일 {int(feats.earn.sum())}건)", "",
             f"## 1차 관문: 비용 전 거래당 평균 ≥ +{GATE * 100:.1f}% 인가 (하루 갭 큰 순 {TOP_DAY}개, 자금관리 없이)", "",
             *GATE_HEAD]
    passed = []
    for code, name, *_ in VARIANTS:
        row, ok = gate_row(name, sel[code], days)
        lines.append(row)
        if ok and code != "F7":
            passed.append(code)
    lines += ["", f"**당일판 관문 통과 변형: {', '.join(passed) if passed else '없음 → b05 탈락'}**"]
    if passed:
        lines += ["", "## 민감도 (F1 기준. 최소 갭 0%는 자료 준비에서 1% 미만을 빼서 못 함)", "", *GATE_HEAD]
        lines += [gate_row(name, sel[code], days)[0] for code, name, *_ in SENS]

    f1 = sel["F1"]
    lines += ["", "## F1을 회사 주식 / 지수 ETF로 나눠 보기 (비용 전)", "", "| 대상 | 신호 수 | 평균 | 중앙값 |", "|---|---|---|---|"]
    for lab, g in (("회사 주식", f1[~f1.etf]), ("지수 ETF", f1[f1.etf])):
        lines.append(f"| {lab} | {len(g)} | {pct(g.gross.mean(), 2)} | {pct(g.gross.median(), 2)} |")
    nt = t[(t.cfg == "base") & (t.hold == "day") & t.etf]
    lines.append(f"| (참고) 지수 ETF 전체, 상위 5개 고르기 전 | {len(nt)} | {pct(nt.gross.mean(), 2)} | {pct(nt.gross.median(), 2)} |")

    bins = [0.01, 0.03, 0.06, np.inf]
    labels = ["1~3%", "3~6%", "6% 이상"]
    lines += ["", "## 갭 크기별 비용 전 거래당 평균 (신호 수)", "", "| 변형 | " + " | ".join(labels) + " |",
              "|---" * (len(labels) + 1) + "|"]
    for code, *_ in VARIANTS:
        s = sel[code]
        if s.empty:
            continue
        g = s.groupby(pd.cut(s.gap, bins, labels=labels, right=False), observed=False).gross
        lines.append(f"| {code} | " + " | ".join(f"{m * 100:.2f}% ({n})" if n else "" for m, n in zip(g.mean(), g.size()))
                     + " |")

    s0 = portfolio.stats(spy_ret)
    lines += ["", f"## 계좌 성과 (당일판만. 손절 시 계좌 0.5% 위험, 한 종목 최대 {MAX_W:.0%}, 동시 {MAX_POS}종목, 하루 -1.5% 중단)", "",
              "| 변형 | 연수익률 | 최대낙폭 | 샤프 | 실제 거래 수 |", "|---|---|---|---|---|",
              f"| SPY 보유 | {pct(s0['cagr'])} | {pct(s0['mdd'])} | {s0['sharpe']:.2f} | |"]
    ytab = {"SPY": portfolio.yearly(spy_ret)}
    for code, name, *_, hold in VARIANTS:
        s = sel[code]
        if s.empty or hold != "day":
            continue
        ret, k = portfolio.daily_returns(s, days, s.cost, max_w=MAX_W, max_pos=MAX_POS)
        st = portfolio.stats(ret)
        lines.append(f"| {name} | {pct(st['cagr'])} | {pct(st['mdd'])} | {st['sharpe']:.2f} | {len(k)} |")
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
    lines += ["", "## 해마다: 계좌 수익률 (당일판)", "", "| 해 | " + " | ".join(yt.columns) + " |",
              "|---" * (len(yt.columns) + 1) + "|"]
    for y, r in yt.iterrows():
        lines.append(f"| {y} | " + " | ".join(pct(x) for x in r) + " |")

    if not f1.empty:
        lines.append(f"\nF1 신호가 많은 종목: {', '.join(f'{k} {v}' for k, v in f1.ticker.value_counts().head(10).items())}")
        lines.append(f"F1 매수 시각 중앙값: 9:30 + {int(f1.entry_mi.median())}분, 실적일 비율 {f1.earn.mean():.0%}")
        top = f1.gross.sort_values(ascending=False)
        lines.append(f"F1 가장 좋은 거래 10개를 빼면 비용 전 평균: {pct(top.iloc[10:].mean(), 2)}")
    text = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
