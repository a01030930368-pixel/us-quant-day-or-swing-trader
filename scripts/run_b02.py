"""b02 데이트레이딩 백테스트 실행: prep_b02 폴더 -> 결과표(마크다운).

    python scripts/run_b02.py --prep prep_b02 --out reports/b02
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daytrade import b02, portfolio  # noqa: E402

COST = 0.0025 + 0.001   # 한쪽 수수료 0.25% + 미끄러짐 0.1%
PAPER_COST = 0.0002     # 논문 수준: 왕복 약 0.04%
GATE = 0.009            # 1차 관문: 비용 전 거래당 평균 >= +0.9%
MIN_N = 200             # 신호가 이보다 적으면 우연일 수 있어 관문 판정에서 뺌
MAX_W, MAX_POS = 0.20, 5

# (코드, 이름, 시간창, 손절, 상위 몇 개, 상대거래량 하한, 갭 하한, 비용)
VARIANTS = [("O1", "O1 논문 (5분, 손절 ATR 10%)", 5, "atr", 20, 1, None, COST),
            ("O2", "O2 5분, 손절 범위 저점", 5, "range", 20, 1, None, COST),
            ("O3", "O3 15분, 손절 범위 저점", 15, "range", 20, 1, None, COST),
            ("O4", "O4 30분, 손절 범위 저점", 30, "range", 20, 1, None, COST),
            ("O5", "O5 60분, 손절 범위 저점", 60, "range", 20, 1, None, COST)]
for w in (5, 15, 30, 60):
    VARIANTS.append((f"O6-{w}", f"O6 {w}분, 상위 5개 + 상대거래량 5배 이상", w, "range", 5, 5, None, COST))
for w in (5, 15, 30, 60):
    VARIANTS.append((f"O7-{w}", f"O7 {w}분, 갭 +3% 이상인 날만", w, "range", 20, 1, 0.03, COST))
VARIANTS.append(("O8", "O8 논문 비용(왕복 0.04%)으로 O1 (참고용)", 5, "atr", 20, 1, None, PAPER_COST))


def load(prep, kind):
    return pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(os.path.join(prep, f"{kind}_*.parquet")))])


def base_trades(prep, feats):
    """뽑힌 종목-날짜마다 (시간창, 손절 방식)별 거래 하나."""
    f = feats.set_index(["ticker", "d"])
    rows = []
    for fn in sorted(glob.glob(os.path.join(prep, "bars_*.parquet"))):
        df = pq.read_table(fn).to_pandas()
        for (tk, d), g in df.groupby(["ticker", "d"], sort=False):
            if (tk, d) not in f.index:
                continue
            x = f.loc[(tk, d)]
            day = {k: g[k].to_numpy() for k in ("mi", "o", "h", "l", "c")}
            for w in (5, 15, 30, 60):
                if not (x[f"rv{w}"] >= 1 and x[f"rk{w}"] <= 25):
                    continue
                for mode in (("atr", "range") if w == 5 else ("range",)):
                    t = b02.trade(day, w, mode, atr=x.atr)
                    if t:
                        rows.append(dict(ticker=tk, d=d, w=w, mode=mode, **t))
        print(f"{os.path.basename(fn)}: base trades so far {len(rows)}", flush=True)
    t = pd.DataFrame(rows).merge(feats, on=["ticker", "d"], how="left")
    t["gap"] = t.o / t.pc - 1
    return t


def pick(t, w, mode, top, min_rv, min_gap):
    s = t[(t.w == w) & (t["mode"] == mode) & (t[f"rk{w}"] <= top) & (t[f"rv{w}"] >= min_rv)]
    if min_gap is not None:
        s = s[s.gap >= min_gap]
    return s.assign(rank=s[f"rk{w}"])


def pct(x):
    return "" if pd.isna(x) else f"{x * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_b02")
    ap.add_argument("--out", default="reports/b02")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spy = load(a.prep, "spy").set_index("d").c.sort_index()
    days = spy.index
    spy_ret = spy.pct_change().fillna(0.0)
    feats = load(a.prep, "feats")
    t = base_trades(a.prep, feats)
    t.to_csv(os.path.join(a.out, "trades.csv.gz"), index=False)
    years = pd.DatetimeIndex(t.d).year
    t["year"] = years

    lines = [f"# b02 백테스트 결과 ({days[0]} ~ {days[-1]}, {len(days)}거래일)", "",
             f"비용: 한쪽 {COST * 100:.2f}% (수수료 0.25% + 미끄러짐 0.1%), O8만 한쪽 {PAPER_COST * 100:.2f}%. "
             f"SPY는 배당 제외 가격 수익률.", "",
             f"## 1차 관문: 비용 전 거래당 평균 ≥ +{GATE * 100:.1f}% 인가 (모든 신호, 자금관리 없이)", "",
             "| 변형 | 신호 수 | 하루 평균 | 비용 전 평균 | 비용 전 중앙값 | 비용 후 평균 | 승률(비용 후) | 평균 손절폭 | 손절/마감 | 관문 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    passed = []
    sel = {}
    for code, name, w, mode, top, rv, gap, cost in VARIANTS:
        s = pick(t, w, mode, top, rv, gap)
        sel[code] = (s, cost)
        if s.empty:
            lines.append(f"| {name} | 0 | | | | | | | | |")
            continue
        net = portfolio.net_return(s.gross, cost)
        ok = s.gross.mean() >= GATE and len(s) >= MIN_N
        if ok and code != "O8":
            passed.append(code)
        rc = s.reason.value_counts()
        lines.append(f"| {name} | {len(s)} | {len(s) / len(days):.1f} | {s.gross.mean() * 100:.2f}% | "
                     f"{s.gross.median() * 100:.2f}% | {net.mean() * 100:.2f}% | {pct((net > 0).mean())} | "
                     f"{s.stop_dist.mean() * 100:.2f}% | {rc.get('손절', 0)}/{rc.get('마감', 0)} | "
                     f"{'통과' if ok else ('탈락' if len(s) >= MIN_N else '신호 부족')} |")
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else '없음 → b02 탈락'}**"]

    s = portfolio.stats(spy_ret)
    lines += ["", f"## 계좌 성과 (손절 시 계좌 0.5% 위험, 한 종목 최대 {MAX_W:.0%}, 동시 {MAX_POS}종목, 하루 -1.5% 중단)", "",
              "| 변형 | 연수익률 | 최대낙폭 | 샤프 | 실제 거래 수 |", "|---|---|---|---|---|",
              f"| SPY 보유 | {pct(s['cagr'])} | {pct(s['mdd'])} | {s['sharpe']:.2f} | |"]
    ytab = {"SPY": portfolio.yearly(spy_ret)}
    for code, name, *_ in VARIANTS:
        sv, cost = sel[code]
        if sv.empty:
            continue
        ret, k = portfolio.daily_returns(sv, days, cost, max_w=MAX_W, max_pos=MAX_POS)
        st = portfolio.stats(ret)
        lines.append(f"| {name} | {pct(st['cagr'])} | {pct(st['mdd'])} | {st['sharpe']:.2f} | {len(k)} |")
        if code in ("O1", "O2", "O3", "O4", "O5", "O8"):
            ytab[code] = portfolio.yearly(ret)

    lines += ["", "## 해마다: 비용 전 거래당 평균 (모든 신호)", "",
              "| 해 | " + " | ".join(c for c, *_ in VARIANTS) + " |", "|---" * (len(VARIANTS) + 1) + "|"]
    for y in sorted(t.year.unique()):
        cells = []
        for code, *_ in VARIANTS:
            sv = sel[code][0]
            g = sv[pd.DatetimeIndex(sv.d).year == y].gross
            cells.append(f"{g.mean() * 100:.2f}%" if len(g) else "")
        lines.append(f"| {y} | " + " | ".join(cells) + " |")
    yt = pd.DataFrame(ytab)
    lines += ["", "## 해마다: 계좌 수익률", "", "| 해 | " + " | ".join(yt.columns) + " |",
              "|---" * (len(yt.columns) + 1) + "|"]
    for y, r in yt.iterrows():
        lines.append(f"| {y} | " + " | ".join(pct(x) for x in r) + " |")

    o4 = sel["O4"][0]
    if not o4.empty:
        lines += ["", "## O4(30분) 신호를 나눠 보면 (비용 전 거래당 평균)", "", "| 구분 | 신호 수 | 비용 전 평균 |", "|---|---|---|"]
        for label, key in [("상대거래량", pd.cut(o4.rv30, [1, 2, 5, 10, np.inf], labels=["1~2배", "2~5배", "5~10배", "10배 이상"])),
                           ("갭", pd.cut(o4.gap, [-np.inf, 0, 0.03, 0.1, np.inf], labels=["갭 하락", "0~3%", "3~10%", "10% 이상"])),
                           ("범위 폭", pd.cut(o4.range_pct, [0, 0.02, 0.05, 0.1, np.inf], labels=["2% 미만", "2~5%", "5~10%", "10% 이상"])),
                           ("청산 이유", o4.reason)]:
            for k, g in o4.groupby(key, observed=True):
                lines.append(f"| {label}: {k} | {len(g)} | {g.gross.mean() * 100:.2f}% |")
        lines.append(f"\n신호가 많은 종목: {', '.join(f'{k} {v}' for k, v in o4.ticker.value_counts().head(15).items())}")
    text = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
