"""b01 데이트레이딩 백테스트 실행: prep 폴더의 bars_*.parquet / daily_*.parquet -> 결과표(마크다운).

    python scripts/run_daytrade.py --prep prep --out reports/b01
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daytrade import b01, portfolio  # noqa: E402

COST = 0.0025 + 0.001  # 한쪽 수수료 0.25% + 미끄러짐 0.1%
COST_ALT = {"수수료 0.25%, 미끄러짐 0": 0.0025, "이벤트 수수료 0.07% + 미끄러짐 0.1%": 0.0017, "비용 0": 0.0}


def load_trades(prep):
    rows = []
    for f in sorted(glob.glob(os.path.join(prep, "bars_*.parquet"))):
        df = pq.read_table(f).to_pandas()
        n = 0
        for (tk, d), g in df.groupby(["ticker", "d"], sort=False):
            day = {c: g[c].to_numpy() for c in ("mi", "o", "h", "l", "c", "v")}
            for t in b01.run_day(day, g.adv.iat[0]):
                rows.append(dict(ticker=tk, d=d, adv=g.adv.iat[0], **t))
            n += 1
        print(f"{os.path.basename(f)}: {n} candidate days, trades so far {len(rows)}", flush=True)
    return pd.DataFrame(rows)


def load_daily(prep):
    return pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(os.path.join(prep, "daily_*.parquet")))])


def pct(x):
    return "" if pd.isna(x) else f"{x * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep")
    ap.add_argument("--out", default="reports/b01")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    daily = load_daily(a.prep)
    spy = daily[daily.ticker == "SPY"].set_index("d").day_close.sort_index()
    days = spy.index
    spy_ret = spy.pct_change().fillna(0.0)
    trades = load_trades(a.prep)
    if trades.empty:
        sys.exit("거래가 하나도 없음")
    trades["rank"] = -trades.base_rel
    trades.to_csv(os.path.join(a.out, "trades.csv.gz"), index=False)

    lines = [f"# b01 백테스트 결과 ({days[0]} ~ {days[-1]}, {len(days)}거래일)", "",
             f"비용: 한쪽 {COST * 100:.2f}% (수수료 0.25% + 미끄러짐 0.1%). SPY는 배당 제외 가격 수익률.", ""]
    s = portfolio.stats(spy_ret)
    lines += ["## 계좌 성과 (자금관리 적용)", "",
              "| 변형 | 연수익률 | 최대낙폭 | 샤프 | 거래 수 | 승률 | 거래당 평균(비용 후) | 거래당 평균(비용 전) | 손절/익절/마감 |",
              "|---|---|---|---|---|---|---|---|---|",
              f"| SPY 보유 | {pct(s['cagr'])} | {pct(s['mdd'])} | {s['sharpe']:.2f} | | | | | |"]
    year_tab = {"SPY": portfolio.yearly(spy_ret)}
    raw = []
    for vi, p in enumerate(b01.VARIANTS):
        t = trades[trades.variant == vi]
        if t.empty:
            lines.append(f"| {p.name} | 거래 없음 | | | 0 | | | | |")
            continue
        ret, k = portfolio.daily_returns(t, days, COST)
        s = portfolio.stats(ret)
        net = portfolio.net_return(k.gross, COST)
        rc = k.reason.value_counts()
        lines.append(f"| {p.name} | {pct(s['cagr'])} | {pct(s['mdd'])} | {s['sharpe']:.2f} | {len(k)} | "
                     f"{pct((net > 0).mean())} | {net.mean() * 100:.2f}% | {k.gross.mean() * 100:.2f}% | "
                     f"{rc.get('손절', 0)}/{rc.get('익절', 0)}/{rc.get('마감', 0)} |")
        if not p.code.startswith("S"):
            year_tab[p.code] = portfolio.yearly(ret)
        tn = portfolio.net_return(t.gross, COST)
        raw.append(f"| {p.name} | {len(t)} | {len(t) / len(days):.1f} | {pct((tn > 0).mean())} | "
                   f"{tn.mean() * 100:.2f}% | {t.gross.mean() * 100:.2f}% | {t.gross.median() * 100:.2f}% |")
    lines += ["", "## 모든 신호 그대로 (자금관리·동시보유 제한 없이, 거래 하나하나의 평균)", "",
              "| 변형 | 신호 수 | 하루 평균 | 승률 | 평균(비용 후) | 평균(비용 전) | 중앙값(비용 전) |",
              "|---|---|---|---|---|---|---|"] + raw
    yt = pd.DataFrame(year_tab)
    lines += ["", "## 해마다 수익률", "", "| 해 | " + " | ".join(yt.columns) + " |",
              "|---" * (len(yt.columns) + 1) + "|"]
    for y, r in yt.iterrows():
        lines.append(f"| {y} | " + " | ".join(pct(x) for x in r) + " |")

    t = trades[trades.variant == 0]
    lines += ["", "## D1 비용을 바꾸면", "", "| 비용 가정 | 연수익률 | 최대낙폭 | 거래당 평균 |", "|---|---|---|---|"]
    for label, c in COST_ALT.items():
        ret, k = portfolio.daily_returns(t, days, c)
        s = portfolio.stats(ret)
        lines.append(f"| {label} | {pct(s['cagr'])} | {pct(s['mdd'])} | {portfolio.net_return(k.gross, c).mean() * 100:.2f}% |")

    t = t.assign(net=portfolio.net_return(t.gross, COST))
    lines += ["", "## D1 신호를 나눠 보면 (모든 신호, 거래 하나하나의 평균)", "",
              "| 구분 | 신호 수 | 승률(비용 후) | 평균(비용 전) | 평균(비용 후) |", "|---|---|---|---|---|"]
    groups = [("20일 평균 거래대금", pd.cut(t.adv, [0, 1e8, 5e8, 2e9, np.inf],
                                       labels=["2천만~1억$", "1억~5억$", "5억~20억$", "20억$ 이상"])),
              ("기준봉 / 평소 하루 거래대금", pd.cut(t.base_rel, [0, 0.1, 0.2, 0.4, np.inf],
                                         labels=["5~10%", "10~20%", "20~40%", "40% 이상"])),
              ("매수가", pd.cut(t.entry_px, [0, 10, 30, 100, np.inf], labels=["5~10$", "10~30$", "30~100$", "100$ 이상"])),
              ("청산 이유", t.reason)]
    for label, key in groups:
        for k, g in t.groupby(key, observed=True):
            lines.append(f"| {label}: {k} | {len(g)} | {pct((g.net > 0).mean())} | {g.gross.mean() * 100:.2f}% | "
                         f"{g.net.mean() * 100:.2f}% |")

    t = t.assign(year=pd.DatetimeIndex(t.d).year, entry_hm=t.entry_mi.map(lambda m: f"{9 + (30 + m) // 60}:{(30 + m) % 60:02d}"))
    lines += ["", "## D1 신호 특징", "",
              f"- 매수 시각 중앙값 {t.entry_hm.iloc[int(np.argsort(t.entry_mi.to_numpy())[len(t) // 2])] if len(t) else ''}, "
              f"손절폭 평균 {t.stop_dist.mean() * 100:.2f}%",
              f"- 기준봉 거래대금 / 20일 평균 하루 거래대금 중앙값 {t.base_rel.median() * 100:.1f}%",
              f"- 신호가 많은 종목: {', '.join(f'{k} {v}' for k, v in t.ticker.value_counts().head(15).items())}"]
    text = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
