"""b03 데이트레이딩 백테스트 실행: prep_b03 폴더 -> 결과표(마크다운).

    python scripts/run_b03.py --prep prep_b03 --out reports/b03
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daytrade import b03, portfolio  # noqa: E402

FEE, SLIP, SLIP_SMALL = 0.0025, 0.001, 0.003
GATE, MIN_N = 0.009, 200
MAX_W, MAX_POS, TOP_DAY = 0.20, 5, 5

# 코드, 이름, 최소 주가, 최소 평소 거래대금, 최소 갭, 장전 기준(비율 / None=실적일), 매수 방식
VARIANTS = [("G1", "G1 기본 (갭 4%, 15분 고가 돌파)", 5, 5e6, 0.04, 0.10, "r15"),
            ("G2", "G2 갭 8% 이상", 5, 5e6, 0.08, 0.10, "r15"),
            ("G3", "G3 첫 5분 범위", 5, 5e6, 0.04, 0.10, "r5"),
            ("G4", "G4 장전 고가 돌파", 5, 5e6, 0.04, 0.10, "pm"),
            ("G5", "G5 소형주 (2달러↑, 평소 100만$↑)", 2, 1e6, 0.04, 0.10, "r15"),
            ("G6", "G6 실적 발표일만 (SEC 8-K 2.02)", 5, 5e6, 0.04, None, "r15"),
            ("G7", "G7 G1 + 2R에서 절반 익절", 5, 5e6, 0.04, 0.10, "r15h"),
            ("G8", "G8 9:31 시가 바로 매수, 손절 장전 저가", 5, 5e6, 0.04, 0.10, "open"),
            # 규칙서에 없는 점검용: 레버리지 ETF처럼 뉴스와 무관하게 갭이 뜨는 상품을 빼고, 실적 공시를 낸 적 있는 회사 주식만
            ("G1c", "G1c G1을 회사 주식만 (실적 공시 이력 있는 종목)", 5, 5e6, 0.04, 0.10, "r15"),
            ("G2c", "G2c G2를 회사 주식만", 5, 5e6, 0.08, 0.10, "r15")]
ENTRIES = ("r15", "r5", "pm", "r15h", "open")


def load(prep, kind):
    return pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(os.path.join(prep, f"{kind}_*.parquet")))])


def earnings_days(prep, days):
    """실적 8-K 접수 시각 -> 그 실적에 반응하는 거래일 (전날 16:00 이후 ~ 당일 9:30 전)."""
    f = os.path.join(prep, "earnings.parquet")
    if not os.path.exists(f):
        return None
    e = pd.read_parquet(f)
    cal = pd.DatetimeIndex(pd.to_datetime(days))
    day = e.accepted.dt.normalize()
    mins = (e.accepted.dt.hour * 60 + e.accepted.dt.minute).to_numpy()
    nxt = cal.searchsorted(day, side="right")  # day 다음 거래일
    geq = cal.searchsorted(day, side="left")   # day 당일(거래일이면) 또는 다음 거래일
    trading = np.isin(day.to_numpy(), cal.to_numpy())
    pos = np.where(mins >= 960, nxt, np.where(mins < 570, geq, np.where(trading, -1, geq)))  # 장중 접수는 뺌
    ok = (pos >= 0) & (pos < len(cal))
    out = pd.DataFrame(dict(ticker=e.ticker.to_numpy()[ok], d=cal[pos[ok]].date))
    return out.drop_duplicates()


def base_trades(prep, feats):
    f = feats.set_index(["ticker", "d"])
    rows = []
    for fn in sorted(glob.glob(os.path.join(prep, "bars_*.parquet"))):
        df = pq.read_table(fn).to_pandas()
        for (tk, d), g in df.groupby(["ticker", "d"], sort=False):
            if (tk, d) not in f.index:
                continue
            x = f.loc[(tk, d)]
            day = {k: g[k].to_numpy() for k in ("mi", "o", "h", "l", "c")}
            res = {"r15": b03.range_breakout(day, 15, x.pc), "r5": b03.range_breakout(day, 5, x.pc),
                   "pm": b03.premarket_breakout(day, x.pc, x.pm_high, x.pm_low),
                   "r15h": b03.range_breakout(day, 15, x.pc, half_r=2.0), "open": b03.open_buy(day, x.pm_low)}
            for k, t in res.items():
                if t:
                    rows.append(dict(ticker=tk, d=d, entry=k, **t))
        print(f"{os.path.basename(fn)}: base trades so far {len(rows)}", flush=True)
    return pd.DataFrame(rows).merge(feats, on=["ticker", "d"], how="left")


def pick(t, earn, price, adv, gap, pm, entry, companies=None):
    s = t[(t.entry == entry) & (t.o >= price) & (t.adv >= adv) & (t.gap >= gap)]
    if companies is not None:
        s = s[s.ticker.isin(companies)]
    if pm is None:
        if earn is None:
            return s.iloc[0:0]
        s = s.merge(earn, on=["ticker", "d"])
    else:
        s = s[s.pm_dv >= np.maximum(2e6, pm * s.adv)]
    s = s.assign(score=s.gap * s.pm_rel)
    s = s.sort_values(["d", "score"], ascending=[True, False]).groupby("d").head(TOP_DAY)
    return s.assign(rank=-s.score, cost=FEE + np.where(s.o < 5, SLIP_SMALL, SLIP))


def pct(x):
    return "" if pd.isna(x) else f"{x * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_b03")
    ap.add_argument("--out", default="reports/b03")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spy = load(a.prep, "spy").set_index("d").c.sort_index()
    days = spy.index
    spy_ret = spy.pct_change().fillna(0.0)
    feats = load(a.prep, "feats")
    earn = earnings_days(a.prep, days)
    t = base_trades(a.prep, feats)
    t.to_csv(os.path.join(a.out, "trades.csv.gz"), index=False)

    lines = [f"# b03 백테스트 결과 ({days[0]} ~ {days[-1]}, {len(days)}거래일)", "",
             "비용: 한쪽 수수료 0.25% + 미끄러짐 0.1% (5달러 미만은 0.3%). SPY는 배당 제외 가격 수익률.",
             f"후보(갭 4%↑ + 장전 거래 5%↑, 가장 느슨한 기준): {len(feats)}건. "
             f"실적 8-K 자료: {'없음 (G6 못 함)' if earn is None else f'{len(earn)}건'}", "",
             f"## 1차 관문: 비용 전 거래당 평균 ≥ +{GATE * 100:.1f}% 인가 (하루 상위 {TOP_DAY}개, 자금관리 없이)", "",
             "| 변형 | 신호 수 | 하루 평균 | 비용 전 평균 | 비용 전 중앙값 | 비용 후 평균 | 승률(비용 후) | 평균 손절폭 | 청산 이유 | 관문 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    sel, passed = {}, []
    companies = set(earn.ticker) if earn is not None else set()
    for code, name, price, adv, gap, pm, entry in VARIANTS:
        s = pick(t, earn, price, adv, gap, pm, entry, companies if code.endswith("c") else None)
        sel[code] = s
        if s.empty:
            lines.append(f"| {name} | 0 | | | | | | | | |")
            continue
        net = portfolio.net_return(s.gross, s.cost)
        ok = s.gross.mean() >= GATE and len(s) >= MIN_N
        if ok:
            passed.append(code)
        rc = ", ".join(f"{k} {v}" for k, v in s.reason.value_counts().items())
        lines.append(f"| {name} | {len(s)} | {len(s) / len(days):.1f} | {s.gross.mean() * 100:.2f}% | "
                     f"{s.gross.median() * 100:.2f}% | {net.mean() * 100:.2f}% | {pct((net > 0).mean())} | "
                     f"{s.stop_dist.mean() * 100:.2f}% | {rc} | "
                     f"{'통과' if ok else ('탈락' if len(s) >= MIN_N else '신호 부족')} |")
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else '없음 → b03 탈락'}**"]

    bins = [0.04, 0.08, 0.15, np.inf]
    labels = ["4~8%", "8~15%", "15% 이상"]
    lines += ["", "## 갭 크기별 비용 전 거래당 평균 (신호 수)", "", "| 변형 | " + " | ".join(labels) + " |",
              "|---" * (len(labels) + 1) + "|"]
    for code, name, *_ in VARIANTS:
        s = sel[code]
        if s.empty:
            continue
        g = s.groupby(pd.cut(s.gap, bins, labels=labels, right=False), observed=False).gross
        lines.append(f"| {code} | " + " | ".join(f"{m * 100:.2f}% ({n})" if n else "" for m, n in zip(g.mean(), g.size()))
                     + " |")

    s0 = portfolio.stats(spy_ret)
    lines += ["", f"## 계좌 성과 (손절 시 계좌 0.5% 위험, 한 종목 최대 {MAX_W:.0%}, 동시 {MAX_POS}종목, 하루 -1.5% 중단)", "",
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
    lines += ["", "## 해마다: 계좌 수익률", "", "| 해 | " + " | ".join(yt.columns) + " |",
              "|---" * (len(yt.columns) + 1) + "|"]
    for y, r in yt.iterrows():
        lines.append(f"| {y} | " + " | ".join(pct(x) for x in r) + " |")
    g1 = sel["G1"]
    if not g1.empty:
        lines.append(f"\nG1 신호가 많은 종목: {', '.join(f'{k} {v}' for k, v in g1.ticker.value_counts().head(15).items())}")
        if not sel["G1c"].empty:
            lines.append(f"G1c 신호가 많은 종목: {', '.join(f'{k} {v}' for k, v in sel['G1c'].ticker.value_counts().head(15).items())}")
        lines.append(f"G1 매수 시각 중앙값: 9:30 + {int(g1.entry_mi.median())}분")
    text = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
