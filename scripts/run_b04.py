"""b04 데이트레이딩 백테스트 실행 (쿨라매기 에피소딕 피벗 당일판): prep_b04 폴더 -> 결과표(마크다운).

    python scripts/run_b04.py --prep prep_b04 --out reports/b04
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daytrade import b04, portfolio  # noqa: E402
from run_b03 import earnings_days  # noqa: E402

FEE, SLIP, SLIP2 = 0.0025, 0.001, 0.002
GATE, MIN_N = 0.009, 200
MAX_W, MAX_POS, TOP_DAY = 0.20, 5, 5
RECENT = 2025  # 2025~2026만 따로

# 매수 설정: 키 -> (첫 몇 분 고가, 거래량 폭발 기준, 매수 마감 분)
CFG = {"base": (1, 0.5, 90), "w5": (5, 0.5, 90), "w30": (30, 0.5, 90), "v100": (1, 1.0, 90),
       "v30": (1, 0.3, 90), "dl30": (1, 0.5, 30), "dl150": (1, 0.5, 150)}
# 코드, 이름, 최소 갭, 실적·뉴스 조건(any = 실적일 또는 장전 거래 10%↑, earn = 실적일만), 120일 수익률 상한, 매수 설정, 보유
VARIANTS = [("E1", "E1 기본 (갭 10%, 1분 고가 돌파)", 0.10, "any", None, "base", "day"),
            ("E2-5", "E2 5분 고가 돌파", 0.10, "any", None, "w5", "day"),
            ("E2-30", "E2 30분 고가 돌파", 0.10, "any", None, "w30", "day"),
            ("E3", "E3 거래량 20분에 평소 하루 100%", 0.10, "any", None, "v100", "day"),
            ("E4", "E4 갭 15% 이상", 0.15, "any", None, "base", "day"),
            ("E5", "E5 실적 발표일만", 0.10, "earn", None, "base", "day"),
            ("E6", "E6 최근 120일 +20% 이하만", 0.10, "any", 0.20, "base", "day"),
            ("E7", "E7 원작 매도 (며칠 보유, 비교용)", 0.10, "any", None, "base", "swing")]
SENS = [("S-gap8", "갭 8% 이상", 0.08, "any", None, "base", "day"),
        ("S-vol30", "거래량 기준 30%", 0.10, "any", None, "v30", "day"),
        ("S-dl10", "매수 마감 10:00", 0.10, "any", None, "dl30", "day"),
        ("S-dl12", "매수 마감 12:00", 0.10, "any", None, "dl150", "day")]


def load(prep, kind):
    return pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(os.path.join(prep, f"{kind}_*.parquet")))])


def daily_features(daily):
    """종목별 직전 20일 평균 거래량(주식 수)·평균 하루 변동폭, 직전 종가 기준 120일 수익률."""
    g = daily.groupby("ticker", sort=False)
    prev = lambda s: s.groupby(daily.ticker).shift(1)  # noqa: E731
    daily = daily.assign(rng=daily.h / daily.l - 1,
                         jump=~(daily.o / g.c.shift(1)).between(0.6, 1.7) & g.c.shift(1).notna())
    roll = lambda s, n: s.groupby(daily.ticker).transform(lambda x: x.rolling(n, min_periods=10).mean())  # noqa: E731
    adv_sh = prev(roll(daily.v, 20))
    adr = prev(roll(daily.rng, 20))
    c1 = prev(daily.c)
    c121 = daily.groupby("ticker").c.shift(121)
    jumps = daily.jump.astype(float).groupby(daily.ticker).transform(lambda x: x.rolling(121, min_periods=1).sum())
    ret120 = (c1 / c121 - 1).where(jumps == 0)  # 그 사이 분할로 보이는 날이 있으면 모름
    return daily[["ticker", "d"]].assign(adv_sh=adv_sh, adr=adr, ret120=ret120)


def base_trades(prep, feats, daily):
    f = feats.set_index(["ticker", "d"])
    hist = {tk: (g.d.to_numpy(), {k: g[k].to_numpy() for k in ("o", "h", "l", "c")})
            for tk, g in daily.groupby("ticker", sort=False)}
    rows = []
    for fn in sorted(glob.glob(os.path.join(prep, "bars_*.parquet"))):
        df = pq.read_table(fn).to_pandas()
        df = df[df.set_index(["ticker", "d"]).index.isin(f.index)]
        for (tk, d), g in df.groupby(["ticker", "d"], sort=False):
            x = f.loc[(tk, d)]
            day = {k: g[k].to_numpy() for k in ("mi", "o", "h", "l", "c", "v")}
            for key, (w, vf, dl) in CFG.items():
                e = b04.find_entry(day, w, x.adv_sh, vf, x.adr, dl)
                if e is None:
                    continue
                t = b04._exit(day, *e)
                if t:
                    rows.append(dict(ticker=tk, d=d, cfg=key, hold="day", **t))
                if key == "base":
                    ds, a = hist[tk]
                    k = np.searchsorted(ds, d)
                    t = b04.swing(day, *e, a["c"][max(0, k - 9):k], {n: v[k + 1:k + 61] for n, v in a.items()})
                    rows.append(dict(ticker=tk, d=d, cfg=key, hold="swing", exit_mi=385, **t))
        print(f"{os.path.basename(fn)}: base trades so far {len(rows)}", flush=True)
    return pd.DataFrame(rows).merge(feats, on=["ticker", "d"], how="left")


def pick(t, gap, news, ret_cap, cfg, hold):
    s = t[(t.cfg == cfg) & (t.hold == hold) & (t.gap >= gap)]
    s = s[s.earn] if news == "earn" else s[s.earn | (s.pm_rel >= 0.10)]
    if ret_cap is not None:
        s = s[s.ret120 <= ret_cap]
    s = s.sort_values(["d", "gap"], ascending=[True, False]).groupby("d").head(TOP_DAY)
    return s.assign(rank=-s.gap, cost=FEE + SLIP)


def pct(x, nd=1):
    return "" if pd.isna(x) else f"{x * 100:.{nd}f}%"


def gate_rows(variants, sel, days):
    out, passed = [], []
    for code, name, *_ in variants:
        s = sel[code]
        if s.empty:
            out.append(f"| {name} | 0 |" + " |" * 11)
            continue
        r = s[pd.DatetimeIndex(s.d).year >= RECENT].gross
        n1, n2 = portfolio.net_return(s.gross, FEE + SLIP), portfolio.net_return(s.gross, FEE + SLIP2)
        ok = s.gross.mean() >= GATE and len(s) >= MIN_N
        if ok:
            passed.append(code)
        rc = ", ".join(f"{k} {v}" for k, v in s.reason.value_counts().items())
        held = f" (평균 {s.days.mean():.1f}일)" if "days" in s and s.days.notna().any() else ""
        out.append(f"| {name} | {len(s)} | {len(s) / len(days):.2f} | {pct(s.gross.mean(), 2)} | {pct(s.gross.median(), 2)} | "
                   f"{pct(r.mean(), 2)} ({len(r)}) | {pct(r.median(), 2)} | {pct(n1.mean(), 2)} | {pct(n2.mean(), 2)} | "
                   f"{pct((n1 > 0).mean())} | {pct(s.stop_dist.mean(), 2)} | {rc}{held} | "
                   f"{'통과' if ok else ('탈락' if len(s) >= MIN_N else '신호 부족')} |")
    return out, passed


GATE_HEAD = ["| 변형 | 신호 수 | 하루 평균 | 비용 전 평균 | 비용 전 중앙값 | 2025~26 평균 (수) | 2025~26 중앙값 | "
             "비용 후 평균 | 비용 후 (미끄러짐 0.2%) | 승률(비용 후) | 평균 손절폭 | 청산 이유 | 관문 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_b04")
    ap.add_argument("--out", default="reports/b04")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spy = load(a.prep, "spy").drop_duplicates("d").set_index("d").c.sort_index()
    days = spy.index
    spy_ret = spy.pct_change().fillna(0.0)
    daily = load(a.prep, "daily").drop_duplicates(["ticker", "d"]).sort_values(["ticker", "d"]).reset_index(drop=True)
    earn = earnings_days(a.prep, days)
    if earn is None:
        raise SystemExit("SEC 실적 자료(earnings.parquet)가 없어 b04를 돌릴 수 없음 (회사 주식 구분과 U2에 필요)")
    companies = set(earn.ticker)
    daily = daily[daily.ticker.isin(companies)].reset_index(drop=True)
    feats = load(a.prep, "feats")
    n_all = len(feats)
    feats = feats[feats.ticker.isin(companies)]
    feats = feats.merge(daily_features(daily), on=["ticker", "d"], how="left")
    feats = feats.merge(earn.assign(earn=True), on=["ticker", "d"], how="left").assign(
        earn=lambda x: x.earn.fillna(False).astype(bool))
    t = base_trades(a.prep, feats, daily)
    t.to_csv(os.path.join(a.out, "trades.csv.gz"), index=False)

    sel = {c: pick(t, *v) for c, _, *v in VARIANTS + SENS}
    lines = [f"# b04 백테스트 결과 ({days[0]} ~ {days[-1]}, {len(days)}거래일)", "",
             "비용: 한쪽 수수료 0.25% + 미끄러짐 0.1% (미끄러짐 0.2%도 함께). SPY는 배당 제외 가격 수익률.",
             f"후보(갭 8%↑, 5달러↑, 평소 거래대금 500만$↑): {n_all}건 중 회사 주식 {len(feats)}건 "
             f"(실적일 {int(feats.earn.sum())}건). 실적 8-K 자료: {len(earn)}건", "",
             f"## 1차 관문: 비용 전 거래당 평균 ≥ +{GATE * 100:.1f}% 인가 (하루 갭 큰 순 {TOP_DAY}개, 자금관리 없이)", "",
             *GATE_HEAD]
    rows, passed = gate_rows(VARIANTS, sel, days)
    lines += rows
    day_pass = [c for c in passed if c != "E7"]
    lines += ["", f"**당일판 관문 통과 변형: {', '.join(day_pass) if day_pass else '없음 → b04 탈락'}**"
              + (" (E7은 비교용이라 판정에서 뺌)" if "E7" in passed else "")]
    if day_pass:
        rows, _ = gate_rows(SENS, sel, days)
        lines += ["", "## 민감도 (E1 기준으로 하나씩 바꿈)", "", *GATE_HEAD, *rows]

    bins = [0.08, 0.10, 0.15, 0.25, np.inf]
    labels = ["8~10%", "10~15%", "15~25%", "25% 이상"]
    lines += ["", "## 갭 크기별 비용 전 거래당 평균 (신호 수)", "", "| 변형 | " + " | ".join(labels) + " |",
              "|---" * (len(labels) + 1) + "|"]
    for code, *_ in VARIANTS + (SENS[:1] if day_pass else []):
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

    e1, e7 = sel["E1"], sel["E7"]
    if not e1.empty:
        lines.append(f"\nE1 신호가 많은 종목: {', '.join(f'{k} {v}' for k, v in e1.ticker.value_counts().head(10).items())}")
        lines.append(f"E1 매수 시각 중앙값: 9:30 + {int(e1.entry_mi.median())}분, 실적일 비율 {e1.earn.mean():.0%}")
        top = e1.gross.sort_values(ascending=False)
        lines.append(f"E1 가장 좋은 거래 10개를 빼면 비용 전 평균: {pct(top.iloc[10:].mean(), 2)}")
    if not e7.empty:
        top = e7.gross.sort_values(ascending=False)
        lines.append(f"E7 가장 좋은 거래 10개를 빼면 비용 전 평균: {pct(top.iloc[10:].mean(), 2)}; "
                     f"보유일 중앙값 {e7.days.median():.0f}일, 가장 좋은 거래: "
                     + ", ".join(f"{r.ticker} {r.d} {r.gross * 100:+.0f}%" for r in e7.nlargest(5, 'gross').itertuples()))
    text = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
