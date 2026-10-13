"""b08 데이트레이딩 백테스트 실행 (IPO 첫날): prep_b08 폴더 -> 결과표(마크다운).

    python scripts/run_b08.py --prep prep_b08 --out reports/b08
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daytrade import b08, portfolio  # noqa: E402

FEE, SLIP, SLIP_LOW = 0.0025, 0.002, 0.001  # 상장 첫날은 미끄러짐 0.2% (왕복 0.9%), 참고로 0.1% (왕복 0.7%)
GATE, MIN_N = 0.011, 100
WEIGHT, MAX_PER_DAY = 0.2, 5
FIRST_YEAR = 1993
PERIODS = [("1993~2003", 1993, 2003), ("2004~2023 (엘론 대학 자료 기간)", 2004, 2023), ("2024~2026", 2024, 2026)]
VARIANTS = [("I1", "I1 기본: 첫 1분봉 종가 매수", dict()),
            ("I2", "I2 첫 체결가(시가) 매수", dict(mode="I2")),
            ("I3", "I3 첫 거래 15분 뒤 매수", dict(delay=15)),
            ("I4", "I4 30분 뒤 시가보다 높을 때만", dict(delay=30, cond="up")),
            ("I5", "I5 30분 뒤 시가보다 낮을 때만", dict(delay=30, cond="down")),
            ("I6", "I6 I1 + 손절 -8%", dict(stop=0.08))]


def read_all(prep, kind):
    fs = sorted(glob.glob(os.path.join(prep, f"{kind}_*.parquet")))
    return pd.concat([pq.read_table(f).to_pandas() for f in fs], ignore_index=True) if fs else pd.DataFrame()


def find_ipos(tick, bars, cal):
    """후보(그해 처음 보인 종목의 첫날) 중 진짜 IPO 첫날만 남기고, 빠진 이유를 센다."""
    tick = tick.copy()
    tick["first_d"], tick["last_d"] = pd.to_datetime(tick.first_d), pd.to_datetime(tick.last_d)
    cand = bars.groupby(["ticker", "d"], sort=False).size().reset_index()[["ticker", "d"]]
    cand["d"] = pd.to_datetime(cand.d)
    cand = cand[cand.d.dt.year >= FIRST_YEAR]
    # U1: 직전 250거래일(약 1년) 안에 그 코드 거래가 있었으면 IPO 아님
    prev = tick.merge(cand, on="ticker")
    prev = prev[prev.last_d < prev.d].groupby(["ticker", "d"]).last_d.max().rename("last_seen").reset_index()
    cand = cand.merge(prev, on=["ticker", "d"], how="left")
    cal = pd.DatetimeIndex(sorted(cal))
    pos = cal.searchsorted(cand.d)
    cand["gap_days"] = np.where(cand.last_seen.notna(), pos - cal.searchsorted(cand.last_seen.fillna(cal[0])), 10 ** 6)
    # U2: 전 거래일에 사라진 코드의 마지막 종가와 첫 시가가 2% 안이면 코드 변경
    gone = tick[["ticker", "last_d", "last_c"]].rename(columns={"ticker": "old"})
    cand["prev_day"] = cal[np.maximum(pos - 1, 0)]
    return cand, gone


def stats_row(name, s, judge=True):
    if s.empty:
        return f"| {name} | 0 |" + " |" * 9, False
    g = s.gross
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 and g.std() > 0 else np.nan
    n1, n2 = portfolio.net_return(g, FEE + SLIP), portfolio.net_return(g, FEE + SLIP_LOW)
    ok = judge and g.mean() >= GATE and len(g) >= MIN_N
    verdict = ("통과" if ok else ("탈락" if len(g) >= MIN_N else "신호 부족")) if judge else "(판정 안 함)"
    return (f"| {name} | {len(g)} | {g.mean() * 100:.2f}% | {g.median() * 100:.2f}% | {(g > 0).mean():.0%} | {t:.2f} | "
            f"{n1.mean() * 100:.2f}% | {n2.mean() * 100:.2f}% | {(n1 > 0).mean():.0%} | {(s.reason == '손절').sum()} | {verdict} |"), ok


HEAD = ("| 변형 | IPO 수 | 비용 전 평균 | 비용 전 중앙값 | 비용 전 승률 | t값 | 비용 후 평균 (왕복 0.9%) | 비용 후 (왕복 0.7%) | "
        "비용 후 승률 | 손절 | 관문 |", "|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_b08")
    ap.add_argument("--out", default="reports/b08")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tick, bars, spy = read_all(a.prep, "tick"), read_all(a.prep, "bars"), read_all(a.prep, "spy")
    spy = spy.drop_duplicates("d").assign(d=lambda x: pd.to_datetime(x.d)).set_index("d").c.sort_index()
    bars = bars.sort_values(["ticker", "d", "mi"])
    days = {(tk, pd.Timestamp(d)): {k: g[k].to_numpy() for k in ("mi", "o", "h", "l", "c", "v")}
            for (tk, d), g in bars.groupby(["ticker", "d"], sort=False)}
    cand, gone = find_ipos(tick, bars, spy.index)

    rows, drop = [], {"직전 1년 안에 거래 있음 (재등장)": 0, "코드 변경으로 보임": 0, "스팩·유닛·워런트": 0,
                      "첫 시가 5달러 미만": 0, "첫 1시간 거래대금 1,000만 달러 미만": 0, "15:00 이후 첫 거래": 0}
    gone_by_day = {d: g for d, g in gone.groupby("last_d")}
    for r in cand.itertuples():
        day = days[(r.ticker, r.d)]
        o0 = day["o"][0]
        if r.gap_days <= 250:
            drop["직전 1년 안에 거래 있음 (재등장)"] += 1
            continue
        g = gone_by_day.get(r.prev_day)
        if g is not None and ((g.old != r.ticker) & ((o0 / g.last_c - 1).abs() < 0.02)).any():
            drop["코드 변경으로 보임"] += 1
            continue
        if b08.is_spac_like(r.ticker, day):
            drop["스팩·유닛·워런트"] += 1
            continue
        if o0 < 5:
            drop["첫 시가 5달러 미만"] += 1
            continue
        dv = b08.dv60(day)
        if dv < 10e6:
            drop["첫 1시간 거래대금 1,000만 달러 미만"] += 1
            continue
        if day["mi"][0] > 330:
            drop["15:00 이후 첫 거래"] += 1
            continue
        base = dict(ticker=r.ticker, d=r.d, first_mi=int(day["mi"][0]), open0=o0, dv60=dv)
        for code, _, kw in VARIANTS:
            t = b08.trade(day, **kw)
            if t:
                rows.append(dict(base, code=code, **t))
    t = pd.DataFrame(rows)
    t["year"] = t.d.dt.year
    i1 = t[t.code == "I1"]
    cut = i1.dv60.quantile(0.7)
    sel = {code: t[t.code == code] for code, _, _ in VARIANTS}
    sel["I7"] = i1[i1.dv60 >= cut]
    names = {code: n for code, n, _ in VARIANTS}
    names["I7"] = f"I7 첫 1시간 거래대금 상위 30% (≥ {cut / 1e6:.0f}백만 달러)"
    t.to_csv(os.path.join(a.out, "trades.csv"), index=False)

    lines = [f"# b08 백테스트 결과 (IPO 첫날, {t.d.min().date()} ~ {t.d.max().date()})", "",
             "비용: 수수료 0.25% + 미끄러짐 0.2% (한쪽 0.45%, 왕복 0.9%). 참고로 미끄러짐 0.1%(왕복 0.7%)도. t값 2 이상이면 대략 '우연은 아님'.",
             f"후보(그해 처음 보인 종목 첫날, 첫 시가 ≥ 4달러, 첫 1시간 거래대금 ≥ 500만 달러) {len(cand)}개 중 IPO로 본 것 {i1.ticker.size}개. 빠진 이유: "
             + ", ".join(f"{k} {v}" for k, v in drop.items()), "",
             f"## 1차 관문: 비용 전 거래당 평균 ≥ +{GATE * 100:.1f}% 인가 (IPO {MIN_N}개 이상)", "", *HEAD]
    passed = []
    for k, s in sel.items():
        row, ok = stats_row(names[k], s)
        lines.append(row)
        if ok:
            passed.append(k)
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else '없음 → b08 탈락'}**"]
    for lab, y0, y1 in PERIODS:
        lines += ["", f"## I8 기간: {lab}", "", *HEAD]
        for k, s in sel.items():
            lines.append(stats_row(names[k], s[(s.year >= y0) & (s.year <= y1)], judge=False)[0])

    lines += ["", "## 해마다 (I1 기본 / I2 시가 매수, 비용 전)", "", "| 해 | IPO 수 | I1 평균 | I1 중앙값 | I2 평균 | I2 중앙값 |", "|---|---|---|---|---|---|"]
    i2 = sel["I2"]
    for y in sorted(t.year.unique()):
        g1, g2 = i1[i1.year == y].gross, i2[i2.year == y].gross
        lines.append(f"| {y} | {len(g1)} | {g1.mean() * 100:.2f}% | {g1.median() * 100:.2f}% | {g2.mean() * 100:.2f}% | {g2.median() * 100:.2f}% |")

    lines += ["", "## 첫 1시간 거래대금 구간별 (I1, 비용 전)", "", "| 구간 | IPO 수 | 평균 | 중앙값 | 승률 |", "|---|---|---|---|---|"]
    for lo, hi in ((10e6, 30e6), (30e6, 100e6), (100e6, 300e6), (300e6, 1e15)):
        g = i1[(i1.dv60 >= lo) & (i1.dv60 < hi)].gross
        lab = f"{lo / 1e6:.0f}~{hi / 1e6:.0f}백만 달러" if hi < 1e15 else f"{lo / 1e6:.0f}백만 달러 이상"
        if len(g):
            lines.append(f"| {lab} | {len(g)} | {g.mean() * 100:.2f}% | {g.median() * 100:.2f}% | {(g > 0).mean():.0%} |")
    lines += ["", "## 첫 거래 시각별 (I1, 비용 전)", "", "| 첫 거래 | IPO 수 | 평균 | 중앙값 |", "|---|---|---|---|"]
    for lo, hi, lab in ((0, 1, "9:30 (바로)"), (1, 90, "9:31~10:59"), (90, 210, "11:00~12:59"), (210, 331, "13:00~15:00")):
        g = i1[(i1.first_mi >= lo) & (i1.first_mi < hi)].gross
        if len(g):
            lines.append(f"| {lab} | {len(g)} | {g.mean() * 100:.2f}% | {g.median() * 100:.2f}% |")

    def fmt(r):
        return f"| {r.d.date()} | {r.ticker} | {r.open0:.2f} | {9 + (30 + r.first_mi) // 60}:{(30 + r.first_mi) % 60:02d} | {r.dv60 / 1e6:.0f} | {r.gross * 100:+.1f}% |"
    th = ["| 날짜 | 종목 | 첫 시가 | 첫 거래 | 첫 1시간 거래대금(백만 달러) | I1 비용 전 |", "|---|---|---|---|---|---|"]
    lines += ["", "## 눈으로 확인: 첫 1시간 거래대금 상위 20개 (I1)", "", *th] + [fmt(r) for r in i1.nlargest(20, "dv60").itertuples()]
    lines += ["", "## I1 가장 좋은 10개", "", *th] + [fmt(r) for r in i1.nlargest(10, "gross").itertuples()]
    lines += ["", "## I1 가장 나쁜 10개", "", *th] + [fmt(r) for r in i1.nsmallest(10, "gross").itertuples()]
    top = i1.gross.sort_values(ascending=False)
    lines.append(f"\nI1 가장 좋은 10개를 빼면 비용 전 평균: {top.iloc[10:].mean() * 100:.2f}%")

    cal = spy.index[spy.index >= pd.Timestamp(f"{FIRST_YEAR}-01-01")]
    s0 = portfolio.stats(spy[cal].pct_change().fillna(0.0))
    lines += ["", f"## 계좌 성과 (IPO 하나에 계좌 {WEIGHT:.0%}, 하루 최대 {MAX_PER_DAY}개, 비용 왕복 0.9%, {FIRST_YEAR}~)", "",
              "| 변형 | 연수익률 | 최대낙폭 |", "|---|---|---|", f"| SPY 보유 (배당 제외) | {s0['cagr'] * 100:.1f}% | {s0['mdd'] * 100:.1f}% |"]
    for k, s in sel.items():
        if s.empty:
            continue
        s = s.sort_values("dv60", ascending=False).groupby("d").head(MAX_PER_DAY)
        r = (WEIGHT * portfolio.net_return(s.gross, FEE + SLIP)).groupby(s.d).sum().reindex(cal, fill_value=0.0)
        st = portfolio.stats(r)
        lines.append(f"| {names[k]} | {st['cagr'] * 100:.1f}% | {st['mdd'] * 100:.1f}% |")
    text = "\n".join(lines) + "\n"
    open(os.path.join(a.out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
