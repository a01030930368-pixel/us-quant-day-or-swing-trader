"""Backtest every swing candidate and write reports/swing_report_<universe>.md.

    python scripts/run_backtest.py
    python scripts/run_backtest.py --universe all --start 2010-01-01 --split 2018-01-01 --cost-bps 20
    python scripts/run_backtest.py --set round2      # the earlier momentum/gap/52-week candidates
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swing import engine, metrics  # noqa: E402
from swing.data import load_ohlc  # noqa: E402
from swing.legends import LEGENDS  # noqa: E402
from swing.legends2 import LEGENDS2  # noqa: E402
from swing.refine import REFINED  # noqa: E402
from swing.earnings_legends import EARNINGS, EPS_MIN, SALES_MIN  # noqa: E402
from swing.fundamentals import growth_flags, load_facts  # noqa: E402
from swing.strategies import ALL, MomentumSwing  # noqa: E402

TITLES = {"legends": "전설 1그룹 규칙", "legends2": "전설 2그룹 규칙", "round2": "후보 전략",
          "refine": "와인스타인·다바스 다듬기", "legends_all": "전설 1·2그룹 규칙",
          "earnings": "전설 규칙 + 실적 조건"}
SETS = {"legends": LEGENDS, "legends2": LEGENDS2, "round2": ALL, "refine": REFINED,
        "legends_all": LEGENDS + LEGENDS2, "earnings": EARNINGS}
RULE_CHAPTERS = {"legends": "12장", "legends2": "14장", "refine": "12·14장",
                 "legends_all": "12·14장", "earnings": "12·14·17장"}
from swing.universe import MARKET, UNIVERSE_LABELS, UNIVERSES  # noqa: E402


def stats(ret: pd.Series, trades: pd.DataFrame | None = None,
          exposure: pd.Series | None = None) -> dict:
    s = metrics.summarize(ret)
    row = {
        "연수익률": s["cagr"],
        "월평균(복리)": (1 + s["cagr"]) ** (1 / 12) - 1,
        "최대낙폭": s["mdd"],
        "샤프": s["sharpe"],
        "수익 난 달": s.get("month_win_rate", float("nan")),
        "최악의 달": s.get("worst_month", float("nan")),
    }
    if trades is not None:
        # A half sale is its own row; weight rows by the share of the position they sold,
        # so one bought position counts once.
        part = trades["part"] if "part" in trades else pd.Series(1.0, index=trades.index)
        n = float(part.sum())
        row["거래수"] = round(n)
        row["승률"] = float(((trades["return"] > 0) * part).sum() / n) if n else float("nan")
        row["평균 거래수익"] = float((trades["return"] * part).sum() / n) if n else float("nan")
        row["평균 보유일"] = float((trades["days"] * part).sum() / n) if n else float("nan")
        if "weight" in trades and n:
            row["평균 비중"] = float((trades["weight"] * part).sum() / n)
    if exposure is not None:
        row["평균 투자비중"] = float(exposure.mean())
    return row


PCT = {"평균 비중", "연수익률", "월평균(복리)", "최대낙폭", "수익 난 달", "최악의 달", "승률", "평균 거래수익", "평균 투자비중"}


def table(rows: dict[str, dict]) -> str:
    cols = list(next(iter(rows.values())).keys())
    out = ["| 전략 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in rows.items():
        cells = []
        for k in cols:
            v = r.get(k, float("nan"))
            if pd.isna(v):
                cells.append("-")
            elif k in PCT:
                cells.append(f"{v * 100:.1f}%")
            elif k in ("거래수",):
                cells.append(f"{int(v)}")
            else:
                cells.append(f"{v:.2f}" if k == "샤프" else f"{v:.1f}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def yearly(series: dict[str, pd.Series]) -> str:
    frame = pd.DataFrame({k: (1 + r).groupby(r.index.year).prod() - 1 for k, r in series.items()})
    out = ["| 연도 | " + " | ".join(frame.columns) + " |", "|---" * (len(frame.columns) + 1) + "|"]
    for y, r in frame.iterrows():
        out.append(f"| {y} | " + " | ".join(f"{v * 100:+.1f}%" for v in r) + " |")
    return "\n".join(out)


def correlations(series: dict[str, pd.Series], refs: dict[str, pd.Series], since) -> str:
    """Correlation of monthly returns in the OOS period: 1 = always move together."""
    def monthly(r):
        r = r[r.index >= since]
        return (1 + r).groupby(r.index.to_period("M")).prod() - 1
    out = ["| 전략 | " + " | ".join(refs) + " |", "|---" * (len(refs) + 1) + "|"]
    for name, r in series.items():
        m = monthly(r)
        vals = [m.corr(monthly(ref)) for ref in refs.values()]
        out.append(f"| {name} | " + " | ".join("-" if pd.isna(v) else f"{v:.2f}" for v in vals) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01", help="백테스트 시작일")
    ap.add_argument("--split", default="2018-01-01", help="검증(OOS) 구간 시작일")
    ap.add_argument("--cost-bps", type=float, default=20.0, help="편도 거래비용 (bp)")
    ap.add_argument("--universe", default="large", choices=sorted(UNIVERSES),
                    help="large: 대형주, all: 대형주 + 중형주, small: 소형주, every: 전부")
    ap.add_argument("--set", default="legends", choices=sorted(SETS),
                    help="legends: 전설 1그룹(와인스타인·미너비니·본드·오닐), legends2: 2그룹(다바스·쿨라매기·래쉬키·켈), round2: 1·2차 후보, refine: 와인스타인·다바스 다듬기, legends_all: 1·2그룹 전부, earnings: 실적 조건 비교")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    warmup = str(pd.Timestamp(args.start).year - 1) + "-01-01"   # 200-day averages need a year
    data = load_ohlc(UNIVERSES[args.universe] + [MARKET], start=warmup)
    split = pd.Timestamp(args.split)
    universe = [t for t in data["Close"].columns if t != MARKET]

    earn_lines = []
    if args.set == "earnings":
        flags, cover = growth_flags(load_facts(universe), data["Close"].index, EPS_MIN, SALES_MIN)
        if cover["eps"] == 0:
            raise RuntimeError("SEC 실적 데이터를 하나도 받지 못했습니다. 접속이 막혔는지 확인하세요.")
        data["EarnOK"] = flags
        since = flags.index >= pd.Timestamp(args.start)
        earn_lines = [
            f"- 실적 데이터: 미국 증권거래위원회(SEC) 공시. {len(universe)}개 중 EPS는 {cover['eps']}개, "
            f"매출은 {cover['sales']}개 종목에서 받았습니다(은행 등 매출 항목이 다른 회사는 빠짐). "
            "공시한 다음 날부터만 씁니다.",
            f"- 실적 조건을 통과한 종목 수: 하루 평균 {flags[since].sum(axis=1).mean():.0f}개",
        ]
    spy = data["Close"][MARKET]
    spy_ret = spy[spy.index >= pd.Timestamp(args.start)].pct_change().fillna(0.0)
    results = {}
    for cls in SETS[args.set]:
        strat = cls().prepare(data, market=MARKET)
        results[strat.label] = (strat, engine.run(strat, cost_bps=args.cost_bps, start=args.start))

    def period(mask_fn) -> dict[str, dict]:
        rows = {}
        for label, (_, res) in results.items():
            m = mask_fn(res.returns.index)
            tr = res.trades
            tm = mask_fn(pd.to_datetime(tr["entry_date"])) if len(tr) else []
            rows[label] = stats(res.returns[m], tr[tm] if len(tr) else tr, res.exposure[m])
        rows["(비교) S&P500 그냥 보유"] = stats(spy_ret[mask_fn(spy_ret.index)])
        return rows

    full = period(lambda ix: ix >= pd.Timestamp(args.start))
    ins = period(lambda ix: (ix >= pd.Timestamp(args.start)) & (ix < split))
    oos = period(lambda ix: ix >= split)

    # Cost sensitivity: same strategies at double the trading cost.
    dbl = {}
    for label, (strat, _) in results.items():
        r = engine.run(strat, cost_bps=args.cost_bps * 2, start=args.start)
        m = r.returns.index >= split
        tr = r.trades[pd.to_datetime(r.trades["entry_date"]) >= split] if len(r.trades) else r.trades
        dbl[label] = stats(r.returns[m], tr, r.exposure[m])

    mom = engine.run(MomentumSwing().prepare(data, market=MARKET), cost_bps=args.cost_bps,
                     start=args.start).returns
    corr = correlations({k: v[1].returns for k, v in results.items()},
                        {"S&P500": spy_ret, "주간 모멘텀(이 앱)": mom}, split)

    end = spy.index[-1].date()
    lines = [
        f"# 미국주식 스윙 {TITLES[args.set]} 백테스트 "
        f"({UNIVERSE_LABELS[args.universe].removeprefix('미국 ')})",
        "",
        f"- 기간: {args.start} ~ {end} (학습 구간 {args.start} ~ {split.date()}, "
        f"**검증 구간 {split.date()} ~ {end}**)",
        f"- 종목: {UNIVERSE_LABELS[args.universe]} {len(universe)}개 "
        "(`swing/universe.py`). 보유 종목 수와 비중은 전략마다 다릅니다(아래 설명).",
        f"- 거래비용: 사고팔 때 각각 {args.cost_bps:.0f}bp ({args.cost_bps / 100:.2f}%)",
        "- 신호는 종가로 판단하고 다음 날 시가에 사고팝니다. 손절은 장중 저가로 확인합니다.",
        *([f"- 규칙은 문서 {RULE_CHAPTERS[args.set]} 매매 규칙서 그대로이고, "
            "이 데이터에 맞춰 조정하지 않았습니다. 시장 필터도 전략마다 다릅니다."]
          if args.set != "round2" else [
            "- 모든 전략은 S&P500이 200일 평균 아래면 새로 사지 않습니다(하락장 필터).",
            "- 파라미터는 널리 알려진 기본값 그대로이고, 이 데이터에 맞춰 조정하지 않았습니다."]),
        *earn_lines,
        "- 절반 매도는 거래 1건으로 따로 셉니다. 승률·평균 거래수익은 이 건들을 모두 포함합니다.",
        "",
        "## 전략 설명",
        "",
    ]
    for label, (strat, _) in results.items():
        lines.append(f"- **{label}**: {strat.description}")
    lines += [
        "",
        "## 검증 구간 성과 (가장 중요)",
        "",
        "학습 구간 숫자는 참고용이고, 전략을 고를 때는 이 표를 봅니다.",
        "",
        table(oos),
        "",
        "## 전체 기간",
        "",
        table(full),
        "",
        "## 학습 구간",
        "",
        table(ins),
        "",
        f"## 거래비용 2배({args.cost_bps * 2:.0f}bp)일 때 검증 구간",
        "",
        "비용이 늘어도 버티는지 봅니다. 크게 무너지면 실전에서 위험합니다.",
        "",
        table(dbl),
        "",
        "## 다른 전략과 같이 움직이는 정도 (검증 구간 월수익률 상관계수)",
        "",
        "1에 가까우면 같이 오르고 같이 내립니다. 기존 앱과 함께 돌린다면 낮을수록 위험이 분산됩니다. "
        "'주간 모멘텀(이 앱)'은 기존 퀀트 앱의 ETF 모멘텀과 같은 전략은 아니고 비슷한 성격의 참고용입니다.",
        "",
        corr,
        "",
        "## 연도별 수익률",
        "",
        yearly({**{k: v[1].returns[v[1].returns.index >= pd.Timestamp(args.start)]
                   for k, v in results.items()}, "S&P500": spy_ret}),
        "",
        "## 매도 이유",
        "",
    ]
    for label, (_, res) in results.items():
        if len(res.trades):
            tr = res.trades.assign(contrib=res.trades["return"] * res.trades["weight"]
                                   * res.trades["part"])
            g = tr.groupby("reason").agg(n=("return", "size"), avg=("return", "mean"),
                                         contrib=("contrib", "sum"))
            parts = [f"{r} {int(x.n)}회(평균 {x.avg * 100:+.1f}%, 계좌 기여 합계 {x.contrib * 100:+.0f}%p)"
                     for r, x in g.iterrows()]
            lines.append(f"- **{label}**: " + ", ".join(parts))
    lines += [
        "",
        "## 주의",
        "",
        *(["- 종목 목록이 지금의 대형주라서, 그동안 망하거나 작아진 회사가 빠져 있습니다. "
           "실제보다 결과가 좋게 나오므로 상한선으로 보세요."] if args.universe != "small" else [
           "- 종목 목록이 지금의 소형주입니다. 그동안 망한 회사가 빠져 결과가 좋아 보이는 쪽과, "
           "크게 성장해 대형주가 된 회사가 빠져 결과가 나빠 보이는 쪽이 함께 있습니다."]),
        "- 소수점 주식으로 계산했습니다. 실제로는 1주 단위라 작은 계좌일수록 조금 달라집니다.",
        "- 세금(해외주식 양도세 22%)과 환율 변동은 넣지 않았습니다.",
        "",
    ]
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    suffix = args.universe if args.set == "legends" else f"{args.set}_{args.universe}"
    (out / f"swing_report_{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    for label, (strat, res) in results.items():
        res.trades.to_csv(out / f"trades_{suffix}_{strat.name}.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
