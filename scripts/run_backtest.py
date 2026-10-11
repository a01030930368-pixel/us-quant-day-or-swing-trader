"""Backtest every swing candidate and write reports/swing_report.md.

    python scripts/run_backtest.py
    python scripts/run_backtest.py --start 2010-01-01 --split 2018-01-01 --cost-bps 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swing import engine, metrics  # noqa: E402
from swing.data import load_ohlc  # noqa: E402
from swing.strategies import ALL  # noqa: E402
from swing.universe import LARGE_CAPS, MARKET  # noqa: E402


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
        row["거래수"] = len(trades)
        row["승률"] = float((trades["return"] > 0).mean()) if len(trades) else float("nan")
        row["평균 거래수익"] = float(trades["return"].mean()) if len(trades) else float("nan")
        row["평균 보유일"] = float(trades["days"].mean()) if len(trades) else float("nan")
    if exposure is not None:
        row["평균 투자비중"] = float(exposure.mean())
    return row


PCT = {"연수익률", "월평균(복리)", "최대낙폭", "수익 난 달", "최악의 달", "승률", "평균 거래수익", "평균 투자비중"}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01", help="백테스트 시작일")
    ap.add_argument("--split", default="2018-01-01", help="검증(OOS) 구간 시작일")
    ap.add_argument("--cost-bps", type=float, default=20.0, help="편도 거래비용 (bp)")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    warmup = str(pd.Timestamp(args.start).year - 1) + "-01-01"   # 200-day averages need a year
    data = load_ohlc(LARGE_CAPS + [MARKET], start=warmup)
    split = pd.Timestamp(args.split)
    universe = [t for t in data["Close"].columns if t != MARKET]

    spy = data["Close"][MARKET]
    spy_ret = spy[spy.index >= pd.Timestamp(args.start)].pct_change().fillna(0.0)
    results = {}
    for cls in ALL:
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

    end = spy.index[-1].date()
    lines = [
        "# 미국주식 스윙 후보 전략 백테스트",
        "",
        f"- 기간: {args.start} ~ {end} (학습 구간 {args.start} ~ {split.date()}, "
        f"**검증 구간 {split.date()} ~ {end}**)",
        f"- 종목: 미국 대형주 {len(universe)}개 (`swing/universe.py`), 동시 보유 최대 5종목, 종목당 계좌의 20%",
        f"- 거래비용: 사고팔 때 각각 {args.cost_bps:.0f}bp ({args.cost_bps / 100:.2f}%)",
        "- 신호는 종가로 판단하고 다음 날 시가에 사고팝니다. 손절은 장중 저가로 확인합니다.",
        "- 세 전략 모두 S&P500이 200일 평균 아래면 새로 사지 않습니다(하락장 필터).",
        "- 파라미터는 널리 알려진 기본값 그대로이고, 이 데이터에 맞춰 조정하지 않았습니다.",
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
            g = res.trades.groupby("reason")["return"].agg(["count", "mean"])
            parts = [f"{r} {int(n)}회(평균 {m * 100:+.1f}%)" for r, (n, m) in g.iterrows()]
            lines.append(f"- **{label}**: " + ", ".join(parts))
    lines += [
        "",
        "## 주의",
        "",
        "- 종목 목록이 지금의 대형주라서, 그동안 망하거나 작아진 회사가 빠져 있습니다. "
        "실제보다 결과가 좋게 나오므로 상한선으로 보세요.",
        "- 소수점 주식으로 계산했습니다. 실제로는 1주 단위라 작은 계좌일수록 조금 달라집니다.",
        "- 세금(해외주식 양도세 22%)과 환율 변동은 넣지 않았습니다.",
        "",
    ]
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "swing_report.md").write_text("\n".join(lines), encoding="utf-8")
    for label, (strat, res) in results.items():
        res.trades.to_csv(out / f"trades_{strat.name}.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
