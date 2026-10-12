"""Backtest every swing candidate and write reports/swing_report_<universe>.md.

    python scripts/run_backtest.py
    python scripts/run_backtest.py --universe all --start 2010-01-01 --split 2018-01-01 --cost-bps 25
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
from swing.earnings_legends import EARNINGS, EPS_MIN, SALES_MIN, WIDE  # noqa: E402
from swing.fundamentals import growth_flags, load_facts  # noqa: E402
from swing.reversion import REVERSION  # noqa: E402
from swing.quality import QUALITY, GrossProfitability, profitability_table  # noqa: E402
from swing.announce import ANNOUNCE, announcement_tables, load_announcements  # noqa: E402
from swing.insider import EVENT_MIN, INSIDER, InsiderDollar, load_purchases, purchase_table  # noqa: E402
from swing.blend import blend  # noqa: E402
from swing.sewer import SEWER  # noqa: E402
from swing.pead import HOLD_DAYS, PEAD, PEAD_SET, SUE_MIN, sue_table  # noqa: E402
from swing.strategies import ALL, MomentumSwing  # noqa: E402

TITLES = {"legends": "전설 1그룹 규칙", "legends2": "전설 2그룹 규칙", "round2": "후보 전략",
          "refine": "와인스타인·다바스 다듬기", "legends_all": "전설 1·2그룹 규칙",
          "earnings": "전설 규칙 + 실적 조건", "earnings_wide": "오닐·미너비니 + 실적 조건",
          "pead": "실적 서프라이즈 후 추세(PEAD)",
          "reversion": "단기 과매도 반등",
          "quality": "총이익률(품질) 장기 보유",
          "announce": "실적 발표일 기준 서프라이즈",
          "insider": "내부자 매수 따라가기",
          "sewer": "하수구 돌파 (200일선 두 번째 돌파)",
          "blend": "여러 전략 합치기"}
SETS = {"legends": LEGENDS, "legends2": LEGENDS2, "round2": ALL, "refine": REFINED,
        "legends_all": LEGENDS + LEGENDS2, "earnings": EARNINGS, "earnings_wide": WIDE, "pead": PEAD_SET, "reversion": REVERSION, "quality": QUALITY, "announce": ANNOUNCE, "insider": INSIDER, "sewer": SEWER,
        "blend": [GrossProfitability, PEAD, InsiderDollar, MomentumSwing]}
RULE_CHAPTERS = {"legends": "12장", "legends2": "14장", "refine": "12·14장",
                 "legends_all": "12·14장", "earnings": "12·14·17장", "earnings_wide": "12·14·17장", "pead": "19장", "reversion": "20장", "quality": "19·22장", "announce": "19·23장", "insider": "24장", "blend": "19·22·24·25장", "sewer": "26장"}
from swing.universe import MARKET, UNIVERSE_LABELS, tickers  # noqa: E402


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
    cols = list(max(rows.values(), key=len).keys())          # mixes have no trade columns
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
    ap.add_argument("--cost-bps", type=float, default=25.0, help="편도 거래비용 (bp, 키움 미국주식 기본 수수료 0.25%%)")
    ap.add_argument("--universe", default="large", choices=sorted(UNIVERSE_LABELS),
                    help="large: 대형주, all: 대형주 + 중형주, small: 소형주, every: 전부, sp1500: S&P 1500")
    ap.add_argument("--set", default="legends", choices=sorted(SETS),
                    help="legends: 전설 1그룹(와인스타인·미너비니·본드·오닐), legends2: 2그룹(다바스·쿨라매기·래쉬키·켈), round2: 1·2차 후보, refine: 와인스타인·다바스 다듬기, legends_all: 1·2그룹 전부, earnings: 실적 조건 비교, earnings_wide: 오닐·미너비니 + 실적(넓은 종목용), pead: 실적 서프라이즈 후 추세, reversion: 단기 과매도 반등, quality: 총이익률 장기 보유, announce: 실적 발표일 기준 서프라이즈, insider: 내부자 매수 따라가기, blend: 여러 전략 합치기, sewer: 하수구 돌파(200일선 두 번째 돌파)")
    ap.add_argument("--low-cost-bps", type=float, default=None,
                    help="이 비용(편도 bp)으로도 다시 계산해 표를 덧붙임 (예: 키움 이벤트 수수료 7)")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    warmup = str(pd.Timestamp(args.start).year - 1) + "-01-01"   # 200-day averages need a year
    data = load_ohlc(tickers(args.universe) + [MARKET], start=warmup)
    split = pd.Timestamp(args.split)
    universe = [t for t in data["Close"].columns if t != MARKET]

    earn_lines = []
    if args.set in ("earnings", "earnings_wide"):
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
    if args.set == "quality":
        facts = load_facts(universe)
        gpa, gcovered = profitability_table(facts, data["Close"].index)
        if gcovered == 0:
            raise RuntimeError("SEC 실적 데이터를 하나도 받지 못했습니다. 접속이 막혔는지 확인하세요.")
        data["GPA"] = gpa
        data["SUE"], _ = sue_table(facts, data["Close"].index)
        since = gpa.index >= pd.Timestamp(args.start)
        earn_lines = [
            f"- 실적 데이터: 미국 증권거래위원회(SEC) 공시. {len(universe)}개 중 {gcovered}개 종목에서 "
            "총이익률을 계산했습니다(은행·보험처럼 매출원가가 없는 회사는 빠짐). 공시 다음 날부터 씁니다.",
            f"- 하루 평균 {gpa[since].notna().sum(axis=1).mean():.0f}개 종목의 총이익률을 알 수 있었습니다.",
        ]
    if args.set == "blend":
        idx = data["Close"].index
        facts = load_facts(universe)
        data["GPA"], gcovered = profitability_table(facts, idx)
        data["SUE"], scovered = sue_table(facts, idx)
        data["IBUY"] = purchase_table(load_purchases(idx[0].year, idx[-1]), idx, universe)
        if gcovered == 0 or scovered == 0 or data["IBUY"].to_numpy().sum() == 0:
            raise RuntimeError("SEC 데이터를 받지 못했습니다. 접속이 막혔는지 확인하세요.")
        earn_lines = [
            f"- SEC 공시: 총이익률 {gcovered}개, 실적 서프라이즈 {scovered}개, "
            f"내부자 매수 {int((data['IBUY'] > 0).any().sum())}개 종목.",
            "- 각 전략은 자기 장의 규칙 그대로 계좌의 정해진 몫에서 돌고, 매달 초 목표 비중으로 다시 맞춥니다. "
            "옮기는 돈에도 사고팔 때 거래비용을 매깁니다.",
        ]
    if args.set == "insider":
        idx = data["Close"].index
        buys = purchase_table(load_purchases(idx[0].year, idx[-1]), idx, universe)
        if buys.to_numpy().sum() == 0:
            raise RuntimeError("SEC 내부자 거래 데이터를 하나도 받지 못했습니다. 접속이 막혔는지 확인하세요.")
        data["IBUY"] = buys
        since = idx >= pd.Timestamp(args.start)
        years = since.sum() / 252
        b = buys[since]
        earn_lines = [
            "- 내부자 거래: SEC 내부자 거래 데이터(Form 4). 임원·이사의 장내 매수(P)만 셉니다. "
            "공시한 다음 날부터 씁니다.",
            f"- {len(universe)}개 중 {int((b > 0).any().sum())}개 종목에서 내부자 매수가 있었습니다. "
            f"매수가 공시된 날은 1년에 평균 {(b > 0).sum().sum() / years:.0f}번(종목×날짜), "
            f"그중 하루 {EVENT_MIN / 1000:.0f}천 달러 이상은 {(b >= EVENT_MIN).sum().sum() / years:.0f}번입니다.",
        ]
    if args.set == "announce":
        facts = load_facts(universe)
        ann, sue, cover = announcement_tables(facts, load_announcements(universe), data["Close"].index)
        if cover["announce"] == 0:
            raise RuntimeError("SEC 실적 발표(8-K) 데이터를 하나도 받지 못했습니다. 접속이 막혔는지 확인하세요.")
        data["ANN"], data["SUE_ANN"] = ann, sue
        since = ann.index >= pd.Timestamp(args.start)
        years = since.sum() / 252
        earn_lines = [
            f"- 실적 발표일: SEC에 낸 실적 발표 공시(8-K 2.02항). {len(universe)}개 중 {cover['announce']}개 종목, "
            f"1년에 평균 {ann[since].notna().sum().sum() / years:.0f}번의 발표를 찾았습니다.",
            f"- 그중 {cover['matched']}개 종목은 분기 EPS를 발표일에 연결해 서프라이즈(SUE)를 계산했습니다. "
            f"SUE {SUE_MIN:.0f} 이상 신호는 1년에 평균 {(sue[since] >= SUE_MIN).sum().sum() / years:.0f}번입니다.",
            "- 발표 반응일 다음 날 종가에 판단하고 그다음 날 시가에 삽니다. "
            f"보유 기간은 {HOLD_DAYS}거래일로 고정입니다.",
        ]
    if args.set == "pead":
        sue, covered = sue_table(load_facts(universe), data["Close"].index)
        if covered == 0:
            raise RuntimeError("SEC 실적 데이터를 하나도 받지 못했습니다. 접속이 막혔는지 확인하세요.")
        data["SUE"] = sue
        since = sue.index >= pd.Timestamp(args.start)
        per_year = (sue[since] >= SUE_MIN).sum().sum() / (since.sum() / 252)
        earn_lines = [
            f"- 실적 데이터: 미국 증권거래위원회(SEC) 공시. {len(universe)}개 중 {covered}개 종목에서 "
            "실적 서프라이즈(SUE)를 계산했습니다. 공시한 다음 날 종가에 판단하고 그다음 날 시가에 삽니다.",
            f"- 서프라이즈 {SUE_MIN:.0f} 이상 신호: 1년에 평균 {per_year:.0f}번 (전체 공시 중 "
            f"{(sue[since] >= SUE_MIN).sum().sum() / max(sue[since].notna().sum().sum(), 1) * 100:.0f}%)",
            f"- 보유 기간은 {HOLD_DAYS}거래일(약 3개월)로 고정입니다.",
        ]
    spy = data["Close"][MARKET]
    spy_ret = spy[spy.index >= pd.Timestamp(args.start)].pct_change().fillna(0.0)
    results = {}
    for cls in SETS[args.set]:
        strat = cls().prepare(data, market=MARKET)
        results[strat.label] = (strat, engine.run(strat, cost_bps=args.cost_bps, start=args.start))

    mixes = {}
    if args.set == "blend":
        parts = list(results)
        mixes = {"혼합 3종 (총이익률·실적 서프라이즈·내부자, 1/3씩)": {k: 1.0 for k in parts[:3]},
                 "혼합 4종 (위 3종 + 주간 모멘텀, 1/4씩)": {k: 1.0 for k in parts}}

    def mixed(runs: dict, cost: float) -> dict[str, pd.Series]:
        rets = {k: v.returns[v.returns.index >= pd.Timestamp(args.start)] for k, v in runs.items()}
        return {name: blend({k: rets[k] for k in w}, w, cost) for name, w in mixes.items()}

    mix_ret = mixed({k: v[1] for k, v in results.items()}, args.cost_bps)

    def period(mask_fn) -> dict[str, dict]:
        rows = {}
        for name, r in mix_ret.items():
            rows[name] = stats(r[mask_fn(r.index)])
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
    dbl, dbl_runs = {}, {}
    for label, (strat, _) in results.items():
        r = dbl_runs[label] = engine.run(strat, cost_bps=args.cost_bps * 2, start=args.start)
        m = r.returns.index >= split
        tr = r.trades[pd.to_datetime(r.trades["entry_date"]) >= split] if len(r.trades) else r.trades
        dbl[label] = stats(r.returns[m], tr, r.exposure[m])
    dbl = {**{name: stats(r[r.index >= split]) for name, r in mixed(dbl_runs, args.cost_bps * 2).items()},
           **dbl}

    # Optional: the same strategies at a lower cost (e.g. a discounted commission, limit orders).
    low_lines = []
    if args.low_cost_bps is not None:
        low_runs = {label: engine.run(strat, cost_bps=args.low_cost_bps, start=args.start)
                    for label, (strat, _) in results.items()}

        def low_period(mask_fn) -> dict[str, dict]:
            rows = {}
            for label, r in low_runs.items():
                tr = r.trades
                tm = mask_fn(pd.to_datetime(tr["entry_date"])) if len(tr) else []
                rows[label] = stats(r.returns[mask_fn(r.returns.index)], tr[tm] if len(tr) else tr,
                                    r.exposure[mask_fn(r.exposure.index)])
            rows["(비교) S&P500 그냥 보유"] = stats(spy_ret[mask_fn(spy_ret.index)])
            return rows

        low_lines = [
            f"## 거래비용 {args.low_cost_bps:g}bp일 때 (키움 이벤트 수수료 0.07% 수준)",
            "",
            "달러를 들고 지정가로 거래하면 환전 비용과 미끄러짐이 거의 없어 비용이 이 정도까지 내려갑니다.",
            "",
            "검증 구간:",
            "",
            table(low_period(lambda ix: ix >= split)),
            "",
            "학습 구간:",
            "",
            table(low_period(lambda ix: (ix >= pd.Timestamp(args.start)) & (ix < split))),
            "",
        ]

    mom = engine.run(MomentumSwing().prepare(data, market=MARKET), cost_bps=args.cost_bps,
                     start=args.start).returns
    corr = correlations({**mix_ret, **{k: v[1].returns for k, v in results.items()}},
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
    for name in mixes:
        lines.append(f"- **{name}**: 아래 전략들을 한 계좌에 나눠 담고 매달 초 같은 비중으로 다시 맞춥니다.")
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
        *low_lines,
        "## 다른 전략과 같이 움직이는 정도 (검증 구간 월수익률 상관계수)",
        "",
        "1에 가까우면 같이 오르고 같이 내립니다. 기존 앱과 함께 돌린다면 낮을수록 위험이 분산됩니다. "
        "'주간 모멘텀(이 앱)'은 기존 퀀트 앱의 ETF 모멘텀과 같은 전략은 아니고 비슷한 성격의 참고용입니다.",
        "",
        corr,
        "",
        "## 연도별 수익률",
        "",
        yearly({**mix_ret, **{k: v[1].returns[v[1].returns.index >= pd.Timestamp(args.start)]
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
    if args.set == "sewer":
        # The deep-drop rule picks stocks that crashed and came back; list who made the money
        # so we can check how many only joined today's index after recovering.
        lines += ["", "## 계좌 기여가 가장 큰 거래 15건", ""]
        for label, (_, res) in results.items():
            if not len(res.trades):
                continue
            tr = res.trades.assign(contrib=res.trades["return"] * res.trades["weight"]
                                   * res.trades["part"]).sort_values("contrib", ascending=False)
            share = tr["contrib"].head(10).sum() / tr["contrib"].sum() if tr["contrib"].sum() > 0 else float("nan")
            lines += [f"**{label}** (상위 10건이 계좌 기여 합계의 {share * 100:.0f}%)", "",
                      "| 종목 | 매수일 | 매도일 | 수익률 | 계좌 기여 |", "|---|---|---|---|---|"]
            lines += [f"| {x.ticker} | {x.entry_date} | {x.exit_date} | {x['return'] * 100:+.0f}% | "
                      f"{x.contrib * 100:+.1f}%p |" for _, x in tr.head(15).iterrows()]
            lines.append("")
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
