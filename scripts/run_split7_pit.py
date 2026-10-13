"""규칙서 10 생존 편향 점검: 그때의 S&P 500 종목(상장폐지 포함) vs 지금 남아 있는 종목.

가격은 무료 1분봉(mito0o852/OHLCV-1m)을 하루로 묶은 표(daily_YYYY.parquet, 데이트레이딩 워크플로가 만든 것)를 씁니다.
두 계산 모두 같은 가격 자료에서 대상 종목만 다르게 해서, 차이가 곧 생존 편향입니다.

python scripts/run_split7_pit.py --daily prep_daily --out reports/split7_pit.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swing import engine, metrics  # noqa: E402
from swing.pit import load_daily, membership, sp500_tables  # noqa: E402
from swing.split7 import E1, E5, F1, F4, F8  # noqa: E402

START = "2018-01-02"
VARIANTS = [E1, E5, F1, F4, F8]


def line(name: str, res: engine.Result) -> str:
    ret = res.returns[res.returns.index >= START]
    tr = res.trades[pd.to_datetime(res.trades["exit_date"]) >= pd.Timestamp(START)]
    gone = tr[tr["reason"].str.startswith("거래 중단")]
    held = res.open_positions
    return (f"| {name} | {metrics.cagr(ret) * 100:.1f}% | {metrics.max_drawdown(ret) * 100:.1f}% | {len(tr)} | "
            f"{(tr['return'] > 0).mean() * 100:.1f}% | {tr['return'].mean() * 100:+.1f}% | "
            f"{len(gone)} ({gone['return'].mean() * 100 if len(gone) else 0:+.0f}%) | {len(held)} | "
            f"{res.exposure[res.exposure.index >= START].mean() * 100:.0f}% |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", default="prep_daily")
    ap.add_argument("--out", default="reports/split7_pit.md")
    ap.add_argument("--cost-bps", type=float, default=30)
    a = ap.parse_args()
    paths = sorted(Path(a.daily).rglob("daily_*.parquet"))
    current, changes = sp500_tables()
    names = set(current) | set(changes["added"]) | set(changes["removed"]) - {""}
    probe = pd.read_parquet(paths[0], columns=["d"])
    first_day = pd.to_datetime(probe["d"]).min()
    rem = changes[(changes["date"] >= first_day) & (changes["removed"] != "")]
    left = dict(zip(rem["removed"], rem["date"]))
    wide, info = load_daily(paths, names | {"SPY"}, left)
    dates = wide["Close"].index
    pit = membership(dates, current, changes).reindex(columns=wide["Close"].columns, fill_value=False)
    pit = pit.drop(columns=["SPY"], errors="ignore")
    now = pd.DataFrame(False, index=dates, columns=pit.columns)
    now.loc[:, [t for t in current if t in now.columns]] = True
    ever = pit.loc[pit.index >= START].any()
    removed_names = sorted(set(ever[ever].index) - set(current))
    have = [t for t in removed_names if wide["Close"][t].notna().any()]
    lines = [
        "# 규칙서 10 생존 편향 점검: 그때의 S&P 500 (상장폐지 포함) vs 지금 S&P 500",
        "",
        f"- 기간: {START} ~ {dates[-1].date()} (가격 자료가 2026년 3월까지라 여기서 끝납니다). 편도 비용 {a.cost_bps / 100:.2f}%.",
        "- 가격: 무료 1분봉(mito0o852/OHLCV-1m, 상장폐지 종목 포함)을 정규장 하루로 묶음. 배당 보정 없음, 주식분할은 "
        f"밤사이 갭이 분할 비율과 맞으면 되돌림 ({info['splits']}건 찾음) [근사].",
        f"- 그때 구성 종목: 위키백과 S&P 500 현재 목록 + 변경 표({len(changes)}건)를 거꾸로 적용. 검증 구간에 한 번이라도 들어 "
        f"있던 종목 {int(ever.sum())}개 중 지금은 빠진 종목 {len(removed_names)}개, 그중 가격이 있는 종목 {len(have)}개.",
        f"- 빠진 뒤 10거래일 넘게 가격이 끊기면 그 뒤를 자름({info['cut']}개). 들고 있던 종목의 가격이 10거래일 넘게 없으면 "
        "마지막 가격에 판 것으로 봄('거래 중단').",
        "- 왼쪽 'PIT'는 그날 S&P 500에 들어 있던 종목만 새로 삼(빠진 뒤에도 들고 있던 것은 계속 관리). 오른쪽 '지금'은 "
        "지금 S&P 500 종목만 삼(보통 백테스트 방식). 가격 자료는 같습니다.",
        "",
        "| 규칙 | 연수익률 | 최대낙폭 | 거래수 | 승률 | 평균 거래수익 | 거래 중단 매도 (평균) | 끝까지 못 판 종목 | 평균 투자비중 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    spy = wide["Close"]["SPY"].pct_change().fillna(0.0)
    spy = spy[spy.index >= START]
    data = {**wide}
    notes = []
    for cls in VARIANTS:
        for tag, member in (("PIT", pit), ("지금", now)):
            strat = cls().prepare({**data, "MEMBER": member}, market="SPY")
            res = engine.run(strat, cost_bps=a.cost_bps)
            lines.append(line(f"{cls.label.split()[0]} {tag}", res))
            if tag == "PIT":
                tr = res.trades.sort_values("return").head(8)
                worst = ", ".join(f"{r.ticker} {r['return'] * 100:+.0f}% ({r.reason})" for _, r in tr.iterrows())
                notes.append(f"- {cls.label.split()[0]} PIT 가장 나쁜 거래: {worst}")
                held = sorted(res.open_positions, key=lambda p: p["entry_date"])[:8]
                if held:
                    notes.append(f"- {cls.label.split()[0]} PIT 끝까지 못 판 종목(오래된 순 8개): "
                                 + ", ".join(f"{p['ticker']} ({p['entry_date']}~)" for p in held))
    lines.append(f"| S&P500 보유 | {metrics.cagr(spy) * 100:.1f}% | {metrics.max_drawdown(spy) * 100:.1f}% | - | - | - | - | - | - |")
    lines += ["", *notes, "", "- 규칙 설명은 본 보고서(규칙서 10)와 같습니다. E1·E5·F1·F4·F8만 돌렸습니다(시장 필터를 쓰는 변형은 "
              "SPY 200일선 계산에 필요한 과거 자료가 짧아서 뺌).",
              "- 지금 빠진 종목 중 가격이 있는 것: " + ", ".join(have[:60]) + (" …" if len(have) > 60 else "")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
