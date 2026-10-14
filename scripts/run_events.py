"""Backtest one of rulebooks 12-21 (event trades) and write reports/events_<set>.md.

    python scripts/run_events.py --set preearn                  # Yahoo prices, today's S&P 500 + 400
    python scripts/run_events.py --set taxloss --pit prep_daily # S&P 500 members of the day, delisted included

Each rule's trades are counted twice: every signal (the papers' way, no
account limits) and an account of equal slots (see swing/events.py).
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swing import bulkowski, divmonth, elder, epgap, nonews, nr7, preearn, spindex, taxloss  # noqa: E402
from swing.events import (GATE_GROSS, GATE_N, Panel, account_stats, frame, md_table, periods_of,  # noqa: E402
                          simulate, top_trades, trade_stats, yearly_trades)
from swing.universe import MARKET  # noqa: E402


@dataclass
class Spec:
    title: str
    rulebook: str
    start: str
    periods: list
    slots: int
    needs: tuple
    pit_ok: bool
    rules: list
    extra: object = None


def spec_of(mod) -> Spec:
    return Spec(mod.TITLE, mod.RULEBOOK, mod.START, mod.PERIODS, mod.SLOTS, mod.NEEDS, mod.PIT_OK, mod.RULES,
                getattr(mod, "extra", None))


SPECS = {"preearn": spec_of(preearn), "epgap": spec_of(epgap), "nonews": spec_of(nonews), "elder": spec_of(elder),
         "divmonth": spec_of(divmonth), "taxloss": spec_of(taxloss), "nr7": spec_of(nr7),
         "bulkowski": spec_of(bulkowski),
         "spadd": Spec("S&P 500 편입 직전 매수 (지수 편입 효과)", "18", spindex.START, spindex.PERIODS, spindex.SLOTS,
                       spindex.NEEDS, True, spindex.ADD_RULES, spindex.extra_add),
         "spdel": Spec("S&P 500 퇴출 종목 반등", "19", spindex.START, spindex.PERIODS, spindex.SLOTS, spindex.NEEDS,
                       True, spindex.DEL_RULES, spindex.extra_del)}
PIT_START = "2018-01-02"


def yahoo_name(raw: str) -> str | None:
    """fja05680 list name -> Yahoo ticker; names with a -YYYYMM suffix were reused later and are dropped."""
    raw = raw.strip()
    if not raw or re.search(r"-\d{6}$", raw):
        return None
    return raw.replace(".", "-")


def sp_member_frame(dates: pd.DatetimeIndex, columns: list[str], snaps: pd.DataFrame) -> np.ndarray:
    """S&P 500 membership for the price columns (each day uses the latest list on or before it)."""
    from swing.pit import norm

    col = {norm(t): j for j, t in enumerate(columns)}
    snaps = snaps.sort_values("date")
    out = np.zeros((len(dates), len(columns)), bool)
    starts = [int(dates.searchsorted(d)) for d in snaps["date"]] + [len(dates)]
    for j, lst in enumerate(snaps["tickers"]):
        a, b = (0 if j == 0 else starts[j]), starts[j + 1]
        if b <= a:
            continue
        ks = [col[norm(x)] for x in str(lst).split(",") if yahoo_name(x) and norm(x) in col]
        out[a:b, ks] = True
    return out


def load_main(name: str, sp: Spec) -> tuple[dict, list[str], dict, list[str]]:
    from swing.data import load_ohlc
    from swing.universe import sp_info, tickers

    warmup = f"{int(sp.start[:4]) - 1}-01-01"
    notes, ext = [], {}
    if name in ("spadd", "spdel"):
        from swing.pit import sp500_snapshots

        snaps = sp500_snapshots()
        snaps = snaps[snaps["date"] >= pd.Timestamp(warmup)]
        names = sorted({y for lst in snaps["tickers"] for x in str(lst).split(",") if (y := yahoo_name(x))})
        data = load_ohlc(names + [MARKET], start=warmup)
        cols = [t for t in data["Close"].columns if t != MARKET]
        ext["sp_member"] = sp_member_frame(data["Close"].index, cols, snaps)
        notes.append(f"- 종목: {warmup[:4]}년 뒤 한 번이라도 S&P 500에 들었던 {len(names)}개 이름 중 야후에서 가격을 받은 "
                     f"{len(cols)}개 (상장폐지·인수된 종목은 대부분 빠짐 → 아래 '그때 S&P 500 점검'이 판정 기준).")
        return data, cols, ext, notes
    data = load_ohlc(tickers("sp1500") + [MARKET], start=warmup)
    info = data["INFO"] = sp_info()
    cols = [t for t in data["Close"].columns if t != MARKET]
    ext["INFO"] = info
    members = [t for t in cols if info.reindex([t])["index"].isin((500, 400)).iloc[0]]
    notes.append(f"- 종목: 지금 S&P 500 + 400 구성 종목 {len(members)}개 (야후 가격, 배당 보정). 그동안 빠진 종목은 없음 [근사].")
    return data, cols, ext, notes


def load_pit(daily: str) -> tuple[dict, list[str], dict, list[str]]:
    from swing.pit import load_daily, norm, snapshot_exits, snapshot_members, sp500_snapshots

    paths = sorted(Path(daily).rglob("daily_*.parquet"))
    first_day = pd.to_datetime(pd.read_parquet(paths[0], columns=["d"])["d"]).min()
    snaps = sp500_snapshots()
    names = set(snapshot_members(pd.DatetimeIndex([first_day]), snaps).columns)
    names |= {norm(x) for x in str(snaps.sort_values("date")["tickers"].iloc[-1]).split(",")}
    left = {t: d for t, d in snapshot_exits(snaps).items() if d >= first_day}
    names |= set(left)
    wide, info = load_daily(paths, names | {"SPY"}, left)
    member = snapshot_members(wide["Close"].index, snaps)
    member = member.reindex(columns=wide["Close"].columns, fill_value=False).drop(columns=["SPY"], errors="ignore")
    cols = [t for t in wide["Close"].columns if t != "SPY"]
    notes = [f"- 종목: 그날그날 S&P 500 구성 종목(github.com/fja05680/sp500), 상장폐지 종목 포함 {len(cols)}개. "
             f"가격은 무료 1분봉(mito0o852/OHLCV-1m)을 하루로 묶은 것, 배당 보정 없음, 분할은 밤사이 갭으로 되돌림 "
             f"({info['splits']}건) [근사]. S&P 400은 구성 이력이 없어 뺌."]
    return wide, cols, {"MEMBER": member}, notes


def build_ext(sp: Spec, panel: Panel, ext: dict, pit: bool = False) -> list[str]:
    notes = []
    if "earnings" in sp.needs or "news" in sp.needs:
        from swing.sec_filings import earnings_days, load_submissions, news_days

        want = [t for k, t in enumerate(panel.tickers) if panel.member[:, k].any()]
        since = "2003-01-01" if "earnings" in sp.needs else "2009-06-01"
        subs = load_submissions(want, since=since)
        if not any(subs.values()):
            raise RuntimeError("SEC 공시 목록을 하나도 받지 못했습니다. 접속이 막혔는지 확인하세요.")
        missing = [k for k, t in enumerate(panel.tickers) if not subs.get(t)]
        if pit:
            panel.elig[:, missing] = False
            gone = sum(1 for k in missing if panel.member[:, k].any())
            notes.append(f"- SEC 회사 번호를 못 찾은 종목(대부분 상장폐지·이름 바뀐 종목) {gone}개는 뺐습니다. 이 점검은 "
                         "'나중에 지수에 들어온 종목을 미리 고르는' 치우침만 없애고, 상장폐지 종목은 여전히 빠져 있습니다.")
        if "earnings" in sp.needs:
            ext["earnings"] = earnings_days(subs, panel.tickers, panel.dates)
            got = sum(1 for v in ext["earnings"].values() if len(v))
            per = sum(len(v) for v in ext["earnings"].values()) / max(got, 1)
            notes.append(f"- 실적 발표일: SEC 8-K 2.02항(2004년 8월 전은 12항) 접수 시각. {len(want)}개 중 {got}개 종목, "
                         f"종목당 평균 {per:.0f}번. 16:00 뒤 접수는 다음 거래일이 반응일.")
        if "news" in sp.needs:
            ext["news"] = news_days(subs, panel.tickers, panel.dates)
            got = sum(1 for v in ext["news"].values() if len(v))
            per = sum(len(v) for v in ext["news"].values()) / max(got, 1) / max(len(panel.dates) / 252, 1)
            notes.append(f"- 공시(뉴스 대신): 8-K·8-K/A·10-Q·10-K 접수 시각. {got}개 종목, 종목당 1년 평균 {per:.0f}건. "
                         "9:30 전 접수는 그날 시가부터, 그 뒤는 다음 날 시가부터 알려진 것으로 봄.")
    if "dividends" in sp.needs:
        from swing.dividends import load_dividends

        want = [t for k, t in enumerate(panel.tickers) if panel.member[:, k].any()]
        div, raw = load_dividends(want, start=str(panel.dates[0].date()))
        amount = np.zeros(panel.C.shape)
        for t in div.columns:
            s = div[t]
            s = s[s > 0]
            if t not in panel.col or s.empty:
                continue
            pos = panel.dates.searchsorted(pd.DatetimeIndex(s.index).tz_localize(None))
            ok = pos < len(panel.dates)
            np.add.at(amount[:, panel.col[t]], pos[ok], s.to_numpy()[ok])
        rawc = raw.copy()
        rawc.index = pd.DatetimeIndex(rawc.index).tz_localize(None)
        ext["div_amount"] = amount
        ext["raw_close"] = rawc.reindex(index=panel.dates, columns=panel.tickers).ffill().to_numpy(float)
        ext["ex_days"] = {k: np.flatnonzero(amount[:, k] > 0) for k in range(amount.shape[1])}
        got = sum(1 for v in ext["ex_days"].values() if len(v))
        notes.append(f"- 배당: 야후 배당 기록(배당락일·금액). {len(want)}개 중 {got}개 종목이 배당을 줬음. "
                     "배당수익률은 배당 보정하지 않은 종가로 계산.")
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SPECS))
    ap.add_argument("--cost-bps", type=float, default=30.0)
    ap.add_argument("--split", default="2018-01-01")
    ap.add_argument("--pit", default=None, help="무료 1분봉 하루 표 폴더 (그때 S&P 500 점검)")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    sp = SPECS[a.set]
    split = pd.Timestamp(a.split)
    if a.pit:
        if not (sp.pit_ok or a.set in ("preearn", "epgap", "nonews")):
            raise SystemExit(f"{a.set}: SEC·배당 자료가 상장폐지 종목엔 없어 그때 S&P 500 점검을 하지 않습니다.")
        data, cols, ext, notes = load_pit(a.pit)
        panel = Panel(data, MARKET, member=ext.pop("MEMBER"))
        if "sp_member" in sp.needs:
            ext["sp_member"] = panel.member.copy()
            if a.set in ("spadd", "spdel"):
                panel.elig = panel.liquid
        start = PIT_START
        bounds = ["2018"]
    else:
        data, cols, ext, notes = load_main(a.set, sp)
        panel = Panel(data, MARKET) if a.set not in ("spadd", "spdel") else Panel(
            {k: v for k, v in data.items() if k != "INFO"}, MARKET)
        start = sp.start
        bounds = sp.periods
    notes += build_ext(sp, panel, ext, pit=bool(a.pit))
    end = panel.dates[-1]
    periods = periods_of(bounds, end)
    cost = a.cost_bps / 10_000
    spy = pd.Series(panel.spy, index=panel.dates)
    s0 = pd.Timestamp(start)

    frames, accts, accts2 = {}, {}, {}
    for name, label, fn in sp.rules:
        trades = [t for t in fn(panel, ext) if panel.dates[t.ei] >= s0]
        frames[name] = frame(trades, panel, a.cost_bps)
        accts[name] = simulate(trades, panel, a.cost_bps, sp.slots, start)
        accts2[name] = simulate(trades, panel, a.cost_bps * 2, sp.slots, start)
        print(f"{name}: {len(trades)} trades", flush=True)

    def cut(df, lo, hi):
        d = pd.to_datetime(df["entry_date"])
        return df[(d >= lo) & (d < hi)]

    tag = " — 그때 S&P 500 점검 (상장폐지 포함)" if a.pit else ""
    lines = [f"# 규칙서 {sp.rulebook}: {sp.title}{tag}", "",
             f"- 기간: {start} ~ {end.date()}, 판정 구간 {split.date()} ~. 편도 비용 {a.cost_bps / 100:.2f}% "
             f"(수수료 0.25% + 미끄러짐 0.05%).",
             *notes,
             f"- 주가 ≥ $10, 20일 평균 거래대금 ≥ $20M (규칙서 19는 $5·$10M).",
             f"- 거래당 숫자는 모든 신호를 다 산 경우(논문 방식). 계좌는 {sp.slots}칸 × {100 / sp.slots:.0f}%, "
             "같은 날 신호가 많으면 점수 순, 빈칸이 없으면 못 삼.",
             f"- 관문: 판정 구간 비용 전 거래당 ≥ +{GATE_GROSS * 100:.1f}%, 거래 {GATE_N}번 이상.", "",
             "## 규칙", ""]
    lines += [f"- **{name}**: {label}" for name, label, _ in sp.rules]
    for pname, lo, hi in periods:
        rows = {name: trade_stats(cut(df, lo, hi)) for name, df in frames.items()}
        lines += ["", f"## 거래당 숫자 {pname}" + (" (판정)" if lo >= split else ""), "", md_table(rows)]
    for pname, lo, hi in ([("개발 구간", s0, split)] if s0 < split else []) + [("판정 구간", split, None)]:
        rows = {name: account_stats(acc, spy, lo, hi) for name, acc in accts.items()}
        lines += ["", f"## 계좌 ({pname})", "", md_table(rows)]
    rows = {}
    for name, df in frames.items():
        x = cut(df, split, end + pd.Timedelta(days=1)).copy()
        x["net"] = (1 + x["gross"]) * (1 - 2 * cost) / (1 + 2 * cost) - 1
        rows[name] = {**{k: v for k, v in trade_stats(x).items() if k in ("거래수", "비용 후 평균", "승률")},
                      **{k: v for k, v in account_stats(accts2[name], spy, split).items() if k in ("계좌 연수익", "최대낙폭")}}
    lines += ["", f"## 비용 2배({a.cost_bps * 2 / 100:.2f}%)일 때 판정 구간", "", md_table(rows)]
    first = sp.rules[0][0]
    lines += ["", f"## 해마다 ({first}, 모든 신호)", "", yearly_trades(frames[first])]
    lines += ["", "## 판정 구간 매도 이유와 가장 좋은 거래 10건", ""]
    for name, df in frames.items():
        x = cut(df, split, end + pd.Timedelta(days=1))
        if x.empty:
            continue
        g = x.groupby("reason")["net"].agg(["size", "mean"])
        lines.append(f"- **{name}** 매도 이유: " + ", ".join(f"{r} {int(v['size'])}번({v['mean'] * 100:+.1f}%)"
                                                         for r, v in g.iterrows()))
        lines.append(f"  - 상위 10건: {top_trades(x)}")
    if sp.extra is not None:
        lines += ["", *sp.extra(panel, ext, {n: df for n, df in frames.items()}, split)]
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    suffix = f"{a.set}{'_pit' if a.pit else ''}"
    (out / f"events_{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, df in frames.items():
        df.to_csv(out / f"events_trades_{suffix}_{name}.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
