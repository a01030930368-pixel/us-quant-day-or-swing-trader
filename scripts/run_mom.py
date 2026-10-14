"""b10~b15 데이트레이딩 백테스트 실행 (급등주·갭 종목 장중 매매): prep_mom 폴더 -> 규칙서별 결과표(마크다운).

    python scripts/run_mom.py --prep prep_mom --rule b10 --out reports/b10
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daytrade import mom, portfolio  # noqa: E402
from run_b03 import earnings_days  # noqa: E402

FEE, SLIP, SLIP_LOW_PX = 0.0025, 0.001, 0.003  # 5달러 미만은 미끄러짐 0.3%
EXTRA_B13 = 0.003
GATE, GATE_LOW_PX, MIN_N = 0.009, 0.013, 200
MAX_W, MAX_POS, TOP_DAY = 0.20, 5, 5
RECENT = 2025


def load(prep, kind):
    fs = sorted(glob.glob(os.path.join(prep, f"{kind}_*.parquet")))
    return pd.concat([pq.read_table(f).to_pandas() for f in fs], ignore_index=True) if fs else pd.DataFrame()


def rng(r):
    return (r.pc - r.pl) / (r.ph - r.pl) if r.ph > r.pl else 1.0


# 규칙서별: kind, 제목, 후보 걸러내기(row -> bool, 회의·실적 정보는 row에 있음), 변형 목록 [(코드, 이름, 함수(row, day, frac) -> 거래)]
def rules():
    R = {}
    R["b10"] = ("up", "급등주 그날 고가 돌파", lambda r: True, [
        ("H1", "H1 기본 (10시 +15%, 상대 거래량 5배, 고가 재돌파 + 거래량 2배, 2R·9EMA)", lambda r, d, f: mom.b10(d, r.pc, r.adv_sh, f)),
        ("H2-10", "H2 10시 상승 기준 +10%", lambda r, d, f: mom.b10(d, r.pc, r.adv_sh, f, up=0.10)),
        ("H2-30", "H2 10시 상승 기준 +30%", lambda r, d, f: mom.b10(d, r.pc, r.adv_sh, f, up=0.30)),
        ("H3", "H3 익절 없이 (9EMA 이탈 또는 15:55)", lambda r, d, f: mom.b10(d, r.pc, r.adv_sh, f, target=False)),
        ("H4", "H4 매수 10:00~12:00만", lambda r, d, f: mom.b10(d, r.pc, r.adv_sh, f, t_hi=150)),
        ("H5", "H5 거래량 조건 없이 고가 넘는 순간 매수", lambda r, d, f: mom.b10(d, r.pc, r.adv_sh, f, at_touch=True)),
        ("H6", "H6 주가 5~50달러만", lambda r, d, f: _px(mom.b10(d, r.pc, r.adv_sh, f), 5))])
    base11 = lambda r, up=0.20: r.pc >= r.ppc * (1 + up) and r.pv >= 5 * r.padv_sh and 1 <= r.pc <= 50 and r.pdv >= 1e7  # noqa: E731
    strong = lambda r: rng(r) >= 0.7 and r.o >= r.pc  # noqa: E731
    line11 = lambda r: max(r.ph, r.pm_high) if not np.isnan(r.pm_high) else r.ph  # noqa: E731
    R["b11"] = ("d2", "둘째 날 이어가기", lambda r: base11(r, 0.10), [
        ("S1", "S1 기본 (전날 +20%·거래량 5배·강한 마감, 갭 하락 아님, 기준선 돌파)", lambda r, d, f: mom.b11(d, line11(r)) if base11(r) and strong(r) else None),
        ("S2", "S2 강한 마감·갭 하락 조건 없이", lambda r, d, f: mom.b11(d, line11(r)) if base11(r) else None),
        ("S3-10", "S3 전날 +10% 이상", lambda r, d, f: mom.b11(d, line11(r)) if base11(r, 0.10) and strong(r) else None),
        ("S3-40", "S3 전날 +40% 이상", lambda r, d, f: mom.b11(d, line11(r)) if base11(r, 0.40) and strong(r) else None),
        ("S4-e", "S4 전날이 실적 발표일인 것만", lambda r, d, f: mom.b11(d, line11(r)) if base11(r) and strong(r) and r.earn_prev else None),
        ("S4-x", "S4 실적일 제외", lambda r, d, f: mom.b11(d, line11(r)) if base11(r) and strong(r) and not r.earn_prev else None),
        ("S5", "S5 익절 없이 (9EMA 이탈 또는 15:55)", lambda r, d, f: mom.b11(d, line11(r), target=False) if base11(r) and strong(r) else None),
        ("S6", "S6 기준선을 전날 고가만", lambda r, d, f: mom.b11(d, r.ph) if base11(r) and strong(r) else None)])
    R["b12"] = ("up", "첫 눌림 VWAP 반등", lambda r: True, [
        ("W1", "W1 기본 (+10%, 첫 VWAP 눌림 후 반등, 손절 눌림 저점/-2%, 익절 +5%/그날 고가)", lambda r, d, f: mom.b12(d, r.pc, r.adv_sh, f)),
        ("W2", "W2 +20% 이상만", lambda r, d, f: mom.b12(d, r.pc, r.adv_sh, f, up=0.20)),
        ("W3", "W3 VWAP 대신 5분봉 9EMA 눌림", lambda r, d, f: mom.b12(d, r.pc, r.adv_sh, f, line="ema")),
        ("W4", "W4 손절 눌림 저점만 (-2% 상한 없음)", lambda r, d, f: mom.b12(d, r.pc, r.adv_sh, f, cap2=False)),
        ("W5", "W5 익절 없이 15:55까지", lambda r, d, f: mom.b12(d, r.pc, r.adv_sh, f, target=False)),
        ("W6", "W6 매수 9:45~11:30만", lambda r, d, f: mom.b12(d, r.pc, r.adv_sh, f, t_hi=120))])
    R["b13"] = ("up", "불 플래그", lambda r: r.pc <= 20, [
        ("L1", "L1 기본 (1~20달러, +20%, 깃대 10봉 +5%, 깃발 2~6봉, 10분 시간 손절)", lambda r, d, f: mom.b13(d, r.pc, r.adv_sh, f)),
        ("L2", "L2 깃발 1~3봉 (마이크로 풀백)", lambda r, d, f: mom.b13(d, r.pc, r.adv_sh, f, flag_len=(1, 3))),
        ("L3", "L3 깃대 +8% 이상", lambda r, d, f: mom.b13(d, r.pc, r.adv_sh, f, pole_min=0.08)),
        ("L4", "L4 시간 손절 없이 (2R·깃대 길이·15:55)", lambda r, d, f: mom.b13(d, r.pc, r.adv_sh, f, time_stop=None)),
        ("L5", "L5 주가 5~20달러만", lambda r, d, f: mom.b13(d, r.pc, r.adv_sh, f, lo_px=5.0)),
        ("L6", "L6 9:30~10:30만", lambda r, d, f: mom.b13(d, r.pc, r.adv_sh, f, t_hi=60))])
    base14 = lambda r, gap=0.05: (r.earn_prev and r.po >= r.ppc * (1 + gap) and r.pc >= r.po and rng(r) >= 0.75  # noqa: E731
                                  and r.pv >= 3 * r.padv_sh and r.pc >= 5 and r.o >= r.pc * 0.97)
    R["b14"] = ("ear", "실적 둘째 날", lambda r: base14(r), [
        ("A1", "A1 기본 (실적 갭 +5%·강한 마감·대형주 제외, 전날 고가/첫 15분 고가 중 낮은 쪽 돌파)", lambda r, d, f: mom.b14(d, _line14(r, d)) if r.adv_rank > 500 else None),
        ("A2", "A2 갭 +10% 이상", lambda r, d, f: mom.b14(d, _line14(r, d)) if r.adv_rank > 500 and base14(r, 0.10) else None),
        ("A3-all", "A3 대형주 포함", lambda r, d, f: mom.b14(d, _line14(r, d))),
        ("A3-big", "A3 대형주(평소 거래대금 상위 500)만", lambda r, d, f: mom.b14(d, _line14(r, d)) if r.adv_rank <= 500 else None),
        ("A4", "A4 기준선을 전날 고가만", lambda r, d, f: mom.b14(d, r.ph) if r.adv_rank > 500 else None),
        ("A5", "A5 (비교용, 스윙) A1 매수 후 3거래일째 종가까지", lambda r, d, f: _swing(r, mom.b14(d, _line14(r, d))) if r.adv_rank > 500 else None),
        ("A6", "A6 손절 없이 15:55까지", lambda r, d, f: mom.b14(d, _line14(r, d), use_stop=False) if r.adv_rank > 500 else None)])
    R["b15"] = ("r2g", "레드 투 그린", lambda r: True, [
        ("G1", "G1 기본 (-2~-10% 갭 하락, 거래량 1.5배로 전날 종가 회복, 손절 그날 저가, 익절 전날 고가/2R 중 먼 쪽)", lambda r, d, f: mom.b15(d, r.pc, r.ph, r.adv_sh, f)),
        ("G2", "G2 보수적: 회복 뒤 전날 종가 근처까지 눌렸다 다시 위로", lambda r, d, f: mom.b15(d, r.pc, r.ph, r.adv_sh, f, conservative=True)),
        ("G3", "G3 손절을 돌파 1분봉 저가로", lambda r, d, f: mom.b15(d, r.pc, r.ph, r.adv_sh, f, bar_stop=True)),
        ("G4", "G4 거래량 기준 2배", lambda r, d, f: mom.b15(d, r.pc, r.ph, r.adv_sh, f, vol_mult=2.0)),
        ("G5-e", "G5 실적 발표일만", lambda r, d, f: mom.b15(d, r.pc, r.ph, r.adv_sh, f) if r.earn else None),
        ("G5-x", "G5 실적일 제외", lambda r, d, f: mom.b15(d, r.pc, r.ph, r.adv_sh, f) if not r.earn else None),
        ("G6", "G6 익절 없이 15:55까지", lambda r, d, f: mom.b15(d, r.pc, r.ph, r.adv_sh, f, target=False))])
    return R


def _px(t, lo):
    return t if t and t["entry_px"] >= lo else None


def _line14(r, d):
    first15 = d["h"][d["mi"] < 15]
    return min(r.ph, first15.max()) if len(first15) else r.ph


SWING = {}  # (ticker, d) -> 3거래일째 종가 (main에서 채움)


def _swing(r, t):
    if not t or t["reason"] == "손절":
        return t
    c3 = SWING.get((r.ticker, r.d))
    if c3 is None:
        return None
    return dict(t, exit_mi=-1, exit_px=c3, gross=c3 / t["entry_px"] - 1, reason="3일째 종가")


def pct(x, nd=1):
    return "" if pd.isna(x) else f"{x * 100:.{nd}f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_mom")
    ap.add_argument("--rule", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"reports/{a.rule}"
    os.makedirs(out, exist_ok=True)
    kind, title, keep, variants = rules()[a.rule]
    spy = load(a.prep, "spy").drop_duplicates("d").assign(d=lambda x: pd.to_datetime(x.d)).set_index("d").c.sort_index()
    days = spy.index
    prev_day = pd.Series(days[:-1], index=days[1:])
    earn = earnings_days(a.prep, days)
    if earn is None:
        raise SystemExit("SEC 실적 자료(earnings.parquet)가 없어 돌릴 수 없음 (회사 주식 구분에 필요)")
    earn["d"] = pd.to_datetime(earn.d)
    companies = set(earn.ticker)
    ekeys = set(zip(earn.ticker, earn.d))
    frac = mom.relvol_curve(load(a.prep, "prof"))
    feats = load(a.prep, "feats")
    feats["d"] = pd.to_datetime(feats.d)
    feats = feats[feats[f"k_{kind}"].fillna(False).astype(bool)]
    n_all = len(feats)
    feats = feats[feats.ticker.isin(companies)].copy()
    feats["earn"] = [(t, d) in ekeys for t, d in zip(feats.ticker, feats.d)]
    feats["earn_prev"] = [(t, prev_day.get(d)) in ekeys for t, d in zip(feats.ticker, feats.d)]
    feats = feats[[bool(keep(r)) for r in feats.itertuples()]]
    if a.rule == "b14":
        daily = load(a.prep, "daily").drop_duplicates(["ticker", "d"]).assign(d=lambda x: pd.to_datetime(x.d))
        for tk, g in daily.sort_values("d").groupby("ticker"):
            ds, cs = g.d.to_numpy(), g.c.to_numpy()
            for d in feats.d[feats.ticker == tk]:
                k = np.searchsorted(ds, np.datetime64(d))
                if k + 2 < len(ds):
                    SWING[(tk, d)] = cs[k + 2]
    fidx = {(r.ticker, r.d): r for r in feats.itertuples()}

    rows = []
    for fn in sorted(glob.glob(os.path.join(a.prep, "bars_*.parquet"))):
        df = pq.read_table(fn).to_pandas()
        df["d"] = pd.to_datetime(df.d)
        df = df[df.set_index(["ticker", "d"]).index.isin(list(fidx))]
        for (tk, d), g in df.groupby(["ticker", "d"], sort=False):
            r = fidx[(tk, d)]
            day = {k: g[k].to_numpy() for k in ("mi", "o", "h", "l", "c", "v")}
            for code, _, fn_ in variants:
                t = fn_(r, day, frac)
                if t:
                    rows.append(dict(ticker=tk, d=d, code=code, earn=r.earn, **t))
        print(f"{os.path.basename(fn)}: trades so far {len(rows)}", flush=True)
    t = pd.DataFrame(rows, columns=["ticker", "d", "code", "earn", "entry_mi", "entry_px", "stop_dist", "exit_mi", "exit_px", "gross", "reason"])
    t["d"] = pd.to_datetime(t.d)
    t["low_px"] = t.entry_px < 5
    t["cost"] = FEE + np.where(t.low_px, SLIP_LOW_PX, SLIP)
    t["hurdle"] = np.where(t.low_px, GATE_LOW_PX, GATE)
    t.to_csv(os.path.join(out, "trades.csv.gz"), index=False)

    def pick(code):
        s = t[t.code == code].sort_values(["d", "entry_mi"]).groupby("d").head(TOP_DAY)
        return s.assign(rank=s.entry_mi, stop_dist=s.stop_dist.fillna(portfolio.RISK / MAX_W))
    sel = {code: pick(code) for code, *_ in variants}
    extra = a.rule == "b13"
    head = ("| 변형 | 신호 수 | 비용 전 평균 | 비용 전 중앙값 | 승률 | t값 | 2025~26 평균 (수) | 관문 초과분 | 비용 후 평균 | "
            + ("비용 후 (미끄러짐 +0.3%) | " if extra else "") + "5달러 미만 비율 | 평균 손절폭 | 청산 이유 | 관문 |")
    lines = [f"# {a.rule} 백테스트 결과: {title} ({days[0].date()} ~ {days[-1].date()})", "",
             "비용: 수수료 0.25% + 미끄러짐 0.1% (한쪽), 매수가 5달러 미만이면 미끄러짐 0.3%. 관문: 비용 전 거래당 ≥ +0.9% (5달러 미만 거래는 +1.3%) → "
             "'관문 초과분'(거래마다 비용 전 수익 − 자기 관문의 평균)이 0 이상이면 통과. 상대 거래량 = 그때까지 거래량 ÷ (20일 평균 하루 거래량 × 그 시각까지 보통 나오는 비율). "
             "회사 주식만 (SEC 공시 회사 목록, 상장폐지 회사는 빠지는 한계).",
             f"후보(자료 준비 기준) {n_all}건 중 회사 주식·규칙서 조건 통과 {len(feats)}건", "",
             f"## 1차 관문 (하루 먼저 나온 신호 {TOP_DAY}개까지, 자금관리 없이)", "", head,
             "|---" * (head.count("|") - 1) + "|"]
    passed = []
    for code, name, _ in variants:
        s = sel[code]
        if s.empty:
            lines.append(f"| {name} | 0 |" + " |" * (head.count("|") - 3))
            continue
        g = s.gross
        tv = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan
        r = s[s.d.dt.year >= RECENT].gross
        exc = (g - s.hurdle).mean()
        n1 = portfolio.net_return(g, s.cost)
        ok = exc >= 0 and len(s) >= MIN_N and code != "A5"
        if ok:
            passed.append(code)
        rc = ", ".join(f"{k} {v}" for k, v in s.reason.value_counts().items())
        lines.append(f"| {name} | {len(s)} | {pct(g.mean(), 2)} | {pct(g.median(), 2)} | {pct((g > 0).mean())} | {tv:.2f} | "
                     f"{pct(r.mean(), 2)} ({len(r)}) | {exc * 100:+.2f}%p | {pct(n1.mean(), 2)} | "
                     + (f"{pct(portfolio.net_return(g, s.cost + EXTRA_B13).mean(), 2)} | " if extra else "")
                     + f"{pct(s.low_px.mean())} | {pct(s.stop_dist[t.loc[s.index, 'stop_dist'].notna()].mean(), 2)} | {rc} | "
                     + ("(비교용)" if code == "A5" else ("통과" if ok else ("탈락" if len(s) >= MIN_N else "신호 부족"))) + " |")
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else f'없음 → {a.rule} 탈락'}**"]

    base = sel[variants[0][0]]
    lines += ["", f"## 매수가 구간별 ({variants[0][0]}, 비용 전)", "", "| 매수가 | 신호 수 | 평균 | 중앙값 | 승률 |", "|---|---|---|---|---|"]
    for lo, hi in ((1, 5), (5, 20), (20, 1e9)):
        g = base[(base.entry_px >= lo) & (base.entry_px < hi)].gross
        if len(g):
            lines.append(f"| {lo}~{hi if hi < 1e9 else ''}달러 | {len(g)} | {pct(g.mean(), 2)} | {pct(g.median(), 2)} | {pct((g > 0).mean())} |")

    codes = [c for c, *_ in variants if not sel[c].empty]
    lines += ["", "## 해마다: 비용 전 거래당 평균 (신호 수)", "", "| 해 | " + " | ".join(codes) + " |", "|---" * (len(codes) + 1) + "|"]
    for y in sorted(t.d.dt.year.unique()):
        cells = []
        for c in codes:
            g = sel[c][sel[c].d.dt.year == y].gross
            cells.append(f"{g.mean() * 100:.2f}% ({len(g)})" if len(g) else "")
        lines.append(f"| {y} | " + " | ".join(cells) + " |")

    spy_ret = spy.pct_change().fillna(0.0)
    s0 = portfolio.stats(spy_ret)
    lines += ["", f"## 계좌 성과 (손절 시 계좌 0.5% 위험, 한 종목 최대 {MAX_W:.0%}, 동시 {MAX_POS}종목, 하루 -1.5% 중단)", "",
              "| 변형 | 연수익률 | 최대낙폭 | 샤프 | 실제 거래 수 |", "|---|---|---|---|---|",
              f"| SPY 보유 (배당 제외) | {pct(s0['cagr'])} | {pct(s0['mdd'])} | {s0['sharpe']:.2f} | |"]
    for code, name, _ in variants:
        s = sel[code]
        if s.empty or code == "A5":
            continue
        ret, k = portfolio.daily_returns(s.assign(exit_mi=s.exit_mi.where(s.exit_mi >= 0, 999)), days, s.cost, max_w=MAX_W, max_pos=MAX_POS)
        st = portfolio.stats(ret)
        lines.append(f"| {name} | {pct(st['cagr'])} | {pct(st['mdd'])} | {st['sharpe']:.2f} | {len(k)} |")
    if not base.empty:
        lines.append(f"\n{variants[0][0]} 신호가 많은 종목: {', '.join(f'{k} {v}' for k, v in base.ticker.value_counts().head(10).items())}")
        lines.append(f"{variants[0][0]} 매수 시각 중앙값: 9:30 + {int(base.entry_mi.median())}분, 실적일 비율 {base.earn.mean():.0%}")
        top = base.gross.sort_values(ascending=False)
        lines.append(f"{variants[0][0]} 가장 좋은 거래 10개를 빼면 비용 전 평균: {pct(top.iloc[10:].mean(), 2)}")
    text = "\n".join(lines) + "\n"
    open(os.path.join(out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
