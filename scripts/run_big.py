"""b19(20% 넘는 갭 첫 1분 매수)·b20(장중 8-K 공시 직후 급등)·b24(마감 30분 낙폭과대 반등) 백테스트.

    python scripts/run_big.py --prep prep_big --sec sec --rule b19 --out reports/b19
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daytrade import portfolio  # noqa: E402
from daytrade.b08 import bad_prints  # noqa: E402
from daytrade.mom import simulate  # noqa: E402

FEE, SLIP, SLIP_LOW = 0.0025, 0.001, 0.003
GATE, GATE_LOW, MIN_N = 0.009, 0.013, 30
CLOSE, HALF_CLOSE = 389, 209
TICK = 0.01


# ---------------------------------------------------------------- 공통
def read(prep, kind):
    fs = sorted(glob.glob(os.path.join(prep, f"{kind}_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    if "d" in df:
        df["d"] = pd.to_datetime(df.d)
    return df


def day_dicts(bars):
    bars = bars.sort_values(["ticker", "d", "mi"])
    return {(tk, d): {k: g[k].to_numpy() for k in ("mi", "o", "h", "l", "c", "v")} for (tk, d), g in bars.groupby(["ticker", "d"], sort=False)}


def load_sec(sec, cal):
    """회사 주식 목록(실적 8-K를 낸 회사), 그 회사 첫 공시일, 실적 반응일, 8-K 전체."""
    k8 = pd.read_parquet(os.path.join(sec, "8k.parquet"))
    comp = pd.read_parquet(os.path.join(sec, "companies.parquet"))
    earn_tk = set(k8.ticker[k8["items"].str.contains("2.02", regex=False)])
    first = comp[comp.ticker.isin(earn_tk)].groupby("ticker").first_filing.min()
    e = k8[k8["items"].str.contains("2.02", regex=False)]
    day = e.accepted.dt.normalize()
    mins = (e.accepted.dt.hour * 60 + e.accepted.dt.minute).to_numpy()
    nxt = cal.searchsorted(day, side="right")
    geq = cal.searchsorted(day, side="left")
    pos = np.where(mins >= 960, nxt, geq)  # 장 마감 뒤 -> 다음 거래일, 그 밖(장전·장중) -> 그날(휴일이면 다음 거래일)
    ok = pos < len(cal)
    earn = set(zip(e.ticker.to_numpy()[ok], cal[pos[ok]]))
    return first, earn, k8


def company_mask(df, first):
    f = df.ticker.map(first)
    return f.notna() & (f <= df.d + pd.Timedelta(days=30))


def tstat(g):
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g.std() > 0 else np.nan


def pct(x, k=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{k}f}%"


def costs(s):
    low = s.entry_px < 5
    return s.assign(hurdle=np.where(low, GATE_LOW, GATE), cost=FEE + np.where(low, SLIP_LOW, SLIP))


HEAD = ("| 변형 | 거래 수 | 비용 전 평균 | 중앙값 | 승률 | t값 | 2025~26 평균 (수) | 상위 10개 뺀 평균 | 관문 초과분 | 비용 후 평균 | 평균 손절폭 | 관문 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|")


def row(name, s, judge=True):
    if s is None or s.empty:
        return f"| {name} | 0 |" + " |" * 11, False
    g, y = s.gross, s.d.dt.year
    ex = (g - s.hurdle).mean()
    net = portfolio.net_return(g, s.cost)
    if not judge:
        v, ok = "(판정 안 함)", False
    elif len(g) < MIN_N:
        v, ok = "표본 부족", False
    else:
        ok = ex >= 0
        v = "통과" if ok else "탈락"
    sd = s.stop_dist.mean() if "stop_dist" in s else np.nan
    return (f"| {name} | {len(g)} | {pct(g.mean())} | {pct(g.median())} | {(g > 0).mean():.0%} | {tstat(g):.2f} | "
            f"{pct(g[y >= 2025].mean())} ({int((y >= 2025).sum())}) | {pct(g.sort_values(ascending=False).iloc[10:].mean())} | "
            f"{ex * 100:+.2f}%p | {pct(net.mean())} | {pct(sd)} | {v} |"), ok


def common_tables(sel, names, judge, main, periods, spy, title, note, top_n=5):
    lines = [title, "", note, "", "## 1차 관문", "", *HEAD]
    passed = []
    for c, s in sel.items():
        r, ok = row(names[c], s, judge=c in judge)
        lines.append(r)
        if ok:
            passed.append(c)
    lines += ["", f"**관문 통과 변형: {', '.join(passed) if passed else '없음 → 탈락'}**", ""]
    lines += ["## 기간 나눠 보기 (비용 전 평균, 괄호는 수)", "", "| 변형 | " + " | ".join(p[0] for p in periods) + " |", "|---" * (len(periods) + 1) + "|"]
    for c, s in sel.items():
        if s.empty:
            continue
        y = s.d.dt.year
        lines.append(f"| {names[c]} | " + " | ".join(f"{pct(s.gross[(y >= a) & (y <= b)].mean())} ({int(((y >= a) & (y <= b)).sum())})" for _, a, b in periods) + " |")
    b = sel[main]
    lines += ["", f"## 매수가 구간별 ({names[main].split(' ')[0]}, 비용 전)", "", "| 매수가 | 수 | 평균 | 중앙값 | 승률 |", "|---|---|---|---|---|"]
    for lo, hi in ((1, 5), (5, 20), (20, 50), (50, 1e9)):
        g = b.gross[(b.entry_px >= lo) & (b.entry_px < hi)]
        lines.append(f"| {lo}~{'' if hi > 1e8 else hi}달러 | {len(g)} | {pct(g.mean())} | {pct(g.median())} | {(g > 0).mean() if len(g) else 0:.0%} |")
    lines += ["", f"## 해마다 ({names[main].split(' ')[0]}, 비용 전 평균, 괄호는 수)", "", "| 해 | 수 | 평균 | 중앙값 |", "|---|---|---|---|"]
    for yy, g in b.groupby(b.d.dt.year).gross:
        lines.append(f"| {yy} | {len(g)} | {pct(g.mean())} | {pct(g.median())} |")
    lines += ["", f"신호가 많았던 종목 ({names[main].split(' ')[0]}): " + ", ".join(f"{k} {v}" for k, v in b.ticker.value_counts().head(10).items())]
    if "reason" in b and len(b):
        lines.append(f"\n청산 이유: {', '.join(f'{k} {v}' for k, v in b.reason.value_counts().items())}. 매수 시각 중앙값 9:30 + {int(b.entry_mi.median())}분.")
    th = ["| 날짜 | 종목 | 매수가 | 비용 전 |", "|---|---|---|---|"]
    lines += ["", "가장 좋은 10개 / 가장 나쁜 10개:", "", *th] + [f"| {r.d.date()} | {r.ticker} | {r.entry_px:.2f} | {r.gross * 100:+.1f}% |" for r in pd.concat([b.nlargest(10, "gross"), b.nsmallest(10, "gross")]).itertuples()]
    cal = spy.index
    s0 = portfolio.stats(spy.pct_change().fillna(0.0))
    lines += ["", f"## 계좌 기준 (손절 때 계좌 0.5% 위험, 한 종목 최대 20%, 동시 {top_n}종목, 하루 -1.5%면 중단)", "",
              "| 변형 | 연수익률 | 최대낙폭 | 실제 거래 수 |", "|---|---|---|---|",
              f"| SPY 보유 (배당 제외, {cal[0].year}~) | {pct(s0['cagr'], 1)} | {pct(s0['mdd'], 1)} | |"]
    for c, s in sel.items():
        if s.empty or c not in judge:
            continue
        k = s.assign(rank=range(len(s)), stop_dist=(s.stop_dist if "stop_dist" in s else pd.Series(np.nan, s.index)).fillna(portfolio.RISK / 0.2))
        r, taken = portfolio.daily_returns(k, cal, k.cost, max_w=0.2, max_pos=top_n)
        st = portfolio.stats(r)
        lines.append(f"| {names[c]} | {pct(st['cagr'], 1)} | {pct(st['mdd'], 1)} | {len(taken)} |")
    return lines


def close_index(mi, half=False):
    return int(np.searchsorted(mi, HALF_CLOSE if half else CLOSE, side="right")) - 1


def at_close_of(day, t, half=False):
    """t분 1분봉(없으면 그 전 마지막) 종가에 매수 -> 마감 종가."""
    mi, c = day["mi"], day["c"]
    e = int(np.searchsorted(mi, t, side="right")) - 1
    x = close_index(mi, half)
    if e < 0 or x <= e:
        return None
    return dict(entry_mi=int(mi[e]), entry_px=c[e], exit_mi=int(mi[x]), exit_px=c[x], gross=c[x] / c[e] - 1, reason="마감", stop_dist=np.nan)


# ---------------------------------------------------------------- b19
def rule_b19(a):
    feats, bars, spy = read(a.prep, "feats"), read(a.prep, "b19"), read(a.prep, "spy")
    spy = spy.drop_duplicates("d").set_index("d").c.sort_index()
    cal = pd.DatetimeIndex(spy.index)
    first, earn, _ = load_sec(a.sec, cal)
    f = feats[feats.kind == "b19"].copy()
    f = f[company_mask(f, first)]
    days = day_dicts(bars)
    rows, nbad = [], 0
    for r in f.itertuples():
        day = days.get((r.ticker, r.d))
        if day is None or day["mi"][0] > 1:
            continue
        if bad_prints(day):
            nbad += 1
            continue
        o0 = day["o"][0]
        gap = o0 / r.pc - 1
        dv1 = day["c"][0] * day["v"][0]
        base = dict(ticker=r.ticker, d=r.d, gap=gap, dv1=dv1, pm_dv=r.pm_dv, pm_ok=bool(o0 >= 0.98 * r.pm_high) if r.pm_high == r.pm_high else False,
                    earn=(r.ticker, r.d) in earn)
        for code, t in (("t0", at_close_of(day, 0)), ("t5", at_close_of(day, 5)), ("t30", at_close_of(day, 30))):
            if t:
                rows.append(dict(base, mode=code, **t))
        lo5 = day["l"][day["mi"] < 5].min() - TICK
        px = day["c"][0]
        if px > lo5 and len(day["mi"]) > 1:
            sim = simulate(day, 1, px, lo5, None, until=CLOSE)
            if sim:
                rows.append(dict(base, mode="stop", entry_mi=int(day["mi"][0]), entry_px=px, stop_dist=(px - lo5) / px, **sim))
    t = costs(pd.DataFrame(rows))
    t = t[(t.dv1 >= 0.5e6) & (t.pm_dv >= 1e6)]

    def top5(s):
        return s.sort_values(["d", "pm_dv"], ascending=[True, False]).groupby("d").head(5)
    m, g = t["mode"], t.gap
    sel = {"X1": top5(t[(m == "t0") & (g >= 0.20)]),
           "X2-15": top5(t[(m == "t0") & (g >= 0.15) & (g < 0.20)]), "X2-20": top5(t[(m == "t0") & (g >= 0.20) & (g < 0.40)]),
           "X2-40": top5(t[(m == "t0") & (g >= 0.40)]),
           "X3-5": top5(t[(m == "t5") & (g >= 0.20)]), "X3-30": top5(t[(m == "t30") & (g >= 0.20)]),
           "X4": top5(t[(m == "stop") & (g >= 0.20)]),
           "X5-e": top5(t[(m == "t0") & (g >= 0.20) & t.earn]), "X5-x": top5(t[(m == "t0") & (g >= 0.20) & ~t.earn]),
           "X6": top5(t[(m == "t0") & (g >= 0.20) & (t.entry_px >= 5)]),
           "X7": top5(t[(m == "t0") & (g >= 0.20) & t.pm_ok])}
    names = {"X1": "X1 기본 (갭 ≥ +20%, 첫 1분봉 종가 매수 → 15:59 종가)", "X2-15": "X2 갭 15~20%", "X2-20": "X2 갭 20~40%",
             "X2-40": "X2 갭 40% 이상", "X3-5": "X3 9:35 1분봉 종가 매수", "X3-30": "X3 10:00 1분봉 종가 매수",
             "X4": "X4 손절 첫 5분 저가 아래", "X5-e": "X5 실적 발표일만", "X5-x": "X5 실적일 제외", "X6": "X6 매수가 5달러 이상만",
             "X7": "X7 시가 ≥ 장전 고가 × 0.98"}
    note = (f"자료: 무료 1분봉 {spy.index[0].date()} ~ {spy.index[-1].date()}. 회사 주식만: SEC 실적 공시(8-K 2.02)를 낸 회사 목록에 있고, "
            "그 회사의 첫 SEC 공시가 그날보다 앞서는 종목 (지금 상장된 회사 기준이라 상장폐지 회사는 빠짐, SEC 전자공시가 1993~1996년에 시작돼 초기 표본이 적음). "
            "주식분할로 보이는 갭(1.5·2·3·4·5·10배 ±3%) 제외: 이 때문에 갭 +45~55%, +94~106% 구간은 빠짐. "
            f"자료 오류 날 {nbad}개 뺌. 조건: 전날 종가 ≥ 2달러, 첫 1분봉 거래대금 ≥ 50만 달러, 장전 거래대금 ≥ 100만 달러, 하루 장전 거래대금 큰 5개. "
            "비용: 수수료 0.25% + 미끄러짐 0.1% (5달러 미만 0.3%). 관문: 비용 전 ≥ +0.9% (5달러 미만 +1.3%) → 관문 초과분 ≥ 0.")
    lines = common_tables(sel, names, set(names), "X1", [("1993~2003", 1993, 2003), ("2004~2017", 2004, 2017), ("2018~2026", 2018, 2026)],
                          spy, "# b19 백테스트 결과: 20% 넘는 갭 상승, 첫 1분 매수 → 종가", note)
    return t, lines


# ---------------------------------------------------------------- b24
def rule_b24(a):
    feats, bars, spy = read(a.prep, "feats"), read(a.prep, "b24"), read(a.prep, "spy")
    spy = spy.drop_duplicates("d").set_index("d").c.sort_index()
    cal = pd.DatetimeIndex(spy.index)
    first, earn, _ = load_sec(a.sec, cal)
    f = feats[feats.kind == "b24"].copy()
    f = f[company_mask(f, first)]
    days = day_dicts(bars)
    rows, nbad = [], 0
    for r in f.itertuples():
        day = days.get((r.ticker, r.d))
        if day is None:
            continue
        if bad_prints(day):
            nbad += 1
            continue
        hf = bool(r.half)
        mi = day["mi"]
        x = close_index(mi, hf)
        base = dict(ticker=r.ticker, d=r.d, earn=(r.ticker, r.d) in earn, half=hf)
        for code, t_in, ref in (("m30", 180 if hf else 360, r.c_cut), ("m60", 150 if hf else 330, r.c_cut60)):
            e = int(np.searchsorted(mi, t_in))
            if e >= len(mi) or mi[e] > t_in + 2 or x <= e or not ref == ref:
                continue
            dvc = r.dv_cut if code == "m30" else r.dv_cut60
            rows.append(dict(base, mode=code, chg=ref / r.pc - 1, chg_o=ref / r.o - 1, dvc=dvc, entry_mi=int(mi[e]), entry_px=day["o"][e],
                             exit_mi=int(mi[x]), exit_px=day["c"][x], gross=day["c"][x] / day["o"][e] - 1, reason="마감", stop_dist=np.nan))
    t = costs(pd.DataFrame(rows))
    t = t[t.dvc >= 20e6]

    def top5(s, key="chg", asc=True):
        return s.sort_values(["d", key], ascending=[True, asc]).groupby("d").head(5)
    m30, m60 = t["mode"] == "m30", t["mode"] == "m60"
    q1 = t[m30 & (t.chg <= -0.08)]
    sel = {"Q1": top5(q1), "Q2-15": top5(t[m30 & (t.chg <= -0.15)]), "Q2-25": top5(t[m30 & (t.chg <= -0.25)]),
           "Q3": top5(t[m30 & (t.chg_o <= -0.08)], "chg_o"), "Q4": top5(t[m60 & (t.chg <= -0.08)]),
           "Q5-x": top5(q1[~q1.earn]), "Q5-e": top5(q1[q1.earn]),
           "Q6": top5(t[m30 & (t.chg >= 0.08)], asc=False), "Q7": top5(q1[q1.d.dt.year >= 2018])}
    names = {"Q1": "Q1 기본 (15:29에 전날 종가 대비 -8% 이하, 15:30 시가 매수 → 15:59 종가)", "Q2-15": "Q2 -15% 이하", "Q2-25": "Q2 -25% 이하",
             "Q3": "Q3 그날 시가 대비 -8% 이하", "Q4": "Q4 14:59에 -8% 이하, 15:00 매수", "Q5-x": "Q5 실적일 제외", "Q5-e": "Q5 실적일만",
             "Q6": "Q6 (참고) +8% 이상 오른 종목 15:30 → 15:59 (공매도 불가라 판정 안 함)", "Q7": "Q7 2018년 이후만"}
    judge = set(names) - {"Q6"}
    note = (f"자료: 무료 1분봉 {spy.index[0].date()} ~ {spy.index[-1].date()}. 회사 주식만 (b19와 같은 SEC 회사 목록 방법). "
            f"조건: 전날 종가 ≥ 5달러, 그때까지 거래대금 ≥ 2,000만 달러, 하락 큰 5개. 반나절 장은 12:30 매수 → 12:59. 자료 오류 날 {nbad}개 뺌. "
            "비용: 수수료 0.25% + 미끄러짐 0.1% (5달러 미만 0.3%). 관문: 비용 전 ≥ +0.9% → 관문 초과분 ≥ 0.")
    lines = common_tables(sel, names, judge, "Q1", [("1993~2003", 1993, 2003), ("2004~2017", 2004, 2017), ("2018~2026", 2018, 2026)],
                          spy, "# b24 백테스트 결과: 장 마감 30분 낙폭과대 반등", note)
    lines += ["", "## 하락 폭 구간별 (Q1 후보 전체, 하루 5개 제한 없이, 비용 전)", "", "| 하락 폭 | 수 | 평균 | 중앙값 | 승률 |", "|---|---|---|---|---|"]
    for lo, hi in ((-0.15, -0.08), (-0.25, -0.15), (-1.0, -0.25)):
        g = q1.gross[(q1.chg > lo) & (q1.chg <= hi)]
        lines.append(f"| {-hi * 100:.0f}~{-lo * 100:.0f}% | {len(g)} | {pct(g.mean())} | {pct(g.median())} | {(g > 0).mean() if len(g) else 0:.0%} |")
    return t, lines


# ---------------------------------------------------------------- b20
def event_trade(day, t, frac, adv_sh, thr=0.03, mult=5.0, stop=True):
    """t분에 공시 접수. P0 = t 전 마지막 1분봉 종가. t~t+5분 안의 1분봉 종가가 P0×(1+thr) 이상이고
    그때까지 거래량 ≥ 같은 5분의 평소 거래량 × mult면 다음 1분봉 시가 매수. 손절 P0 - 0.01 (8% 넘으면 건너뜀)."""
    mi, o, c, v = day["mi"], day["o"], day["c"], day["v"]
    k0 = int(np.searchsorted(mi, t))
    if k0 == 0 or k0 >= len(mi):
        return None
    p0 = c[k0 - 1]
    expect = adv_sh * frac[t:t + 5].sum()
    if not expect > 0:
        return None
    cum = 0.0
    for j in range(k0, len(mi)):
        if mi[j] >= t + 5:
            return None
        cum += v[j]
        if c[j] >= p0 * (1 + thr) and cum >= mult * expect:
            e = j + 1
            if e >= len(mi) or mi[e] > mi[j] + 3:
                return None
            px = o[e]
            sp = p0 - TICK if stop else None
            if stop and (sp >= px or (px - sp) / px > 0.08):
                return None
            r = simulate(day, e, px, sp, None, until=CLOSE)
            if r is None:
                return None
            return dict(signal_mi=int(mi[j]), p0=p0, entry_mi=int(mi[e]), entry_px=px, stop_dist=(px - sp) / px if stop else np.nan, **r)
    return None


def rule_b20(a):
    feats, bars, spy = read(a.prep, "feats"), read(a.prep, "b20"), read(a.prep, "spy")
    spy = spy.drop_duplicates("d").set_index("d").c.sort_index()
    cal = pd.DatetimeIndex(spy.index)
    first, earn, k8 = load_sec(a.sec, cal)
    prof_y = {}
    for f_ in sorted(glob.glob(os.path.join(a.prep, "prof_*.parquet"))):
        p = pd.read_parquet(f_).set_index("mi").v.reindex(range(390), fill_value=0.0)
        prof_y[int(f_[-12:-8])] = (p / p.sum()).to_numpy() if p.sum() > 0 else None
    f = feats[feats.kind.isin(["b20", "b20c"])].copy()
    f = f[company_mask(f, first)].set_index(["ticker", "d"])
    days = day_dicts(bars)
    k8 = k8.assign(d=k8.accepted.dt.normalize(), t=k8.accepted.dt.hour * 60 + k8.accepted.dt.minute - 570)
    k8 = k8[(k8.t >= 5) & (k8.t <= 300) & k8.d.isin(cal)].sort_values("accepted")
    rows = []
    for (tk, d), g in k8.groupby(["ticker", "d"], sort=False):
        if (tk, d) not in f.index:
            continue
        r = f.loc[(tk, d)]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        day = days.get((tk, d))
        frac = prof_y.get(d.year)
        if day is None or frac is None or bad_prints(day):
            continue
        items = ",".join(sorted(set(",".join(g["items"]).split(",")) - {""}))
        base = dict(ticker=tk, d=d, items=items, accepted=g.accepted.iloc[0], nc=r.nc, kind="8-K")
        for code, kw in (("K1", {}), ("K2-5", dict(thr=0.05)), ("K2-10", dict(thr=0.10)), ("K4", dict(stop=False))):
            for t in g.t.astype(int).unique():
                tr = event_trade(day, t, frac, r.adv_sh, **kw)
                if tr:
                    rows.append(dict(base, mode=code, **tr))
                    break
    # 대조군: 공시 없는 날, 9:35~14:30 아무 때나 같은 조건
    for (tk, d), r in f[f.kind == "b20c"].iterrows():
        day = days.get((tk, d))
        frac = prof_y.get(d.year)
        if day is None or frac is None or bad_prints(day):
            continue
        for t in day["mi"][(day["mi"] >= 5) & (day["mi"] <= 300)]:
            tr = event_trade(day, int(t), frac, r.adv_sh)
            if tr:
                rows.append(dict(ticker=tk, d=d, items="", accepted=pd.NaT, nc=r.nc, kind="none", mode="K6", **tr))
                break
    t = costs(pd.DataFrame(rows))
    k1 = t[(t["mode"] == "K1") & (t.kind == "8-K")]
    k5 = k1.dropna(subset=["nc"]).copy()
    k5["gross"] = np.where(k5.reason == "손절", k5.gross, k5.nc / k5.entry_px - 1)
    has = lambda s, it: s[s["items"].str.split(",").apply(lambda xs: any(x in it for x in xs))]  # noqa: E731
    sel = {"K1": k1, "K2-5": t[t["mode"] == "K2-5"], "K2-10": t[t["mode"] == "K2-10"],
           "K3-101": has(k1, {"1.01"}), "K3-202": has(k1, {"2.02"}), "K3-78": has(k1, {"7.01", "8.01"}), "K3-502": has(k1, {"5.02"}),
           "K4": t[t["mode"] == "K4"], "K5": k5, "K6": t[t["mode"] == "K6"]}
    names = {"K1": "K1 기본 (장중 8-K 접수 후 5분 안 +3%·거래량 5배, 다음 1분봉 시가 매수, 손절 공시 직전 가격 아래)",
             "K2-5": "K2 +5% 이상", "K2-10": "K2 +10% 이상", "K3-101": "K3 항목 1.01 (중요 계약) 포함", "K3-202": "K3 항목 2.02 (실적) 포함",
             "K3-78": "K3 항목 7.01·8.01 (기타 발표) 포함", "K3-502": "K3 항목 5.02 (임원 변경) 포함", "K4": "K4 손절 없이",
             "K5": "K5 (비교용) 다음 날 종가까지", "K6": "K6 (대조군) 공시 없이 같은 조건으로 오른 날 (하루 최대 15개 무작위 표본)"}
    judge = {"K1", "K2-5", "K2-10", "K3-101", "K3-202", "K3-78", "K3-502", "K4"}
    n_ev = int(k8.groupby(["ticker", "d"]).ngroups)
    note = (f"자료: 무료 1분봉 + SEC 8-K 접수 시각(세계표준시를 미국 동부시간으로 바꿈). 9:35~14:30 접수된 8-K가 있는 종목-날짜 {n_ev}개 중 "
            "접수 후 5분 안에 3% 넘게 오른 적 있는 날, "
            "회사 주식(b19와 같은 방법)·전날 종가 ≥ 2달러·20일 평균 거래대금 ≥ 100만 달러인 날만. "
            "'같은 5분의 평소 거래량' = 20일 평균 하루 거래량 × 그해 시장 전체의 그 5분 거래량 비중 (종목별 시간대 자료 대신 [해석]). "
            "비용: 수수료 0.25% + 미끄러짐 0.1% (5달러 미만 0.3%). 관문: 비용 전 ≥ +0.9% → 관문 초과분 ≥ 0.")
    lines = common_tables(sel, names, judge, "K1", [("2001~2009", 2001, 2009), ("2010~2017", 2010, 2017), ("2018~2026", 2018, 2026)],
                          spy, "# b20 백테스트 결과: 장중 8-K 공시 직후 급등 따라가기", note)
    al = read(a.prep, "align")
    lines += ["", "## 접수 시각 확인 (모든 장중 8-K, 그 시각부터 5분 동안 움직인 크기 평균)", "", "| 접수 시각 기준 | 5분 움직임 평균 | 수 |", "|---|---|---|"]
    if len(al):
        for off, g in al.groupby("off"):
            lines.append(f"| {off:+d}분부터 | {pct((g.r * g.n).sum() / g.n.sum(), 3)} | {int(g.n.sum())} |")
    lines.append("\n접수 시각(0분)에서 움직임이 가장 크면 시각이 맞게 맞춰진 거예요.")
    smp = k1.sample(min(10, len(k1)), random_state=1) if len(k1) else k1
    lines += ["", "## 접수 시각 확인 샘플 10건 (K1)", "", "| 종목 | 접수 시각 (동부) | 항목 | 공시 직전 가격 | 신호 시각 | 매수 시각 | 매수가 | 비용 전 |", "|---|---|---|---|---|---|---|---|"]
    for r in smp.itertuples():
        hm = lambda m: f"{9 + (30 + m) // 60}:{(30 + m) % 60:02d}"  # noqa: E731
        lines.append(f"| {r.ticker} | {r.accepted} | {r.items} | {r.p0:.2f} | {hm(r.signal_mi)} | {hm(r.entry_mi)} | {r.entry_px:.2f} | {r.gross * 100:+.1f}% |")
    return t, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_big")
    ap.add_argument("--sec", default="sec")
    ap.add_argument("--rule", required=True, choices=["b19", "b20", "b24"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"reports/{a.rule}"
    os.makedirs(out, exist_ok=True)
    t, lines = {"b19": rule_b19, "b20": rule_b20, "b24": rule_b24}[a.rule](a)
    t.to_csv(os.path.join(out, "trades.csv.gz"), index=False)
    text = "\n".join(lines) + "\n"
    open(os.path.join(out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
