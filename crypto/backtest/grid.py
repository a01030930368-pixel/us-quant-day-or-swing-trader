"""역추세 규칙 격자 탐색: 개발(2018.03~2021.12) 에서 고르고 검증(2022.01~2026.09) 에서 확인.

사용: python grid.py <데이터폴더> <심볼> <결과.csv> [비용시나리오]
비용 시나리오
- base : maker 0.02%, taker 0.05%, 슬리피지 0.03%, 펀딩 0.01%/8h (바이낸스 선물 일반 등급)
- harsh: maker 0.05%, taker 0.075%, 슬리피지 0.05%, 펀딩 0.01%/8h (BitMEX 일반 taker 수준)
- bitmex2018: maker -0.025%(리베이트), taker 0.075% (워뇨띠 당시 조건)
- zero: 비용 0 (규칙 자체에 힘이 있는지만 확인)
격자: 네 번째 뒤 다섯 번째 인자 'swing' 이면 며칠짜리 역추세(GRID_SWING)
"""
import glob, itertools, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fade import run, signals  # noqa: E402

COST = {"base": (0.0002, 0.0005, 0.0003, 0.0001), "harsh": (0.0005, 0.00075, 0.0005, 0.0001),
        # 워뇨띠가 매매하던 때의 BitMEX: 지정가는 오히려 0.025% 리베이트를 받음
        "bitmex2018": (-0.00025, 0.00075, 0.0003, 0.0001),
        "zero": (0.0, 0.0, 0.0, 0.0)}
DEV = ("2018-03-01", "2022-01-01")
VAL = ("2022-01-01", "2026-10-01")


def load(data, sym):
    px = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{data}/{sym}_1m_*.parquet")))
    px = px.set_index("t").sort_index()
    px = px[~px.index.duplicated()].asfreq("1min")
    px[["o", "h", "l", "c"]] = px[["o", "h", "l", "c"]].ffill()
    px["v"] = px.v.fillna(0)
    return px


def stats(eq, idx, tr):
    s = pd.Series(eq, idx)
    d = s.resample("1D").last()
    yrs = (idx[-1] - idx[0]).days / 365.25
    cagr = d.iloc[-1] ** (1 / yrs) - 1 if d.iloc[-1] > 0 else -1
    dr = d.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    mdd = (d / d.cummax() - 1).min()
    n = len(tr)
    win = (tr[:, 3] > 0).mean() if n else np.nan
    avg = tr[:, 3].mean() if n else np.nan
    return dict(cagr=cagr, mdd=mdd, sharpe=sh, trades_yr=n / yrs, win=win, avg_tr=avg, final=d.iloc[-1])


def hold(px, a, b):
    c = px.c.loc[a:b]
    d = c.resample("1D").last()
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    return dict(cagr=(c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1, mdd=(d / d.cummax() - 1).min(),
                sharpe=d.pct_change().mean() / d.pct_change().std() * np.sqrt(365))


GRID = dict(L=[15, 60, 240], Z=[2.5, 3.5, 5.0], V=[0.0, 1.5], off=[0.0, 0.003],
            tp=[0.004, 0.008, 0.015], sl=[0.02, 0.05, 0.5], H=[240, 1440, 4320], ladder=[0, 1])

GRID_SWING = dict(L=[1440, 4320, 10080], Z=[2.0, 3.0], V=[0.0], off=[0.0, 0.01],
                  tp=[0.02, 0.04, 0.08], sl=[0.05, 0.1, 0.5], H=[4320, 10080, 30240], ladder=[0, 1])

if __name__ == "__main__":
    data, sym, out = sys.argv[1:4]
    cost = sys.argv[4] if len(sys.argv) > 4 else "base"
    mk, tk, sp, fd = COST[cost]
    if len(sys.argv) > 5 and sys.argv[5] == "swing":
        GRID = GRID_SWING
    step = 0.03 if GRID is GRID_SWING else 0.01
    px = load(data, sym)
    rows = []
    sig_cache = {}
    t0 = time.time()
    for per, (a, b) in [("dev", DEV), ("val", VAL)]:
        p = px.loc[a:b]
        if len(p) == 0:
            continue
        o, h, l, c, v = (p[k].to_numpy(np.float64) for k in "ohlcv")
        for L, Z, V in itertools.product(GRID["L"], GRID["Z"], GRID["V"]):
            sg = signals(c, v, L, Z, V)
            for off, tp, sl, H, lad in itertools.product(GRID["off"], GRID["tp"], GRID["sl"], GRID["H"], GRID["ladder"]):
                eq, tr = run(o, h, l, c, sg, off, 60 if GRID is GRID_SWING else 15, lad, step, tp, sl, H, mk, tk, sp, fd, 1.0)
                r = stats(eq, p.index, tr)
                r.update(per=per, L=L, Z=Z, V=V, off=off, tp=tp, sl=sl, H=H, ladder=lad)
                rows.append(r)
        hb = hold(px, a, b); hb.update(per=per, L=-1)
        rows.append(hb)
        print(per, "done", round(time.time() - t0), "s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
