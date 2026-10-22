"""워뇨띠의 실제 진입(시각·방향·평균단가)에 기계적 청산 규칙을 붙여 봅니다.

진입 자체에 힘이 있는지(=무엇을 언제 샀나), 청산 솜씨가 중요한지 가르기 위한 실험.
입력: 데이터 폴더, 작업 폴더(episodes.parquet)
"""
import glob, itertools, sys
import numpy as np
import pandas as pd

data, work = sys.argv[1:3]
px = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{data}/BTCUSDT_1m_20*.parquet")))
px = px.set_index("t").sort_index(); px = px[~px.index.duplicated()].asfreq("1min").ffill()
h, l, c = px.h.to_numpy(float), px.l.to_numpy(float), px.c.to_numpy(float)
e = pd.read_parquet(f"{work}/episodes.parquet")
e = e[(e.symbol == "XBTUSD") & e.t1.notna()].reset_index(drop=True)
i0 = px.index.searchsorted(e.t0.values) + 1      # 진입 다음 봉부터 청산 판단
side = e.side.to_numpy(); ent = c[i0 - 1]           # 바이낸스 가격 기준 진입가 (거래소 차이 제거)
COSTS = dict(maker=0.0002, taker=0.0005, slip=0.0003)


def replay(tp, sl, H):
    out = np.empty(len(e))
    for k in range(len(e)):
        s, p, a = side[k], ent[k], i0[k]
        b = min(a + H, len(c) - 1)
        hh, ll = h[a:b + 1], l[a:b + 1]
        if s > 0:
            hit_sl = np.flatnonzero(ll <= p * (1 - sl)); hit_tp = np.flatnonzero(hh > p * (1 + tp))
        else:
            hit_sl = np.flatnonzero(hh >= p * (1 + sl)); hit_tp = np.flatnonzero(ll < p * (1 - tp))
        jsl = hit_sl[0] if len(hit_sl) else 10**9; jtp = hit_tp[0] if len(hit_tp) else 10**9
        if jsl <= jtp and jsl < 10**9:
            out[k] = -sl - COSTS["maker"] - COSTS["taker"] - COSTS["slip"]
        elif jtp < 10**9:
            out[k] = tp - 2 * COSTS["maker"]
        else:
            out[k] = s * (c[b] / p - 1) - COSTS["maker"] - COSTS["taker"] - COSTS["slip"]
    return out


rows = []
for tp, sl, H in itertools.product([0.004, 0.008, 0.015, 0.03], [0.02, 0.05, 0.5], [240, 1440, 4320]):
    r = replay(tp, sl, H)
    rows.append(dict(tp=tp, sl=sl, H=H, avg=r.mean(), win=(r > 0).mean(), sum_=r.sum()))
res = pd.DataFrame(rows)
# 기준: 같은 진입 시각에 방향만 무작위로 뒤집기 / 그가 실제로 낸 성과
rng = np.random.default_rng(1)
actual = e.ret.to_numpy() - 2 * COSTS["maker"]
print("실제 청산(평균단가 기준, maker 수수료 차감) 평균", actual.mean().round(5), "승률", (actual > 0).mean().round(3))
print(res.sort_values("avg", ascending=False).round(5).to_string())
side0 = side.copy()
for trial in range(3):
    side = rng.choice([-1, 1], len(e))
    r = replay(0.008, 0.5, 1440)
    print("무작위 방향 (tp0.8%, 손절없음, 1일) 평균", r.mean().round(5), "승률", (r > 0).mean().round(3))
side = side0
for yr in sorted(e.t0.dt.year.unique()):
    m = (e.t0.dt.year == yr).to_numpy()
    r = replay(0.008, 0.5, 1440)[m]
    print(yr, "tp0.8% 1일:", r.mean().round(5), (r > 0).mean().round(3), " 실제:", actual[m].mean().round(5))
