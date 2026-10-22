"""scaling.py 격자: 진입 신호 4종 × 크기 방식 3종 × 청산 2종 × 숫자 조합. BTC·ETH, 개발/검증.

진입 신호
- random  : 하루 한 번 무작위 시각·무작위 방향 (씨앗 3개) → 크기 방식 효과만 보기
- fade    : 1시간 등락이 평소의 3.5배 이상이면 반대로 (1차 연구 규칙)
- breakout: 7일 최고가 돌파 → 롱, 7일 최저가 이탈 → 숏 (추세)
- clone   : 따라하기 모델 점수 상위 0.5% (BTC, 2020-07 이후만)
사용: python scaling_grid.py <데이터폴더> <작업폴더> <출력.csv>
"""
import itertools, sys
import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from scaling import run_scale  # noqa: E402
from grid import load, stats, COST, DEV, VAL  # noqa: E402
from fade import signals  # noqa: E402

data, work, out = sys.argv[1:4]
mk, tk, sp, fd = COST["base"]
G = dict(step=[0.005, 0.01, 0.02], tp=[0.01, 0.02, 0.04], sl=[0.02, 0.05, 0.1],
         trail=[0.01, 0.02, 0.04], H=[1440, 4320, 10080])
rows = []
for sym in ["BTCUSDT", "ETHUSDT"]:
    px = load(data, sym)
    for per, (a, b) in [("dev", DEV), ("val", VAL)]:
        p = px.loc[a:b]
        o, h, l, c, v = (p[k].to_numpy(np.float64) for k in "ohlcv")
        sigs = {}
        for seed in range(3):
            rng = np.random.default_rng(seed)
            s = np.zeros(len(c), np.int64)
            days = np.arange(0, len(c) - 1440, 1440)
            s[days + rng.integers(0, 1440, len(days))] = rng.choice([-1, 1], len(days))
            sigs[f"random{seed}"] = s
        sigs["fade"] = signals(c, v, 60, 3.5, 0.0)
        cs = pd.Series(c, p.index)
        hi = cs.rolling(10080).max().shift(1); lo = cs.rolling(10080).min().shift(1)
        bo = np.zeros(len(c), np.int64)
        bo[(cs > hi).to_numpy() & (cs.shift(1) <= hi.shift(1)).to_numpy()] = 1
        bo[(cs < lo).to_numpy() & (cs.shift(1) >= lo.shift(1)).to_numpy()] = -1
        sigs["breakout"] = bo
        if sym == "BTCUSDT":
            P = pd.read_parquet(f"{work}/clone_scores.parquet")
            tune = P.loc["2020-07-01":"2021-12-31"]
            thr = np.quantile((tune.pl + tune.ps).to_numpy(), 0.995)
            pp = P.loc[a:b]
            s5 = pd.Series(np.where((pp.pl + pp.ps) >= thr, np.where(pp.pl > pp.ps, 1, -1), 0),
                           index=pp.index + pd.Timedelta(minutes=4))
            s = s5.reindex(p.index).fillna(0).astype(np.int64).to_numpy().copy()
            if per == "dev":
                s[: p.index.searchsorted(pd.Timestamp("2020-07-01"))] = 0   # 학습 구간은 제외
            sigs["clone"] = s
        for name, sg in sigs.items():
            for mode, ek in itertools.product([0, 1, 2], [0, 1]):
                for step, x, sl, H in itertools.product(G["step"], G["tp"] if ek == 0 else G["trail"], G["sl"], G["H"]):
                    if mode == 0 and step != G["step"][0]:
                        continue  # 한 번에 사기는 step 무관
                    eq, tr = run_scale(o, h, l, c, sg, mode, ek, step, x, sl, x, H, mk, tk, sp, fd)
                    r = stats(eq, p.index, tr)
                    r.update(sym=sym, per=per, sig=name, mode=mode, exit=ek, step=step, x=x, sl=sl, H=H)
                    rows.append(r)
        print(sym, per, len(rows), flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
