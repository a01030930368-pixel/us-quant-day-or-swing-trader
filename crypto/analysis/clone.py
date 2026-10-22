"""워뇨띠 따라하기 모델: 5분봉마다 '그가 지금 롱/숏을 새로 시작할까'를 맞히도록 학습합니다.

기간
- 학습: 2018-03 ~ 2020-06 (그의 매매로 모델을 맞춤)
- 조정: 2020-07 ~ 2021-12 (모델이 처음 보는 그의 매매. 신호 문턱·청산 방식은 여기서만 고름)
- 검증: 2022-01 ~ 2026-09 (그의 장부 이후, 아무것도 고르지 않은 구간)
사용: python clone.py <데이터폴더> <작업폴더>
"""
import itertools, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/backtest")
from fade import run  # noqa: E402
from grid import load, stats, hold, COST  # noqa: E402

data, work = sys.argv[1:3]
f = pd.read_parquet(f"{work}/feat5.parquet")
e = pd.read_parquet(f"{work}/episodes.parquet")
e = e[(e.symbol == "XBTUSD") & e.t1.notna()]
# 진입 결정 직전에 끝난 5분봉에 꼬리표
lab = pd.Series(0, index=f.index, dtype="int8")
ix = e.t0.dt.floor("5min") - pd.Timedelta(minutes=5)
for t, s in zip(ix, e.side):
    if t in lab.index:
        lab.loc[t] = 1 if s > 0 else 2
X = f.drop(columns=["c"])
TR = ("2018-03-05", "2020-06-30 23:55")
TU = ("2020-07-01", "2021-12-31 23:55")
VA = ("2022-01-01", "2026-09-30 23:55")
xtr, ytr = X.loc[TR[0]:TR[1]], lab.loc[TR[0]:TR[1]]
xtu, ytu = X.loc[TU[0]:TU[1]], lab.loc[TU[0]:TU[1]]
print("진입 꼬리표: 학습", (ytr > 0).sum(), "조정", (ytu > 0).sum())

w = np.where(ytr > 0, 50.0, 1.0)
m = lgb.LGBMClassifier(objective="multiclass", n_estimators=400, learning_rate=0.03, num_leaves=15,
                       min_child_samples=200, subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
                       reg_lambda=5.0, verbose=-1)
m.fit(xtr, ytr, sample_weight=w)
p = m.predict_proba(xtu)
enter = p[:, 1] + p[:, 2]
print("조정 구간: '들어갈까' AUC", round(roc_auc_score(ytu > 0, enter), 3))
mask = (ytu > 0).to_numpy()
dir_ok = ((p[mask, 1] > p[mask, 2]) == (ytu[mask] == 1).to_numpy()).mean()
print("조정 구간: 그가 들어간 때 방향 맞힌 비율", round(dir_ok, 3))
imp = pd.Series(m.feature_importances_, X.columns).sort_values(ascending=False)
print("많이 쓴 재료:", ", ".join(imp.head(12).index))

# 전체 기간 점수
P = pd.DataFrame(m.predict_proba(X.loc[TU[0]:]), index=X.loc[TU[0]:].index, columns=["p0", "pl", "ps"])
P.to_parquet(f"{work}/clone_scores.parquet")

# 매매: 점수가 문턱을 넘으면 더 큰 쪽으로 지정가 진입 (1분봉 시뮬레이터)
px = load(data, "BTCUSDT")
mk, tk, sp, fd = COST["base"]
rows = []
for per, (a, b) in [("tune", TU), ("val", VA)]:
    p1 = px.loc[a:b]
    o, h, l, c = (p1[k].to_numpy(np.float64) for k in "ohlc")
    pp = P.loc[a:b]
    score = (pp.pl + pp.ps).to_numpy()
    side = np.where(pp.pl > pp.ps, 1, -1)
    # 문턱은 조정 구간 점수 분포의 상위 q (그 문턱 값을 검증에도 그대로 사용)
    for q in [0.995, 0.999, 0.9997]:
        thr = np.quantile((P.loc[TU[0]:TU[1]].pl + P.loc[TU[0]:TU[1]].ps).to_numpy(), q)
        sig5 = pd.Series(np.where(score >= thr, side, 0), index=pp.index + pd.Timedelta(minutes=4))
        sig = sig5.reindex(p1.index).fillna(0).astype(np.int64).to_numpy()
        for tp, sl, H in itertools.product([0.004, 0.008, 0.015, 0.03], [0.02, 0.05, 0.5], [240, 1440, 4320]):
            eq, tr = run(o, h, l, c, sig, 0.0, 15, 0, 0.01, tp, sl, H, mk, tk, sp, fd, 1.0)
            r = stats(eq, p1.index, tr)
            r.update(per=per, q=q, tp=tp, sl=sl, H=H)
            rows.append(r)
    hb = hold(px, a, b); hb.update(per=per, q=-1); rows.append(hb)
res = pd.DataFrame(rows)
res.to_csv(f"{work}/clone_bt.csv", index=False)
K = ["q", "tp", "sl", "H"]
t = res[res.per == "tune"].set_index(K).join(res[res.per == "val"].set_index(K), lsuffix="_t", rsuffix="_v")
t = t[t.trades_yr_t >= 5]
pd.set_option("display.width", 250)
print("조합", len(t), "조정에서 플러스", (t.cagr_t > 0).mean().round(3), "검증에서 플러스", (t.cagr_v > 0).mean().round(3))
print(t.sort_values("sharpe_t", ascending=False).head(10)[["cagr_t", "mdd_t", "sharpe_t", "trades_yr_t", "win_t",
                                                          "cagr_v", "mdd_v", "sharpe_v", "trades_yr_v", "win_v"]].round(3).to_string())
print(res[res.q == -1][["per", "cagr", "mdd", "sharpe"]].round(3).to_string())
