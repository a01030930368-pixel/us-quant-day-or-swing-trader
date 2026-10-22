"""매매 시점의 차트 상황(직전 등락, 거래량, 이평선 거리 등)을 무작위 시점과 비교합니다.

입력: 데이터 폴더(BTCUSDT_1m_*.parquet), episodes.parquet, orders_xbt.parquet
출력: entry_features.parquet, 표 출력
"""
import glob, sys
import numpy as np
import pandas as pd

data_dir, work = sys.argv[1], sys.argv[2]
px = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{data_dir}/BTCUSDT_1m_*.parquet")))
px = px.set_index("t").sort_index()
px = px[~px.index.duplicated()].asfreq("1min").ffill()
c, v = px.c.astype(float), px.v.fillna(0).astype(float)


def feats(times):
    """times 직전(이미 끝난 1분봉 기준) 특징."""
    t = pd.DatetimeIndex(times).floor("1min") - pd.Timedelta(minutes=1)
    out = {}
    for k, m in [("r5m", 5), ("r15m", 15), ("r1h", 60), ("r4h", 240), ("r24h", 1440), ("r3d", 4320), ("r7d", 10080)]:
        out[k] = (c.reindex(t).values / c.reindex(t - pd.Timedelta(minutes=m)).values - 1)
    v15 = v.rolling(15).sum(); v1d = v.rolling(1440).sum() / 96
    out["vol15_x"] = (v15 / v1d).reindex(t).values
    for k, m in [("ma1h_200", 200 * 60), ("ma4h_50", 50 * 240), ("ma1d_20", 20 * 1440)]:
        out["d_" + k] = (c / c.rolling(m, min_periods=m // 2).mean() - 1).reindex(t).values
    hi = c.rolling(1440).max(); lo = c.rolling(1440).min()
    out["pos24h"] = ((c - lo) / (hi - lo)).reindex(t).values
    hi7 = c.rolling(10080).max(); lo7 = c.rolling(10080).min()
    out["pos7d"] = ((c - lo7) / (hi7 - lo7)).reindex(t).values
    vol = np.log(c).diff().rolling(1440).std() * np.sqrt(1440)
    out["dvol"] = vol.reindex(t).values
    return pd.DataFrame(out, index=times)


if __name__ == "__main__":
    e = pd.read_parquet(f"{work}/episodes.parquet")
    e = e[(e.symbol == "XBTUSD") & e.t1.notna()].reset_index(drop=True)
    f = feats(e.t0.values)
    f["side"] = e.side.values; f["xbt"] = e.xbt.values; f["year"] = e.t0.dt.year.values
    f["hold_min"] = e.hold_min.values; f["maxpos"] = e.maxpos.values
    # 무작위 기준: 같은 기간 아무 시각
    rng = np.random.default_rng(0)
    rt = pd.to_datetime(rng.integers(e.t0.min().value, e.t0.max().value, 20000))
    base = feats(rt.values)
    f.to_parquet(f"{work}/entry_features.parquet"); base.to_parquet(f"{work}/base_features.parquet")
    cols = [k for k in f.columns if k.startswith(("r", "d_")) and k not in ("year",)]
    rows = []
    for k in cols:
        rows.append(dict(feat=k,
                         long_med=np.nanmedian(f.loc[f.side > 0, k]), short_med=np.nanmedian(f.loc[f.side < 0, k]),
                         base_med=np.nanmedian(base[k]),
                         signed_med=np.nanmedian(f[k] * f.side)))
    print(pd.DataFrame(rows).round(4).to_string())
    for k in ["vol15_x", "pos24h", "pos7d", "dvol"]:
        print(k, "long", np.nanmedian(f.loc[f.side > 0, k]).round(3), "short", np.nanmedian(f.loc[f.side < 0, k]).round(3),
              "base", np.nanmedian(base[k]).round(3))
