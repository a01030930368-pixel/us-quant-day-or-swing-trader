"""5분봉 학습 재료(특징) 만들기: BTC 여러 시간대 등락·변동성·거래량·이평선·범위 위치 + ETH 대비 움직임 + 시각.

모든 특징은 그 5분봉이 끝난 시점까지의 정보만 씁니다.
사용: python features.py <데이터폴더> <출력.parquet>
"""
import glob, sys
import numpy as np
import pandas as pd


def load(data, sym):
    px = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{data}/{sym}_1m_*.parquet")))
    px = px.set_index("t").sort_index()
    px = px[~px.index.duplicated()].asfreq("1min")
    px[["o", "h", "l", "c"]] = px[["o", "h", "l", "c"]].ffill()
    px["v"] = px.v.fillna(0)
    return px


def bars5(px):
    return px.resample("5min").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()


def build(data):
    b = bars5(load(data, "BTCUSDT"))
    e = bars5(load(data, "ETHUSDT")).reindex(b.index).ffill()
    c, v = b.c, b.v
    lc = np.log(c)
    f = pd.DataFrame(index=b.index)
    sd = lc.diff().rolling(288, min_periods=144).std()           # 하루 5분 변동성
    for k, n in [("5m", 1), ("15m", 3), ("30m", 6), ("1h", 12), ("2h", 24), ("4h", 48), ("12h", 144),
                 ("1d", 288), ("3d", 864), ("7d", 2016), ("30d", 8640)]:
        r = lc - lc.shift(n)
        f["r" + k] = r
        f["z" + k] = r / (sd * np.sqrt(n))
        f["eth_rel" + k] = (np.log(e.c) - np.log(e.c).shift(n)) - r
    f["vol1d"] = sd * np.sqrt(288)
    f["vol_ratio"] = sd / lc.diff().rolling(288 * 30, min_periods=288 * 10).std()
    vm = v.rolling(288 * 7, min_periods=288).mean()
    for k, n in [("5m", 1), ("15m", 3), ("1h", 12), ("4h", 48)]:
        f["v" + k] = v.rolling(n).sum() / (vm * n)
    for k, n in [("1h", 12), ("4h", 48), ("1d", 288), ("3d", 864), ("7d", 2016), ("30d", 8640)]:
        hi, lo = b.h.rolling(n).max(), b.l.rolling(n).min()
        f["pos" + k] = (c - lo) / (hi - lo)
    for k, n in [("ma4h", 48), ("ma1d", 288), ("ma7d", 2016), ("ma50d", 14400), ("ma200d", 57600)]:
        f["d_" + k] = c / c.rolling(n, min_periods=n // 2).mean() - 1
    rng = (b.h - b.l) / c
    f["bar_range"] = rng / rng.rolling(288).median()
    f["wick_up"] = (b.h - np.maximum(b.o, c)) / (b.h - b.l).replace(0, np.nan)
    f["wick_dn"] = (np.minimum(b.o, c) - b.l) / (b.h - b.l).replace(0, np.nan)
    f["hour"] = f.index.hour
    f["dow"] = f.index.dayofweek
    f["c"] = c
    return f.astype("float32")


if __name__ == "__main__":
    build(sys.argv[1]).to_parquet(sys.argv[2])
