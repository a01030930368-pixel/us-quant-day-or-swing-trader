"""규칙서 b25용 비트코인 가격: 바이낸스 공개 자료(data.binance.vision) BTCUSDT 1분봉 (2017-08 ~).

받을 수 없으면 코인베이스 공개 API 1시간봉(BTC-USD)으로 대신한다.
-> btc.parquet (t: 미국 동부시간 1분봉(또는 1시간봉) 시작 시각, c: 종가, src: 'binance-1m' | 'coinbase-1h')
"""
import argparse
import io
import json
import os
import time
import urllib.request
import zipfile

import pandas as pd

BINANCE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-{}.zip"
COINBASE = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600&start={}&end={}"
UA = {"User-Agent": "Mozilla/5.0 (research)"}


def fetch(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 403, 451):
                raise
            print("retry", url, e, flush=True)
        except Exception as e:
            print("retry", url, e, flush=True)
        time.sleep(5 * (k + 1))
    raise RuntimeError(url)


def binance(months):
    out = []
    for mo in months:
        try:
            raw = fetch(BINANCE.format(mo))
        except Exception as e:
            print(mo, "binance failed:", e, flush=True)
            if not out:
                return None  # 처음부터 안 되면 다른 곳으로
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 4], names=["ts", "c"])
        df = df[pd.to_numeric(df.ts, errors="coerce").notna()].astype({"ts": "int64", "c": "float64"})
        unit = "us" if df.ts.iloc[0] > 1e14 else "ms"  # 2025년부터 마이크로초
        df["t"] = pd.to_datetime(df.ts, unit=unit, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        out.append(df[["t", "c"]])
        print(mo, len(df), flush=True)
    return pd.concat(out).assign(src="binance-1m") if out else None


def coinbase(start, end):
    rows, t = [], pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC")
    while t < end:
        t2 = min(t + pd.Timedelta(hours=300), end)
        j = json.loads(fetch(COINBASE.format(t.isoformat(), t2.isoformat())))
        rows += [(r[0], r[4]) for r in j]  # [time, low, high, open, close, volume]
        t = t2
        time.sleep(0.4)
    df = pd.DataFrame(rows, columns=["ts", "c"]).drop_duplicates("ts").sort_values("ts")
    df["t"] = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return df[["t", "c"]].assign(src="coinbase-1h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prep_etf/btc.parquet")
    ap.add_argument("--first", default="2017-08")
    ap.add_argument("--last", default="2026-03")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    months = [p.strftime("%Y-%m") for p in pd.period_range(a.first, a.last, freq="M")]
    df = binance(months)
    if df is None:
        print("binance unavailable -> coinbase hourly", flush=True)
        df = coinbase(f"{a.first}-01", (pd.Period(a.last, "M") + 1).start_time.strftime("%Y-%m-%d"))
    df = df.drop_duplicates("t").sort_values("t")
    print("btc rows", len(df), df.t.min(), df.t.max(), df.src.iloc[0], flush=True)
    df.to_parquet(a.out)


if __name__ == "__main__":
    main()
