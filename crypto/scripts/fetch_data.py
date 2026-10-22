"""코인 가격 데이터(1분봉)와 BitMEX 펀딩비를 받아 parquet 로 저장합니다.

GitHub Actions 에서만 돌립니다 (작업 컨테이너는 거래소 접속이 막혀 있음).
- Binance 현물 1분봉: data.binance.vision 월별 zip
- BitMEX XBTUSD/ETHUSD 펀딩비: 공개 API
"""
import io, os, sys, time, zipfile, json
import urllib.request
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "data"
os.makedirs(OUT, exist_ok=True)
SYMS = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
MONTHS = pd.period_range("2017-08", "2026-09", freq="M")
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "qvol", "n", "tbb", "tbq", "ign"]


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa
            if getattr(e, "code", None) == 404:
                return None
            time.sleep(2 ** i)
    print("FAIL", url)
    return None


for s in SYMS:
    frames = []
    for m in MONTHS:
        url = f"https://data.binance.vision/data/spot/monthly/klines/{s}/1m/{s}-1m-{m}.zip"
        b = get(url)
        if not b:
            continue
        z = zipfile.ZipFile(io.BytesIO(b))
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLS)
        t = df["open_time"].astype("int64")
        t = t.where(t < 10**14, t // 1000)  # 2025년부터 마이크로초 단위
        df = pd.DataFrame({
            "t": pd.to_datetime(t, unit="ms"),
            "o": df["open"].astype("float32"), "h": df["high"].astype("float32"),
            "l": df["low"].astype("float32"), "c": df["close"].astype("float32"),
            "v": df["qvol"].astype("float32"),
        })
        frames.append(df)
    df = pd.concat(frames).drop_duplicates("t").sort_values("t")
    for y, g in df.groupby(df["t"].dt.year):
        g.to_parquet(f"{OUT}/{s}_1m_{y}.parquet", index=False, compression="zstd")
    print(s, len(df), df["t"].min(), df["t"].max())

# BitMEX 펀딩비 (8시간마다)
for sym in ["XBTUSD", "ETHUSD"]:
    rows, start = [], 0
    while True:
        url = (f"https://www.bitmex.com/api/v1/funding?symbol={sym}&count=500&start={start}"
               "&reverse=false")
        b = get(url)
        if not b:
            break
        js = json.loads(b)
        if not js:
            break
        rows += js
        start += len(js)
        time.sleep(1.2)
    if rows:
        f = pd.DataFrame(rows)[["timestamp", "fundingRate"]]
        f.to_csv(f"{OUT}/funding_{sym}.csv", index=False)
        print("funding", sym, len(f))
    else:
        print("funding", sym, "none")
