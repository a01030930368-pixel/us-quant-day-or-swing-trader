"""무료 분봉 데이터 출처 탐색 2차: 가장 유력한 HF 데이터셋을 깊게 확인한다."""
import io, json, re, urllib.request, urllib.parse
import pandas as pd, pyarrow.parquet as pq

UA = {"User-Agent": "Mozilla/5.0 (probe)"}
def get(url, raw=False):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
            d = r.read(); return r.status, d if raw else d.decode("utf-8", "replace")
    except Exception as e:
        return "ERR", str(e)
def sec(t): print("\n" + "=" * 20, t, "=" * 20, flush=True)

MITO = "mito0o852/OHLCV-1m"
def mito_month(m):
    st, d = get(f"https://huggingface.co/datasets/{MITO}/resolve/main/data/ohlcv_{m}.parquet", raw=True)
    if st != 200: print(m, st, d[:200]); return None
    df = pq.read_table(io.BytesIO(d)).to_pandas(); print(m, "bytes", len(d), "rows", len(df), "tickers", df.ticker.nunique()); return df

sec("mito README body (without split list)")
st, rd = get(f"https://huggingface.co/datasets/{MITO}/raw/main/README.md")
body = rd.split("---", 2)[-1]
print(body[:8000])
st, tree = get(f"https://huggingface.co/api/datasets/{MITO}/tree/main/data")
t = json.loads(tree); print("files", len(t), "first", t[0]["path"], "last", t[-1]["path"])
sz = {x["path"][-15:-8]: x["size"] for x in t}
for y in ["2000", "2005", "2010", "2015", "2018", "2020", "2022", "2024", "2025", "2026"]:
    print(y, {k: round(v/1e6) for k, v in sz.items() if k.startswith(y)})
st, commits = get(f"https://huggingface.co/api/datasets/{MITO}/commits/main")
if st == 200:
    for c in json.loads(commits)[:10]: print("commit", c.get("date"), c.get("title"))

# 2020-08 (AAPL 4:1 split on 8/31), 2023-03 (SIVB collapse)
for m in ["2020-08", "2023-03"]:
    sec("mito " + m)
    df = mito_month(m)
    if df is None: continue
    df["et"] = df.timestamp.dt.tz_convert("America/New_York")
    for tk in ["AAPL", "SIVB", "SPY", "TSLA", "MARA"]:
        s = df[df.ticker == tk]
        if s.empty: print(tk, "MISSING"); continue
        days = s.et.dt.date.nunique()
        print(tk, "bars", len(s), "days", days, "bars/day", round(len(s)/days), "first", s.et.min(), "last", s.et.max())
        dd = s.groupby(s.et.dt.date).agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"),
                                          v=("volume", "sum"), first=("et", "min"), last=("et", "max"), n=("open", "size"))
        print(dd.head(4).to_string()); print(dd.tail(3).to_string())
    a = df[df.ticker == "AAPL"].set_index("et")
    print("AAPL first 12 bars of a day:"); print(a.iloc[:12].to_string())
    print("share of bars outside 9:30-16:00:", round(((df.et.dt.hour*60+df.et.dt.minute < 570) | (df.et.dt.hour >= 16)).mean(), 3))

sec("other HF candidates")
for repo in ["paperswithbacktest/Stocks-1Min-Price", "sujalgawas/trade-1min-stock-5-yrs", "deerfieldgreen/stk-intraday-data-1min",
             "vessel888/us-stock-data", "vessel888/OHLCV-1m", "ggaddam/OHLCV-1m", "Rrishab/OHLCV-1m", "idleengine/OHLCV-1m",
             "siddharthmb/stocks-ohlcv", "benstaf/nasdaq_2013_2023"]:
    st, b = get(f"https://huggingface.co/api/datasets/{repo}?full=true")
    if st != 200: print(repo, st); continue
    info = json.loads(b); files = [s["rfilename"] for s in info.get("siblings", [])]
    st2, tr = get(f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true")
    tot = sum(x.get("size", 0) for x in json.loads(tr) if x.get("type") == "file") if st2 == 200 else None
    st3, rd = get(f"https://huggingface.co/datasets/{repo}/raw/main/README.md")
    rd = rd if st3 == 200 else ""
    lic = (info.get("cardData") or {}).get("license")
    print(f"\n## {repo} gated={info.get('gated')} license={lic} files={len(files)} bytes={tot} dl={info.get('downloads')}")
    print("  sample files:", files[:8], "...", files[-4:])
    txt = rd.split("---", 2)[-1] if rd.startswith("---") else rd
    print("  README:", re.sub(r"\s+", " ", txt)[:1500])

sec("Kaggle search")
for q in ["stock minute data", "1 minute stock", "intraday stock us", "sp500 minute", "nasdaq 1min"]:
    st, b = get("https://www.kaggle.com/api/v1/datasets/list?search=" + urllib.parse.quote(q) + "&sortBy=votes")
    if st != 200: print(q, st, b[:200]); continue
    for d in json.loads(b)[:20]:
        print(f"  [{q}] {d.get('titleNullable')} | {d.get('urlNullable')} | {round((d.get('totalBytesNullable') or 0)/1e6)}MB | {d.get('licenseNameNullable')} | {d.get('subtitleNullable')}")
