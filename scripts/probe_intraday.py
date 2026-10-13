"""무료 분봉 데이터 출처에 실제로 접속되는지, 무엇이 들어 있는지 확인하는 탐색 스크립트."""
import json, re, sys, urllib.request, urllib.parse

UA = {"User-Agent": "Mozilla/5.0 (probe)"}

def get(url, n=None, raw=False):
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(n) if n else r.read()
            return r.status, data if raw else data.decode("utf-8", "replace")
    except Exception as e:
        return "ERR", str(e)

def sec(t):
    print("\n" + "=" * 20, t, "=" * 20, flush=True)

def hf_dataset(repo):
    sec("HF " + repo)
    st, body = get(f"https://huggingface.co/api/datasets/{repo}?full=true")
    print("status", st)
    if st != 200:
        print(body[:500]); return
    info = json.loads(body)
    print("downloads", info.get("downloads"), "likes", info.get("likes"), "lastModified", info.get("lastModified"))
    print("cardData", json.dumps(info.get("cardData"), ensure_ascii=False)[:1500])
    files = [s["rfilename"] for s in info.get("siblings", [])]
    print("n files", len(files))
    for f in files[:60]: print("  ", f)
    if len(files) > 60: print("   ... last:", files[-20:])
    st, readme = get(f"https://huggingface.co/datasets/{repo}/raw/main/README.md")
    print("README status", st); print(readme[:6000] if isinstance(readme, str) else "")
    # sizes via tree API
    st, tree = get(f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true")
    if st == 200:
        t = json.loads(tree)
        tot = sum(x.get("size", 0) for x in t if x.get("type") == "file")
        print("tree entries", len(t), "total bytes (first page)", tot)
        for x in t[:40]: print("  ", x.get("type"), x.get("path"), x.get("size"))
    return files

def hf_search():
    sec("HF search")
    seen = {}
    for q in ["1min", "1m", "minute", "intraday", "ohlcv", "stock-minute", "us-stock", "sp500", "polygon", "nasdaq"]:
        st, body = get(f"https://huggingface.co/api/datasets?search={q}&limit=100&sort=downloads")
        if st != 200: print(q, st, body[:200]); continue
        for d in json.loads(body):
            seen[d["id"]] = (d.get("downloads"), d.get("lastModified", "")[:10])
    for k, v in sorted(seen.items(), key=lambda kv: -(kv[1][0] or 0)):
        print(f"  {k}  dl={v[0]}  mod={v[1]}")

def try_parquet(repo, path):
    sec(f"sample {repo}/{path}")
    try:
        import pyarrow.parquet as pq, io
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"
        st, data = get(url, raw=True)
        print("status", st, "bytes", len(data) if st == 200 else data[:300])
        if st != 200: return
        if path.endswith(".parquet"):
            t = pq.read_table(io.BytesIO(data)).to_pandas()
        else:
            import pandas as pd
            t = pd.read_csv(io.BytesIO(data), compression="infer")
        print(t.dtypes); print(t.head(5).to_string()); print(t.tail(3).to_string()); print("rows", len(t))
        for c in t.columns:
            if "tick" in c.lower() or "symbol" in c.lower():
                u = t[c].unique(); print("unique tickers", len(u), list(u[:30]))
    except Exception as e:
        print("ERR", e)

def page(url, n=4000):
    sec(url)
    st, body = get(url)
    print("status", st)
    if isinstance(body, str):
        txt = re.sub(r"<script.*?</script>|<style.*?</style>", " ", body, flags=re.S)
        txt = re.sub(r"<[^>]+>", " ", txt); txt = re.sub(r"\s+", " ", txt)
        print(txt[:n])

if __name__ == "__main__":
    hf_search()
    for repo in ["mito0o852/OHLCV-1m", "elkassabgi/hfdatalibrary"]:
        files = hf_dataset(repo) or []
        cands = [f for f in files if f.endswith((".parquet", ".csv", ".csv.gz"))]
        # 작은 표본 하나와, 상장폐지 종목이 있는지 보려고 이름에 SIVB/FRC/BBBY 있는 파일
        for f in cands[:1] + [f for f in cands if re.search(r"SIVB|FRC|BBBY|TWTR|ATVI", f)][:3]:
            try_parquet(repo, f)
    for u in ["https://hfdatalibrary.com/", "https://hfdatalibrary.com/pages/data",
              "https://hfdatalibrary.com/pages/download", "https://hfdatalibrary.com/pages/license",
              "https://firstratedata.com/free-intraday-data"]:
        page(u)
    sec("Alpha Vantage demo IBM 2009-01 1min")
    st, b = get("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=1min&month=2009-01&outputsize=full&apikey=demo")
    print(st, b[:800] if isinstance(b, str) else b)
    sec("Yahoo 1m")
    st, b = get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1m&range=7d")
    print(st, b[:300])
    sec("Kaggle API anon")
    st, b = get("https://www.kaggle.com/api/v1/datasets/list?search=1+minute+stock&sortBy=votes")
    print(st, b[:3000])
