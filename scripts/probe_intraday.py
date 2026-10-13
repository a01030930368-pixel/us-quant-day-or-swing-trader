"""무료 분봉 데이터 탐색 3차: mito0o852/OHLCV-1m 의 빠진 날·빠진 분 비율을 여러 해에 걸쳐 잰다."""
import io, urllib.request
import pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc

UA = {"User-Agent": "Mozilla/5.0 (probe)"}
TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "META", "GOOGL", "AMD", "NFLX", "INTC", "JPM", "XOM", "BA", "SPY", "QQQ",
           "MARA", "PLTR", "GME", "AMC", "RIOT", "SOFI", "F", "CSCO", "KO", "WMT", "UBER", "COIN", "SMCI", "CVNA", "UPST"]
MONTHS = ["2005-03", "2010-03", "2015-03", "2018-03", "2019-03", "2021-03", "2022-03"] + [f"2023-{m:02d}" for m in range(1, 13)] + ["2024-03", "2025-03", "2026-03"]

def load(m):
    url = f"https://huggingface.co/datasets/mito0o852/OHLCV-1m/resolve/main/data/ohlcv_{m}.parquet"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
        data = r.read()
    t = pq.read_table(io.BytesIO(data))
    n_all, n_tk = t.num_rows, len(pc.unique(t["ticker"]))
    t = t.filter(pc.is_in(t["ticker"], value_set=__import__("pyarrow").array(TICKERS)))
    return t.to_pandas(), n_all, n_tk

rows = []
for m in MONTHS:
    try:
        df, n_all, n_tk = load(m)
    except Exception as e:
        print(m, "ERR", e, flush=True); continue
    et = df.timestamp.dt.tz_convert("America/New_York")
    mins = et.dt.hour * 60 + et.dt.minute
    df = df[(mins >= 570) & (mins < 960)].assign(d=et.dt.date)
    spy_days = df[df.ticker == "SPY"].d.nunique()
    all_days = df.d.nunique()
    per = df.groupby(["ticker", "d"]).size().unstack(0)
    for tk in TICKERS:
        if tk not in per: continue
        s = per[tk].dropna()
        if s.empty: continue
        rows.append(dict(month=m, ticker=tk, days=len(s), mkt_days=all_days, short_days=int((s < 0.5 * 390).sum()),
                         med_bars=int(s.median())))
    print(f"{m}: rows {n_all:,} tickers {n_tk:,} trading days {all_days} SPY days {spy_days}", flush=True)
r = pd.DataFrame(rows)
print("\n=== per month: tickers present, median regular-session bars/day (of 390), ticker-days with < half the bars, missing ticker-days")
g = r.groupby("month").apply(lambda x: pd.Series(dict(tickers=len(x), med_bars=int(x.med_bars.median()),
        short=int(x.short_days.sum()), missing=int((x.mkt_days - x.days).sum()), tdays=int(x.days.sum()))))
print(g.to_string())
print("\n=== 2023 per ticker: med bars/day, short days, missing days")
y = r[r.month.str.startswith("2023")].groupby("ticker").agg(med=("med_bars", "median"), short=("short_days", "sum"),
        days=("days", "sum"), mkt=("mkt_days", "sum"))
y["missing"] = y.mkt - y.days
print(y.to_string())
