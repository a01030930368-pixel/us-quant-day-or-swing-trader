"""규칙서 b17(IPO 둘째 날)·b18(IPO 베이스 돌파)용 자료 뽑기.

ipos.parquet(scripts/ipo_list.py)에서 그해(year) 상장한 IPO만 골라, 그해 1월 ~ 다음 해 3월 월 파일을 받아:
  1) ipod_YYYY.parquet: 상장일(0) ~ 60거래일째 정규장 하루 표 (ticker, d, k=몇 거래일째, o, h, l, c, v, dv)
  2) ipob_YYYY.parquet: 정규장 1분봉 — 둘째 날(k=1) 전부, 그리고 b18 후보일(k 5~40, 그날 고가가 상장 후 어제까지 최고가 H를 넘고,
     H가 5거래일 넘게 전, H 이후 저가 ≥ H×0.8, 어제 종가 > 첫 시가, 직전 5일 평균 거래대금 ≥ 1,000만 달러, 어제 종가 ≥ 5달러)과
     그 직전 5거래일 (같은 시간대 거래량 비교용). 열: ticker, d, mi, o, h, l, c, v
거래일이 10일(달력) 넘게 끊기면 그 뒤는 다른 종목일 수 있어 버린다.
"""
import argparse
import os
import time

import duckdb
import numpy as np
import pandas as pd

from prep import download, load_month

MAX_K = 60


def candidates(g):
    """g: 한 IPO의 하루 표(k 순). b18 후보일의 k 목록."""
    h, l, c, dv, o0 = g.h.to_numpy(), g.l.to_numpy(), g.c.to_numpy(), g.dv.to_numpy(), g.o.iloc[0]
    out = []
    for k in range(5, min(41, len(g))):
        i_h = int(np.argmax(h[:k]))
        H = h[i_h]
        if k - i_h < 5 or l[i_h:k].min() < 0.8 * H or not c[k - 1] > o0 or c[k - 1] < 5:
            continue
        if dv[k - 5:k].mean() < 10e6 or not h[k] > H:
            continue
        out.append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--ipos", default="ipo/ipos.parquet")
    ap.add_argument("--last-month", default="2026-03")
    ap.add_argument("--out", default="prep_ipo")
    ap.add_argument("--tmp", default="/tmp/ohlcv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    ipos = pd.read_parquet(a.ipos)
    ipos = ipos[pd.to_datetime(ipos.d0).dt.year == a.year][["ticker", "d0"]]
    ipos["d0"] = pd.to_datetime(ipos.d0).dt.date
    print(len(ipos), "IPOs in", a.year, flush=True)
    con = duckdb.connect(os.path.join(a.tmp, "work.duckdb"))
    con.execute("SET memory_limit='4GB'; SET preserve_insertion_order=false")
    con.register("ipos_df", ipos)
    con.execute("CREATE OR REPLACE TABLE ipos AS SELECT ticker, d0::DATE d0 FROM ipos_df")
    con.execute("CREATE OR REPLACE TABLE x (ticker VARCHAR, d DATE, mi INTEGER, o DOUBLE, h DOUBLE, l DOUBLE, c DOUBLE, v DOUBLE)")
    months = [f"{a.year}-{i:02d}" for i in range(1, 13)] + [f"{a.year + 1}-{i:02d}" for i in range(1, 4)]
    path = os.path.join(a.tmp, "month.parquet")
    for mo in [m for m in months if m <= a.last_month] if len(ipos) else []:
        t0 = time.time()
        if not download(mo, path):
            print(mo, "not found", flush=True)
            continue
        load_month(con, path)
        con.execute("""INSERT INTO x SELECT m.ticker, m.d, mi, o, h, l, c, v FROM m JOIN ipos
                       ON m.ticker = ipos.ticker AND m.d >= ipos.d0 AND m.d <= ipos.d0 + INTERVAL 120 DAY""")
        print(f"{mo}: {time.time() - t0:.0f}s rows {con.execute('SELECT count(*) FROM x').fetchone()[0]}", flush=True)
        os.remove(path)
    bars = con.execute("SELECT * FROM x ORDER BY ticker, d, mi").df()
    if bars.empty:
        bars = pd.DataFrame(columns=["ticker", "d", "mi", "o", "h", "l", "c", "v"])
    bars["d"] = pd.to_datetime(bars.d)
    daily = (bars.assign(dv=bars.c * bars.v).groupby(["ticker", "d"], sort=True)
             .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"), dv=("dv", "sum")).reset_index())
    keep_d, keep_b = [], set()
    for tk, g in daily.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        gap = g.d.diff().dt.days.fillna(0).to_numpy()
        cut = np.nonzero(gap > 10)[0]
        g = g.iloc[:cut[0] if len(cut) else len(g)].iloc[:MAX_K + 1].assign(k=lambda z: np.arange(len(z)))
        keep_d.append(g)
        ks = {1} | {j for k in candidates(g) for j in range(k - 5, k + 1)}
        keep_b.update((tk, g.d[k]) for k in ks if 0 < k < len(g) or k == 0)
    daily = pd.concat(keep_d, ignore_index=True) if keep_d else daily.assign(k=0)
    idx = pd.MultiIndex.from_frame(bars[["ticker", "d"]])
    mb = bars[idx.isin(list(keep_b))]
    print("daily rows", len(daily), "minute rows kept", len(mb), "days", len(keep_b), flush=True)
    daily.to_parquet(os.path.join(a.out, f"ipod_{a.year}.parquet"))
    mb.to_parquet(os.path.join(a.out, f"ipob_{a.year}.parquet"))


if __name__ == "__main__":
    main()
