"""규칙서 b21~b23(3배 ETF)과 b25(비트코인 관련주)용 자료 뽑기: 정해진 종목들의 장전 + 정규장 1분봉만.

한 해 단위로 월 파일을 받아 TICKERS의 4:00~15:59 1분봉을 남긴다 -> bars_YYYY.parquet (mi = 9:30부터 몇 분째, 장전은 음수)
(BGU는 2012년 이름이 SPXL로 바뀌기 전 Direxion 3배 S&P)
"""
import argparse
import os
import time

import duckdb

from prep import download

TICKERS = ("SPY", "QQQ", "IWM", "SMH", "SPXL", "BGU", "UPRO", "TQQQ", "TNA", "SOXL", "FAS", "LABU", "TECL",
           "MSTR", "COIN", "MARA", "RIOT", "CLSK", "HUT", "BITF", "GBTC", "BITO", "IBIT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--last-month", default="2026-03")
    ap.add_argument("--out", default="prep_etf")
    ap.add_argument("--tmp", default="/tmp/ohlcv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    con = duckdb.connect()
    con.execute("CREATE TABLE bars (ticker VARCHAR, d DATE, mi INTEGER, o DOUBLE, h DOUBLE, l DOUBLE, c DOUBLE, v DOUBLE)")
    tl = ", ".join(f"'{t}'" for t in TICKERS)
    path = os.path.join(a.tmp, "month.parquet")
    for mo in [f"{a.year}-{i:02d}" for i in range(1, 13)]:
        if mo > a.last_month:
            break
        t0 = time.time()
        if not download(mo, path):
            print(mo, "not found", flush=True)
            continue
        con.execute(f"""
            INSERT INTO bars
            SELECT ticker, d, mi, o, h, l, c, v FROM (
                SELECT ticker, open::DOUBLE o, high::DOUBLE h, low::DOUBLE l, close::DOUBLE c, volume::DOUBLE v,
                       CAST(timezone('America/New_York', "timestamp") AS DATE) d,
                       hour(timezone('America/New_York', "timestamp")) * 60
                         + minute(timezone('America/New_York', "timestamp")) - 570 mi
                FROM read_parquet('{path}') WHERE ticker IN ({tl}))
            WHERE mi >= -330 AND mi < 390 AND v > 0 AND c > 0""")
        n = con.execute(f"SELECT ticker, count(*) FROM bars WHERE strftime(d, '%Y-%m') = '{mo}' GROUP BY 1 ORDER BY 1").fetchall()
        print(f"{mo}: {time.time() - t0:.0f}s", " ".join(f"{t}:{k}" for t, k in n), flush=True)
        os.remove(path)
    con.execute(f"COPY (SELECT * FROM bars ORDER BY ticker, d, mi) TO '{a.out}/bars_{a.year}.parquet' "
                f"(FORMAT parquet, COMPRESSION zstd)")


if __name__ == "__main__":
    main()
