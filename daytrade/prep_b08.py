"""규칙서 b08(IPO 첫날)용 자료 뽑기. 한 해 단위로 월 파일(앞해 12월 포함)을 받아:

  1) tick_YYYY.parquet: 그해 종목코드마다 처음/마지막 거래일, 첫 1분봉 시가, 마지막 종가 (코드 변경·재상장 판정용)
  2) bars_YYYY.parquet: 그해 '처음 보이는' 종목(앞해 12월과 그해 앞선 날에 없던 코드)의 첫날 정규장 1분봉.
     단 첫 시가 >= 4달러, 첫 거래 후 60분 거래대금 >= 500만 달러인 것만 (규칙서 5달러/1,000만 달러보다 느슨하게).
  3) spy_YYYY.parquet: SPY 하루 종가 (비교용)
진짜 IPO인지(직전 250거래일 거래 없음, 코드 변경 아님, 스팩 아님)는 run_b08.py가 여러 해 tick 표를 모아 가린다.
"""
import argparse
import os
import time

import duckdb

from prep import download, load_month

MIN_OPEN = 4.0
MIN_DV60 = 5e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--last-month", default="2026-03")
    ap.add_argument("--out", default="prep_b08")
    ap.add_argument("--tmp", default="/tmp/ohlcv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    con = duckdb.connect(os.path.join(a.tmp, "work.duckdb"))
    con.execute("SET memory_limit='4GB'; SET preserve_insertion_order=false")
    con.execute("CREATE OR REPLACE TABLE seen (ticker VARCHAR)")
    con.execute("CREATE OR REPLACE TABLE tk (ticker VARCHAR, first_d DATE, first_o DOUBLE, last_d DATE, last_c DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE bars (ticker VARCHAR, d DATE, mi INTEGER, o DOUBLE, h DOUBLE, l DOUBLE, c DOUBLE, v DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE spy (d DATE, c DOUBLE)")
    months = [f"{a.year - 1}-12"] + [f"{a.year}-{i:02d}" for i in range(1, 13)]
    months = [mo for mo in months if mo <= a.last_month]
    path = os.path.join(a.tmp, "month.parquet")
    for i, mo in enumerate(months):
        t0 = time.time()
        if not download(mo, path):
            print(mo, "not found", flush=True)
            continue
        load_month(con, path)
        if i > 0:
            con.execute("""
                INSERT INTO tk
                SELECT ticker, min(d), arg_min(o, epoch(d) + mi * 60), max(d), arg_max(c, epoch(d) + mi * 60)
                FROM m GROUP BY ticker""")
            con.execute("""
                INSERT INTO spy SELECT d, arg_max(c, mi) FROM m WHERE ticker = 'SPY' GROUP BY d""")
            con.execute(f"""
                INSERT INTO bars
                WITH nf AS (
                    SELECT ticker, min(d) d FROM m WHERE ticker NOT IN (SELECT ticker FROM seen) GROUP BY ticker),
                day1 AS (SELECT m.* FROM m JOIN nf USING (ticker, d)),
                q AS (
                    SELECT ticker, d FROM (
                        SELECT ticker, d, min(mi) OVER (PARTITION BY ticker) m0, mi, o, c, v,
                               first_value(o) OVER (PARTITION BY ticker ORDER BY mi) o0
                        FROM day1)
                    WHERE mi < m0 + 60 GROUP BY ticker, d
                    HAVING any_value(o0) >= {MIN_OPEN} AND sum(c * v) >= {MIN_DV60})
                SELECT day1.ticker, day1.d, mi, o, h, l, c, v FROM day1 JOIN q USING (ticker, d)""")
        con.execute("INSERT INTO seen SELECT DISTINCT ticker FROM m WHERE ticker NOT IN (SELECT ticker FROM seen)")
        n = con.execute("SELECT count(DISTINCT ticker) FROM bars").fetchone()[0]
        print(f"{mo}: {time.time() - t0:.0f}s, seen {con.execute('SELECT count(*) FROM seen').fetchone()[0]}, "
              f"new-ticker days kept so far {n}", flush=True)
        os.remove(path)
    # tk는 달마다 한 줄씩 들어갔으니 해 전체로 합친다
    con.execute(f"""
        COPY (SELECT ticker, min(first_d) first_d, arg_min(first_o, first_d) first_o,
                     max(last_d) last_d, arg_max(last_c, last_d) last_c
              FROM tk GROUP BY ticker ORDER BY ticker)
        TO '{a.out}/tick_{a.year}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    con.execute(f"COPY (SELECT * FROM bars ORDER BY ticker, d, mi) TO '{a.out}/bars_{a.year}.parquet' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT * FROM spy ORDER BY d) TO '{a.out}/spy_{a.year}.parquet' (FORMAT parquet, COMPRESSION zstd)")


if __name__ == "__main__":
    main()
