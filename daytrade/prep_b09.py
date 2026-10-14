"""규칙서 b09(장중 급락 반등)용 자료 뽑기. 받기·하루 표는 prep_b04.py와 같다.

한 해 단위로 후보 = 전날 종가 >= 5달러, 직전 20일 평균 거래대금 >= 2,000만 달러,
그날 정규장 저가 <= 전날 종가 x 0.90, 주식분할로 보이는 갭 제외 (회사 주식 구분과 실적일은 백테스트 쪽에서)
  -> feats_YYYY.parquet (전날·그 전 종가 4개, 20일 평균 거래대금·거래량), bars_YYYY.parquet (후보의 정규장 1분봉), spy_YYYY.parquet
"""
import argparse
import os
import time

import duckdb

from prep import download
from prep_b03 import SPLIT_RATIOS, load_month
from prep_b04 import add_daily

MAX_LOW = 0.90


def select(con, month_start):
    split = " OR ".join(f"abs(o / pc * {r} - 1) < 0.03" for r in SPLIT_RATIOS)  # r:1 분할이면 시가 = 전날 종가 / r
    con.execute("DROP TABLE IF EXISTS feats")
    con.execute(f"""
        CREATE TABLE feats AS
        WITH a AS (
            SELECT *, lag(c, 1) OVER w pc, lag(c, 2) OVER w pc2, lag(c, 3) OVER w pc3, lag(c, 4) OVER w pc4,
                   avg(dv) OVER w20 adv, avg(v) OVER w20 adv_sh, count(dv) OVER w20 adv_n
            FROM daily WHERE o IS NOT NULL
            WINDOW w AS (PARTITION BY ticker ORDER BY d),
                   w20 AS (PARTITION BY ticker ORDER BY d ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT ticker, d, o, l, pc, pc2, pc3, pc4, 1 - l / pc dd, adv, adv_sh
        FROM a
        WHERE d >= DATE '{month_start}' AND adv_n >= 10 AND pc >= 5 AND adv >= 20e6
          AND l <= pc * {MAX_LOW} AND NOT ({split})""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--last-month", default="2026-03")
    ap.add_argument("--out", default="prep_b09")
    ap.add_argument("--tmp", default="/tmp/ohlcv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    con = duckdb.connect(os.path.join(a.tmp, "work_b09.duckdb"))
    con.execute("SET memory_limit='4GB'; SET preserve_insertion_order=false")
    con.execute("CREATE OR REPLACE TABLE daily (ticker VARCHAR, d DATE, o DOUBLE, h DOUBLE, l DOUBLE, c DOUBLE, "
                "v DOUBLE, dv DOUBLE, pm_dv DOUBLE, pm_high DOUBLE, pm_low DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE bars (ticker VARCHAR, d DATE, mi INTEGER, o DOUBLE, h DOUBLE, l DOUBLE, "
                "c DOUBLE, v DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE allfeats (ticker VARCHAR, d DATE, o DOUBLE, l DOUBLE, pc DOUBLE, pc2 DOUBLE, "
                "pc3 DOUBLE, pc4 DOUBLE, dd DOUBLE, adv DOUBLE, adv_sh DOUBLE)")
    months = [f"{a.year - 1}-12"] + [f"{a.year}-{i:02d}" for i in range(1, 13)]
    months = [mo for mo in months if mo <= a.last_month]
    path = os.path.join(a.tmp, "month.parquet")
    for i, mo in enumerate(months):
        t0 = time.time()
        if not download(mo, path):
            print(mo, "not found", flush=True)
            continue
        t1 = time.time()
        load_month(con, path)
        add_daily(con)
        if i > 0:
            select(con, f"{mo}-01")
            con.execute("INSERT INTO allfeats SELECT * FROM feats")
            con.execute("INSERT INTO bars SELECT m.* FROM m JOIN feats USING (ticker, d) WHERE m.mi >= 0")
            n = con.execute("SELECT count(*), count(DISTINCT d) FROM feats").fetchone()
            print(f"{mo}: download {t1 - t0:.0f}s, process {time.time() - t1:.0f}s, picked {n[0]} over {n[1]} days",
                  flush=True)
        os.remove(path)
    con.execute(f"COPY (SELECT * FROM allfeats ORDER BY d, ticker) TO '{a.out}/feats_{a.year}.parquet' "
                f"(FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT * FROM bars ORDER BY d, ticker, mi) TO '{a.out}/bars_{a.year}.parquet' "
                f"(FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT d, c FROM daily WHERE ticker = 'SPY' AND year(d) = {a.year} ORDER BY d) "
                f"TO '{a.out}/spy_{a.year}.parquet' (FORMAT parquet)")
    print("bars rows", con.execute("SELECT count(*) FROM bars").fetchone()[0])


if __name__ == "__main__":
    main()
