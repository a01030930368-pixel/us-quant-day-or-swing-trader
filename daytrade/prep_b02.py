"""규칙서 b02(주목 종목 시가 범위 돌파)용 자료 뽑기. 받기·정규장 거르기는 prep.py와 같다.

한 해 단위로:
  1) 전 종목 하루 시가·고가·저가·종가·거래량과 첫 5/15/30/60분 거래량 (14일 평균을 내려고 앞의 12월부터)
  2) 대상 = 시가 >= 5달러, 직전 14일 평균 하루 거래량 >= 100만 주, 14일 ATR >= 0.5달러,
     직전 15일 안에 주식분할로 보이는 날(시가/전날 종가 < 0.6 또는 > 1.7)이 없음 (오늘 포함)
  3) 시간 창(5/15/30/60분)마다 상대거래량 = 오늘 그 창 거래량 / 직전 14일 같은 창 평균 거래량 (>= 1) 상위 25개
  -> feats_YYYY.parquet (뽑힌 종목-날짜와 지표), bars_YYYY.parquet (그 종목-날짜의 정규장 1분봉), spy_YYYY.parquet
"""
import argparse
import os
import time

import duckdb

from prep import download, load_month

WINDOWS = (5, 15, 30, 60)
TOP_N = 25  # 규칙서는 상위 20. 민감도 때 순위가 조금 바뀌어도 되게 여유를 둠
SPLIT_LO, SPLIT_HI = 0.6, 1.7


def add_daily(con):
    vw = ", ".join(f"sum(v) FILTER (WHERE mi < {w}) v{w}" for w in WINDOWS)
    con.execute(f"""
        INSERT INTO daily
        SELECT ticker, d, arg_min(o, mi) o, max(h) h, min(l) l, arg_max(c, mi) c, sum(v) vol, {vw}
        FROM m GROUP BY ticker, d""")


def select(con, month_start):
    avgs = ", ".join(f"avg(v{w}) OVER w14 av{w}" for w in WINDOWS)
    rv = ", ".join(f"coalesce(v{w}, 0) / nullif(av{w}, 0) rv{w}" for w in WINDOWS)
    ranks = ", ".join(f"row_number() OVER (PARTITION BY d ORDER BY rv{w} DESC) rk{w}" for w in WINDOWS)
    pick = " OR ".join(f"(rv{w} >= 1 AND rk{w} <= {TOP_N})" for w in WINDOWS)
    con.execute("DROP TABLE IF EXISTS feats")
    con.execute(f"""
        CREATE TABLE feats AS
        WITH a AS (
            SELECT *, lag(c) OVER (PARTITION BY ticker ORDER BY d) pc FROM daily),
        b AS (
            SELECT *, greatest(h, pc) - least(l, pc) tr,
                   (o / pc < {SPLIT_LO} OR o / pc > {SPLIT_HI})::INT bad
            FROM a),
        f AS (
            SELECT ticker, d, o, pc, v5, v15, v30, v60,
                   avg(tr) OVER w14 atr, avg(vol) OVER w14 avgvol, count(tr) OVER w14 n,
                   max(bad) OVER (PARTITION BY ticker ORDER BY d ROWS BETWEEN 15 PRECEDING AND CURRENT ROW) split,
                   {avgs}
            FROM b
            WINDOW w14 AS (PARTITION BY ticker ORDER BY d ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING)),
        g AS (
            SELECT *, {rv} FROM f
            WHERE d >= DATE '{month_start}' AND n = 14 AND coalesce(split, 1) = 0
              AND o >= 5 AND avgvol >= 1e6 AND atr >= 0.5),
        r AS (SELECT *, {ranks} FROM g)
        SELECT * FROM r WHERE {pick}""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--last-month", default="2026-03")
    ap.add_argument("--out", default="prep_b02")
    ap.add_argument("--tmp", default="/tmp/ohlcv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    con = duckdb.connect(os.path.join(a.tmp, "work_b02.duckdb"))
    con.execute("SET memory_limit='4GB'; SET preserve_insertion_order=false")
    vcols = ", ".join(f"v{w} DOUBLE" for w in WINDOWS)
    con.execute(f"CREATE OR REPLACE TABLE daily (ticker VARCHAR, d DATE, o DOUBLE, h DOUBLE, l DOUBLE, c DOUBLE, "
                f"vol DOUBLE, {vcols})")
    con.execute("CREATE OR REPLACE TABLE bars (ticker VARCHAR, d DATE, mi INTEGER, o DOUBLE, h DOUBLE, l DOUBLE, "
                "c DOUBLE, v DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE allfeats AS SELECT * FROM daily WHERE false")
    months = [f"{a.year - 1}-12"] + [f"{a.year}-{i:02d}" for i in range(1, 13)]
    months = [mo for mo in months if mo <= a.last_month]
    path = os.path.join(a.tmp, "month.parquet")
    first = True
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
            if first:
                con.execute("CREATE OR REPLACE TABLE allfeats AS SELECT * FROM feats")
                first = False
            else:
                con.execute("INSERT INTO allfeats SELECT * FROM feats")
            con.execute("INSERT INTO bars SELECT m.* FROM m JOIN (SELECT DISTINCT ticker, d FROM feats) USING (ticker, d)")
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
