"""규칙서 b10~b15(급등주·갭 종목 장중 매매)가 함께 쓰는 자료 뽑기. 받기·하루 표는 prep_b04.py와 같다.

한 해 단위로, 전 종목 하루 표(정규장 시가·고가·저가·종가·거래량·거래대금, 장전 고가)에서 후보를 고른다.
(회사 주식 구분과 실적일은 백테스트 쪽에서 SEC 자료로)
  kind 'up'  (b10·b12·b13): 전날 종가 1~50달러, 그날 정규장 고가 >= 전날 종가 x 1.10, 그날 거래대금 >= 500만 달러
  kind 'd2'  (b11): 전날 종가 >= 그 전날 종가 x 1.10, 전날 거래량 >= 20일 평균 x 3, 전날 종가 1~50달러, 전날 거래대금 >= 1,000만 달러
  kind 'ear' (b14): 전날 시가 >= 그 전날 종가 x 1.05, 전날 종가 >= 전날 시가, 전날 거래량 >= 20일 평균 x 3, 전날 종가 >= 5달러
  kind 'r2g' (b15): 시가가 전날 종가 대비 -2% ~ -10%, 시가 > 전날 저가, 전날 종가 >= 그 전날 x 0.90, 1~50달러, 20일 평균 거래대금 >= 500만 달러
주식분할로 보이는 갭(분할 비율 ±3%)은 뺀다.
  -> feats_YYYY.parquet (후보 하루마다 한 줄, 해당 kind 표시와 전날 값들, 20일 평균 거래대금 순위)
     bars_YYYY.parquet (후보의 정규장 1분봉), daily_YYYY.parquet (후보 종목의 하루 표, b14 3일 보유용)
     prof_YYYY.parquet (시각별 '그날 거래량 중 그때까지 나온 비율' 중앙값, 상대 거래량 계산용), spy_YYYY.parquet
"""
import argparse
import os
import time

import duckdb

from prep import download
from prep_b03 import SPLIT_RATIOS, load_month
from prep_b04 import add_daily


def select(con, month_start):
    split_o = " OR ".join(f"abs(o / pc / {r} - 1) < 0.03 OR abs(o / pc * {r} - 1) < 0.03" for r in SPLIT_RATIOS)
    split_p = " OR ".join(f"abs(po / ppc / {r} - 1) < 0.03 OR abs(po / ppc * {r} - 1) < 0.03" for r in SPLIT_RATIOS)
    con.execute("DROP TABLE IF EXISTS feats")
    con.execute(f"""
        CREATE TABLE feats AS
        WITH a AS (
            SELECT *, lag(c, 1) OVER w pc, lag(c, 2) OVER w ppc, lag(o, 1) OVER w po, lag(h, 1) OVER w ph,
                   lag(l, 1) OVER w pl, lag(v, 1) OVER w pv, lag(dv, 1) OVER w pdv,
                   avg(dv) OVER w20 adv, avg(v) OVER w20 adv_sh, count(dv) OVER w20 adv_n,
                   avg(v) OVER w20p padv_sh
            FROM daily WHERE o IS NOT NULL
            WINDOW w AS (PARTITION BY ticker ORDER BY d),
                   w20 AS (PARTITION BY ticker ORDER BY d ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
                   w20p AS (PARTITION BY ticker ORDER BY d ROWS BETWEEN 21 PRECEDING AND 2 PRECEDING)),
        b AS (
            SELECT *, rank() OVER (PARTITION BY d ORDER BY adv DESC) adv_rank,
                   pc BETWEEN 1 AND 50 AND h >= pc * 1.10 AND dv >= 5e6 AND NOT ({split_o}) k_up,
                   pc >= ppc * 1.10 AND pv >= 3 * padv_sh AND pc BETWEEN 1 AND 50 AND pdv >= 1e7 AND NOT ({split_p}) k_d2,
                   po >= ppc * 1.05 AND pc >= po AND pv >= 3 * padv_sh AND pc >= 5 AND NOT ({split_p}) k_ear,
                   o <= pc * 0.98 AND o >= pc * 0.90 AND o > pl AND pc >= ppc * 0.90 AND pc BETWEEN 1 AND 50
                     AND adv >= 5e6 AND NOT ({split_o}) k_r2g
            FROM a WHERE adv_n >= 10 AND pc IS NOT NULL AND ppc IS NOT NULL)
        SELECT ticker, d, o, h, l, c, v, dv, pm_high, pc, ppc, po, ph, pl, pv, pdv, adv, adv_sh, padv_sh, adv_rank,
               k_up, k_d2, k_ear, k_r2g
        FROM b WHERE d >= DATE '{month_start}' AND (k_up OR k_d2 OR k_ear OR k_r2g)""")


def profile(con, month):
    """시각(5분 단위)별로 '그날 정규장 거래량 중 그때까지 나온 비율'의 중앙값 (거래대금 100만 달러 이상 종목-날짜)."""
    con.execute(f"""
        INSERT INTO prof
        WITH tot AS (SELECT ticker, d, sum(v) tv FROM m WHERE mi >= 0 GROUP BY ticker, d HAVING sum(c * v) >= 1e6),
        b AS (SELECT ticker, d, mi // 5 k, sum(v) bv FROM m WHERE mi >= 0 GROUP BY ticker, d, k),
        cu AS (SELECT b.k, sum(bv) OVER (PARTITION BY b.ticker, b.d ORDER BY b.k) / tot.tv f
               FROM b JOIN tot USING (ticker, d))
        SELECT '{month}', k * 5 + 5, median(f), count(*) FROM cu GROUP BY k""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--last-month", default="2026-03")
    ap.add_argument("--out", default="prep_mom")
    ap.add_argument("--tmp", default="/tmp/ohlcv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    con = duckdb.connect(os.path.join(a.tmp, "work_mom.duckdb"))
    con.execute("SET memory_limit='5GB'; SET preserve_insertion_order=false")
    con.execute("CREATE OR REPLACE TABLE daily (ticker VARCHAR, d DATE, o DOUBLE, h DOUBLE, l DOUBLE, c DOUBLE, "
                "v DOUBLE, dv DOUBLE, pm_dv DOUBLE, pm_high DOUBLE, pm_low DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE bars (ticker VARCHAR, d DATE, mi INTEGER, o DOUBLE, h DOUBLE, l DOUBLE, "
                "c DOUBLE, v DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE prof (month VARCHAR, t INTEGER, f DOUBLE, n BIGINT)")
    allfeats = None
    months = [f"{a.year - 1}-11", f"{a.year - 1}-12"] + [f"{a.year}-{i:02d}" for i in range(1, 13)]
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
        if i > 1:  # 앞의 두 달은 20일 평균용
            select(con, f"{mo}-01")
            if allfeats is None:
                con.execute("CREATE OR REPLACE TABLE allfeats AS SELECT * FROM feats")
                allfeats = True
            else:
                con.execute("INSERT INTO allfeats SELECT * FROM feats")
            con.execute("INSERT INTO bars SELECT m.* FROM m JOIN feats USING (ticker, d) WHERE m.mi >= 0")
            if i % 3 == 2:  # 석 달에 한 번이면 충분
                profile(con, mo)
            n = con.execute("SELECT count(*), sum(k_up::INT), sum(k_d2::INT), sum(k_ear::INT), sum(k_r2g::INT) FROM feats").fetchone()
            print(f"{mo}: download {t1 - t0:.0f}s, process {time.time() - t1:.0f}s, picked {n[0]} "
                  f"(up {n[1]}, d2 {n[2]}, ear {n[3]}, r2g {n[4]})", flush=True)
        os.remove(path)
    con.execute(f"COPY (SELECT * FROM allfeats ORDER BY d, ticker) TO '{a.out}/feats_{a.year}.parquet' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT * FROM bars ORDER BY d, ticker, mi) TO '{a.out}/bars_{a.year}.parquet' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"""COPY (SELECT ticker, d, o, h, l, c FROM daily
                         WHERE o IS NOT NULL AND ticker IN (SELECT DISTINCT ticker FROM allfeats WHERE k_ear)
                         ORDER BY ticker, d) TO '{a.out}/daily_{a.year}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    con.execute(f"COPY (SELECT * FROM prof ORDER BY month, t) TO '{a.out}/prof_{a.year}.parquet' (FORMAT parquet)")
    con.execute(f"COPY (SELECT d, c FROM daily WHERE ticker = 'SPY' AND year(d) = {a.year} ORDER BY d) "
                f"TO '{a.out}/spy_{a.year}.parquet' (FORMAT parquet)")
    print("bars rows", con.execute("SELECT count(*) FROM bars").fetchone()[0])


if __name__ == "__main__":
    main()
