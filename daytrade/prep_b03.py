"""규칙서 b03(뉴스 갭 상승 갭 앤 고)용 자료 뽑기. 받기는 prep.py와 같고, 이번에는 장전 거래(4:00~9:29)도 쓴다.

한 해 단위로:
  1) 전 종목 하루 정규장 시가·종가·거래대금, 장전 거래대금·고가·저가 (20일 평균을 내려고 앞의 12월부터)
  2) 후보 = 시가 >= 2달러, 직전 20일 평균 하루 거래대금 >= 100만 달러, 갭(시가/전날 종가 - 1) >= +4%,
     장전 거래대금 >= max(200만 달러, 평소의 5%), 주식분할로 보이는 갭은 제외 (가장 느슨한 기준, 변형별로 다시 거름)
  -> feats_YYYY.parquet, bars_YYYY.parquet (후보의 정규장 1분봉), spy_YYYY.parquet
"""
import argparse
import os
import time

import duckdb

from prep import download

SPLIT_RATIOS = (1.5, 2, 3, 4, 5, 10)  # 갭 상승으로 보이는 역분할 비율 (1:N 병합)


def load_month(con, path):
    """장전 + 정규장 1분봉. mi = 9:30부터 몇 분째 (장전은 음수, 4:00 = -330)."""
    con.execute("DROP TABLE IF EXISTS m")
    con.execute(f"""
        CREATE TABLE m AS
        SELECT ticker, d, mi, open::DOUBLE o, high::DOUBLE h, low::DOUBLE l, close::DOUBLE c, volume::DOUBLE v
        FROM (
            SELECT ticker, open, high, low, close, volume,
                   CAST(timezone('America/New_York', "timestamp") AS DATE) d,
                   hour(timezone('America/New_York', "timestamp")) * 60
                     + minute(timezone('America/New_York', "timestamp")) - 570 mi
            FROM read_parquet('{path}'))
        WHERE mi >= -330 AND mi < 390 AND volume > 0 AND close > 0""")


def add_daily(con):
    con.execute("""
        INSERT INTO daily
        SELECT ticker, d,
               arg_min(o, mi) FILTER (WHERE mi >= 0) o, arg_max(c, mi) FILTER (WHERE mi >= 0) c,
               sum(c * v) FILTER (WHERE mi >= 0) dv,
               coalesce(sum(c * v) FILTER (WHERE mi < 0), 0) pm_dv,
               max(h) FILTER (WHERE mi < 0) pm_high, min(l) FILTER (WHERE mi < 0) pm_low
        FROM m GROUP BY ticker, d""")


def select(con, month_start):
    split = " OR ".join(f"abs(o / pc / {r} - 1) < 0.03" for r in SPLIT_RATIOS)
    con.execute("DROP TABLE IF EXISTS feats")
    con.execute(f"""
        CREATE TABLE feats AS
        WITH a AS (
            SELECT *, lag(c) OVER (PARTITION BY ticker ORDER BY d) pc,
                   avg(dv) OVER w20 adv, count(dv) OVER w20 adv_n
            FROM daily WHERE o IS NOT NULL
            WINDOW w20 AS (PARTITION BY ticker ORDER BY d ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT ticker, d, o, pc, o / pc - 1 gap, adv, pm_dv, pm_dv / adv pm_rel, pm_high, pm_low
        FROM a
        WHERE d >= DATE '{month_start}' AND adv_n >= 10 AND o >= 2 AND adv >= 1e6
          AND o / pc - 1 >= 0.04 AND pm_dv >= greatest(2e6, 0.05 * adv) AND NOT ({split})""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--last-month", default="2026-03")
    ap.add_argument("--out", default="prep_b03")
    ap.add_argument("--tmp", default="/tmp/ohlcv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    con = duckdb.connect(os.path.join(a.tmp, "work_b03.duckdb"))
    con.execute("SET memory_limit='4GB'; SET preserve_insertion_order=false")
    con.execute("CREATE OR REPLACE TABLE daily (ticker VARCHAR, d DATE, o DOUBLE, c DOUBLE, dv DOUBLE, pm_dv DOUBLE, "
                "pm_high DOUBLE, pm_low DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE bars (ticker VARCHAR, d DATE, mi INTEGER, o DOUBLE, h DOUBLE, l DOUBLE, "
                "c DOUBLE, v DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE allfeats (ticker VARCHAR, d DATE, o DOUBLE, pc DOUBLE, gap DOUBLE, adv DOUBLE, "
                "pm_dv DOUBLE, pm_rel DOUBLE, pm_high DOUBLE, pm_low DOUBLE)")
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
