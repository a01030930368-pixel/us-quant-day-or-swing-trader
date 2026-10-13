"""규칙서 b04(쿨라매기 에피소딕 피벗 당일판)용 자료 뽑기. 받기·장전 처리는 prep_b03.py와 같다.

한 해 단위로:
  1) 전 종목 하루 정규장 시가·고가·저가·종가·거래량·거래대금, 장전 거래대금·고가·저가 -> daily_YYYY.parquet
     (20일 평균 거래량·평균 하루 변동폭, 120일 수익률(E6), 며칠 보유(E7)를 백테스트 쪽에서 계산)
  2) 후보 = 시가 >= 5달러, 직전 20일 평균 거래대금 >= 500만 달러, 갭 >= +8%, 주식분할로 보이는 갭 제외
     (가장 느슨한 기준. 실적·장전 거래·거래량 폭발·회사 주식 조건은 백테스트 쪽에서 거름)
  -> feats_YYYY.parquet, bars_YYYY.parquet (후보의 정규장 1분봉, 거래량 포함), spy_YYYY.parquet
"""
import argparse
import os
import time

import duckdb

from prep import download
from prep_b03 import SPLIT_RATIOS, load_month


def add_daily(con):
    con.execute("""
        INSERT INTO daily
        SELECT ticker, d,
               arg_min(o, mi) FILTER (WHERE mi >= 0) o, max(h) FILTER (WHERE mi >= 0) h,
               min(l) FILTER (WHERE mi >= 0) l, arg_max(c, mi) FILTER (WHERE mi >= 0) c,
               sum(v) FILTER (WHERE mi >= 0) v, sum(c * v) FILTER (WHERE mi >= 0) dv,
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
        WHERE d >= DATE '{month_start}' AND adv_n >= 10 AND o >= 5 AND adv >= 5e6
          AND o / pc - 1 >= 0.08 AND NOT ({split})""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--first-month", default="", help="이 달부터 받음 (기본: 앞 해 12월). 첫 해는 120일 수익률용으로 더 앞에서")
    ap.add_argument("--last-month", default="2026-03")
    ap.add_argument("--out", default="prep_b04")
    ap.add_argument("--tmp", default="/tmp/ohlcv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    con = duckdb.connect(os.path.join(a.tmp, "work_b04.duckdb"))
    con.execute("SET memory_limit='4GB'; SET preserve_insertion_order=false")
    con.execute("CREATE OR REPLACE TABLE daily (ticker VARCHAR, d DATE, o DOUBLE, h DOUBLE, l DOUBLE, c DOUBLE, "
                "v DOUBLE, dv DOUBLE, pm_dv DOUBLE, pm_high DOUBLE, pm_low DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE bars (ticker VARCHAR, d DATE, mi INTEGER, o DOUBLE, h DOUBLE, l DOUBLE, "
                "c DOUBLE, v DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE allfeats (ticker VARCHAR, d DATE, o DOUBLE, pc DOUBLE, gap DOUBLE, adv DOUBLE, "
                "pm_dv DOUBLE, pm_rel DOUBLE, pm_high DOUBLE, pm_low DOUBLE)")
    first = a.first_month or f"{a.year - 1}-12"
    months = []
    y, mth = int(first[:4]), int(first[5:7])
    while f"{y}-{mth:02d}" <= min(f"{a.year}-12", a.last_month):
        months.append(f"{y}-{mth:02d}")
        y, mth = (y + 1, 1) if mth == 12 else (y, mth + 1)
    path = os.path.join(a.tmp, "month.parquet")
    for mo in months:
        t0 = time.time()
        if not download(mo, path):
            print(mo, "not found", flush=True)
            continue
        t1 = time.time()
        load_month(con, path)
        add_daily(con)
        if mo.startswith(str(a.year)):
            select(con, f"{mo}-01")
            con.execute("INSERT INTO allfeats SELECT * FROM feats")
            con.execute("INSERT INTO bars SELECT m.* FROM m JOIN feats USING (ticker, d) WHERE m.mi >= 0")
            n = con.execute("SELECT count(*), count(DISTINCT d) FROM feats").fetchone()
            print(f"{mo}: download {t1 - t0:.0f}s, process {time.time() - t1:.0f}s, picked {n[0]} over {n[1]} days",
                  flush=True)
        else:
            print(f"{mo}: history only, download {t1 - t0:.0f}s", flush=True)
        os.remove(path)
    con.execute(f"COPY (SELECT * FROM allfeats ORDER BY d, ticker) TO '{a.out}/feats_{a.year}.parquet' "
                f"(FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT * FROM bars ORDER BY d, ticker, mi) TO '{a.out}/bars_{a.year}.parquet' "
                f"(FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT ticker, d, o, h, l, c, v FROM daily WHERE o IS NOT NULL ORDER BY ticker, d) "
                f"TO '{a.out}/daily_{a.year}.parquet' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT d, c FROM daily WHERE ticker = 'SPY' AND year(d) = {a.year} ORDER BY d) "
                f"TO '{a.out}/spy_{a.year}.parquet' (FORMAT parquet)")
    print("bars rows", con.execute("SELECT count(*) FROM bars").fetchone()[0])


if __name__ == "__main__":
    main()
