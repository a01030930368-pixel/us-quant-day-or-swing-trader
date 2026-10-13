"""무료 1분봉(mito0o852/OHLCV-1m)에서 b01 백테스트에 필요한 것만 뽑아 둔다.

한 해(year) 단위로 돈다. 월 파일(약 0.4GB)을 하나씩 받아서:
  1) 전 종목 하루 거래대금(정규장) -> daily_YYYY.parquet  (20일 평균 거래대금, SPY 비교용)
  2) 후보 종목-날짜의 정규장 1분봉 -> bars_YYYY.parquet
후보 = 주가 >= 5달러, 직전 20일 평균 거래대금 >= 2,000만 달러, 그리고
       11:00 전에 시작한 3분봉 중 양봉이면서 거래대금 >= max(300만 달러, 20일 평균의 5%)인 봉이 하나라도 있는 날.
(가장 느슨한 기준이라 D1~D6과 민감도 점검이 모두 이 안에서 고를 수 있다.)

거래대금은 가격 x 거래량이라 주식분할 보정이 필요 없다. 시각은 미국 동부시간, 1분봉 시각은 그 분의 시작.
"""
import argparse
import os
import time
import urllib.request

import duckdb

URL = "https://huggingface.co/datasets/mito0o852/OHLCV-1m/resolve/main/data/ohlcv_{}.parquet"
MIN_PRICE = 5.0
MIN_ADV = 20e6
MIN_BASE_DV = 3e6
BASE_ADV_FRAC = 0.05
BASE_LAST_MIN = 90  # 3분봉 시작이 9:30+90분(=11:00) 전


def download(month, path):
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL.format(month), headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=900) as r, open(path, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            print(month, "retry", e, flush=True)
        except Exception as e:  # 네트워크 끊김
            print(month, "retry", e, flush=True)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"download failed {month}")


def load_month(con, path):
    """정규장(9:30~16:00) 1분봉만 남겨 m 테이블로. mi = 9:30부터 몇 분째."""
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
        WHERE mi >= 0 AND mi < 390 AND volume > 0 AND close > 0""")


def add_daily(con):
    con.execute("""
        INSERT INTO daily
        SELECT ticker, d, sum(c * v) dv, arg_min(o, mi) day_open, arg_max(c, mi) day_close, count(*) nbars
        FROM m GROUP BY ticker, d""")


def candidates(con, month_start):
    """이 달 후보 종목-날짜와 20일 평균 거래대금."""
    con.execute("DROP TABLE IF EXISTS cand")
    con.execute(f"""
        CREATE TABLE cand AS
        WITH adv AS (
            SELECT ticker, d, day_open,
                   avg(dv) OVER w adv, count(dv) OVER w adv_n
            FROM daily
            WINDOW w AS (PARTITION BY ticker ORDER BY d ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)),
        b3 AS (
            SELECT ticker, d, mi // 3 k, arg_min(o, mi) o, arg_max(c, mi) c, sum(c * v) dv
            FROM m WHERE mi < {BASE_LAST_MIN} GROUP BY ticker, d, k)
        SELECT DISTINCT a.ticker, a.d, a.adv
        FROM b3 JOIN adv a USING (ticker, d)
        WHERE a.d >= DATE '{month_start}' AND a.adv_n >= 10 AND a.adv >= {MIN_ADV} AND a.day_open >= {MIN_PRICE}
          AND b3.c > b3.o AND b3.dv >= greatest({MIN_BASE_DV}, {BASE_ADV_FRAC} * a.adv)""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--last-month", default="2026-03")
    ap.add_argument("--out", default="prep")
    ap.add_argument("--tmp", default="/tmp/ohlcv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    con = duckdb.connect(os.path.join(a.tmp, "work.duckdb"))
    con.execute("SET memory_limit='4GB'; SET preserve_insertion_order=false")
    con.execute("CREATE OR REPLACE TABLE daily (ticker VARCHAR, d DATE, dv DOUBLE, day_open DOUBLE, day_close DOUBLE, nbars BIGINT)")
    con.execute("CREATE OR REPLACE TABLE bars (ticker VARCHAR, d DATE, mi INTEGER, o DOUBLE, h DOUBLE, l DOUBLE, c DOUBLE, v DOUBLE, adv DOUBLE)")
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
        if i > 0:  # 앞의 12월은 20일 평균용으로만 씀
            candidates(con, f"{mo}-01")
            con.execute("""
                INSERT INTO bars SELECT m.*, cand.adv FROM m JOIN cand USING (ticker, d)""")
            n = con.execute("SELECT count(*), count(DISTINCT d) FROM cand").fetchone()
            print(f"{mo}: download {t1 - t0:.0f}s, process {time.time() - t1:.0f}s, "
                  f"candidates {n[0]} over {n[1]} days", flush=True)
        os.remove(path)
    con.execute(f"COPY (SELECT * FROM daily WHERE year(d) = {a.year} ORDER BY ticker, d) "
                f"TO '{a.out}/daily_{a.year}.parquet' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT * FROM bars ORDER BY d, ticker, mi) "
                f"TO '{a.out}/bars_{a.year}.parquet' (FORMAT parquet, COMPRESSION zstd)")
    print("bars rows", con.execute("SELECT count(*) FROM bars").fetchone()[0])


if __name__ == "__main__":
    main()
