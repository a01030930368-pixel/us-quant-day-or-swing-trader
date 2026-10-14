"""규칙서 b19(20% 넘는 갭 첫 1분 매수)·b20(장중 8-K 공시 직후 급등)·b24(마감 30분 낙폭과대 반등)용 자료 뽑기.

한 해 단위로 월 파일(앞해 12월 포함, 20일 평균용)을 받아 장전 + 정규장 1분봉을 읽고:
  feats_YYYY.parquet: 고른 종목-날짜의 하루 정보 (전날 종가 pc, 시가 o, 20일 평균 거래대금 adv·거래량 adv_sh,
                      장전 거래대금 pm_dv·고가 pm_high, 마감 30분 전 가격 c_cut·거래대금 dv_cut (60분 전 c_cut60·dv_cut60), 다음 날 종가 nc, 반나절 장 여부 half, kind)
  b19_YYYY.parquet: 갭 ≥ +15%, 전날 종가 ≥ 2달러, 분할로 보이지 않는 날의 정규장 1분봉
  b24_YYYY.parquet: 마감 30분 전 가격이 전날 종가(또는 시가) 대비 -8% 이하 / 전날 종가 대비 +8% 이상, 그때까지 거래대금 ≥ 2,000만 달러,
                    전날 종가 ≥ 5달러인 날의 마감 60분 1분봉
  b20_YYYY.parquet: 장중(9:35~14:30) 8-K가 접수된 종목-날짜(전날 종가 ≥ 2달러)와, 대조군으로 공시 없이 5분 안에 3% 넘게 오른
                    종목-날짜(하루 최대 15개, 무작위)의 정규장 1분봉
  prof_YYYY.parquet: 시장 전체 1분마다 거래량 비중 (같은 시간대 거래량 추정용)
  spy_YYYY.parquet: SPY 하루 종가
"""
import argparse
import os
import time

import duckdb
import pandas as pd

from prep import download
from prep_b03 import SPLIT_RATIOS, load_month


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--sec", default="sec/8k.parquet")
    ap.add_argument("--last-month", default="2026-03")
    ap.add_argument("--out", default="prep_big")
    ap.add_argument("--tmp", default="/tmp/ohlcv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    con = duckdb.connect(os.path.join(a.tmp, "work_big.duckdb"))
    con.execute("SET memory_limit='5GB'; SET preserve_insertion_order=false")
    if os.path.exists(a.sec):
        k = pd.read_parquet(a.sec, columns=["ticker", "accepted"])
        k = k[k.accepted.dt.year == a.year]
        mins = k.accepted.dt.hour * 60 + k.accepted.dt.minute - 570
        k = k.assign(d=k.accepted.dt.date, t=mins)[(mins >= 5) & (mins <= 300)][["ticker", "d", "t"]].drop_duplicates()
        print("intraday 8-K ticker-days", len(k), flush=True)
    else:
        print("no SEC file -> b20 events skipped", flush=True)
        k = pd.DataFrame({"ticker": pd.Series(dtype=str), "d": pd.Series(dtype="datetime64[ns]"), "t": pd.Series(dtype=int)})
    con.register("k8", k)
    con.execute("CREATE OR REPLACE TABLE ev AS SELECT DISTINCT ticker, d::DATE d, t::INTEGER t FROM k8")
    con.execute("CREATE OR REPLACE TABLE align (off INTEGER, r DOUBLE)")
    con.execute("""CREATE OR REPLACE TABLE daily (ticker VARCHAR, d DATE, o DOUBLE, c DOUBLE, dv DOUBLE, vsh DOUBLE,
                   pm_dv DOUBLE, pm_high DOUBLE, c_cut DOUBLE, dv_cut DOUBLE, c_cut60 DOUBLE, dv_cut60 DOUBLE, half BOOLEAN)""")
    con.execute("CREATE OR REPLACE TABLE feats (ticker VARCHAR, d DATE, kind VARCHAR, o DOUBLE, pc DOUBLE, adv DOUBLE, adv_sh DOUBLE, "
                "pm_dv DOUBLE, pm_high DOUBLE, c_cut DOUBLE, dv_cut DOUBLE, c_cut60 DOUBLE, dv_cut60 DOUBLE, half BOOLEAN)")
    for t in ("b19", "b24", "b20"):
        con.execute(f"CREATE OR REPLACE TABLE {t} (ticker VARCHAR, d DATE, mi INTEGER, o DOUBLE, h DOUBLE, l DOUBLE, c DOUBLE, v DOUBLE)")
    con.execute("CREATE OR REPLACE TABLE prof (mi INTEGER, v DOUBLE)")
    split = " OR ".join(f"abs(o / pc / {r} - 1) < 0.03 OR abs(o / pc * {r} - 1) < 0.03" for r in SPLIT_RATIOS)
    months = [f"{a.year - 1}-12"] + [f"{a.year}-{i:02d}" for i in range(1, 13)]
    path = os.path.join(a.tmp, "month.parquet")
    for i, mo in enumerate([m for m in months if m <= a.last_month]):
        t0 = time.time()
        if not download(mo, path):
            print(mo, "not found", flush=True)
            continue
        load_month(con, path)
        con.execute("""
            CREATE OR REPLACE TABLE sd AS
            SELECT d, max(mi) <= 215 half FROM m WHERE ticker = 'SPY' AND mi >= 0 GROUP BY d""")
        con.execute("""
            INSERT INTO daily
            SELECT m.ticker, m.d,
                   arg_min(o, mi) FILTER (WHERE mi >= 0), arg_max(c, mi) FILTER (WHERE mi >= 0),
                   sum(c * v) FILTER (WHERE mi >= 0), sum(v) FILTER (WHERE mi >= 0),
                   coalesce(sum(c * v) FILTER (WHERE mi < 0), 0), max(h) FILTER (WHERE mi < 0),
                   arg_max(c, mi) FILTER (WHERE mi >= 0 AND mi <= CASE WHEN sd.half THEN 179 ELSE 359 END),
                   sum(c * v) FILTER (WHERE mi >= 0 AND mi <= CASE WHEN sd.half THEN 179 ELSE 359 END),
                   arg_max(c, mi) FILTER (WHERE mi >= 0 AND mi <= CASE WHEN sd.half THEN 149 ELSE 329 END),
                   sum(c * v) FILTER (WHERE mi >= 0 AND mi <= CASE WHEN sd.half THEN 149 ELSE 329 END),
                   any_value(sd.half)
            FROM m JOIN sd USING (d) GROUP BY m.ticker, m.d""")
        if i == 0:
            os.remove(path)
            continue
        con.execute(f"""
            CREATE OR REPLACE TABLE f AS
            WITH x AS (
                SELECT *, lag(c) OVER (PARTITION BY ticker ORDER BY d) pc,
                       avg(dv) OVER w adv, avg(vsh) OVER w adv_sh, count(dv) OVER w n
                FROM daily WHERE o IS NOT NULL
                WINDOW w AS (PARTITION BY ticker ORDER BY d ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
            SELECT ticker, d, o, pc, adv, adv_sh, pm_dv, pm_high, c_cut, dv_cut, c_cut60, dv_cut60, half
            FROM x WHERE d >= DATE '{mo}-01' AND n >= 10 AND pc > 0 AND NOT ({split})""")
        con.execute("""CREATE OR REPLACE TABLE s19 AS SELECT * FROM f WHERE o / pc >= 1.15 AND pc >= 2""")
        con.execute("""CREATE OR REPLACE TABLE s24 AS SELECT * FROM f WHERE pc >= 5 AND (
                       (dv_cut >= 20e6 AND (c_cut / pc <= 0.92 OR c_cut / o <= 0.92 OR c_cut / pc >= 1.08))
                       OR (dv_cut60 >= 20e6 AND c_cut60 / pc <= 0.92))""")
        # 장중 8-K: 접수 직전 종가 대비 5분 안에 +3% 넘게 오른 적이 있는 날만 1분봉을 남긴다 (실행 단계에서 거래량까지 다시 확인)
        con.execute("""
            CREATE OR REPLACE TABLE evm AS
            SELECT ev.ticker, ev.d, ev.t,
                   arg_max(m.c, m.mi) FILTER (WHERE m.mi < ev.t) p0,
                   max(m.c) FILTER (WHERE m.mi >= ev.t AND m.mi < ev.t + 5) mx
            FROM ev JOIN m ON m.ticker = ev.ticker AND m.d = ev.d AND m.mi >= 0
            GROUP BY ev.ticker, ev.d, ev.t""")
        con.execute("""CREATE OR REPLACE TABLE s20 AS SELECT f.* FROM f
                       SEMI JOIN (SELECT ticker, d FROM evm WHERE mx >= 1.03 * p0) e USING (ticker, d)
                       WHERE pc >= 2 AND adv >= 1e6""")
        # 접수 시각 확인용: 모든 장중 8-K의 접수 시각 앞뒤 5분 움직임 크기
        for off in (-60, -10, -5, 0, 5, 60):
            con.execute(f"""
                INSERT INTO align
                SELECT {off}, abs(b / a - 1) FROM (
                    SELECT arg_max(m.c, m.mi) FILTER (WHERE m.mi < ev.t + {off}) a,
                           arg_max(m.c, m.mi) FILTER (WHERE m.mi < ev.t + {off} + 5) b,
                           count(*) FILTER (WHERE m.mi >= ev.t + {off} AND m.mi < ev.t + {off} + 5) n
                    FROM ev JOIN m ON m.ticker = ev.ticker AND m.d = ev.d AND m.mi >= 0
                    GROUP BY ev.ticker, ev.d, ev.t)
                WHERE n > 0 AND a > 0""")
        # 대조군: 공시 없이 5분 안에 3% 넘게 오른 날 (대략 고른 뒤 실행 단계에서 규칙대로 다시 확인)
        con.execute("""
            CREATE OR REPLACE TABLE s20c AS
            WITH r AS (
                SELECT ticker, d, c / min(c) OVER (PARTITION BY ticker, d ORDER BY mi ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) ratio
                FROM m WHERE mi >= 0 AND mi <= 305),
            hit AS (SELECT DISTINCT ticker, d FROM r WHERE ratio >= 1.03),
            cand AS (
                SELECT f.* FROM f JOIN hit USING (ticker, d) ANTI JOIN (SELECT DISTINCT ticker, d FROM ev) e USING (ticker, d)
                WHERE f.pc >= 2 AND f.adv >= 1e6),
            ranked AS (SELECT *, row_number() OVER (PARTITION BY d ORDER BY hash(ticker || d::VARCHAR)) rk FROM cand)
            SELECT * EXCLUDE (rk) FROM ranked WHERE rk <= 15""")
        for kind, tb, cond in (("b19", "s19", "m.mi >= 0"), ("b24", "s24", "m.mi >= (CASE WHEN s24.half THEN 150 ELSE 330 END)"),
                               ("b20", "s20", "m.mi >= 0"), ("b20c", "s20c", "m.mi >= 0")):
            con.execute(f"""INSERT INTO feats SELECT ticker, d, '{kind}', o, pc, adv, adv_sh, pm_dv, pm_high, c_cut, dv_cut, c_cut60, dv_cut60, half FROM {tb}""")
            dest = "b20" if kind.startswith("b20") else kind
            con.execute(f"INSERT INTO {dest} SELECT m.* FROM m JOIN {tb} USING (ticker, d) WHERE {cond}")
        con.execute("""INSERT INTO prof SELECT mi, sum(v) FROM m JOIN sd USING (d)
                       WHERE mi >= 0 AND mi < 390 AND NOT sd.half AND c >= 2 GROUP BY mi""")
        n = [con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("s19", "s24", "s20", "s20c")]
        print(f"{mo}: {time.time() - t0:.0f}s  b19 {n[0]}  b24 {n[1]}  b20 {n[2]}  control {n[3]}", flush=True)
        os.remove(path)
    # 다음 날 종가 (b20 K5 비교용, 그해 마지막 날은 비어 있음)
    con.execute("""CREATE OR REPLACE TABLE feats AS SELECT feats.*, nx.nc FROM feats LEFT JOIN (
                   SELECT ticker, d, lead(c) OVER (PARTITION BY ticker ORDER BY d) nc FROM daily) nx USING (ticker, d)""")
    for t in ("feats", "b19", "b24", "b20"):
        con.execute(f"COPY (SELECT * FROM {t} ORDER BY ticker, d) TO '{a.out}/{t}_{a.year}.parquet' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT off, avg(r) r, count(*) n FROM align GROUP BY off ORDER BY off) TO '{a.out}/align_{a.year}.parquet' (FORMAT parquet)")
    con.execute(f"COPY (SELECT mi, sum(v) v FROM prof GROUP BY mi ORDER BY mi) TO '{a.out}/prof_{a.year}.parquet' (FORMAT parquet)")
    con.execute(f"COPY (SELECT d, c FROM daily WHERE ticker = 'SPY' AND year(d) = {a.year} ORDER BY d) TO '{a.out}/spy_{a.year}.parquet' (FORMAT parquet)")


if __name__ == "__main__":
    main()
