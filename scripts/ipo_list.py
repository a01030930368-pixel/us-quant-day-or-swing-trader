"""b16~b18용 IPO 첫날 목록: prep_b08 폴더(b08과 같은 방법) -> ipos.parquet (ticker, d0, o0, first_mi, dv60).

    python scripts/ipo_list.py --prep prep_b08 --out ipo/ipos.parquet
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_b08 import ipo_days, read_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_b08")
    ap.add_argument("--out", default="ipo/ipos.parquet")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tick, bars, spy = read_all(a.prep, "tick"), read_all(a.prep, "bars"), read_all(a.prep, "spy")
    spy = spy.drop_duplicates("d").assign(d=lambda x: pd.to_datetime(x.d)).set_index("d").c.sort_index()
    ipos, drop, _ = ipo_days(tick, bars.sort_values(["ticker", "d", "mi"]), spy)
    df = pd.DataFrame([dict(ticker=tk, d0=d, o0=day["o"][0], first_mi=int(day["mi"][0]), dv60=dv) for tk, d, day, dv in ipos])
    print(len(df), "IPOs;", drop, flush=True)
    print(df.groupby(df.d0.dt.year).size().to_string(), flush=True)
    df.to_parquet(a.out)


if __name__ == "__main__":
    main()
