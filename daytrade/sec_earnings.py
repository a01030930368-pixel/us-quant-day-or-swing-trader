"""b03 변형 G6용: SEC 공시에서 실적 발표(8-K 항목 2.02) 시각을 모은다.

SEC가 무료로 주는 전체 공시 목록(submissions.zip)에서, 지금 상장된 회사(company_tickers.json)의
8-K 중 항목에 2.02(실적 발표)가 있는 것의 접수 시각을 뽑는다. 상장폐지된 회사는 빠진다(한계).
-> earnings.parquet (ticker, accepted: 미국 동부시간 기준 접수 시각)
"""
import argparse
import io
import json
import os
import time
import urllib.request
import zipfile

import pandas as pd

UA = {"User-Agent": os.environ.get("SEC_USER_AGENT", "us-quant-day-or-swing-trader backtest research bot")}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"


def get(url, path=None):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900) as r:
                if path is None:
                    return r.read()
                with open(path, "wb") as f:
                    while chunk := r.read(1 << 22):
                        f.write(chunk)
                return path
        except Exception as e:
            print("retry", url, e, flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed {url}")


def rows_from(block, tickers):
    out = []
    for form, acc, items in zip(block.get("form", []), block.get("acceptanceDateTime", []), block.get("items", [])):
        if form == "8-K" and "2.02" in (items or ""):
            for tk in tickers:
                out.append((tk, acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prep_b03/earnings.parquet")
    ap.add_argument("--tmp", default="/tmp/sec")
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tk = json.loads(get(TICKERS_URL))
    cik_tickers = {}
    for v in tk.values():
        cik_tickers.setdefault(int(v["cik_str"]), []).append(v["ticker"].replace("-", "."))
    print("companies with tickers", len(cik_tickers), flush=True)
    zpath = get(BULK_URL, os.path.join(a.tmp, "submissions.zip"))
    print("zip size", os.path.getsize(zpath), flush=True)
    rows = []
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.startswith("CIK"):
                continue
            cik = int(name[3:13])
            if cik not in cik_tickers:
                continue
            j = json.loads(z.read(name))
            block = j.get("filings", {}).get("recent", j)  # 본 파일은 filings.recent, 추가 쪽 파일은 바로 목록
            rows += rows_from(block, cik_tickers[cik])
    df = pd.DataFrame(rows, columns=["ticker", "acc"]).drop_duplicates()
    # acceptanceDateTime은 끝에 Z가 붙어 있지만 EDGAR 접수 시각(동부시간)으로 본다. 아래 시각 분포로 확인
    df["accepted"] = pd.to_datetime(df.acc.str.slice(0, 19))
    print("8-K 2.02 filings", len(df), "tickers", df.ticker.nunique(), flush=True)
    print("hour histogram:\n", df.accepted.dt.hour.value_counts().sort_index().to_string(), flush=True)
    df[["ticker", "accepted"]].to_parquet(a.out)


if __name__ == "__main__":
    main()
