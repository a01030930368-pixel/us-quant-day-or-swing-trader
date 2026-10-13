"""규칙서 b16~b20용 SEC 공시 정리 (submissions.zip, 무료).

지금 상장된 회사(company_tickers.json)마다:
  companies.parquet: ticker, cik, sic, first_filing(첫 공시일), first_periodic(첫 10-K/10-Q/20-F/40-F 접수일),
                     first_425(합병 관련 425 공시 첫 접수일), acq_name(옛 이름에 'ACQUISITION' 같은 스팩식 이름이 있었나), periodic(정기보고서를 낸 회사인가)
  8k.parquet: ticker, accepted(미국 동부시간 접수 시각), items  (모든 8-K, 시각 없는 옛 공시는 뺌)
상장폐지된 회사는 company_tickers.json에 없어서 빠진다 (한계). 같은 코드를 나중에 다른 회사가 쓰면 잘못 이어질 수 있어
이를 쓰는 쪽에서 '그 회사 첫 공시일이 그날보다 앞서는지'로 확인한다.
"""
import argparse
import json
import os
import re
import zipfile

import pandas as pd

from sec_earnings import BULK_URL, TICKERS_URL, get

PERIODIC = {"10-K", "10-Q", "20-F", "40-F", "10-K405", "10-KSB", "10-QSB", "10-KT", "10-QT"}
ACQ = re.compile(r"ACQUISITION|SPAC|BLANK CHECK|CAPITAL CORP\b|MERGER CORP", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sec")
    ap.add_argument("--tmp", default="/tmp/sec")
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    tk = json.loads(get(TICKERS_URL))
    cik_tickers = {}
    for v in tk.values():
        cik_tickers.setdefault(int(v["cik_str"]), []).append(v["ticker"].replace("-", "."))
    print("companies with tickers", len(cik_tickers), flush=True)
    zpath = get(BULK_URL, os.path.join(a.tmp, "submissions.zip"))
    print("zip size", os.path.getsize(zpath), flush=True)
    info, k8 = {}, []
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.startswith("CIK"):
                continue
            cik = int(name[3:13])
            if cik not in cik_tickers:
                continue
            j = json.loads(z.read(name))
            r = info.setdefault(cik, dict(sic=None, first_filing=None, first_periodic=None, first_425=None, acq_name=False))
            if "filings" in j:
                r["sic"] = int(j["sic"]) if str(j.get("sic") or "").isdigit() else None
                r["acq_name"] = bool(ACQ.search(" ".join(f.get("name", "") for f in j.get("formerNames", []))))
                block = j["filings"]["recent"]
            else:
                block = j
            forms, dates = block.get("form", []), block.get("filingDate", [])
            accs, items = block.get("acceptanceDateTime", []), block.get("items", [""] * len(forms))
            for f, d, acc, it in zip(forms, dates, accs, items):
                if r["first_filing"] is None or d < r["first_filing"]:
                    r["first_filing"] = d
                if f in PERIODIC and (r["first_periodic"] is None or d < r["first_periodic"]):
                    r["first_periodic"] = d
                if f == "425" and (r["first_425"] is None or d < r["first_425"]):
                    r["first_425"] = d
                if f == "8-K" and acc and "T00:00:00" not in acc:
                    k8.append((cik, acc, it or ""))
    comp = pd.DataFrame([dict(cik=c, ticker=t, **info[c]) for c, ts in cik_tickers.items() if c in info for t in ts])
    comp["periodic"] = comp.first_periodic.notna()
    for c in ("first_filing", "first_periodic", "first_425"):
        comp[c] = pd.to_datetime(comp[c])
    comp.to_parquet(os.path.join(a.out, "companies.parquet"))
    print("companies", len(comp), "sic 6770:", int((comp.sic == 6770).sum()), "with 425:", int(comp.first_425.notna().sum()), flush=True)
    df = pd.DataFrame(k8, columns=["cik", "acc", "items"]).drop_duplicates()
    # acceptanceDateTime은 세계표준시(UTC) -> 동부시간
    df["accepted"] = pd.to_datetime(df.acc, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = df.merge(comp[["cik", "ticker"]], on="cik")[["ticker", "cik", "accepted", "items"]]
    print("8-K", len(df), "hour histogram:\n", df.accepted.dt.hour.value_counts().sort_index().to_string(), flush=True)
    df.to_parquet(os.path.join(a.out, "8k.parquet"))


if __name__ == "__main__":
    main()
