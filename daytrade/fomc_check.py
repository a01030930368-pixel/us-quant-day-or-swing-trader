"""daytrade/fomc_dates.csv(기억으로 만든 FOMC 정례 회의 표)를 연준 홈페이지 일정표와 대조한다.

연준 페이지에서 회의 날짜(이틀 회의면 둘째 날)와 'Press Conference' 링크 유무를 뽑아,
표에 없는 정례 회의 / 표에만 있는 회의 / 기자회견 여부가 다른 회의를 출력한다. 결과는 fomc_fed.csv로도 남긴다.
"""
import argparse
import html
import re
import time
import urllib.request

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (us-quant-day-or-swing-trader backtest research)"}
HIST = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{}.htm"
CURRENT = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
PAT = re.compile(rf"\b({MONTHS})(?:/({MONTHS}))?\s+(\d{{1,2}})(?:\s*-\s*(\d{{1,2}}))?\b")


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            print("retry", url, e, flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def text_of(page):
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", page, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))


def meetings(text, year_of):
    """year_of(pos) -> 그 위치의 연도. (날짜, 기자회견, 임시 여부) 목록."""
    out = []
    ms = list(PAT.finditer(text))
    for k, m in enumerate(ms):
        chunk = text[m.end(): ms[k + 1].start() if k + 1 < len(ms) else len(text)]
        head = chunk[:80]
        before = text[max(0, m.start() - 25): m.start()]
        if re.search(r"Released|Meeting -", before):  # 회의록 공개일 등 다른 날짜
            continue
        if not re.match(r"\s*(\*|\(unscheduled\))?\s*(Meeting|Statement|Press Conference|Minutes|Implementation|\(un)", chunk):
            continue
        y = year_of(m.start())
        if y is None:
            continue
        mon = m.group(2) or m.group(1)
        day = int(m.group(4) or m.group(3))
        try:
            d = pd.Timestamp(f"{mon} {day} {y}").date()
        except ValueError:
            continue
        out.append(dict(date=d, presser="Press Conference" in chunk[:1500],
                        unscheduled=bool(re.search(r"unscheduled|notation vote|conference call", head + chunk[:200], re.I))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="daytrade/fomc_dates.csv")
    ap.add_argument("--out", default="fomc_fed.csv")
    ap.add_argument("--first", type=int, default=2011)
    ap.add_argument("--last", type=int, default=2026)
    a = ap.parse_args()
    found = []
    cur = get(CURRENT)
    if cur:
        t = text_of(cur)
        years = [(m.start(), int(m.group(1))) for m in re.finditer(r"(20\d\d) FOMC Meetings", t)]
        yo = lambda p: next((y for s, y in reversed(years) if s <= p), None)  # noqa: E731
        found += meetings(t, yo)
    for y in range(a.first, a.last + 1):
        page = get(HIST.format(y))
        if page:
            t = text_of(page)
            found += meetings(t, lambda p, y=y: y)
    fed = pd.DataFrame(found)
    if fed.empty:
        print("연준 페이지를 읽지 못함")
        return
    fed = fed[(pd.to_datetime(fed.date).dt.year >= a.first) & (pd.to_datetime(fed.date).dt.year <= a.last)]
    fed = fed.groupby("date", as_index=False).agg(presser=("presser", "max"), unscheduled=("unscheduled", "max"))
    fed.to_csv(a.out, index=False)
    mine = pd.read_csv(a.table, parse_dates=["date"])
    mine["date"] = mine.date.dt.date
    sched = fed[~fed.unscheduled]
    m = mine.merge(sched, on="date", how="outer", suffixes=("", "_fed"), indicator=True)
    print(f"연준 페이지에서 찾은 회의 {len(fed)}개 (임시 {int(fed.unscheduled.sum())}개), 표 {len(mine)}개")
    print("표에만 있음:", ", ".join(str(d) for d in m[m._merge == "left_only"].date) or "없음")
    print("연준에만 있음(정례로 보임):", ", ".join(str(d) for d in m[m._merge == "right_only"].date) or "없음")
    both = m[m._merge == "both"]
    diff = both[both.presser.astype(bool) != both.presser_fed.astype(bool)]
    print("기자회견 여부 다름:", ", ".join(f"{r.date}(표 {int(r.presser)}, 연준 {int(r.presser_fed)})" for r in diff.itertuples()) or "없음")
    print("임시 회의:", ", ".join(str(d) for d in fed[fed.unscheduled].date) or "없음")


if __name__ == "__main__":
    main()
