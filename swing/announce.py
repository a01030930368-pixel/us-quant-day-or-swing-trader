"""Earnings drift measured from the real announcement day, not the later 10-Q filing.

Chapter 19's PEAD used the date of the 10-Q/10-K filing, which is often days
or weeks after the earnings press release, so part of the drift was gone
before we bought. Companies file the press release itself with the SEC as an
8-K under item 2.02 ("Results of Operations"), usually within minutes of
publishing it, and the SEC's free submissions data lists every 8-K with its
items and the time it was accepted. This module uses that as the
announcement time.

The announcement return (EAR) idea is from Brandt, Kishore, Santa-Clara and
Venkatachalam, "Earnings Announcements are Full of Surprises" (2008): the
stock's 3-day return around the announcement, minus the market's, predicts
the next 60 days as well as the earnings number does, and the two together
work better than either (about 7.5% a year for EAR alone, 12.5% combined,
long-short, 1987-2004; weaker but present in the largest 1,000 stocks).

Rules, fixed before the first run (doc chapter 23):
- Reaction day: the 8-K's acceptance date, or the next trading day when it
  was accepted at or after 16:00 New York time [근사: the SEC labels the time
  as UTC but it is New York time].
- EAR = stock return minus S&P 500 return from the close two days before the
  reaction day to the close the day after it [원전: days -1 to +1].
- Signal day = the day after the reaction day; buy at the next open
  [원전: enter on the second trading day after the announcement].
- "Top fifth" = at or above the 80th percentile of all announcement returns
  in the previous 63 trading days [원전 uses quintiles of the previous
  quarter's announcements].
- 60 trading days, 10 positions of 10%, no new buys while the S&P 500 is below
  its 200-day average (as chapter 19).
A. SUE >= 2 on the announcement day (the quarter's EPS, matched to the latest
   item 2.02 8-K between the quarter end and the 10-Q/10-K filing) [근사: we
   take the EPS from the 10-Q; the press release normally shows the same number].
B. A, and the announcement return is in the top fifth [원전: long both top quintiles].
C. Announcement return in the top fifth, no EPS needed (covers banks too).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .fundamentals import CACHE_DIR, EPS_TAGS, _get, cik_of, quarterly
from .pead import PEAD, sue_events

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
MORE_URL = "https://data.sec.gov/submissions/{name}"
EAR_LOOKBACK = 63          # trading days of past announcements for the top-fifth cut
TOP_FIFTH = 0.8
MATCH_DAYS = 60            # an 8-K more than 60 days before the 10-Q is not its release


def _earnings_8k(block: dict) -> list[str]:
    """Acceptance times of the item 2.02 8-Ks in one block of the submissions JSON."""
    forms, items = block.get("form", []), block.get("items", [])
    acc = block.get("acceptanceDateTime", [])
    return [a for f, it, a in zip(forms, items, acc) if f == "8-K" and "2.02" in (it or "")]


def load_announcements(tickers: list[str], use_cache: bool = True,
                       since: str = "2008-01-01") -> dict[str, list[str]]:
    """Item 2.02 8-K acceptance times per ticker (ISO strings, New York time), cached on disk."""
    cik = cik_of(use_cache)
    folder = CACHE_DIR / "announce"
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in tickers:
        c = cik(t)
        if c is None:
            out[t] = []
            continue
        path = folder / f"{c}.json"
        if use_cache and path.exists():
            out[t] = json.loads(path.read_text())
            continue
        sub = _get(SUBMISSIONS_URL.format(cik=c))
        times = _earnings_8k(sub.get("filings", {}).get("recent", {}))
        for f in sub.get("filings", {}).get("files", []):
            if f.get("filingTo", "9999") >= since:     # older pages: big filers have thousands of forms
                time.sleep(0.12)
                times += _earnings_8k(_get(MORE_URL.format(name=f["name"])))
        out[t] = sorted(set(a for a in times if a[:10] >= since))
        path.write_text(json.dumps(out[t]))
        time.sleep(0.12)                     # SEC limit: at most 10 requests a second
    return out


def reaction_days(accepted: list[str], index: pd.DatetimeIndex) -> pd.Series:
    """Position in `index` of the first close after each announcement, keyed by its date."""
    if not accepted:
        return pd.Series(dtype=int)
    ts = pd.to_datetime([a[:19] for a in accepted])
    after_close = (ts.hour * 60 + ts.minute) >= 16 * 60
    day = ts.normalize()
    pos = np.where(after_close, index.searchsorted(day, side="right"), index.searchsorted(day, side="left"))
    s = pd.Series(pos, index=day)
    return s[s < len(index)]


def announcement_tables(facts: dict[str, dict], announcements: dict[str, list[str]],
                        index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Two tables on each signal day (the day after the reaction day), NaN elsewhere.

    ANN: 1.0 on every earnings-announcement signal day.
    SUE_ANN: the quarter's SUE when the quarter could be matched to its 8-K.
    """
    ann = pd.DataFrame(np.nan, index=index, columns=list(announcements))
    sue = ann.copy()
    cover = {"announce": 0, "matched": 0}
    for t, accepted in announcements.items():
        react = reaction_days(accepted, index)
        if react.empty:
            continue
        cover["announce"] += 1
        sig = react + 1
        sig = sig[sig < len(index)]
        ann.iloc[sig.values, ann.columns.get_loc(t)] = 1.0
        ev = sue_events(quarterly(EPS_TAGS, facts.get(t, {}), "USD/shares")).dropna()
        hit = False
        for row in ev.itertuples(index=False):
            window = sig[(sig.index > row.end) & (sig.index <= row.filed)
                         & (sig.index >= row.filed - pd.Timedelta(days=MATCH_DAYS))]
            if len(window):
                sue.iat[window.iloc[-1], sue.columns.get_loc(t)] = row.sue
                hit = True
        cover["matched"] += int(hit)
    return ann, sue, cover


def top_fifth_cut(ear: pd.DataFrame, lookback: int = EAR_LOOKBACK, q: float = TOP_FIFTH) -> pd.Series:
    """Per day, the q-quantile of all announcement returns in the previous `lookback` days."""
    vals = ear.to_numpy()
    out = np.full(len(ear), np.nan)
    for i in range(lookback, len(ear)):
        past = vals[i - lookback:i]
        past = past[~np.isnan(past)]
        if len(past) >= 20:
            out[i] = np.quantile(past, q)
    return pd.Series(out, index=ear.index)


class AnnouncementPEAD(PEAD):
    name = "pead_announce"
    label = "실적 서프라이즈 (발표일 기준)"
    description = ("19장 규칙(SUE 2 이상, 60거래일 보유, 10종목)과 같지만, 분기보고서 공시일이 아니라 "
                   "실적 발표(8-K) 날을 기준으로 삽니다. 발표 반응일 다음 날 종가에 판단하고 그다음 날 시가에 삽니다.")

    def _prepare(self) -> None:
        self.sue = self.extra["SUE_ANN"].reindex(index=self.c.index)
        ann = self.extra["ANN"].reindex(index=self.c.index)
        r = self.c / self.c.shift(3) - 1                  # close of day -2 to close of day +1
        self.ear = r.sub(r[self.market], axis=0).where(ann.notna())
        self.cut = top_fifth_cut(self.ear[[t for t in self.ear.columns if t != self.market]])


class AnnouncementPEADReaction(AnnouncementPEAD):
    name = "pead_announce_ear"
    label = "실적 서프라이즈 + 발표 반응 상위 20%"
    description = ("위 규칙에, 발표 전날부터 다음 날까지 3일 수익률(S&P500 대비)이 최근 3개월 실적 "
                   "발표 종목 중 상위 20% 안일 때만 삽니다.")

    def signal(self, i: int) -> pd.Series:
        s = super().signal(i)
        return s[self.ear.iloc[i].reindex(s.index) >= self.cut.iloc[i]]


class AnnouncementReturn(AnnouncementPEAD):
    name = "ear"
    label = "발표 반응 상위 20%"
    description = ("EPS는 보지 않고, 실적 발표 3일 수익률(S&P500 대비)이 최근 3개월 상위 20%면 삽니다. "
                   "반응이 큰 순서로 10종목, 60거래일 보유.")

    def signal(self, i: int) -> pd.Series:
        e = self.ear.iloc[i].reindex(self.tradable)
        return e[e >= self.cut.iloc[i]]


ANNOUNCE = [AnnouncementPEAD, AnnouncementPEADReaction, AnnouncementReturn]
