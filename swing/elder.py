"""Rulebook 15: Alexander Elder's triple screen (weekly tide, daily wave, buy-stop entry).

Screen 1: the last completed week's MACD(12,26,9) histogram rose.
Screen 2: today's 2-day Force Index is below zero (a dip).
Screen 3: a buy-stop one cent above today's high, lowered to each next day's
high for up to 3 days while screens 1 and 2 still hold. Stop one cent under
the lower of the last two lows; sell at the next open after a week whose
histogram fell; 40 days at most. From /mnt/project-files/rulebooks/15-엘더-삼중스크린.md.

Weeks end on the week's last trading day; a week counts as complete at that
day's close, so a Friday close can use its own week [추가].
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .events import Panel, Trade, buy_stop_fill, walk

TITLE = "엘더 삼중 스크린"
RULEBOOK = "15"
START = "2010-01-01"
PERIODS = ["2010", "2018"]
SLOTS = 10
NEEDS = ()
PIT_OK = True

TRIES, MAX_DAYS = 3, 40


def _ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(span=n, min_periods=n, adjust=False).mean()


def _hist(close: pd.DataFrame) -> pd.DataFrame:
    macd = _ema(close, 12) - _ema(close, 26)
    return macd - _ema(macd, 9)


def weekly_tables(p: Panel) -> dict:
    """Weekly MACD histogram and 13-week EMA, mapped to each day as of its close (last complete week)."""
    if "elder_w" in p._cache:
        return p._cache["elder_w"]
    pos = pd.Series(np.arange(len(p.dates)), index=p.dates)
    last_day = pos.groupby(p.dates.to_period("W-FRI")).max().to_numpy()
    cols = p.tickers + ["__SPY__"]
    allc = np.hstack([p.C, p.spy[:, None]])
    wc = pd.DataFrame(allc[last_day], columns=cols)
    h, e13 = _hist(wc).to_numpy(), _ema(wc, 13).to_numpy()
    week_of_day = np.searchsorted(last_day, np.arange(len(p.dates)), side="right") - 1   # last week ended by day i

    def at(x, lag):
        w = week_of_day - lag
        out = np.full((len(p.dates), x.shape[1]), np.nan)
        ok = w >= 0
        out[ok] = x[w[ok]]
        return out
    t = {"h0": at(h, 0), "h1": at(h, 1), "e0": at(e13, 0), "e1": at(e13, 1),
         "week_end": np.isin(np.arange(len(p.dates)), last_day)}
    p._cache["elder_w"] = t
    return t


def daily_tables(p: Panel) -> dict:
    if "elder_d" in p._cache:
        return p._cache["elder_d"]
    C = pd.DataFrame(p.C)
    fi = (C.diff() * pd.DataFrame(p.V)).ewm(span=2, min_periods=2, adjust=False).mean().to_numpy()
    e13 = _ema(C, 13).to_numpy()
    dh = _hist(C).to_numpy()
    lo5 = pd.DataFrame(p.L).rolling(5, min_periods=5).min()
    hi5 = pd.DataFrame(p.H).rolling(5, min_periods=5).max()
    k = ((C - lo5) / (hi5 - lo5) * 100).rolling(3, min_periods=3).mean().to_numpy()
    t = {"fi": fi, "e13": e13, "dh": dh, "stoch": k}
    p._cache["elder_d"] = t
    return t


def rule(p: Panel, ext: dict, *, tide: str = "macd", wave: str = "force", exit: str = "weekly",
         spy_tide: bool = False) -> list[Trade]:
    w, d = weekly_tables(p), daily_tables(p)
    C, O, H, L = p.C, p.O, p.H, p.L
    n, m = C.shape
    up_h = w["h0"] > w["h1"]
    up_e = w["e0"] > w["e1"]
    screen1 = {"macd": up_h, "ema": up_e, "impulse": up_h & up_e}[tide][:, :m]
    if spy_tide:
        screen1 = screen1 & (w["h0"][:, m] > w["h1"][:, m])[:, None]
    screen2 = {"force": d["fi"] < 0, "stoch": d["stoch"] < 30, "value": C < d["e13"]}[wave]
    setup = screen1 & screen2 & p.elig
    down_week = (w["h0"] < w["h1"])[:, :m] & w["week_end"][:, None]
    e13 = d["e13"]
    red = np.zeros_like(setup)
    red[1:] = (e13[1:] < e13[:-1]) & (d["dh"][1:] < d["dh"][:-1])
    stats = ext.setdefault("_elder", {})
    orders = fills = 0
    out = []
    for k in range(m):
        busy = -1
        days = np.flatnonzero(setup[:, k])
        for s in days:
            if s <= busy or s + 1 >= n:
                continue
            orders += 1
            level = H[s, k] + 0.01
            got = None
            for q in range(s + 1, min(s + 1 + TRIES, n)):
                fill = buy_stop_fill(O[q, k], H[q, k], level)
                if fill is not None:
                    got = (q, fill)
                    break
                if not setup[q, k]:                            # screens no longer hold: cancel
                    break
                if np.isfinite(H[q, k]):
                    level = min(level, H[q, k] + 0.01)
            if got is None:
                busy = max(busy, s)
                continue
            fills += 1
            q, ep = got
            stop = np.nanmin(L[max(q - 2, 0):q, k]) - 0.01
            if not np.isfinite(stop) or stop >= ep:
                continue
            kw = {}
            if exit == "weekly":
                kw["open_rule"] = lambda j: "주간 MACD 히스토그램 하락" if down_week[j, k] else None
            elif exit == "impulse":
                kw["close_rule"] = lambda j: "일봉 임펄스 빨강" if red[j, k] and not red[j - 1, k] else None
            elif exit == "target":
                kw["target"] = ep + 2 * (ep - stop)
            legs = walk(p, k, q, ep, at_close=False, stop=stop, max_day=q + MAX_DAYS, **kw)
            tr = Trade(k, q, ep, legs, score=-d["fi"][s, k] / max(p.C[s, k] * p.V[s, k], 1))
            out.append(tr)
            busy = tr.last()
    stats[(tide, wave, exit, spy_tide)] = (orders, fills)
    return out


RULES = [
    ("T1", "T1 주간 MACD 히스토그램↑ + Force Index(2)<0 + 고가 돌파 대기 주문", partial(rule)),
    ("T2", "T2 1번 화면 = 13주 지수평균↑", partial(rule, tide="ema")),
    ("T3", "T3 1번 화면 = 주봉 임펄스 녹색", partial(rule, tide="impulse")),
    ("T4", "T4 2번 화면 = 스토캐스틱(5,3)<30", partial(rule, wave="stoch")),
    ("T5", "T5 2번 화면 = 종가 < 13일 지수평균", partial(rule, wave="value")),
    ("T6", "T6 매도 = 일봉 임펄스 빨강 전환 종가", partial(rule, exit="impulse")),
    ("T7", "T7 매도 = 2R 익절 / 손절", partial(rule, exit="target")),
    ("T8", "T8 SPY 주봉도 1번 화면 통과할 때만", partial(rule, spy_tide=True)),
]


def extra(p: Panel, ext: dict, frames: dict[str, pd.DataFrame], split: pd.Timestamp) -> list[str]:
    lines = ["## 대기 주문 체결률", ""]
    for (tide, wave, ex, spy), (o, f) in ext.get("_elder", {}).items():
        lines.append(f"- 1번 {tide}, 2번 {wave}, 매도 {ex}{', SPY 주봉' if spy else ''}: 주문 {o}번 중 체결 {f}번 "
                     f"({f / max(o, 1) * 100:.0f}%)")
    return lines
