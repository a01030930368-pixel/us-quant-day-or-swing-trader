"""Event-trade backtests for rulebooks 12-21 (strategies from papers and books).

These rules are short trades around a dated event (an earnings release, an
ex-dividend day, an index change, a chart pattern). Each rule is written as a
function that walks one stock at a time and returns every trade it would have
made, with no account limits:

    Trade(k, ei, ep, legs, score)
      k      column of the stock
      ei     entry day (position in the date index), ep the fill price
             (a buy at day ei's close has ep = close; a buy at the open has ep = open)
      legs   [(share, exit day, exit price, reason), ...]; shares add up to 1
      score  ranks same-day signals when the account has fewer free slots

A rule may only use data up to the day it decides on (the signal day's close,
or the fill day's open/high for a buy-stop), and each exit only data up to
that exit; tests/test_events.py checks this by changing the future.

Two kinds of numbers come out:
1. Per trade, every signal taken (the papers' way): mean before and after
   costs, median, win rate, the mean without the 10 best trades, by period.
2. An account: equal slots (10 x 10% unless the rule says otherwise), signals
   filled in score order while slots and cash last, marked to the close every
   day. Yearly return and drawdown are compared with holding SPY.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics

MIN_PRICE = 10.0            # [원전 공통 조건]
MIN_DOLLAR_VOL = 20e6       # 20-day average dollar volume [공통 조건]
MEMBERS = (500, 400)        # S&P 500 + 400
GATE_GROSS, GATE_N = 0.009, 30


@dataclass
class Trade:
    k: int
    ei: int
    ep: float
    legs: list = field(default_factory=list)       # (share, exit day, exit price, reason)
    score: float = 0.0
    tag: str = ""

    def gross(self) -> float:
        return sum(s * (xp / self.ep - 1) for s, _, xp, _ in self.legs)

    def net(self, cost: float) -> float:
        return sum(s * (xp * (1 - cost) / (self.ep * (1 + cost)) - 1) for s, _, xp, _ in self.legs)

    def last(self) -> int:
        return max(x for _, x, _, _ in self.legs)


class Panel:
    """Numpy arrays of one price set, shared by every rule of a run."""

    def __init__(self, d: dict[str, pd.DataFrame], market: str = "SPY", member: pd.DataFrame | None = None,
                 min_price: float = MIN_PRICE, min_dollar_vol: float = MIN_DOLLAR_VOL):
        c = d["Close"]
        self.dates = c.index
        self.tickers = [t for t in c.columns if t != market]
        t = self.tickers
        self.O, self.H, self.L, self.C, self.V = (d[f][t].to_numpy(float)
                                                  for f in ("Open", "High", "Low", "Close", "Volume"))
        self.spy = c[market].to_numpy(float) if market in c else None
        self.spy_open = d["Open"][market].to_numpy(float) if market in c else None
        if member is None:
            info = d.get("INFO")
            if info is not None:
                m = info.reindex(t)["index"].isin(MEMBERS).to_numpy()
                member = np.broadcast_to(m[None, :], self.C.shape)
            else:
                member = np.ones(self.C.shape, bool)
        else:
            member = member.reindex(index=self.dates, columns=t, fill_value=False).to_numpy(bool)
        self.member = member
        value = pd.DataFrame(self.C * self.V).rolling(20, min_periods=20).mean().to_numpy()
        self.liquid = (value >= min_dollar_vol) & (self.C >= min_price)
        self.elig = self.member & self.liquid
        self.col = {tk: k for k, tk in enumerate(t)}
        self._cache: dict = {}

    def sma(self, n: int, what: str = "C") -> np.ndarray:
        key = ("sma", n, what)
        if key not in self._cache:
            self._cache[key] = pd.DataFrame(getattr(self, what)).rolling(n, min_periods=n).mean().to_numpy()
        return self._cache[key]

    def ema(self, n: int) -> np.ndarray:
        key = ("ema", n)
        if key not in self._cache:
            self._cache[key] = pd.DataFrame(self.C).ewm(span=n, min_periods=n, adjust=False).mean().to_numpy()
        return self._cache[key]

    def spy_sma(self, n: int) -> np.ndarray:
        return pd.Series(self.spy).rolling(n, min_periods=n).mean().to_numpy()

    def series(self, k: int):
        return self.O[:, k], self.H[:, k], self.L[:, k], self.C[:, k]


def last_valid(c: np.ndarray, i: int) -> tuple[int, float]:
    """The last day at or before i with a price (a stock that stopped trading is sold there)."""
    j = i
    while j > 0 and not np.isfinite(c[j]):
        j -= 1
    return j, c[j]


STOPPED = 10                # no price for this many days = the stock stopped trading


def stopped_at(c: np.ndarray, a: int, b: int) -> int | None:
    """First day in (a, b] after which the stock has no price for STOPPED days, else None."""
    j = a + 1
    while j <= min(b, len(c) - 1):
        if not np.isfinite(c[j]) and not np.isfinite(c[j:j + STOPPED]).any():
            return j
        j += 1
    return None


def close_out(c: np.ndarray, ei: int, xi: int, reason: str, share: float = 1.0) -> tuple:
    """Sell `share` at day xi's close, or at the last close if trading stopped first (or xi is past the data)."""
    end = len(c) - 1
    if xi > end:
        xi, reason = end, "기간 끝"
    s = stopped_at(c, ei, xi)
    if s is not None:
        j, px = last_valid(c, s - 1)
        return (share, j, px, "거래 중단")
    j, px = last_valid(c, xi)
    return (share, j, px, reason)


# ---------------------------------------------------------------- account
@dataclass
class Account:
    equity: pd.Series
    exposure: pd.Series
    taken: list


def simulate(trades: list[Trade], panel: Panel, cost_bps: float, slots: int = 10,
             start: str | None = None) -> Account:
    """Fill trades into an account of equal slots, best score first on each entry day."""
    cost = cost_bps / 10_000
    C = panel.C
    n = len(panel.dates)
    first = 0 if start is None else int(panel.dates.searchsorted(pd.Timestamp(start)))
    by_day: dict[int, list[Trade]] = {}
    for tr in trades:
        if tr.ei >= first:
            by_day.setdefault(tr.ei, []).append(tr)
    cash, prev = 1.0, 1.0
    held: dict[int, dict] = {}                 # stock column -> position
    exits: dict[int, list] = {}                # day -> [(column, share of initial, price)]
    last = pd.DataFrame(C).ffill().to_numpy()
    eq, ex, taken = np.full(n, np.nan), np.zeros(n), []
    w = 1 / slots
    for i in range(first, n):
        for k, share, px in exits.pop(i, []):
            pos = held[k]
            sh = pos["shares0"] * share
            cash += sh * px * (1 - cost)
            pos["shares"] -= sh
            if pos["shares"] <= pos["shares0"] * 1e-9:
                del held[k]
        for tr in sorted(by_day.get(i, []), key=lambda x: -x.score):
            if len(held) >= slots or tr.k in held or not np.isfinite(tr.ep) or tr.ep <= 0:
                continue
            spend = min(prev * w, cash / (1 + cost))
            if spend < prev * w * 0.5:
                break
            sh = spend / tr.ep
            cash -= spend * (1 + cost)
            held[tr.k] = {"shares": sh, "shares0": sh}
            taken.append(tr)
            same_day = []
            for share, xi, xp, _ in tr.legs:
                if xi == i:
                    same_day.append((share, xp))
                else:
                    exits.setdefault(xi, []).append((tr.k, share, xp))
            for share, xp in same_day:             # bought and sold the same day (a stop on day one)
                cash += sh * share * xp * (1 - cost)
                held[tr.k]["shares"] -= sh * share
            if held[tr.k]["shares"] <= sh * 1e-9:
                del held[tr.k]
        value = cash + sum(p["shares"] * last[i, k] for k, p in held.items())
        eq[i], ex[i] = value, (1 - cash / value) if value > 0 else 0.0
        prev = value
    idx = panel.dates[first:]
    return Account(pd.Series(eq[first:], index=idx), pd.Series(ex[first:], index=idx), taken)


# ---------------------------------------------------------------- tables
def frame(trades: list[Trade], panel: Panel, cost_bps: float) -> pd.DataFrame:
    cost = cost_bps / 10_000
    spy = pd.Series(panel.spy).ffill().to_numpy()
    rows = [{"ticker": panel.tickers[t.k], "entry_date": panel.dates[t.ei], "entry_price": t.ep,
             "exit_date": panel.dates[t.last()], "gross": t.gross(), "net": t.net(cost),
             "excess": t.gross() - (spy[t.last()] / spy[t.ei] - 1),
             "days": t.last() - t.ei,
             "reason": "/".join(sorted({r for _, _, _, r in t.legs})), "score": t.score, "tag": t.tag}
            for t in trades]
    cols = ["ticker", "entry_date", "entry_price", "exit_date", "gross", "net", "excess", "days", "reason", "score",
            "tag"]
    return pd.DataFrame(rows, columns=cols)


def trade_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"거래수": 0}
    g, r = df["gross"], df["net"]
    top = r.sort_values(ascending=False).iloc[10:] if n > 10 else r.iloc[:0]
    t = r.mean() / (r.std(ddof=1) / np.sqrt(n)) if n > 2 and r.std(ddof=1) > 0 else np.nan
    ex = df["excess"] if "excess" in df else pd.Series(np.nan, index=df.index)
    return {"거래수": n, "비용 전 평균": g.mean(), "SPY 대비 평균(전)": ex.mean(), "비용 후 평균": r.mean(),
            "중앙값(후)": r.median(),
            "승률": (r > 0).mean(), "보유일": df["days"].mean(),
            "상위10 뺀 평균(후)": top.mean() if len(top) else np.nan, "t(거래)": t,
            "관문": "통과" if (g.mean() >= GATE_GROSS and n >= GATE_N) else "탈락"}


def account_stats(acc: Account, spy: pd.Series, a: pd.Timestamp, b: pd.Timestamp | None = None) -> dict:
    def cut(s):
        m = s.index >= a
        if b is not None:
            m &= s.index < b
        return s[m]
    r = cut(acc.equity.pct_change().fillna(0.0))
    s = cut(spy.pct_change().fillna(0.0))
    if len(r) < 2:
        return {}
    return {"계좌 연수익": metrics.cagr(r), "최대낙폭": metrics.max_drawdown(r),
            "투자비중": cut(acc.exposure).mean(), "SPY 연수익": metrics.cagr(s)}


PCT = {"비용 전 평균", "SPY 대비 평균(전)", "비용 후 평균", "중앙값(후)", "승률", "상위10 뺀 평균(후)", "계좌 연수익", "최대낙폭",
       "투자비중", "SPY 연수익"}


def md_table(rows: dict[str, dict]) -> str:
    cols: list[str] = []
    for r in rows.values():
        for k in r:
            if k not in cols:
                cols.append(k)
    if not cols:
        return "(자료 없음)"
    out = ["| 규칙 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in rows.items():
        cells = []
        for k in cols:
            v = r.get(k, np.nan)
            if isinstance(v, str):
                cells.append(v)
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("-")
            elif k in PCT:
                cells.append(f"{v * 100:+.2f}%" if "평균" in k or "중앙" in k else f"{v * 100:.1f}%")
            elif k == "거래수":
                cells.append(f"{int(v)}")
            else:
                cells.append(f"{v:.1f}" if k == "보유일" else f"{v:.2f}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def periods_of(bounds: list[str], end: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """['2004','2010','2018'] -> [('2004~2009', ...), ('2010~2017', ...), ('2018~', ...)]."""
    out = []
    for j, y in enumerate(bounds):
        a = pd.Timestamp(f"{y}-01-01")
        b = pd.Timestamp(f"{bounds[j + 1]}-01-01") if j + 1 < len(bounds) else end + pd.Timedelta(days=1)
        name = f"{y}~{int(bounds[j + 1]) - 1}" if j + 1 < len(bounds) else f"{y}~"
        out.append((name, a, b))
    return out


def yearly_trades(df: pd.DataFrame) -> str:
    if df.empty:
        return "(거래 없음)"
    g = df.groupby(pd.to_datetime(df["entry_date"]).dt.year)
    out = ["| 연도 | 거래수 | 비용 전 평균 | 비용 후 평균 | 승률 |", "|---|---|---|---|---|"]
    for y, x in g:
        out.append(f"| {y} | {len(x)} | {x['gross'].mean() * 100:+.2f}% | {x['net'].mean() * 100:+.2f}% | "
                   f"{(x['net'] > 0).mean() * 100:.0f}% |")
    return "\n".join(out)


def top_trades(df: pd.DataFrame, n: int = 10) -> str:
    if df.empty:
        return "(거래 없음)"
    x = df.sort_values("net", ascending=False).head(n)
    return ", ".join(f"{r.ticker} {str(r.entry_date)[:10]} {r.net * 100:+.0f}%" for r in x.itertuples())


# ---------------------------------------------------------------- one position, day by day
def walk(p: Panel, k: int, ei: int, ep: float, *, at_close: bool, stop: float | None = None,
         target: float | None = None, max_day: int | None = None, close_rule=None, open_rule=None,
         partial=None, max_reason: str = "보유 기간 끝") -> list:
    """Follow one position from its entry to the last sale and return its legs.

    at_close    True if bought at day ei's close (stops and targets start the next day),
                False if bought during day ei (open or buy-stop; that day's low and high count,
                the stop first, the cautious order)
    stop        stop price: the low touching it sells there, an open below it sells at the open
    target      profit target: the high reaching it sells there, an open above it at the open
    max_day     sell at this day's close at the latest
    close_rule  f(j) -> reason: sell the rest at day j's close
    open_rule   f(j) -> reason: decided at day j's close, sell the rest at day j+1's open
    partial     f(j, state) -> (share, reason) or None: sell a share of the first size at day j's close;
                state = {"stop": ..., "left": ...} may be changed (e.g. move the stop to the entry price)
    """
    O, H, L, C = p.series(k)
    n = len(C)
    legs, left = [], 1.0
    st = {"stop": stop, "left": 1.0, "ep": ep}
    pending = None
    last = n - 1 if max_day is None else min(max_day, n - 1)
    j = ei
    while True:
        first = j == ei
        if not (first and at_close):
            if not np.isfinite(C[j]):
                if stopped_at(C, j - 1, j) is not None:
                    jj, px = last_valid(C, j - 1)
                    legs.append((left, jj, px, "거래 중단"))
                    return legs
                if j >= last:
                    jj, px = last_valid(C, j)
                    legs.append((left, jj, px, max_reason if j == last and max_day is not None else "기간 끝"))
                    return legs
                j += 1
                continue
            o = O[j] if np.isfinite(O[j]) else C[j]
            if pending:
                legs.append((left, j, o, pending))
                return legs
            s = st["stop"]
            if s is not None:
                if not first and o <= s:
                    legs.append((left, j, o, "손절 (갭 하락)"))
                    return legs
                if L[j] <= s:
                    legs.append((left, j, min(s, o) if not first else s, "손절"))
                    return legs
            if target is not None and H[j] >= target:
                legs.append((left, j, o if (not first and o > target) else target, "익절"))
                return legs
            if partial is not None:
                st["left"] = left
                got = partial(j, st)
                if got:
                    share, why = got
                    legs.append((share, j, C[j], why))
                    left -= share
            if close_rule is not None:
                why = close_rule(j)
                if why:
                    legs.append((left, j, C[j], why))
                    return legs
            if j >= last:
                legs.append((left, j, C[j], max_reason if max_day is not None and j == max_day else "기간 끝"))
                return legs
            if open_rule is not None:
                why = open_rule(j)
                if why:
                    pending = why
        elif j >= last:
            legs.append((left, j, C[j], "기간 끝"))
            return legs
        j += 1


def one_per_stock(signals: np.ndarray, make) -> list[Trade]:
    """Walk signals (day, column) in time order per stock; skip a signal while that stock is held.

    make(s, k) -> Trade or None.
    """
    out = []
    busy: dict[int, int] = {}
    for s, k in sorted(map(tuple, np.argwhere(signals)), key=lambda x: (x[1], x[0])):
        if s <= busy.get(k, -1):
            continue
        tr = make(int(s), int(k))
        if tr is not None and tr.legs:
            out.append(tr)
            busy[k] = tr.last()
    return out


def buy_stop_fill(o: float, h: float, level: float) -> float | None:
    """Buy-stop at `level`: the open if it opens above, the level if the high reaches it, else no fill."""
    if not np.isfinite(o) or not np.isfinite(h):
        return None
    if o >= level:
        return o
    if h >= level:
        return level
    return None
