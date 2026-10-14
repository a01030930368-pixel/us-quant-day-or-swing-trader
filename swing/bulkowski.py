"""Rulebook 21: Thomas Bulkowski's chart patterns as machine rules (high tight flag, Eve & Eve double bottom).

Bulkowski's numbers (HTF +39% / Eve & Eve +50% average rise) are to the
"ultimate high" before costs, his own comparison figure, not a trade result;
the report shows both. Machine rules stand in for the human eye [근사].
Numbers from /mnt/project-files/rulebooks/21-불코우스키-차트패턴-하이타이트플래그-이브더블바텀.md.

Bottoms [해석]: a day whose low is the lowest of the last 20 days and lower
than each of the next 5 (so it is known 5 days later). Breakouts need the
second bottom confirmed, i.e. at least 5 days after it.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .events import Panel, Trade, walk

TITLE = "불코우스키 차트 패턴 (하이 타이트 플래그, 이브 앤 이브 이중 바닥)"
RULEBOOK = "21"
START = "2010-01-01"
PERIODS = ["2010", "2018"]
SLOTS = 10
NEEDS = ()
PIT_OK = True

POLE_DAYS, FLAG = 40, (3, 20)
HOLD = 30


def htf_signals(p: Panel, k: int, pole: float) -> list[tuple]:
    """(breakout day, flag low, pole high, pole start low) for each high tight flag."""
    H, L, C = p.H[:, k], p.L[:, k], p.C[:, k]
    n = len(C)
    lo60 = pd.Series(L).rolling(POLE_DAYS + FLAG[1] + 1, min_periods=POLE_DAYS).min().to_numpy()
    out = []
    for i in range(POLE_DAYS + FLAG[1] + 1, n):
        if not (np.isfinite(C[i]) and C[i] / lo60[i] >= pole):     # cheap filter
            continue
        a, b = i - FLAG[1] - 1, i - FLAG[0] - 1                    # the pole high is 3..20 days before i
        seg = H[a:b + 1]
        if not np.isfinite(seg).any():
            continue
        m = a + int(np.nanargmax(seg))
        top = H[m]
        flag = slice(m + 1, i)
        if np.nanmax(H[flag]) > top:
            continue
        start = np.nanmin(L[m - POLE_DAYS:m + 1])
        if not top / start >= pole:
            continue
        flag_low = np.nanmin(L[flag])
        if flag_low < 0.8 * top:
            continue
        if C[i] > top and C[i - 1] <= top:
            out.append((i, flag_low, top, start))
    return out


def bottoms(L: np.ndarray) -> np.ndarray:
    s = pd.Series(L)
    low20 = s.rolling(21, min_periods=21).min().to_numpy()
    nxt5 = s[::-1].rolling(5, min_periods=5).min()[::-1].shift(-1).to_numpy()
    return np.flatnonzero((L == low20) & (L < nxt5))


def eve(L: np.ndarray, j: int) -> bool:
    w = np.r_[L[max(j - 5, 0):j], L[j + 1:j + 6]]
    return int(np.sum(w <= L[j] * 1.03)) >= 3


def dbl_signals(p: Panel, k: int, shape: bool) -> list[tuple]:
    """(breakout day, lower bottom, confirmation price, first bottom day, second bottom day)."""
    H, L, C = p.H[:, k], p.L[:, k], p.C[:, k]
    n = len(C)
    bs = bottoms(L)
    out, used = [], -1
    for b2i, j2 in enumerate(bs):
        for j1 in bs[:b2i][::-1]:
            if j2 - j1 < 20:
                continue
            if j2 - j1 > 65:
                break
            l1, l2 = L[j1], L[j2]
            lower = min(l1, l2)
            if abs(l1 - l2) / lower > 0.04 or np.nanmin(L[j1:j2 + 1]) < lower:
                continue
            peak = np.nanmax(H[j1:j2 + 1])
            if peak < 1.10 * lower:
                continue
            if j1 < 60 or not np.nanmax(H[j1 - 60:j1 + 1]) >= l1 / 0.8:
                continue
            if shape and not (eve(L, j1) and eve(L, j2)):
                continue
            for i in range(j2 + 1, min(j2 + 61, n)):
                if L[i] < lower:
                    break
                if C[i] > peak:
                    if i >= j2 + 5 and i > used:
                        out.append((i, lower, peak, j1, j2))
                        used = i
                    break
            break                                       # only the closest valid first bottom
    return out


def rule(p: Panel, ext: dict, *, kind: str = "htf", pole: float = 1.9, shape: bool = True, vol: bool = False,
         market: bool = False, ma_exit: bool = False) -> list[Trade]:
    C = p.C
    avg50 = pd.DataFrame(p.V).rolling(50, min_periods=50).mean().shift(1).to_numpy()
    spy_ok = p.spy > p.spy_sma(200)
    ma20, ma50 = p.sma(20), p.sma(50)
    found = ext.setdefault("_patterns", {})
    out = []
    for k in range(C.shape[1]):
        busy = -1
        sigs = htf_signals(p, k, pole) if kind == "htf" else dbl_signals(p, k, shape)
        for sg in sigs:
            i = sg[0]
            if i <= busy or not p.elig[i, k]:
                continue
            if vol and not p.V[i, k] >= 1.5 * avg50[i, k]:
                continue
            if market and not spy_ok[i]:
                continue
            ep = C[i, k]
            if kind == "htf":
                legs = walk(p, k, i, ep, at_close=True, stop=sg[1], max_day=i + HOLD,
                            close_rule=lambda j: "종가 < 20일 평균" if C[j, k] < ma20[j, k] else None)
            elif ma_exit:
                legs = walk(p, k, i, ep, at_close=True, stop=sg[1] - 0.01, max_day=i + 60,
                            close_rule=lambda j: "종가 < 50일 평균" if C[j, k] < ma50[j, k] else None)
            else:
                legs = walk(p, k, i, ep, at_close=True, stop=sg[1] - 0.01, target=sg[2] + (sg[2] - sg[1]),
                            max_day=i + HOLD)
            tr = Trade(k, i, ep, legs)
            out.append(tr)
            if not (vol or market or ma_exit):
                found.setdefault((kind, pole, shape), []).append((k, sg))
            busy = tr.last()
    return out


RULES = [
    ("P1", "P1 하이 타이트 플래그: 40일 안 +90% 깃대, 3~20일 깃발(되돌림 20% 이내), 종가 돌파", partial(rule)),
    ("P2", "P2 이브 앤 이브 이중 바닥: 두 바닥 4% 안, 20~65일 간격, 사이 반등 10%+, 둥근 바닥, 확인가 종가 돌파",
     partial(rule, kind="dbl")),
    ("P3", "P3 P2에서 모양 조건 없음 (모든 이중 바닥)", partial(rule, kind="dbl", shape=False)),
    ("P4", "P4 P2 + 확인 날 거래량 50일 평균 1.5배", partial(rule, kind="dbl", vol=True)),
    ("P5a", "P5 P1 + SPY 200일 평균 위", partial(rule, market=True)),
    ("P5b", "P5 P2 + SPY 200일 평균 위", partial(rule, kind="dbl", market=True)),
    ("P6", "P6 P2 익절 없이 종가 < 50일 평균 매도 (최대 60일)", partial(rule, kind="dbl", ma_exit=True)),
    ("P7", "P7 P1 깃대 +50% (덜 엄격)", partial(rule, pole=1.5)),
]

SPARK = "▁▂▃▄▅▆▇█"


def spark(x: np.ndarray) -> str:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return ""
    lo, hi = x.min(), x.max()
    return "".join(SPARK[int((v - lo) / (hi - lo + 1e-12) * 7.999)] for v in x)


def ultimate(C: np.ndarray, i: int, k: int) -> float:
    """Bulkowski's ultimate high: the highest close before a 20% fall from it (or 250 days)."""
    top = C[i, k]
    for j in range(i + 1, min(i + 251, len(C))):
        c = C[j, k]
        if not np.isfinite(c):
            continue
        top = max(top, c)
        if c <= 0.8 * top:
            break
    return top / C[i, k] - 1


def extra(p: Panel, ext: dict, frames: dict[str, pd.DataFrame], split: pd.Timestamp) -> list[str]:
    lines = ["## 불코우스키 숫자와 비교 (돌파 뒤 '가장 높이 간 곳', 비용 전)", "",
             "돌파 종가부터 20% 떨어지기 전 가장 높은 종가까지(최대 250일). 불코우스키: HTF 평균 +39%, 이브 앤 이브 +50%.", "",
             "| 패턴 | 개수 | 평균 | 중앙값 | 해마다 개수 |", "|---|---|---|---|---|"]
    C = p.C
    for (kind, pole, shape), items in ext.get("_patterns", {}).items():
        uh = np.array([ultimate(C, sg[0], k) for k, sg in items])
        yrs = pd.Series([p.dates[sg[0]].year for _, sg in items]).value_counts().sort_index()
        name = f"HTF 깃대 +{(pole - 1) * 100:.0f}%" if kind == "htf" else f"이중 바닥{' (이브)' if shape else ' (모양 무관)'}"
        lines.append(f"| {name} | {len(uh)} | {np.mean(uh) * 100:+.1f}% | {np.median(uh) * 100:+.1f}% | "
                     + ", ".join(f"{y}:{c}" for y, c in yrs.items()) + " |")
    rng = np.random.default_rng(21)
    lines += ["", "## 눈으로 확인할 무작위 패턴 (종가 모양, ▁ 낮음 █ 높음)", ""]
    for (kind, pole, shape), items in ext.get("_patterns", {}).items():
        if (kind, pole) == ("htf", 1.5) or (kind == "dbl" and not shape):
            continue
        pick = rng.choice(len(items), size=min(10 if kind == "htf" else 20, len(items)), replace=False) if items else []
        lines.append(f"**{'하이 타이트 플래그' if kind == 'htf' else '이브 앤 이브 이중 바닥'}** ({len(pick)}개)")
        lines.append("")
        for q in sorted(pick):
            k, sg = items[q]
            t = p.tickers[k]
            i = sg[0]
            if kind == "htf":
                a = max(i - 60, 0)
                desc = (f"{t} 돌파 {p.dates[i].date()} 종가 {C[i, k]:.2f}, 깃대 시작 저가 {sg[3]:.2f} → 고가 {sg[2]:.2f}, "
                        f"깃발 저가 {sg[1]:.2f}")
            else:
                a = max(sg[3] - 20, 0)
                desc = (f"{t} 바닥 {p.dates[sg[3]].date()}·{p.dates[sg[4]].date()} (낮은 바닥 {sg[1]:.2f}), 확인가 {sg[2]:.2f}, "
                        f"돌파 {p.dates[i].date()} 종가 {C[i, k]:.2f}")
            lines.append(f"- {desc}  `{spark(C[a:i + 6, k])}`")
        lines.append("")
    return lines
