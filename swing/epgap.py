"""Rulebook 13: earnings gap-up, held for days to weeks (Qullamaggie's episodic pivot, swing version).

On the earnings reaction day the stock gaps up 5%+ on 3x volume and holds the
gap. Buy at that close, stop at the day's low, sell a third on the third day
if in profit (and move the stop to the buy price), the rest when the close
falls below the 10-day average. Numbers from
/mnt/project-files/rulebooks/13-실적갭상승-스윙보유-에피소딕피벗.md.

[해석] "sell a third on day 3 and raise the stop (only if in profit)": both
happen only when day 3's close is above the buy price.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .events import Panel, Trade, buy_stop_fill, one_per_stock, walk

TITLE = "실적 갭 상승 스윙 보유 (에피소딕 피벗)"
RULEBOOK = "13"
START = "2010-01-01"
PERIODS = ["2010", "2018"]
SLOTS = 10
NEEDS = ("earnings",)
PIT_OK = False

GAP, VOL_X, MAX_DAYS, PART_DAY = 0.05, 3.0, 60, 3


def _signals(p: Panel, ext: dict, gap: float, quiet: bool, earnings: bool) -> np.ndarray:
    C, O, V = p.C, p.O, p.V
    prev = np.vstack([np.full((1, C.shape[1]), np.nan), C[:-1]])
    g = O / prev - 1
    avg50 = pd.DataFrame(V).rolling(50, min_periods=50).mean().shift(1).to_numpy()
    sig = (g >= gap) & (V >= VOL_X * avg50) & (C > O) & (C > p.sma(50)) & p.elig
    if quiet:
        before = np.full_like(C, np.nan)
        before[61:] = prev[61:] / C[:-61] - 1                  # 60 days up to the day before the gap
        sig &= before <= 0.10
    ear = np.zeros_like(sig)
    for k, r in ext["earnings"].items():
        ear[r[r < len(C)], k] = True
    near = ear.copy()                                          # G8: no release that day or the day before/after
    near[1:] |= ear[:-1]
    near[:-1] |= ear[1:]
    return sig & ear if earnings else sig & ~near


def rule(p: Panel, ext: dict, *, gap: float = GAP, ma: int = 10, next_open: bool = False, third: bool = True,
         quiet: bool = False, pullback: bool = False, earnings: bool = True) -> list[Trade]:
    sig = _signals(p, ext, gap, quiet, earnings)
    C, O, H, L = p.C, p.O, p.H, p.L
    avg = p.sma(ma)
    ma10 = p.sma(10)
    n = len(C)

    def exits(k, ei, ep, at_close, stop):
        def part(j, st):
            if third and j == ei + PART_DAY and C[j, k] > st["ep"]:
                st["stop"] = max(st["stop"], st["ep"])
                return (1 / 3, "3일째 1/3 매도")
            return None
        return walk(p, k, ei, ep, at_close=at_close, stop=stop, max_day=ei + MAX_DAYS,
                    partial=part if third else None,
                    open_rule=lambda j: f"종가 < {ma}일 평균" if C[j, k] < avg[j, k] else None)

    def make(s, k):
        stop = L[s, k]
        if pullback:                                           # G7: first dip to the 10-day line, then a higher high
            for q in range(s + 1, min(s + 11, n - 1)):
                if not np.isfinite(C[q, k]) or not np.isfinite(ma10[q, k]):
                    continue
                if abs(C[q, k] / ma10[q, k] - 1) <= 0.02:
                    level = H[q, k] + 0.01
                    fill = buy_stop_fill(O[q + 1, k], H[q + 1, k], level)
                    if fill is None:
                        return None
                    return Trade(k, q + 1, fill, exits(k, q + 1, fill, False, L[q, k] - 0.01), tag="눌림")
            return None
        if next_open:
            if s + 1 >= n or not np.isfinite(O[s + 1, k]) or O[s + 1, k] <= stop:
                return None
            return Trade(k, s + 1, O[s + 1, k], exits(k, s + 1, O[s + 1, k], False, stop))
        return Trade(k, s, C[s, k], exits(k, s, C[s, k], True, stop), score=O[s, k] / C[s - 1, k])
    return one_per_stock(sig, make)


RULES = [
    ("G1", "G1 실적 갭 +5%·거래량 3배 종가 매수, 3일째 1/3, 나머지 10일선", partial(rule)),
    ("G2", "G2 나머지 매도 = 20일 평균", partial(rule, ma=20)),
    ("G3", "G3 갭 +10% 이상", partial(rule, gap=0.10)),
    ("G4", "G4 다음 날 시가 매수", partial(rule, next_open=True)),
    ("G5", "G5 1/3 매도 없이 전량 10일선", partial(rule, third=False)),
    ("G6", "G6 갭 전 60일 상승률 10% 이하 (잠자던 종목)", partial(rule, quiet=True)),
    ("G7", "G7 첫 눌림(10일선 ±2%) 뒤 전날 고가 돌파 매수", partial(rule, pullback=True)),
    ("G8", "G8 대조군: 실적 아닌 날의 같은 갭", partial(rule, earnings=False)),
]


def extra(p: Panel, ext: dict, frames: dict[str, pd.DataFrame], split: pd.Timestamp) -> list[str]:
    lines = ["## 1차 매도 전에 손절된 비율", ""]
    for name, df in frames.items():
        if df.empty or name in ("G4", "G7"):
            continue
        r = df["reason"]
        early = (r.str.startswith("손절") & ~r.str.contains("3일째")).mean()
        lines.append(f"- {name}: {early * 100:.0f}% (신호 날 저가 손절이 1/3 매도보다 먼저)")
    return lines
