"""Legend rule sets with the earnings condition they all insisted on.

O'Neil (C in CAN SLIM) and Minervini (SEPA) buy only stocks whose latest
quarter showed strong earnings and sales growth. Weinstein and Darvas did not
require it, but Darvas later called himself a "techno-fundamentalist" and
bought only stocks with rising earnings, so we test the same gate on them.

The gate [원전, O'Neil "How to Make Money in Stocks"]:
    latest reported quarter EPS +25% or more versus the same quarter a year earlier,
    and sales +25% or more.
Minervini's own minimum is +20% EPS [원전]; we use one number for all.

The runner puts a daily True/False table in d["EarnOK"] (see fundamentals.py).
"""
from __future__ import annotations

import pandas as pd

from .legends import ONeil, Minervini, Weinstein
from .legends2 import Darvas
from .refine import WeinsteinDarvas

EPS_MIN = 0.25                       # [원전] O'Neil
SALES_MIN = 0.25                     # [원전] O'Neil (or accelerating; not used)
GATE_TEXT = " 여기에 최근 분기 EPS와 매출이 1년 전 같은 분기보다 모두 25% 이상 늘었을 때만 삽니다."


class EarningsGate:
    """Mix-in: keep only buy candidates that pass the earnings condition today."""

    def prepare(self, d: dict[str, pd.DataFrame], market: str = "SPY"):
        self.earn_ok = d["EarnOK"].reindex(index=d["Close"].index).fillna(False).astype(bool)
        return super().prepare(d, market)

    def candidates(self, i: int) -> list[str]:
        row = self.earn_ok.iloc[i]
        return [t for t in super().candidates(i) if bool(row.get(t, False))]


def gated(cls):
    return type(cls.__name__ + "Earnings", (EarningsGate, cls), {
        "name": cls.name + "_earn",
        "label": cls.label + " + 실적",
        "description": cls.description + GATE_TEXT,
    })


BASES = [Minervini, ONeil, Weinstein, Darvas, WeinsteinDarvas]
EARNINGS = [c for base in BASES for c in (base, gated(base))]
