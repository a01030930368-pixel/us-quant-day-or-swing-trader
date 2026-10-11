"""Refinements of the two legend rule sets that made money in both periods.

Weinstein (chapter 12) and Darvas (chapter 14) were the only rule sets with
positive returns before and after 2018, but they sat mostly in cash (35% and
16% of the account invested on average). These variants change only how much
money each signal gets and how the two share one account; the buy and sell
rules are unchanged. The variants were fixed before running them, so the test
does not pick the best of many tries.
"""
from __future__ import annotations

import pandas as pd

from .legends import Weinstein
from .legends2 import Darvas
from .strategies import Strategy


class WeinsteinFocused(Weinstein):
    """Weinstein with 5 slots of 20%: his own advice for a small account."""
    name = "weinstein5"
    label = "와인스타인 5종목 집중"
    description = ("와인스타인 규칙 그대로, 최대 5종목에 20%씩 넣습니다. 와인스타인 책의 "
                   "'계좌가 작으면 5~6종목' 조언을 따릅니다.")
    max_positions = 5                # [원전] 5-6 stocks under $25K


class WeinsteinDarvas(Strategy):
    """Weinstein and Darvas signals in one account; each stock keeps its own rules."""
    name = "weinstein_darvas"
    label = "와인스타인 + 다바스 한 계좌"
    description = ("한 계좌에서 두 전략의 매수 신호를 모두 받습니다(같은 날이면 와인스타인 먼저). "
                   "산 종목은 그 신호를 낸 전략의 손절·매도 규칙을 그대로 따릅니다. 최대 6종목, "
                   "한 종목에 계좌의 1/6.")
    max_positions = 6                # [근사]

    def __init__(self):
        self.subs = [Weinstein(), Darvas()]
        self.pending: dict[str, Strategy] = {}
        self.owner: dict[str, Strategy] = {}

    def _prepare(self) -> None:
        d = {"Open": self.o, "High": self.h, "Low": self.l, "Close": self.c, "Volume": self.v}
        for s in self.subs:
            s.prepare(d, self.market)

    def candidates(self, i: int) -> list[str]:
        self.pending = {}
        for s in self.subs:
            for t in s.candidates(i):
                self.pending.setdefault(t, s)
        return list(self.pending)

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        return self.pending[t].accept_entry(t, i, price)

    def initial_stop(self, t: str, i: int, price: float) -> float | None:
        self.owner[t] = self.pending[t]          # called once, when the buy fills
        return self.owner[t].initial_stop(t, i, price)

    def update_stop(self, pos: dict, i: int) -> float | None:
        return self.owner[pos["ticker"]].update_stop(pos, i)

    def exit_reason(self, pos: dict, i: int):
        return self.owner[pos["ticker"]].exit_reason(pos, i)

    def take_profit(self, pos: dict, i: int):
        return self.owner[pos["ticker"]].take_profit(pos, i)


REFINED = [Weinstein, WeinsteinFocused, Darvas, WeinsteinDarvas]
