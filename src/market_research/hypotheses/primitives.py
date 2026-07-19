from __future__ import annotations

import operator
from collections.abc import Callable

import polars as pl

from market_research.hypotheses.hypothesis import Hypothesis

_OPS: dict[str, Callable[[pl.Series, float], pl.Series]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
}


class Threshold:
    """Leaf predicate: `feature <op> value` -> 1.0 where true, 0.0 elsewhere.

    A null feature value (not yet available, e.g. still inside its lookback window) is
    never silently treated as a signal — it resolves to 0.0 (flat), not True or an
    error, so an unresolved feature can't fabricate a position.
    """

    def __init__(self, feature: str, op: str, value: float) -> None:
        if op not in _OPS:
            raise ValueError(f"Unsupported operator: {op!r}, expected one of {sorted(_OPS)}")
        self.feature = feature
        self.op = op
        self.value = value

    def positions(self, features: pl.DataFrame) -> pl.Series:
        condition = _OPS[self.op](features[self.feature], self.value)
        return condition.fill_null(False).cast(pl.Float64).rename("position")

    def complexity(self) -> int:
        return 1

    def config(self) -> dict:
        return {"type": "Threshold", "feature": self.feature, "op": self.op, "value": self.value}


class And:
    """Both children active -> 1.0, else 0.0 (multiplication of 0/1-valued series)."""

    def __init__(self, left: Hypothesis, right: Hypothesis) -> None:
        self.left = left
        self.right = right

    def positions(self, features: pl.DataFrame) -> pl.Series:
        return (self.left.positions(features) * self.right.positions(features)).rename("position")

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()

    def config(self) -> dict:
        return {"type": "And", "left": self.left.config(), "right": self.right.config()}


class Or:
    """Either child active -> 1.0, else 0.0 (inclusive-or over 0/1-valued series)."""

    def __init__(self, left: Hypothesis, right: Hypothesis) -> None:
        self.left = left
        self.right = right

    def positions(self, features: pl.DataFrame) -> pl.Series:
        left = self.left.positions(features)
        right = self.right.positions(features)
        return (left + right - left * right).rename("position")

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()

    def config(self) -> dict:
        return {"type": "Or", "left": self.left.config(), "right": self.right.config()}


class Not:
    """Inverts a 0/1-valued position series."""

    def __init__(self, child: Hypothesis) -> None:
        self.child = child

    def positions(self, features: pl.DataFrame) -> pl.Series:
        return (1.0 - self.child.positions(features)).rename("position")

    def complexity(self) -> int:
        return 1 + self.child.complexity()

    def config(self) -> dict:
        return {"type": "Not", "child": self.child.config()}
