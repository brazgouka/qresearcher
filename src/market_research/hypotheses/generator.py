from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

from market_research.hypotheses.hypothesis import Hypothesis
from market_research.hypotheses.primitives import And, Threshold


@dataclass(frozen=True)
class FeatureSpec:
    """Declares the search space for one feature: which operators and threshold
    values are worth trying against it."""

    name: str
    ops: tuple[str, ...]
    values: tuple[float, ...]


class HypothesisGenerator:
    """Enumerates every `Threshold` leaf over the declared feature specs, plus every
    pairwise `And` combination of two distinct leaves, up to `max_complexity`.

    The generator and the trial registry are co-designed (spec §4): the search space
    is reportable at all times via `search_space_size()`. That method returns the
    length of the same list `generate()` yields from — one materialized enumeration,
    not a separately-maintained formula that could drift from what's actually produced.

    OR/NOT-inclusive generation and richer primitives (crossovers, rankings — spec §4)
    are later widening steps, not this one.
    """

    def __init__(self, feature_specs: list[FeatureSpec], max_complexity: int = 3) -> None:
        if max_complexity < 1:
            raise ValueError("max_complexity must be >= 1")
        self.feature_specs = feature_specs
        self.max_complexity = max_complexity
        self._hypotheses: list[Hypothesis] = self._build()

    def _build(self) -> list[Hypothesis]:
        leaves: list[Hypothesis] = [
            Threshold(spec.name, op, value)
            for spec in self.feature_specs
            for op in spec.ops
            for value in spec.values
        ]
        hypotheses = list(leaves)

        if self.max_complexity >= 3:
            for left, right in combinations(leaves, 2):
                hypotheses.append(And(left, right))

        return hypotheses

    def search_space_size(self) -> int:
        return len(self._hypotheses)

    def generate(self) -> Iterator[Hypothesis]:
        yield from self._hypotheses
