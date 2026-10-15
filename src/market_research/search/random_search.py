from __future__ import annotations

import numpy as np
import polars as pl

from market_research.backtest import Regime
from market_research.hypotheses import HypothesisGenerator
from market_research.search.base import BaseSearch, SearchResult

__all__ = ["RandomSearch", "SearchResult"]


class RandomSearch(BaseSearch):
    """Evaluates a random sample, without replacement, of a `HypothesisGenerator`'s
    search space, instead of every candidate.

    Effective trial counting here is identical to grid search: sampling without
    replacement from an already-fixed, non-adaptive space means the trials actually
    evaluated are exactly the trials logged — no adaptive-selection bias to account
    for. That stops being true once search becomes adaptive (Bayesian/GA), which is
    why counting "effective" trials there is called out in the spec (§4) as a harder,
    separate problem for a later increment.
    """

    def run(
        self,
        generator: HypothesisGenerator,
        n_samples: int,
        data: pl.DataFrame,
        regime: Regime,
        seed: int = 0,
    ) -> list[SearchResult]:
        population = list(generator.generate())
        if not 1 <= n_samples <= len(population):
            raise ValueError(f"n_samples must be between 1 and {len(population)}, got {n_samples}")

        rng = np.random.default_rng(seed)
        sampled_indices = rng.choice(len(population), size=n_samples, replace=False)
        return [self._evaluate(population[i], data, regime, seed) for i in sampled_indices]
