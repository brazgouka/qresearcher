from __future__ import annotations

import polars as pl

from market_research.backtest import Regime
from market_research.hypotheses import HypothesisGenerator
from market_research.search.base import BaseSearch, SearchResult

__all__ = ["GridSearch", "SearchResult"]


class GridSearch(BaseSearch):
    """Exhaustively evaluates every hypothesis a `HypothesisGenerator` produces.

    See `BaseSearch` for the shared reserve -> evaluate -> record -> validate
    mechanics; this class only decides *which* candidates to evaluate — all of them.
    """

    def run(
        self,
        generator: HypothesisGenerator,
        data: pl.DataFrame,
        regime: Regime,
        seed: int = 0,
    ) -> list[SearchResult]:
        return [self._evaluate(h, data, regime, seed) for h in generator.generate()]
