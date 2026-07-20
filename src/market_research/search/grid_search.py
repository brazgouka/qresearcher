from __future__ import annotations

from dataclasses import dataclass

import polars as pl

import market_research
from market_research.backtest import Backtester, CostModel, Regime, SizingModel
from market_research.hypotheses import Hypothesis, HypothesisGenerator
from market_research.registry import TrialRegistry
from market_research.storage import content_hash
from market_research.validation import ValidationVerdict, Validator, permutation_test


@dataclass(frozen=True)
class SearchResult:
    trial_id: str
    hypothesis: Hypothesis
    verdict: ValidationVerdict


class GridSearch:
    """Exhaustively evaluates every hypothesis a `HypothesisGenerator` produces.

    Each candidate is reserved in the registry the instant it's emitted (spec §4),
    evaluated (backtest + permutation-test null model), then its result is recorded.
    The Bonferroni correction each candidate faces is read from
    `registry.count_trials()` *after* recording — a running total across this search
    and, since the registry is meant to persist, the platform's whole history — not a
    fixed "final count for this run only."
    """

    def __init__(
        self,
        backtester: Backtester,
        validator: Validator,
        registry: TrialRegistry,
        costs: CostModel,
        sizing: SizingModel,
        code_version: str = market_research.__version__,
        n_permutations: int = 300,
    ) -> None:
        self.backtester = backtester
        self.validator = validator
        self.registry = registry
        self.costs = costs
        self.sizing = sizing
        self.code_version = code_version
        self.n_permutations = n_permutations

    def run(
        self,
        generator: HypothesisGenerator,
        data: pl.DataFrame,
        regime: Regime,
        seed: int = 0,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for hypothesis in generator.generate():
            config = hypothesis.config()
            trial_id = self.registry.reserve_trial(
                content_hash("hypothesis_trial", config), config, seed, self.code_version
            )

            signal = hypothesis.positions(data)
            permutation = permutation_test(
                self.backtester,
                data,
                signal,
                self.costs,
                self.sizing,
                n_permutations=self.n_permutations,
                seed=seed,
            )
            self.registry.record_result(
                trial_id, regime, permutation.observed_sharpe, permutation.p_value
            )

            global_trial_count = self.registry.count_trials()
            verdict = self.validator.validate(permutation, global_trial_count, regime)
            results.append(SearchResult(trial_id=trial_id, hypothesis=hypothesis, verdict=verdict))

        return results
