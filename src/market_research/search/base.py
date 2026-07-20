from __future__ import annotations

from dataclasses import dataclass

import polars as pl

import market_research
from market_research.backtest import Backtester, CostModel, Regime, SizingModel
from market_research.hypotheses import Hypothesis
from market_research.registry import TrialRegistry
from market_research.storage import content_hash
from market_research.validation import ValidationVerdict, Validator, permutation_test


@dataclass(frozen=True)
class SearchResult:
    trial_id: str
    hypothesis: Hypothesis
    verdict: ValidationVerdict


class BaseSearch:
    """Shared per-candidate evaluation loop: reserve -> backtest + permutation test ->
    record -> validate.

    Concrete search strategies (`GridSearch`, `RandomSearch`, ...) differ only in
    *which* and *how many* candidates from a `HypothesisGenerator` they choose to
    evaluate — the evaluation mechanics, and the registry/validator bookkeeping that
    makes multiple-testing correction possible, are identical and shared here so they
    can never silently drift apart between strategies.
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

    def _evaluate(
        self, hypothesis: Hypothesis, data: pl.DataFrame, regime: Regime, seed: int
    ) -> SearchResult:
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
        return SearchResult(trial_id=trial_id, hypothesis=hypothesis, verdict=verdict)
