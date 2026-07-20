from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from market_research.backtest import CostModel, EventDrivenBacktester, SizingModel
from market_research.features import RSI, FeatureEngine
from market_research.hypotheses import FeatureSpec, HypothesisGenerator
from market_research.registry import DuckDBTrialRegistry
from market_research.search import RandomSearch
from market_research.validation import BonferroniPermutationValidator


def _dates(n: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]


def _synthetic_features(n: int = 150, seed: int = 21) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.012, n)
    closes = 100.0 * np.cumprod(1 + returns)
    frame = pl.DataFrame(
        {
            "symbol": ["X"] * n,
            "date": _dates(n),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * n,
        }
    )
    return FeatureEngine([RSI(period=14)]).compute(frame)


def _make_search(registry: DuckDBTrialRegistry) -> RandomSearch:
    return RandomSearch(
        EventDrivenBacktester(),
        BonferroniPermutationValidator(alpha=0.05),
        registry,
        CostModel(),
        SizingModel(),
        code_version="test",
        n_permutations=50,
    )


def test_random_search_evaluates_exactly_n_samples() -> None:
    data = _synthetic_features()
    generator = HypothesisGenerator(
        [FeatureSpec("rsi_14", ("<", ">"), (20.0, 25.0, 30.0, 70.0, 75.0))], max_complexity=1
    )
    registry = DuckDBTrialRegistry()

    results = _make_search(registry).run(
        generator, n_samples=3, data=data, regime="in_sample", seed=1
    )

    assert len(results) == 3
    assert registry.count_trials() == 3


def test_random_search_samples_without_replacement() -> None:
    data = _synthetic_features()
    generator = HypothesisGenerator(
        [FeatureSpec("rsi_14", ("<", ">"), (20.0, 25.0, 30.0, 70.0, 75.0))], max_complexity=1
    )
    registry = DuckDBTrialRegistry()

    results = _make_search(registry).run(
        generator, n_samples=5, data=data, regime="in_sample", seed=2
    )

    configs = [r.hypothesis.config() for r in results]
    unique_configs = {frozenset(c.items()) for c in configs}
    assert len(unique_configs) == 5  # all 10 leaves available -> 5 sampled must be distinct


def test_random_search_is_deterministic_given_seed() -> None:
    data = _synthetic_features()
    generator = HypothesisGenerator(
        [FeatureSpec("rsi_14", ("<", ">"), (20.0, 25.0, 30.0, 70.0, 75.0))], max_complexity=1
    )

    results_a = _make_search(DuckDBTrialRegistry()).run(
        generator, n_samples=3, data=data, regime="in_sample", seed=7
    )
    results_b = _make_search(DuckDBTrialRegistry()).run(
        generator, n_samples=3, data=data, regime="in_sample", seed=7
    )

    configs_a = [r.hypothesis.config() for r in results_a]
    configs_b = [r.hypothesis.config() for r in results_b]
    assert configs_a == configs_b


def test_random_search_rejects_n_samples_larger_than_population() -> None:
    data = _synthetic_features()
    generator = HypothesisGenerator([FeatureSpec("rsi_14", ("<",), (25.0, 30.0))], max_complexity=1)
    registry = DuckDBTrialRegistry()

    with pytest.raises(ValueError):
        _make_search(registry).run(generator, n_samples=5, data=data, regime="in_sample", seed=1)


def test_random_search_rejects_n_samples_below_one() -> None:
    data = _synthetic_features()
    generator = HypothesisGenerator([FeatureSpec("rsi_14", ("<",), (25.0,))], max_complexity=1)
    registry = DuckDBTrialRegistry()

    with pytest.raises(ValueError):
        _make_search(registry).run(generator, n_samples=0, data=data, regime="in_sample", seed=1)
