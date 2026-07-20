from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from market_research.backtest import CostModel, EventDrivenBacktester, SizingModel
from market_research.features import RSI, FeatureEngine
from market_research.hypotheses import FeatureSpec, HypothesisGenerator
from market_research.registry import DuckDBTrialRegistry
from market_research.search import GridSearch, SearchResult
from market_research.validation import BonferroniPermutationValidator, ValidationVerdict


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


def test_grid_search_reserves_and_records_every_candidate() -> None:
    data = _synthetic_features()
    generator = HypothesisGenerator(
        [FeatureSpec("rsi_14", ("<",), (25.0, 30.0, 35.0))], max_complexity=1
    )
    registry = DuckDBTrialRegistry()
    search = GridSearch(
        EventDrivenBacktester(),
        BonferroniPermutationValidator(alpha=0.05),
        registry,
        CostModel(),
        SizingModel(),
        code_version="test",
        n_permutations=50,
    )

    results = search.run(generator, data, regime="in_sample", seed=1)

    assert len(results) == generator.search_space_size() == 3
    assert registry.count_trials() == 3

    all_trial_ids = set(registry.all_trials()["trial_id"].to_list())
    result_trial_ids = {r.trial_id for r in results}
    assert all_trial_ids == result_trial_ids


def test_grid_search_results_have_well_formed_verdicts() -> None:
    data = _synthetic_features()
    generator = HypothesisGenerator([FeatureSpec("rsi_14", ("<",), (25.0, 30.0))], max_complexity=1)
    registry = DuckDBTrialRegistry()
    search = GridSearch(
        EventDrivenBacktester(),
        BonferroniPermutationValidator(alpha=0.05),
        registry,
        CostModel(),
        SizingModel(),
        code_version="test",
        n_permutations=50,
    )

    results = search.run(generator, data, regime="in_sample", seed=2)

    assert len(results) == 2
    for result in results:
        assert isinstance(result, SearchResult)
        assert isinstance(result.verdict, ValidationVerdict)
        assert result.verdict.regime == "in_sample"
        assert isinstance(result.verdict.survives, bool)


def test_grid_search_all_evaluated_trials_have_recorded_results() -> None:
    data = _synthetic_features()
    generator = HypothesisGenerator([FeatureSpec("rsi_14", ("<",), (25.0, 30.0))], max_complexity=1)
    registry = DuckDBTrialRegistry()
    search = GridSearch(
        EventDrivenBacktester(),
        BonferroniPermutationValidator(alpha=0.05),
        registry,
        CostModel(),
        SizingModel(),
        code_version="test",
        n_permutations=50,
    )

    search.run(generator, data, regime="in_sample", seed=3)

    all_trials = registry.all_trials()
    assert (all_trials["status"] == "evaluated").all()
    assert all_trials["observed_sharpe"].null_count() == 0


def test_global_trial_count_grows_across_successive_searches() -> None:
    data = _synthetic_features()
    registry = DuckDBTrialRegistry()
    search = GridSearch(
        EventDrivenBacktester(),
        BonferroniPermutationValidator(alpha=0.05),
        registry,
        CostModel(),
        SizingModel(),
        code_version="test",
        n_permutations=50,
    )

    first_generator = HypothesisGenerator(
        [FeatureSpec("rsi_14", ("<",), (25.0,))], max_complexity=1
    )
    search.run(first_generator, data, regime="in_sample", seed=4)
    assert registry.count_trials() == 1

    second_generator = HypothesisGenerator(
        [FeatureSpec("rsi_14", ("<",), (30.0,))], max_complexity=1
    )
    search.run(second_generator, data, regime="in_sample", seed=5)
    assert registry.count_trials() == 2
