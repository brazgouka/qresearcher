from __future__ import annotations

import polars as pl
import pytest

from market_research.hypotheses import FeatureSpec, HypothesisGenerator


def test_search_space_size_matches_leaf_count_when_complexity_one() -> None:
    specs = [
        FeatureSpec("rsi_14", ("<", ">"), (25.0, 75.0)),
        FeatureSpec("zscore_20d", ("<",), (-1.0, 1.0)),
    ]
    generator = HypothesisGenerator(specs, max_complexity=1)
    # 2 ops * 2 values for rsi_14 (4) + 1 op * 2 values for zscore_20d (2) = 6 leaves
    assert generator.search_space_size() == 6
    assert len(list(generator.generate())) == 6


def test_search_space_size_includes_pairwise_and_combinations() -> None:
    specs = [
        FeatureSpec("rsi_14", ("<",), (25.0, 75.0)),
        FeatureSpec("zscore_20d", ("<",), (-1.0,)),
    ]
    generator = HypothesisGenerator(specs, max_complexity=3)
    # 3 leaves total (2 + 1); pairwise And combos = C(3,2) = 3 -> 3 + 3 = 6
    assert generator.search_space_size() == 6


def test_low_complexity_excludes_and_combinations() -> None:
    specs = [FeatureSpec("rsi_14", ("<",), (25.0, 75.0))]
    generator = HypothesisGenerator(specs, max_complexity=1)
    assert generator.search_space_size() == 2  # just the 2 leaves, no And nodes


def test_generate_yields_the_same_count_as_search_space_size() -> None:
    specs = [
        FeatureSpec("rsi_14", ("<", ">="), (20.0, 30.0, 70.0)),
        FeatureSpec("zscore_20d", ("<", ">"), (-1.0, 0.0, 1.0)),
    ]
    generator = HypothesisGenerator(specs, max_complexity=3)
    assert len(list(generator.generate())) == generator.search_space_size()


def test_every_generated_hypothesis_respects_max_complexity() -> None:
    specs = [
        FeatureSpec("rsi_14", ("<",), (25.0, 75.0)),
        FeatureSpec("zscore_20d", ("<",), (-1.0, 1.0)),
    ]
    generator = HypothesisGenerator(specs, max_complexity=3)
    for hyp in generator.generate():
        assert hyp.complexity() <= 3


def test_every_generated_hypothesis_computes_positions_cleanly() -> None:
    specs = [
        FeatureSpec("rsi_14", ("<", ">"), (25.0, 75.0)),
        FeatureSpec("zscore_20d", ("<", ">"), (-1.0, 1.0)),
    ]
    generator = HypothesisGenerator(specs, max_complexity=3)
    features = pl.DataFrame({"rsi_14": [10.0, 30.0, 50.0], "zscore_20d": [-2.0, 0.0, 1.5]})

    for hyp in generator.generate():
        positions = hyp.positions(features)
        assert positions.len() == 3
        assert set(positions.to_list()) <= {0.0, 1.0}


def test_rejects_max_complexity_below_one() -> None:
    with pytest.raises(ValueError):
        HypothesisGenerator([FeatureSpec("rsi_14", ("<",), (25.0,))], max_complexity=0)
