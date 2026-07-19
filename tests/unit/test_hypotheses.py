from __future__ import annotations

import polars as pl
import pytest

from market_research.hypotheses import And, Not, Or, Threshold


@pytest.fixture
def features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "rsi_14": [10.0, 30.0, 50.0, None, 90.0],
            "zscore_20d": [-2.0, -0.5, 0.0, 1.0, None],
        }
    )


def test_threshold_less_than(features: pl.DataFrame) -> None:
    positions = Threshold("rsi_14", "<", 25).positions(features)
    assert positions.to_list() == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_threshold_null_feature_resolves_to_flat(features: pl.DataFrame) -> None:
    positions = Threshold("rsi_14", "<", 100).positions(features)
    # row 3 has rsi_14=None -> must be 0.0 (flat), not fabricated as True
    assert positions.to_list()[3] == 0.0


def test_threshold_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError):
        Threshold("rsi_14", "~=", 25)


def test_and_requires_both_conditions(features: pl.DataFrame) -> None:
    hyp = And(Threshold("rsi_14", "<", 55), Threshold("zscore_20d", "<", -0.1))
    positions = hyp.positions(features)
    # row0: rsi=10<55 T, z=-2<-0.1 T -> 1.0
    # row1: rsi=30<55 T, z=-0.5<-0.1 T -> 1.0
    # row2: rsi=50<55 T, z=0.0<-0.1 F -> 0.0
    assert positions.to_list()[:3] == [1.0, 1.0, 0.0]


def test_or_requires_either_condition(features: pl.DataFrame) -> None:
    hyp = Or(Threshold("rsi_14", "<", 15), Threshold("zscore_20d", "<", -0.1))
    positions = hyp.positions(features)
    # row0: rsi<15 T -> 1.0 regardless of z
    # row2: rsi=50 F, z=0.0 F -> 0.0
    assert positions.to_list()[0] == 1.0
    assert positions.to_list()[2] == 0.0


def test_not_inverts(features: pl.DataFrame) -> None:
    hyp = Not(Threshold("rsi_14", "<", 25))
    positions = hyp.positions(features)
    assert positions.to_list() == [0.0, 1.0, 1.0, 1.0, 1.0]


def test_complexity_counts_nodes() -> None:
    leaf = Threshold("rsi_14", "<", 25)
    assert leaf.complexity() == 1

    combo = And(Threshold("rsi_14", "<", 25), Threshold("zscore_20d", "<", -1))
    assert combo.complexity() == 3

    nested = Not(And(Threshold("rsi_14", "<", 25), Threshold("zscore_20d", "<", -1)))
    assert nested.complexity() == 4


def test_config_is_deterministic_and_reflects_structure() -> None:
    hyp = And(Threshold("rsi_14", "<", 25), Threshold("zscore_20d", "<", -1.0))
    config = hyp.config()
    assert config == {
        "type": "And",
        "left": {"type": "Threshold", "feature": "rsi_14", "op": "<", "value": 25},
        "right": {"type": "Threshold", "feature": "zscore_20d", "op": "<", "value": -1.0},
    }
    # identical structure -> identical config, independent of object identity
    hyp2 = And(Threshold("rsi_14", "<", 25), Threshold("zscore_20d", "<", -1.0))
    assert hyp2.config() == config
