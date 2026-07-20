from __future__ import annotations

import pytest

from market_research.registry import DuckDBTrialRegistry


def test_reserve_trial_returns_a_unique_id() -> None:
    registry = DuckDBTrialRegistry()
    id_a = registry.reserve_trial("hash_a", {"type": "Threshold"}, 1, "v1")
    id_b = registry.reserve_trial("hash_b", {"type": "Threshold"}, 2, "v1")
    assert id_a != id_b


def test_count_trials_reflects_reservations_even_before_evaluation() -> None:
    registry = DuckDBTrialRegistry()
    assert registry.count_trials() == 0
    registry.reserve_trial("hash_a", {"type": "Threshold"}, 1, "v1")
    assert registry.count_trials() == 1
    registry.reserve_trial("hash_b", {"type": "Threshold"}, 2, "v1")
    assert registry.count_trials() == 2


def test_reserved_trial_starts_pending_then_becomes_evaluated() -> None:
    registry = DuckDBTrialRegistry()
    trial_id = registry.reserve_trial("hash_a", {"type": "Threshold"}, 1, "v1")

    pending = registry.get_trial(trial_id)
    assert pending.status == "pending"
    assert pending.regime is None
    assert pending.observed_sharpe is None
    assert pending.permutation_p_value is None

    registry.record_result(trial_id, "holdout", 1.23, 0.004)

    evaluated = registry.get_trial(trial_id)
    assert evaluated.status == "evaluated"
    assert evaluated.regime == "holdout"
    assert evaluated.observed_sharpe == pytest.approx(1.23)
    assert evaluated.permutation_p_value == pytest.approx(0.004)


def test_record_result_round_trips_config() -> None:
    registry = DuckDBTrialRegistry()
    config = {"type": "And", "left": {"type": "Threshold", "feature": "rsi_14"}}
    trial_id = registry.reserve_trial("hash_a", config, 7, "v1.2.3")
    registry.record_result(trial_id, "in_sample", 0.5, 0.02)

    record = registry.get_trial(trial_id)
    assert record.config_hash == "hash_a"
    assert record.config == config
    assert record.seed == 7
    assert record.code_version == "v1.2.3"


def test_record_result_on_missing_trial_raises_keyerror() -> None:
    registry = DuckDBTrialRegistry()
    with pytest.raises(KeyError):
        registry.record_result("does-not-exist", "in_sample", 0.5, 0.02)


def test_get_trial_missing_raises_keyerror() -> None:
    registry = DuckDBTrialRegistry()
    with pytest.raises(KeyError):
        registry.get_trial("does-not-exist")


def test_all_trials_includes_survivors_and_rejects_alike() -> None:
    # invariant #2: discarding a trial without logging it is a leakage bug — the
    # registry must never silently drop a "failed" trial from the ledger.
    registry = DuckDBTrialRegistry()
    survivor_id = registry.reserve_trial("hash_ok", {"type": "Threshold"}, 1, "v1")
    registry.record_result(survivor_id, "in_sample", 1.5, 0.001)
    rejected_id = registry.reserve_trial("hash_bad", {"type": "Threshold"}, 2, "v1")
    registry.record_result(rejected_id, "in_sample", 0.1, 0.9)

    all_ids = registry.all_trials()["trial_id"].to_list()
    assert survivor_id in all_ids
    assert rejected_id in all_ids
    assert registry.count_trials() == 2


def test_registry_persists_across_connections(tmp_path) -> None:
    db_path = str(tmp_path / "trials.duckdb")
    registry = DuckDBTrialRegistry(db_path)
    registry.reserve_trial("hash_a", {"type": "Threshold"}, 1, "v1")
    del registry

    reopened = DuckDBTrialRegistry(db_path)
    assert reopened.count_trials() == 1
