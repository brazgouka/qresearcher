from __future__ import annotations

import pytest

from market_research.registry import DuckDBTrialRegistry


def test_log_trial_returns_a_unique_id() -> None:
    registry = DuckDBTrialRegistry()
    id_a = registry.log_trial("hash_a", {"type": "Threshold"}, 1, "v1", "in_sample", 0.5, 0.01)
    id_b = registry.log_trial("hash_b", {"type": "Threshold"}, 2, "v1", "in_sample", 0.3, 0.4)
    assert id_a != id_b


def test_count_trials_reflects_number_logged() -> None:
    registry = DuckDBTrialRegistry()
    assert registry.count_trials() == 0
    registry.log_trial("hash_a", {"type": "Threshold"}, 1, "v1", "in_sample", 0.5, 0.01)
    assert registry.count_trials() == 1
    registry.log_trial("hash_b", {"type": "Threshold"}, 2, "v1", "in_sample", 0.3, 0.4)
    assert registry.count_trials() == 2


def test_get_trial_round_trips_logged_data() -> None:
    registry = DuckDBTrialRegistry()
    config = {"type": "And", "left": {"type": "Threshold", "feature": "rsi_14"}}
    trial_id = registry.log_trial("hash_a", config, 7, "v1.2.3", "holdout", 1.23, 0.004)

    record = registry.get_trial(trial_id)
    assert record.trial_id == trial_id
    assert record.config_hash == "hash_a"
    assert record.config == config
    assert record.seed == 7
    assert record.code_version == "v1.2.3"
    assert record.regime == "holdout"
    assert record.observed_sharpe == pytest.approx(1.23)
    assert record.permutation_p_value == pytest.approx(0.004)


def test_get_trial_missing_raises_keyerror() -> None:
    registry = DuckDBTrialRegistry()
    with pytest.raises(KeyError):
        registry.get_trial("does-not-exist")


def test_all_trials_includes_survivors_and_rejects_alike() -> None:
    # invariant #2: discarding a trial without logging it is a leakage bug — the
    # registry must never silently drop a "failed" trial from the ledger.
    registry = DuckDBTrialRegistry()
    survivor_id = registry.log_trial(
        "hash_ok", {"type": "Threshold"}, 1, "v1", "in_sample", 1.5, 0.001
    )
    rejected_id = registry.log_trial(
        "hash_bad", {"type": "Threshold"}, 2, "v1", "in_sample", 0.1, 0.9
    )

    all_ids = registry.all_trials()["trial_id"].to_list()
    assert survivor_id in all_ids
    assert rejected_id in all_ids
    assert registry.count_trials() == 2


def test_registry_persists_across_connections(tmp_path) -> None:
    db_path = str(tmp_path / "trials.duckdb")
    registry = DuckDBTrialRegistry(db_path)
    registry.log_trial("hash_a", {"type": "Threshold"}, 1, "v1", "in_sample", 0.5, 0.01)
    del registry

    reopened = DuckDBTrialRegistry(db_path)
    assert reopened.count_trials() == 1
