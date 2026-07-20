from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import polars as pl

from market_research.registry.trial_registry import TrialRecord
from market_research.storage import DuckDBStore

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id VARCHAR PRIMARY KEY,
    config_hash VARCHAR,
    config VARCHAR,
    seed INTEGER,
    code_version VARCHAR,
    status VARCHAR,
    regime VARCHAR,
    observed_sharpe DOUBLE,
    permutation_p_value DOUBLE,
    logged_at VARCHAR
)
"""


class DuckDBTrialRegistry:
    """Append-only trial ledger backed by an embedded DuckDB connection.

    No server process to run (spec §9.5's "small correct system" preference), and it
    reuses the same `DuckDBStore` the storage module already provides rather than
    opening a second, parallel notion of "how we talk to DuckDB."
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._store = DuckDBStore(db_path)
        self._store.execute(_CREATE_TABLE)

    def reserve_trial(self, config_hash: str, config: dict, seed: int, code_version: str) -> str:
        trial_id = str(uuid.uuid4())
        logged_at = datetime.now(UTC).isoformat()
        self._store.execute(
            "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                trial_id,
                config_hash,
                json.dumps(config, sort_keys=True),
                seed,
                code_version,
                "pending",
                None,
                None,
                None,
                logged_at,
            ],
        )
        return trial_id

    def record_result(
        self,
        trial_id: str,
        regime: str,
        observed_sharpe: float,
        permutation_p_value: float,
    ) -> None:
        if self._store.query("SELECT 1 FROM trials WHERE trial_id = ?", [trial_id]).height == 0:
            raise KeyError(f"No reserved trial with id {trial_id!r}")
        self._store.execute(
            """
            UPDATE trials
            SET status = 'evaluated', regime = ?, observed_sharpe = ?, permutation_p_value = ?
            WHERE trial_id = ?
            """,
            [regime, observed_sharpe, permutation_p_value, trial_id],
        )

    def count_trials(self) -> int:
        return int(self._store.query("SELECT COUNT(*) AS n FROM trials")["n"][0])

    def get_trial(self, trial_id: str) -> TrialRecord:
        frame = self._store.query("SELECT * FROM trials WHERE trial_id = ?", [trial_id])
        if frame.height == 0:
            raise KeyError(f"No trial with id {trial_id!r}")
        row = frame.row(0, named=True)
        return TrialRecord(
            trial_id=row["trial_id"],
            config_hash=row["config_hash"],
            config=json.loads(row["config"]),
            seed=row["seed"],
            code_version=row["code_version"],
            status=row["status"],
            regime=row["regime"],
            observed_sharpe=row["observed_sharpe"],
            permutation_p_value=row["permutation_p_value"],
            logged_at=row["logged_at"],
        )

    def all_trials(self) -> pl.DataFrame:
        return self._store.query("SELECT * FROM trials ORDER BY logged_at")
