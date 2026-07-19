from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import polars as pl


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    config_hash: str
    config: dict
    seed: int
    code_version: str
    regime: str
    observed_sharpe: float
    permutation_p_value: float
    logged_at: str


class TrialRegistry(Protocol):
    """The append-only ledger every search writes to (spec §1.2, §3).

    Every hypothesis ever evaluated is logged here — survivors and rejects alike.
    `count_trials()` is what the Bonferroni correction in `validation.Validator`
    divides by, so this registry is the thing standing between "we found an edge" and
    "we found the best of 10,000 coin flips."
    """

    def log_trial(
        self,
        config_hash: str,
        config: dict,
        seed: int,
        code_version: str,
        regime: str,
        observed_sharpe: float,
        permutation_p_value: float,
    ) -> str: ...

    def count_trials(self) -> int: ...

    def get_trial(self, trial_id: str) -> TrialRecord: ...

    def all_trials(self) -> pl.DataFrame: ...
