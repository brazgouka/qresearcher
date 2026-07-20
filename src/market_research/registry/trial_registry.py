from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import polars as pl

TrialStatus = Literal["pending", "evaluated"]


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    config_hash: str
    config: dict
    seed: int
    code_version: str
    status: TrialStatus
    regime: str | None
    observed_sharpe: float | None
    permutation_p_value: float | None
    logged_at: str


class TrialRegistry(Protocol):
    """The append-only ledger every search writes to (spec §1.2, §3, §4).

    Reservation and evaluation are two separate calls: "the generator cannot emit a
    hypothesis without the registry recording it" (spec §4) means the moment a
    candidate is produced — not once its backtest finishes — it must already be in the
    ledger. `count_trials()` counts every reserved row, evaluated or not; this is what
    the Bonferroni correction in `validation.Validator` divides by, so this registry is
    the thing standing between "we found an edge" and "we found the best of 10,000
    coin flips."
    """

    def reserve_trial(
        self, config_hash: str, config: dict, seed: int, code_version: str
    ) -> str: ...

    def record_result(
        self,
        trial_id: str,
        regime: str,
        observed_sharpe: float,
        permutation_p_value: float,
    ) -> None: ...

    def count_trials(self) -> int: ...

    def get_trial(self, trial_id: str) -> TrialRecord: ...

    def all_trials(self) -> pl.DataFrame: ...
