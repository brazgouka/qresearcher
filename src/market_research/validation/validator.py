from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from market_research.backtest import Regime
from market_research.validation.permutation import PermutationTestResult


@dataclass(frozen=True)
class ValidationVerdict:
    regime: Regime
    global_trial_count: int
    observed_sharpe: float
    permutation_p_value: float
    corrected_alpha: float
    survives: bool


class Validator(Protocol):
    def validate(
        self,
        permutation: PermutationTestResult,
        global_trial_count: int,
        regime: Regime,
    ) -> ValidationVerdict: ...


class BonferroniPermutationValidator:
    """Rejects a hypothesis unless its permutation-test p-value clears a
    Bonferroni-corrected significance bar: alpha / global_trial_count.

    This is the mechanism behind spec §1.2: multiple-testing correction computed
    against the *total* number of trials ever evaluated (read from the trial
    registry), not just the surviving candidate's own history. Running 10,000
    backtests and reporting the best one guarantees some will look significant at an
    uncorrected alpha=0.05 — dividing by the trial count is what makes that
    impossible to hide.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def validate(
        self,
        permutation: PermutationTestResult,
        global_trial_count: int,
        regime: Regime,
    ) -> ValidationVerdict:
        corrected_alpha = self.alpha / max(global_trial_count, 1)
        survives = permutation.p_value < corrected_alpha
        return ValidationVerdict(
            regime=regime,
            global_trial_count=global_trial_count,
            observed_sharpe=permutation.observed_sharpe,
            permutation_p_value=permutation.p_value,
            corrected_alpha=corrected_alpha,
            survives=survives,
        )
