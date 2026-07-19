from __future__ import annotations

from market_research.validation import (
    BonferroniPermutationValidator,
    PermutationTestResult,
    ValidationVerdict,
)


def test_survives_when_p_value_beats_corrected_alpha() -> None:
    result = PermutationTestResult(observed_sharpe=1.2, null_sharpes=[0.1] * 100, p_value=0.0)
    verdict = BonferroniPermutationValidator(alpha=0.05).validate(
        result, global_trial_count=100, regime="in_sample"
    )
    assert verdict.survives is True
    assert verdict.corrected_alpha == 0.05 / 100


def test_rejected_when_p_value_only_beats_uncorrected_alpha() -> None:
    # p=0.03 would pass an uncorrected alpha=0.05, but not once divided by 10,000 trials
    result = PermutationTestResult(observed_sharpe=0.4, null_sharpes=[0.1] * 100, p_value=0.03)
    verdict = BonferroniPermutationValidator(alpha=0.05).validate(
        result, global_trial_count=10_000, regime="in_sample"
    )
    assert verdict.survives is False


def test_more_trials_makes_survival_strictly_harder() -> None:
    result = PermutationTestResult(observed_sharpe=0.5, null_sharpes=[0.1] * 100, p_value=0.001)
    validator = BonferroniPermutationValidator(alpha=0.05)

    lenient = validator.validate(result, global_trial_count=10, regime="in_sample")
    strict = validator.validate(result, global_trial_count=1_000_000, regime="in_sample")

    assert lenient.survives is True
    assert strict.survives is False


def test_verdict_carries_regime_and_trial_count_through() -> None:
    result = PermutationTestResult(observed_sharpe=0.5, null_sharpes=[], p_value=0.0)
    verdict = BonferroniPermutationValidator().validate(
        result, global_trial_count=42, regime="holdout"
    )
    assert isinstance(verdict, ValidationVerdict)
    assert verdict.regime == "holdout"
    assert verdict.global_trial_count == 42
