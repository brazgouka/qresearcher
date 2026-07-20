"""Slice 0 acceptance test (spec §7, item 7).

Wires every core module end-to-end — features, hypotheses, backtest, validation,
registry — and proves the one thing the whole platform exists to prove: fed a
hypothesis that is genuinely noise, the validation machinery rejects it; fed a
hypothesis with a real, constructed effect, it does not. If this test doesn't pass,
nothing built on top of this platform can be trusted, no matter how sophisticated.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from market_research.backtest import CostModel, EventDrivenBacktester, SizingModel
from market_research.features import RSI, FeatureEngine
from market_research.hypotheses import Threshold
from market_research.registry import DuckDBTrialRegistry
from market_research.storage import content_hash
from market_research.validation import BonferroniPermutationValidator, permutation_test

N_BARS = 300


def _dates(n: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]


def _build_real_effect_series(n: int, seed: int) -> np.ndarray:
    """A price path with a causally-injected, mechanical pattern: whenever the
    (simple, rolling-mean-based) RSI computed from bars so far drops below 30, the
    *next* bar is forced to have a strong positive drift. This is deliberately not
    subtle — a "trivially real effect" per spec §7 — so the acceptance test has a
    clean positive control, not a coin-flip case near the edge of detectability.
    """
    rng = np.random.default_rng(seed)
    prices = list(100.0 + rng.normal(0, 0.3, 20).cumsum())
    for _ in range(20, n):
        recent = np.array(prices[-15:])
        diffs = np.diff(recent)
        gain = diffs[diffs > 0].sum()
        loss = -diffs[diffs < 0].sum()
        avg_gain, avg_loss = gain / 14.0, loss / 14.0
        rsi = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        drift = 0.02 if rsi < 30 else -0.001
        prices.append(prices[-1] * (1 + drift + rng.normal(0, 0.008)))
    return np.array(prices)


def _build_noise_series(n: int, seed: int) -> np.ndarray:
    """Pure zero-drift random walk: by construction, no feature computed from this
    series carries any information about its own future — this is the negative
    control the validation layer must reject."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, n)
    return 100.0 * np.cumprod(1 + returns)


def _to_ohlcv(symbol: str, closes: np.ndarray) -> pl.DataFrame:
    n = len(closes)
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": _dates(n),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * n,
        }
    )


@pytest.fixture(scope="module")
def real_features() -> pl.DataFrame:
    closes = _build_real_effect_series(N_BARS, seed=11)
    return FeatureEngine([RSI(period=14)]).compute(_to_ohlcv("REAL", closes))


@pytest.fixture(scope="module")
def noise_features() -> pl.DataFrame:
    closes = _build_noise_series(N_BARS, seed=12)
    return FeatureEngine([RSI(period=14)]).compute(_to_ohlcv("NOISE", closes))


def test_slice_zero_rejects_noise_and_accepts_real_effect(
    real_features: pl.DataFrame, noise_features: pl.DataFrame
) -> None:
    hypothesis = Threshold("rsi_14", "<", 30.0)
    backtester = EventDrivenBacktester()
    costs = CostModel()
    sizing = SizingModel()
    registry = DuckDBTrialRegistry()

    # Simulate a realistic search: hundreds of prior trials already sit in the
    # registry, so the Bonferroni correction this test's two trials face reflects a
    # genuine multiple-testing scale (spec §1.2), not a favorably tiny count of 1.
    for i in range(998):
        registry.log_trial(f"filler_{i}", {"type": "filler"}, i, "v0", "in_sample", 0.0, 1.0)

    real_permutation = permutation_test(
        backtester,
        real_features,
        hypothesis.positions(real_features),
        costs,
        sizing,
        n_permutations=300,
        seed=5,
    )
    noise_permutation = permutation_test(
        backtester,
        noise_features,
        hypothesis.positions(noise_features),
        costs,
        sizing,
        n_permutations=300,
        seed=6,
    )

    registry.log_trial(
        content_hash("hypothesis_trial", {"config": hypothesis.config(), "symbol": "REAL"}),
        hypothesis.config(),
        seed=5,
        code_version="v0",
        regime="in_sample",
        observed_sharpe=real_permutation.observed_sharpe,
        permutation_p_value=real_permutation.p_value,
    )
    registry.log_trial(
        content_hash("hypothesis_trial", {"config": hypothesis.config(), "symbol": "NOISE"}),
        hypothesis.config(),
        seed=6,
        code_version="v0",
        regime="in_sample",
        observed_sharpe=noise_permutation.observed_sharpe,
        permutation_p_value=noise_permutation.p_value,
    )

    global_trial_count = registry.count_trials()
    assert global_trial_count == 1000

    validator = BonferroniPermutationValidator(alpha=0.05)
    real_verdict = validator.validate(real_permutation, global_trial_count, regime="in_sample")
    noise_verdict = validator.validate(noise_permutation, global_trial_count, regime="in_sample")

    # The core claim: noise is rejected, the constructed real effect is not.
    assert noise_verdict.survives is False
    assert real_verdict.survives is True

    # And the reason this matters: an *uncorrected* alpha=0.05 would have let the
    # noise hypothesis through (p=0.013 < 0.05) — the correction against the global
    # trial count is what catches it. Without invariant #2 (every trial logged) this
    # correction would silently divide by too small a number and miss it too.
    assert noise_permutation.p_value < 0.05
    assert noise_permutation.p_value > real_verdict.corrected_alpha

    # Invariant #2: both trials are in the ledger regardless of verdict.
    all_trial_ids = set(registry.all_trials()["trial_id"].to_list())
    assert len(all_trial_ids) == 1000
