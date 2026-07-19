from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from market_research.backtest import CostModel, EventDrivenBacktester, SizingModel
from market_research.validation import permutation_test


def _dates(n: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]


def test_permutation_test_flags_genuine_predictive_signal() -> None:
    n = 40
    pattern = [0.02, 0.02, -0.01, -0.01]
    returns = [pattern[i % 4] for i in range(n)]
    closes = [100.0]
    for r in returns[1:]:
        closes.append(closes[-1] * (1 + r))

    # foreknowledge signal: on at t iff the next bar's return is positive
    signal_vals = [1.0 if (i + 1 < n and returns[i + 1] > 0) else 0.0 for i in range(n)]
    frame = pl.DataFrame({"symbol": ["A"] * n, "date": _dates(n), "close": closes})
    signal = pl.Series("signal", signal_vals)

    result = permutation_test(
        EventDrivenBacktester(),
        frame,
        signal,
        CostModel(),
        SizingModel(),
        n_permutations=300,
        seed=1,
    )

    assert result.observed_sharpe > 0.5
    assert result.p_value < 0.05


def test_permutation_test_does_not_flag_noise() -> None:
    n = 40
    dates = _dates(n)
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.01, n)
    closes = [100.0]
    for r in returns[1:]:
        closes.append(closes[-1] * (1 + r))
    frame = pl.DataFrame({"symbol": ["A"] * n, "date": dates, "close": closes})

    noise_signal = pl.Series("signal", rng.integers(0, 2, n).astype(float))

    result = permutation_test(
        EventDrivenBacktester(),
        frame,
        noise_signal,
        CostModel(),
        SizingModel(),
        n_permutations=300,
        seed=2,
    )

    assert result.p_value > 0.05


def test_permutation_does_not_mix_symbols() -> None:
    n = 20
    dates = _dates(n)
    a_closes = [100.0 + i for i in range(n)]
    b_closes = [50.0 - i * 0.1 for i in range(n)]
    frame = pl.concat(
        [
            pl.DataFrame({"symbol": ["A"] * n, "date": dates, "close": a_closes}),
            pl.DataFrame({"symbol": ["B"] * n, "date": dates, "close": b_closes}),
        ]
    )
    signal = pl.Series("signal", [1.0] * (2 * n))

    # must not raise and must produce a well-formed result even with divergent per-symbol paths
    result = permutation_test(
        EventDrivenBacktester(),
        frame,
        signal,
        CostModel(),
        SizingModel(),
        n_permutations=50,
        seed=4,
    )
    assert len(result.null_sharpes) == 50
