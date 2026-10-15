from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from market_research.backtest import Backtester, CostModel, SizingModel


@dataclass(frozen=True)
class PermutationTestResult:
    observed_sharpe: float
    null_sharpes: list[float]
    p_value: float


class _FixedSignal:
    """A throwaway Hypothesis whose positions() returns a pre-computed series.

    Lets the permutation test reuse the real Backtester (return + cost math) for every
    null draw instead of reimplementing that arithmetic separately, where it could
    silently drift out of sync with the real thing.
    """

    def __init__(self, signal: pl.Series) -> None:
        self._signal = signal

    def positions(self, features: pl.DataFrame) -> pl.Series:
        return self._signal

    def complexity(self) -> int:
        return 0

    def config(self) -> dict:
        return {"type": "_FixedSignal"}


def permutation_test(
    backtester: Backtester,
    data: pl.DataFrame,
    signal: pl.Series,
    costs: CostModel,
    sizing: SizingModel,
    n_permutations: int = 500,
    seed: int = 42,
) -> PermutationTestResult:
    """Null-model benchmark: "random-entry-same-exposure" (spec §6).

    Compares the observed strategy Sharpe against the distribution obtained by
    randomly permuting *when* the signal fires — independently within each symbol, so
    permutation never mixes bars across instruments — while holding the real price
    path and the signal's own time-in-market distribution fixed. A hypothesis with
    genuine predictive power should be a clear outlier against this null; a noise
    hypothesis should not.
    """
    observed_result = backtester.run(_FixedSignal(signal), data, costs, sizing, regime="in_sample")
    observed_sharpe = _sharpe(observed_result.frame["strategy_return"])

    symbols = data["symbol"]
    rng = np.random.default_rng(seed)
    null_sharpes = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = _permute_within_symbol(signal, symbols, rng)
        result = backtester.run(_FixedSignal(shuffled), data, costs, sizing, regime="in_sample")
        null_sharpes[i] = _sharpe(result.frame["strategy_return"])

    p_value = float(np.mean(null_sharpes >= observed_sharpe))
    return PermutationTestResult(
        observed_sharpe=observed_sharpe,
        null_sharpes=null_sharpes.tolist(),
        p_value=p_value,
    )


def _permute_within_symbol(
    signal: pl.Series, symbols: pl.Series, rng: np.random.Generator
) -> pl.Series:
    values = signal.to_numpy().copy()
    symbol_values = symbols.to_numpy()
    for sym in np.unique(symbol_values):
        idx = np.where(symbol_values == sym)[0]
        values[idx] = values[rng.permutation(idx)]
    return pl.Series(signal.name, values)


def _sharpe(returns: pl.Series) -> float:
    values = returns.drop_nulls().to_numpy()
    if len(values) < 2 or values.std(ddof=1) == 0:
        return 0.0
    return float(values.mean() / values.std(ddof=1))
