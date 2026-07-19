from __future__ import annotations

from typing import Protocol

import polars as pl

from market_research.backtest.cost_model import CostModel, SizingModel
from market_research.backtest.result import BacktestResult, Regime
from market_research.hypotheses import Hypothesis


class Backtester(Protocol):
    """Consumes a hypothesis + a features-augmented OHLCV frame -> trades + equity curve.

    `data` must already contain both price columns (symbol, date, close) and any
    feature columns `hypothesis` references — i.e. the output of
    `FeatureEngine.compute()`, not raw OHLCV. `regime` is required, not defaulted:
    every result must explicitly declare which regime it belongs to (spec §1.6).
    """

    def run(
        self,
        hypothesis: Hypothesis,
        data: pl.DataFrame,
        costs: CostModel,
        sizing: SizingModel,
        *,
        regime: Regime,
    ) -> BacktestResult: ...


class EventDrivenBacktester:
    """Vectorized event-driven backtest.

    The signal decided at bar t (using only data available at t, since Hypothesis
    reads already-computed, backward-looking features) is applied to the return
    realized from t to t+1 — "signal today, trade tomorrow's return" — so the backtest
    engine itself introduces no look-ahead independent of whether the features do
    (spec §1.1). Cost is charged once, at the bar where held exposure changes.
    """

    def run(
        self,
        hypothesis: Hypothesis,
        data: pl.DataFrame,
        costs: CostModel,
        sizing: SizingModel,
        *,
        regime: Regime,
    ) -> BacktestResult:
        frame = data.sort(["symbol", "date"])
        frame = frame.with_columns(hypothesis.positions(frame).alias("signal"))

        frame = frame.with_columns(
            pl.col("close").pct_change().over("symbol").alias("asset_return")
        )
        frame = frame.with_columns(
            pl.col("signal").shift(1).over("symbol").fill_null(0.0).alias("position")
        )
        frame = frame.with_columns(
            (pl.col("position") * sizing.target_gross_exposure).alias("exposure")
        )
        frame = frame.with_columns(
            (pl.col("exposure") - pl.col("exposure").shift(1).over("symbol").fill_null(0.0))
            .abs()
            .alias("turnover")
        )
        frame = frame.with_columns((pl.col("turnover") * costs.cost_rate).alias("cost"))
        frame = frame.with_columns(
            (
                pl.col("exposure") * pl.col("asset_return").fill_null(0.0) - pl.col("cost")
            ).alias("strategy_return")
        )
        frame = frame.with_columns(
            (1.0 + pl.col("strategy_return")).cum_prod().over("symbol").alias("equity")
        )

        columns = [
            "symbol",
            "date",
            "position",
            "asset_return",
            "cost",
            "strategy_return",
            "equity",
        ]
        return BacktestResult(frame=frame.select(columns), regime=regime)
