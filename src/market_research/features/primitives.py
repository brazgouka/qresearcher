from __future__ import annotations

import polars as pl


class ReturnOverN:
    """N-bar simple return: close / close.shift(n) - 1."""

    def __init__(self, n: int = 5) -> None:
        self.n = n
        self.name = f"return_{n}d"
        self.lookback = n

    def transform(self, frame: pl.LazyFrame) -> pl.Expr:
        close = pl.col("close")
        return ((close / close.shift(self.n) - 1.0).over("symbol")).alias(self.name)


class RollingZScore:
    """Z-score of close over a trailing window: (close - rolling_mean) / rolling_std."""

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.name = f"zscore_{window}d"
        self.lookback = window

    def transform(self, frame: pl.LazyFrame) -> pl.Expr:
        close = pl.col("close")
        mean = close.rolling_mean(window_size=self.window).over("symbol")
        std = close.rolling_std(window_size=self.window).over("symbol")
        return ((close - mean) / std).alias(self.name)


class RSI:
    """Relative Strength Index: rolling-mean gain/loss ratio over `period` bars."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.name = f"rsi_{period}"
        self.lookback = period + 1  # +1: diff() consumes one prior bar

    def transform(self, frame: pl.LazyFrame) -> pl.Expr:
        delta = pl.col("close").diff().over("symbol")
        gain = pl.when(delta > 0).then(delta).otherwise(0.0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
        avg_gain = gain.rolling_mean(window_size=self.period).over("symbol")
        avg_loss = loss.rolling_mean(window_size=self.period).over("symbol")
        rs = avg_gain / avg_loss
        return (100.0 - 100.0 / (1.0 + rs)).alias(self.name)


class VolumeRatio:
    """Current volume vs. the average of the prior `window` bars (excludes the current bar)."""

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.name = f"volume_ratio_{window}d"
        self.lookback = window + 1

    def transform(self, frame: pl.LazyFrame) -> pl.Expr:
        volume = pl.col("volume")
        avg_prior_volume = volume.shift(1).rolling_mean(window_size=self.window).over("symbol")
        return (volume / avg_prior_volume).alias(self.name)


class RealizedVolatility:
    """Rolling standard deviation of simple returns over `window` bars."""

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.name = f"realized_vol_{window}d"
        self.lookback = window + 1

    def transform(self, frame: pl.LazyFrame) -> pl.Expr:
        returns = pl.col("close").pct_change().over("symbol")
        return returns.rolling_std(window_size=self.window).over("symbol").alias(self.name)
