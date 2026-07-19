from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


def _synthetic_ohlcv(symbol: str, n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    returns = rng.normal(loc=0.0005, scale=0.01, size=n)
    close = 100.0 * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = rng.integers(1000, 5000, n)
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def multi_symbol_ohlcv() -> pl.DataFrame:
    return pl.concat([_synthetic_ohlcv("AAA", 60, seed=1), _synthetic_ohlcv("BBB", 60, seed=2)])
