from __future__ import annotations

import polars as pl

from market_research.features.feature import Feature


class FeatureEngine:
    """Computes a set of Features over an OHLCV frame.

    Sorting by ["symbol", "date"] happens once, here — individual Feature
    implementations only need to express their (backward-looking, per-symbol) transform
    and can rely on row order already being correct.
    """

    def __init__(self, features: list[Feature]) -> None:
        self._features = features

    @property
    def max_lookback(self) -> int:
        return max((f.lookback for f in self._features), default=0)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        lazy = frame.lazy().sort(["symbol", "date"])
        exprs = [f.transform(lazy) for f in self._features]
        return lazy.with_columns(exprs).collect()
