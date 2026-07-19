from __future__ import annotations

from typing import Protocol

import polars as pl


class Feature(Protocol):
    """A pure, backward-looking transform from an OHLCV frame to a named column.

    Look-ahead safety is structural, not conventional: `transform` must return a
    `pl.Expr` built only from Polars' backward-looking primitives (`shift`, `diff`,
    `rolling_*`, `pct_change`, cumulative ops) over a frame already sorted by
    `["symbol", "date"]`. There is no parameter here for a centered window or for
    seeing rows beyond the current one — the interface itself forbids it.

    `lookback` is the number of prior bars the feature needs before its first valid
    (non-null) value; callers use it to know how much history to fetch and how many
    leading rows to discard.
    """

    name: str
    lookback: int

    def transform(self, frame: pl.LazyFrame) -> pl.Expr: ...
