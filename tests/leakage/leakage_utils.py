from __future__ import annotations

import math
from typing import Any

import polars as pl


def assert_no_lookahead(feature: Any, frame: pl.DataFrame, symbol: str) -> None:
    """A feature's value at row t must be identical whether computed on the full
    series or on a series truncated to rows [0, t] — i.e. it cannot see beyond t.

    This is the leakage-regression test the spec (§8) requires per feature: a feature
    that only passes because of an accidental Polars laziness quirk gets caught here.
    """
    symbol_frame = frame.filter(pl.col("symbol") == symbol).sort("date")
    full = (
        symbol_frame.lazy()
        .with_columns(feature.transform(symbol_frame.lazy()))
        .collect()
    )
    n = symbol_frame.height

    checked = 0
    for t in range(feature.lookback, n):
        truncated_input = symbol_frame.head(t + 1)
        truncated = (
            truncated_input.lazy()
            .with_columns(feature.transform(truncated_input.lazy()))
            .collect()
        )
        full_value = full[feature.name][t]
        truncated_value = truncated[feature.name][t]
        assert _values_equal(full_value, truncated_value), (
            f"{feature.name} at t={t} (symbol={symbol}): full-series value "
            f"{full_value!r} != truncated-series value {truncated_value!r} "
            "(look-ahead leak)"
        )
        checked += 1

    assert checked > 0, "no rows were checked — lookback too large for the test fixture"


def _values_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b
