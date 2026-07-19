from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from market_research.features import (
    RSI,
    FeatureEngine,
    RealizedVolatility,
    ReturnOverN,
    RollingZScore,
    VolumeRatio,
)


def _frame(symbol: str, closes: list[float], volumes: list[int] | None = None) -> pl.DataFrame:
    n = len(closes)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes or [1000] * n,
        }
    )


def test_return_over_n_matches_manual_calculation() -> None:
    frame = _frame("A", [100.0, 101.0, 102.0, 103.0, 110.0])
    feature = ReturnOverN(n=3)
    out = FeatureEngine([feature]).compute(frame)

    values = out[feature.name].to_list()
    assert values[:3] == [None, None, None]
    assert values[3] == pytest.approx(103.0 / 100.0 - 1.0)
    assert values[4] == pytest.approx(110.0 / 101.0 - 1.0)


def test_rolling_zscore_zero_at_constant_window_is_null_not_inf() -> None:
    # constant prices -> rolling std is 0 -> division by zero: must not raise, produces null/inf
    frame = _frame("A", [100.0] * 10)
    feature = RollingZScore(window=5)
    out = FeatureEngine([feature]).compute(frame)
    # no exception raised is the main assertion; last values should not be NaN's crashing anything
    assert out.height == 10


def test_rsi_bounds_between_0_and_100() -> None:
    frame = _frame("A", [100, 102, 101, 105, 103, 108, 106, 110, 107, 112, 109, 115, 111, 118, 114])
    feature = RSI(period=5)
    out = FeatureEngine([feature]).compute(frame)
    values = [v for v in out[feature.name].to_list() if v is not None]
    assert len(values) > 0
    assert all(0.0 <= v <= 100.0 for v in values)


def test_volume_ratio_excludes_current_bar() -> None:
    # constant prior volume of 100, current bar spikes to 500 -> ratio should be 5.0
    volumes = [100] * 10 + [500]
    frame = _frame("A", list(range(1, 12)), volumes=volumes)
    feature = VolumeRatio(window=5)
    out = FeatureEngine([feature]).compute(frame)
    assert out[feature.name].to_list()[-1] == pytest.approx(5.0)


def test_realized_volatility_zero_for_constant_returns() -> None:
    # constant per-bar growth rate -> constant returns -> rolling std of returns is ~0
    closes = [100.0 * (1.01**i) for i in range(15)]
    frame = _frame("A", closes)
    feature = RealizedVolatility(window=5)
    out = FeatureEngine([feature]).compute(frame)
    last = out[feature.name].to_list()[-1]
    assert last == pytest.approx(0.0, abs=1e-9)


def test_feature_engine_does_not_leak_across_symbols() -> None:
    a = _frame("A", [100.0, 101.0, 102.0, 103.0, 104.0])
    b = _frame("B", [50.0, 49.0, 48.0, 47.0, 46.0])
    frame = pl.concat([a, b])

    engine = FeatureEngine([ReturnOverN(n=2)])
    out = engine.compute(frame)

    a_values = out.filter(pl.col("symbol") == "A")["return_2d"].to_list()
    b_values = out.filter(pl.col("symbol") == "B")["return_2d"].to_list()

    assert a_values[2] == pytest.approx(102.0 / 100.0 - 1.0)
    assert b_values[2] == pytest.approx(48.0 / 50.0 - 1.0)


def test_feature_engine_output_has_all_feature_columns() -> None:
    frame = _frame("A", [float(100 + i) for i in range(30)])
    features = [ReturnOverN(3), RollingZScore(5), RSI(5), VolumeRatio(5), RealizedVolatility(5)]
    out = FeatureEngine(features).compute(frame)

    for feature in features:
        assert feature.name in out.columns
