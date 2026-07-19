from __future__ import annotations

import polars as pl
import pytest
from leakage_utils import assert_no_lookahead

from market_research.features import (
    RSI,
    RealizedVolatility,
    ReturnOverN,
    RollingZScore,
    VolumeRatio,
)

FEATURES = [
    ReturnOverN(n=5),
    RollingZScore(window=20),
    RSI(period=14),
    VolumeRatio(window=20),
    RealizedVolatility(window=20),
]


@pytest.mark.parametrize("feature", FEATURES, ids=lambda f: f.name)
@pytest.mark.parametrize("symbol", ["AAA", "BBB"])
def test_feature_has_no_lookahead(feature, symbol: str, multi_symbol_ohlcv: pl.DataFrame) -> None:
    assert_no_lookahead(feature, multi_symbol_ohlcv, symbol)
