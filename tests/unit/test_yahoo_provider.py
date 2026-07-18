from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from market_research.data import YahooDataProvider
from market_research.storage import ContentStore


def _fake_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="D", tz="America/New_York", name="Date")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(len(index))],
            "High": [101.0 + i for i in range(len(index))],
            "Low": [99.0 + i for i in range(len(index))],
            "Close": [100.5 + i for i in range(len(index))],
            "Volume": [1000 + i for i in range(len(index))],
        },
        index=index,
    )


def _empty_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def test_get_ohlcv_returns_tidy_frame_with_expected_schema() -> None:
    provider = YahooDataProvider(universe=["AAPL"], history_fn=_fake_history)
    frame = provider.get_ohlcv(["AAPL"], date(2024, 1, 2), date(2024, 1, 3))

    assert frame.columns == ["symbol", "date", "open", "high", "low", "close", "volume"]
    assert set(frame["symbol"].to_list()) == {"AAPL"}
    assert frame["date"].dtype.__class__.__name__ == "Date"


def test_get_ohlcv_concatenates_multiple_symbols_sorted() -> None:
    provider = YahooDataProvider(universe=["AAPL", "MSFT"], history_fn=_fake_history)
    frame = provider.get_ohlcv(["MSFT", "AAPL"], date(2024, 1, 2), date(2024, 1, 3))

    assert frame["symbol"].to_list() == sorted(frame["symbol"].to_list())
    assert set(frame["symbol"].unique().to_list()) == {"AAPL", "MSFT"}


def test_get_ohlcv_skips_symbols_with_empty_history() -> None:
    provider = YahooDataProvider(universe=["DELISTED"], history_fn=_empty_history)
    frame = provider.get_ohlcv(["DELISTED"], date(2024, 1, 2), date(2024, 1, 3))
    assert frame.height == 0
    assert frame.columns == ["symbol", "date", "open", "high", "low", "close", "volume"]


def test_get_ohlcv_rejects_non_daily_freq() -> None:
    provider = YahooDataProvider(universe=["AAPL"], history_fn=_fake_history)
    with pytest.raises(ValueError):
        provider.get_ohlcv(["AAPL"], date(2024, 1, 2), date(2024, 1, 3), freq="1h")  # type: ignore[arg-type]


def test_get_ohlcv_caches_through_content_store(tmp_path) -> None:
    calls = {"count": 0}

    def counting_history(symbol: str, start: date, end: date) -> pd.DataFrame:
        calls["count"] += 1
        return _fake_history(symbol, start, end)

    store = ContentStore(tmp_path)
    provider = YahooDataProvider(universe=["AAPL"], store=store, history_fn=counting_history)

    first = provider.get_ohlcv(["AAPL"], date(2024, 1, 2), date(2024, 1, 3))
    second = provider.get_ohlcv(["AAPL"], date(2024, 1, 2), date(2024, 1, 3))

    assert calls["count"] == 1
    assert first.equals(second)


def test_point_in_time_universe_returns_fixed_universe_regardless_of_as_of() -> None:
    provider = YahooDataProvider(universe=["AAPL", "MSFT"], history_fn=_fake_history)
    assert provider.point_in_time_universe(date(2020, 1, 1)) == ["AAPL", "MSFT"]
    assert provider.point_in_time_universe(date(2024, 1, 1)) == ["AAPL", "MSFT"]
