from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pandas as pd
import polars as pl
import yfinance as yf

from market_research.data.calendar import MarketCalendar
from market_research.data.provider import Frequency
from market_research.storage import ContentStore

_OHLCV_SCHEMA = {
    "symbol": pl.Utf8,
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
}

HistoryFn = Callable[[str, date, date], pd.DataFrame]


def _yfinance_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    return yf.Ticker(symbol).history(start=start.isoformat(), end=end.isoformat(), auto_adjust=True)


class YahooDataProvider:
    """Yahoo Finance dev-stub `DataProvider` — for local development only.

    NOTE — survivorship bias: `point_in_time_universe` returns the fixed universe passed
    at construction time, not the constituents as they actually existed on `as_of`.
    Delisted/renamed symbols are silently absent. Acceptable for Slice 0 (proving the
    validation machinery rejects noise), but must not be used for a real research run —
    Polygon/Alpaca point-in-time universes are the fix (spec §7 "widen" phase).
    """

    def __init__(
        self,
        universe: list[str],
        store: ContentStore | None = None,
        exchange: str = "XNYS",
        history_fn: HistoryFn | None = None,
    ) -> None:
        self._universe = list(universe)
        self._store = store
        self._calendar = MarketCalendar(exchange)
        self._history_fn = history_fn or _yfinance_history

    @property
    def calendar(self) -> MarketCalendar:
        return self._calendar

    def point_in_time_universe(self, as_of: date) -> list[str]:
        return list(self._universe)

    def get_ohlcv(
        self, symbols: list[str], start: date, end: date, freq: Frequency = "1d"
    ) -> pl.DataFrame:
        if freq != "1d":
            raise ValueError(f"YahooDataProvider only supports daily bars, got freq={freq!r}")

        params = {
            "symbols": sorted(symbols),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "freq": freq,
        }
        if self._store is not None and self._store.exists("ohlcv_yahoo", params):
            return self._store.read("ohlcv_yahoo", params)

        frame = self._download(symbols, start, end)
        if self._store is not None:
            self._store.write("ohlcv_yahoo", params, frame)
        return frame

    def _download(self, symbols: list[str], start: date, end: date) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for symbol in symbols:
            history = self._history_fn(symbol, start, end)
            if history.empty:
                continue
            tidy = history.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            tidy.columns = ["date", "open", "high", "low", "close", "volume"]
            frame = pl.from_pandas(tidy).with_columns(
                pl.lit(symbol).alias("symbol"),
                pl.col("date").dt.date(),
                pl.col("volume").cast(pl.Int64),
            )
            frames.append(frame.select(list(_OHLCV_SCHEMA)))

        if not frames:
            return pl.DataFrame(schema=_OHLCV_SCHEMA)
        return pl.concat(frames).sort(["symbol", "date"])
