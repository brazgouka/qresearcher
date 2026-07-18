from __future__ import annotations

from datetime import date
from typing import Literal, Protocol

import polars as pl

from market_research.data.calendar import MarketCalendar

# Slice 0 only supports daily bars — intraday is a "widen" phase concern (spec §7).
Frequency = Literal["1d"]


class DataProvider(Protocol):
    """Uniform, calendar-aware, point-in-time access to market data.

    `point_in_time_universe` exists specifically to prevent survivorship bias: a
    provider must be able to say which symbols existed *as of* a given date, not just
    which symbols exist today.
    """

    @property
    def calendar(self) -> MarketCalendar: ...

    def get_ohlcv(
        self, symbols: list[str], start: date, end: date, freq: Frequency = "1d"
    ) -> pl.DataFrame: ...

    def point_in_time_universe(self, as_of: date) -> list[str]: ...
