from __future__ import annotations

from datetime import date

import exchange_calendars as xcals


class MarketCalendar:
    """Thin wrapper over exchange_calendars — the one place calendar/session logic lives.

    Kept separate from the primary Polars dataframe pipeline: this is trading-day/session
    bookkeeping (is a given day open, what sessions fall in a range), not market data.
    """

    def __init__(self, exchange: str = "XNYS") -> None:
        self.exchange = exchange
        self._cal = xcals.get_calendar(exchange)

    def is_session(self, day: date) -> bool:
        return bool(self._cal.is_session(day.isoformat()))

    def sessions_in_range(self, start: date, end: date) -> list[date]:
        sessions = self._cal.sessions_in_range(start.isoformat(), end.isoformat())
        return [ts.date() for ts in sessions]
