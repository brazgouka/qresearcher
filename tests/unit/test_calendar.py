from __future__ import annotations

from datetime import date

from market_research.data import MarketCalendar


def test_is_session_true_on_trading_day() -> None:
    cal = MarketCalendar("XNYS")
    assert cal.is_session(date(2024, 1, 2)) is True


def test_is_session_false_on_holiday() -> None:
    cal = MarketCalendar("XNYS")
    assert cal.is_session(date(2024, 1, 1)) is False


def test_sessions_in_range_excludes_holiday() -> None:
    cal = MarketCalendar("XNYS")
    sessions = cal.sessions_in_range(date(2024, 1, 1), date(2024, 1, 5))
    assert date(2024, 1, 1) not in sessions
    assert sessions == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
