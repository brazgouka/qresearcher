from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from market_research.labeling import TripleBarrierLabeler

SETTLING = [100.0, 101.0, 100.0, 101.0, 100.0]  # establishes a nonzero vol estimate by index 4


def _dates(n: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]


def _prices(symbol: str, closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"symbol": [symbol] * len(closes), "date": _dates(len(closes)), "close": closes}
    )


def _events(symbol: str, t0_indices: list[int], all_dates: list[date]) -> pl.DataFrame:
    return pl.DataFrame(
        {"symbol": [symbol] * len(t0_indices), "t0": [all_dates[i] for i in t0_indices]}
    )


@pytest.fixture
def labeler() -> TripleBarrierLabeler:
    return TripleBarrierLabeler(pt_mult=2.0, sl_mult=2.0, max_holding_bars=5, vol_window=3)


def test_profit_take_barrier_hit_first(labeler: TripleBarrierLabeler) -> None:
    closes = SETTLING + [150.0, 151.0, 152.0, 153.0, 154.0]  # +50% on first post-event bar
    prices = _prices("PT", closes)
    events = _events("PT", [4], prices["date"].to_list())

    out = labeler.label(prices, events)

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["label"] == 1
    assert row["t1"] == prices["date"].to_list()[5]
    assert row["ret"] == pytest.approx(0.5)


def test_stop_loss_barrier_hit_first(labeler: TripleBarrierLabeler) -> None:
    closes = SETTLING + [50.0, 49.0, 48.0, 47.0, 46.0]  # -50% on first post-event bar
    prices = _prices("SL", closes)
    events = _events("SL", [4], prices["date"].to_list())

    out = labeler.label(prices, events)

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["label"] == -1
    assert row["t1"] == prices["date"].to_list()[5]
    assert row["ret"] == pytest.approx(-0.5)


def test_time_barrier_hit_with_positive_return(labeler: TripleBarrierLabeler) -> None:
    # stays within the ~2.3%-wide barriers the whole way, ends modestly positive
    closes = SETTLING + [100.5, 100.8, 100.3, 100.6, 100.9]
    prices = _prices("TIME", closes)
    events = _events("TIME", [4], prices["date"].to_list())

    out = labeler.label(prices, events)

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["label"] == 1
    assert row["t1"] == prices["date"].to_list()[9]  # horizon = i0(4) + max_holding_bars(5)
    assert 0.0 < row["ret"] < 0.023  # positive but small, well inside the pt barrier


def test_time_barrier_hit_with_negative_return(labeler: TripleBarrierLabeler) -> None:
    closes = SETTLING + [99.7, 99.4, 99.8, 99.5, 99.2]
    prices = _prices("TIMENEG", closes)
    events = _events("TIMENEG", [4], prices["date"].to_list())

    out = labeler.label(prices, events)

    row = out.row(0, named=True)
    assert row["label"] == -1
    assert -0.023 < row["ret"] < 0.0


def test_event_before_volatility_is_available_is_skipped(labeler: TripleBarrierLabeler) -> None:
    closes = SETTLING + [100.5, 100.8, 100.3, 100.6, 100.9]
    prices = _prices("EARLY", closes)
    # t0 at index 0: vol_window=3 needs 3 returns of history, unavailable this early
    events = _events("EARLY", [0], prices["date"].to_list())

    out = labeler.label(prices, events)

    assert out.height == 0


def test_isolated_event_has_sample_weight_one(labeler: TripleBarrierLabeler) -> None:
    closes = SETTLING + [150.0, 151.0, 152.0, 153.0, 154.0]
    prices = _prices("ISO", closes)
    events = _events("ISO", [4], prices["date"].to_list())

    out = labeler.label(prices, events)

    assert out.row(0, named=True)["sample_weight"] == pytest.approx(1.0)


def test_overlapping_events_have_lower_sample_weight_than_isolated(
    labeler: TripleBarrierLabeler,
) -> None:
    closes = SETTLING + [100.3, 100.6, 100.2, 100.5, 100.8, 101.0]
    prices = _prices("OVERLAP", closes)
    all_dates = prices["date"].to_list()
    # two entries one bar apart -> their [t0, t1] spans overlap heavily
    events = _events("OVERLAP", [4, 5], all_dates)

    out = labeler.label(prices, events)

    assert out.height == 2
    weights = out.sort("t0")["sample_weight"].to_list()
    assert all(w < 1.0 for w in weights)
