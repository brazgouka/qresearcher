from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from market_research.backtest import CostModel, EventDrivenBacktester, SizingModel
from market_research.hypotheses import Threshold


def _dates(n: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]


@pytest.fixture
def priced_features() -> pl.DataFrame:
    # exact +10% daily growth -> asset_return is exactly 0.10 every bar after the first
    closes = [100.0, 110.0, 121.0, 133.1, 146.41, 161.051]
    go_signal = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]  # "go long" condition first true at index 2
    return pl.DataFrame(
        {
            "symbol": ["A"] * 6,
            "date": _dates(6),
            "close": closes,
            "go_signal": go_signal,
        }
    )


def test_position_lags_signal_by_one_bar_no_lookahead(priced_features: pl.DataFrame) -> None:
    hyp = Threshold("go_signal", "==", 1.0)
    result = EventDrivenBacktester().run(
        hyp, priced_features, CostModel(), SizingModel(), regime="in_sample"
    )
    positions = result.frame["position"].to_list()
    # signal first true at index 2 -> position (held next bar) first becomes 1 at index 3
    assert positions == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_strategy_return_excludes_the_bar_that_produced_the_signal(
    priced_features: pl.DataFrame,
) -> None:
    hyp = Threshold("go_signal", "==", 1.0)
    result = EventDrivenBacktester().run(
        hyp, priced_features, CostModel(), SizingModel(), regime="in_sample"
    )
    strategy_returns = result.frame["strategy_return"].to_list()
    # index 2 is the bar the signal turned on -> must earn nothing from that bar's own move
    assert strategy_returns[2] == pytest.approx(0.0)


def test_cost_charged_once_on_entry_then_zero_while_holding(priced_features: pl.DataFrame) -> None:
    costs = CostModel(spread_bps=10.0, slippage_bps=5.0, commission_bps=1.0)
    hyp = Threshold("go_signal", "==", 1.0)
    result = EventDrivenBacktester().run(
        hyp, priced_features, costs, SizingModel(), regime="in_sample"
    )

    frame = result.frame
    cost_rate = costs.cost_rate
    assert cost_rate == pytest.approx(16.0 / 10_000.0)

    strategy_returns = frame["strategy_return"].to_list()
    # entry at index 3: full asset_return(0.10) minus one entry cost
    assert strategy_returns[3] == pytest.approx(0.10 - cost_rate)
    # already holding at index 4, 5: no further cost, full asset_return
    assert strategy_returns[4] == pytest.approx(0.10)
    assert strategy_returns[5] == pytest.approx(0.10)


def test_equity_curve_compounds_strategy_returns(priced_features: pl.DataFrame) -> None:
    costs = CostModel(spread_bps=10.0, slippage_bps=5.0, commission_bps=1.0)
    hyp = Threshold("go_signal", "==", 1.0)
    result = EventDrivenBacktester().run(
        hyp, priced_features, costs, SizingModel(), regime="in_sample"
    )

    equity = result.frame["equity"].to_list()
    expected_3 = 1.0 * (1 + (0.10 - costs.cost_rate))
    expected_4 = expected_3 * 1.10
    expected_5 = expected_4 * 1.10
    assert equity[3] == pytest.approx(expected_3)
    assert equity[4] == pytest.approx(expected_4)
    assert equity[5] == pytest.approx(expected_5)


def test_regime_is_passed_through(priced_features: pl.DataFrame) -> None:
    hyp = Threshold("go_signal", "==", 1.0)
    result = EventDrivenBacktester().run(
        hyp, priced_features, CostModel(), SizingModel(), regime="holdout"
    )
    assert result.regime == "holdout"


def test_symbols_do_not_leak_into_each_other() -> None:
    a = pl.DataFrame(
        {
            "symbol": ["A"] * 4,
            "date": _dates(4),
            "close": [100.0, 200.0, 100.0, 200.0],  # huge swings
            "go_signal": [1.0, 1.0, 1.0, 1.0],
        }
    )
    b = pl.DataFrame(
        {
            "symbol": ["B"] * 4,
            "date": _dates(4),
            "close": [50.0, 50.5, 51.0, 51.5],  # calm, steady
            "go_signal": [1.0, 1.0, 1.0, 1.0],
        }
    )
    frame = pl.concat([a, b])
    hyp = Threshold("go_signal", "==", 1.0)
    result = EventDrivenBacktester().run(hyp, frame, CostModel(), SizingModel(), regime="in_sample")

    b_returns = result.frame.filter(pl.col("symbol") == "B")["strategy_return"].to_list()
    # B's returns must reflect only B's calm price path, not A's huge swings
    assert all(abs(r) < 0.05 for r in b_returns if r is not None)
