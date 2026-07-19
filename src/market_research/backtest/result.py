from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

# In-sample vs out-of-sample vs holdout must always be explicit (spec §1.6) —
# a number without this label is not reportable.
Regime = Literal["in_sample", "out_of_sample", "holdout"]


@dataclass(frozen=True)
class BacktestResult:
    """Output of a single hypothesis backtest.

    `frame` carries per-bar detail: symbol, date, position, asset_return, cost,
    strategy_return, equity (cumulative product of 1+strategy_return). `regime` is
    supplied by the caller, not inferred — the Backtester only knows the data it was
    handed, not what role that data plays in the broader validation protocol.
    """

    frame: pl.DataFrame
    regime: Regime
