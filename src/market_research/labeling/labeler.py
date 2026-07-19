from __future__ import annotations

from typing import Protocol

import polars as pl


class Labeler(Protocol):
    """Maps price paths + candidate event start times to labels + t1 + sample weights.

    Output schema: `symbol, t0, t1, label (-1/0/+1), ret, sample_weight`.

    `t1` is the time each label *resolves* — purged CV needs this (not just `t0`) to
    embargo correctly, since a label isn't settled until t1 (spec §1.3). `sample_weight`
    downweights labels whose [t0, t1] spans overlap heavily with other labels
    (label-uniqueness weighting, spec §5) so overlapping outcomes don't get overcounted.
    """

    def label(self, prices: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame: ...
