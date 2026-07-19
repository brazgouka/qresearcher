from __future__ import annotations

from datetime import date

import polars as pl


class TripleBarrierLabeler:
    """Path-dependent labeling: for each event start t0, find which of three barriers
    (profit-take, stop-loss, time) the price path touches first.

    Barrier widths are `pt_mult`/`sl_mult` * a trailing volatility estimate at t0, so
    they scale with the instrument's own regime rather than being a fixed percentage
    (Lopez de Prado, *Advances in Financial Machine Learning*, ch. 3). If neither
    profit-take nor stop-loss is touched within `max_holding_bars`, the label is the
    sign of the return at the time (vertical) barrier.
    """

    def __init__(
        self,
        pt_mult: float = 2.0,
        sl_mult: float = 2.0,
        max_holding_bars: int = 10,
        vol_window: int = 20,
    ) -> None:
        self.pt_mult = pt_mult
        self.sl_mult = sl_mult
        self.max_holding_bars = max_holding_bars
        self.vol_window = vol_window

    def label(self, prices: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
        prices = prices.sort(["symbol", "date"])
        vol = self._volatility(prices)

        rows: list[dict] = []
        for symbol in events["symbol"].unique().to_list():
            symbol_prices = prices.filter(pl.col("symbol") == symbol)
            symbol_vol = vol.filter(pl.col("symbol") == symbol)
            symbol_events = events.filter(pl.col("symbol") == symbol)

            dates = symbol_prices["date"].to_list()
            closes = symbol_prices["close"].to_list()
            vols = symbol_vol["vol"].to_list()
            date_index = {d: i for i, d in enumerate(dates)}

            for t0 in symbol_events["t0"].to_list():
                row = self._label_one(symbol, t0, date_index, dates, closes, vols)
                if row is not None:
                    rows.append(row)

        labeled = pl.DataFrame(
            rows,
            schema={
                "symbol": pl.Utf8,
                "t0": pl.Date,
                "t1": pl.Date,
                "label": pl.Int64,
                "ret": pl.Float64,
            },
        )
        weights = _uniqueness_weights(labeled)
        return labeled.join(weights, on=["symbol", "t0"], how="left")

    def _label_one(
        self,
        symbol: str,
        t0: date,
        date_index: dict[date, int],
        dates: list[date],
        closes: list[float],
        vols: list[float | None],
    ) -> dict | None:
        i0 = date_index.get(t0)
        if i0 is None:
            return None
        sigma = vols[i0]
        if sigma is None or sigma <= 0:
            return None  # not enough history for a volatility estimate yet

        p0 = closes[i0]
        upper = p0 * (1 + self.pt_mult * sigma)
        lower = p0 * (1 - self.sl_mult * sigma)
        horizon = min(i0 + self.max_holding_bars, len(closes) - 1)

        label = 0
        t1_idx = horizon
        for i in range(i0 + 1, horizon + 1):
            if closes[i] >= upper:
                label, t1_idx = 1, i
                break
            if closes[i] <= lower:
                label, t1_idx = -1, i
                break
        else:
            ret_at_horizon = closes[horizon] / p0 - 1.0
            label = 1 if ret_at_horizon > 0 else (-1 if ret_at_horizon < 0 else 0)

        ret = closes[t1_idx] / p0 - 1.0
        return {"symbol": symbol, "t0": t0, "t1": dates[t1_idx], "label": label, "ret": ret}

    def _volatility(self, prices: pl.DataFrame) -> pl.DataFrame:
        return (
            prices.lazy()
            .with_columns(
                pl.col("close")
                .pct_change()
                .over("symbol")
                .rolling_std(window_size=self.vol_window)
                .over("symbol")
                .alias("vol")
            )
            .select(["symbol", "date", "vol"])
            .collect()
        )


def _uniqueness_weights(labeled: pl.DataFrame) -> pl.DataFrame:
    """Average uniqueness per label: 1 / (concurrently active labels) at each bar the
    label spans, averaged over [t0, t1]. Labels overlapping many others are downweighted.
    """
    if labeled.height == 0:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "t0": pl.Date, "sample_weight": pl.Float64})

    weights: list[dict] = []
    for symbol in labeled["symbol"].unique().to_list():
        sub = labeled.filter(pl.col("symbol") == symbol)
        t0s = sub["t0"].to_list()
        t1s = sub["t1"].to_list()

        all_dates = sorted({*t0s, *t1s})
        concurrency = dict.fromkeys(all_dates, 0)
        for t0, t1 in zip(t0s, t1s, strict=True):
            for d in all_dates:
                if t0 <= d <= t1:
                    concurrency[d] += 1

        for t0, t1 in zip(t0s, t1s, strict=True):
            span_dates = [d for d in all_dates if t0 <= d <= t1]
            uniqueness = sum(1.0 / concurrency[d] for d in span_dates) / len(span_dates)
            weights.append({"symbol": symbol, "t0": t0, "sample_weight": uniqueness})

    schema = {"symbol": pl.Utf8, "t0": pl.Date, "sample_weight": pl.Float64}
    return pl.DataFrame(weights, schema=schema)
