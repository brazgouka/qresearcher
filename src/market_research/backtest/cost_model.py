from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Transaction cost assumptions, pessimistic by default (spec §1.5).

    An edge that only survives at zero cost is presumed to be spread capture, not a
    real effect — so defaults here are deliberately not flattering: half-spread,
    adverse slippage, and commission are all charged on every unit of position change.
    """

    spread_bps: float = 10.0
    slippage_bps: float = 5.0
    commission_bps: float = 1.0

    @property
    def cost_rate(self) -> float:
        """Fraction of traded notional charged per unit of |position change|."""
        return (self.spread_bps + self.slippage_bps + self.commission_bps) / 10_000.0


@dataclass(frozen=True)
class SizingModel:
    """Maps a 0/1 hypothesis signal to a fraction of capital at risk when "on"."""

    target_gross_exposure: float = 1.0
