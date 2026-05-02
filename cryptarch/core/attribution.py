"""Per-layer P&L attribution.

Daily report decomposes the total bankroll change into per-layer
contributions. This makes it possible to:

  - See which layer is currently producing returns
  - Spot regressions in any one layer
  - Justify allocator tilts with realized data

Mathematics
-----------
For each layer over a window [t0, t1]:

  realized_pnl   = sum of closed-position PnL booked in window
  funding_pnl    = sum of funding payments collected (L1 specifically)
  unrealized_pnl = sum of (mark_to_market - cost_basis) for still-open
                   positions at t1
  fees_pnl       = -sum of fees paid
  total_pnl      = realized + funding + unrealized + fees

Sum across layers = portfolio-level P&L change in the window.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerPnL:
    """P&L breakdown for one layer over a window."""
    layer: str
    realized_pnl_usd: float       # closed positions
    funding_pnl_usd: float        # only L1 typically
    unrealized_pnl_usd: float     # mark-to-market on open positions
    fees_pnl_usd: float           # -fees (negative)
    n_trades_closed: int          # closed positions in the window
    n_open_at_close: int          # still-open at end of window

    @property
    def total_pnl_usd(self) -> float:
        return (self.realized_pnl_usd + self.funding_pnl_usd
                + self.unrealized_pnl_usd + self.fees_pnl_usd)


@dataclass(frozen=True)
class AttributionReport:
    """Multi-layer attribution over a window."""
    window_start: str             # ISO timestamp
    window_end: str
    bankroll_start_usd: float
    bankroll_end_usd: float
    layers: tuple[LayerPnL, ...]

    @property
    def total_pnl_usd(self) -> float:
        return sum(l.total_pnl_usd for l in self.layers)

    @property
    def return_pct(self) -> float:
        if self.bankroll_start_usd <= 0:
            return 0.0
        return self.total_pnl_usd / self.bankroll_start_usd

    def best_layer(self) -> LayerPnL | None:
        if not self.layers:
            return None
        return max(self.layers, key=lambda l: l.total_pnl_usd)

    def worst_layer(self) -> LayerPnL | None:
        if not self.layers:
            return None
        return min(self.layers, key=lambda l: l.total_pnl_usd)


def compute_attribution(
    realized_by_layer: dict[str, float],
    funding_by_layer: dict[str, float],
    unrealized_by_layer: dict[str, float],
    fees_by_layer: dict[str, float],
    n_closed_by_layer: dict[str, int],
    n_open_by_layer: dict[str, int],
    bankroll_start_usd: float,
    bankroll_end_usd: float,
    window_start: str,
    window_end: str,
) -> AttributionReport:
    """Combine per-layer metrics into an AttributionReport. Pure-function;
    inputs come from store queries."""
    all_layers = (
        set(realized_by_layer) | set(funding_by_layer)
        | set(unrealized_by_layer) | set(fees_by_layer)
        | set(n_closed_by_layer) | set(n_open_by_layer)
    )
    layer_pnls = tuple(
        LayerPnL(
            layer=name,
            realized_pnl_usd=realized_by_layer.get(name, 0.0),
            funding_pnl_usd=funding_by_layer.get(name, 0.0),
            unrealized_pnl_usd=unrealized_by_layer.get(name, 0.0),
            fees_pnl_usd=fees_by_layer.get(name, 0.0),
            n_trades_closed=n_closed_by_layer.get(name, 0),
            n_open_at_close=n_open_by_layer.get(name, 0),
        )
        for name in sorted(all_layers)
    )
    return AttributionReport(
        window_start=window_start,
        window_end=window_end,
        bankroll_start_usd=bankroll_start_usd,
        bankroll_end_usd=bankroll_end_usd,
        layers=layer_pnls,
    )
