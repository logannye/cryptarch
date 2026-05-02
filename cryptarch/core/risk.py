"""Cross-layer risk netting + portfolio-level exposure metrics.

The naive total_at_risk (sum of position notionals) overstates risk
because hedged positions partially offset:

  - L1 funding-arb: long spot + short perp on same underlying = ~0 delta
  - L2 cascade ladder: long spot ladder = +delta exposure
  - L3 strangle: small directional delta (≈0 by design)

If L1 has a $1000 BTC funding arb and L2 has a $500 BTC ladder filled,
the *netted* portfolio delta is just the L2 leg (+$500 of BTC). The
sum-of-notionals would call this $1500 of risk.

This module computes net delta per underlying for portfolio-level
analytics. It informs (but doesn't replace) the per-layer hard caps —
those still bound concentration and worst-case loss.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetExposure:
    """Per-underlying net delta exposure across all open positions."""
    underlying: str             # "BTC" | "ETH" | "DOGE" etc.
    long_notional_usd: float    # sum of all long-side positions on this underlying
    short_notional_usd: float   # sum of all short-side positions
    net_notional_usd: float     # long - short (signed)


@dataclass(frozen=True)
class PortfolioRisk:
    """Aggregate risk view across all layers."""
    gross_at_risk_usd: float    # sum of |position_notional|
    net_at_risk_usd: float      # sum of |net_notional| per underlying
    underlyings: tuple[NetExposure, ...]


def compute_portfolio_risk(positions: list[dict]) -> PortfolioRisk:
    """Aggregate net delta per underlying.

    Each position dict should have:
      - underlying: str (the base asset symbol — "BTC", "ETH", etc.)
      - notional_usd: float
      - direction: str — "long" | "short"

    Hedged L1 positions submit two entries (spot long + perp short on
    same underlying). After netting, they contribute ~0 to the
    portfolio's directional risk.
    """
    by_underlying: dict[str, dict[str, float]] = {}
    for p in positions:
        u = p.get("underlying")
        if not u:
            continue
        notional = float(p.get("notional_usd", 0))
        direction = p.get("direction", "long")
        agg = by_underlying.setdefault(u, {"long": 0.0, "short": 0.0})
        if direction == "long":
            agg["long"] += notional
        elif direction == "short":
            agg["short"] += notional

    exposures: list[NetExposure] = []
    gross = 0.0
    net_abs = 0.0
    for u, agg in by_underlying.items():
        net = agg["long"] - agg["short"]
        exposures.append(NetExposure(
            underlying=u,
            long_notional_usd=agg["long"],
            short_notional_usd=agg["short"],
            net_notional_usd=net,
        ))
        gross += agg["long"] + agg["short"]
        net_abs += abs(net)

    return PortfolioRisk(
        gross_at_risk_usd=gross,
        net_at_risk_usd=net_abs,
        underlyings=tuple(sorted(exposures, key=lambda e: -abs(e.net_notional_usd))),
    )


def netting_efficiency(risk: PortfolioRisk) -> float:
    """Fraction of gross exposure netted by hedging. 1.0 = fully hedged.
    Empty portfolio → 0.0."""
    if risk.gross_at_risk_usd <= 0:
        return 0.0
    return 1.0 - (risk.net_at_risk_usd / risk.gross_at_risk_usd)
