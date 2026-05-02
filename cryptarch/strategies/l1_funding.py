"""Layer 1: Funding-rate arbitrage (delta-neutral basis trade).

Mechanics
---------
A perpetual futures contract trades at a price that may differ from the
underlying spot. The funding mechanism — paid every 8 hours on most
exchanges — keeps the perp tethered to spot:

  perp_price > spot_price  → funding > 0  → longs PAY shorts every 8h
  perp_price < spot_price  → funding < 0  → shorts pay longs every 8h

Strategy
--------
When funding is sufficiently positive on a perp:
  1. LONG spot for some notional N (we pay spot_price for N/spot_price units)
  2. SHORT perp for the same notional N (we sell N/perp_price units)
  3. Net delta ≈ 0 (price moves don't matter)
  4. We RECEIVE funding payments every 8h (~33% APR at 0.0003 per 8h)
  5. When funding compresses, we close both legs

The math
--------
Let F_8h = funding rate per 8h period (e.g. 0.0003 = 0.03%)
Let N = total notional we deploy (split equally between spot and short-perp)

  Daily funding income:  F_8h × 3 × N / 2     (the perp leg only is N/2)

  Wait — actually: notional on each side is N/2 if we split a budget N
  between spot and perp. But we usually express "deployed capital" as
  the SUM (N), and "position notional per side" as N/2.

  Yield/day from funding (per dollar deployed):
      r_daily = F_8h × 3 × (N/2) / N = F_8h × 1.5

  At F_8h = 0.0003 (0.03% per 8h, common):
      r_daily = 0.045% (≈16% APR)

  At F_8h = 0.0010 (0.1% per 8h, hot bull market):
      r_daily = 0.15% (≈55% APR)

Risk dimensions
---------------
1. Funding flip: if the rate goes negative, we now PAY funding. Need
   to monitor and close when sustained negative.
2. Basis risk: if we close both legs separately, perp/spot can diverge
   between executions. Use simultaneous order placement.
3. Liquidation risk on the short perp: the bot must maintain margin if
   spot moves up sharply (perp position loses, but spot leg gains —
   net P&L roughly zero, but exchange may liquidate the perp before we
   can post margin from the spot leg).
4. Counter-party risk: exchange insolvency. Diversify across exchanges.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FundingArbCandidate:
    """A potential funding-rate arb on one (spot, perp) pair."""
    spot_exchange: str
    spot_symbol: str            # e.g. "BTC/USDT"
    perp_exchange: str
    perp_symbol: str            # e.g. "BTC/USDT:USDT"
    spot_price: float
    perp_price: float
    funding_rate_8h: float      # e.g. 0.0003 = 0.03% per 8h

    @property
    def basis_pct(self) -> float:
        """(perp - spot) / spot. Positive when perp trades premium."""
        if self.spot_price <= 0:
            return 0.0
        return (self.perp_price - self.spot_price) / self.spot_price

    @property
    def expected_daily_yield_pct(self) -> float:
        """Per dollar of TOTAL deployed (spot + perp combined). The perp
        leg is half the total notional, so daily yield = funding × 3 × 0.5."""
        return self.funding_rate_8h * 3 * 0.5

    @property
    def expected_apr_pct(self) -> float:
        return self.expected_daily_yield_pct * 365


def is_attractive(
    candidate: FundingArbCandidate,
    min_funding_8h: float,
    min_basis_pct: float,
    max_basis_pct: float,
) -> bool:
    """Decide whether this candidate is worth entering.

    Three conditions must all hold:
      1. Funding rate is high enough to clear our threshold (covers fees)
      2. Basis is positive (perp >= spot) — confirms perp premium structurally
      3. Basis isn't already extreme (mean-reversion risk if too high)
    """
    if candidate.funding_rate_8h < min_funding_8h:
        return False
    if candidate.basis_pct < min_basis_pct:
        return False
    if candidate.basis_pct > max_basis_pct:
        return False
    return True


@dataclass(frozen=True)
class PositionPlan:
    """Sizing plan for a funding-arb position."""
    candidate: FundingArbCandidate
    total_notional_usd: float       # spot + perp leg combined
    spot_notional_usd: float        # half (we long spot)
    perp_notional_usd: float        # half (we short perp)
    spot_size_base: float           # quantity of base asset to buy on spot
    perp_size_base: float           # quantity of base asset to short on perp
    expected_daily_pnl_usd: float
    expected_apr_pct: float


def plan_position(
    candidate: FundingArbCandidate,
    total_capital_usd: float,
) -> PositionPlan:
    """Given a budget, compute the sizes for each leg.

    We split equally between spot and perp by NOTIONAL value. Note that
    if spot_price ≈ perp_price (near-zero basis), the base sizes are
    nearly equal too — perfect delta-neutrality.
    """
    if total_capital_usd <= 0:
        raise ValueError(f"total_capital_usd must be positive, got {total_capital_usd}")
    spot_notional = total_capital_usd / 2.0
    perp_notional = total_capital_usd / 2.0
    spot_size = spot_notional / candidate.spot_price
    perp_size = perp_notional / candidate.perp_price
    daily_pnl = total_capital_usd * candidate.expected_daily_yield_pct
    return PositionPlan(
        candidate=candidate,
        total_notional_usd=total_capital_usd,
        spot_notional_usd=spot_notional,
        perp_notional_usd=perp_notional,
        spot_size_base=spot_size,
        perp_size_base=perp_size,
        expected_daily_pnl_usd=daily_pnl,
        expected_apr_pct=candidate.expected_apr_pct,
    )


@dataclass(frozen=True)
class HedgeDeviation:
    """How far off-hedge a current position has drifted."""
    delta_base: float           # signed; positive = we're net-long base
    delta_pct_of_position: float    # delta_base × spot_price / position_notional


def compute_hedge_deviation(
    spot_size_base: float,
    perp_size_base: float,
    current_spot_price: float,
    position_notional_usd: float,
) -> HedgeDeviation:
    """Net delta = spot_long - perp_short. Each in base units. We want this
    to be ≈ 0. Express as fraction of position notional for thresholding."""
    delta = spot_size_base - perp_size_base
    delta_usd = delta * current_spot_price
    pct = delta_usd / position_notional_usd if position_notional_usd > 0 else 0.0
    return HedgeDeviation(delta_base=delta, delta_pct_of_position=pct)


def should_close_position(
    candidate: FundingArbCandidate,
    min_funding_8h_to_hold: float,
) -> bool:
    """Close the position when funding compresses below the hold-threshold.

    The hold-threshold is typically lower than the entry threshold (we
    don't want to flap in and out around a single number). e.g. enter at
    +0.03% per 8h, hold until below +0.01%."""
    return candidate.funding_rate_8h < min_funding_8h_to_hold
