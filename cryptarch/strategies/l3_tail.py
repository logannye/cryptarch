"""Layer 3: Tail-hedge via OTM strangles on Deribit.

Mechanics
---------
Continuously hold a long out-of-the-money strangle:
  BUY 30-60 day OTM call (strike = spot × (1 + otm_pct))
  BUY 30-60 day OTM put  (strike = spot × (1 - otm_pct))

Mathematical role in the portfolio
----------------------------------
A long strangle has STRICT CONVEXITY in the underlying price: payoff
P(spot_at_expiry) = max(0, spot - K_call) + max(0, K_put - spot) - premium.

Daily P&L for held strangle:
  d/dt P = -theta            (time decay, negative; the daily premium burn)
  d/dσ P = +vega             (positive vol exposure; we WIN when IV rises)
  d²/dS² P = +gamma          (positive convexity; we WIN as price moves either way)

The L1+L2 layers are SHORT vol (collect premium / fade dislocations) — they
lose during regime shifts. L3 is LONG vol — it WINS during regime shifts.
The negative correlation reduces portfolio variance while preserving
expected return. Empirically: L3 costs ~0.05%/day in theta during quiet
periods, pays out 5-15% during major vol regime changes (Mar 2020,
May 2021, FTX collapse).

Sizing discipline
-----------------
The cost of L3 (daily theta) must be self-funded by L1+L2 yield. We cap
daily theta budget at a fraction of cumulative L1+L2 PnL. If theta
exceeds budget, reduce position size at next roll.

Roll discipline
---------------
When DTE drops below `target_dte_min` (default 30 days), we close the
existing strangle and open a fresh one at `target_dte_target` (45 days).
Don't let positions decay into gamma-spike territory — short-DTE options
have wild greeks that misbehave near expiry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


# ── Strike selection ────────────────────────────────────────────────


def select_strangle_strikes(
    spot: float,
    otm_pct: float = 0.20,
    strike_step: float = 1000.0,
) -> tuple[float, float]:
    """Pick (call_strike, put_strike) for an OTM strangle anchored at spot.

    Strikes are rounded to `strike_step` because exchanges only list
    discrete strikes (e.g. Deribit BTC strikes are usually $1k/$2k apart).

    Args:
        spot: current spot price
        otm_pct: how far OTM each leg is (default 20%)
        strike_step: rounding granularity for strikes (Deribit BTC ~ $1000)
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if not 0 < otm_pct < 1:
        raise ValueError(f"otm_pct must be in (0, 1), got {otm_pct}")
    if strike_step <= 0:
        raise ValueError(f"strike_step must be positive, got {strike_step}")

    raw_call = spot * (1.0 + otm_pct)
    raw_put = spot * (1.0 - otm_pct)
    # Round CALL up (further OTM = cheaper); PUT down (further OTM = cheaper)
    call_strike = math.ceil(raw_call / strike_step) * strike_step
    put_strike = math.floor(raw_put / strike_step) * strike_step
    return call_strike, put_strike


# ── Strangle valuation ──────────────────────────────────────────────


@dataclass(frozen=True)
class OptionLeg:
    """One option contract within a strangle position."""
    instrument_name: str        # e.g. "BTC-31MAY26-95000-C"
    underlying: str             # "BTC" | "ETH"
    expiry: datetime
    strike: float
    option_type: str            # "C" | "P"
    contracts: float            # number of contracts (1 contract = 1 BTC on Deribit)
    entry_premium_usd: float    # USD paid per contract at entry
    entry_iv: float             # implied volatility at entry (e.g. 0.65)


@dataclass(frozen=True)
class StranglePosition:
    """Long strangle: one call + one put."""
    call: OptionLeg
    put: OptionLeg
    spot_at_open: float
    opened_at: datetime

    @property
    def total_premium_usd(self) -> float:
        return (self.call.entry_premium_usd * self.call.contracts
                + self.put.entry_premium_usd * self.put.contracts)

    @property
    def expiry(self) -> datetime:
        # Both legs share the same expiry by design.
        return self.call.expiry


def compute_strangle_breakeven(
    position: StranglePosition,
) -> tuple[float, float]:
    """Lower and upper breakeven prices at expiry.

    For a long strangle, the position is profitable at expiry iff
    spot ≤ put_strike - premium_per_contract OR
    spot ≥ call_strike + premium_per_contract.

    We assume equal sizing on both legs (premium split evenly per side).
    """
    n_call = position.call.contracts
    n_put = position.put.contracts
    if n_call <= 0 or n_put <= 0:
        return 0.0, float("inf")
    upper = position.call.strike + (position.call.entry_premium_usd
                                    + position.put.entry_premium_usd) / n_call
    lower = position.put.strike - (position.call.entry_premium_usd
                                   + position.put.entry_premium_usd) / n_put
    return max(0.0, lower), upper


# ── Time-to-expiry helpers ─────────────────────────────────────────


def days_to_expiry(expiry: datetime, now: datetime | None = None) -> float:
    """Calendar days remaining until expiry. Negative if past expiry."""
    if now is None:
        now = datetime.now(timezone.utc)
    return (expiry - now).total_seconds() / 86400.0


def should_roll(expiry: datetime, target_dte_min: int = 30,
                now: datetime | None = None) -> bool:
    """Roll the strangle when DTE drops below threshold. We don't let
    options decay into gamma-spike territory."""
    return days_to_expiry(expiry, now) < target_dte_min


def select_target_expiry(
    available_expiries: list[datetime],
    target_dte: int = 45,
    now: datetime | None = None,
) -> datetime | None:
    """Of the available expiries, pick the one closest to target_dte.

    We prefer 45 days because:
      - Long enough to avoid gamma-spike territory (>30 days)
      - Short enough that we benefit from gamma if vol moves (vs LEAPs
        with wide greeks)
      - Liquidity is best at 30-60 day tenor

    Args:
        available_expiries: list of expiry datetimes from the option chain
        target_dte: ideal DTE for new positions (default 45)
        now: current time (override for testing)
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if not available_expiries:
        return None
    # Filter to expiries in the future.
    future = [e for e in available_expiries if e > now]
    if not future:
        return None
    # Pick the one closest to target_dte days out.
    return min(future, key=lambda e: abs(days_to_expiry(e, now) - target_dte))


# ── Sizing — theta budget ─────────────────────────────────────────


def daily_theta_cost(
    call_premium_usd: float,
    put_premium_usd: float,
    contracts: float,
    days_to_expiry_now: float,
) -> float:
    """Approximate daily theta = total premium / DTE (linear decay).

    This is a first-order estimate; real theta is non-linear and accelerates
    near expiry. For position sizing it's good enough — over a 30-60 day
    tenor, average theta ≈ premium/DTE.

    Args:
        call_premium_usd: premium PER CONTRACT for the call leg
        put_premium_usd: same for put
        contracts: contracts per side (assumed equal)
        days_to_expiry_now: DTE at the time we open
    """
    if days_to_expiry_now <= 0:
        return 0.0
    total_premium = (call_premium_usd + put_premium_usd) * contracts
    return total_premium / days_to_expiry_now


def max_contracts_within_theta_budget(
    daily_theta_budget_usd: float,
    call_premium_usd: float,
    put_premium_usd: float,
    days_to_expiry_now: float,
) -> float:
    """Largest position size that stays within the daily theta budget.

    daily_theta = (call + put) × contracts / DTE   →
    contracts = budget × DTE / (call + put)
    """
    if daily_theta_budget_usd <= 0:
        return 0.0
    if days_to_expiry_now <= 0:
        return 0.0
    if (call_premium_usd + put_premium_usd) <= 0:
        return 0.0
    return daily_theta_budget_usd * days_to_expiry_now / (call_premium_usd + put_premium_usd)
