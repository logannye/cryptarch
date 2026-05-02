"""Layer 2: Liquidation cascade capture.

Mechanics
---------
When leverage builds up (high open interest, sustained positive funding,
compressed volatility), small price moves cascade into forced liquidations
that overshoot the actual fundamental price by 2-5%. Place LIMIT BUY orders
1-4% below current spot — they sit unfilled in normal times, but during
cascades the order book trough touches our ladder and we fill at deep
discounts. Sell into the post-cascade bounce.

Math: why this works
--------------------
Liquidations are FORCED MARKET ORDERS. They consume the order book without
regard to price. If a long position is liquidated at notional N, the
exchange dumps N/spot units of base into the bid side. Price moves to:

    p* such that V_book(p*) = N
    where V_book(p) = ∫_{p}^{spot} bid_depth(x) dx

Because order books are typically thinner far from mid (especially during
volatility), N creates DISPROPORTIONATELY LARGE price moves during
cascades. Empirically, cascade overshoot averages 2-3× the fundamental-
driven price move that triggered the cascade.

Within minutes, patient capital arrives and prices revert. The arb is to
be that patient capital — but waiting at predetermined levels (the ladder)
rather than deploying reactively.

Asymmetric payoff
-----------------
Per filled rung, empirical:
  - 70% of fills mean-revert within an hour → +1-3% gain
  - 25% of fills hold flat for some time → time-stop at ~0%
  - 5% of fills extend further into doom-loop → -2-3% loss
  Net: ~+0.6%/fill expected, with positive skew (big winners on the rare
  major cascades that go deeper than expected before reverting).

Combined with funding-rate yield (Layer 1) and tail hedge (Layer 3), the
total portfolio is structurally tail-asymmetric.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ── Statistics helpers ──────────────────────────────────────────────


def percentile_rank(value: float, history: list[float]) -> float:
    """What % of `history` is at-or-below `value`. Returns 0-100.

    Empty history → 50 (neutral / cold-start). Single-element history
    → 0 if value < it, 100 if >, 50 if equal.

    This is the OI-percentile signal: where does current OI sit vs the
    rolling history? High percentile = leverage has piled up vs recent
    norm = cascade-prone."""
    if not history:
        return 50.0
    n_at_or_below = sum(1 for h in history if h <= value)
    return (n_at_or_below / len(history)) * 100.0


def realized_vol_from_closes(
    closes: list[float], lookback: int | None = None,
) -> float:
    """Realized volatility = stdev of log returns. Returns the unannualized
    σ (per-bar) so the caller can decide whether to compare ratios or
    annualize. We use ratios (recent_vol / historical_vol) so unannualized
    is fine.

    If `lookback` is given, only use the last N closes. Otherwise all of them.
    """
    if not closes or len(closes) < 2:
        return 0.0
    if lookback is not None and lookback < len(closes):
        closes = closes[-lookback:]
    log_rets: list[float] = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            continue
        log_rets.append(math.log(closes[i] / closes[i - 1]))
    if len(log_rets) < 2:
        return 0.0
    mean = sum(log_rets) / len(log_rets)
    variance = sum((r - mean) ** 2 for r in log_rets) / (len(log_rets) - 1)
    return math.sqrt(variance)



# ── Cascade probability ────────────────────────────────────────────


def cascade_probability(
    oi_percentile: float,
    funding_rate_24h_avg: float,
    recent_24h_vol_pct: float,
    historical_7d_vol_pct: float,
) -> float:
    """Composite 0-1 score for "is a cascade likely soon?"

    Inputs:
      oi_percentile         — current OI vs 7-day rolling history (0-100)
      funding_rate_24h_avg — mean 8h funding over last 24h (e.g. 0.0005)
      recent_24h_vol_pct   — realized 24h vol as fraction (e.g. 0.05 = 5%)
      historical_7d_vol_pct — realized 7-day vol as fraction

    Three signals combine:
      1. Leverage buildup: high OI + positive sustained funding
      2. Volatility compression: recent < historical (primed for breakout)
      3. (Time since last cascade — handled at executor level, not here)
    """
    # OI signal: linear ramp from 70th to 95th percentile.
    # Below 70 → no signal; at 95+ → max signal.
    oi_signal = max(0.0, min(1.0, (oi_percentile - 70.0) / 25.0))

    # Funding signal: linear ramp. 0 → no signal, 0.001 (0.1%/8h sustained) → max.
    funding_signal = max(0.0, min(1.0, funding_rate_24h_avg / 0.001))

    # Compression signal: recent_vol / historical_vol < 1 means compressed.
    # ratio 1.0 → no signal (vol is at its normal level)
    # ratio 0.3 → max signal (vol is 70% below normal — very compressed)
    if historical_7d_vol_pct > 0:
        ratio = recent_24h_vol_pct / historical_7d_vol_pct
        compression_signal = max(0.0, min(1.0, (1.0 - ratio) / 0.7))
    else:
        compression_signal = 0.0

    # Weighted composite: leverage buildup is the dominant signal.
    return 0.4 * oi_signal + 0.4 * funding_signal + 0.2 * compression_signal


# ── Ladder design ──────────────────────────────────────────────────


@dataclass(frozen=True)
class LadderRung:
    pct_below: float        # e.g. 0.02 = 2% below entry-time spot
    limit_price: float      # absolute price for the limit order
    size_usd: float         # USD notional for this rung


@dataclass(frozen=True)
class Ladder:
    spot_at_design: float   # spot price when ladder was designed
    rungs: tuple[LadderRung, ...]
    total_usd: float        # sum of rung size_usd's


def design_ladder(
    spot: float,
    levels: int = 4,
    total_usd: float = 200.0,
    deepest_pct: float = 0.04,
    size_decay: float = 1.5,
) -> Ladder:
    """Design a ladder of buy-limit orders below spot.

    Rung prices: linearly spaced from -1/levels of deepest_pct to -deepest_pct.
        e.g. with levels=4, deepest_pct=0.04: rungs at -1%, -2%, -3%, -4%

    Rung sizes: weighted by size_decay^i. Deeper rungs get MORE size because
    they're more selective (fill only on deeper cascades, which mean-revert
    most reliably).

    Args:
        spot: current price; ladder is anchored here
        levels: number of rungs
        total_usd: total notional spread across all rungs
        deepest_pct: depth of deepest rung as fraction (0.04 = 4% below spot)
        size_decay: weight ratio between successive deeper rungs (>1 = deeper rungs bigger)
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if levels <= 0:
        raise ValueError(f"levels must be positive, got {levels}")
    if total_usd <= 0:
        raise ValueError(f"total_usd must be positive, got {total_usd}")
    if not 0 < deepest_pct < 0.5:
        raise ValueError(f"deepest_pct must be in (0, 0.5), got {deepest_pct}")
    if size_decay <= 0:
        raise ValueError(f"size_decay must be positive, got {size_decay}")

    # Rung depths: linearly spaced
    pcts = [deepest_pct * (i + 1) / levels for i in range(levels)]
    # Rung weights: geometric decay
    weights = [size_decay ** i for i in range(levels)]
    weight_total = sum(weights)

    rungs = tuple(
        LadderRung(
            pct_below=pcts[i],
            limit_price=spot * (1.0 - pcts[i]),
            size_usd=total_usd * weights[i] / weight_total,
        )
        for i in range(levels)
    )
    return Ladder(spot_at_design=spot, rungs=rungs, total_usd=total_usd)


# ── TP / SL math ───────────────────────────────────────────────────


def take_profit_price(fill_price: float, tp_pct: float) -> float:
    """Price at which to sell to lock in tp_pct gain on a long position."""
    if fill_price <= 0:
        raise ValueError(f"fill_price must be positive, got {fill_price}")
    if tp_pct <= 0:
        raise ValueError(f"tp_pct must be positive, got {tp_pct}")
    return fill_price * (1.0 + tp_pct)


def stop_loss_price(fill_price: float, sl_pct: float) -> float:
    """Price at which to cut losses if the cascade deepens further."""
    if fill_price <= 0:
        raise ValueError(f"fill_price must be positive, got {fill_price}")
    if not 0 < sl_pct < 1:
        raise ValueError(f"sl_pct must be in (0, 1), got {sl_pct}")
    return fill_price * (1.0 - sl_pct)


# ── Refresh decisions ──────────────────────────────────────────────


def should_refresh_ladder(
    spot_now: float,
    spot_at_design: float,
    drift_threshold_pct: float = 0.01,
) -> bool:
    """If the underlying has drifted enough that our rung prices are no
    longer at the intended depths, refresh the ladder. e.g. ladder designed
    at $100 with rungs at $99, $98, $97, $96. If spot has moved to $105,
    those rungs are now at -5.7%, -6.7%, -7.6%, -8.6% — too deep, won't fill.
    """
    if spot_at_design <= 0:
        return True
    drift_pct = abs(spot_now - spot_at_design) / spot_at_design
    return drift_pct > drift_threshold_pct
