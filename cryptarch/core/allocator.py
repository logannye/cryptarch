"""Dynamic capital allocator.

Mathematical premise
--------------------
With three near-zero-correlated edges, the optimal portfolio weight is
roughly proportional to the conditional Sharpe ratio of each (Markowitz /
Kelly). But the relevant signal isn't *recent realized Sharpe* — it's
*currently-available opportunity*. Funding-rate APRs, cascade probabilities,
and IV compression each have observable signals that predict near-term P&L
better than backward-looking Sharpe alone.

This allocator answers: "given what each layer can see RIGHT NOW, how
should we tilt capital from the static 60/25/15 baseline?"

Per-layer signals
-----------------
L1 (funding arb): max APR currently available across the 30-pair scan.
   Score = clamp(max_apr / 50%, 0, 1). When funding pays 50%+ APR
   somewhere in our universe, that's a hot regime — over-allocate.

L2 (cascade): max cascade_probability across symbols in the L2 universe.
   Already 0-1 by construction. When some symbol scores > 0.7, leverage
   has piled up and a cascade is imminent — over-allocate to be ready
   with bigger ladders.

L3 (tail hedge): IV compression score = 1 - (current_IV / historical_IV_baseline).
   When IV is unusually low, OTM strangles are CHEAP — disproportionately
   asymmetric to load up. When IV is high, theta cost eats budget — pull back.

Tilt magnitude is bounded so we never starve any layer. Default ±20pp from
baseline allocation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSignals:
    """Snapshot of forward-looking opportunity per layer."""
    l1_max_apr_pct: float           # best funding APR currently visible (e.g. 25.0 = 25%)
    l2_max_cascade_score: float     # 0-1, highest cascade probability across symbols
    l3_iv_compression_score: float  # 0-1, where 1 = IV at multi-month low


@dataclass(frozen=True)
class AllocationDecision:
    """Per-layer fraction of bankroll. Always sums to 1.0."""
    l1_pct: float
    l2_pct: float
    l3_pct: float
    rationale: str

    @property
    def as_dict(self) -> dict[str, float]:
        return {"l1_funding": self.l1_pct, "l2_cascade": self.l2_pct, "l3_tail": self.l3_pct}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _l1_signal_score(max_apr_pct: float) -> float:
    """Map L1's max APR to a 0-1 attractiveness score.

    50% APR is the point at which we say "this is a hot regime worth
    over-allocating to." Linear ramp 0% → 50% → cap at 1.0.
    """
    return _clamp(max_apr_pct / 50.0, 0.0, 1.0)


def _l2_signal_score(max_cascade_score: float) -> float:
    """L2 score is already 0-1 from cascade_probability — pass through."""
    return _clamp(max_cascade_score, 0.0, 1.0)


def _l3_signal_score(iv_compression: float) -> float:
    """L3 score is the IV-compression score. Pass through but threshold at 0.3:
    below that level the tail hedge isn't notably cheaper than baseline."""
    if iv_compression < 0.3:
        return 0.0
    return _clamp((iv_compression - 0.3) / 0.7, 0.0, 1.0)


def compute_target_allocation(
    signals: LayerSignals,
    base_l1: float = 0.60,
    base_l2: float = 0.25,
    base_l3: float = 0.15,
    max_tilt_pp: float = 0.20,
) -> AllocationDecision:
    """Compute target allocation given current signals.

    Each layer's score (0-1) tilts its allocation up by `score × max_tilt_pp`.
    The sum of all tilts comes from the OTHER two layers proportionally,
    so we always sum to 1.0.

    Args:
        signals: per-layer opportunity scores
        base_l1/l2/l3: static baseline (must sum to 1.0)
        max_tilt_pp: max ± shift in allocation per layer (0.20 = ±20pp)
    """
    if abs(base_l1 + base_l2 + base_l3 - 1.0) > 1e-6:
        raise ValueError(
            f"base allocations must sum to 1.0; got "
            f"{base_l1} + {base_l2} + {base_l3} = {base_l1 + base_l2 + base_l3}",
        )
    if not 0 <= max_tilt_pp <= 0.5:
        raise ValueError(f"max_tilt_pp must be in [0, 0.5], got {max_tilt_pp}")

    s1 = _l1_signal_score(signals.l1_max_apr_pct)
    s2 = _l2_signal_score(signals.l2_max_cascade_score)
    s3 = _l3_signal_score(signals.l3_iv_compression_score)

    # First pass: each layer's tilt-up amount.
    tilt_l1 = s1 * max_tilt_pp
    tilt_l2 = s2 * max_tilt_pp
    tilt_l3 = s3 * max_tilt_pp
    total_tilt_up = tilt_l1 + tilt_l2 + tilt_l3

    # The tilt-up must be funded from the layers WITHOUT a tilt-up signal,
    # proportionally. If all three have tilt signals, no funding pool — the
    # tilt scales down proportionally to keep the sum at 1.0.
    if total_tilt_up == 0.0:
        return AllocationDecision(
            l1_pct=base_l1, l2_pct=base_l2, l3_pct=base_l3,
            rationale="no_signal_baseline",
        )

    # Funding pool: sum of (base_i × non-tilted-fraction) for each layer.
    # If a layer has signal s_i, its "non-tilted fraction" is (1 - s_i).
    # We pull funding proportionally to base × (1 - s_i).
    fund_weight_l1 = base_l1 * (1.0 - s1)
    fund_weight_l2 = base_l2 * (1.0 - s2)
    fund_weight_l3 = base_l3 * (1.0 - s3)
    fund_total = fund_weight_l1 + fund_weight_l2 + fund_weight_l3

    if fund_total <= 0:
        # All layers maxed out — equal cap, no shift.
        return AllocationDecision(
            l1_pct=base_l1, l2_pct=base_l2, l3_pct=base_l3,
            rationale="all_layers_maxed",
        )

    # Funding pull per layer proportional to (1 - s_i). Sum of pulls = total_tilt_up.
    pull_l1 = total_tilt_up * fund_weight_l1 / fund_total
    pull_l2 = total_tilt_up * fund_weight_l2 / fund_total
    pull_l3 = total_tilt_up * fund_weight_l3 / fund_total

    target_l1 = base_l1 + tilt_l1 - pull_l1
    target_l2 = base_l2 + tilt_l2 - pull_l2
    target_l3 = base_l3 + tilt_l3 - pull_l3

    # Floor: don't let any layer go below 50% of baseline (preserves diversification).
    target_l1 = max(target_l1, base_l1 * 0.5)
    target_l2 = max(target_l2, base_l2 * 0.5)
    target_l3 = max(target_l3, base_l3 * 0.5)

    # Renormalize to sum to 1.0.
    total = target_l1 + target_l2 + target_l3
    target_l1 /= total
    target_l2 /= total
    target_l3 /= total

    # Rationale string for logging
    parts = []
    if s1 > 0.1:
        parts.append(f"L1↑(apr={signals.l1_max_apr_pct:.0f}%)")
    if s2 > 0.1:
        parts.append(f"L2↑(cascade={signals.l2_max_cascade_score:.2f})")
    if s3 > 0.1:
        parts.append(f"L3↑(iv_compression={signals.l3_iv_compression_score:.2f})")
    rationale = ",".join(parts) if parts else "near_baseline"

    return AllocationDecision(
        l1_pct=target_l1, l2_pct=target_l2, l3_pct=target_l3,
        rationale=rationale,
    )
