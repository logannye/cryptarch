"""Hard safeguards enforced at order-submit time.

These bypass softer checks (config flags) and refuse orders that would
violate any system invariant. The principle: every guard is a pure
function — we can test the math without an exchange connection.

Key invariants (from project memory):
  1. Σ position_notional ≤ bankroll × max_total_deployed_pct
  2. Single-position notional ≤ max_per_position_usd
  3. Layer-specific caps stay within layer's allocation
  4. Order has an idempotency key
  5. Live orders only when enable_live_orders=true
"""
from __future__ import annotations

from dataclasses import dataclass

from cryptarch.core.config import Settings


class GuardViolation(Exception):
    """Raised when an order would violate a safety invariant."""

    def __init__(self, code: str, msg: str):
        self.code = code
        super().__init__(f"[{code}] {msg}")


@dataclass(frozen=True)
class OrderRequest:
    """Pre-submission order intent. Immutable; gets validated by every
    safeguard before being passed to the exchange client."""
    exchange: str           # e.g. "binance"
    symbol: str             # e.g. "BTC/USDT" or "BTC/USDT:USDT" for perps
    side: str               # "buy" | "sell"
    size_usd: float         # notional in USD
    limit_price: float      # quote price
    layer: str              # "l1_funding" | "l2_cascade" | "l3_tail"
    client_order_id: str    # idempotency key (must be unique)
    is_live: bool           # true = real order, false = dry-run record


def check_order(
    order: OrderRequest,
    settings: Settings,
    current_total_at_risk_usd: float,
    layer_already_deployed_usd: float,
    seen_client_order_ids: set[str],
    layer_cap_usd: float | None = None,
) -> None:
    """Run every safeguard on an order. Raises GuardViolation on failure;
    returns None if the order is safe to submit.

    Caller must pass the CURRENT state from the DB:
      - total at-risk across all open positions
      - this layer's currently-deployed notional
      - set of client_order_ids already used (idempotency)
      - optional layer_cap_usd override: lets the caller substitute the
        dynamic-allocator's view of a layer's cap for the static config
        value. The executor sizes orders against dynamic allocation; if
        the safeguard kept enforcing the static cap they could disagree
        and rungs the executor sized within budget would get rejected.
    """
    if order.size_usd <= 0:
        raise GuardViolation("invalid_size", f"size_usd must be positive, got {order.size_usd}")

    if order.limit_price <= 0:
        raise GuardViolation("invalid_price", f"limit_price must be positive, got {order.limit_price}")

    # 1. Single-position cap
    if order.size_usd > settings.max_per_position_usd:
        raise GuardViolation(
            "max_per_position",
            f"size_usd {order.size_usd:.2f} > max_per_position_usd {settings.max_per_position_usd:.2f}",
        )

    # 2. Total deployed cap
    new_total = current_total_at_risk_usd + order.size_usd
    if new_total > settings.max_total_deployed_usd:
        raise GuardViolation(
            "max_total_deployed",
            f"would push total at-risk to {new_total:.2f} > "
            f"{settings.max_total_deployed_usd:.2f} ({settings.max_total_deployed_pct*100:.0f}% of bankroll)",
        )

    # 3. Layer-specific cap. The override is authoritative when supplied;
    # otherwise fall back to the static config value for that layer.
    static_layer_cap = {
        "l1_funding": settings.alloc_layer_1_usd,
        "l2_cascade": settings.alloc_layer_2_usd,
        "l3_tail":    settings.alloc_layer_3_usd,
    }.get(order.layer)
    if static_layer_cap is None:
        raise GuardViolation(
            "unknown_layer",
            f"layer must be one of l1_funding | l2_cascade | l3_tail, got {order.layer!r}",
        )
    layer_cap = layer_cap_usd if layer_cap_usd is not None else static_layer_cap
    new_layer_deployed = layer_already_deployed_usd + order.size_usd
    if new_layer_deployed > layer_cap:
        raise GuardViolation(
            "layer_cap_exceeded",
            f"layer {order.layer} would deploy {new_layer_deployed:.2f} > cap {layer_cap:.2f}",
        )

    # 4. Idempotency
    if not order.client_order_id:
        raise GuardViolation("missing_client_order_id", "every order must have a client_order_id")
    if order.client_order_id in seen_client_order_ids:
        raise GuardViolation(
            "duplicate_client_order_id",
            f"client_order_id {order.client_order_id!r} already submitted",
        )

    # 5. Live-order gate
    if order.is_live and not settings.enable_live_orders:
        raise GuardViolation(
            "live_orders_disabled",
            "is_live=True but settings.enable_live_orders=False — refusing to submit",
        )

    # 6. Layer-enabled gate
    layer_enabled = {
        "l1_funding": settings.layer_1_funding_arb_enabled,
        "l2_cascade": settings.layer_2_cascade_capture_enabled,
        "l3_tail":    settings.layer_3_tail_hedge_enabled,
    }[order.layer]
    if not layer_enabled:
        raise GuardViolation(
            "layer_disabled",
            f"layer {order.layer} is disabled in config",
        )
