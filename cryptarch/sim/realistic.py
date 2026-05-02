"""Realistic dry-run simulator.

Lesson from polybot: a maker-fill simulation that always "fills" at the
limit price produces fictional PnL. Real maker orders only fill if a
counter-party crosses to your price. We model this faithfully here:

  - For LIMIT orders: fill iff the live order book has a crossing
    counter-party within `tolerance` of our limit. Otherwise the order
    sits unfilled (we record it as `pending`).
  - For MARKET orders: fill at the actual best ask/bid plus a small
    slippage estimate based on size vs depth.

This module is provider-agnostic — it consumes an `OrderBookSnapshot`
and produces a `SimulatedFill` decision. The exchange client supplies
the snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OrderBookLevel:
    """One price level in the order book (price, size in base units)."""
    price: float
    size: float


@dataclass(frozen=True)
class OrderBookSnapshot:
    """Aggregated order book for one symbol at one moment in time."""
    bids: tuple[OrderBookLevel, ...]    # sorted high→low
    asks: tuple[OrderBookLevel, ...]    # sorted low→high

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


@dataclass(frozen=True)
class SimulatedFill:
    """The result of attempting to fill an order against an order book.

    `filled=True` means the simulator believes a real order would have
    filled at the given `avg_fill_price` and `filled_size_usd`.
    `filled=False` is the realistic outcome for many maker orders —
    the order sits, doesn't fill, and we record it as `pending`. Reason
    is one of: 'maker_unfillable', 'no_book', 'partial_fillable_below_threshold'.
    """
    filled: bool
    avg_fill_price: float | None = None
    filled_size_usd: float = 0.0
    reason: str = ""


def simulate_market_buy(
    book: OrderBookSnapshot,
    size_usd: float,
    max_slippage_pct: float = 0.005,
) -> SimulatedFill:
    """Walk the ask side of the book to fill `size_usd`. Reject if the
    average fill price exceeds best_ask × (1 + max_slippage_pct).
    """
    if book.best_ask is None:
        return SimulatedFill(filled=False, reason="no_book")

    remaining_usd = size_usd
    cost = 0.0
    filled_base = 0.0
    for level in book.asks:
        level_capacity_usd = level.price * level.size
        take_usd = min(remaining_usd, level_capacity_usd)
        take_base = take_usd / level.price
        cost += take_usd
        filled_base += take_base
        remaining_usd -= take_usd
        if remaining_usd <= 1e-9:
            break

    if remaining_usd > 1e-9:
        # Not enough depth — partial fill. Prefer to reject.
        return SimulatedFill(
            filled=False,
            reason=f"insufficient_depth (filled {size_usd - remaining_usd:.2f} of {size_usd:.2f})",
        )

    avg_price = cost / filled_base if filled_base > 0 else 0.0
    if avg_price > book.best_ask * (1 + max_slippage_pct):
        return SimulatedFill(
            filled=False,
            reason=f"slippage_too_high (avg {avg_price:.4f} vs best_ask {book.best_ask:.4f})",
        )
    return SimulatedFill(
        filled=True, avg_fill_price=avg_price, filled_size_usd=size_usd, reason="market_filled"
    )


def simulate_market_sell(
    book: OrderBookSnapshot,
    size_usd: float,
    max_slippage_pct: float = 0.005,
) -> SimulatedFill:
    """Walk the bid side. Reject if avg sell price < best_bid × (1 - slippage)."""
    if book.best_bid is None:
        return SimulatedFill(filled=False, reason="no_book")

    remaining_usd = size_usd
    proceeds = 0.0
    filled_base = 0.0
    for level in book.bids:
        level_capacity_usd = level.price * level.size
        take_usd = min(remaining_usd, level_capacity_usd)
        take_base = take_usd / level.price
        proceeds += take_usd
        filled_base += take_base
        remaining_usd -= take_usd
        if remaining_usd <= 1e-9:
            break

    if remaining_usd > 1e-9:
        return SimulatedFill(
            filled=False,
            reason=f"insufficient_depth (filled {size_usd - remaining_usd:.2f} of {size_usd:.2f})",
        )

    avg_price = proceeds / filled_base if filled_base > 0 else 0.0
    if avg_price < book.best_bid * (1 - max_slippage_pct):
        return SimulatedFill(
            filled=False,
            reason=f"slippage_too_high (avg {avg_price:.4f} vs best_bid {book.best_bid:.4f})",
        )
    return SimulatedFill(
        filled=True, avg_fill_price=avg_price, filled_size_usd=size_usd, reason="market_filled"
    )


def simulate_limit_maker(
    book: OrderBookSnapshot,
    side: Literal["buy", "sell"],
    limit_price: float,
    size_usd: float,
    fill_tolerance_pct: float = 0.0005,
) -> SimulatedFill:
    """A LIMIT maker order. We treat it as fillable iff the book has a
    counter-party within `fill_tolerance_pct` of our limit AT THE TIME
    we evaluate. This is a snapshot model — it captures whether a fill
    is realistic right now.

    For BUY: we fill iff best_ask <= limit_price × (1 + fill_tolerance_pct).
    For SELL: we fill iff best_bid >= limit_price × (1 - fill_tolerance_pct).

    This is the polybot v12.5 lesson: don't pretend orders fill when no
    real seller is at our price.

    Note: a more sophisticated simulator would model time-to-fill (the
    order sits, market moves toward our price, then fills). For now we
    use the simple snapshot model — orders that aren't fillable right
    now are simply rejected. The caller can re-evaluate next cycle.
    """
    if side == "buy":
        if book.best_ask is None:
            return SimulatedFill(filled=False, reason="no_book")
        threshold = limit_price * (1 + fill_tolerance_pct)
        if book.best_ask > threshold:
            return SimulatedFill(
                filled=False,
                reason=f"maker_unfillable (best_ask {book.best_ask:.6f} > limit {limit_price:.6f} + tolerance)",
            )
        # Fill at our limit (we became the new best bid; market crossed us)
        return SimulatedFill(
            filled=True, avg_fill_price=limit_price, filled_size_usd=size_usd,
            reason="maker_filled_at_limit",
        )
    else:    # sell
        if book.best_bid is None:
            return SimulatedFill(filled=False, reason="no_book")
        threshold = limit_price * (1 - fill_tolerance_pct)
        if book.best_bid < threshold:
            return SimulatedFill(
                filled=False,
                reason=f"maker_unfillable (best_bid {book.best_bid:.6f} < limit {limit_price:.6f} - tolerance)",
            )
        return SimulatedFill(
            filled=True, avg_fill_price=limit_price, filled_size_usd=size_usd,
            reason="maker_filled_at_limit",
        )
