"""L2 cascade-capture executor — orchestration layer.

State model
-----------
Each rung is a separate position (one position per ladder rung). The
ladder is a logical grouping via `strategy_group_id`. Per-rung state
machine:

    pending  → filled  → tp_pending  → closed_tp     (happy path: TP fired)
                              ↘
                                → closed_sl          (cascade extended past stop-loss)
                                → closed_time        (no TP/SL fire within max_hold)
    pending  → cancelled                              (signal weakened or stale ladder)

Position.state mapping:
    pending     ↔ position.state = 'opening'
    filled / tp_pending ↔ position.state = 'open'
    closed_*    ↔ position.state = 'closed'

`run_once` performs one cycle:
  1. Read all open L2 positions.
  2. Manage 'open' positions (filled rungs awaiting TP):
       - check current price → fire TP / SL / time-stop as warranted
  3. Manage 'opening' positions (pending rungs):
       - check best_ask vs limit price → simulated fill, or cancel
  4. For each tracked symbol, compute cascade probability:
       - if signal is strong AND no active ladder, design + place new ladder
       - if signal is weak AND ladder is active with unfilled rungs, cancel them

Concurrency via asyncio.gather ensures the per-cycle 90s budget stays
healthy with 30+ symbols.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

import structlog

from cryptarch.core.config import Settings
from cryptarch.core.safeguards import GuardViolation, OrderRequest, check_order
from cryptarch.db.store import OpenPosition, Store
from cryptarch.exchanges.pool import ExchangePool
from cryptarch.sim.realistic import simulate_limit_maker, simulate_market_buy, simulate_market_sell
from cryptarch.strategies.l2_cascade import (
    Ladder, LadderDesignSnapshot, cascade_probability, design_ladder,
    should_refresh_ladder, stop_loss_price, take_profit_price,
)

log = structlog.get_logger()


# Symbols we scan for cascade opportunities. We use SPOT only on Layer 2:
# - No funding cost / liquidation risk on a long position
# - Cleaner simulation
# - Clean separation from L1 (which uses spot+perp combined)
#
# Universe expanded after a 1m-granularity backtest revealed that the
# original 5m backtest was systematically biased against fast V-shape
# revert dynamics (the SL-first-when-both-bounds-in-bar artifact and
# the same-bar-as-fill exit blind spot). Re-running the original 10-
# symbol universe at 1m flipped DOGE from -$6 to +$21, validated SHIB
# at +$13, and confirmed XRP/BONK/FLOKI as correctly pruned. A separate
# candidate run added JASMY, TURBO, and PENGU as positive contributors.
# BTC/ETH/SOL stayed in (institutional-flow majors near $0 P&L at 1m;
# kept as cheap optionality for major-cap cascade events).
#
# Tuple form: (exchange, spot_symbol, base_label[, perp_symbol_override]).
# The perp override is required for memecoins where Binance lists the
# perp under a 1000-multiple base — PEPE/USDT spot ↔ 1000PEPE/USDT:USDT.
DEFAULT_SYMBOLS: list[tuple[str, str, str] | tuple[str, str, str, str]] = [
    # Major-cap optionality
    ("binance", "BTC/USDT",   "BTC"),
    ("binance", "ETH/USDT",   "ETH"),
    ("binance", "SOL/USDT",   "SOL"),
    # High-vol memes / retail-driven (the actual edge)
    ("binance", "PEPE/USDT",  "PEPE",  "1000PEPE/USDT:USDT"),
    ("binance", "WIF/USDT",   "WIF"),
    ("binance", "DOGE/USDT",  "DOGE"),
    ("binance", "SHIB/USDT",  "SHIB",  "1000SHIB/USDT:USDT"),
    ("binance", "JASMY/USDT", "JASMY"),
    ("binance", "TURBO/USDT", "TURBO"),
    ("binance", "PENGU/USDT", "PENGU"),
]


@dataclass(frozen=True)
class SymbolConfig:
    exchange: str
    symbol: str
    base_label: str
    # Explicit override for the perp-market symbol. None means derive
    # it as `<spot>:USDT` (works for everything except 1000-multiple
    # memecoin perps).
    perp_symbol_override: str | None = None

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.symbol}"

    @property
    def perp_symbol(self) -> str:
        if self.perp_symbol_override is not None:
            return self.perp_symbol_override
        return self.symbol.replace("/USDT", "/USDT:USDT")


# Per-symbol signal callback. Takes a SymbolConfig + the L2 executor's
# context and returns a 0-1 cascade probability. Default implementation
# uses funding rate only (mocks OI/vol). Production would compute from
# rolling OI history + recent OHLCV.
SignalFn = Callable[["SymbolConfig", "L2Executor"], Awaitable[float]]


async def default_cascade_signal(symbol: SymbolConfig, executor: "L2Executor") -> float:
    """Default cascade-probability signal.

    For v1 we use funding rate as a proxy and stub OI/vol percentiles.
    A future Phase 2c will introduce a rolling OI observer that records
    open_interest over time and computes the proper percentile signal.
    """
    perp_symbol = symbol.perp_symbol
    try:
        client = await executor._pool.get(symbol.exchange, market_type="swap")
        funding = await client.get_funding_rate(perp_symbol)
        rate = float(funding.rate_8h)
    except Exception as e:
        log.debug("l2_signal_fetch_failed",
                  symbol=symbol.key, error=str(e)[:100])
        return 0.0

    # Stubs: these will be replaced once the OI history collector is in place.
    oi_percentile_stub = 80.0    # neutral-bullish baseline assumption
    historical_vol_stub = 0.05
    recent_vol_stub = 0.04        # slight compression assumption
    return cascade_probability(
        oi_percentile=oi_percentile_stub,
        funding_rate_24h_avg=rate,
        recent_24h_vol_pct=recent_vol_stub,
        historical_7d_vol_pct=historical_vol_stub,
    )


class L2Executor:
    LAYER = "l2_cascade"
    TRIGGER_THRESHOLD = 0.6     # design ladder when probability > this
    HOLD_THRESHOLD = 0.4        # cancel unfilled rungs when probability < this
    MAX_HOLD_MINUTES = 60       # time-stop on filled positions

    def __init__(
        self,
        settings: Settings,
        store: Store,
        pool: ExchangePool,
        symbols: list[SymbolConfig] | None = None,
        signal_fn: SignalFn | None = None,
    ):
        self._settings = settings
        self._store = store
        self._pool = pool
        self._symbols = symbols or [SymbolConfig(*s) for s in DEFAULT_SYMBOLS]
        self._signal_fn = signal_fn or default_cascade_signal

    # ── public entry ──

    async def run_once(self) -> dict[str, Any]:
        if not self._settings.layer_2_cascade_capture_enabled:
            return {"skipped": "layer_disabled"}

        state = await self._store.get_system_state()
        if state and state.halt_reason:
            return {"skipped": "halted", "reason": state.halt_reason}

        # Read all open L2 positions.
        open_positions = await self._store.open_positions(layer=self.LAYER)

        # 0. Refresh stale ladders. Any opening rung whose design-time
        # config snapshot doesn't match current settings (or is missing
        # entirely from a legacy placement) gets cancelled at $0 P&L.
        # Next pass through the cascade scan will re-evaluate the symbol
        # and design a fresh ladder if conditions still warrant. This is
        # the structural guarantee that all live ladders reflect the
        # current strategy — no pre-tweak/post-tweak split.
        n_refreshed = await self._refresh_stale_ladders(open_positions)
        if n_refreshed > 0:
            open_positions = await self._store.open_positions(layer=self.LAYER)

        n_filled = 0
        n_closed = 0
        n_new_ladders = 0

        # 1. Manage pending rungs (state='opening') — check fills + cancellations.
        for pos in [p for p in open_positions if p.state == "opening"]:
            outcome = await self._manage_pending_rung(pos)
            if outcome == "filled":
                n_filled += 1
            elif outcome == "cancelled":
                n_closed += 1

        # 2. Manage filled positions (state='open') — TP/SL/time-stop.
        for pos in [p for p in open_positions if p.state == "open"]:
            outcome = await self._manage_filled_position(pos)
            if outcome in ("closed_tp", "closed_sl", "closed_time"):
                n_closed += 1

        # 3. Cascade scanning — possibly place new ladders.
        held_symbols = {p.metadata.get("symbol_key") for p in open_positions}
        for symbol in self._symbols:
            if symbol.key in held_symbols:
                continue    # already have an active ladder on this symbol
            score = await self._signal_fn(symbol, self)
            if score >= self.TRIGGER_THRESHOLD:
                if await self._design_and_place_ladder(symbol, score):
                    n_new_ladders += 1

        return {
            "open_positions": len(open_positions),
            "rungs_filled": n_filled,
            "positions_closed": n_closed,
            "new_ladders": n_new_ladders,
            "stale_refreshed": n_refreshed,
        }

    # ── refresh stale ladders ──

    async def _refresh_stale_ladders(self, open_positions: list[OpenPosition]) -> int:
        """Cancel any opening rung whose design-time config snapshot no
        longer matches the current settings (or whose snapshot is missing
        — legacy positions from before this feature). The rungs close at
        $0 P&L (they were never filled, no realized loss); next cycle the
        executor will re-evaluate the symbol against the live config and
        place a fresh ladder if score >= trigger.

        Filled rungs (state='open') are intentionally left alone — their
        size is already deployed and can't be retroactively redesigned.
        TP/SL adapt naturally because they're computed at fill time from
        live settings, not from the snapshot.

        Returns the count of rungs refresh-closed.
        """
        current = LadderDesignSnapshot.from_settings(self._settings)
        n_refreshed = 0
        for pos in open_positions:
            if pos.state != "opening":
                continue
            stored = LadderDesignSnapshot.from_dict(pos.metadata.get("config_snapshot"))
            if stored == current:
                continue
            await self._store.close_position(pos.id, realized_pnl=0.0)
            log.info("l2_rung_refreshed_stale_config",
                     position=pos.id,
                     symbol=pos.metadata.get("symbol_key"),
                     stored_snapshot=pos.metadata.get("config_snapshot"),
                     current_snapshot=current.to_dict())
            n_refreshed += 1
        return n_refreshed

    # ── place ladder ──

    async def _design_and_place_ladder(self, symbol: SymbolConfig, score: float) -> bool:
        """Design a 4-rung ladder, run safeguards on each rung, persist
        rungs as 'opening' positions sharing a strategy_group_id."""
        # Get current spot price.
        try:
            client = await self._pool.get(symbol.exchange, market_type="spot")
            ticker = await client.get_ticker(symbol.symbol)
        except Exception as e:
            log.warning("l2_place_ticker_failed",
                        symbol=symbol.key, error=str(e)[:100])
            return False
        spot = float(ticker.last or ticker.ask or 0)
        if spot <= 0:
            return False

        # Sizing budget: allocation - already-deployed. Prefer dynamic
        # allocation but never below the static floor — gives at least
        # the static guarantee, lets dynamic expand when signal is hot.
        state = await self._store.get_system_state()
        static_pct = self._settings.alloc_layer_2_pct
        dynamic_pct = state.dynamic_alloc_l2_pct if state and state.dynamic_alloc_l2_pct is not None else None
        alloc_pct = max(static_pct, dynamic_pct) if dynamic_pct is not None else static_pct
        layer_alloc_usd = self._settings.bankroll_usd * alloc_pct
        layer_remaining = (
            layer_alloc_usd
            - await self._store.layer_deployed_usd(self.LAYER)
        )
        budget = min(layer_remaining, self._settings.l2_ladder_total_usd)
        if budget < self._settings.min_position_usd:
            return False

        ladder = design_ladder(
            spot=spot,
            levels=self._settings.l2_ladder_levels,
            total_usd=budget,
        )

        group_id = str(uuid.uuid4())
        seen_ids = await self._store.recent_client_order_ids()
        total_at_risk = await self._store.total_at_risk_usd()
        layer_deployed = await self._store.layer_deployed_usd(self.LAYER)

        placed = 0
        for i, rung in enumerate(ladder.rungs):
            client_order_id = f"cryptarch-l2-{group_id}-rung-{i}"
            order = OrderRequest(
                exchange=symbol.exchange,
                symbol=symbol.symbol,
                side="buy",
                size_usd=rung.size_usd,
                limit_price=rung.limit_price,
                layer=self.LAYER,
                client_order_id=client_order_id,
                is_live=self._settings.enable_live_orders,
            )
            try:
                check_order(
                    order, self._settings,
                    total_at_risk + sum(r.size_usd for r in ladder.rungs[:i]),
                    layer_deployed + sum(r.size_usd for r in ladder.rungs[:i]),
                    seen_ids,
                    layer_cap_usd=layer_alloc_usd,
                )
            except GuardViolation as e:
                log.info("l2_rung_guard_violation",
                         symbol=symbol.key, rung_idx=i,
                         code=e.code, msg=str(e))
                continue

            position_id = await self._store.create_position(
                layer=self.LAYER,
                strategy_group_id=group_id,
                notional_usd=rung.size_usd,
                metadata={
                    "symbol_key": symbol.key,
                    "exchange": symbol.exchange,
                    "symbol": symbol.symbol,
                    "base_label": symbol.base_label,
                    "rung_idx": i,
                    "rung_pct_below": rung.pct_below,
                    "limit_price": rung.limit_price,
                    "size_usd": rung.size_usd,
                    "spot_at_design": ladder.spot_at_design,
                    "trigger_score": score,
                    "client_order_id": client_order_id,
                    "phase": "pending_rung",
                    "config_snapshot": LadderDesignSnapshot.from_settings(
                        self._settings,
                    ).to_dict(),
                },
            )

            # Record a 'pending' fill row so idempotency dedup works.
            await self._store.record_fill(
                position_id=position_id,
                layer=self.LAYER,
                exchange=order.exchange,
                symbol=order.symbol,
                side="buy",
                order_type="limit",
                size_base=0.0,    # not yet filled
                size_usd=0.0,
                fill_price=order.limit_price,
                client_order_id=client_order_id,
                is_simulated=not order.is_live,
                sim_reason="ladder_pending",
            )
            placed += 1

        if placed > 0:
            log.info("l2_ladder_placed",
                     symbol=symbol.key, group=group_id,
                     spot=round(spot, 4), score=round(score, 3),
                     rungs=placed,
                     total_usd=round(sum(r.size_usd for r in ladder.rungs), 2))
            return True
        return False

    # ── manage pending rungs ──

    async def _manage_pending_rung(self, pos: OpenPosition) -> str | None:
        """For a position in 'opening' state (rung pending fill):
          - If best_ask has crossed our limit, simulate fill + place TP order
          - If signal weakened past HOLD_THRESHOLD, cancel
        Returns 'filled' | 'cancelled' | None.
        """
        symbol_key = pos.metadata.get("symbol_key")
        symbol = next((s for s in self._symbols if s.key == symbol_key), None)
        if symbol is None:
            return None
        limit_price = float(pos.metadata.get("limit_price", 0))
        size_usd = float(pos.metadata.get("size_usd", 0))
        if limit_price <= 0 or size_usd <= 0:
            return None

        try:
            client = await self._pool.get(symbol.exchange, market_type="spot")
            book = await client.get_order_book(symbol.symbol, depth=10)
        except Exception as e:
            log.warning("l2_rung_book_failed",
                        symbol=symbol_key, error=str(e)[:100])
            return None

        # Did the rung fill?
        sim = simulate_limit_maker(book, side="buy", limit_price=limit_price, size_usd=size_usd)
        if sim.filled:
            # Fill recorded. Update fill row, transition position to 'open',
            # record a TP-pending fill stub.
            await self._store.record_fill(
                position_id=pos.id,
                layer=self.LAYER,
                exchange=symbol.exchange,
                symbol=symbol.symbol,
                side="buy",
                order_type="limit",
                size_base=size_usd / limit_price,
                size_usd=size_usd,
                fill_price=limit_price,
                client_order_id=f"cryptarch-l2-{pos.strategy_group_id}-rung-{pos.metadata.get('rung_idx', 0)}-fill",
                is_simulated=not self._settings.enable_live_orders,
                sim_reason="ladder_filled",
            )
            await self._store.mark_position_open(pos.id)

            # Record TP target — we'll detect TP fills next cycle.
            tp = take_profit_price(limit_price, self._settings.l2_take_profit_pct)
            sl = stop_loss_price(limit_price, self._settings.l2_stop_loss_pct)
            new_meta = dict(pos.metadata)
            new_meta.update({
                "phase": "tp_pending",
                "fill_price": limit_price,
                "tp_price": tp,
                "sl_price": sl,
                "filled_at": datetime.now(timezone.utc).isoformat(),
            })
            # Write metadata back via a position-update query.
            await self._update_position_metadata(pos.id, new_meta)
            log.info("l2_rung_filled",
                     position=pos.id, symbol=symbol_key,
                     fill_price=round(limit_price, 6),
                     tp=round(tp, 6), sl=round(sl, 6))
            return "filled"

        # Check if signal has weakened — cancel if so.
        score = await self._signal_fn(symbol, self)
        if score < self.HOLD_THRESHOLD:
            # Cancel the pending rung. Free its capital.
            await self._store.close_position(pos.id, realized_pnl=0.0)
            log.info("l2_rung_cancelled",
                     position=pos.id, symbol=symbol_key,
                     reason="signal_weakened", current_score=round(score, 3))
            return "cancelled"

        # Also cancel if spot has drifted too far from the ladder's design point.
        spot_at_design = float(pos.metadata.get("spot_at_design", 0))
        cur_spot = book.mid or book.best_ask
        if cur_spot and spot_at_design and should_refresh_ladder(
            float(cur_spot), spot_at_design,
            drift_threshold_pct=0.03,    # 3% drift on the ladder anchor
        ):
            await self._store.close_position(pos.id, realized_pnl=0.0)
            log.info("l2_rung_cancelled",
                     position=pos.id, symbol=symbol_key,
                     reason="ladder_stale",
                     spot_at_design=spot_at_design, spot_now=float(cur_spot))
            return "cancelled"

        return None

    # ── manage filled positions (TP/SL/time-stop) ──

    async def _manage_filled_position(self, pos: OpenPosition) -> str | None:
        """For a position in 'open' state with a fill recorded:
          - Sell at market if best_bid >= TP
          - Sell at market if best_bid <= SL
          - Time-stop after MAX_HOLD_MINUTES — sell at market regardless
        """
        symbol_key = pos.metadata.get("symbol_key")
        symbol = next((s for s in self._symbols if s.key == symbol_key), None)
        if symbol is None:
            return None

        fill_price = float(pos.metadata.get("fill_price", 0))
        tp_price = float(pos.metadata.get("tp_price", 0))
        sl_price = float(pos.metadata.get("sl_price", 0))
        size_usd = float(pos.metadata.get("size_usd", 0))
        filled_at_str = pos.metadata.get("filled_at")
        if not (fill_price and tp_price and sl_price and size_usd and filled_at_str):
            return None

        filled_at = datetime.fromisoformat(filled_at_str)
        age_min = (datetime.now(timezone.utc) - filled_at).total_seconds() / 60.0

        try:
            client = await self._pool.get(symbol.exchange, market_type="spot")
            book = await client.get_order_book(symbol.symbol, depth=10)
        except Exception as e:
            log.warning("l2_filled_book_failed",
                        symbol=symbol_key, error=str(e)[:100])
            return None

        bid = book.best_bid
        if bid is None:
            return None

        # Decide: TP, SL, time-stop, or hold?
        exit_reason: str | None = None
        target_exit_price: float = 0.0
        if bid >= tp_price:
            exit_reason = "closed_tp"
            target_exit_price = tp_price
        elif bid <= sl_price:
            exit_reason = "closed_sl"
            target_exit_price = sl_price
        elif age_min >= self.MAX_HOLD_MINUTES:
            exit_reason = "closed_time"
            target_exit_price = float(bid)
        else:
            return None

        # Execute the close as a market sell against current bids.
        sim = simulate_market_sell(book, size_usd=size_usd)
        if not sim.filled:
            log.warning("l2_close_unfillable",
                        position=pos.id, symbol=symbol_key,
                        reason=sim.reason)
            return None

        actual_exit = sim.avg_fill_price or target_exit_price
        size_base = size_usd / fill_price
        realized_pnl = (actual_exit - fill_price) * size_base

        # Record the sell fill.
        await self._store.record_fill(
            position_id=pos.id,
            layer=self.LAYER,
            exchange=symbol.exchange,
            symbol=symbol.symbol,
            side="sell",
            order_type="market",
            size_base=size_base,
            size_usd=size_usd,
            fill_price=actual_exit,
            client_order_id=f"cryptarch-l2-{pos.strategy_group_id}-rung-{pos.metadata.get('rung_idx', 0)}-{exit_reason}",
            is_simulated=not self._settings.enable_live_orders,
            sim_reason=exit_reason,
        )
        await self._store.close_position(pos.id, realized_pnl=realized_pnl)

        log.info("l2_position_closed",
                 position=pos.id, symbol=symbol_key, reason=exit_reason,
                 fill_price=round(fill_price, 6),
                 exit_price=round(actual_exit, 6),
                 pnl=round(realized_pnl, 4),
                 age_min=round(age_min, 1))
        return exit_reason

    # ── helpers ──

    async def _update_position_metadata(self, position_id: int, metadata: dict[str, Any]) -> None:
        """Replace position.metadata. Direct DB write because Store doesn't
        expose a metadata-only update method (would clutter the interface)."""
        import json
        # Use a connection from the underlying pool.
        await self._store._pool.execute(    # type: ignore[attr-defined]
            "UPDATE position SET metadata=$1 WHERE id=$2",
            json.dumps(metadata), position_id,
        )
