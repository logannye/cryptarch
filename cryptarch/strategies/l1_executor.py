"""L1 funding-arb executor — orchestration layer.

`run_once(state)` performs one cycle:
  1. Fetch (spot, perp, funding) tuples for each configured pair (in parallel)
  2. Build FundingArbCandidate for each
  3. Open new positions for attractive candidates we don't already hold
  4. Monitor open positions: harvest funding payments, close if conditions met

Each call is bounded by the engine's per-cycle timeout (90s).
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from cryptarch.core.config import Settings
from cryptarch.core.safeguards import GuardViolation, OrderRequest, check_order
from cryptarch.db.store import OpenPosition, Store
from cryptarch.exchanges.pool import ExchangePool
from cryptarch.sim.realistic import simulate_limit_maker
from cryptarch.strategies.l1_funding import (
    FundingArbCandidate, is_attractive, plan_position, should_close_position,
)

log = structlog.get_logger()


# Pairs we scan. Format: (spot_exchange, spot_symbol, perp_exchange, perp_symbol, base_label).
# Single-exchange pairs first (simpler — same balance, same fees, same margin engine).
# Cross-exchange will need additional Phase 1c work for transfer logistics.
# Curated list: high-volume Binance perp pairs with active funding-rate
# markets. Selection criteria: top-30 USDT-perp volume on Binance, active
# liquidity in both spot and perp, base assets that historically show
# meaningful funding-rate variance (so the bot has opportunities to
# exploit). Memecoins and ultra-thin alts excluded — they're either
# under-leveraged (no funding signal) or have illiquid spot markets.
#
# (spot_exchange, spot_symbol, perp_exchange, perp_symbol, base_label,
#  perp_multiplier). perp_multiplier is 1000 for memecoins where the
# perp contract represents 1000 base tokens (PEPE, SHIB, FLOKI, BONK)
# and 1 for everything else.
DEFAULT_PAIRS: list[tuple[str, str, str, str, str] | tuple[str, str, str, str, str, int]] = [
    # ── Top-cap (always-on liquidity) ──
    ("binance", "BTC/USDT",   "binance", "BTC/USDT:USDT",   "BTC"),
    ("binance", "ETH/USDT",   "binance", "ETH/USDT:USDT",   "ETH"),
    ("binance", "BNB/USDT",   "binance", "BNB/USDT:USDT",   "BNB"),
    ("binance", "SOL/USDT",   "binance", "SOL/USDT:USDT",   "SOL"),
    ("binance", "XRP/USDT",   "binance", "XRP/USDT:USDT",   "XRP"),

    # ── Major L1 / L2 ──
    ("binance", "ADA/USDT",   "binance", "ADA/USDT:USDT",   "ADA"),
    ("binance", "AVAX/USDT",  "binance", "AVAX/USDT:USDT",  "AVAX"),
    ("binance", "DOT/USDT",   "binance", "DOT/USDT:USDT",   "DOT"),
    ("binance", "NEAR/USDT",  "binance", "NEAR/USDT:USDT",  "NEAR"),
    ("binance", "ATOM/USDT",  "binance", "ATOM/USDT:USDT",  "ATOM"),
    ("binance", "ARB/USDT",   "binance", "ARB/USDT:USDT",   "ARB"),
    ("binance", "OP/USDT",    "binance", "OP/USDT:USDT",    "OP"),
    ("binance", "APT/USDT",   "binance", "APT/USDT:USDT",   "APT"),
    ("binance", "SUI/USDT",   "binance", "SUI/USDT:USDT",   "SUI"),
    ("binance", "TIA/USDT",   "binance", "TIA/USDT:USDT",   "TIA"),
    ("binance", "SEI/USDT",   "binance", "SEI/USDT:USDT",   "SEI"),
    ("binance", "INJ/USDT",   "binance", "INJ/USDT:USDT",   "INJ"),
    ("binance", "TON/USDT",   "binance", "TON/USDT:USDT",   "TON"),

    # ── Alt majors ──
    ("binance", "LINK/USDT",  "binance", "LINK/USDT:USDT",  "LINK"),
    ("binance", "LTC/USDT",   "binance", "LTC/USDT:USDT",   "LTC"),
    ("binance", "UNI/USDT",   "binance", "UNI/USDT:USDT",   "UNI"),
    ("binance", "AAVE/USDT",  "binance", "AAVE/USDT:USDT",  "AAVE"),

    # ── Memecoins (highest funding-rate variance — biggest edge) ──
    # When retail piles long into memes, funding can hit 0.1-1.0% per 8h
    # (= 100-1000%+ APR). This is where the asymmetric upside lives.
    # The 1000-prefix is because 1 PEPE/SHIB is worth fractions of a cent;
    # exchanges trade 1000-multiples for tick sizing. Math still holds
    # because basis is in pct, not absolute.
    # Memecoin perps not on Binance (POPCAT, MEW, TURBO, SATS, NEIRO)
    # are reachable via Bybit/OKX integrations — to add in Phase 1c.
    ("binance", "DOGE/USDT",  "binance", "DOGE/USDT:USDT",       "DOGE"),
    ("binance", "PEPE/USDT",  "binance", "1000PEPE/USDT:USDT",   "PEPE",  1000),
    ("binance", "SHIB/USDT",  "binance", "1000SHIB/USDT:USDT",   "SHIB",  1000),
    ("binance", "FLOKI/USDT", "binance", "1000FLOKI/USDT:USDT",  "FLOKI", 1000),
    ("binance", "BONK/USDT",  "binance", "1000BONK/USDT:USDT",   "BONK",  1000),
    ("binance", "WIF/USDT",   "binance", "WIF/USDT:USDT",        "WIF"),
    ("binance", "BOME/USDT",  "binance", "BOME/USDT:USDT",       "BOME"),
    ("binance", "ORDI/USDT",  "binance", "ORDI/USDT:USDT",       "ORDI"),
]


@dataclass(frozen=True)
class PairConfig:
    spot_exchange: str
    spot_symbol: str
    perp_exchange: str
    perp_symbol: str
    base_label: str
    # Multiplier on perp size: e.g. 1000 for "1000PEPE" perps where
    # 1 perp contract represents 1000 PEPE tokens. Required to make
    # perp_price comparable to spot_price (per unit of base token).
    perp_multiplier: int = 1

    @property
    def group_key(self) -> str:
        return f"{self.spot_exchange}:{self.spot_symbol}|{self.perp_exchange}:{self.perp_symbol}"


class L1Executor:
    LAYER = "l1_funding"

    def __init__(
        self, settings: Settings, store: Store, pool: ExchangePool,
        pairs: list[PairConfig] | None = None,
    ):
        self._settings = settings
        self._store = store
        self._pool = pool
        self._pairs = pairs or [
            PairConfig(*(t if len(t) == 6 else (*t, 1)))    # type: ignore
            for t in DEFAULT_PAIRS
        ]

    # ── public entry ──

    async def run_once(self) -> dict[str, Any]:
        """One cycle of: scan → consider new positions → manage open."""
        if not self._settings.layer_1_funding_arb_enabled:
            return {"skipped": "layer_disabled"}

        state = await self._store.get_system_state()
        if state and state.halt_reason:
            return {"skipped": "halted", "reason": state.halt_reason}

        # 1. Snapshot open L1 positions to compute capacity + dedup pairs.
        open_positions = await self._store.open_positions(layer=self.LAYER)
        held_group_keys = {
            p.metadata.get("pair_group_key") for p in open_positions
        }

        # 2. Scan all configured pairs in parallel.
        candidates = await self._scan_candidates()

        # 3. Manage existing positions first (close decisions take priority).
        n_closed = 0
        for pos in open_positions:
            cand = self._candidate_for_position(pos, candidates)
            if cand is None:
                continue
            if should_close_position(
                cand,
                min_funding_8h_to_hold=self._settings.l1_min_funding_rate_8h * 0.33,
                # hold-threshold is 1/3 of entry threshold (avoid flap)
            ):
                if await self._close_position(pos, cand):
                    n_closed += 1

        # 4. Consider new entries.
        n_opened = 0
        for cand in candidates:
            pair_key = self._pair_key_for_candidate(cand)
            if pair_key in held_group_keys:
                continue    # already in this market
            if not is_attractive(
                cand,
                min_funding_8h=self._settings.l1_min_funding_rate_8h,
                min_basis_pct=self._settings.l1_min_basis_pct,
                max_basis_pct=self._settings.l1_max_basis_pct,
            ):
                continue
            if await self._open_position(cand):
                n_opened += 1
                held_group_keys.add(pair_key)

        return {
            "candidates": len(candidates),
            "open_positions": len(open_positions),
            "opened": n_opened,
            "closed": n_closed,
        }

    # ── scanning ──

    async def _scan_candidates(self) -> list[FundingArbCandidate]:
        """Fetch (spot price, perp price, funding) for every configured pair
        in parallel. With 30+ pairs, sequential fetching would blow our
        per-cycle budget; parallel finishes in roughly the slowest single
        request's time."""
        # Pre-resolve clients (idempotent + cached in pool)
        # Then fetch all pairs concurrently via asyncio.gather.
        tasks = [self._fetch_one(pair) for pair in self._pairs]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[FundingArbCandidate] = []
        for pair, outcome in zip(self._pairs, outcomes):
            if isinstance(outcome, Exception):
                log.warning("l1_scan_pair_failed",
                            pair=pair.group_key, error=str(outcome)[:120])
                continue
            if outcome is not None:
                results.append(outcome)
        return results

    async def _fetch_one(self, pair: PairConfig) -> FundingArbCandidate | None:
        spot_client = await self._pool.get(pair.spot_exchange, market_type="spot")
        perp_client = await self._pool.get(pair.perp_exchange, market_type="swap")
        spot_ticker, perp_ticker, funding = await asyncio.gather(
            spot_client.get_ticker(pair.spot_symbol),
            perp_client.get_ticker(pair.perp_symbol),
            perp_client.get_funding_rate(pair.perp_symbol),
        )
        spot_price = (spot_ticker.last or spot_ticker.ask or 0)
        perp_price_raw = (perp_ticker.last or perp_ticker.ask or 0)
        if spot_price <= 0 or perp_price_raw <= 0:
            return None
        # Normalize per-token prices for fair basis comparison. e.g.
        # 1000PEPE perp at $0.004 / 1000 = $0.000004 per PEPE = matches spot.
        perp_price = float(perp_price_raw) / pair.perp_multiplier
        return FundingArbCandidate(
            spot_exchange=pair.spot_exchange,
            spot_symbol=pair.spot_symbol,
            perp_exchange=pair.perp_exchange,
            perp_symbol=pair.perp_symbol,
            spot_price=float(spot_price),
            perp_price=perp_price,
            funding_rate_8h=float(funding.rate_8h),
        )

    def _pair_key_for_candidate(self, c: FundingArbCandidate) -> str:
        return f"{c.spot_exchange}:{c.spot_symbol}|{c.perp_exchange}:{c.perp_symbol}"

    def _candidate_for_position(
        self, pos: OpenPosition, candidates: list[FundingArbCandidate],
    ) -> FundingArbCandidate | None:
        key = pos.metadata.get("pair_group_key")
        if not key:
            return None
        for c in candidates:
            if self._pair_key_for_candidate(c) == key:
                return c
        return None

    # ── open ──

    async def _open_position(self, cand: FundingArbCandidate) -> bool:
        """Build orders for both legs, run safeguards, simulate fills,
        record in DB. Returns True if opened, False if any leg refused."""
        # Sizing: capital allocated to L1 / max position size, whichever smaller.
        # Prefer dynamic allocation but never below the static floor — gives
        # at least the static guarantee, lets dynamic expand when funding is
        # hot. Same value is passed to check_order so the safeguards agree.
        state = await self._store.get_system_state()
        static_pct = self._settings.alloc_layer_1_pct
        dynamic_pct = state.dynamic_alloc_l1_pct if state and state.dynamic_alloc_l1_pct is not None else None
        alloc_pct = max(static_pct, dynamic_pct) if dynamic_pct is not None else static_pct
        layer_alloc_usd = self._settings.bankroll_usd * alloc_pct
        layer_remaining = (
            layer_alloc_usd
            - await self._store.layer_deployed_usd(self.LAYER)
        )
        budget = min(layer_remaining, self._settings.max_per_position_usd)
        if budget < self._settings.min_position_usd:    # bankroll-relative floor; fees eat tiny positions
            return False

        plan = plan_position(cand, total_capital_usd=budget)
        group_id = str(uuid.uuid4())
        pair_key = self._pair_key_for_candidate(cand)

        # Build order requests for both legs.
        spot_order = OrderRequest(
            exchange=cand.spot_exchange,
            symbol=cand.spot_symbol,
            side="buy",
            size_usd=plan.spot_notional_usd,
            limit_price=cand.spot_price,
            layer=self.LAYER,
            client_order_id=f"cryptarch-l1-{group_id}-spot",
            is_live=self._settings.enable_live_orders,
        )
        perp_order = OrderRequest(
            exchange=cand.perp_exchange,
            symbol=cand.perp_symbol,
            side="sell",
            size_usd=plan.perp_notional_usd,
            limit_price=cand.perp_price,
            layer=self.LAYER,
            client_order_id=f"cryptarch-l1-{group_id}-perp",
            is_live=self._settings.enable_live_orders,
        )

        # Pre-flight: safeguards must pass for BOTH legs before we touch either.
        try:
            total_at_risk = await self._store.total_at_risk_usd()
            layer_deployed = await self._store.layer_deployed_usd(self.LAYER)
            seen_ids = await self._store.recent_client_order_ids()
            check_order(spot_order, self._settings, total_at_risk, layer_deployed, seen_ids,
                        layer_cap_usd=layer_alloc_usd)
            # Conceptually the second leg is part of the same total — pass updated counters.
            check_order(
                perp_order, self._settings,
                total_at_risk + spot_order.size_usd,
                layer_deployed + spot_order.size_usd,
                seen_ids | {spot_order.client_order_id},
                layer_cap_usd=layer_alloc_usd,
            )
        except GuardViolation as e:
            log.info("l1_open_guard_violation",
                     pair=pair_key, code=e.code, msg=str(e))
            return False

        # Create position record (state='opening').
        position_id = await self._store.create_position(
            layer=self.LAYER,
            strategy_group_id=group_id,
            notional_usd=plan.total_notional_usd,
            metadata={
                "pair_group_key": pair_key,
                "base_label": next((p.base_label for p in self._pairs
                                    if self._pair_key_for_candidate(cand) == p.group_key), ""),
                "entry_funding_rate_8h": cand.funding_rate_8h,
                "entry_basis_pct": cand.basis_pct,
                "entry_spot_price": cand.spot_price,
                "entry_perp_price": cand.perp_price,
                "expected_apr_pct": cand.expected_apr_pct,
                "spot_size_base": plan.spot_size_base,
                "perp_size_base": plan.perp_size_base,
            },
        )

        # Submit both legs (sim or live).
        spot_filled = await self._submit_leg(
            order=spot_order, position_id=position_id,
            order_type="limit",
            base_size=plan.spot_size_base,
            market_type="spot",
        )
        perp_filled = await self._submit_leg(
            order=perp_order, position_id=position_id,
            order_type="limit",
            base_size=plan.perp_size_base,
            market_type="swap",
        )

        if spot_filled and perp_filled:
            await self._store.mark_position_open(position_id)
            log.info("l1_position_opened",
                     position_id=position_id, pair=pair_key,
                     notional=round(plan.total_notional_usd, 2),
                     funding_8h=round(cand.funding_rate_8h, 6),
                     expected_apr=round(cand.expected_apr_pct * 100, 2))
            return True

        # If only one leg filled, we have a directional position — the
        # delta-neutral assumption breaks. In live mode this is dangerous;
        # in simulation we leave it for human review by marking halted.
        if spot_filled != perp_filled:
            log.error("l1_one_leg_failed",
                      position_id=position_id, pair=pair_key,
                      spot_filled=spot_filled, perp_filled=perp_filled)
            await self._store.set_halt_reason(
                f"L1 partial fill at position {position_id} "
                f"({pair_key}); manual reconciliation required")
        return False

    async def _submit_leg(
        self,
        order: OrderRequest,
        position_id: int,
        order_type: str,
        base_size: float,
        market_type: str,
    ) -> bool:
        """Submit one leg. In dry-run, run the realistic simulator
        against the live order book. In live mode, place via exchange.
        Returns True on fill, False otherwise."""
        client = await self._pool.get(order.exchange, market_type=market_type)
        book = await client.get_order_book(order.symbol, depth=10)

        if not order.is_live:
            sim = simulate_limit_maker(
                book, side=order.side,
                limit_price=order.limit_price, size_usd=order.size_usd,
            )
            await self._store.record_fill(
                position_id=position_id,
                layer=self.LAYER,
                exchange=order.exchange,
                symbol=order.symbol,
                side=order.side,
                order_type=order_type,
                size_base=base_size if sim.filled else 0.0,
                size_usd=order.size_usd if sim.filled else 0.0,
                fill_price=sim.avg_fill_price or order.limit_price,
                client_order_id=order.client_order_id,
                is_simulated=True,
                sim_reason=sim.reason,
                exchange_order_id=None,
            )
            if not sim.filled:
                log.info("l1_leg_unfilled_sim",
                         pair=f"{order.exchange}:{order.symbol}",
                         side=order.side, limit=order.limit_price,
                         best_ask=book.best_ask, best_bid=book.best_bid,
                         reason=sim.reason)
            return sim.filled

        # Live placement
        try:
            exch_order_id = await client.submit_limit_order(
                symbol=order.symbol, side=order.side,
                size_base=base_size, limit_price=order.limit_price,
                post_only=True, client_order_id=order.client_order_id,
            )
            await self._store.record_fill(
                position_id=position_id,
                layer=self.LAYER,
                exchange=order.exchange,
                symbol=order.symbol,
                side=order.side,
                order_type=order_type,
                size_base=base_size,
                size_usd=order.size_usd,
                fill_price=order.limit_price,
                client_order_id=order.client_order_id,
                is_simulated=False,
                sim_reason="live_post_only",
                exchange_order_id=exch_order_id,
            )
            return True
        except Exception as e:
            log.error("l1_live_submit_failed",
                      pair=f"{order.exchange}:{order.symbol}",
                      side=order.side, error=str(e)[:120])
            return False

    # ── close ──

    async def _close_position(
        self, pos: OpenPosition, cand: FundingArbCandidate,
    ) -> bool:
        """Close both legs of a funding-arb position. PnL = funding income
        + small basis-change PnL."""
        # We don't actually compute basis-change PnL here in dry-run; in
        # live mode the actual fills produce real prices. For now we
        # estimate realized PnL as accrued funding.
        pair_key = pos.metadata.get("pair_group_key", "")
        spot_size_base = float(pos.metadata.get("spot_size_base", 0))
        perp_size_base = float(pos.metadata.get("perp_size_base", 0))

        await self._store.mark_position_closing(pos.id)

        # Build close orders (opposite sides of the original).
        spot_close = OrderRequest(
            exchange=cand.spot_exchange, symbol=cand.spot_symbol,
            side="sell", size_usd=pos.notional_usd / 2,
            limit_price=cand.spot_price,
            layer=self.LAYER,
            client_order_id=f"cryptarch-l1-{pos.strategy_group_id}-spot-close",
            is_live=self._settings.enable_live_orders,
        )
        perp_close = OrderRequest(
            exchange=cand.perp_exchange, symbol=cand.perp_symbol,
            side="buy", size_usd=pos.notional_usd / 2,
            limit_price=cand.perp_price,
            layer=self.LAYER,
            client_order_id=f"cryptarch-l1-{pos.strategy_group_id}-perp-close",
            is_live=self._settings.enable_live_orders,
        )

        # Same idempotency + safeguard logic as open.
        try:
            check_order(
                spot_close, self._settings,
                await self._store.total_at_risk_usd(),
                await self._store.layer_deployed_usd(self.LAYER),
                await self._store.recent_client_order_ids(),
            )
        except GuardViolation as e:
            log.info("l1_close_guard_violation",
                     position=pos.id, code=e.code, msg=str(e))
            # Restore state — we'll retry next cycle
            await self._store.mark_position_open(pos.id)
            return False

        s_filled = await self._submit_leg(
            order=spot_close, position_id=pos.id, order_type="limit",
            base_size=spot_size_base, market_type="spot",
        )
        p_filled = await self._submit_leg(
            order=perp_close, position_id=pos.id, order_type="limit",
            base_size=perp_size_base, market_type="swap",
        )

        if s_filled and p_filled:
            funding_collected = await self._store.total_funding_collected(pos.id)
            # Approximation: treat close-fill price as entry-fill price (delta-neutral
            # so basis-change PnL is bounded and small). Real PnL booking happens
            # when we get actual fill receipts; for now funding is the dominant term.
            realized = funding_collected
            await self._store.close_position(pos.id, realized)
            log.info("l1_position_closed",
                     position=pos.id, pair=pair_key,
                     funding_collected=round(funding_collected, 4),
                     duration_h=round(
                         (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 3600.0, 2,
                     ))
            return True

        if s_filled != p_filled:
            log.error("l1_close_partial",
                      position=pos.id, pair=pair_key,
                      spot_close=s_filled, perp_close=p_filled)
            await self._store.set_halt_reason(
                f"L1 partial close at position {pos.id} ({pair_key})")
        return False
