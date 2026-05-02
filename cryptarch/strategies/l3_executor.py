"""L3 tail-hedge executor.

Each cycle:
  1. Check open L3 strangle (one position with two legs in metadata)
  2. If no strangle exists → open one (size capped by daily theta budget)
  3. If existing strangle is ≤ target_dte_min from expiry → roll
  4. Otherwise → hold

Sizing
------
The executor caps position size such that daily theta cost stays within
`l3_daily_theta_budget_usd` (default $10/day). At default 45-DTE
strangle with ~$2,500 total premium, that's roughly 0.18 BTC
worth of contracts ($9k notional).

Dry-run vs live
---------------
In dry-run mode (`enable_live_orders=False`), the executor still reads
real Deribit option chains and quotes — so the recorded position is
genuinely realistic (real IV, real premiums) — but no orders submit.
This lets us track simulated theta decay vs real vol regime moves
honestly.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from cryptarch.core.config import Settings
from cryptarch.db.store import OpenPosition, Store
from cryptarch.exchanges.deribit_options import (
    DeribitOptionsClient, OptionInstrument, OptionQuote, parse_instrument_name,
)
from cryptarch.exchanges.pool import ExchangePool
from cryptarch.strategies.l3_tail import (
    OptionLeg, StranglePosition, daily_theta_cost,
    days_to_expiry, max_contracts_within_theta_budget,
    select_strangle_strikes, select_target_expiry, should_roll,
)

log = structlog.get_logger()


@dataclass(frozen=True)
class UnderlyingConfig:
    """One underlying we're hedging (BTC, ETH).

    The bot may run a separate strangle per underlying; we treat them as
    independent positions. For v1 we default to BTC only; ETH can be
    enabled by extending DEFAULT_UNDERLYINGS.
    """
    symbol_id: str          # "BTC", "ETH"
    spot_exchange: str      # exchange to use for fetching spot price
    spot_symbol: str        # e.g. "BTC/USDT"
    deribit_currency: str   # "BTC" or "ETH" (Deribit's underlying selector)


DEFAULT_UNDERLYINGS: list[UnderlyingConfig] = [
    UnderlyingConfig(
        symbol_id="BTC",
        spot_exchange="binance",
        spot_symbol="BTC/USDT",
        deribit_currency="BTC",
    ),
]


class L3Executor:
    LAYER = "l3_tail"
    TARGET_DTE = 45            # ideal DTE when opening fresh
    MIN_DTE_BEFORE_ROLL = 30   # roll when DTE drops below this
    OTM_PCT_DEFAULT = 0.20

    def __init__(
        self,
        settings: Settings,
        store: Store,
        pool: ExchangePool,
        underlyings: list[UnderlyingConfig] | None = None,
    ):
        self._settings = settings
        self._store = store
        self._pool = pool
        self._underlyings = underlyings or list(DEFAULT_UNDERLYINGS)

    # ── public entry ──

    async def run_once(self) -> dict[str, Any]:
        if not self._settings.layer_3_tail_hedge_enabled:
            return {"skipped": "layer_disabled"}

        state = await self._store.get_system_state()
        if state and state.halt_reason:
            return {"skipped": "halted", "reason": state.halt_reason}

        open_positions = await self._store.open_positions(layer=self.LAYER)

        n_opened = 0
        n_rolled = 0
        for underlying in self._underlyings:
            held = next(
                (p for p in open_positions
                 if p.metadata.get("underlying") == underlying.symbol_id),
                None,
            )
            if held is None:
                if await self._open_strangle(underlying):
                    n_opened += 1
            else:
                if await self._maybe_roll(held, underlying):
                    n_rolled += 1
        return {
            "open_positions": len(open_positions),
            "opened": n_opened,
            "rolled": n_rolled,
        }

    # ── open strangle ──

    async def _open_strangle(self, u: UnderlyingConfig) -> bool:
        """Identify OTM strikes at target_dte, fetch quotes, size to
        theta budget, persist position record."""
        # Spot
        try:
            spot_client = await self._pool.get(u.spot_exchange, market_type="spot")
            ticker = await spot_client.get_ticker(u.spot_symbol)
        except Exception as e:
            log.warning("l3_spot_fetch_failed", underlying=u.symbol_id,
                        error=str(e)[:100])
            return False
        spot = float(ticker.last or ticker.ask or 0)
        if spot <= 0:
            return False

        # Deribit chain
        try:
            deribit_base = await self._pool.get("deribit", market_type="option")
            deribit = DeribitOptionsClient(deribit_base)
            instruments = await deribit.list_instruments(currency=u.deribit_currency)
        except Exception as e:
            log.warning("l3_deribit_fetch_failed", underlying=u.symbol_id,
                        error=str(e)[:100])
            return False
        if not instruments:
            log.info("l3_no_instruments", underlying=u.symbol_id)
            return False

        # Pick expiry + strikes
        expiries = sorted({i.expiry for i in instruments})
        target_expiry = select_target_expiry(expiries, target_dte=self.TARGET_DTE)
        if target_expiry is None:
            return False
        chain_at_expiry = await deribit.filter_by_expiry(instruments, target_expiry)

        target_call_strike, target_put_strike = select_strangle_strikes(
            spot=spot, otm_pct=self._settings.l3_otm_pct, strike_step=1000.0,
        )
        call_inst = DeribitOptionsClient.find_closest_strike(
            chain_at_expiry, target_call_strike, "C")
        put_inst = DeribitOptionsClient.find_closest_strike(
            chain_at_expiry, target_put_strike, "P")
        if call_inst is None or put_inst is None:
            log.info("l3_no_matching_strikes", underlying=u.symbol_id,
                     target_call=target_call_strike, target_put=target_put_strike)
            return False

        # Quotes
        call_q = await deribit.get_quote(call_inst, spot_usd=spot)
        put_q = await deribit.get_quote(put_inst, spot_usd=spot)
        call_premium = call_q.ask_usd or 0
        put_premium = put_q.ask_usd or 0
        if call_premium <= 0 or put_premium <= 0:
            log.info("l3_no_quotes", underlying=u.symbol_id,
                     call=call_premium, put=put_premium)
            return False

        # Size to theta budget
        dte = days_to_expiry(target_expiry)
        contracts = max_contracts_within_theta_budget(
            daily_theta_budget_usd=self._settings.l3_daily_theta_budget_usd,
            call_premium_usd=call_premium,
            put_premium_usd=put_premium,
            days_to_expiry_now=dte,
        )
        # Round down to a contract size Deribit accepts (typically 0.1 BTC steps).
        contracts = round(contracts * 10) / 10.0
        if contracts <= 0:
            log.info("l3_size_too_small", underlying=u.symbol_id,
                     budget=self._settings.l3_daily_theta_budget_usd,
                     premium_total=call_premium + put_premium, dte=dte)
            return False

        notional_usd = (call_premium + put_premium) * contracts
        group_id = str(uuid.uuid4())

        # Record as a single position with both legs in metadata.
        position_id = await self._store.create_position(
            layer=self.LAYER,
            strategy_group_id=group_id,
            notional_usd=notional_usd,
            metadata={
                "underlying": u.symbol_id,
                "spot_at_open": spot,
                "expiry_iso": target_expiry.isoformat(),
                "dte_at_open": round(dte, 2),
                "call_instrument": call_inst.instrument_name,
                "call_strike": call_inst.strike,
                "call_premium_usd": call_premium,
                "call_iv": call_q.mark_iv,
                "put_instrument": put_inst.instrument_name,
                "put_strike": put_inst.strike,
                "put_premium_usd": put_premium,
                "put_iv": put_q.mark_iv,
                "contracts": contracts,
                "daily_theta_estimate": daily_theta_cost(
                    call_premium, put_premium, contracts, dte),
                "is_simulated": not self._settings.enable_live_orders,
            },
        )

        # Record fills (or simulated equivalents).
        await self._store.record_fill(
            position_id=position_id,
            layer=self.LAYER,
            exchange="deribit",
            symbol=call_inst.instrument_name,
            side="buy",
            order_type="limit",
            size_base=contracts,
            size_usd=call_premium * contracts,
            fill_price=call_premium,
            client_order_id=f"cryptarch-l3-{group_id}-call",
            is_simulated=not self._settings.enable_live_orders,
            sim_reason="strangle_open",
        )
        await self._store.record_fill(
            position_id=position_id,
            layer=self.LAYER,
            exchange="deribit",
            symbol=put_inst.instrument_name,
            side="buy",
            order_type="limit",
            size_base=contracts,
            size_usd=put_premium * contracts,
            fill_price=put_premium,
            client_order_id=f"cryptarch-l3-{group_id}-put",
            is_simulated=not self._settings.enable_live_orders,
            sim_reason="strangle_open",
        )
        await self._store.mark_position_open(position_id)

        log.info("l3_strangle_opened",
                 position=position_id, underlying=u.symbol_id,
                 spot=round(spot, 2),
                 call_strike=call_inst.strike, call_premium_usd=round(call_premium, 2),
                 put_strike=put_inst.strike, put_premium_usd=round(put_premium, 2),
                 contracts=contracts, dte=round(dte, 1),
                 daily_theta=round(daily_theta_cost(
                     call_premium, put_premium, contracts, dte), 4))
        return True

    # ── roll ──

    async def _maybe_roll(self, pos: OpenPosition, u: UnderlyingConfig) -> bool:
        """If DTE < threshold, close and reopen at a fresh expiry."""
        expiry_iso = pos.metadata.get("expiry_iso")
        if not expiry_iso:
            return False
        try:
            expiry = datetime.fromisoformat(expiry_iso)
        except (ValueError, TypeError):
            return False
        if not should_roll(expiry, target_dte_min=self.MIN_DTE_BEFORE_ROLL):
            return False    # still has runway

        # Close current position. PnL = total premium burned so far,
        # estimated. Real exit value would come from selling the legs
        # back into the Deribit chain — for simulation we mark the
        # closing premiums to 0 (worst-case theta-only outcome). A
        # future refinement would mark-to-market via live quotes at
        # close time and book the actual residual value.
        await self._store.close_position(pos.id, realized_pnl=-float(pos.notional_usd))
        log.info("l3_strangle_rolled_close",
                 position=pos.id, underlying=u.symbol_id,
                 dte_at_close=round(days_to_expiry(expiry), 2))

        # Open a fresh strangle.
        if await self._open_strangle(u):
            log.info("l3_strangle_rolled_open", underlying=u.symbol_id)
        return True
