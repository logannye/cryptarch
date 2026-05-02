"""Entry point for `python -m cryptarch`.

Phase 0 commands:
  init-db   — apply schema.sql
  status    — show system_state + open positions
  audit     — read-only audit of exchange connectivity + balances

Future phases will add: scan, run, settle, etc.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import asyncpg
import click
import structlog

from cryptarch.core.config import Settings


def _configure_logging() -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


@click.group()
def cli():
    """cryptarch — three-layer crypto trading system."""
    _configure_logging()


@cli.command("init-db")
def init_db_cmd():
    """Apply schema.sql to the configured database."""
    asyncio.run(_init_db())


async def _init_db() -> None:
    settings = Settings()
    schema_path = Path(__file__).parent / "db" / "schema.sql"
    sql = schema_path.read_text()
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            await conn.execute(sql)
        print(f"Schema applied to {settings.database_url}")
    finally:
        await pool.close()


@cli.command("status")
def status_cmd():
    """Print current system_state + open positions."""
    asyncio.run(_status())


async def _status() -> None:
    settings = Settings()
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            state = await conn.fetchrow("SELECT * FROM system_state WHERE id=1")
            n_open = await conn.fetchval(
                "SELECT COUNT(*) FROM position WHERE state IN ('opening', 'open')")
            n_fills_24h = await conn.fetchval(
                "SELECT COUNT(*) FROM fill WHERE placed_at > NOW() - INTERVAL '24 hours'")
        if state is None:
            print("no system_state row — run init-db first")
            return
        print(json.dumps({
            "bankroll_usd": float(state["bankroll_usd"]),
            "cash_usd": float(state["cash_usd"]),
            "total_at_risk_usd": float(state["total_at_risk_usd"]),
            "daily_pnl": float(state["daily_pnl"]),
            "cumulative_pnl": float(state["cumulative_pnl"]),
            "deployment_stage": state["deployment_stage"],
            "halt_reason": state["halt_reason"],
            "open_positions": int(n_open or 0),
            "fills_24h": int(n_fills_24h or 0),
            "live_orders_enabled": settings.enable_live_orders,
            "layers_enabled": {
                "l1_funding": settings.layer_1_funding_arb_enabled,
                "l2_cascade": settings.layer_2_cascade_capture_enabled,
                "l3_tail":    settings.layer_3_tail_hedge_enabled,
            },
        }, indent=2, default=str))
    finally:
        await pool.close()


@cli.command("scan")
def scan_cmd():
    """One-shot L1 scan: print current funding-arb candidates and attractiveness verdicts."""
    asyncio.run(_scan())


async def _scan() -> None:
    """Use the executor's parallel scan path so we get the same fast
    fetch as the live engine. Read-only — no DB required."""
    from unittest.mock import MagicMock
    from cryptarch.exchanges.pool import ExchangePool
    from cryptarch.strategies.l1_executor import L1Executor
    from cryptarch.strategies.l1_funding import is_attractive

    settings = Settings()
    pool = ExchangePool(settings)
    try:
        # Scan-only mode: stub the store; we never call any of its methods
        # because we only invoke _scan_candidates which doesn't touch DB.
        executor = L1Executor(settings, store=MagicMock(), pool=pool)
        candidates = await executor._scan_candidates()    # parallel
        # Sort by APR desc so the most interesting candidates print first.
        candidates.sort(key=lambda c: c.expected_apr_pct, reverse=True)
        for cand in candidates:
            label = next(
                (p.base_label for p in executor._pairs
                 if p.spot_symbol == cand.spot_symbol), cand.spot_symbol,
            )
            attractive = is_attractive(
                cand,
                min_funding_8h=settings.l1_min_funding_rate_8h,
                min_basis_pct=settings.l1_min_basis_pct,
                max_basis_pct=settings.l1_max_basis_pct,
            )
            print(json.dumps({
                "pair": label,
                "spot": round(cand.spot_price, 6),
                "perp": round(cand.perp_price, 6),
                "basis_pct": round(cand.basis_pct * 100, 4),
                "funding_8h_pct": round(cand.funding_rate_8h * 100, 4),
                "expected_apr_pct": round(cand.expected_apr_pct * 100, 2),
                "attractive": attractive,
            }))
        n_attractive = sum(
            1 for c in candidates
            if is_attractive(
                c,
                settings.l1_min_funding_rate_8h,
                settings.l1_min_basis_pct,
                settings.l1_max_basis_pct,
            )
        )
        print(json.dumps({
            "total_pairs_scanned": len(candidates),
            "attractive": n_attractive,
        }))
    finally:
        await pool.close_all()


@cli.command("run")
def run_cmd():
    """Start the engine — runs forever, scanning + managing L1 positions."""
    asyncio.run(_run())


async def _run() -> None:
    from cryptarch.core.engine import Engine
    from cryptarch.db.store import Store
    from cryptarch.exchanges.pool import ExchangePool
    from cryptarch.strategies.l1_executor import L1Executor
    from cryptarch.strategies.l2_executor import L2Executor, SymbolConfig, DEFAULT_SYMBOLS
    from cryptarch.strategies.l2_signal import CascadeSignal, OIObserver
    from cryptarch.strategies.l3_executor import L3Executor
    from cryptarch.strategies.allocator_executor import AllocatorExecutor
    settings = Settings()
    pool = ExchangePool(settings)
    db_pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=5)
    store = Store(db_pool)
    try:
        engine = Engine(settings, store, pool)
        l1: L1Executor | None = None
        cascade_signal_for_alloc: CascadeSignal | None = None
        l2_symbols_for_alloc: list[SymbolConfig] = []
        if settings.layer_1_funding_arb_enabled:
            l1 = L1Executor(settings, store, pool)
            engine.register("l1_funding", l1.run_once)
        if settings.layer_2_cascade_capture_enabled:
            symbols = [SymbolConfig(*s) for s in DEFAULT_SYMBOLS]
            l2_symbols_for_alloc = symbols
            # OI observer runs alongside L2 to bootstrap the rolling history
            # that the signal needs. Register it whether or not L2 is firing
            # so history accumulates during cold start.
            observer = OIObserver(store, pool, symbols)
            engine.register("oi_observer", observer.run_once)
            # Real signal function (replaces stubbed default).
            signal = CascadeSignal(store, pool)
            cascade_signal_for_alloc = signal
            l2 = L2Executor(settings, store, pool, symbols=symbols, signal_fn=signal)
            engine.register("l2_cascade", l2.run_once)
        if settings.layer_3_tail_hedge_enabled:
            l3 = L3Executor(settings, store, pool)
            engine.register("l3_tail", l3.run_once)

        # Phase 4: dynamic allocator. Only register if at least L1 is up
        # (it needs L1 to compute the funding signal).
        if l1 is not None and cascade_signal_for_alloc is not None:
            allocator = AllocatorExecutor(
                settings, store, pool,
                l1_executor=l1,
                l2_signal_fn=cascade_signal_for_alloc,
                l2_symbols=l2_symbols_for_alloc,
            )
            engine.register("allocator", allocator.run_once)
        await engine.run_forever()
    finally:
        await pool.close_all()
        await db_pool.close()


@cli.command("audit")
def audit_cmd():
    """Phase 0 audit: verify exchange connectivity + balances + book quality.
    Read-only; safe to run anytime. Mirrors polybot's Phase 0 pattern."""
    asyncio.run(_audit())


async def _audit() -> None:
    from cryptarch.exchanges.ccxt_client import CCXTClient
    settings = Settings()
    print("=" * 70)
    print("CRYPTOARB PHASE 0 AUDIT (read-only)")
    print("=" * 70)

    # We try Binance spot + perp; expand as we wire more exchanges.
    targets: list[tuple[str, str, str, str]] = [
        ("binance", "spot", settings.binance_api_key, settings.binance_api_secret),
        ("binance", "swap", settings.binance_api_key, settings.binance_api_secret),
    ]

    for ex_id, market_type, key, secret in targets:
        print(f"\n[{ex_id} {market_type}]")
        if not key:
            print(f"  credentials missing — skipping live balance check")
            client = CCXTClient(ex_id, market_type=market_type)
        else:
            client = CCXTClient(ex_id, key, secret, market_type=market_type)
        try:
            await client.start()
            print(f"  connected; markets loaded")
            # Sample order book + ticker
            symbol = "BTC/USDT" if market_type == "spot" else "BTC/USDT:USDT"
            try:
                ticker = await client.get_ticker(symbol)
                print(f"  {symbol} ticker: bid={ticker.bid} ask={ticker.ask} last={ticker.last}")
                book = await client.get_order_book(symbol, depth=5)
                print(f"  {symbol} book: best_bid={book.best_bid} best_ask={book.best_ask} spread={book.spread}")
            except Exception as e:
                print(f"  market-data fetch failed: {e}")
            if market_type == "swap":
                try:
                    fr = await client.get_funding_rate(symbol)
                    print(f"  {symbol} funding 8h: {fr.rate_8h:+.6f} ({fr.rate_8h * 3 * 365 * 100:+.2f}% APR)")
                except Exception as e:
                    print(f"  funding-rate fetch failed: {e}")
            if key:
                try:
                    balance = await client.get_balance()
                    print(f"  balance: cash=${balance.cash_usd:.2f} positions=${balance.position_notional_usd:.2f}")
                except Exception as e:
                    print(f"  balance fetch failed: {e}")
        except Exception as e:
            print(f"  FAILED: {e}")
        finally:
            await client.close()

    print("\n" + "=" * 70)
    print("Phase 0 audit complete.")
    print("=" * 70)


if __name__ == "__main__":
    cli()
