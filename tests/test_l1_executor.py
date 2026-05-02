"""Tests for L1Executor orchestration. Mocks exchange + store so we can
verify the decision logic without real network/DB access."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptarch.core.config import Settings
from cryptarch.db.store import OpenPosition
from cryptarch.exchanges.base import FundingRate, Ticker
from cryptarch.sim.realistic import OrderBookLevel, OrderBookSnapshot
from cryptarch.strategies.l1_executor import L1Executor, PairConfig


def _settings(**overrides) -> Settings:
    base = dict(
        bankroll_usd=2000.0,
        alloc_layer_1_pct=0.60,
        alloc_layer_2_pct=0.25,
        alloc_layer_3_pct=0.15,
        max_total_deployed_pct=0.50,
        max_per_position_usd=500.0,
        enable_live_orders=False,
        layer_1_funding_arb_enabled=True,
        l1_min_funding_rate_8h=0.0003,
        l1_min_basis_pct=0.001,
        l1_max_basis_pct=0.020,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _book(asks: list[tuple[float, float]], bids: list[tuple[float, float]]) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        asks=tuple(OrderBookLevel(p, s) for p, s in asks),
        bids=tuple(OrderBookLevel(p, s) for p, s in bids),
    )


def _mock_pool_with_market(
    spot_price: float, perp_price: float, funding_8h: float,
    spot_book: OrderBookSnapshot | None = None,
    perp_book: OrderBookSnapshot | None = None,
):
    """Build an ExchangePool mock that returns deterministic prices/books
    for whatever (exchange, market_type) is asked for."""
    spot_book = spot_book or _book(
        asks=[(spot_price * 1.0001, 1000)],
        bids=[(spot_price * 0.9999, 1000)],
    )
    perp_book = perp_book or _book(
        asks=[(perp_price * 1.0001, 1000)],
        bids=[(perp_price * 0.9999, 1000)],
    )

    spot_client = MagicMock()
    spot_client.get_ticker = AsyncMock(return_value=Ticker(
        exchange="binance", symbol="BTC/USDT",
        bid=spot_price - 1, ask=spot_price + 1, last=spot_price,
        fetched_at=datetime.now(timezone.utc),
    ))
    spot_client.get_order_book = AsyncMock(return_value=spot_book)

    perp_client = MagicMock()
    perp_client.get_ticker = AsyncMock(return_value=Ticker(
        exchange="binance", symbol="BTC/USDT:USDT",
        bid=perp_price - 1, ask=perp_price + 1, last=perp_price,
        fetched_at=datetime.now(timezone.utc),
    ))
    perp_client.get_funding_rate = AsyncMock(return_value=FundingRate(
        exchange="binance", symbol="BTC/USDT:USDT",
        rate_8h=funding_8h,
        next_funding_time=datetime.now(timezone.utc),
        fetched_at=datetime.now(timezone.utc),
    ))
    perp_client.get_order_book = AsyncMock(return_value=perp_book)

    pool = MagicMock()
    async def get(exchange_id, market_type="spot"):
        return spot_client if market_type == "spot" else perp_client
    pool.get = get
    return pool, spot_client, perp_client


def _mock_store(
    open_positions: list[OpenPosition] | None = None,
    total_at_risk: float = 0.0,
    layer_deployed: float = 0.0,
    state_halt_reason: str | None = None,
):
    store = MagicMock()
    state = MagicMock()
    state.halt_reason = state_halt_reason
    state.dynamic_alloc_l1_pct = None
    state.dynamic_alloc_l2_pct = None
    state.dynamic_alloc_l3_pct = None
    store.get_system_state = AsyncMock(return_value=state)
    store.open_positions = AsyncMock(return_value=open_positions or [])
    store.total_at_risk_usd = AsyncMock(return_value=total_at_risk)
    store.layer_deployed_usd = AsyncMock(return_value=layer_deployed)
    store.recent_client_order_ids = AsyncMock(return_value=set())
    store.create_position = AsyncMock(return_value=42)
    store.mark_position_open = AsyncMock()
    store.mark_position_closing = AsyncMock()
    store.close_position = AsyncMock()
    store.record_fill = AsyncMock(return_value=1)
    store.total_funding_collected = AsyncMock(return_value=0.0)
    store.set_halt_reason = AsyncMock()
    return store


# ── attractive candidate gets opened ──


@pytest.mark.asyncio
async def test_opens_attractive_candidate(monkeypatch):
    settings = _settings()
    pool, spot, perp = _mock_pool_with_market(
        spot_price=50000.0, perp_price=50100.0,    # 20bp basis
        funding_8h=0.0010,                          # 0.1% per 8h, hot
    )
    store = _mock_store()
    pairs = [PairConfig("binance", "BTC/USDT", "binance", "BTC/USDT:USDT", "BTC")]
    executor = L1Executor(settings, store, pool, pairs=pairs)

    result = await executor.run_once()

    assert result["candidates"] == 1
    assert result["opened"] == 1
    store.create_position.assert_awaited_once()
    # Both legs got recorded as filled.
    assert store.record_fill.await_count == 2
    store.mark_position_open.assert_awaited_once_with(42)


# ── unattractive candidate is skipped ──


@pytest.mark.asyncio
async def test_skips_unattractive_funding():
    settings = _settings()
    pool, _, _ = _mock_pool_with_market(
        spot_price=50000.0, perp_price=50100.0,
        funding_8h=0.00001,    # tiny funding, below threshold
    )
    store = _mock_store()
    pairs = [PairConfig("binance", "BTC/USDT", "binance", "BTC/USDT:USDT", "BTC")]
    executor = L1Executor(settings, store, pool, pairs=pairs)
    result = await executor.run_once()
    assert result["opened"] == 0
    store.create_position.assert_not_called()


@pytest.mark.asyncio
async def test_skips_inverted_basis():
    settings = _settings()
    pool, _, _ = _mock_pool_with_market(
        spot_price=50000.0, perp_price=49500.0,    # perp BELOW spot
        funding_8h=0.0010,
    )
    store = _mock_store()
    pairs = [PairConfig("binance", "BTC/USDT", "binance", "BTC/USDT:USDT", "BTC")]
    executor = L1Executor(settings, store, pool, pairs=pairs)
    result = await executor.run_once()
    assert result["opened"] == 0


# ── refuses to fill when book is too wide (polybot lesson) ──


@pytest.mark.asyncio
async def test_refuses_unfillable_spot_book():
    settings = _settings()
    # Bid 0.001, ask 0.999 — Polymarket-style joke book on spot leg.
    spot_book = _book(asks=[(0.999, 1000)], bids=[(0.001, 1000)])
    perp_book = _book(asks=[(50100, 1000)], bids=[(50050, 1000)])
    pool, _, _ = _mock_pool_with_market(
        spot_price=50000.0, perp_price=50100.0, funding_8h=0.0010,
        spot_book=spot_book, perp_book=perp_book,
    )
    store = _mock_store()
    pairs = [PairConfig("binance", "BTC/USDT", "binance", "BTC/USDT:USDT", "BTC")]
    executor = L1Executor(settings, store, pool, pairs=pairs)
    result = await executor.run_once()
    # Spot leg can't fill in simulator → position halted (one leg failed)
    # One leg recorded as "filled" with size 0 (perp side); other refused.
    # The store's halt_reason should be set.
    store.set_halt_reason.assert_awaited()


# ── halted state skips entirely ──


@pytest.mark.asyncio
async def test_halted_state_skips_cycle():
    settings = _settings()
    pool, _, _ = _mock_pool_with_market(
        spot_price=50000.0, perp_price=50100.0, funding_8h=0.0010,
    )
    store = _mock_store(state_halt_reason="manual halt for testing")
    pairs = [PairConfig("binance", "BTC/USDT", "binance", "BTC/USDT:USDT", "BTC")]
    executor = L1Executor(settings, store, pool, pairs=pairs)
    result = await executor.run_once()
    assert result == {"skipped": "halted", "reason": "manual halt for testing"}
    store.create_position.assert_not_called()


# ── disabled layer is no-op ──


@pytest.mark.asyncio
async def test_disabled_layer_no_op():
    settings = _settings(layer_1_funding_arb_enabled=False)
    pool = MagicMock()
    store = _mock_store()
    executor = L1Executor(settings, store, pool, pairs=[])
    result = await executor.run_once()
    assert result == {"skipped": "layer_disabled"}
    pool.get.assert_not_called() if hasattr(pool.get, "assert_not_called") else None


# ── existing position dedup ──


@pytest.mark.asyncio
async def test_does_not_double_open_same_pair():
    settings = _settings()
    pool, _, _ = _mock_pool_with_market(
        spot_price=50000.0, perp_price=50100.0, funding_8h=0.0010,
    )
    # Already hold a position on this pair.
    existing = OpenPosition(
        id=99, layer="l1_funding", strategy_group_id="g1",
        state="open", notional_usd=200.0, realized_pnl_usd=0.0,
        opened_at=datetime.now(timezone.utc),
        metadata={"pair_group_key": "binance:BTC/USDT|binance:BTC/USDT:USDT"},
    )
    store = _mock_store(open_positions=[existing])
    pairs = [PairConfig("binance", "BTC/USDT", "binance", "BTC/USDT:USDT", "BTC")]
    executor = L1Executor(settings, store, pool, pairs=pairs)
    result = await executor.run_once()
    assert result["opened"] == 0
