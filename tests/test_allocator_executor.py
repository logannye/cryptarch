"""Tests for AllocatorExecutor — verifies signal aggregation + DB write.
Mocks L1, L2 signal, and pool to test orchestration without network."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptarch.core.config import Settings
from cryptarch.exchanges.base import Candle
from cryptarch.strategies.allocator_executor import AllocatorExecutor
from cryptarch.strategies.l1_funding import FundingArbCandidate
from cryptarch.strategies.l2_executor import SymbolConfig


def _settings(**kw) -> Settings:
    base = dict(
        bankroll_usd=2000.0,
        alloc_layer_1_pct=0.60, alloc_layer_2_pct=0.25, alloc_layer_3_pct=0.15,
        max_total_deployed_pct=0.50, max_per_position_usd=500.0,
        enable_live_orders=False,
    )
    base.update(kw)
    return Settings(_env_file=None, **base)


def _make_pool_with_btc_candles(closes: list[float]):
    now = datetime.now(timezone.utc)
    candles = [
        Candle(ts=now, open=c, high=c, low=c, close=c, volume=1.0)
        for c in closes
    ]
    spot_client = MagicMock()
    spot_client.get_ohlcv = AsyncMock(return_value=candles)
    pool = MagicMock()
    async def _get(exchange_id, market_type="spot"):
        return spot_client
    pool.get = _get
    return pool


def _l1_executor_with_candidates(candidates):
    l1 = MagicMock()
    l1._scan_candidates = AsyncMock(return_value=candidates)
    return l1


# ── interval gating ──


@pytest.mark.asyncio
async def test_skips_until_interval_reached():
    pool = _make_pool_with_btc_candles([100] * 60)
    store = MagicMock()
    store.set_dynamic_allocation = AsyncMock()
    l1 = _l1_executor_with_candidates([])
    signal_fn = AsyncMock(return_value=0.0)
    alloc = AllocatorExecutor(
        settings=_settings(), store=store, pool=pool,
        l1_executor=l1, l2_signal_fn=signal_fn,
        l2_symbols=[], recompute_every_n_cycles=5,
    )
    # First 4 cycles should skip
    for _ in range(4):
        result = await alloc.run_once()
        assert result.get("skipped") == "interval_not_reached"
    # 5th cycle should compute
    result = await alloc.run_once()
    assert "skipped" not in result
    store.set_dynamic_allocation.assert_awaited_once()


# ── signal aggregation ──


@pytest.mark.asyncio
async def test_l1_signal_takes_max_apr_across_pairs():
    candidates = [
        FundingArbCandidate(
            spot_exchange="b", spot_symbol="BTC/USDT",
            perp_exchange="b", perp_symbol="BTC/USDT:USDT",
            spot_price=50000, perp_price=50100, funding_rate_8h=0.0003,
        ),
        FundingArbCandidate(
            spot_exchange="b", spot_symbol="ETH/USDT",
            perp_exchange="b", perp_symbol="ETH/USDT:USDT",
            spot_price=2000, perp_price=2010, funding_rate_8h=0.0010,
        ),
    ]
    pool = _make_pool_with_btc_candles([100] * 60)
    store = MagicMock()
    store.set_dynamic_allocation = AsyncMock()
    l1 = _l1_executor_with_candidates(candidates)
    signal_fn = AsyncMock(return_value=0.0)
    alloc = AllocatorExecutor(
        settings=_settings(), store=store, pool=pool,
        l1_executor=l1, l2_signal_fn=signal_fn,
        l2_symbols=[], recompute_every_n_cycles=1,
    )
    result = await alloc.run_once()
    # ETH funding 0.001 × 3 × 0.5 × 365 × 100 = 54.75% APR
    # → would tilt L1 up
    args = store.set_dynamic_allocation.await_args
    assert args.kwargs["l1"] > 0.60    # tilted up


@pytest.mark.asyncio
async def test_l2_signal_takes_max_score_across_symbols():
    pool = _make_pool_with_btc_candles([100] * 60)
    store = MagicMock()
    store.set_dynamic_allocation = AsyncMock()
    l1 = _l1_executor_with_candidates([])
    # Two symbols; one returns 0.9, the other 0.3 — max should win
    async def signal_fn(sym, executor):
        return 0.9 if sym.symbol == "BTC/USDT" else 0.3
    symbols = [
        SymbolConfig("binance", "BTC/USDT", "BTC"),
        SymbolConfig("binance", "ETH/USDT", "ETH"),
    ]
    alloc = AllocatorExecutor(
        settings=_settings(), store=store, pool=pool,
        l1_executor=l1, l2_signal_fn=signal_fn,
        l2_symbols=symbols, recompute_every_n_cycles=1,
    )
    await alloc.run_once()
    args = store.set_dynamic_allocation.await_args
    # L2 cascade score 0.9 → tilts L2 up
    assert args.kwargs["l2"] > 0.25


@pytest.mark.asyncio
async def test_l3_compression_signal_from_candles():
    # Recent prices flat (low vol), historical oscillates (high vol)
    # → recent vol < historical vol → compression score > 0
    historical = [100, 105, 95, 105, 95, 105, 95] * 10    # 70 bars oscillating
    recent = [100] * 10                                    # 10 flat bars
    closes = historical[:-10] + recent                     # combine
    pool = _make_pool_with_btc_candles(closes)
    store = MagicMock()
    store.set_dynamic_allocation = AsyncMock()
    l1 = _l1_executor_with_candidates([])
    signal_fn = AsyncMock(return_value=0.0)
    alloc = AllocatorExecutor(
        settings=_settings(), store=store, pool=pool,
        l1_executor=l1, l2_signal_fn=signal_fn,
        l2_symbols=[], recompute_every_n_cycles=1,
    )
    await alloc.run_once()
    args = store.set_dynamic_allocation.await_args
    # IV compression should tilt L3 up
    assert args.kwargs["l3"] > 0.15


# ── no signal → baseline ──


@pytest.mark.asyncio
async def test_no_signals_writes_baseline():
    pool = _make_pool_with_btc_candles([100] * 60)
    store = MagicMock()
    store.set_dynamic_allocation = AsyncMock()
    l1 = _l1_executor_with_candidates([])
    signal_fn = AsyncMock(return_value=0.0)
    alloc = AllocatorExecutor(
        settings=_settings(), store=store, pool=pool,
        l1_executor=l1, l2_signal_fn=signal_fn,
        l2_symbols=[], recompute_every_n_cycles=1,
    )
    result = await alloc.run_once()
    args = store.set_dynamic_allocation.await_args
    assert args.kwargs["l1"] == pytest.approx(0.60)
    assert args.kwargs["l2"] == pytest.approx(0.25)
    assert args.kwargs["l3"] == pytest.approx(0.15)


# ── failure resilience ──


@pytest.mark.asyncio
async def test_l1_scan_failure_returns_zero_signal():
    pool = _make_pool_with_btc_candles([100] * 60)
    store = MagicMock()
    store.set_dynamic_allocation = AsyncMock()
    l1 = MagicMock()
    l1._scan_candidates = AsyncMock(side_effect=Exception("network"))
    signal_fn = AsyncMock(return_value=0.0)
    alloc = AllocatorExecutor(
        settings=_settings(), store=store, pool=pool,
        l1_executor=l1, l2_signal_fn=signal_fn,
        l2_symbols=[], recompute_every_n_cycles=1,
    )
    # Should not raise
    result = await alloc.run_once()
    # L1 signal becomes 0 → baseline allocation
    args = store.set_dynamic_allocation.await_args
    assert args.kwargs["l1"] == pytest.approx(0.60)
