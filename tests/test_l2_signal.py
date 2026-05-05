"""Tests for CascadeSignal and OIObserver. Mocks store + exchange so we
verify wiring without real connections."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptarch.exchanges.base import Candle, FundingRate, OpenInterest
from cryptarch.strategies.l2_executor import SymbolConfig
from cryptarch.strategies.l2_signal import (
    MIN_OI_HISTORY_LEN, CascadeSignal, OIObserver,
)


def _candles_flat(n: int = 168, price: float = 100.0) -> list[Candle]:
    now = datetime.now(timezone.utc)
    return [
        Candle(ts=now - timedelta(hours=n - i), open=price, high=price,
               low=price, close=price, volume=1.0)
        for i in range(n)
    ]


def _candles_compressed(n: int = 168) -> list[Candle]:
    """Older candles oscillate ±5%; recent 24 are flat (compressed vol)."""
    now = datetime.now(timezone.utc)
    out = []
    for i in range(n):
        if i < n - 24:
            # Older bars: alternating up/down
            price = 100 * (1.05 if i % 2 == 0 else 0.95)
        else:
            # Recent 24: flat
            price = 100.0
        out.append(Candle(ts=now - timedelta(hours=n - i),
                          open=price, high=price, low=price,
                          close=price, volume=1.0))
    return out


def _make_pool_for_signal(
    current_oi_usd: float,
    funding_8h: float,
    candles: list[Candle],
):
    perp_client = MagicMock()
    perp_client.get_open_interest = AsyncMock(return_value=OpenInterest(
        exchange="binance", symbol="BTC/USDT:USDT",
        oi_usd=current_oi_usd, oi_base=0.0,
        fetched_at=datetime.now(timezone.utc),
    ))
    perp_client.get_funding_rate = AsyncMock(return_value=FundingRate(
        exchange="binance", symbol="BTC/USDT:USDT",
        rate_8h=funding_8h,
        next_funding_time=datetime.now(timezone.utc),
        fetched_at=datetime.now(timezone.utc),
    ))
    spot_client = MagicMock()
    spot_client.get_ohlcv = AsyncMock(return_value=candles)
    pool = MagicMock()
    async def _get(exchange_id, market_type="spot"):
        return spot_client if market_type == "spot" else perp_client
    pool.get = _get
    return pool, spot_client, perp_client


def _store_with_oi_history(history: list[float]):
    store = MagicMock()
    store.oi_history = AsyncMock(return_value=history)
    store.record_oi_observation = AsyncMock()
    store.prune_oi_history = AsyncMock(return_value=0)
    return store


# ── CascadeSignal ──


@pytest.mark.asyncio
async def test_cold_start_returns_zero():
    store = _store_with_oi_history([])    # no history yet
    pool, _, _ = _make_pool_for_signal(1e9, 0.0005, _candles_flat())
    signal = CascadeSignal(store, pool)
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    score = await signal(sym, executor=None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_short_history_returns_zero():
    # < MIN_OI_HISTORY_LEN observations
    short_hist = [1e9] * (MIN_OI_HISTORY_LEN - 1)
    store = _store_with_oi_history(short_hist)
    pool, _, _ = _make_pool_for_signal(1e9, 0.001, _candles_flat())
    signal = CascadeSignal(store, pool)
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    score = await signal(sym, executor=None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_high_oi_high_funding_compressed_vol_max_signal():
    # Current OI at top percentile, funding 0.001+, vol compressed
    # → all three signals at max → composite ≈ 1.0
    history = [1e9] * 100    # all observations at 1B; current 2B → 100th percentile
    store = _store_with_oi_history(history)
    pool, _, _ = _make_pool_for_signal(
        current_oi_usd=2e9,
        funding_8h=0.001,
        candles=_candles_compressed(),
    )
    signal = CascadeSignal(store, pool)
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    score = await signal(sym, executor=None)
    assert score >= 0.6    # well above the 0.6 trigger threshold


@pytest.mark.asyncio
async def test_low_oi_no_funding_normal_vol_zero_signal():
    # Current OI at low percentile, no funding, no compression
    history = [2e9] * 100    # all observations at 2B; current 1B → 0th percentile
    store = _store_with_oi_history(history)
    pool, _, _ = _make_pool_for_signal(
        current_oi_usd=1e9,
        funding_8h=0.0,
        candles=_candles_flat(),    # zero vol either way → ratio undefined → no compression
    )
    signal = CascadeSignal(store, pool)
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    score = await signal(sym, executor=None)
    assert score < 0.2


@pytest.mark.asyncio
async def test_funding_fetch_failure_returns_zero():
    store = _store_with_oi_history([1e9] * 100)
    pool, _, perp = _make_pool_for_signal(2e9, 0.001, _candles_flat())
    perp.get_funding_rate = AsyncMock(side_effect=Exception("network"))
    signal = CascadeSignal(store, pool)
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    score = await signal(sym, executor=None)
    assert score == 0.0


# ── OIObserver ──


@pytest.mark.asyncio
async def test_observer_records_oi_for_each_symbol():
    store = _store_with_oi_history([])
    pool, _, _ = _make_pool_for_signal(1.5e9, 0.0, _candles_flat())
    symbols = [
        SymbolConfig("binance", "BTC/USDT", "BTC"),
        SymbolConfig("binance", "ETH/USDT", "ETH"),
    ]
    observer = OIObserver(store, pool, symbols)
    result = await observer.run_once()
    assert result["recorded"] == 2
    assert store.record_oi_observation.await_count == 2


@pytest.mark.asyncio
async def test_observer_skips_zero_oi():
    store = _store_with_oi_history([])
    pool, _, perp = _make_pool_for_signal(0, 0.0, _candles_flat())
    symbols = [SymbolConfig("binance", "BTC/USDT", "BTC")]
    observer = OIObserver(store, pool, symbols)
    result = await observer.run_once()
    assert result["recorded"] == 0
    store.record_oi_observation.assert_not_called()


@pytest.mark.asyncio
async def test_observer_handles_fetch_failure_gracefully():
    store = _store_with_oi_history([])
    pool, _, perp = _make_pool_for_signal(1.5e9, 0.0, _candles_flat())
    perp.get_open_interest = AsyncMock(side_effect=Exception("network"))
    symbols = [
        SymbolConfig("binance", "BTC/USDT", "BTC"),
        SymbolConfig("binance", "ETH/USDT", "ETH"),
    ]
    observer = OIObserver(store, pool, symbols)
    result = await observer.run_once()
    assert result["recorded"] == 0    # both failed silently


@pytest.mark.asyncio
async def test_observer_uses_configured_perp_symbol_for_memecoin():
    """Regression: PEPE/SHIB/FLOKI/BONK perps live under a 1000-prefix
    on Binance. The OI observer must request the configured perp symbol,
    not naively append `:USDT` to the spot."""
    store = _store_with_oi_history([])
    pool, _, perp = _make_pool_for_signal(1.5e9, 0.0, _candles_flat())
    symbols = [
        SymbolConfig("binance", "PEPE/USDT", "PEPE",
                     perp_symbol_override="1000PEPE/USDT:USDT"),
    ]
    observer = OIObserver(store, pool, symbols)
    await observer.run_once()
    perp.get_open_interest.assert_awaited_once_with("1000PEPE/USDT:USDT")


@pytest.mark.asyncio
async def test_cascade_signal_uses_configured_perp_symbol_for_memecoin():
    """Regression: CascadeSignal must use the configured perp symbol
    when fetching OI and funding for memecoin pairs."""
    history = [1.0e9] * 30    # enough to clear MIN_OI_HISTORY_LEN
    store = _store_with_oi_history(history)
    pool, _, perp = _make_pool_for_signal(1.5e9, 0.0005, _candles_flat())
    signal = CascadeSignal(store, pool)
    sym = SymbolConfig("binance", "PEPE/USDT", "PEPE",
                       perp_symbol_override="1000PEPE/USDT:USDT")
    await signal(sym, executor=None)
    perp.get_open_interest.assert_awaited_once_with("1000PEPE/USDT:USDT")
    perp.get_funding_rate.assert_awaited_once_with("1000PEPE/USDT:USDT")


@pytest.mark.asyncio
async def test_observer_prunes_periodically():
    store = _store_with_oi_history([])
    pool, _, _ = _make_pool_for_signal(1.5e9, 0.0, _candles_flat())
    symbols = [SymbolConfig("binance", "BTC/USDT", "BTC")]
    # Force prune every 2 cycles for testability
    observer = OIObserver(store, pool, symbols, prune_every_n_cycles=2)
    await observer.run_once()    # cycle 1 — no prune
    store.prune_oi_history.assert_not_called()
    await observer.run_once()    # cycle 2 — prune fires
    store.prune_oi_history.assert_awaited_once()
