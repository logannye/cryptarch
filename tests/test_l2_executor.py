"""L2Executor orchestration tests. Mocks store + exchange to verify the
ladder placement, fill detection, and TP/SL state machine without real
network or DB access."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptarch.core.config import Settings
from cryptarch.db.store import OpenPosition
from cryptarch.exchanges.base import Ticker
from cryptarch.sim.realistic import OrderBookLevel, OrderBookSnapshot
from cryptarch.strategies.l2_executor import L2Executor, SymbolConfig


class TestL2UniverseConcentration:
    """The L2 universe is concentrated on the symbols where 90-day backtest
    showed positive expected value: BTC/ETH/SOL as low-cost majors and
    PEPE/WIF as the high-vol memecoins that actually mean-revert. XRP,
    DOGE, SHIB, FLOKI, BONK were dropped because they showed neutral-to-
    negative cascade-capture P&L over the same window."""

    def test_universe_is_concentrated_to_five_symbols(self):
        from cryptarch.strategies.l2_executor import DEFAULT_SYMBOLS
        bases = {row[2] for row in DEFAULT_SYMBOLS}
        assert bases == {"BTC", "ETH", "SOL", "PEPE", "WIF"}

    def test_memecoins_carry_perp_symbol_override(self):
        from cryptarch.strategies.l2_executor import DEFAULT_SYMBOLS
        by_base = {row[2]: row for row in DEFAULT_SYMBOLS}
        # PEPE perp on Binance is listed as 1000PEPE — must override.
        assert by_base["PEPE"][3] == "1000PEPE/USDT:USDT"
        # WIF doesn't need an override (its spot and perp share the base).
        assert len(by_base["WIF"]) == 3


class TestSizingScalesWithBankroll:
    """Every dollar amount in the system is derived from bankroll × pct.
    A $50k bankroll moves 10× the notional of a $5k one, with zero code
    changes. Hardcoded $ values are an anti-pattern."""

    def test_l2_ladder_default_pct_is_12pct(self):
        """Smaller ladders (was 0.20) so multiple concurrent ladders fit
        without the deepest rungs hitting layer-cap rejection."""
        from cryptarch.core.config import Settings
        assert Settings(_env_file=None).l2_ladder_pct == 0.12

    def test_l2_ladder_total_scales_with_bankroll(self):
        from cryptarch.core.config import Settings
        assert Settings(_env_file=None, bankroll_usd=2_000.0).l2_ladder_total_usd == 240.0
        assert Settings(_env_file=None, bankroll_usd=5_000.0).l2_ladder_total_usd == 600.0
        assert Settings(_env_file=None, bankroll_usd=50_000.0).l2_ladder_total_usd == 6_000.0

    def test_alloc_layer_2_default_is_40pct(self):
        """L2 cap raised from 25% to 40% of bankroll. Lets multiple
        full ladders fit at the same time on a $5k+ bankroll instead
        of saturating after the first."""
        from cryptarch.core.config import Settings
        assert Settings(_env_file=None).alloc_layer_2_pct == 0.40

    def test_max_per_position_scales_with_bankroll(self):
        from cryptarch.core.config import Settings
        assert Settings(_env_file=None, bankroll_usd=2_000.0).max_per_position_usd == 500.0
        assert Settings(_env_file=None, bankroll_usd=50_000.0).max_per_position_usd == 12_500.0

    def test_l3_theta_budget_scales_with_bankroll(self):
        from cryptarch.core.config import Settings
        assert Settings(_env_file=None, bankroll_usd=2_000.0).l3_daily_theta_budget_usd == 10.0
        assert Settings(_env_file=None, bankroll_usd=50_000.0).l3_daily_theta_budget_usd == 250.0

    def test_min_position_scales_with_bankroll(self):
        """The minimum-trade-size floor (was hardcoded $20 in executors) is
        also bankroll-relative — at $50k bankroll, $20 fills are noise."""
        from cryptarch.core.config import Settings
        assert Settings(_env_file=None, bankroll_usd=2_000.0).min_position_usd == 20.0
        assert Settings(_env_file=None, bankroll_usd=50_000.0).min_position_usd == 500.0


class TestSymbolConfigPerpSymbol:
    """The L2 stack converts a spot symbol to its corresponding perp for
    OI and funding-rate fetches. For most pairs this is just appending
    `:USDT`, but Binance lists 1000-multiple memecoin perps under a
    different base (e.g. PEPE/USDT spot ↔ 1000PEPE/USDT:USDT perp),
    which the naive `.replace()` mapping silently breaks."""

    def test_default_appends_usdt_suffix(self):
        sym = SymbolConfig("binance", "BTC/USDT", "BTC")
        assert sym.perp_symbol == "BTC/USDT:USDT"

    def test_memecoin_override_uses_explicit_perp_symbol(self):
        sym = SymbolConfig(
            "binance", "PEPE/USDT", "PEPE",
            perp_symbol_override="1000PEPE/USDT:USDT",
        )
        assert sym.perp_symbol == "1000PEPE/USDT:USDT"


def _settings(**overrides) -> Settings:
    base = dict(
        bankroll_usd=2000.0,
        alloc_layer_1_pct=0.60,
        alloc_layer_2_pct=0.25,
        alloc_layer_3_pct=0.15,
        max_total_deployed_pct=0.50,
        max_per_position_pct=0.25,
        enable_live_orders=False,
        layer_2_cascade_capture_enabled=True,
        l2_ladder_levels=4,
        l2_ladder_pct=0.10,    # 10% of $2k = $200 for the legacy test scenario
        l2_take_profit_pct=0.012,
        l2_stop_loss_pct=0.030,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _book(asks: list[tuple[float, float]], bids: list[tuple[float, float]]) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        asks=tuple(OrderBookLevel(p, s) for p, s in asks),
        bids=tuple(OrderBookLevel(p, s) for p, s in bids),
    )


def _make_pool(spot_price: float, book: OrderBookSnapshot | None = None):
    book = book or _book(
        asks=[(spot_price * 1.0001, 1000)],
        bids=[(spot_price * 0.9999, 1000)],
    )
    client = MagicMock()
    client.get_ticker = AsyncMock(return_value=Ticker(
        exchange="binance", symbol="BTC/USDT",
        bid=spot_price - 1, ask=spot_price + 1, last=spot_price,
        fetched_at=datetime.now(timezone.utc),
    ))
    client.get_order_book = AsyncMock(return_value=book)
    pool = MagicMock()
    async def _get(exchange_id, market_type="spot"):
        return client
    pool.get = _get
    return pool, client


def _make_store(open_positions=None, total_at_risk=0.0, layer_deployed=0.0,
                state_halt_reason=None):
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
    store.create_position = AsyncMock(side_effect=lambda **kw: 100 + kw.get("notional_usd", 0))
    store.mark_position_open = AsyncMock()
    store.close_position = AsyncMock()
    store.record_fill = AsyncMock(return_value=1)
    store._pool = MagicMock()
    store._pool.execute = AsyncMock()
    return store


# ── ladder config snapshot + stale-refresh ──


class TestLadderConfigSnapshot:
    """When systemic strategy parameters (ladder size, levels, bankroll)
    change, opening rungs from the prior config become structurally stale.
    The bot must detect this and refresh them so all live ladders reflect
    the current strategy."""

    def test_snapshot_captures_design_relevant_settings(self):
        from cryptarch.strategies.l2_cascade import LadderDesignSnapshot
        s = _settings(bankroll_usd=5000.0, l2_ladder_pct=0.12, l2_ladder_levels=4)
        snap = LadderDesignSnapshot.from_settings(s)
        assert snap.bankroll_usd == 5000.0
        assert snap.l2_ladder_pct == 0.12
        assert snap.l2_ladder_levels == 4

    def test_snapshot_round_trips_through_dict(self):
        from cryptarch.strategies.l2_cascade import LadderDesignSnapshot
        original = LadderDesignSnapshot(
            bankroll_usd=5000.0, l2_ladder_pct=0.12, l2_ladder_levels=4,
        )
        roundtripped = LadderDesignSnapshot.from_dict(original.to_dict())
        assert roundtripped == original

    def test_snapshot_from_missing_dict_is_none(self):
        from cryptarch.strategies.l2_cascade import LadderDesignSnapshot
        assert LadderDesignSnapshot.from_dict(None) is None
        assert LadderDesignSnapshot.from_dict({}) is None

    def test_snapshot_from_partial_dict_is_none(self):
        """Legacy positions saved before this feature have no snapshot —
        from_dict returns None to signal 'unknown vintage' so the refresh
        loop treats them as stale."""
        from cryptarch.strategies.l2_cascade import LadderDesignSnapshot
        assert LadderDesignSnapshot.from_dict({"l2_ladder_pct": 0.20}) is None


@pytest.mark.asyncio
async def test_placed_ladder_records_config_snapshot_in_metadata():
    """Each rung must carry the design-time config so we can detect later
    if it has gone stale relative to the live config."""
    settings = _settings(bankroll_usd=5000.0, l2_ladder_pct=0.12)
    pool, _ = _make_pool(spot_price=100.0)
    store = _make_store()
    create_calls: list[dict] = []
    async def capture_create(**kw):
        create_calls.append(kw)
        return 100 + kw.get("notional_usd", 0)
    store.create_position = AsyncMock(side_effect=capture_create)

    async def signal_high(symbol, executor):
        return 0.7
    executor = L2Executor(
        settings, store, pool,
        symbols=[SymbolConfig("binance", "BTC/USDT", "BTC")],
        signal_fn=signal_high,
    )
    await executor.run_once()
    assert create_calls, "expected at least one rung placed"
    for call in create_calls:
        snap = call["metadata"].get("config_snapshot")
        assert snap is not None
        assert snap["bankroll_usd"] == 5000.0
        assert snap["l2_ladder_pct"] == 0.12
        assert snap["l2_ladder_levels"] == 4


@pytest.mark.asyncio
async def test_opening_rung_with_stale_snapshot_is_refreshed():
    """A rung designed under l2_ladder_pct=0.20 but running against current
    l2_ladder_pct=0.12 should be cancelled at $0 P&L. Next cycle the
    executor will re-evaluate the symbol and design a fresh ladder."""
    settings = _settings(bankroll_usd=5000.0, l2_ladder_pct=0.12)
    pool, _ = _make_pool(spot_price=100.0)
    stale_pos = OpenPosition(
        id=42, layer="l2_cascade", strategy_group_id="g1", state="opening",
        notional_usd=100.0, realized_pnl_usd=0.0, opened_at=datetime.now(timezone.utc),
        metadata={
            "symbol_key": "binance:BTC/USDT",
            "symbol": "BTC/USDT",
            "limit_price": 99.0,
            "size_usd": 100.0,
            "spot_at_design": 100.0,
            "config_snapshot": {
                "bankroll_usd": 5000.0,
                "l2_ladder_pct": 0.20,    # ← stale (current is 0.12)
                "l2_ladder_levels": 4,
            },
        },
    )
    store = _make_store(open_positions=[stale_pos])
    async def signal_low(symbol, executor):
        return 0.3    # below trigger so no new placement
    executor = L2Executor(
        settings, store, pool,
        symbols=[SymbolConfig("binance", "BTC/USDT", "BTC")],
        signal_fn=signal_low,
    )
    await executor.run_once()
    store.close_position.assert_awaited_with(42, realized_pnl=0.0)


@pytest.mark.asyncio
async def test_opening_rung_with_missing_snapshot_is_refreshed():
    """Legacy positions placed before the snapshot feature existed have
    no config_snapshot in metadata — they should be treated as stale and
    refreshed on the next cycle."""
    settings = _settings(bankroll_usd=5000.0, l2_ladder_pct=0.12)
    pool, _ = _make_pool(spot_price=100.0)
    legacy_pos = OpenPosition(
        id=99, layer="l2_cascade", strategy_group_id="legacy", state="opening",
        notional_usd=100.0, realized_pnl_usd=0.0, opened_at=datetime.now(timezone.utc),
        metadata={
            "symbol_key": "binance:BTC/USDT",
            "symbol": "BTC/USDT",
            "limit_price": 99.0,
            "size_usd": 100.0,
            "spot_at_design": 100.0,
            # no config_snapshot key at all
        },
    )
    store = _make_store(open_positions=[legacy_pos])
    async def signal_low(symbol, executor):
        return 0.3
    executor = L2Executor(
        settings, store, pool,
        symbols=[SymbolConfig("binance", "BTC/USDT", "BTC")],
        signal_fn=signal_low,
    )
    await executor.run_once()
    store.close_position.assert_awaited_with(99, realized_pnl=0.0)


@pytest.mark.asyncio
async def test_opening_rung_with_matching_snapshot_is_not_refreshed():
    """Rungs whose snapshot matches current config must be left alone — no
    cancellation should fire just because we ran the refresh check."""
    settings = _settings(bankroll_usd=5000.0, l2_ladder_pct=0.12)
    pool, _ = _make_pool(spot_price=100.0)
    fresh_pos = OpenPosition(
        id=7, layer="l2_cascade", strategy_group_id="g1", state="opening",
        notional_usd=100.0, realized_pnl_usd=0.0, opened_at=datetime.now(timezone.utc),
        metadata={
            "symbol_key": "binance:BTC/USDT",
            "symbol": "BTC/USDT",
            "limit_price": 99.0,
            "size_usd": 100.0,
            "spot_at_design": 100.0,
            "rung_idx": 0,
            "rung_pct_below": 0.01,
            "config_snapshot": {
                "bankroll_usd": 5000.0,
                "l2_ladder_pct": 0.12,
                "l2_ladder_levels": 4,
            },
        },
    )
    # signal at 0.5 — above hold threshold so won't be cancelled for that
    # reason either, leaving only "stale config" as a possible reason.
    store = _make_store(open_positions=[fresh_pos])
    async def signal_mid(symbol, executor):
        return 0.5
    executor = L2Executor(
        settings, store, pool,
        symbols=[SymbolConfig("binance", "BTC/USDT", "BTC")],
        signal_fn=signal_mid,
    )
    await executor.run_once()
    store.close_position.assert_not_awaited()


@pytest.mark.asyncio
async def test_filled_rung_is_not_refreshed_even_if_snapshot_stale():
    """Once a rung is filled (state='open'), its size is established —
    we can't re-deploy capital that's already in a position. The refresh
    is for opening (pending) rungs only."""
    settings = _settings(bankroll_usd=5000.0, l2_ladder_pct=0.12)
    pool, _ = _make_pool(spot_price=100.0)
    filled_pos = OpenPosition(
        id=11, layer="l2_cascade", strategy_group_id="g1", state="open",
        notional_usd=100.0, realized_pnl_usd=0.0, opened_at=datetime.now(timezone.utc),
        metadata={
            "symbol_key": "binance:BTC/USDT",
            "symbol": "BTC/USDT",
            "fill_price": 99.0,
            "tp_price": 100.18,
            "sl_price": 96.03,
            "size_usd": 100.0,
            "filled_at": datetime.now(timezone.utc).isoformat(),
            "config_snapshot": {
                "bankroll_usd": 2000.0,    # stale — bankroll has since changed
                "l2_ladder_pct": 0.20,
                "l2_ladder_levels": 4,
            },
        },
    )
    store = _make_store(open_positions=[filled_pos])
    async def signal_low(symbol, executor):
        return 0.3
    executor = L2Executor(
        settings, store, pool,
        symbols=[SymbolConfig("binance", "BTC/USDT", "BTC")],
        signal_fn=signal_low,
    )
    await executor.run_once()
    # close_position may be called for OTHER reasons (TP/SL/timestop) inside
    # _manage_filled_position, but NOT with realized_pnl=0.0 from the refresh.
    # The refresh path is the only one that closes opening positions at $0.
    refresh_calls = [
        c for c in store.close_position.await_args_list
        if c.kwargs.get("realized_pnl") == 0.0 or (c.args and c.args[1] == 0.0)
    ]
    # Refresh would close at $0; filled positions must not be among those.
    assert all(
        not (c.args and c.args[0] == 11) for c in refresh_calls
    ), f"filled position 11 was refresh-closed: {refresh_calls}"


# ── trigger logic ──


@pytest.mark.asyncio
async def test_high_signal_places_ladder():
    settings = _settings()
    pool, _ = _make_pool(spot_price=50000)
    store = _make_store()
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    executor = L2Executor(
        settings, store, pool, symbols=[sym],
        signal_fn=lambda s, e: _const(0.85),    # strong signal
    )
    result = await executor.run_once()
    assert result["new_ladders"] == 1
    # 4 rungs created
    assert store.create_position.await_count == 4


@pytest.mark.asyncio
async def test_low_signal_does_nothing():
    settings = _settings()
    pool, _ = _make_pool(spot_price=50000)
    store = _make_store()
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    executor = L2Executor(
        settings, store, pool, symbols=[sym],
        signal_fn=lambda s, e: _const(0.30),    # below threshold
    )
    result = await executor.run_once()
    assert result["new_ladders"] == 0
    store.create_position.assert_not_called()


# ── existing-ladder dedup ──


@pytest.mark.asyncio
async def test_does_not_place_second_ladder_for_same_symbol():
    settings = _settings()
    pool, _ = _make_pool(spot_price=50000)
    existing = OpenPosition(
        id=1, layer="l2_cascade", strategy_group_id="g1",
        state="opening", notional_usd=50.0, realized_pnl_usd=0,
        opened_at=datetime.now(timezone.utc),
        metadata={"symbol_key": "binance:BTC/USDT", "limit_price": 49500,
                  "size_usd": 50, "rung_idx": 0,
                  "spot_at_design": 50000},
    )
    store = _make_store(open_positions=[existing])
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    executor = L2Executor(
        settings, store, pool, symbols=[sym],
        signal_fn=lambda s, e: _const(0.85),
    )
    result = await executor.run_once()
    assert result["new_ladders"] == 0


# ── pending rung fill ──


@pytest.mark.asyncio
async def test_pending_rung_fills_when_book_crosses():
    settings = _settings()
    # Book has best_ask at $49,500 — equals our rung's limit
    book = _book(asks=[(49500, 1000)], bids=[(49490, 1000)])
    pool, _ = _make_pool(spot_price=49500, book=book)

    pending = OpenPosition(
        id=1, layer="l2_cascade", strategy_group_id="g1",
        state="opening", notional_usd=50.0, realized_pnl_usd=0,
        opened_at=datetime.now(timezone.utc),
        metadata={
            "symbol_key": "binance:BTC/USDT",
            "exchange": "binance", "symbol": "BTC/USDT",
            "limit_price": 49500, "size_usd": 50,
            "rung_idx": 0, "spot_at_design": 50000,
        },
    )
    store = _make_store(open_positions=[pending])
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    executor = L2Executor(
        settings, store, pool, symbols=[sym],
        signal_fn=lambda s, e: _const(0.85),
    )
    result = await executor.run_once()
    assert result["rungs_filled"] == 1
    store.mark_position_open.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_pending_rung_holds_when_book_above():
    settings = _settings()
    # Best ask far above our limit — won't fill
    book = _book(asks=[(50000, 1000)], bids=[(49500, 1000)])
    pool, _ = _make_pool(spot_price=50000, book=book)

    pending = OpenPosition(
        id=1, layer="l2_cascade", strategy_group_id="g1",
        state="opening", notional_usd=50.0, realized_pnl_usd=0,
        opened_at=datetime.now(timezone.utc),
        metadata={
            "symbol_key": "binance:BTC/USDT",
            "exchange": "binance", "symbol": "BTC/USDT",
            "limit_price": 49000, "size_usd": 50,    # 1k below ask
            "rung_idx": 1, "spot_at_design": 50000,
        },
    )
    store = _make_store(open_positions=[pending])
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    executor = L2Executor(
        settings, store, pool, symbols=[sym],
        signal_fn=lambda s, e: _const(0.85),
    )
    result = await executor.run_once()
    assert result["rungs_filled"] == 0
    store.mark_position_open.assert_not_called()


@pytest.mark.asyncio
async def test_pending_rung_cancels_when_signal_weakens():
    settings = _settings()
    book = _book(asks=[(50000, 1000)], bids=[(49500, 1000)])
    pool, _ = _make_pool(spot_price=50000, book=book)

    pending = OpenPosition(
        id=1, layer="l2_cascade", strategy_group_id="g1",
        state="opening", notional_usd=50.0, realized_pnl_usd=0,
        opened_at=datetime.now(timezone.utc),
        metadata={
            "symbol_key": "binance:BTC/USDT",
            "exchange": "binance", "symbol": "BTC/USDT",
            "limit_price": 49000, "size_usd": 50, "rung_idx": 1,
            "spot_at_design": 50000,
            # Current snapshot so the refresh path leaves this position
            # alone and the signal-weakening cancellation can fire.
            "config_snapshot": {
                "bankroll_usd": settings.bankroll_usd,
                "l2_ladder_pct": settings.l2_ladder_pct,
                "l2_ladder_levels": settings.l2_ladder_levels,
            },
        },
    )
    store = _make_store(open_positions=[pending])
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    executor = L2Executor(
        settings, store, pool, symbols=[sym],
        signal_fn=lambda s, e: _const(0.20),    # signal weakened
    )
    result = await executor.run_once()
    assert result["positions_closed"] == 1
    store.close_position.assert_awaited_once_with(1, realized_pnl=0.0)


# ── filled position TP/SL/time-stop ──


@pytest.mark.asyncio
async def test_filled_position_takes_profit():
    settings = _settings()
    # Fill price was 50000; TP = 50000 × 1.012 = 50600; bid > 50600
    book = _book(asks=[(50700, 1000)], bids=[(50650, 1000)])
    pool, _ = _make_pool(spot_price=50650, book=book)

    filled = OpenPosition(
        id=1, layer="l2_cascade", strategy_group_id="g1",
        state="open", notional_usd=50.0, realized_pnl_usd=0,
        opened_at=datetime.now(timezone.utc),
        metadata={
            "symbol_key": "binance:BTC/USDT",
            "exchange": "binance", "symbol": "BTC/USDT",
            "limit_price": 50000, "size_usd": 50, "rung_idx": 0,
            "spot_at_design": 50000,
            "fill_price": 50000, "tp_price": 50600, "sl_price": 48500,
            "filled_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    store = _make_store(open_positions=[filled])
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    executor = L2Executor(settings, store, pool, symbols=[sym],
                         signal_fn=lambda s, e: _const(0.5))
    result = await executor.run_once()
    assert result["positions_closed"] == 1
    # close_position called with positive PnL
    args = store.close_position.await_args
    assert args.kwargs["realized_pnl"] > 0


@pytest.mark.asyncio
async def test_filled_position_stops_loss():
    settings = _settings()
    # Bid at 48400 < SL of 48500 → stop-loss fires
    book = _book(asks=[(48450, 1000)], bids=[(48400, 1000)])
    pool, _ = _make_pool(spot_price=48400, book=book)

    filled = OpenPosition(
        id=1, layer="l2_cascade", strategy_group_id="g1",
        state="open", notional_usd=50.0, realized_pnl_usd=0,
        opened_at=datetime.now(timezone.utc),
        metadata={
            "symbol_key": "binance:BTC/USDT",
            "exchange": "binance", "symbol": "BTC/USDT",
            "limit_price": 50000, "size_usd": 50, "rung_idx": 0,
            "spot_at_design": 50000,
            "fill_price": 50000, "tp_price": 50600, "sl_price": 48500,
            "filled_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    store = _make_store(open_positions=[filled])
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    executor = L2Executor(settings, store, pool, symbols=[sym],
                         signal_fn=lambda s, e: _const(0.5))
    result = await executor.run_once()
    assert result["positions_closed"] == 1
    args = store.close_position.await_args
    assert args.kwargs["realized_pnl"] < 0


@pytest.mark.asyncio
async def test_filled_position_time_stops():
    settings = _settings()
    # Bid hovering near fill → no TP/SL trigger; but age > 60min → time-stop
    book = _book(asks=[(50050, 1000)], bids=[(50000, 1000)])
    pool, _ = _make_pool(spot_price=50000, book=book)

    filled = OpenPosition(
        id=1, layer="l2_cascade", strategy_group_id="g1",
        state="open", notional_usd=50.0, realized_pnl_usd=0,
        opened_at=datetime.now(timezone.utc) - timedelta(minutes=90),
        metadata={
            "symbol_key": "binance:BTC/USDT",
            "exchange": "binance", "symbol": "BTC/USDT",
            "limit_price": 50000, "size_usd": 50, "rung_idx": 0,
            "spot_at_design": 50000,
            "fill_price": 50000, "tp_price": 50600, "sl_price": 48500,
            "filled_at": (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat(),
        },
    )
    store = _make_store(open_positions=[filled])
    sym = SymbolConfig("binance", "BTC/USDT", "BTC")
    executor = L2Executor(settings, store, pool, symbols=[sym],
                         signal_fn=lambda s, e: _const(0.5))
    result = await executor.run_once()
    assert result["positions_closed"] == 1


# ── disabled / halted ──


@pytest.mark.asyncio
async def test_disabled_layer_skips():
    settings = _settings(layer_2_cascade_capture_enabled=False)
    pool = MagicMock()
    store = _make_store()
    executor = L2Executor(settings, store, pool, symbols=[])
    result = await executor.run_once()
    assert result == {"skipped": "layer_disabled"}


@pytest.mark.asyncio
async def test_halted_state_skips():
    settings = _settings()
    pool = MagicMock()
    store = _make_store(state_halt_reason="manual halt")
    executor = L2Executor(settings, store, pool, symbols=[])
    result = await executor.run_once()
    assert result["skipped"] == "halted"


# ── helpers ──


async def _const(v: float) -> float:
    return v


# Adapter so `signal_fn=lambda s, e: _const(0.85)` works (the lambda
# captures the awaitable).
class _ConstSignal:
    def __init__(self, value: float):
        self.value = value
    async def __call__(self, sym, exec):
        return self.value
