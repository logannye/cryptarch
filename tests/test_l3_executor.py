"""L3Executor orchestration tests. Mocks store + exchange pool + Deribit
chain so we verify the strangle lifecycle (open / hold / roll) without
real network access."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptarch.core.config import Settings
from cryptarch.db.store import OpenPosition
from cryptarch.exchanges.base import Ticker
from cryptarch.exchanges.deribit_options import OptionInstrument, OptionQuote
from cryptarch.strategies.l3_executor import L3Executor, UnderlyingConfig


def _settings(**overrides) -> Settings:
    base = dict(
        bankroll_usd=2000.0,
        alloc_layer_1_pct=0.60,
        alloc_layer_2_pct=0.25,
        alloc_layer_3_pct=0.15,
        max_total_deployed_pct=0.50,
        max_per_position_usd=500.0,
        enable_live_orders=False,
        layer_3_tail_hedge_enabled=True,
        l3_daily_theta_budget_usd=10.0,
        l3_target_days_to_expiry=45,
        l3_otm_pct=0.20,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _option(strike: float, opt_type: str, expiry: datetime) -> OptionInstrument:
    return OptionInstrument(
        instrument_name=f"BTC-{expiry.strftime('%d%b%y').upper()}-{int(strike)}-{opt_type}",
        underlying="BTC",
        expiry=expiry,
        strike=strike,
        option_type=opt_type,
    )


def _make_pool_for_l3(
    spot_price: float = 50000.0,
    instruments: list[OptionInstrument] | None = None,
    call_quote: OptionQuote | None = None,
    put_quote: OptionQuote | None = None,
):
    """Build an ExchangePool mock that returns:
       - spot ticker for binance/spot
       - Deribit option chain via DeribitOptionsClient
    """
    spot_client = MagicMock()
    spot_client.get_ticker = AsyncMock(return_value=Ticker(
        exchange="binance", symbol="BTC/USDT",
        bid=spot_price - 5, ask=spot_price + 5, last=spot_price,
        fetched_at=datetime.now(timezone.utc),
    ))

    # The DeribitOptionsClient wraps a CCXTClient. We mock the wrapper
    # methods directly via patching the DeribitOptionsClient class.
    deribit_inner = MagicMock()
    deribit_inner._client = MagicMock()    # CCXTClient internal

    pool = MagicMock()
    async def _get(exchange_id, market_type="spot"):
        if exchange_id == "deribit":
            return deribit_inner
        return spot_client
    pool.get = _get
    return pool, spot_client, deribit_inner, instruments, call_quote, put_quote


def _store():
    store = MagicMock()
    store.get_system_state = AsyncMock(return_value=MagicMock(halt_reason=None))
    store.open_positions = AsyncMock(return_value=[])
    store.create_position = AsyncMock(return_value=42)
    store.mark_position_open = AsyncMock()
    store.close_position = AsyncMock()
    store.record_fill = AsyncMock(return_value=1)
    return store


# ── disabled / halted ──


@pytest.mark.asyncio
async def test_disabled_layer_no_op():
    settings = _settings(layer_3_tail_hedge_enabled=False)
    pool, *_ = _make_pool_for_l3()
    store = _store()
    exec = L3Executor(settings, store, pool)
    result = await exec.run_once()
    assert result == {"skipped": "layer_disabled"}


@pytest.mark.asyncio
async def test_halted_state_skips():
    settings = _settings()
    pool, *_ = _make_pool_for_l3()
    store = _store()
    store.get_system_state = AsyncMock(return_value=MagicMock(halt_reason="manual"))
    exec = L3Executor(settings, store, pool)
    result = await exec.run_once()
    assert result["skipped"] == "halted"


# ── open new strangle ──


@pytest.mark.asyncio
async def test_opens_strangle_when_none_held():
    settings = _settings()
    spot = 50000.0
    expiry = datetime.now(timezone.utc) + timedelta(days=45)
    instruments = [
        _option(40000.0, "P", expiry),
        _option(45000.0, "P", expiry),
        _option(55000.0, "C", expiry),
        _option(60000.0, "C", expiry),
    ]
    pool, *_  = _make_pool_for_l3(spot_price=spot, instruments=instruments)
    store = _store()

    # Patch DeribitOptionsClient to return our test data.
    with patch("cryptarch.strategies.l3_executor.DeribitOptionsClient") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.list_instruments = AsyncMock(return_value=instruments)
        mock_inst.filter_by_expiry = AsyncMock(side_effect=lambda ins, exp: [
            i for i in ins if i.expiry.date() == exp.date()
        ])
        mock_inst.get_quote = AsyncMock(return_value=OptionQuote(
            instrument=instruments[0],    # placeholder
            bid_usd=1500.0, ask_usd=1500.0, mark_iv=0.65,
            fetched_at=datetime.now(timezone.utc),
        ))
        # find_closest_strike is a static method — keep the real one
        from cryptarch.exchanges.deribit_options import DeribitOptionsClient
        mock_cls.find_closest_strike = DeribitOptionsClient.find_closest_strike
        mock_cls.return_value = mock_inst

        exec_ = L3Executor(settings, store, pool)
        result = await exec_.run_once()

    assert result["opened"] == 1
    store.create_position.assert_awaited_once()
    # Two fills recorded: call + put
    assert store.record_fill.await_count == 2
    store.mark_position_open.assert_awaited_once_with(42)


@pytest.mark.asyncio
async def test_skips_when_quotes_missing():
    settings = _settings()
    expiry = datetime.now(timezone.utc) + timedelta(days=45)
    instruments = [
        _option(40000.0, "P", expiry),
        _option(60000.0, "C", expiry),
    ]
    pool, *_ = _make_pool_for_l3(spot_price=50000.0)
    store = _store()
    with patch("cryptarch.strategies.l3_executor.DeribitOptionsClient") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.list_instruments = AsyncMock(return_value=instruments)
        mock_inst.filter_by_expiry = AsyncMock(return_value=instruments)
        # No quote available
        mock_inst.get_quote = AsyncMock(return_value=OptionQuote(
            instrument=instruments[0],
            bid_usd=None, ask_usd=None, mark_iv=None,
            fetched_at=datetime.now(timezone.utc),
        ))
        from cryptarch.exchanges.deribit_options import DeribitOptionsClient
        mock_cls.find_closest_strike = DeribitOptionsClient.find_closest_strike
        mock_cls.return_value = mock_inst

        exec_ = L3Executor(settings, store, pool)
        result = await exec_.run_once()

    assert result["opened"] == 0
    store.create_position.assert_not_called()


@pytest.mark.asyncio
async def test_skips_when_no_instruments():
    settings = _settings()
    pool, *_ = _make_pool_for_l3()
    store = _store()
    with patch("cryptarch.strategies.l3_executor.DeribitOptionsClient") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.list_instruments = AsyncMock(return_value=[])
        mock_cls.return_value = mock_inst
        exec_ = L3Executor(settings, store, pool)
        result = await exec_.run_once()
    assert result["opened"] == 0


# ── hold ──


@pytest.mark.asyncio
async def test_holds_when_strangle_has_runway():
    settings = _settings()
    far_expiry = datetime.now(timezone.utc) + timedelta(days=45)
    held = OpenPosition(
        id=1, layer="l3_tail", strategy_group_id="g1",
        state="open", notional_usd=1500.0, realized_pnl_usd=0,
        opened_at=datetime.now(timezone.utc),
        metadata={
            "underlying": "BTC",
            "expiry_iso": far_expiry.isoformat(),
        },
    )
    pool, *_ = _make_pool_for_l3()
    store = _store()
    store.open_positions = AsyncMock(return_value=[held])
    exec_ = L3Executor(settings, store, pool)
    result = await exec_.run_once()
    assert result["opened"] == 0
    assert result["rolled"] == 0


# ── roll ──


@pytest.mark.asyncio
async def test_rolls_when_dte_below_threshold():
    settings = _settings()
    near_expiry = datetime.now(timezone.utc) + timedelta(days=20)    # < 30 day threshold
    held = OpenPosition(
        id=1, layer="l3_tail", strategy_group_id="g1",
        state="open", notional_usd=1500.0, realized_pnl_usd=0,
        opened_at=datetime.now(timezone.utc),
        metadata={
            "underlying": "BTC",
            "expiry_iso": near_expiry.isoformat(),
        },
    )
    new_expiry = datetime.now(timezone.utc) + timedelta(days=45)
    new_instruments = [
        _option(40000.0, "P", new_expiry),
        _option(60000.0, "C", new_expiry),
    ]
    pool, *_ = _make_pool_for_l3(spot_price=50000.0)
    store = _store()
    store.open_positions = AsyncMock(return_value=[held])

    with patch("cryptarch.strategies.l3_executor.DeribitOptionsClient") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.list_instruments = AsyncMock(return_value=new_instruments)
        mock_inst.filter_by_expiry = AsyncMock(return_value=new_instruments)
        mock_inst.get_quote = AsyncMock(return_value=OptionQuote(
            instrument=new_instruments[0],
            bid_usd=1500.0, ask_usd=1500.0, mark_iv=0.65,
            fetched_at=datetime.now(timezone.utc),
        ))
        from cryptarch.exchanges.deribit_options import DeribitOptionsClient
        mock_cls.find_closest_strike = DeribitOptionsClient.find_closest_strike
        mock_cls.return_value = mock_inst

        exec_ = L3Executor(settings, store, pool)
        result = await exec_.run_once()

    assert result["rolled"] == 1
    store.close_position.assert_awaited_once()    # closed the old one
    # And opened a new one (create_position called)
    store.create_position.assert_awaited_once()
