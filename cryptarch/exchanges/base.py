"""Provider-agnostic exchange interface.

We use ccxt-async under the hood for portability across 100+ exchanges,
but the bot codes against this abstract interface so we can swap or mock
any specific exchange (e.g. for tests, or to replace ccxt with native
clients later for latency-critical paths).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from cryptarch.sim.realistic import OrderBookSnapshot


@dataclass(frozen=True)
class FundingRate:
    """A single funding-rate observation for a perp."""
    exchange: str
    symbol: str             # e.g. "BTC/USDT:USDT"
    rate_8h: float          # current funding rate, 8h period (positive = longs pay)
    next_funding_time: datetime
    fetched_at: datetime


@dataclass(frozen=True)
class OpenInterest:
    """Current open-interest snapshot for a perp."""
    exchange: str
    symbol: str
    oi_usd: float           # USD-denominated open interest
    oi_base: float          # native (base-token-denominated) OI when available
    fetched_at: datetime


@dataclass(frozen=True)
class Candle:
    """One OHLCV candle."""
    ts: datetime            # candle close time
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Ticker:
    """Quick bid/ask/last summary."""
    exchange: str
    symbol: str
    bid: float | None
    ask: float | None
    last: float | None
    fetched_at: datetime


@dataclass(frozen=True)
class Balance:
    """USDC/USDT balance + open-position notional on one exchange."""
    exchange: str
    cash_usd: float
    position_notional_usd: float


class ExchangeClient(ABC):
    """Async exchange interface. Every concrete exchange must implement
    these methods. Read-only methods come first; write methods are gated
    by the live-orders safeguard."""

    name: str

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    # ── read-only ──

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBookSnapshot:
        """Fetch order book snapshot. Used by simulator + live-fill checks."""

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker: ...

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> FundingRate:
        """Current 8h funding rate for a perp symbol. Raises if not a perp."""

    @abstractmethod
    async def get_open_interest(self, symbol: str) -> OpenInterest:
        """Current open-interest snapshot. Raises if not a perp."""

    @abstractmethod
    async def get_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 168,
    ) -> list[Candle]:
        """Historical candles. Default = 7 days of hourly bars."""

    @abstractmethod
    async def get_balance(self) -> Balance: ...

    # ── write (gated) ──

    @abstractmethod
    async def submit_limit_order(
        self, symbol: str, side: str, size_base: float,
        limit_price: float, post_only: bool = True,
        client_order_id: str | None = None,
    ) -> str:
        """Submit a limit order. Returns the exchange's order id."""

    @abstractmethod
    async def submit_market_order(
        self, symbol: str, side: str, size_base: float,
        client_order_id: str | None = None,
    ) -> str: ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool: ...

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> dict: ...
