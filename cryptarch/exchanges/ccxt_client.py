"""CCXT-backed exchange client. Wraps ccxt.async_support for any of
~100 exchanges with a uniform interface.

Each exchange has quirks. We handle the common ones explicitly:
  - Binance: USDT-margined perps live under the `swap` market type
  - Bybit: unified-account vs spot/contract; symbol format BTC/USDT:USDT
  - OKX: separate trading-passphrase header
  - Coinbase: spot only; no perps
  - Deribit: options-first; perps and futures separately
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from cryptarch.exchanges.base import (
    Balance, Candle, ExchangeClient, FundingRate, OpenInterest, Ticker,
)
from cryptarch.sim.realistic import OrderBookLevel, OrderBookSnapshot

log = structlog.get_logger()


class CCXTClient(ExchangeClient):
    """Async ccxt-backed exchange client."""

    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        market_type: str = "spot",    # "spot" | "swap" (perps) | "future" | "option"
    ):
        self.name = exchange_id
        self._exchange_id = exchange_id
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._market_type = market_type
        self._client: Any = None

    async def start(self) -> None:
        # Lazy import so importing this module doesn't fail without ccxt.
        import ccxt.async_support as ccxt_async    # type: ignore
        ex_class = getattr(ccxt_async, self._exchange_id)
        config: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": self._market_type},
        }
        if self._api_key:
            config["apiKey"] = self._api_key
            config["secret"] = self._api_secret
        if self._api_passphrase:
            config["password"] = self._api_passphrase
        self._client = ex_class(config)
        await self._client.load_markets()
        log.info("ccxt_client_started",
                 exchange=self._exchange_id, market_type=self._market_type,
                 markets=len(self._client.markets))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBookSnapshot:
        raw = await self._client.fetch_order_book(symbol, limit=depth)
        return OrderBookSnapshot(
            asks=tuple(OrderBookLevel(price=p, size=s) for p, s in raw.get("asks", [])),
            bids=tuple(OrderBookLevel(price=p, size=s) for p, s in raw.get("bids", [])),
        )

    async def get_ticker(self, symbol: str) -> Ticker:
        raw = await self._client.fetch_ticker(symbol)
        return Ticker(
            exchange=self._exchange_id,
            symbol=symbol,
            bid=raw.get("bid"),
            ask=raw.get("ask"),
            last=raw.get("last"),
            fetched_at=datetime.now(timezone.utc),
        )

    async def get_funding_rate(self, symbol: str) -> FundingRate:
        raw = await self._client.fetch_funding_rate(symbol)
        rate = float(raw.get("fundingRate") or 0)
        next_ts = raw.get("nextFundingTimestamp")
        next_dt = (
            datetime.fromtimestamp(next_ts / 1000, tz=timezone.utc)
            if next_ts else datetime.now(timezone.utc)
        )
        return FundingRate(
            exchange=self._exchange_id,
            symbol=symbol,
            rate_8h=rate,
            next_funding_time=next_dt,
            fetched_at=datetime.now(timezone.utc),
        )

    async def get_open_interest(self, symbol: str) -> OpenInterest:
        raw = await self._client.fetch_open_interest(symbol)
        # CCXT normalizes to: {openInterestAmount (base), openInterestValue (USD), ...}
        oi_base = float(raw.get("openInterestAmount") or 0)
        oi_usd = float(raw.get("openInterestValue") or 0)
        # Some exchanges only return base; fall back to last price × base if needed.
        if oi_usd == 0 and oi_base > 0:
            try:
                ticker = await self._client.fetch_ticker(symbol)
                oi_usd = oi_base * float(ticker.get("last") or ticker.get("close") or 0)
            except Exception:
                pass
        return OpenInterest(
            exchange=self._exchange_id,
            symbol=symbol,
            oi_usd=oi_usd,
            oi_base=oi_base,
            fetched_at=datetime.now(timezone.utc),
        )

    async def get_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 168,
    ) -> list[Candle]:
        raw = await self._client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        candles: list[Candle] = []
        for row in raw or []:
            try:
                ts_ms, o, h, l, c, v = row[:6]
                candles.append(Candle(
                    ts=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                    open=float(o), high=float(h), low=float(l),
                    close=float(c), volume=float(v),
                ))
            except (ValueError, TypeError) as e:
                log.debug("ohlcv_row_skipped", error=str(e))
                continue
        return candles

    async def get_balance(self) -> Balance:
        raw = await self._client.fetch_balance()
        # USDC/USDT, whichever is present; sum them (treat as $1 each).
        cash = 0.0
        for stable in ("USDC", "USDT"):
            entry = raw.get(stable, {})
            if isinstance(entry, dict):
                cash += float(entry.get("free", 0) or 0)
        # Position notional comes from positions endpoint for perps.
        notional = 0.0
        if self._market_type == "swap":
            try:
                positions = await self._client.fetch_positions()
                for p in positions:
                    notional += abs(float(p.get("notional") or 0))
            except Exception as e:
                log.warning("fetch_positions_failed",
                            exchange=self._exchange_id, error=str(e))
        return Balance(
            exchange=self._exchange_id,
            cash_usd=cash,
            position_notional_usd=notional,
        )

    async def submit_limit_order(
        self, symbol: str, side: str, size_base: float,
        limit_price: float, post_only: bool = True,
        client_order_id: str | None = None,
    ) -> str:
        params: dict[str, Any] = {}
        if post_only:
            params["postOnly"] = True
        if client_order_id:
            params["clientOrderId"] = client_order_id
        order = await self._client.create_order(
            symbol=symbol, type="limit", side=side, amount=size_base,
            price=limit_price, params=params,
        )
        log.info("ccxt_limit_submitted", exchange=self._exchange_id,
                 symbol=symbol, side=side, size=size_base,
                 price=limit_price, order_id=order.get("id"))
        return str(order.get("id", ""))

    async def submit_market_order(
        self, symbol: str, side: str, size_base: float,
        client_order_id: str | None = None,
    ) -> str:
        params: dict[str, Any] = {}
        if client_order_id:
            params["clientOrderId"] = client_order_id
        order = await self._client.create_order(
            symbol=symbol, type="market", side=side, amount=size_base, params=params,
        )
        log.info("ccxt_market_submitted", exchange=self._exchange_id,
                 symbol=symbol, side=side, size=size_base,
                 order_id=order.get("id"))
        return str(order.get("id", ""))

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            await self._client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            log.warning("ccxt_cancel_failed", exchange=self._exchange_id,
                        order_id=order_id, error=str(e))
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> dict:
        return await self._client.fetch_order(order_id, symbol)
