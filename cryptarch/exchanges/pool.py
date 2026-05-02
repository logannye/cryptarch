"""Pool of exchange clients keyed by (exchange_id, market_type).

Strategy code asks the pool for a client; the pool lazily creates and
caches them. On engine shutdown, all clients close cleanly.

Concurrency: a per-key asyncio.Lock prevents the thundering herd where
N parallel `get()` calls each create a fresh client before the first
one finishes loading markets.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

from cryptarch.core.config import Settings
from cryptarch.exchanges.base import ExchangeClient
from cryptarch.exchanges.ccxt_client import CCXTClient

log = structlog.get_logger()


class ExchangePool:
    """Lazy-initialized client pool with per-key locking."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._clients: dict[tuple[str, str], ExchangeClient] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    def _credentials_for(self, exchange_id: str) -> dict[str, str]:
        s = self._settings
        if exchange_id in ("kraken", "krakenfutures"):
            return {"api_key": s.kraken_api_key, "api_secret": s.kraken_api_secret}
        if exchange_id == "binance":
            return {"api_key": s.binance_api_key, "api_secret": s.binance_api_secret}
        if exchange_id == "bybit":
            return {"api_key": s.bybit_api_key, "api_secret": s.bybit_api_secret}
        if exchange_id == "okx":
            return {
                "api_key": s.okx_api_key, "api_secret": s.okx_api_secret,
                "api_passphrase": s.okx_api_passphrase,
            }
        if exchange_id == "coinbase":
            return {"api_key": s.coinbase_api_key, "api_secret": s.coinbase_api_secret}
        if exchange_id == "deribit":
            return {"api_key": s.deribit_api_key, "api_secret": s.deribit_api_secret}
        return {}

    async def get(self, exchange_id: str, market_type: str = "spot") -> ExchangeClient:
        key = (exchange_id, market_type)
        # Fast path: client already exists.
        if key in self._clients:
            return self._clients[key]
        # Slow path: take the per-key lock so only one task creates the client.
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Re-check inside lock — another task may have created it while we waited.
            if key in self._clients:
                return self._clients[key]
            creds = self._credentials_for(exchange_id)
            client = CCXTClient(exchange_id, market_type=market_type, **creds)
            await client.start()
            self._clients[key] = client
            log.info("exchange_client_initialized",
                     exchange=exchange_id, market_type=market_type,
                     has_credentials=bool(creds.get("api_key")))
            return client

    async def close_all(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception as e:
                log.warning("exchange_client_close_failed",
                            exchange=client.name, error=str(e))
        self._clients.clear()
