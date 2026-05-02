"""Async engine — main loop with per-cycle hard timeout.

The engine doesn't know about strategies — it iterates through registered
executors, calls `run_once()` on each with a 90s timeout, sleeps, repeats.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import structlog

from cryptarch.core.config import Settings
from cryptarch.db.store import Store
from cryptarch.exchanges.pool import ExchangePool

log = structlog.get_logger()


# Type alias for a strategy executor's run_once.
RunOnce = Callable[[], Awaitable[dict[str, Any]]]


class Engine:
    """Hosts strategy executors. Each cycle:
       1. Iterate executors; each gets a 90s budget.
       2. Reconcile total_at_risk every reconcile_interval.
       3. Sleep scan_interval.
    """

    def __init__(self, settings: Settings, store: Store, pool: ExchangePool):
        self._settings = settings
        self._store = store
        self._pool = pool
        self._executors: list[tuple[str, RunOnce]] = []
        self._last_reconcile = datetime.now(timezone.utc)
        self._cycle_count = 0

    def register(self, name: str, run_once: RunOnce) -> None:
        self._executors.append((name, run_once))
        log.info("executor_registered", name=name)

    async def run_forever(self) -> None:
        log.info("engine_starting",
                 executors=[name for name, _ in self._executors],
                 enable_live_orders=self._settings.enable_live_orders,
                 scan_interval=self._settings.scan_interval_seconds)
        while True:
            await self._cycle()
            await asyncio.sleep(self._settings.scan_interval_seconds)

    async def _cycle(self) -> None:
        self._cycle_count += 1
        cycle_started = datetime.now(timezone.utc)

        for name, run_once in self._executors:
            started = datetime.now(timezone.utc)
            try:
                result = await asyncio.wait_for(run_once(), timeout=90)
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                log.info("executor_cycle",
                         name=name, count=self._cycle_count,
                         elapsed_s=round(elapsed, 2), result=result)
            except asyncio.TimeoutError:
                log.error("executor_cycle_timeout",
                          name=name, count=self._cycle_count)
            except Exception as e:
                log.error("executor_cycle_error",
                          name=name, count=self._cycle_count,
                          error=str(e)[:200])

        # Reconcile periodically.
        if (datetime.now(timezone.utc) - self._last_reconcile).total_seconds() \
                >= self._settings.reconcile_interval_seconds:
            try:
                await self._store.reconcile_at_risk()
                self._last_reconcile = datetime.now(timezone.utc)
                log.info("at_risk_reconciled")
            except Exception as e:
                log.error("reconcile_failed", error=str(e))

        elapsed_total = (datetime.now(timezone.utc) - cycle_started).total_seconds()
        log.info("cycle_complete",
                 count=self._cycle_count,
                 elapsed_s=round(elapsed_total, 2))
