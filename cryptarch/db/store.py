"""Async DB access layer.

Single store object wraps an asyncpg pool and exposes domain operations.
Strategy code calls these methods; raw SQL stays here.

Idempotency is enforced at the DB level via the UNIQUE constraint on
fill.client_order_id — any duplicate submission raises a violation that
the safeguard layer also catches independently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg
import structlog

log = structlog.get_logger()


@dataclass(frozen=True)
class SystemState:
    bankroll_usd: float
    cash_usd: float
    total_at_risk_usd: float
    daily_pnl: float
    cumulative_pnl: float
    deployment_stage: str
    halt_reason: str | None
    # Dynamic allocations (None = fall back to static config)
    dynamic_alloc_l1_pct: float | None = None
    dynamic_alloc_l2_pct: float | None = None
    dynamic_alloc_l3_pct: float | None = None
    dynamic_alloc_rationale: str | None = None


@dataclass(frozen=True)
class OpenPosition:
    id: int
    layer: str
    strategy_group_id: str
    state: str
    notional_usd: float
    realized_pnl_usd: float
    opened_at: datetime
    metadata: dict[str, Any]


class Store:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    # ── system state ──

    async def get_system_state(self) -> SystemState | None:
        row = await self._pool.fetchrow("SELECT * FROM system_state WHERE id=1")
        if row is None:
            return None
        return SystemState(
            bankroll_usd=float(row["bankroll_usd"]),
            cash_usd=float(row["cash_usd"]),
            total_at_risk_usd=float(row["total_at_risk_usd"]),
            daily_pnl=float(row["daily_pnl"]),
            cumulative_pnl=float(row["cumulative_pnl"]),
            deployment_stage=row["deployment_stage"],
            halt_reason=row["halt_reason"],
            dynamic_alloc_l1_pct=(float(row["dynamic_alloc_l1_pct"])
                                  if row.get("dynamic_alloc_l1_pct") is not None else None),
            dynamic_alloc_l2_pct=(float(row["dynamic_alloc_l2_pct"])
                                  if row.get("dynamic_alloc_l2_pct") is not None else None),
            dynamic_alloc_l3_pct=(float(row["dynamic_alloc_l3_pct"])
                                  if row.get("dynamic_alloc_l3_pct") is not None else None),
            dynamic_alloc_rationale=row.get("dynamic_alloc_rationale"),
        )

    async def set_dynamic_allocation(
        self, l1: float, l2: float, l3: float, rationale: str,
    ) -> None:
        await self._pool.execute(
            """UPDATE system_state SET
                   dynamic_alloc_l1_pct=$1,
                   dynamic_alloc_l2_pct=$2,
                   dynamic_alloc_l3_pct=$3,
                   dynamic_alloc_rationale=$4,
                   dynamic_alloc_updated_at=NOW(),
                   last_updated=NOW()
               WHERE id=1""",
            l1, l2, l3, rationale,
        )

    async def set_halt_reason(self, reason: str | None) -> None:
        await self._pool.execute(
            "UPDATE system_state SET halt_reason=$1, last_updated=NOW() WHERE id=1",
            reason,
        )

    # ── deployed-capital queries (for safeguards) ──

    async def total_at_risk_usd(self) -> float:
        v = await self._pool.fetchval(
            "SELECT COALESCE(SUM(notional_usd), 0) FROM position "
            "WHERE state IN ('opening', 'open')"
        )
        return float(v or 0)

    async def layer_deployed_usd(self, layer: str) -> float:
        v = await self._pool.fetchval(
            "SELECT COALESCE(SUM(notional_usd), 0) FROM position "
            "WHERE state IN ('opening', 'open') AND layer = $1",
            layer,
        )
        return float(v or 0)

    async def recent_client_order_ids(self, hours: int = 48) -> set[str]:
        rows = await self._pool.fetch(
            "SELECT client_order_id FROM fill "
            "WHERE placed_at > NOW() - ($1 || ' hours')::interval",
            str(hours),
        )
        return {r["client_order_id"] for r in rows}

    # ── positions ──

    async def open_positions(self, layer: str | None = None) -> list[OpenPosition]:
        if layer is None:
            rows = await self._pool.fetch(
                "SELECT * FROM position WHERE state IN ('opening', 'open') ORDER BY opened_at"
            )
        else:
            rows = await self._pool.fetch(
                "SELECT * FROM position WHERE state IN ('opening', 'open') AND layer = $1 "
                "ORDER BY opened_at",
                layer,
            )
        out = []
        for r in rows:
            md = r["metadata"]
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except json.JSONDecodeError:
                    md = {}
            out.append(OpenPosition(
                id=r["id"], layer=r["layer"],
                strategy_group_id=r["strategy_group_id"],
                state=r["state"],
                notional_usd=float(r["notional_usd"]),
                realized_pnl_usd=float(r["realized_pnl_usd"]),
                opened_at=r["opened_at"],
                metadata=md or {},
            ))
        return out

    async def create_position(
        self, layer: str, strategy_group_id: str,
        notional_usd: float, metadata: dict[str, Any],
    ) -> int:
        return await self._pool.fetchval(
            """INSERT INTO position (layer, strategy_group_id, state, notional_usd, metadata)
               VALUES ($1, $2, 'opening', $3, $4) RETURNING id""",
            layer, strategy_group_id, notional_usd, json.dumps(metadata),
        )

    async def mark_position_open(self, position_id: int) -> None:
        await self._pool.execute(
            "UPDATE position SET state='open' WHERE id=$1", position_id,
        )

    async def mark_position_closing(self, position_id: int) -> None:
        await self._pool.execute(
            "UPDATE position SET state='closing' WHERE id=$1", position_id,
        )

    async def close_position(self, position_id: int, realized_pnl: float) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """UPDATE position SET state='closed', closed_at=NOW(),
                                            realized_pnl_usd=$1 WHERE id=$2""",
                    realized_pnl, position_id,
                )
                # Move position notional out of at-risk and book PnL.
                # Note: total_at_risk is also recomputed by the engine reconciliation
                # loop; this keeps it accurate between reconciliations.
                row = await conn.fetchrow(
                    "SELECT notional_usd FROM position WHERE id=$1", position_id,
                )
                if row is None:
                    return
                notional = float(row["notional_usd"])
                await conn.execute(
                    """UPDATE system_state SET
                           cash_usd = cash_usd + $1,
                           total_at_risk_usd = GREATEST(0, total_at_risk_usd - $2),
                           daily_pnl = daily_pnl + $1,
                           cumulative_pnl = cumulative_pnl + $1,
                           last_updated = NOW()
                       WHERE id=1""",
                    realized_pnl, notional,
                )

    # ── fills ──

    async def record_fill(
        self,
        position_id: int | None,
        layer: str,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        size_base: float,
        size_usd: float,
        fill_price: float,
        client_order_id: str,
        is_simulated: bool,
        sim_reason: str = "",
        exchange_order_id: str | None = None,
        fee_usd: float = 0.0,
    ) -> int | None:
        try:
            return await self._pool.fetchval(
                """INSERT INTO fill (
                       position_id, layer, exchange, symbol, side, order_type,
                       size_base, size_usd, fill_price, fee_usd,
                       is_simulated, sim_reason, client_order_id, exchange_order_id)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                   RETURNING id""",
                position_id, layer, exchange, symbol, side, order_type,
                size_base, size_usd, fill_price, fee_usd,
                is_simulated, sim_reason, client_order_id, exchange_order_id,
            )
        except asyncpg.UniqueViolationError:
            log.warning("fill_duplicate_rejected", client_order_id=client_order_id)
            return None

    # ── funding events ──

    async def record_funding_event(
        self, position_id: int, exchange: str, symbol: str,
        rate_8h: float, notional_usd: float, payment_usd: float,
        is_simulated: bool, paid_at: datetime,
    ) -> int:
        return await self._pool.fetchval(
            """INSERT INTO funding_event (
                   position_id, exchange, symbol, rate_8h, notional_usd,
                   payment_usd, is_simulated, paid_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id""",
            position_id, exchange, symbol, rate_8h, notional_usd,
            payment_usd, is_simulated, paid_at,
        )

    async def total_funding_collected(self, position_id: int) -> float:
        v = await self._pool.fetchval(
            "SELECT COALESCE(SUM(payment_usd), 0) FROM funding_event WHERE position_id=$1",
            position_id,
        )
        return float(v or 0)

    # ── OI observations (Layer 2 cascade signal) ──

    async def record_oi_observation(
        self, exchange: str, symbol: str, oi_usd: float,
    ) -> None:
        await self._pool.execute(
            "INSERT INTO oi_observation (exchange, symbol, open_interest_usd) "
            "VALUES ($1, $2, $3)",
            exchange, symbol, oi_usd,
        )

    async def oi_history(
        self, exchange: str, symbol: str, hours: int = 168,
    ) -> list[float]:
        """Return OI observations for the symbol over the last `hours`
        (default 7 days). Used to compute the percentile signal."""
        rows = await self._pool.fetch(
            "SELECT open_interest_usd FROM oi_observation "
            "WHERE exchange=$1 AND symbol=$2 "
            "AND observed_at > NOW() - ($3 || ' hours')::interval "
            "ORDER BY observed_at",
            exchange, symbol, str(hours),
        )
        return [float(r["open_interest_usd"]) for r in rows]

    async def prune_oi_history(self, retain_days: int = 14) -> int:
        """Delete observations older than `retain_days`. Returns deleted count."""
        result = await self._pool.execute(
            "DELETE FROM oi_observation "
            "WHERE observed_at < NOW() - ($1 || ' days')::interval",
            str(retain_days),
        )
        # asyncpg returns "DELETE N" string
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    # ── reconcile total_at_risk ──

    async def reconcile_at_risk(self) -> None:
        """Recompute system_state.total_at_risk_usd from authoritative
        position table. Run periodically; protects against drift."""
        v = await self._pool.fetchval(
            "SELECT COALESCE(SUM(notional_usd), 0) FROM position "
            "WHERE state IN ('opening', 'open')"
        )
        await self._pool.execute(
            "UPDATE system_state SET total_at_risk_usd=$1, last_updated=NOW() WHERE id=1",
            float(v or 0),
        )
