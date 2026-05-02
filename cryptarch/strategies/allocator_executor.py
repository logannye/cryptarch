"""AllocatorExecutor — periodic recompute of dynamic capital allocation.

Gathers per-layer opportunity signals and writes a target allocation to
system_state. The L1/L2/L3 executors read this on each cycle and use it
in place of the static config (when present).

Run cadence: every few minutes is enough — we're tilting allocation,
not trading. Avoid recomputing every 30s (noisy) or every hour (slow
to react when funding spikes).
"""
from __future__ import annotations

import asyncio
import structlog

from cryptarch.core.allocator import LayerSignals, compute_target_allocation
from cryptarch.core.config import Settings
from cryptarch.db.store import Store
from cryptarch.exchanges.pool import ExchangePool
from cryptarch.strategies.l1_executor import L1Executor, PairConfig
from cryptarch.strategies.l2_executor import SymbolConfig
from cryptarch.strategies.l2_signal import CascadeSignal
from cryptarch.strategies.l1_funding import FundingArbCandidate

log = structlog.get_logger()


class AllocatorExecutor:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        pool: ExchangePool,
        l1_executor: L1Executor,
        l2_signal_fn,                          # CascadeSignal instance
        l2_symbols: list[SymbolConfig],
        recompute_every_n_cycles: int = 10,    # at 30s cadence → every 5 min
    ):
        self._settings = settings
        self._store = store
        self._pool = pool
        self._l1 = l1_executor
        self._l2_signal = l2_signal_fn
        self._l2_symbols = l2_symbols
        self._recompute_every = recompute_every_n_cycles
        self._cycle_count = 0

    async def run_once(self) -> dict:
        self._cycle_count += 1
        if self._cycle_count % self._recompute_every != 0:
            return {"skipped": "interval_not_reached", "cycles_until_next":
                    self._recompute_every - (self._cycle_count % self._recompute_every)}

        # Gather signals concurrently.
        l1_signal_task = self._compute_l1_signal()
        l2_signal_task = self._compute_l2_signal()
        l3_signal_task = self._compute_l3_signal()
        l1_max_apr, l2_max_score, l3_compression = await asyncio.gather(
            l1_signal_task, l2_signal_task, l3_signal_task,
            return_exceptions=False,
        )

        signals = LayerSignals(
            l1_max_apr_pct=l1_max_apr,
            l2_max_cascade_score=l2_max_score,
            l3_iv_compression_score=l3_compression,
        )
        decision = compute_target_allocation(
            signals,
            base_l1=self._settings.alloc_layer_1_pct,
            base_l2=self._settings.alloc_layer_2_pct,
            base_l3=self._settings.alloc_layer_3_pct,
        )
        await self._store.set_dynamic_allocation(
            l1=decision.l1_pct, l2=decision.l2_pct, l3=decision.l3_pct,
            rationale=decision.rationale,
        )
        log.info("allocator_updated",
                 l1=round(decision.l1_pct, 4),
                 l2=round(decision.l2_pct, 4),
                 l3=round(decision.l3_pct, 4),
                 l1_apr=round(l1_max_apr, 1),
                 l2_score=round(l2_max_score, 3),
                 l3_compression=round(l3_compression, 3),
                 rationale=decision.rationale)
        return {
            "l1_pct": round(decision.l1_pct, 4),
            "l2_pct": round(decision.l2_pct, 4),
            "l3_pct": round(decision.l3_pct, 4),
            "rationale": decision.rationale,
        }

    # ── per-layer signal collection ──

    async def _compute_l1_signal(self) -> float:
        """Best APR currently visible across the L1 funding-arb scan."""
        try:
            candidates = await self._l1._scan_candidates()
        except Exception as e:
            log.warning("alloc_l1_scan_failed", error=str(e)[:120])
            return 0.0
        if not candidates:
            return 0.0
        # APR is funding × 3 × 0.5 × 365 × 100 (% terms)
        max_apr = max(c.expected_apr_pct for c in candidates) * 100.0
        return max(0.0, max_apr)

    async def _compute_l2_signal(self) -> float:
        """Highest cascade probability across all tracked symbols."""
        if not self._l2_symbols:
            return 0.0
        try:
            scores = await asyncio.gather(
                *(self._l2_signal(sym, None) for sym in self._l2_symbols),
                return_exceptions=True,
            )
        except Exception as e:
            log.warning("alloc_l2_scan_failed", error=str(e)[:120])
            return 0.0
        valid = [s for s in scores if isinstance(s, (int, float))]
        if not valid:
            return 0.0
        return max(valid)

    async def _compute_l3_signal(self) -> float:
        """IV compression score for BTC (the L3 underlying).

        For v1 we use a simple proxy: compare current realized vol to
        a longer-term baseline. A future refinement would query Deribit
        for current mark IV vs historical IV. The compression score is
        between 0 and 1 where 1 means current vol is much lower than
        historical baseline (cheap options regime).
        """
        try:
            client = await self._pool.get("binance", market_type="spot")
            candles = await client.get_ohlcv("BTC/USDT", timeframe="1d", limit=60)
        except Exception as e:
            log.warning("alloc_l3_ohlcv_failed", error=str(e)[:120])
            return 0.0
        if len(candles) < 30:
            return 0.0
        from cryptarch.strategies.l2_cascade import realized_vol_from_closes
        closes = [c.close for c in candles]
        recent = realized_vol_from_closes(closes, lookback=10)
        long_term = realized_vol_from_closes(closes, lookback=60)
        if long_term <= 0:
            return 0.0
        ratio = recent / long_term
        # ratio 1.0 → no compression; ratio 0.5 → significant compression
        # Map ratio 0.5 → score 1.0; ratio 1.0 → score 0
        return max(0.0, min(1.0, (1.0 - ratio) / 0.5))
