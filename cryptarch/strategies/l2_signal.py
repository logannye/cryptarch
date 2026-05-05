"""Real cascade-probability signal for Layer 2.

Replaces the stub in l2_executor with one that uses:
  - OI percentile from rolling 7-day store-backed history
  - Funding rate from live exchange
  - Realized-volatility ratio from OHLCV (24h vs 7d)

If insufficient OI history is available (cold-start, < 24 observations),
we conservatively return 0 — refusing to trigger ladders until we have
real percentile data.
"""
from __future__ import annotations

import structlog

from cryptarch.db.store import Store
from cryptarch.exchanges.pool import ExchangePool
from cryptarch.strategies.l2_cascade import (
    cascade_probability, percentile_rank, realized_vol_from_closes,
)
from cryptarch.strategies.l2_executor import SymbolConfig

log = structlog.get_logger()


# Need at least this many OI observations before the percentile is
# considered meaningful. With the default observer interval (1/min)
# this is roughly 24 hours of accumulation.
MIN_OI_HISTORY_LEN = 24


class CascadeSignal:
    """Produces a 0-1 cascade probability for a symbol from live + history."""

    def __init__(self, store: Store, pool: ExchangePool):
        self._store = store
        self._pool = pool

    async def __call__(self, symbol: SymbolConfig, executor) -> float:
        """Signature matches L2Executor's signal_fn type: takes the
        symbol and the executor (unused), returns a 0-1 score."""
        del executor    # not needed; we have direct access to store + pool

        # 1. OI percentile from store history
        history = await self._store.oi_history(symbol.exchange, symbol.symbol, hours=168)
        if len(history) < MIN_OI_HISTORY_LEN:
            log.debug("l2_signal_oi_cold_start",
                      symbol=symbol.key, history_len=len(history))
            return 0.0
        # Get current OI
        try:
            perp_symbol = symbol.perp_symbol
            client = await self._pool.get(symbol.exchange, market_type="swap")
            oi = await client.get_open_interest(perp_symbol)
            current_oi = oi.oi_usd
        except Exception as e:
            log.debug("l2_signal_oi_fetch_failed",
                      symbol=symbol.key, error=str(e)[:100])
            return 0.0
        if current_oi <= 0:
            return 0.0
        oi_pct = percentile_rank(current_oi, history)

        # 2. Funding rate (current 8h)
        try:
            funding = await client.get_funding_rate(perp_symbol)
            funding_rate = float(funding.rate_8h)
        except Exception as e:
            log.debug("l2_signal_funding_fetch_failed",
                      symbol=symbol.key, error=str(e)[:100])
            return 0.0

        # 3. Volatility compression from OHLCV (1h candles, 7 days = 168 bars)
        try:
            spot_client = await self._pool.get(symbol.exchange, market_type="spot")
            candles = await spot_client.get_ohlcv(symbol.symbol, timeframe="1h", limit=168)
        except Exception as e:
            log.debug("l2_signal_ohlcv_fetch_failed",
                      symbol=symbol.key, error=str(e)[:100])
            return 0.0
        if len(candles) < 50:
            return 0.0    # not enough history for vol calc
        closes = [c.close for c in candles]
        recent_vol = realized_vol_from_closes(closes, lookback=24)
        historical_vol = realized_vol_from_closes(closes, lookback=168)

        score = cascade_probability(
            oi_percentile=oi_pct,
            funding_rate_24h_avg=funding_rate,
            recent_24h_vol_pct=recent_vol,
            historical_7d_vol_pct=historical_vol,
        )
        log.debug("l2_signal_computed",
                  symbol=symbol.key, score=round(score, 3),
                  oi_pct=round(oi_pct, 1),
                  funding=round(funding_rate, 6),
                  vol_ratio=round(recent_vol / historical_vol if historical_vol else 0, 3))
        return score


class OIObserver:
    """Periodic OI recorder. Registered with the engine; runs every
    `interval_seconds` and records current OI for every L2 symbol.

    Bootstraps the rolling history that CascadeSignal needs. Until at
    least MIN_OI_HISTORY_LEN observations exist per symbol, the signal
    returns 0 (refusing to trigger).
    """

    def __init__(
        self,
        store: Store,
        pool: ExchangePool,
        symbols: list[SymbolConfig],
        prune_days: int = 14,
        prune_every_n_cycles: int = 60,    # at 1/min cadence, prune hourly
    ):
        self._store = store
        self._pool = pool
        self._symbols = symbols
        self._prune_days = prune_days
        self._prune_every = prune_every_n_cycles
        self._cycle_count = 0

    async def run_once(self) -> dict:
        self._cycle_count += 1
        n_recorded = 0
        for symbol in self._symbols:
            try:
                perp_symbol = symbol.perp_symbol
                client = await self._pool.get(symbol.exchange, market_type="swap")
                oi = await client.get_open_interest(perp_symbol)
                if oi.oi_usd > 0:
                    await self._store.record_oi_observation(
                        symbol.exchange, symbol.symbol, oi.oi_usd,
                    )
                    n_recorded += 1
            except Exception as e:
                log.debug("oi_observe_failed",
                          symbol=symbol.key, error=str(e)[:100])
                continue
        result: dict = {"recorded": n_recorded, "symbols": len(self._symbols)}

        # Periodic pruning to keep the table bounded.
        if self._cycle_count % self._prune_every == 0:
            try:
                deleted = await self._store.prune_oi_history(retain_days=self._prune_days)
                result["pruned"] = deleted
            except Exception as e:
                log.warning("oi_prune_failed", error=str(e))
        return result
