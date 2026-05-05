"""One-shot read-only probe: print live L2 cascade-probability breakdown for
each tracked symbol. Mirrors what CascadeSignal computes inside the engine,
but renders the per-component contributions instead of returning a scalar."""
from __future__ import annotations

import asyncio
import json

import asyncpg

from cryptarch.core.config import Settings
from cryptarch.db.store import Store
from cryptarch.exchanges.pool import ExchangePool
from cryptarch.strategies.l2_cascade import (
    cascade_probability, percentile_rank, realized_vol_from_closes,
)
from cryptarch.strategies.l2_executor import DEFAULT_SYMBOLS, SymbolConfig


async def probe() -> None:
    settings = Settings()
    pool = ExchangePool(settings)
    db_pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=2)
    store = Store(db_pool)
    rows: list[dict] = []
    try:
        symbols = [
            SymbolConfig(*s) if len(s) == 4 else SymbolConfig(*s, None)
            for s in DEFAULT_SYMBOLS
        ]
        for symbol in symbols:
            history = await store.oi_history(
                symbol.exchange, symbol.symbol, hours=168,
            )
            try:
                perp_symbol = symbol.perp_symbol
                client = await pool.get(symbol.exchange, market_type="swap")
                oi = await client.get_open_interest(perp_symbol)
                cur_oi = oi.oi_usd
                fr = await client.get_funding_rate(perp_symbol)
                funding_rate = float(fr.rate_8h)
                spot_client = await pool.get(symbol.exchange, market_type="spot")
                candles = await spot_client.get_ohlcv(
                    symbol.symbol, timeframe="1h", limit=168,
                )
            except Exception as e:
                rows.append({"symbol": symbol.key, "error": str(e)[:120]})
                continue
            closes = [c.close for c in candles]
            recent_vol = realized_vol_from_closes(closes, lookback=24)
            historical_vol = realized_vol_from_closes(closes, lookback=168)
            oi_pct = percentile_rank(cur_oi, history) if len(history) >= 24 else None
            ratio = (recent_vol / historical_vol) if historical_vol else 0.0

            score = None
            if oi_pct is not None:
                score = cascade_probability(
                    oi_percentile=oi_pct,
                    funding_rate_24h_avg=funding_rate,
                    recent_24h_vol_pct=recent_vol,
                    historical_7d_vol_pct=historical_vol,
                )

            oi_signal = max(0.0, min(1.0, ((oi_pct or 0.0) - 70.0) / 25.0)) if oi_pct is not None else 0.0
            funding_signal = max(0.0, min(1.0, funding_rate / 0.001))
            compression_signal = (
                max(0.0, min(1.0, (1.0 - ratio) / 0.7)) if historical_vol else 0.0
            )
            rows.append({
                "symbol": symbol.key,
                "oi_history_n": len(history),
                "current_oi_usd": round(cur_oi, 0),
                "oi_pct": round(oi_pct, 1) if oi_pct is not None else None,
                "oi_signal": round(oi_signal, 3),
                "funding_8h_pct": round(funding_rate * 100, 4),
                "funding_signal": round(funding_signal, 3),
                "vol_ratio": round(ratio, 3),
                "compression_signal": round(compression_signal, 3),
                "score": round(score, 3) if score is not None else None,
                "above_trigger_0.6": score is not None and score >= 0.6,
                "above_hold_0.4": score is not None and score >= 0.4,
            })
    finally:
        await pool.close_all()
        await db_pool.close()

    rows.sort(key=lambda r: (r.get("score") or -1), reverse=True)
    for r in rows:
        print(json.dumps(r))
    n_trigger = sum(1 for r in rows if r.get("above_trigger_0.6"))
    n_hold = sum(1 for r in rows if r.get("above_hold_0.4"))
    print(json.dumps({
        "total": len(rows),
        "above_0.6_trigger": n_trigger,
        "above_0.4_hold": n_hold,
    }))


if __name__ == "__main__":
    asyncio.run(probe())
