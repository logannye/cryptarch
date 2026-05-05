"""Backtest the proposed L2 tail-rung addition against ~90 days of 5m
candles across the L2 universe.

For every 5m bar we treat as an "anchor" (i.e., as if the L2 signal had
just fired and a ladder was placed at that bar's close), we simulate two
designs side-by-side:

  - CURRENT  : 4 rungs at -1/-2/-3/-4% with the existing geometric size
               weights summing to $200.
  - PROPOSED : same 4 rungs PLUS 2 tail rungs at -6% and -10% with $10
               each (total $220, +10% capital).

Each rung's lifecycle: pending → filled (low touches limit within 60min) →
exited via TP (+1.2% from fill) / SL (-3% from fill) / time-stop (60min
after fill, exit at close). Realized P&L is the per-rung exit minus fill,
in USD on the rung's notional.

Per-anchor outcomes are aggregated into:
  - Cascade-frequency tables (depth bucket × symbol)
  - Per-design fill rate, total P&L, P&L distribution
  - The marginal contribution of the tail rungs alone
  - Tail-rung impact during the worst N drawdowns (the "asymmetric upside"
    use case)
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean, median, quantiles

from cryptarch.core.config import Settings
from cryptarch.exchanges.pool import ExchangePool
from cryptarch.strategies.l2_executor import DEFAULT_SYMBOLS, SymbolConfig

LOOKBACK_DAYS = 90
TIMEFRAME = "5m"
BAR_MS = 5 * 60 * 1000
LADDER_TTL_BARS = 12          # 60 min for unfilled rungs (current bot rule)
POST_FILL_HORIZON_BARS = 12   # 60 min hold after fill
TP_PCT = 0.012
SL_PCT = 0.03
ANCHOR_STRIDE_BARS = 12       # one anchor per hour (avoid hyper-correlated overlapping ladders)


@dataclass(frozen=True)
class Rung:
    pct_below: float
    size_usd: float


CURRENT_LADDER: list[Rung] = [
    Rung(0.01, 24.62),
    Rung(0.02, 36.92),
    Rung(0.03, 55.39),
    Rung(0.04, 83.08),
]

# Same first 4 rungs, plus 2 tail rungs. We add capital ($20) rather than
# redistribute, so the comparison is "current ladder vs current+insurance".
TAIL_RUNGS: list[Rung] = [
    Rung(0.06, 10.0),
    Rung(0.10, 10.0),
]
PROPOSED_LADDER: list[Rung] = CURRENT_LADDER + TAIL_RUNGS


def simulate_rung(bars: list[list], anchor_idx: int, anchor_close: float, rung: Rung) -> dict:
    """Returns: {filled, fill_idx, fill_price, exit_idx, exit_price, exit_reason, pnl_usd}."""
    limit = anchor_close * (1 - rung.pct_below)
    n = len(bars)
    fill_idx = None
    for j in range(anchor_idx + 1, min(anchor_idx + 1 + LADDER_TTL_BARS, n)):
        if bars[j][3] <= limit:    # bar low touches limit
            fill_idx = j
            break
    if fill_idx is None:
        return {"filled": False, "pnl_usd": 0.0, "exit_reason": "unfilled"}
    size_base = rung.size_usd / limit
    tp = limit * (1 + TP_PCT)
    sl = limit * (1 - SL_PCT)
    exit_idx = None
    exit_price = None
    exit_reason = None
    for k in range(fill_idx + 1, min(fill_idx + 1 + POST_FILL_HORIZON_BARS, n)):
        high, low, close = bars[k][2], bars[k][3], bars[k][4]
        # Conservative ordering: if both TP and SL are inside the bar's range,
        # assume SL fires first (worst-case for our strategy).
        if low <= sl:
            exit_idx = k
            exit_price = sl
            exit_reason = "sl"
            break
        if high >= tp:
            exit_idx = k
            exit_price = tp
            exit_reason = "tp"
            break
    if exit_price is None:
        exit_idx = min(fill_idx + POST_FILL_HORIZON_BARS, n - 1)
        exit_price = bars[exit_idx][4]    # close at time-stop bar
        exit_reason = "time"
    pnl = (exit_price - limit) * size_base
    return {
        "filled": True,
        "fill_idx": fill_idx,
        "fill_price": limit,
        "exit_idx": exit_idx,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl_usd": pnl,
    }


def simulate_ladder(bars: list[list], anchor_idx: int, ladder: list[Rung]) -> dict:
    if anchor_idx + LADDER_TTL_BARS + POST_FILL_HORIZON_BARS >= len(bars):
        return None
    anchor_close = bars[anchor_idx][4]
    rung_results = [simulate_rung(bars, anchor_idx, anchor_close, r) for r in ladder]
    return {
        "anchor_close": anchor_close,
        "rungs": rung_results,
        "total_pnl_usd": sum(r["pnl_usd"] for r in rung_results),
        "n_filled": sum(1 for r in rung_results if r["filled"]),
    }


def compute_max_drawdown_pct(bars: list[list], anchor_idx: int, horizon_bars: int) -> float:
    """Max % drawdown from anchor close over [anchor_idx+1, anchor_idx+horizon]."""
    anchor_close = bars[anchor_idx][4]
    end = min(anchor_idx + 1 + horizon_bars, len(bars))
    min_low = min((b[3] for b in bars[anchor_idx + 1 : end]), default=anchor_close)
    return (anchor_close - min_low) / anchor_close


async def fetch_5m_bars(ccxt_client, symbol: str, lookback_days: int) -> list[list]:
    """Paginate 5m candles back lookback_days. Returns ccxt-format bars:
    [ts_ms, open, high, low, close, volume]."""
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - lookback_days * 24 * 3600 * 1000
    out: list[list] = []
    cur = start_ms
    while cur < end_ms:
        try:
            batch = await ccxt_client.fetch_ohlcv(
                symbol, timeframe=TIMEFRAME, since=cur, limit=1000,
            )
        except Exception as e:
            print(f"  fetch failed at cur={cur}: {str(e)[:120]}", file=sys.stderr)
            break
        if not batch:
            break
        out.extend(batch)
        cur = batch[-1][0] + BAR_MS
        if len(batch) < 1000:
            break
        # Polite delay to avoid rate limits.
        await asyncio.sleep(0.05)
    return out


def summarize_per_design(label: str, results: list[dict]) -> dict:
    total_pnl = sum(r["total_pnl_usd"] for r in results)
    pnls = [r["total_pnl_usd"] for r in results]
    n_with_any_fill = sum(1 for r in results if r["n_filled"] > 0)
    return {
        "label": label,
        "n_anchors": len(results),
        "n_anchors_with_fill": n_with_any_fill,
        "total_pnl_usd": round(total_pnl, 2),
        "avg_pnl_per_anchor_usd": round(total_pnl / max(1, len(results)), 4),
        "pnl_p10": round(quantiles(pnls, n=10)[0], 4) if len(pnls) >= 10 else None,
        "pnl_median": round(median(pnls), 4) if pnls else 0.0,
        "pnl_p90": round(quantiles(pnls, n=10)[8], 4) if len(pnls) >= 10 else None,
        "pnl_min": round(min(pnls), 4) if pnls else 0.0,
        "pnl_max": round(max(pnls), 4) if pnls else 0.0,
    }


async def main() -> None:
    settings = Settings()
    pool = ExchangePool(settings)

    # We'll only backtest the 6 symbols where the L2 signal had OI history
    # in production. The 4 memecoins (PEPE/SHIB/FLOKI/BONK) are now
    # observing post-fix but had no OI data in our window — including
    # them here would give us per-event ladder math but not the bot's
    # actual past behaviour.
    universe = [
        SymbolConfig(*s) if len(s) == 4 else SymbolConfig(*s, None)
        for s in DEFAULT_SYMBOLS
    ]

    all_per_design = {"current": [], "proposed": []}
    all_per_symbol_summary: dict[str, dict] = {}
    deep_drawdowns: list[dict] = []    # for the "asymmetric upside" view

    try:
        for sym in universe:
            print(f"[fetch] {sym.symbol} 5m × {LOOKBACK_DAYS}d ...", file=sys.stderr)
            spot_client = await pool.get(sym.exchange, market_type="spot")
            bars = await fetch_5m_bars(spot_client._client, sym.symbol, LOOKBACK_DAYS)
            print(f"  got {len(bars)} bars", file=sys.stderr)
            if len(bars) < LADDER_TTL_BARS + POST_FILL_HORIZON_BARS + 10:
                print(f"  insufficient bars; skip", file=sys.stderr)
                continue

            sym_current: list[dict] = []
            sym_proposed: list[dict] = []
            for anchor_idx in range(0, len(bars), ANCHOR_STRIDE_BARS):
                cur = simulate_ladder(bars, anchor_idx, CURRENT_LADDER)
                pro = simulate_ladder(bars, anchor_idx, PROPOSED_LADDER)
                if cur is None or pro is None:
                    continue
                sym_current.append(cur)
                sym_proposed.append(pro)
                # Track deep drawdowns
                dd_24h = compute_max_drawdown_pct(
                    bars, anchor_idx,
                    horizon_bars=LADDER_TTL_BARS + POST_FILL_HORIZON_BARS,
                )
                if dd_24h >= 0.04:
                    deep_drawdowns.append({
                        "symbol": sym.symbol,
                        "anchor_ts": datetime.fromtimestamp(
                            bars[anchor_idx][0] / 1000, tz=timezone.utc).isoformat(),
                        "max_dd_pct": round(dd_24h * 100, 2),
                        "current_pnl": round(cur["total_pnl_usd"], 4),
                        "proposed_pnl": round(pro["total_pnl_usd"], 4),
                        "tail_pnl": round(
                            sum(r["pnl_usd"] for r in pro["rungs"][4:]),
                            4,
                        ),
                    })

            all_per_design["current"].extend(sym_current)
            all_per_design["proposed"].extend(sym_proposed)
            all_per_symbol_summary[sym.symbol] = {
                "current": summarize_per_design("current", sym_current),
                "proposed": summarize_per_design("proposed", sym_proposed),
                "lift_total_usd": round(
                    sum(r["total_pnl_usd"] for r in sym_proposed)
                    - sum(r["total_pnl_usd"] for r in sym_current),
                    2,
                ),
            }
            print(f"  {sym.symbol}: anchors={len(sym_current)} "
                  f"current_pnl={sum(r['total_pnl_usd'] for r in sym_current):.2f} "
                  f"proposed_pnl={sum(r['total_pnl_usd'] for r in sym_proposed):.2f}",
                  file=sys.stderr)
    finally:
        await pool.close_all()

    overall = {
        "lookback_days": LOOKBACK_DAYS,
        "timeframe": TIMEFRAME,
        "anchor_stride_bars": ANCHOR_STRIDE_BARS,
        "current": summarize_per_design("current", all_per_design["current"]),
        "proposed": summarize_per_design("proposed", all_per_design["proposed"]),
    }
    overall["lift_total_usd"] = round(
        overall["proposed"]["total_pnl_usd"] - overall["current"]["total_pnl_usd"], 2,
    )
    overall["lift_avg_per_anchor_usd"] = round(
        overall["proposed"]["avg_pnl_per_anchor_usd"]
        - overall["current"]["avg_pnl_per_anchor_usd"], 6,
    )

    # Drawdown depth distribution (across all symbols)
    dd_buckets = {"4-6%": 0, "6-10%": 0, "10-20%": 0, ">20%": 0}
    for dd in deep_drawdowns:
        d = dd["max_dd_pct"]
        if d < 6:
            dd_buckets["4-6%"] += 1
        elif d < 10:
            dd_buckets["6-10%"] += 1
        elif d < 20:
            dd_buckets["10-20%"] += 1
        else:
            dd_buckets[">20%"] += 1

    # Tail contribution: for each cascade ≥6% drawdown, did the tail rungs
    # actually fire? What's their isolated P&L?
    tail_active = [d for d in deep_drawdowns if d["max_dd_pct"] >= 6.0]
    tail_pnls = [d["tail_pnl"] for d in tail_active]

    # Top 20 worst drawdowns by current ladder loss (the canonical "what
    # happens when things break" panel)
    worst = sorted(deep_drawdowns, key=lambda d: d["current_pnl"])[:20]

    print(json.dumps({
        "overall": overall,
        "per_symbol": all_per_symbol_summary,
        "cascades_4pct_or_more": {
            "total": len(deep_drawdowns),
            "depth_buckets": dd_buckets,
        },
        "cascades_6pct_or_more_tail_impact": {
            "n_events": len(tail_active),
            "tail_total_pnl_usd": round(sum(tail_pnls), 2),
            "tail_avg_pnl_per_event_usd": round(
                sum(tail_pnls) / max(1, len(tail_pnls)), 4),
            "tail_max_pnl_usd": round(max(tail_pnls), 2) if tail_pnls else 0.0,
            "tail_min_pnl_usd": round(min(tail_pnls), 2) if tail_pnls else 0.0,
        },
        "worst_20_cascades_by_current_pnl": worst,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
