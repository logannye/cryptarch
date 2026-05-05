"""Backtest a candidate universe of symbols against the current L2 ladder
design over ~90 days of 5m candles.

Mechanically identical to scripts/backtest_l2_tail_rungs.py but:
  - Iterates a candidate list rather than DEFAULT_SYMBOLS
  - Uses the current production ladder shape only (no comparison against
    a tail-rung variant — that question's already answered)
  - Reports per-symbol P&L in a form directly comparable to the prior
    full-universe table at $600 sizing

Each anchor is placed at every Nth bar (default: every hour), and the
existing rung-fill simulation (low-of-bar touches limit; SL ordering
conservative when both bounds inside same bar; 60min hold) is run.
The aggregate per-symbol P&L is what we use to decide whether each
candidate is worth promoting to production.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median, quantiles

from cryptarch.core.config import Settings
from cryptarch.exchanges.pool import ExchangePool

import os

LOOKBACK_DAYS = 90
# Override via env: TIMEFRAME=1m to capture intra-5m wick dynamics on
# memecoins. Bar count constants below are derived from the chosen tf so
# the lifecycle windows stay constant in wall-clock minutes regardless.
TIMEFRAME = os.environ.get("CRYPTARCH_BACKTEST_TF", "5m")
_TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15}[TIMEFRAME]
BAR_MS = _TF_MINUTES * 60 * 1000
LADDER_TTL_BARS = 60 // _TF_MINUTES         # 60 min for unfilled rungs
POST_FILL_HORIZON_BARS = 60 // _TF_MINUTES  # 60 min hold after fill
ANCHOR_STRIDE_BARS = 60 // _TF_MINUTES      # one anchor per hour
TP_PCT = 0.012
SL_PCT = 0.030


@dataclass(frozen=True)
class Rung:
    pct_below: float
    size_usd: float


# Current production ladder: $5k bankroll × 12% = $600 total, 4 rungs,
# geometric weight decay 1.5. Weights 1/1.5/2.25/3.375 (sum 8.125).
LADDER: list[Rung] = [
    Rung(0.01, 73.85),
    Rung(0.02, 110.77),
    Rung(0.03, 166.15),
    Rung(0.04, 249.23),
]


# Original 10-symbol L2 universe — re-evaluating at finer granularity to
# validate the prior pruning decision (made on 5m data, now suspect after
# the candidate run revealed 5m's structural bias against fast V-shape
# reverts). The five symbols we previously dropped (XRP/DOGE/SHIB/FLOKI/
# BONK) might have been wrongly cut.
CANDIDATES: list[tuple[str, str, str]] = [
    ("BTC/USDT",   "BTC",   "Major reference — institutional flow"),
    ("ETH/USDT",   "ETH",   "Major reference — institutional flow"),
    ("SOL/USDT",   "SOL",   "Major reference — Solana ecosystem"),
    ("XRP/USDT",   "XRP",   "PRUNED at 5m — re-evaluate"),
    ("DOGE/USDT",  "DOGE",  "PRUNED at 5m — re-evaluate"),
    ("PEPE/USDT",  "PEPE",  "REFERENCE: top contributor in production"),
    ("SHIB/USDT",  "SHIB",  "PRUNED at 5m — re-evaluate"),
    ("FLOKI/USDT", "FLOKI", "PRUNED at 5m (worst, -$8.95) — re-evaluate"),
    ("WIF/USDT",   "WIF",   "REFERENCE: second contributor in production"),
    ("BONK/USDT",  "BONK",  "PRUNED at 5m — re-evaluate"),
]


def simulate_rung(bars: list[list], anchor_idx: int, anchor_close: float, rung: Rung) -> dict:
    """Find first fill within LADDER_TTL_BARS; if filled, find first
    TP/SL/timestop within POST_FILL_HORIZON_BARS. Returns per-rung outcome."""
    limit = anchor_close * (1 - rung.pct_below)
    n = len(bars)
    fill_idx = None
    for j in range(anchor_idx + 1, min(anchor_idx + 1 + LADDER_TTL_BARS, n)):
        if bars[j][3] <= limit:
            fill_idx = j
            break
    if fill_idx is None:
        return {"filled": False, "pnl_usd": 0.0}
    size_base = rung.size_usd / limit
    tp = limit * (1 + TP_PCT)
    sl = limit * (1 - SL_PCT)
    exit_price = None
    for k in range(fill_idx + 1, min(fill_idx + 1 + POST_FILL_HORIZON_BARS, n)):
        high, low, _ = bars[k][2], bars[k][3], bars[k][4]
        # Conservative ordering: SL first when both bounds inside the bar
        if low <= sl:
            exit_price = sl
            break
        if high >= tp:
            exit_price = tp
            break
    if exit_price is None:
        exit_idx = min(fill_idx + POST_FILL_HORIZON_BARS, n - 1)
        exit_price = bars[exit_idx][4]
    pnl = (exit_price - limit) * size_base
    return {"filled": True, "pnl_usd": pnl}


def simulate_ladder(bars: list[list], anchor_idx: int) -> dict | None:
    if anchor_idx + LADDER_TTL_BARS + POST_FILL_HORIZON_BARS >= len(bars):
        return None
    anchor_close = bars[anchor_idx][4]
    rungs = [simulate_rung(bars, anchor_idx, anchor_close, r) for r in LADDER]
    return {
        "total_pnl_usd": sum(r["pnl_usd"] for r in rungs),
        "n_filled": sum(1 for r in rungs if r["filled"]),
    }


async def fetch_5m_bars(ccxt_client, symbol: str, lookback_days: int) -> list[list]:
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
            return out
        if not batch:
            break
        out.extend(batch)
        cur = batch[-1][0] + BAR_MS
        if len(batch) < 1000:
            break
        await asyncio.sleep(0.05)
    return out


def summarize(label: str, results: list[dict]) -> dict:
    pnls = [r["total_pnl_usd"] for r in results]
    if not pnls:
        return {"symbol": label, "error": "no anchors processed"}
    fills = [r["n_filled"] for r in results]
    n_winners = sum(1 for p in pnls if p > 0.001)
    n_losers = sum(1 for p in pnls if p < -0.001)
    return {
        "symbol": label,
        "n_anchors": len(pnls),
        "n_anchors_with_fill": sum(1 for f in fills if f > 0),
        "n_winning_ladders": n_winners,
        "n_losing_ladders": n_losers,
        "total_pnl_usd": round(sum(pnls), 2),
        "avg_pnl_per_anchor_usd": round(sum(pnls) / len(pnls), 4),
        "median_pnl": round(median(pnls), 4),
        "p10_pnl": round(quantiles(pnls, n=10)[0], 4) if len(pnls) >= 10 else None,
        "p90_pnl": round(quantiles(pnls, n=10)[8], 4) if len(pnls) >= 10 else None,
        "min_pnl": round(min(pnls), 4),
        "max_pnl": round(max(pnls), 4),
    }


async def main() -> None:
    settings = Settings()
    pool = ExchangePool(settings)
    summaries: list[dict] = []
    failures: list[dict] = []

    try:
        for spot_symbol, base_label, hypothesis in CANDIDATES:
            print(f"[fetch] {spot_symbol} ({base_label}) — {hypothesis}", file=sys.stderr)
            try:
                spot_client = await pool.get("binance", market_type="spot")
                bars = await fetch_5m_bars(spot_client._client, spot_symbol, LOOKBACK_DAYS)
            except Exception as e:
                print(f"  fetch error: {e}", file=sys.stderr)
                failures.append({"symbol": spot_symbol, "error": str(e)[:200]})
                continue
            if len(bars) < LADDER_TTL_BARS + POST_FILL_HORIZON_BARS + 10:
                print(f"  insufficient bars ({len(bars)}); skip", file=sys.stderr)
                failures.append({
                    "symbol": spot_symbol,
                    "error": f"only {len(bars)} bars (need ≥35)",
                })
                continue
            print(f"  got {len(bars)} bars", file=sys.stderr)

            sym_results: list[dict] = []
            for anchor_idx in range(0, len(bars), ANCHOR_STRIDE_BARS):
                r = simulate_ladder(bars, anchor_idx)
                if r is not None:
                    sym_results.append(r)

            summary = summarize(spot_symbol, sym_results)
            summary["base_label"] = base_label
            summary["hypothesis"] = hypothesis
            summary["bars_available"] = len(bars)
            # Approximate days of history actually covered (in case shorter than 90)
            if bars:
                days = (bars[-1][0] - bars[0][0]) / (1000 * 86400)
                summary["days_covered"] = round(days, 1)
            summaries.append(summary)
            print(f"  {base_label}: anchors={summary['n_anchors']} "
                  f"total_pnl=${summary['total_pnl_usd']:+.2f} "
                  f"fill_rate={summary['n_anchors_with_fill']}/{summary['n_anchors']} "
                  f"days={summary.get('days_covered', '?')}",
                  file=sys.stderr)
    finally:
        await pool.close_all()

    # Sort summaries by total_pnl_usd descending so the strongest candidates
    # surface at the top of the JSON output.
    summaries.sort(key=lambda s: s.get("total_pnl_usd", 0), reverse=True)

    print(json.dumps({
        "config": {
            "lookback_days": LOOKBACK_DAYS,
            "timeframe": TIMEFRAME,
            "ladder_total_usd": sum(r.size_usd for r in LADDER),
            "ladder_rungs": [
                {"pct_below": r.pct_below, "size_usd": r.size_usd}
                for r in LADDER
            ],
            "anchor_stride_bars": ANCHOR_STRIDE_BARS,
            "tp_pct": TP_PCT,
            "sl_pct": SL_PCT,
            "post_fill_horizon_bars": POST_FILL_HORIZON_BARS,
        },
        "candidates": summaries,
        "failures": failures,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
