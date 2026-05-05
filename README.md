# cryptarch

Three-layer crypto trading system. Mathematically derived to combine three near-zero-correlated edges into a positive-skew portfolio (tail outcomes are upside).

## The three layers

| Layer | Strategy | E[r]/day | Source of edge |
|-------|----------|----------|----------------|
| **L1** | Funding-rate arbitrage (delta-neutral basis) | 0.05–0.15% in bull regimes; ~0% in calm regimes | Structural perp/spot mispricing — longs pay shorts via funding payments |
| **L2** | Liquidation cascade capture (mean-reversion ladder) | Episodic, ~0.10–0.30% per cascade event | Forced liquidations consume order-book depth and overshoot fundamentals; price reverts within minutes-to-hours |
| **L3** | OTM-options tail hedge (long strangles via Deribit) | -0.05% normal carry, +5–15% on tail moves | Long convexity into vol regime shifts; bounded loss, unbounded upside |

L1 and L2 generate the bread-and-butter return. L3 is insurance that costs theta in the boring case and pays out asymmetrically when something genuinely breaks.

## Status

All three layers are shipped and operating in dry-run mode. Current focus is calibration based on measured outcomes against historical and live data.

| Component | State |
|---|---|
| L1 funding arb | Live, hibernating in current low-funding regime (gate calibrated for retail-bull conditions) |
| L2 cascade capture | Live, concentrated universe; weights re-tuned after live regime observation |
| L3 tail hedge | Live, single BTC strangle at 26-Jun expiry |
| Allocator (dynamic L1/L2/L3 mix) | Live, signal-driven |
| Realistic dry-run simulator | Live; rejects fictional fills (no counter-party at price) |
| Hard pre-trade safeguards | Live, % of bankroll |
| OI observer + 7-day rolling history | Live across full L2 universe |

## L2 universe — concentrated on cascade-positive symbols

A 90-day backtest across 10 candidate symbols (BTC/ETH/SOL/XRP/DOGE + memes) showed the strategy's edge is sharply concentrated:

| Symbol | 90d backtest P&L | Notes |
|---|---|---|
| **PEPE** | **+$49.47** | Frequent shallow dips, fast retail-driven rebounds |
| **WIF** | **+$34.71** | Same dynamic |
| ETH | +$6.14 | Modest contributor |
| SOL | +$4.59 | Modest |
| BTC | +$3.41 | Low signal — institutional flow doesn't reflexively buy dips |
| SHIB | +$2.71 | Marginal |
| DOGE | -$2.06 | Negative |
| XRP | -$4.93 | Negative |
| BONK | -$4.87 | Negative |
| FLOKI | -$8.95 | Distribution-phase trend; "dips" kept dipping |

The current production universe is **BTC, ETH, SOL, PEPE, WIF**. The other five symbols were dropped because they showed neutral-to-negative cascade-capture P&L in the 90-day window. Empirically, mean-reversion on hour-scale dips works on (1) institutional underlyings with deep liquidity (BTC/ETH/SOL — low signal but no drag) and (2) high-vol memecoins with reflexive retail buy-the-dip behavior (PEPE/WIF). Other memes were in distribution-phase trends; their drawdowns kept extending.

## L2 cascade-probability scoring

```
score = 0.55 · oi_signal + 0.20 · funding_signal + 0.25 · compression_signal
```

- **oi_signal**: linear ramp from 70th to 95th OI percentile (vs 7-day rolling history). Leverage buildup is the actual cascade fuel.
- **funding_signal**: linear ramp on funding rate, capped at 0.001/8h.
- **compression_signal**: 24h realized vol vs 7-day realized vol — primed-for-breakout signal.

Trigger threshold: 0.6. Hold threshold: 0.4. Weights were re-tuned from `0.4/0.4/0.2` after live observation showed cascade-primed setups (max OI percentile + meaningful vol compression) being blocked because funding wasn't retail-bullish.

## What didn't work — recorded so we don't redo it

- **Tail rungs at -6% and -10%** as cascade insurance: 90-day backtest measured **negative EV (-$1.37 over the window)**. The fundamental issue is that tail rungs inherit the same SL discipline as standard rungs, so during a real cascade they fill at -10% and immediately stop out at -13%. They're a second bet on the same dynamics, not insurance. Real cascades have momentum; they don't snap back from extreme depth on hour scales.
- **Lowering L1 thresholds to capture bond-like 3–5% APR in calm regimes**: ~$50/year incremental at current funding. Real edge returns when funding regimes shift; the conservative gate doesn't cost much in opportunity but protects against negative-basis traps.

## Bankroll-relative sizing

Every dollar amount in the system is derived from `bankroll_usd × pct`. Bumping `bankroll_usd` is the *single* config change to grow the operation — no other knobs need to move.

| Knob | % of bankroll | $2k bankroll | $5k bankroll | $50k bankroll |
|---|---|---|---|---|
| L1 alloc | 60% | $1,200 | $3,000 | $30,000 |
| L2 alloc | 25% | $500 | $1,250 | $12,500 |
| L3 alloc | 15% | $300 | $750 | $7,500 |
| L2 ladder total | 20% | $400 | $1,000 | $10,000 |
| Max single-position | 25% | $500 | $1,250 | $12,500 |
| Max total deployed | 50% | $1,000 | $2,500 | $25,000 |
| Min position floor | 1% | $20 | $50 | $500 |
| L3 daily theta budget | 0.5% | $10 | $25 | $250 |

The hardcoded `if budget < 20.0` floors that lived in the executors are gone — that minimum is now `bankroll × min_position_pct`.

## Setup

```bash
cd ~/cryptarch
uv venv
uv pip install -e ".[dev]"
cp .env.example .env
createdb cryptarch
uv run python -m cryptarch init-db
```

## CLI

```bash
uv run python -m cryptarch status      # bankroll, cash, at-risk, layers enabled
uv run python -m cryptarch audit       # read-only exchange connectivity check
uv run python -m cryptarch scan        # one-shot L1 funding-arb candidate dump
uv run python -m cryptarch run         # start the engine (foreground)
```

For continuous operation a LaunchAgent + wrapper script is provided in `scripts/`.

## Run tests

```bash
uv run pytest tests/
```

193 tests passing across all three layers, the allocator, the safeguards, the realistic simulator, the OI observer, and the bankroll-scaling invariants.

## Safety architecture

`enable_live_orders=false` by default. Until set to `true`, **no real orders will be submitted regardless of any other config**. The safeguard module enforces this at submit time, not just at config time.

Hard caps (enforced as pure-function checks in `core/safeguards.py`):

| Invariant | Default | Source |
|-----------|---------|--------|
| Single-position notional | ≤ 25% of bankroll | `max_per_position_pct` |
| Total at-risk across all positions | ≤ 50% of bankroll | `max_total_deployed_pct` |
| Per-layer notional | within layer's allocation (60/25/15) | `alloc_layer_*_pct` |
| Idempotency | every order has unique `client_order_id` | `safeguards.recent_client_order_ids` |
| Live-order gate | requires `enable_live_orders=true` AND layer enabled | both must be true |

## Architecture

```
cryptarch/
├── core/
│   ├── config.py              # pydantic settings; bankroll-relative knobs
│   ├── safeguards.py          # pre-trade invariant checks (pure functions)
│   ├── allocator.py           # dynamic L1/L2/L3 split based on signal strength
│   ├── attribution.py         # per-layer P&L attribution
│   ├── engine.py              # cycle scheduler + per-cycle timeout
│   └── risk.py                # at-risk reconciliation
├── exchanges/
│   ├── base.py                # abstract ExchangeClient
│   ├── ccxt_client.py         # CCXT-backed implementation
│   ├── deribit_options.py     # L3 options client
│   └── pool.py                # exchange-client lifecycle
├── sim/
│   └── realistic.py           # fill simulator that doesn't lie
├── strategies/
│   ├── l1_funding.py          # delta-neutral basis math (pure)
│   ├── l1_executor.py         # L1 orchestration
│   ├── l2_cascade.py          # cascade-probability + ladder design (pure)
│   ├── l2_executor.py         # L2 orchestration + universe definition
│   ├── l2_signal.py           # CascadeSignal + OIObserver (live-data wiring)
│   ├── l3_tail.py             # strangle math (pure)
│   ├── l3_executor.py         # L3 orchestration
│   └── allocator_executor.py  # signal-driven dynamic allocation
├── db/
│   ├── schema.sql             # postgres
│   └── store.py               # async DB layer
└── __main__.py                # CLI: init-db, status, audit, scan, run

scripts/
├── ai.cryptarch.trader.plist  # macOS LaunchAgent
├── run_cryptarch.sh           # wrapper (waits for postgres, caffeinate)
├── probe_l2_signal.py         # one-shot: live cascade scores per symbol
└── backtest_l2_tail_rungs.py  # 90-day backtest of ladder design variants
```

## Lessons applied from previous trading bot designs

1. **Realistic dry-run from day one.** The `sim/realistic.py` module rejects "fills" when no real counter-party is at our price. Polybot's pre-v12.5 sim produced $686 of fictional P&L by pretending unfillable orders filled.
2. **Per-cycle hard timeout** so a stuck exchange doesn't deadlock the engine.
3. **Hard caps at submit time, not just config**. `safeguards.check_order` runs before every order; impossible to bypass.
4. **Pure-function math separated from I/O**. Safeguards, simulator math, layer math — all testable without exchange connections.
5. **Idempotency on every order**. Every order has a unique `client_order_id`; duplicates are rejected.
6. **Manual approval for first live trade**. Before flipping `enable_live_orders=true`, an audit runs.
7. **Measure don't reason about leverage.** Strategy refinements are validated against historical data before deployment, not justified by mechanism arguments alone. The tail-rung idea above is recorded specifically because mechanism intuition predicted positive EV and the data said otherwise.
