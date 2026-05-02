# cryptarch

Three-layer crypto trading system. Mathematically derived to combine three near-zero-correlated edges into a high-Sharpe portfolio with positive skew (tail-heavy upside).

## The three layers

| Layer | Strategy | E[r]/day | Sharpe | Source of edge |
|-------|----------|----------|--------|----------------|
| **L1** | Funding-rate arbitrage (delta-neutral basis) | 0.05–0.15% | 3–4 | Structural perp/spot mispricing |
| **L2** | Liquidation cascade capture (mean-reversion ladder) | 0.10–0.30% (episodic) | 1–2 | Forced sellers overshoot fundamentals |
| **L3** | OTM-options tail hedge (Deribit) | -0.05% normal, +5–15% on tails | ~0/∞ | Long convexity into vol regime shifts |

Combined target: **~0.30%/day, Sharpe 3–4**, with positive skew (tail outcomes are upside).

## Status

- **Phase 0 (foundation)**: ✓ done. Project skeleton, hard safeguards, realistic dry-run simulator (no fictional fills), CCXT-backed exchange client, DB schema.
- **Phase 1 (Layer 1 — funding arb)**: in progress.
- **Phase 2 (Layer 2 — cascade capture)**: queued.
- **Phase 3 (Layer 3 — tail hedge)**: queued.
- **Phase 4–5 (allocator + hardening)**: queued.

## Setup

```bash
cd ~/cryptarch
uv venv
uv pip install -e ".[dev]"
cp .env.example .env
# Add ODDS_API_KEY only after Layer 1 is shipping; not needed for dry-run yet.
createdb cryptarch
uv run python -m cryptarch init-db
```

## Run tests

```bash
uv run pytest tests/
```

Currently 31 tests passing on Phase 0:
- `test_safeguards.py` — every safety invariant has explicit test coverage
- `test_simulator.py` — realistic-fill simulator including the polybot-pathological case (bid 0.001 / ask 0.999) it correctly rejects

## Phase 0 audit

Before any live trading, run a connectivity audit:

```bash
uv run python -m cryptarch audit
```

This is read-only. It verifies:
- Exchange APIs are reachable
- Order books are sane (not the polybot 99pp-spread pattern)
- Funding rates are accessible
- Wallet balances are visible

## Safety architecture

`enable_live_orders=false` by default. Until you set it to `true`, **no real orders will be submitted regardless of any other config**. The safeguard module enforces this at submit time, not just config-time.

Hard caps (enforced as pure-function checks in `core/safeguards.py`):

| Invariant | Default |
|-----------|---------|
| Single-position notional | ≤ $500 |
| Total at-risk across all positions | ≤ 50% of bankroll |
| Per-layer notional | within layer's allocation (60/25/15) |
| Idempotency | every order has unique `client_order_id` |
| Live-order gate | requires `enable_live_orders=true` AND layer enabled |

## Architecture

```
cryptarch/
├── core/
│   ├── config.py        # pydantic settings
│   └── safeguards.py    # pre-trade invariant checks (pure functions)
├── exchanges/
│   ├── base.py          # abstract ExchangeClient interface
│   └── ccxt_client.py   # CCXT-backed implementation
├── sim/
│   └── realistic.py     # fill simulator that doesn't lie
├── strategies/          # (Phase 1+) — l1_funding, l2_cascade, l3_tail
├── db/
│   └── schema.sql       # postgres
└── __main__.py          # CLI: init-db, status, audit
```

## Lessons applied from polybot + sportsarb

1. **Realistic dry-run from day one.** The `sim/realistic.py` module rejects "fills" when no real counter-party is at our price. Polybot's pre-v12.5 sim produced $686 of fictional PnL by pretending unfillable orders filled.
2. **Per-cycle hard timeout** (Phase 1+).
3. **Hard caps at submit time, not just config**. `safeguards.check_order` runs before every order; impossible to bypass.
4. **Pure-function math separated from I/O**. Safeguards, simulator math, future layer math — all testable without exchange connections.
5. **Idempotency on every order**. Every order must have a unique `client_order_id`; duplicates are rejected.
6. **Manual approval for first live trade** (Phase 1+). Before flipping `enable_live_orders=true`, an audit runs and the first live trade requires explicit confirmation.
