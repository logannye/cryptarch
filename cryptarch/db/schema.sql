-- cryptarch v0.1 schema. Apply with:
--   uv run python -m cryptarch init-db

-- ── system state (single row, id=1) ──────────────────────────────
CREATE TABLE IF NOT EXISTS system_state (
    id                       INTEGER PRIMARY KEY DEFAULT 1,
    bankroll_usd             NUMERIC(14,2) NOT NULL,
    cash_usd                 NUMERIC(14,2) NOT NULL,
    total_at_risk_usd        NUMERIC(14,2) NOT NULL DEFAULT 0,
    daily_pnl                NUMERIC(14,4) NOT NULL DEFAULT 0,
    cumulative_pnl           NUMERIC(14,4) NOT NULL DEFAULT 0,
    deployment_stage         TEXT NOT NULL DEFAULT 'dry_run',
    -- 'dry_run' = simulator-only against real prices
    -- 'micro_test' = live but per-position cap reduced
    -- 'live' = full operation
    halt_reason              TEXT,    -- if non-null, no new entries until cleared
    -- Dynamic allocations (Phase 4). When NULL, the executor falls back to
    -- the static settings.alloc_layer_*_pct. When the AllocatorExecutor
    -- updates these, executors prefer the dynamic value.
    dynamic_alloc_l1_pct     NUMERIC(5,4),
    dynamic_alloc_l2_pct     NUMERIC(5,4),
    dynamic_alloc_l3_pct     NUMERIC(5,4),
    dynamic_alloc_rationale  TEXT,
    dynamic_alloc_updated_at TIMESTAMPTZ,
    last_updated             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (id = 1)
);

INSERT INTO system_state (id, bankroll_usd, cash_usd, deployment_stage)
VALUES (1, 2000.00, 2000.00, 'dry_run')
ON CONFLICT (id) DO NOTHING;


-- ── positions (logical position; may span multiple legs/orders) ──
CREATE TABLE IF NOT EXISTS position (
    id                  SERIAL PRIMARY KEY,
    layer               TEXT NOT NULL,    -- 'l1_funding' | 'l2_cascade' | 'l3_tail'
    strategy_group_id   TEXT NOT NULL,    -- groups multi-leg positions (e.g. spot+perp pair)
    state               TEXT NOT NULL,    -- 'opening' | 'open' | 'closing' | 'closed'
    opened_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at           TIMESTAMPTZ,
    notional_usd        NUMERIC(14,2) NOT NULL,
    realized_pnl_usd    NUMERIC(14,4) NOT NULL DEFAULT 0,
    metadata            JSONB NOT NULL DEFAULT '{}'    -- strategy-specific context
);

CREATE INDEX IF NOT EXISTS idx_pos_state ON position(state, layer);
CREATE INDEX IF NOT EXISTS idx_pos_group ON position(strategy_group_id);


-- ── fills (each individual order's outcome) ──────────────────────
CREATE TABLE IF NOT EXISTS fill (
    id                   SERIAL PRIMARY KEY,
    position_id          INTEGER REFERENCES position(id) ON DELETE SET NULL,
    layer                TEXT NOT NULL,
    exchange             TEXT NOT NULL,
    symbol               TEXT NOT NULL,
    side                 TEXT NOT NULL,    -- 'buy' | 'sell'
    order_type           TEXT NOT NULL,    -- 'limit' | 'market'
    size_base            NUMERIC(20,10) NOT NULL,
    size_usd             NUMERIC(14,4) NOT NULL,
    fill_price           NUMERIC(20,10) NOT NULL,
    fee_usd              NUMERIC(10,4) NOT NULL DEFAULT 0,
    is_simulated         BOOLEAN NOT NULL,    -- true = dry-run record, false = real fill
    sim_reason           TEXT,                -- if simulated: 'maker_filled_at_limit' etc
    client_order_id      TEXT NOT NULL,       -- idempotency key (simulated or real)
    exchange_order_id    TEXT,                -- exchange's id (null for simulated)
    placed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (client_order_id)
);

CREATE INDEX IF NOT EXISTS idx_fill_position ON fill(position_id);
CREATE INDEX IF NOT EXISTS idx_fill_placed   ON fill(placed_at DESC);


-- ── funding events (Layer 1 yield tracking) ──────────────────────
CREATE TABLE IF NOT EXISTS funding_event (
    id              SERIAL PRIMARY KEY,
    position_id     INTEGER REFERENCES position(id) ON DELETE SET NULL,
    exchange        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    rate_8h         NUMERIC(10,8) NOT NULL,
    notional_usd    NUMERIC(14,4) NOT NULL,
    payment_usd     NUMERIC(14,4) NOT NULL,    -- positive = we received, negative = we paid
    paid_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_simulated    BOOLEAN NOT NULL
);


-- ── cascade events (Layer 2 backtest + live tracking) ────────────
CREATE TABLE IF NOT EXISTS cascade_event (
    id                  SERIAL PRIMARY KEY,
    exchange            TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    detected_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pre_price           NUMERIC(20,10) NOT NULL,
    trough_price        NUMERIC(20,10),
    trough_pct_drop     NUMERIC(8,4),
    recovery_pct        NUMERIC(8,4),
    oi_percentile       NUMERIC(5,2),
    funding_rate_pre    NUMERIC(10,8),
    metadata            JSONB NOT NULL DEFAULT '{}'
);


-- ── OI observations (Layer 2 cascade signal input) ──────────────
-- Rolling 7-day window of OI snapshots; cascade_probability uses the
-- percentile of current OI vs the recent history.
CREATE TABLE IF NOT EXISTS oi_observation (
    id                  SERIAL PRIMARY KEY,
    exchange            TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    open_interest_usd   NUMERIC(18,2) NOT NULL,
    observed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_oi_obs ON oi_observation (exchange, symbol, observed_at DESC);


-- ── option positions (Layer 3) ───────────────────────────────────
CREATE TABLE IF NOT EXISTS option_position (
    id                  SERIAL PRIMARY KEY,
    position_id         INTEGER REFERENCES position(id) ON DELETE CASCADE,
    exchange            TEXT NOT NULL,
    instrument_name     TEXT NOT NULL,    -- e.g. "BTC-31MAY26-50000-C"
    underlying          TEXT NOT NULL,
    expiry              TIMESTAMPTZ NOT NULL,
    strike              NUMERIC(20,2) NOT NULL,
    option_type         TEXT NOT NULL,    -- 'C' | 'P'
    contracts           NUMERIC(14,4) NOT NULL,
    entry_iv            NUMERIC(8,4),
    entry_premium_usd   NUMERIC(14,4) NOT NULL,
    UNIQUE (instrument_name, position_id)
);
