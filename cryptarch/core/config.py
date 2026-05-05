"""Settings — pydantic-settings backed by .env."""
from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Exchange credentials ──
    # Kraken: US-legal, single venue with both spot and perpetual futures
    kraken_api_key: str = ""
    kraken_api_secret: str = ""
    # Binance/Bybit/OKX: outside the US only (geo-blocked from US without VPN)
    binance_api_key: str = ""
    binance_api_secret: str = ""
    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    okx_api_key: str = ""
    okx_api_secret: str = ""
    okx_api_passphrase: str = ""
    coinbase_api_key: str = ""
    coinbase_api_secret: str = ""
    deribit_api_key: str = ""
    deribit_api_secret: str = ""

    # ── Storage ──
    database_url: str = "postgresql://localhost/cryptarch"

    # ── Master safety ──
    enable_live_orders: bool = False
    layer_1_funding_arb_enabled: bool = True
    layer_2_cascade_capture_enabled: bool = False
    layer_3_tail_hedge_enabled: bool = False

    # ── Bankroll + allocation ──
    # The bankroll is the ONLY canonical $ knob. Every other dollar
    # amount in the system is derived from bankroll × pct, so a $50k
    # bankroll automatically moves 10× the notional of a $5k one with
    # no code changes. Do not introduce hardcoded $ values elsewhere.
    bankroll_usd: float = 2000.0
    alloc_layer_1_pct: float = 0.60
    alloc_layer_2_pct: float = 0.25
    alloc_layer_3_pct: float = 0.15

    # ── Hard caps (% of bankroll) ──
    max_total_deployed_pct: float = 0.50
    max_per_position_pct: float = 0.25      # max single-position notional
    min_position_pct: float = 0.01          # min trade size; below this, fees dominate

    # ── Layer 1 ──
    l1_min_funding_rate_8h: float = 0.0003
    l1_min_basis_pct: float = 0.001
    l1_max_basis_pct: float = 0.020

    # ── Layer 2 ──
    l2_ladder_levels: int = 4
    # Per-ladder notional as % of bankroll. 0.20 = 20% of $2k = $400 at
    # default. Concentrated on PEPE/WIF where the 90-day backtest showed
    # actual edge. Per-rung max stays under max_per_position_pct.
    l2_ladder_pct: float = 0.20
    l2_take_profit_pct: float = 0.012
    l2_stop_loss_pct: float = 0.030

    # ── Layer 3 ──
    # Daily theta cost as % of bankroll. 0.005 = 0.5%/day = max $10/day
    # at $2k. Drives strangle contract sizing through theta-budget math.
    l3_daily_theta_budget_pct: float = 0.005
    l3_target_days_to_expiry: int = 45
    l3_otm_pct: float = 0.20

    # ── Cadence ──
    scan_interval_seconds: int = 30
    reconcile_interval_seconds: int = 300

    # ── Alerts ──
    resend_api_key: str = ""
    alert_email: str = ""

    @property
    def alloc_layer_1_usd(self) -> float:
        return self.bankroll_usd * self.alloc_layer_1_pct

    @property
    def alloc_layer_2_usd(self) -> float:
        return self.bankroll_usd * self.alloc_layer_2_pct

    @property
    def alloc_layer_3_usd(self) -> float:
        return self.bankroll_usd * self.alloc_layer_3_pct

    @property
    def max_total_deployed_usd(self) -> float:
        return self.bankroll_usd * self.max_total_deployed_pct

    @property
    def max_per_position_usd(self) -> float:
        return self.bankroll_usd * self.max_per_position_pct

    @property
    def min_position_usd(self) -> float:
        return self.bankroll_usd * self.min_position_pct

    @property
    def l2_ladder_total_usd(self) -> float:
        return self.bankroll_usd * self.l2_ladder_pct

    @property
    def l3_daily_theta_budget_usd(self) -> float:
        return self.bankroll_usd * self.l3_daily_theta_budget_pct

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
