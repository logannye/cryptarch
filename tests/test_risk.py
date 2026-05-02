"""Tests for cross-layer risk netting."""
from __future__ import annotations

import pytest

from cryptarch.core.risk import compute_portfolio_risk, netting_efficiency


class TestNetting:
    def test_empty_portfolio(self):
        risk = compute_portfolio_risk([])
        assert risk.gross_at_risk_usd == 0
        assert risk.net_at_risk_usd == 0
        assert netting_efficiency(risk) == 0

    def test_unhedged_long_position(self):
        positions = [
            {"underlying": "BTC", "notional_usd": 1000, "direction": "long"},
        ]
        risk = compute_portfolio_risk(positions)
        assert risk.gross_at_risk_usd == 1000
        assert risk.net_at_risk_usd == 1000
        assert netting_efficiency(risk) == 0

    def test_l1_hedged_position_nets_to_zero(self):
        # L1 funding-arb: long spot + short perp on same underlying
        positions = [
            {"underlying": "BTC", "notional_usd": 1000, "direction": "long"},   # spot
            {"underlying": "BTC", "notional_usd": 1000, "direction": "short"},  # perp
        ]
        risk = compute_portfolio_risk(positions)
        assert risk.gross_at_risk_usd == 2000
        assert risk.net_at_risk_usd == 0
        assert netting_efficiency(risk) == 1.0

    def test_l1_plus_l2_partial_netting(self):
        # L1 BTC: $1000 long spot + $1000 short perp (net 0)
        # L2 BTC: $500 long spot ladder filled
        positions = [
            {"underlying": "BTC", "notional_usd": 1000, "direction": "long"},
            {"underlying": "BTC", "notional_usd": 1000, "direction": "short"},
            {"underlying": "BTC", "notional_usd": 500,  "direction": "long"},
        ]
        risk = compute_portfolio_risk(positions)
        assert risk.gross_at_risk_usd == 2500
        # Long total = 1500, short = 1000 → net = 500
        assert risk.net_at_risk_usd == 500
        assert netting_efficiency(risk) == pytest.approx(0.8)    # 1 - 500/2500

    def test_multi_underlying_independent(self):
        # BTC hedged perfectly; ETH unhedged long
        positions = [
            {"underlying": "BTC", "notional_usd": 1000, "direction": "long"},
            {"underlying": "BTC", "notional_usd": 1000, "direction": "short"},
            {"underlying": "ETH", "notional_usd": 500,  "direction": "long"},
        ]
        risk = compute_portfolio_risk(positions)
        assert risk.gross_at_risk_usd == 2500
        # BTC: net 0; ETH: net 500
        assert risk.net_at_risk_usd == 500

    def test_exposures_sorted_by_magnitude(self):
        positions = [
            {"underlying": "BTC", "notional_usd": 100, "direction": "long"},
            {"underlying": "ETH", "notional_usd": 1000, "direction": "long"},
            {"underlying": "SOL", "notional_usd": 500, "direction": "long"},
        ]
        risk = compute_portfolio_risk(positions)
        assert risk.underlyings[0].underlying == "ETH"    # largest
        assert risk.underlyings[-1].underlying == "BTC"   # smallest


class TestEfficiency:
    def test_perfect_hedge(self):
        positions = [
            {"underlying": "BTC", "notional_usd": 100, "direction": "long"},
            {"underlying": "BTC", "notional_usd": 100, "direction": "short"},
        ]
        assert netting_efficiency(compute_portfolio_risk(positions)) == 1.0

    def test_no_hedge(self):
        positions = [
            {"underlying": "BTC", "notional_usd": 100, "direction": "long"},
        ]
        assert netting_efficiency(compute_portfolio_risk(positions)) == 0
