"""Tests for the per-layer P&L attribution computation."""
from __future__ import annotations

import pytest

from cryptarch.core.attribution import (
    AttributionReport, LayerPnL, compute_attribution,
)


class TestLayerPnL:
    def test_total_sums_components(self):
        pnl = LayerPnL(
            layer="l1_funding",
            realized_pnl_usd=10.0, funding_pnl_usd=5.0,
            unrealized_pnl_usd=2.0, fees_pnl_usd=-1.0,
            n_trades_closed=3, n_open_at_close=1,
        )
        assert pnl.total_pnl_usd == 16.0


class TestComputeAttribution:
    def test_basic_three_layer_decomposition(self):
        report = compute_attribution(
            realized_by_layer={"l1_funding": 5.0, "l2_cascade": 12.0, "l3_tail": 0.0},
            funding_by_layer={"l1_funding": 3.0},
            unrealized_by_layer={"l1_funding": 1.0, "l3_tail": -0.5},
            fees_by_layer={"l1_funding": -0.20, "l2_cascade": -0.05},
            n_closed_by_layer={"l1_funding": 2, "l2_cascade": 4, "l3_tail": 0},
            n_open_by_layer={"l1_funding": 1, "l2_cascade": 0, "l3_tail": 1},
            bankroll_start_usd=2000.0,
            bankroll_end_usd=2020.25,
            window_start="2026-05-02T00:00:00Z",
            window_end="2026-05-03T00:00:00Z",
        )
        assert len(report.layers) == 3
        # l1_funding total = 5 + 3 + 1 - 0.20 = 8.80
        l1 = next(l for l in report.layers if l.layer == "l1_funding")
        assert l1.total_pnl_usd == pytest.approx(8.80)
        # l2_cascade total = 12 + 0 + 0 - 0.05 = 11.95
        l2 = next(l for l in report.layers if l.layer == "l2_cascade")
        assert l2.total_pnl_usd == pytest.approx(11.95)
        # l3_tail total = 0 + 0 - 0.5 + 0 = -0.5
        l3 = next(l for l in report.layers if l.layer == "l3_tail")
        assert l3.total_pnl_usd == pytest.approx(-0.5)
        # Total = 8.80 + 11.95 - 0.5 = 20.25
        assert report.total_pnl_usd == pytest.approx(20.25)

    def test_return_pct_calculation(self):
        report = compute_attribution(
            realized_by_layer={"l1_funding": 20.0},
            funding_by_layer={}, unrealized_by_layer={}, fees_by_layer={},
            n_closed_by_layer={}, n_open_by_layer={},
            bankroll_start_usd=2000.0, bankroll_end_usd=2020.0,
            window_start="x", window_end="y",
        )
        assert report.return_pct == pytest.approx(0.01)    # 1%

    def test_zero_bankroll_safe(self):
        report = compute_attribution(
            realized_by_layer={}, funding_by_layer={},
            unrealized_by_layer={}, fees_by_layer={},
            n_closed_by_layer={}, n_open_by_layer={},
            bankroll_start_usd=0.0, bankroll_end_usd=0.0,
            window_start="x", window_end="y",
        )
        assert report.return_pct == 0.0

    def test_best_and_worst_layer(self):
        report = compute_attribution(
            realized_by_layer={"l1_funding": 5.0, "l2_cascade": 50.0, "l3_tail": -2.0},
            funding_by_layer={}, unrealized_by_layer={}, fees_by_layer={},
            n_closed_by_layer={}, n_open_by_layer={},
            bankroll_start_usd=1000, bankroll_end_usd=1053,
            window_start="x", window_end="y",
        )
        best = report.best_layer()
        worst = report.worst_layer()
        assert best is not None and best.layer == "l2_cascade"
        assert worst is not None and worst.layer == "l3_tail"

    def test_empty_layers_returns_none_for_best_worst(self):
        report = compute_attribution(
            realized_by_layer={}, funding_by_layer={},
            unrealized_by_layer={}, fees_by_layer={},
            n_closed_by_layer={}, n_open_by_layer={},
            bankroll_start_usd=1000, bankroll_end_usd=1000,
            window_start="x", window_end="y",
        )
        assert report.best_layer() is None
        assert report.worst_layer() is None
