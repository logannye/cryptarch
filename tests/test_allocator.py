"""Tests for the dynamic capital allocator."""
from __future__ import annotations

import pytest

from cryptarch.core.allocator import (
    AllocationDecision, LayerSignals, compute_target_allocation,
)


def _signals(
    l1_apr: float = 0.0,
    l2_cascade: float = 0.0,
    l3_iv_compression: float = 0.0,
) -> LayerSignals:
    return LayerSignals(
        l1_max_apr_pct=l1_apr,
        l2_max_cascade_score=l2_cascade,
        l3_iv_compression_score=l3_iv_compression,
    )


# ── baseline behavior ──


class TestBaseline:
    def test_zero_signals_returns_baseline(self):
        decision = compute_target_allocation(_signals())
        assert decision.l1_pct == pytest.approx(0.60)
        assert decision.l2_pct == pytest.approx(0.25)
        assert decision.l3_pct == pytest.approx(0.15)
        assert decision.rationale == "no_signal_baseline"

    def test_allocation_always_sums_to_one(self):
        # Various signal combinations
        for s in [
            _signals(l1_apr=100, l2_cascade=0, l3_iv_compression=0),
            _signals(l1_apr=0, l2_cascade=1.0, l3_iv_compression=0),
            _signals(l1_apr=0, l2_cascade=0, l3_iv_compression=1.0),
            _signals(l1_apr=100, l2_cascade=1.0, l3_iv_compression=1.0),
            _signals(l1_apr=25, l2_cascade=0.5, l3_iv_compression=0.5),
        ]:
            d = compute_target_allocation(s)
            assert d.l1_pct + d.l2_pct + d.l3_pct == pytest.approx(1.0, abs=1e-9)


# ── layer-specific tilts ──


class TestL1Tilt:
    def test_high_funding_apr_tilts_to_l1(self):
        # 50% APR → score 1.0 → max tilt
        d = compute_target_allocation(_signals(l1_apr=50))
        assert d.l1_pct > 0.60
        # Other layers shrink
        assert d.l2_pct < 0.25
        assert d.l3_pct < 0.15

    def test_l1_tilt_capped_at_max_pp(self):
        # Even at 200% APR, tilt is capped at +20pp from baseline
        d = compute_target_allocation(_signals(l1_apr=200), max_tilt_pp=0.20)
        # Base 0.60 + tilt 0.20 = 0.80; after renormalization may be slightly lower
        assert d.l1_pct <= 0.80 + 0.01

    def test_low_apr_below_threshold_no_tilt(self):
        # 5% APR → score 0.1 → barely any tilt
        d = compute_target_allocation(_signals(l1_apr=5))
        assert d.l1_pct == pytest.approx(0.60, abs=0.03)


class TestL2Tilt:
    def test_high_cascade_score_tilts_to_l2(self):
        d = compute_target_allocation(_signals(l2_cascade=1.0))
        assert d.l2_pct > 0.25
        assert d.l1_pct < 0.60
        assert d.l3_pct < 0.15

    def test_l2_score_passthrough(self):
        # L2 cascade score is already 0-1 (vs L1 needs APR conversion)
        d_half = compute_target_allocation(_signals(l2_cascade=0.5))
        d_full = compute_target_allocation(_signals(l2_cascade=1.0))
        assert d_full.l2_pct > d_half.l2_pct


class TestL3Tilt:
    def test_iv_compression_threshold(self):
        # Below 0.3 compression: no tilt
        d_low = compute_target_allocation(_signals(l3_iv_compression=0.2))
        assert d_low.l3_pct == pytest.approx(0.15, abs=0.001)

    def test_iv_compression_at_max(self):
        # 1.0 compression: full tilt to L3
        d = compute_target_allocation(_signals(l3_iv_compression=1.0))
        assert d.l3_pct > 0.15


# ── floor constraint ──


class TestFloors:
    def test_no_layer_below_50pct_of_baseline(self):
        # If L1 has max signal, L2 and L3 should still get ≥ 0.5 × baseline
        d = compute_target_allocation(_signals(l1_apr=200))
        assert d.l2_pct >= 0.25 * 0.5 - 0.01    # ≥ 12.5%
        assert d.l3_pct >= 0.15 * 0.5 - 0.01    # ≥ 7.5%

    def test_extreme_tilt_doesnt_starve_layers(self):
        # All three at max simultaneously: should distribute
        d = compute_target_allocation(_signals(
            l1_apr=200, l2_cascade=1.0, l3_iv_compression=1.0,
        ))
        # No layer goes to zero
        assert d.l1_pct > 0.30
        assert d.l2_pct > 0.10
        assert d.l3_pct > 0.05


# ── edge cases ──


class TestValidation:
    def test_invalid_base_sum_raises(self):
        with pytest.raises(ValueError):
            compute_target_allocation(
                _signals(), base_l1=0.5, base_l2=0.3, base_l3=0.3,    # sums to 1.1
            )

    def test_invalid_tilt_raises(self):
        with pytest.raises(ValueError):
            compute_target_allocation(_signals(), max_tilt_pp=-0.1)
        with pytest.raises(ValueError):
            compute_target_allocation(_signals(), max_tilt_pp=0.6)


class TestRationale:
    def test_rationale_describes_dominant_signal(self):
        d = compute_target_allocation(_signals(l1_apr=80))
        assert "L1↑" in d.rationale

    def test_rationale_lists_multiple_signals(self):
        d = compute_target_allocation(_signals(l1_apr=80, l2_cascade=0.7))
        assert "L1↑" in d.rationale
        assert "L2↑" in d.rationale
