"""Tests for Layer 2 cascade math. Pure-function correctness of the
cascade probability formula, ladder design, and TP/SL prices."""
from __future__ import annotations

import pytest

from cryptarch.strategies.l2_cascade import (
    LadderRung, Ladder, cascade_probability, design_ladder,
    percentile_rank, realized_vol_from_closes,
    should_refresh_ladder, stop_loss_price, take_profit_price,
)


# ── cascade probability ──


class TestCascadeProbability:
    def test_max_score_when_all_three_signals_max(self):
        # OI 95+, funding 0.001+, recent vol heavily compressed (ratio ≤ 0.3)
        score = cascade_probability(
            oi_percentile=95.0,
            funding_rate_24h_avg=0.0010,
            recent_24h_vol_pct=0.015,
            historical_7d_vol_pct=0.05,
        )
        assert score == pytest.approx(1.0, abs=0.02)

    def test_zero_score_when_no_signals(self):
        # OI 50 (mid-pack), funding 0, recent vol > historical
        score = cascade_probability(
            oi_percentile=50.0,
            funding_rate_24h_avg=0.0,
            recent_24h_vol_pct=0.10,
            historical_7d_vol_pct=0.05,
        )
        assert score == pytest.approx(0.0, abs=0.01)

    def test_oi_signal_dominates_at_high_oi(self):
        score = cascade_probability(
            oi_percentile=95.0, funding_rate_24h_avg=0,
            recent_24h_vol_pct=0.05, historical_7d_vol_pct=0.05,
        )
        # OI alone gives 0.4 weight × 1.0 = 0.4
        assert score == pytest.approx(0.4, abs=0.05)

    def test_funding_signal_dominates_at_high_funding(self):
        score = cascade_probability(
            oi_percentile=70.0,
            funding_rate_24h_avg=0.0010,
            recent_24h_vol_pct=0.05,
            historical_7d_vol_pct=0.05,
        )
        # Funding alone gives 0.4 × 1.0 = 0.4
        assert score == pytest.approx(0.4, abs=0.05)

    def test_compression_alone_is_modest(self):
        # No leverage signals, but vol compressed 50%.
        score = cascade_probability(
            oi_percentile=50.0,
            funding_rate_24h_avg=0.0,
            recent_24h_vol_pct=0.025,
            historical_7d_vol_pct=0.05,
        )
        # Compression: ratio 0.5 → (1-0.5)/0.7 = 0.714; weighted 0.2 = 0.143
        assert score == pytest.approx(0.143, abs=0.02)

    def test_zero_historical_vol_handled_safely(self):
        # Edge: division by zero
        score = cascade_probability(
            oi_percentile=80.0, funding_rate_24h_avg=0.0005,
            recent_24h_vol_pct=0.05, historical_7d_vol_pct=0.0,
        )
        # Should not crash; compression signal = 0
        assert 0 <= score <= 1


# ── ladder design ──


class TestDesignLadder:
    def test_default_ladder_sizes_sum_to_total(self):
        ladder = design_ladder(spot=100.0, levels=4, total_usd=200.0)
        assert sum(r.size_usd for r in ladder.rungs) == pytest.approx(200.0)

    def test_rungs_at_correct_depths(self):
        ladder = design_ladder(spot=100.0, levels=4, total_usd=200.0, deepest_pct=0.04)
        # Linear: 1%, 2%, 3%, 4%
        assert ladder.rungs[0].pct_below == pytest.approx(0.01)
        assert ladder.rungs[1].pct_below == pytest.approx(0.02)
        assert ladder.rungs[2].pct_below == pytest.approx(0.03)
        assert ladder.rungs[3].pct_below == pytest.approx(0.04)

    def test_rung_prices_match_pct_depths(self):
        ladder = design_ladder(spot=100.0, levels=4, total_usd=200.0, deepest_pct=0.04)
        assert ladder.rungs[0].limit_price == pytest.approx(99.0)
        assert ladder.rungs[1].limit_price == pytest.approx(98.0)
        assert ladder.rungs[2].limit_price == pytest.approx(97.0)
        assert ladder.rungs[3].limit_price == pytest.approx(96.0)

    def test_deeper_rungs_have_more_size(self):
        ladder = design_ladder(spot=100.0, levels=4, total_usd=200.0, size_decay=1.5)
        sizes = [r.size_usd for r in ladder.rungs]
        # Sizes should be monotonically increasing (deeper = bigger)
        for i in range(len(sizes) - 1):
            assert sizes[i + 1] > sizes[i]

    def test_uniform_size_decay_1_yields_equal_rungs(self):
        ladder = design_ladder(spot=100.0, levels=4, total_usd=200.0, size_decay=1.0)
        sizes = [r.size_usd for r in ladder.rungs]
        for s in sizes:
            assert s == pytest.approx(50.0)    # 200/4

    def test_zero_spot_raises(self):
        with pytest.raises(ValueError):
            design_ladder(spot=0, levels=4, total_usd=200.0)

    def test_negative_levels_raises(self):
        with pytest.raises(ValueError):
            design_ladder(spot=100, levels=-1, total_usd=200.0)

    def test_zero_usd_raises(self):
        with pytest.raises(ValueError):
            design_ladder(spot=100, levels=4, total_usd=0)

    def test_too_deep_pct_raises(self):
        with pytest.raises(ValueError):
            design_ladder(spot=100, levels=4, total_usd=200.0, deepest_pct=0.6)

    def test_high_decay_concentrates_at_bottom(self):
        ladder = design_ladder(spot=100.0, levels=4, total_usd=400.0, size_decay=3.0)
        sizes = [r.size_usd for r in ladder.rungs]
        # With 3^i weights: 1, 3, 9, 27 = 40; bottom rung gets 27/40 × 400 = $270
        assert sizes[3] > sizes[0] * 20    # massive concentration


# ── TP / SL ──


class TestTakeProfitPrice:
    def test_simple_pct_lift(self):
        assert take_profit_price(100.0, 0.012) == pytest.approx(101.2)

    def test_zero_price_raises(self):
        with pytest.raises(ValueError):
            take_profit_price(0, 0.012)

    def test_zero_pct_raises(self):
        with pytest.raises(ValueError):
            take_profit_price(100, 0)


class TestStopLossPrice:
    def test_simple_pct_drop(self):
        assert stop_loss_price(100.0, 0.03) == pytest.approx(97.0)

    def test_pct_at_one_raises(self):
        with pytest.raises(ValueError):
            stop_loss_price(100, 1.0)

    def test_negative_pct_raises(self):
        with pytest.raises(ValueError):
            stop_loss_price(100, -0.01)


# ── refresh ──


class TestShouldRefreshLadder:
    def test_no_refresh_for_small_drift(self):
        assert not should_refresh_ladder(spot_now=100.5, spot_at_design=100.0)

    def test_refresh_when_drift_exceeds_threshold(self):
        # 2% drift, threshold 1%
        assert should_refresh_ladder(spot_now=102.0, spot_at_design=100.0)

    def test_refresh_handles_downward_drift(self):
        assert should_refresh_ladder(spot_now=98.0, spot_at_design=100.0)

    def test_refresh_with_zero_design_price(self):
        # Bad state — refresh defensively.
        assert should_refresh_ladder(spot_now=100.0, spot_at_design=0.0)


# ── percentile rank ──


class TestPercentileRank:
    def test_value_below_all_history_is_0(self):
        assert percentile_rank(50, [60, 70, 80, 90, 100]) == 0

    def test_value_above_all_history_is_100(self):
        assert percentile_rank(110, [60, 70, 80, 90, 100]) == 100

    def test_median_is_50(self):
        # 80 is at-or-below 3 of 5 values (60, 70, 80) → 60%
        # The "at-or-below" definition for percentile rank.
        assert percentile_rank(80, [60, 70, 80, 90, 100]) == 60

    def test_empty_history_returns_50(self):
        assert percentile_rank(100, []) == 50

    def test_handles_ties_inclusively(self):
        # 5 of 5 are at-or-below 100 → 100%
        assert percentile_rank(100, [60, 70, 80, 90, 100]) == 100


# ── realized vol from closes ──


class TestRealizedVol:
    def test_constant_prices_have_zero_vol(self):
        assert realized_vol_from_closes([100, 100, 100, 100]) == pytest.approx(0)

    def test_steady_growth_has_zero_vol(self):
        # Each return is identical → variance = 0 → σ = 0
        # 100 → 110 → 121 → 133.1: each return is +10%
        prices = [100, 110, 121, 133.1]
        assert realized_vol_from_closes(prices) == pytest.approx(0, abs=1e-9)

    def test_volatile_prices_have_positive_vol(self):
        # Up 5, down 5, up 5, down 5 → mean ≈ 0, variance > 0
        prices = [100, 105, 100, 105, 100]
        assert realized_vol_from_closes(prices) > 0

    def test_lookback_takes_last_n(self):
        # First two are flat; last three are volatile. Lookback=3 should
        # only see the volatile portion.
        prices = [100, 100, 100, 110, 95]
        full = realized_vol_from_closes(prices)
        last3 = realized_vol_from_closes(prices, lookback=3)
        assert last3 > full    # last 3 includes the volatile transitions

    def test_short_input_returns_zero(self):
        assert realized_vol_from_closes([100]) == 0
        assert realized_vol_from_closes([]) == 0

    def test_zero_or_negative_prices_skipped(self):
        # Bad data points are silently skipped, not crashing.
        prices = [100, 0, 100, -5, 100]
        # Only valid log-return pairs survive; should not raise.
        assert realized_vol_from_closes(prices) >= 0
