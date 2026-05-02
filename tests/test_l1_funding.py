"""Tests for Layer 1 funding-arb math. Pure-function correctness."""
from __future__ import annotations

import pytest

from cryptarch.strategies.l1_funding import (
    FundingArbCandidate, compute_hedge_deviation, is_attractive, plan_position,
    should_close_position,
)


def _candidate(
    spot_price: float = 50000.0,
    perp_price: float = 50050.0,    # 10bps premium
    funding_8h: float = 0.0003,
) -> FundingArbCandidate:
    return FundingArbCandidate(
        spot_exchange="binance",
        spot_symbol="BTC/USDT",
        perp_exchange="binance",
        perp_symbol="BTC/USDT:USDT",
        spot_price=spot_price,
        perp_price=perp_price,
        funding_rate_8h=funding_8h,
    )


# ── Basis math ──


def test_basis_positive_when_perp_above_spot():
    c = _candidate(spot_price=50000, perp_price=50100)
    assert c.basis_pct == pytest.approx(0.002)


def test_basis_negative_when_perp_below_spot():
    c = _candidate(spot_price=50000, perp_price=49950)
    assert c.basis_pct == pytest.approx(-0.001)


def test_basis_zero_when_equal():
    c = _candidate(spot_price=50000, perp_price=50000)
    assert c.basis_pct == 0


# ── Yield math ──


def test_daily_yield_at_typical_funding():
    # 0.03% / 8h × 3 / 2 = 0.045% per dollar deployed daily
    c = _candidate(funding_8h=0.0003)
    assert c.expected_daily_yield_pct == pytest.approx(0.00045)


def test_daily_yield_at_hot_funding():
    c = _candidate(funding_8h=0.0010)
    assert c.expected_daily_yield_pct == pytest.approx(0.0015)


def test_apr_calculation():
    c = _candidate(funding_8h=0.0003)
    # 0.045% daily × 365 = ~16.4% APR
    assert c.expected_apr_pct == pytest.approx(0.16425, rel=1e-3)


# ── Attractiveness gates ──


class TestIsAttractive:
    def test_passes_at_threshold_funding(self):
        c = _candidate(funding_8h=0.0003, perp_price=50010)    # 2bp basis
        assert is_attractive(c, min_funding_8h=0.0003, min_basis_pct=0.0001, max_basis_pct=0.02)

    def test_fails_below_funding_threshold(self):
        c = _candidate(funding_8h=0.0001)
        assert not is_attractive(c, min_funding_8h=0.0003, min_basis_pct=0.0, max_basis_pct=0.02)

    def test_fails_with_inverted_basis(self):
        # Perp BELOW spot — funding is paying us but we'd have to short perp at a discount
        c = _candidate(spot_price=50000, perp_price=49500, funding_8h=0.0003)
        assert not is_attractive(c, min_funding_8h=0.0003, min_basis_pct=0.0, max_basis_pct=0.02)

    def test_fails_when_basis_extreme(self):
        # Basis already 5% — high mean-reversion risk
        c = _candidate(spot_price=50000, perp_price=52500, funding_8h=0.001)
        assert not is_attractive(c, min_funding_8h=0.0003, min_basis_pct=0.001, max_basis_pct=0.02)


# ── Position planning ──


class TestPlanPosition:
    def test_splits_equally_between_legs(self):
        c = _candidate()
        plan = plan_position(c, total_capital_usd=200.0)
        assert plan.spot_notional_usd == pytest.approx(100.0)
        assert plan.perp_notional_usd == pytest.approx(100.0)

    def test_base_sizes_match_notionals(self):
        c = _candidate(spot_price=50000, perp_price=50050)
        plan = plan_position(c, total_capital_usd=200.0)
        assert plan.spot_size_base == pytest.approx(100.0 / 50000)
        assert plan.perp_size_base == pytest.approx(100.0 / 50050)

    def test_expected_pnl_matches_yield_formula(self):
        c = _candidate(funding_8h=0.0003)
        plan = plan_position(c, total_capital_usd=1000.0)
        # 0.045% daily on $1000 = $0.45
        assert plan.expected_daily_pnl_usd == pytest.approx(0.45, rel=1e-3)

    def test_zero_capital_raises(self):
        c = _candidate()
        with pytest.raises(ValueError):
            plan_position(c, total_capital_usd=0)

    def test_negative_capital_raises(self):
        c = _candidate()
        with pytest.raises(ValueError):
            plan_position(c, total_capital_usd=-100)


# ── Hedge deviation ──


class TestHedgeDeviation:
    def test_perfect_hedge_has_zero_delta(self):
        d = compute_hedge_deviation(
            spot_size_base=0.002, perp_size_base=0.002,
            current_spot_price=50000, position_notional_usd=200.0,
        )
        assert d.delta_base == 0
        assert d.delta_pct_of_position == 0

    def test_more_spot_than_perp_creates_long_delta(self):
        # We're 0.0001 BTC long after the perp moved differently
        d = compute_hedge_deviation(
            spot_size_base=0.0021, perp_size_base=0.002,
            current_spot_price=50000, position_notional_usd=200.0,
        )
        assert d.delta_base == pytest.approx(0.0001)
        # delta_usd = 5; position_notional = 200; pct = 2.5%
        assert d.delta_pct_of_position == pytest.approx(0.025)

    def test_more_perp_than_spot_creates_short_delta(self):
        d = compute_hedge_deviation(
            spot_size_base=0.002, perp_size_base=0.0025,
            current_spot_price=50000, position_notional_usd=200.0,
        )
        assert d.delta_base == pytest.approx(-0.0005)
        assert d.delta_pct_of_position == pytest.approx(-0.125)


# ── Close-condition ──


class TestShouldClose:
    def test_close_when_funding_drops_below_hold_threshold(self):
        c = _candidate(funding_8h=0.00005)    # 0.005% per 8h — too low
        assert should_close_position(c, min_funding_8h_to_hold=0.0001)

    def test_hold_when_funding_above_threshold(self):
        c = _candidate(funding_8h=0.0002)
        assert not should_close_position(c, min_funding_8h_to_hold=0.0001)

    def test_close_when_funding_negative(self):
        c = _candidate(funding_8h=-0.0001)
        assert should_close_position(c, min_funding_8h_to_hold=0.0001)
