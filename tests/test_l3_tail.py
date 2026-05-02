"""Tests for Layer 3 (tail-hedge) pure-function math."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cryptarch.strategies.l3_tail import (
    OptionLeg, StranglePosition,
    compute_strangle_breakeven, daily_theta_cost,
    days_to_expiry,
    max_contracts_within_theta_budget, select_strangle_strikes,
    select_target_expiry, should_roll,
)


# ── Strike selection ──


class TestSelectStrikes:
    def test_basic_btc_at_50000(self):
        call, put = select_strangle_strikes(spot=50000.0, otm_pct=0.20, strike_step=1000.0)
        # Call strike: ceil(60000 / 1000) × 1000 = 60000
        # Put strike: floor(40000 / 1000) × 1000 = 40000
        assert call == 60000.0
        assert put == 40000.0

    def test_strikes_round_to_step(self):
        # Spot at 50_125, otm 20%; raw call 60150 → 61000 (ceil to 1000)
        # Raw put 40100 → 40000 (floor)
        call, put = select_strangle_strikes(spot=50125.0, otm_pct=0.20, strike_step=1000.0)
        assert call == 61000.0
        assert put == 40000.0

    def test_smaller_step_finer_granularity(self):
        call, put = select_strangle_strikes(spot=2000.0, otm_pct=0.20, strike_step=50.0)
        # Raw call 2400 → 2400 (already on step); raw put 1600 → 1600
        assert call == 2400.0
        assert put == 1600.0

    def test_zero_spot_raises(self):
        with pytest.raises(ValueError):
            select_strangle_strikes(spot=0, otm_pct=0.20)

    def test_invalid_otm_raises(self):
        with pytest.raises(ValueError):
            select_strangle_strikes(spot=50000, otm_pct=1.5)
        with pytest.raises(ValueError):
            select_strangle_strikes(spot=50000, otm_pct=0)


# ── DTE helpers ──


class TestDaysToExpiry:
    def test_future_expiry_positive(self):
        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        expiry = datetime(2026, 5, 15, tzinfo=timezone.utc)
        assert days_to_expiry(expiry, now) == pytest.approx(14.0)

    def test_past_expiry_negative(self):
        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        expiry = datetime(2026, 4, 25, tzinfo=timezone.utc)
        assert days_to_expiry(expiry, now) == pytest.approx(-6.0)


class TestShouldRoll:
    def test_roll_when_dte_below_threshold(self):
        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        expiry = datetime(2026, 5, 25, tzinfo=timezone.utc)    # 24 days
        assert should_roll(expiry, target_dte_min=30, now=now)

    def test_no_roll_when_dte_above_threshold(self):
        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        expiry = datetime(2026, 6, 15, tzinfo=timezone.utc)    # 45 days
        assert not should_roll(expiry, target_dte_min=30, now=now)


class TestSelectTargetExpiry:
    def _expiries(self, now: datetime, days: list[int]) -> list[datetime]:
        return [now + timedelta(days=d) for d in days]

    def test_picks_closest_to_target_dte(self):
        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        expiries = self._expiries(now, [7, 30, 45, 60, 90])
        # Target is 45; closest is exactly 45
        assert select_target_expiry(expiries, target_dte=45, now=now) == expiries[2]

    def test_ignores_past_expiries(self):
        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        expiries = self._expiries(now, [-10, 7, 45, 90])
        # Past one is excluded; from [7, 45, 90], closest to 45 is 45
        assert select_target_expiry(expiries, target_dte=45, now=now) == expiries[2]

    def test_empty_input_returns_none(self):
        assert select_target_expiry([], target_dte=45) is None

    def test_no_future_expiries_returns_none(self):
        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        past = [now - timedelta(days=5), now - timedelta(days=10)]
        assert select_target_expiry(past, target_dte=45, now=now) is None


# ── Strangle valuation ──


def _strangle(
    call_strike: float = 60000.0,
    put_strike: float = 40000.0,
    contracts: float = 1.0,
    call_premium: float = 1500.0,
    put_premium: float = 1200.0,
    expiry_days: int = 45,
) -> StranglePosition:
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(days=expiry_days)
    return StranglePosition(
        call=OptionLeg(
            instrument_name="BTC-test-CALL", underlying="BTC", expiry=expiry,
            strike=call_strike, option_type="C", contracts=contracts,
            entry_premium_usd=call_premium, entry_iv=0.65,
        ),
        put=OptionLeg(
            instrument_name="BTC-test-PUT", underlying="BTC", expiry=expiry,
            strike=put_strike, option_type="P", contracts=contracts,
            entry_premium_usd=put_premium, entry_iv=0.65,
        ),
        spot_at_open=50000.0,
        opened_at=now,
    )


class TestStranglePosition:
    def test_total_premium(self):
        pos = _strangle(call_premium=1500, put_premium=1200, contracts=2)
        # (1500 + 1200) × 2 = 5400
        assert pos.total_premium_usd == 5400.0

    def test_expiry_reflects_legs(self):
        pos = _strangle(expiry_days=60)
        # Call's expiry is canonical
        assert pos.expiry == pos.call.expiry


class TestBreakeven:
    def test_basic_breakeven(self):
        pos = _strangle(
            call_strike=60000, put_strike=40000,
            contracts=1, call_premium=1500, put_premium=1200,
        )
        lower, upper = compute_strangle_breakeven(pos)
        # Total premium = 2700 over 1 contract.
        # Upper: call_strike + total_premium = 60000 + 2700 = 62700
        # Lower: put_strike - total_premium = 40000 - 2700 = 37300
        assert upper == pytest.approx(62700.0)
        assert lower == pytest.approx(37300.0)


# ── Theta + sizing ──


class TestTheta:
    def test_daily_theta_basic(self):
        # Premiums total 2700 per side × 1 contract; DTE 45
        # daily theta = 2700 / 45 = 60
        theta = daily_theta_cost(call_premium_usd=1500, put_premium_usd=1200,
                                 contracts=1, days_to_expiry_now=45)
        assert theta == pytest.approx(60.0)

    def test_theta_zero_when_expired(self):
        assert daily_theta_cost(1500, 1200, 1, 0) == 0
        assert daily_theta_cost(1500, 1200, 1, -5) == 0

    def test_max_contracts_for_budget(self):
        # Budget $5/day, premiums (1500+1200) = 2700, DTE 45
        # contracts = 5 × 45 / 2700 = 0.0833 contracts
        contracts = max_contracts_within_theta_budget(
            daily_theta_budget_usd=5.0,
            call_premium_usd=1500, put_premium_usd=1200,
            days_to_expiry_now=45,
        )
        assert contracts == pytest.approx(0.0833, rel=0.01)

    def test_max_contracts_zero_when_no_budget(self):
        c = max_contracts_within_theta_budget(0, 1500, 1200, 45)
        assert c == 0

    def test_max_contracts_zero_when_zero_dte(self):
        c = max_contracts_within_theta_budget(5, 1500, 1200, 0)
        assert c == 0
