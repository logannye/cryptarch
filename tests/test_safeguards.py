"""Tests for the safeguard invariants. Every code path that could let
through an unsafe order is tested explicitly."""
from __future__ import annotations

import pytest

from cryptarch.core.config import Settings
from cryptarch.core.safeguards import GuardViolation, OrderRequest, check_order


@pytest.fixture
def base_settings(monkeypatch) -> Settings:
    # Settings can pull from .env; ensure tests are deterministic.
    return Settings(
        _env_file=None,
        bankroll_usd=2000.0,
        alloc_layer_1_pct=0.60,
        alloc_layer_2_pct=0.25,
        alloc_layer_3_pct=0.15,
        max_total_deployed_pct=0.50,
        max_per_position_usd=500.0,
        enable_live_orders=False,
        layer_1_funding_arb_enabled=True,
        layer_2_cascade_capture_enabled=True,
        layer_3_tail_hedge_enabled=True,
    )


def _ok_order(**kw) -> OrderRequest:
    """Default valid order for tests; override fields via kw."""
    defaults = dict(
        exchange="binance",
        symbol="BTC/USDT",
        side="buy",
        size_usd=100.0,
        limit_price=50000.0,
        layer="l1_funding",
        client_order_id="abc123",
        is_live=False,
    )
    defaults.update(kw)
    return OrderRequest(**defaults)


# ── happy path ──


def test_valid_order_passes(base_settings):
    check_order(
        _ok_order(),
        base_settings,
        current_total_at_risk_usd=0.0,
        layer_already_deployed_usd=0.0,
        seen_client_order_ids=set(),
    )    # no exception


# ── invalid inputs ──


def test_zero_size_rejected(base_settings):
    with pytest.raises(GuardViolation, match="invalid_size"):
        check_order(
            _ok_order(size_usd=0),
            base_settings, 0, 0, set(),
        )


def test_negative_price_rejected(base_settings):
    with pytest.raises(GuardViolation, match="invalid_price"):
        check_order(
            _ok_order(limit_price=-1),
            base_settings, 0, 0, set(),
        )


# ── max-per-position ──


def test_position_size_at_cap_passes(base_settings):
    check_order(
        _ok_order(size_usd=500.0),
        base_settings, 0, 0, set(),
    )


def test_position_size_over_cap_rejected(base_settings):
    with pytest.raises(GuardViolation, match="max_per_position"):
        check_order(
            _ok_order(size_usd=500.01),
            base_settings, 0, 0, set(),
        )


# ── max-total-deployed ──


def test_total_deployed_at_cap_passes(base_settings):
    # Cap is 50% of $2000 = $1000. Already $900 deployed; new $100 = $1000.
    check_order(
        _ok_order(size_usd=100.0),
        base_settings,
        current_total_at_risk_usd=900.0,
        layer_already_deployed_usd=0.0,
        seen_client_order_ids=set(),
    )


def test_total_deployed_over_cap_rejected(base_settings):
    # $900 + $200 = $1100 > $1000 cap.
    with pytest.raises(GuardViolation, match="max_total_deployed"):
        check_order(
            _ok_order(size_usd=200.0),
            base_settings,
            current_total_at_risk_usd=900.0,
            layer_already_deployed_usd=0.0,
            seen_client_order_ids=set(),
        )


# ── layer caps ──


def test_layer_1_cap_at_60pct(base_settings):
    # L1 alloc = $1200. Already $1100 in L1; new $100 lands at exactly cap.
    check_order(
        _ok_order(layer="l1_funding", size_usd=100.0),
        base_settings, 0, 1100.0, set(),
    )


def test_layer_1_cap_exceeded(base_settings):
    with pytest.raises(GuardViolation, match="layer_cap_exceeded"):
        check_order(
            _ok_order(layer="l1_funding", size_usd=200.0),
            base_settings, 0, 1100.0, set(),
        )


def test_unknown_layer_rejected(base_settings):
    with pytest.raises(GuardViolation, match="unknown_layer"):
        check_order(
            _ok_order(layer="l5_quantum"),
            base_settings, 0, 0, set(),
        )


# ── idempotency ──


def test_missing_client_order_id_rejected(base_settings):
    with pytest.raises(GuardViolation, match="missing_client_order_id"):
        check_order(
            _ok_order(client_order_id=""),
            base_settings, 0, 0, set(),
        )


def test_duplicate_client_order_id_rejected(base_settings):
    with pytest.raises(GuardViolation, match="duplicate_client_order_id"):
        check_order(
            _ok_order(client_order_id="dup1"),
            base_settings, 0, 0,
            seen_client_order_ids={"dup1", "other"},
        )


# ── live-order gate ──


def test_live_order_blocked_when_disabled(base_settings):
    with pytest.raises(GuardViolation, match="live_orders_disabled"):
        check_order(
            _ok_order(is_live=True),
            base_settings, 0, 0, set(),
        )


def test_live_order_allowed_when_enabled(monkeypatch, base_settings):
    settings = Settings(
        _env_file=None,
        bankroll_usd=2000.0, alloc_layer_1_pct=0.60, alloc_layer_2_pct=0.25,
        alloc_layer_3_pct=0.15, max_total_deployed_pct=0.50,
        max_per_position_usd=500.0,
        enable_live_orders=True,
        layer_1_funding_arb_enabled=True,
    )
    check_order(
        _ok_order(is_live=True),
        settings, 0, 0, set(),
    )


# ── per-layer enable gate ──


def test_disabled_layer_rejected():
    settings = Settings(
        _env_file=None,
        bankroll_usd=2000.0, alloc_layer_1_pct=0.60, alloc_layer_2_pct=0.25,
        alloc_layer_3_pct=0.15, max_total_deployed_pct=0.50,
        max_per_position_usd=500.0,
        enable_live_orders=False,
        layer_1_funding_arb_enabled=False,    # disabled
    )
    with pytest.raises(GuardViolation, match="layer_disabled"):
        check_order(
            _ok_order(layer="l1_funding"),
            settings, 0, 0, set(),
        )


# ── interaction tests ──


def test_first_failure_is_reported(base_settings):
    """Multiple violations: the first-checked one is reported, by design."""
    # size=0 + over cap + invalid layer + duplicate id — all violate.
    # We get whichever check fires first. Currently invalid_size is first.
    with pytest.raises(GuardViolation, match="invalid_size"):
        check_order(
            _ok_order(
                size_usd=0,
                client_order_id="dup",
                layer="bogus",
            ),
            base_settings, 0, 0, {"dup"},
        )
