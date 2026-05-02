"""Tests for the realistic dry-run simulator. The whole point of this
module is to NOT lie about fills — these tests pin the behavior."""
from __future__ import annotations

import pytest

from cryptarch.sim.realistic import (
    OrderBookLevel, OrderBookSnapshot,
    simulate_limit_maker, simulate_market_buy, simulate_market_sell,
)


def _book(asks: list[tuple[float, float]], bids: list[tuple[float, float]]) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        asks=tuple(OrderBookLevel(p, s) for p, s in asks),
        bids=tuple(OrderBookLevel(p, s) for p, s in bids),
    )


# ── market buy ──


class TestMarketBuy:
    def test_fills_at_best_ask_with_full_depth(self):
        book = _book(asks=[(50000, 1.0)], bids=[(49990, 1.0)])
        # $1000 at $50000 → 0.02 BTC. Plenty of depth.
        r = simulate_market_buy(book, size_usd=1000.0)
        assert r.filled
        assert r.avg_fill_price == pytest.approx(50000)
        assert r.filled_size_usd == 1000.0

    def test_walks_book_when_top_level_thin(self):
        # Top level only has 0.01 BTC. We need 0.02 → walk to next level.
        book = _book(
            asks=[(50000, 0.01), (50100, 1.0)],
            bids=[(49990, 1.0)],
        )
        r = simulate_market_buy(book, size_usd=1000.0)
        # 0.01 @ 50000 = $500; remaining $500 @ 50100 = 0.00998 BTC.
        # Total filled base = 0.01998, avg = 1000 / 0.01998 ≈ 50050
        assert r.filled
        assert r.avg_fill_price == pytest.approx(50050, rel=1e-3)

    def test_rejects_on_insufficient_depth(self):
        book = _book(asks=[(50000, 0.01)], bids=[(49990, 1.0)])
        # Only $500 of depth available; we want $1000.
        r = simulate_market_buy(book, size_usd=1000.0)
        assert not r.filled
        assert "insufficient_depth" in r.reason

    def test_rejects_on_excessive_slippage(self):
        book = _book(
            asks=[(50000, 0.001), (60000, 1.0)],    # huge gap
            bids=[(49990, 1.0)],
        )
        r = simulate_market_buy(book, size_usd=1000.0, max_slippage_pct=0.01)
        assert not r.filled
        assert "slippage_too_high" in r.reason

    def test_rejects_when_no_book(self):
        book = _book(asks=[], bids=[(49990, 1.0)])
        r = simulate_market_buy(book, size_usd=100)
        assert not r.filled
        assert r.reason == "no_book"


# ── market sell ──


class TestMarketSell:
    def test_fills_at_best_bid(self):
        book = _book(asks=[(50010, 1.0)], bids=[(50000, 1.0)])
        r = simulate_market_sell(book, size_usd=1000.0)
        assert r.filled
        assert r.avg_fill_price == pytest.approx(50000)


# ── limit maker — the key test for honest simulation ──


class TestLimitMaker:
    def test_fills_when_best_ask_at_or_below_limit(self):
        # Buy limit at 50000; best ask is 49995 → market is at our price
        book = _book(asks=[(49995, 1.0)], bids=[(49990, 1.0)])
        r = simulate_limit_maker(book, side="buy", limit_price=50000, size_usd=1000.0)
        assert r.filled
        assert r.avg_fill_price == 50000    # we filled AT our limit (we became the bid)

    def test_fills_when_within_tolerance(self):
        # Buy limit at 50000; best ask 50001 (one tick above). Default
        # tolerance is 5bps = $25 → 50000 + 25 = 50025. 50001 is well within.
        book = _book(asks=[(50001, 1.0)], bids=[(49990, 1.0)])
        r = simulate_limit_maker(book, side="buy", limit_price=50000, size_usd=1000.0)
        assert r.filled

    def test_rejects_when_book_too_far_above_limit(self):
        # Buy limit at 50000; best ask is 50500 (1% above)
        # → no realistic fill; refuse.
        book = _book(asks=[(50500, 1.0)], bids=[(49500, 1.0)])
        r = simulate_limit_maker(book, side="buy", limit_price=50000, size_usd=1000.0)
        assert not r.filled
        assert "maker_unfillable" in r.reason

    def test_polybot_pathological_case(self):
        # The exact pattern that produced fictional polybot PnL:
        # bid 0.001, ask 0.999, our limit at 0.93. Best ask is way above.
        book = _book(asks=[(0.999, 1000.0)], bids=[(0.001, 1000.0)])
        r = simulate_limit_maker(book, side="buy", limit_price=0.93, size_usd=100.0)
        assert not r.filled
        assert "maker_unfillable" in r.reason

    def test_sell_limit_fills_at_or_above_best_bid(self):
        book = _book(asks=[(50100, 1.0)], bids=[(50005, 1.0)])
        # Sell limit at 50000; best bid is 50005 → buyer is above our limit
        r = simulate_limit_maker(book, side="sell", limit_price=50000, size_usd=1000.0)
        assert r.filled

    def test_sell_limit_rejects_when_bid_too_far_below(self):
        book = _book(asks=[(50100, 1.0)], bids=[(49000, 1.0)])
        r = simulate_limit_maker(book, side="sell", limit_price=50000, size_usd=1000.0)
        assert not r.filled
        assert "maker_unfillable" in r.reason

    def test_custom_tolerance(self):
        # Buy limit 50000; ask is 50100 (20bps gap). Default 5bps tolerance
        # rejects. With 30bps tolerance, fill.
        book = _book(asks=[(50100, 1.0)], bids=[(49990, 1.0)])
        assert not simulate_limit_maker(book, "buy", 50000, 1000.0).filled
        r = simulate_limit_maker(book, "buy", 50000, 1000.0, fill_tolerance_pct=0.003)
        assert r.filled


# ── snapshot helpers ──


class TestSnapshot:
    def test_mid_and_spread(self):
        book = _book(asks=[(50010, 1.0)], bids=[(50000, 1.0)])
        assert book.mid == pytest.approx(50005)
        assert book.spread == pytest.approx(10)

    def test_no_book_returns_none(self):
        book = _book(asks=[], bids=[])
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid is None
        assert book.spread is None
