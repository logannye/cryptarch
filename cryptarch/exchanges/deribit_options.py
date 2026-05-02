"""Deribit option-chain helpers.

Deribit is the dominant venue for crypto options (~85% of BTC option
volume). CCXT supports it via the standard interface; this module wraps
those calls in option-chain-aware logic to make strike selection and
quote retrieval clean.

Instrument naming convention on Deribit:
    BTC-31MAY26-95000-C
    └─┬─┘ └──┬──┘ └─┬─┘ │
      │    │     │   └── "C" or "P"
      │    │     └────── strike in USD
      │    └──────────── expiry date "DDMONYY"
      └───────────────── underlying currency
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from cryptarch.exchanges.ccxt_client import CCXTClient

log = structlog.get_logger()


@dataclass(frozen=True)
class OptionInstrument:
    instrument_name: str        # e.g. "BTC-31MAY26-95000-C"
    underlying: str             # "BTC" | "ETH"
    expiry: datetime
    strike: float
    option_type: str            # "C" | "P"


@dataclass(frozen=True)
class OptionQuote:
    instrument: OptionInstrument
    bid_usd: float | None       # Note: Deribit quotes options in BTC; we convert to USD
    ask_usd: float | None
    mark_iv: float | None       # implied volatility (e.g. 0.65 = 65% annualized)
    fetched_at: datetime


def parse_instrument_name(name: str) -> OptionInstrument | None:
    """Parse 'BTC-31MAY26-95000-C' → OptionInstrument. Returns None on failure."""
    parts = name.split("-")
    if len(parts) != 4:
        return None
    underlying, expiry_str, strike_str, opt_type = parts
    if opt_type not in ("C", "P"):
        return None
    try:
        expiry = datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=timezone.utc)
        # Deribit options expire at 08:00 UTC.
        expiry = expiry.replace(hour=8, minute=0, second=0)
        strike = float(strike_str)
    except (ValueError, TypeError):
        return None
    return OptionInstrument(
        instrument_name=name,
        underlying=underlying,
        expiry=expiry,
        strike=strike,
        option_type=opt_type,
    )


class DeribitOptionsClient:
    """Thin convenience wrapper over CCXTClient(deribit, market_type='option')."""

    def __init__(self, base_client: CCXTClient):
        self._client = base_client

    async def list_instruments(self, currency: str = "BTC") -> list[OptionInstrument]:
        """All currently-listed options on Deribit for `currency`."""
        # Underlying CCXT client has loaded markets; filter to options.
        ccxt_inner = self._client._client    # type: ignore[attr-defined]
        if ccxt_inner is None or not ccxt_inner.markets:
            return []
        out: list[OptionInstrument] = []
        for market_id, market in ccxt_inner.markets.items():
            if market.get("type") != "option":
                continue
            symbol = market.get("symbol", "") or ""
            instrument_id = market.get("id", "") or symbol
            parsed = parse_instrument_name(instrument_id)
            if parsed is None:
                continue
            if parsed.underlying != currency:
                continue
            out.append(parsed)
        return out

    async def get_quote(
        self, instrument: OptionInstrument, spot_usd: float | None = None,
    ) -> OptionQuote:
        """Fetch quote for a single option. Deribit prices options in
        the underlying coin (e.g. 0.05 BTC), so caller passes spot to
        convert to USD.
        """
        ccxt_inner = self._client._client    # type: ignore[attr-defined]
        try:
            ticker = await ccxt_inner.fetch_ticker(instrument.instrument_name)
        except Exception as e:
            log.debug("deribit_quote_fetch_failed",
                      instrument=instrument.instrument_name, error=str(e)[:100])
            return OptionQuote(
                instrument=instrument, bid_usd=None, ask_usd=None,
                mark_iv=None, fetched_at=datetime.now(timezone.utc),
            )
        bid_coin = ticker.get("bid")
        ask_coin = ticker.get("ask")
        info = ticker.get("info") or {}
        mark_iv = info.get("mark_iv")
        try:
            mark_iv = float(mark_iv) / 100.0 if mark_iv is not None else None
        except (ValueError, TypeError):
            mark_iv = None

        # Deribit BTC options are priced in BTC. Convert to USD if we have spot.
        if spot_usd is None or spot_usd <= 0:
            bid_usd = float(bid_coin) if bid_coin is not None else None
            ask_usd = float(ask_coin) if ask_coin is not None else None
        else:
            bid_usd = float(bid_coin) * spot_usd if bid_coin is not None else None
            ask_usd = float(ask_coin) * spot_usd if ask_coin is not None else None
        return OptionQuote(
            instrument=instrument, bid_usd=bid_usd, ask_usd=ask_usd,
            mark_iv=mark_iv, fetched_at=datetime.now(timezone.utc),
        )

    async def filter_by_expiry(
        self, instruments: list[OptionInstrument], expiry: datetime,
    ) -> list[OptionInstrument]:
        """Filter to a specific expiry."""
        return [i for i in instruments
                if i.expiry.date() == expiry.date()]

    @staticmethod
    def find_closest_strike(
        instruments: list[OptionInstrument], target_strike: float, option_type: str,
    ) -> OptionInstrument | None:
        """Of the (already filtered) instruments, find the one with strike
        closest to target_strike on the requested side (C/P)."""
        candidates = [i for i in instruments if i.option_type == option_type]
        if not candidates:
            return None
        return min(candidates, key=lambda i: abs(i.strike - target_strike))
