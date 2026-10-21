"""USD price table used to turn token usage into cost.

Source: the model table in the ``claude-api`` skill (cached 2026-10-06), read on 2026-10-21.
Prices are USD per million tokens on the first-party Claude API. Cache reads are listed where
the source states them (Opus 5.5 / Sonnet 5.5 / Sonnet 5: $0.20; Fable 5.1: $0.25; Fable 5:
$1.00) and otherwise use the documented ~0.1x-of-input rate; cache writes use the documented
~1.25x-of-input rate for 5-minute entries. Haiku 5.5 is the <=100K-token-prompt rate.

Unknown models (including the offline fakes) are *not* priced: :func:`estimate_cost` returns
``None`` and the tracer records ``price_known: false`` rather than inventing a number.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentloop.providers.base import Usage

PRICE_TABLE_DATE = "2026-10-06"


@dataclass(frozen=True, slots=True)
class Price:
    """Per-million-token prices in USD."""

    input: float
    output: float
    cache_read: float
    cache_write: float

    def cost(self, usage: Usage) -> float:
        """Cost of ``usage`` in USD."""
        per_m = 1_000_000
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * self.cache_read
            + usage.cache_write_tokens * self.cache_write
        ) / per_m


PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(input=10.00, output=50.00, cache_read=0.25, cache_write=12.50),
    "claude-fable-5": Price(input=10.00, output=50.00, cache_read=1.00, cache_write=12.50),
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
    "claude-opus-5": Price(input=5.00, output=25.00, cache_read=0.50, cache_write=6.25),
    "claude-opus-4-8": Price(input=5.00, output=25.00, cache_read=0.50, cache_write=6.25),
    "claude-opus-4-7": Price(input=5.00, output=25.00, cache_read=0.50, cache_write=6.25),
    "claude-opus-4-6": Price(input=5.00, output=25.00, cache_read=0.50, cache_write=6.25),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
    "claude-sonnet-5": Price(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
    "claude-sonnet-4-6": Price(input=3.00, output=15.00, cache_read=0.30, cache_write=3.75),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
    "claude-haiku-4-5": Price(input=1.00, output=5.00, cache_read=0.10, cache_write=1.25),
}


def price_for(model: str) -> Price | None:
    """Look up a model's prices; ``None`` when the model is not in the table."""
    return PRICES.get(model)


def estimate_cost(model: str, usage: Usage) -> float | None:
    """USD cost of ``usage`` on ``model``, or ``None`` if the model is unpriced."""
    price = price_for(model)
    return None if price is None else price.cost(usage)
