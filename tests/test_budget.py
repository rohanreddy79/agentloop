"""Budgets stop the loop with a partial result attached to the exception."""

from __future__ import annotations

import pytest

from agentloop import (
    Agent,
    Budget,
    BudgetExceeded,
    FakeProvider,
    Usage,
    text_response,
    tool_response,
)
from tests.conftest import echo, nap


def _echo_forever(n: int = 10, **kwargs: object) -> FakeProvider:
    return FakeProvider([tool_response(("echo", {"text": "t"}), text="working", **kwargs)] * n)


async def test_max_steps_raises_with_partial_result() -> None:
    provider = _echo_forever()
    agent = Agent(provider, [echo], budget=Budget(max_steps=3))
    with pytest.raises(
        BudgetExceeded, match=r"budget exceeded \(max_steps\): 3 steps used"
    ) as info:
        await agent.arun("x")
    partial = info.value.result
    assert info.value.limit == "max_steps"
    assert partial.steps == 3
    assert partial.output == "working"
    assert partial.stop_reason == "budget_exceeded"
    assert len(partial.messages) == 1 + 3 * 2
    assert len(provider.calls) == 3


async def test_max_tokens_counts_all_tokens_across_calls() -> None:
    provider = _echo_forever(usage=Usage(400, 100))
    agent = Agent(provider, [echo], budget=Budget(max_tokens=1000))
    with pytest.raises(BudgetExceeded) as info:
        await agent.arun("x")
    assert info.value.limit == "max_tokens"
    assert info.value.result.steps == 3  # 500 + 500 = 1000 is allowed, 1500 is not
    assert info.value.result.usage.total_tokens == 1500


async def test_budget_does_not_abort_a_run_that_just_finished() -> None:
    provider = FakeProvider(
        [
            tool_response(("echo", {"text": "t"}), usage=Usage(400, 100)),
            text_response("done", usage=Usage(400, 100)),
        ]
    )
    result = await Agent(provider, [echo], budget=Budget(max_tokens=600)).arun("x")
    assert result.output == "done"
    assert result.usage.total_tokens == 1000


async def test_max_seconds_is_checked_before_each_call() -> None:
    provider = FakeProvider([("nap", {"seconds": 0.15})] * 5)
    agent = Agent(provider, [nap], budget=Budget(max_seconds=0.1))
    with pytest.raises(BudgetExceeded) as info:
        await agent.arun("x")
    assert info.value.limit == "max_seconds"
    assert info.value.result.steps == 1
    assert info.value.result.elapsed_s >= 0.1


async def test_max_cost_uses_the_price_table() -> None:
    provider = _echo_forever(usage=Usage(input_tokens=100_000))
    provider.model = "claude-opus-5-5"  # $4 / MTok input -> $0.40 per step
    agent = Agent(provider, [echo], budget=Budget(max_cost_usd=0.5))
    with pytest.raises(BudgetExceeded, match=r"\$0.8000 spent") as info:
        await agent.arun("x")
    assert info.value.limit == "max_cost_usd"
    assert info.value.result.cost_usd == pytest.approx(0.8)


async def test_unpriced_model_never_trips_the_cost_budget() -> None:
    provider = FakeProvider([("echo", {"text": "a"}), ("echo", {"text": "b"}), "done"])
    result = await Agent(provider, [echo], budget=Budget(max_cost_usd=0.0001)).arun("x")
    assert result.output == "done"
    assert result.cost_usd == 0.0


def test_budget_is_serialisable() -> None:
    assert Budget(3, 10, 0.5, 2.0).to_dict() == {
        "max_steps": 3,
        "max_tokens": 10,
        "max_cost_usd": 0.5,
        "max_seconds": 2.0,
    }
