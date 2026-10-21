"""Traces are written as parseable JSONL with usage, cost and errors; the renderer reads them."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from agentloop import (
    Agent,
    Budget,
    BudgetExceeded,
    FakeProvider,
    ScriptExhaustedError,
    Tracer,
    Usage,
    read_trace,
    render_trace,
)
from agentloop.pricing import PRICES, estimate_cost, price_for
from tests.conftest import boom, echo


async def test_trace_file_is_written_and_parseable(tmp_path: Path) -> None:
    trace = tmp_path / "run.jsonl"
    provider = FakeProvider(
        [("echo", {"text": "a"}), ("boom", {}), "done"], model="claude-opus-5-5"
    )
    agent = Agent(provider, [echo, boom], tracer=Tracer(trace))
    result = await agent.arun("task text")
    assert result.trace_path == trace

    events = read_trace(trace)
    kinds = [e["event"] for e in events]
    assert kinds == ["run_start", "tool_call", "step", "tool_call", "step", "step", "run_end"]
    assert len({e["run_id"] for e in events}) == 1
    assert all("ts" in e for e in events)

    start = events[0]
    assert start["task"] == "task text"
    assert start["provider"] == "fake" and start["model"] == "claude-opus-5-5"
    assert start["budget"]["max_steps"] == 10

    steps = [e for e in events if e["event"] == "step"]
    assert steps[0]["price_known"] is True and steps[0]["cost_usd"] > 0
    assert steps[0]["tool_calls"][0]["name"] == "echo"
    assert steps[0]["messages_delta"][0]["role"] == "assistant"
    assert steps[0]["messages_delta"][1]["tool_results"][0]["content"] == "a"
    assert steps[0]["usage"]["total_tokens"] > 0
    assert steps[2]["text"] == "done" and steps[2]["stop_reason"] == "end_turn"

    tools = [e for e in events if e["event"] == "tool_call"]
    assert tools[0]["is_error"] is False and tools[0]["result"] == "a"
    assert tools[1]["is_error"] is True and tools[1]["result"] == "ValueError: boom"
    assert all(e["latency_ms"] >= 0 for e in tools)

    end = events[-1]
    assert end["status"] == "ok" and end["steps"] == 3 and end["output"] == "done"
    assert end["usage"]["total_tokens"] == result.usage.total_tokens
    assert end["cost_usd"] == pytest.approx(result.cost_usd)
    assert end["cost_usd"] == pytest.approx(sum(s["cost_usd"] for s in steps))


async def test_unpriced_model_is_flagged_not_zeroed_silently(tmp_path: Path) -> None:
    trace = tmp_path / "run.jsonl"
    await Agent(FakeProvider(["ok"]), tracer=Tracer(trace)).arun("x")
    step = next(e for e in read_trace(trace) if e["event"] == "step")
    assert step["price_known"] is False and step["cost_usd"] == 0.0


async def test_budget_exceeded_and_errors_are_recorded(tmp_path: Path) -> None:
    trace = tmp_path / "run.jsonl"
    provider = FakeProvider([("echo", {"text": "a"})] * 3)
    agent = Agent(provider, [echo], budget=Budget(max_steps=1), tracer=Tracer(trace))
    with pytest.raises(BudgetExceeded):
        await agent.arun("x")
    assert read_trace(trace)[-1]["status"] == "budget_exceeded"

    trace2 = tmp_path / "run2.jsonl"
    agent = Agent(FakeProvider([("echo", {"text": "a"})]), [echo], tracer=Tracer(trace2))
    with pytest.raises(ScriptExhaustedError):
        await agent.arun("x")
    assert read_trace(trace2)[-1]["status"] == "error"


async def test_render_trace_shows_steps_tools_and_errors(tmp_path: Path) -> None:
    trace = tmp_path / "run.jsonl"
    provider = FakeProvider([("boom", {}), ("echo", {"text": "fine"}), "all good"])
    await Agent(provider, [echo, boom], tracer=Tracer(trace)).arun("render me")
    console = Console(record=True, width=120, force_terminal=False, no_color=True)
    render_trace(read_trace(trace), console)
    text = console.export_text()
    assert "render me" in text
    assert "status: ok" in text
    assert "boom error: ValueError: boom" in text
    assert "echo ok: fine" in text
    assert "all good" in text
    # the model row for a step is printed before that step's tool rows
    assert text.index("-> boom") < text.index("boom error")


def test_price_table_lookup_and_cost() -> None:
    assert price_for("not-a-model") is None
    assert estimate_cost("not-a-model", Usage(1, 1)) is None
    million = Usage(1_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert estimate_cost("claude-opus-5-5", million) == pytest.approx(4 + 20 + 0.2 + 5)
    assert estimate_cost("claude-haiku-5-5", Usage(output_tokens=2_000_000)) == pytest.approx(1.0)
    assert {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"} <= set(PRICES)
