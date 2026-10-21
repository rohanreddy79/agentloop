"""The loop: ends on text, feeds results back, recovers from tool errors, runs calls in parallel."""

from __future__ import annotations

import time

import pytest

from agentloop import (
    Agent,
    FakeProvider,
    Memory,
    Message,
    ScriptExhaustedError,
    StepEvent,
    ToolResult,
    Usage,
    text_response,
    tool_response,
)
from tests.conftest import add, boom, echo, nap


async def test_loop_ends_on_text_only_response() -> None:
    provider = FakeProvider(["hello"])
    agent = Agent(provider, [echo], system="be brief")
    result = await agent.arun("hi")
    assert result.output == "hello"
    assert result.steps == 1
    assert result.stop_reason == "end_turn"
    assert result.parsed is None
    assert [m.role for m in result.messages] == ["user", "assistant"]
    request = provider.calls[0]
    assert request.system == "be brief"
    assert [t.name for t in request.tools] == ["echo"]
    assert request.messages == (Message.user("hi"),)
    assert request.max_tokens == 4096


def test_run_is_a_blocking_wrapper() -> None:
    assert Agent(FakeProvider(["ok"])).run("x").output == "ok"


async def test_tool_calls_are_executed_and_results_fed_back() -> None:
    provider = FakeProvider([("add", {"a": 2, "b": 3}), "sum is 5"])
    result = await Agent(provider, [add]).arun("add")
    assert result.steps == 2
    assert result.output == "sum is 5"
    second = provider.calls[1].messages
    assert [m.role for m in second] == ["user", "assistant", "user"]
    call = second[1].tool_calls[0]
    assert call.name == "add"
    assert second[2].tool_results == (ToolResult(call.id, "5"),)


async def test_tool_error_is_returned_to_model_and_loop_continues() -> None:
    provider = FakeProvider(
        [("boom", {"message": "nope"}), ("echo", {"text": "recovered"}), "done"]
    )
    result = await Agent(provider, [boom, echo]).arun("go")
    error = provider.calls[1].messages[-1].tool_results[0]
    assert error.is_error
    assert error.content == "ValueError: nope"
    recovered = provider.calls[2].messages[-1].tool_results[0]
    assert recovered == ToolResult(recovered.tool_call_id, "recovered")
    assert result.steps == 3
    assert result.output == "done"


async def test_unknown_tool_call_does_not_crash_the_loop() -> None:
    provider = FakeProvider([("ghost", {}), "ok"])
    result = await Agent(provider, [echo]).arun("x")
    error = provider.calls[1].messages[-1].tool_results[0]
    assert error.is_error and "unknown tool 'ghost'" in error.content
    assert result.output == "ok"


async def test_parallel_tool_calls_run_concurrently() -> None:
    sleeps = [("nap", {"seconds": 0.2})] * 3
    provider = FakeProvider([sleeps, "done"])
    started = time.monotonic()
    result = await Agent(provider, [nap]).arun("sleep")
    elapsed = time.monotonic() - started
    assert elapsed < 0.45, f"3 x 0.2s tools took {elapsed:.2f}s; expected concurrent execution"
    calls = provider.calls[1].messages[1].tool_calls
    results = provider.calls[1].messages[2].tool_results
    assert [r.content for r in results] == ["slept 0.2"] * 3
    assert [r.tool_call_id for r in results] == [c.id for c in calls]
    assert result.output == "done"


async def test_on_step_hook_receives_each_step() -> None:
    events: list[StepEvent] = []
    provider = FakeProvider([("echo", {"text": "a"}), "fin"])
    result = await Agent(provider, [echo], on_step=events.append).arun("x")
    assert [e.step for e in events] == [1, 2]
    assert events[0].tool_results[0].content == "a"
    assert events[1].tool_results == ()
    assert events[1].usage == result.usage
    assert events[0].run_id == result.run_id


async def test_usage_is_accumulated_across_steps() -> None:
    provider = FakeProvider(
        [
            tool_response(("echo", {"text": "a"}), usage=Usage(100, 20)),
            text_response("done", usage=Usage(50, 10, cache_read_tokens=7)),
        ]
    )
    result = await Agent(provider, [echo]).arun("x")
    assert result.usage == Usage(150, 30, 7, 0)
    assert result.usage.total_tokens == 187


async def test_memory_persists_across_runs() -> None:
    provider = FakeProvider(["one", "two"])
    agent = Agent(provider)
    await agent.arun("a")
    await agent.arun("b")
    assert [m.role for m in provider.calls[1].messages] == ["user", "assistant", "user"]
    assert len(agent.memory) == 4


async def test_memory_window_limits_the_provider_view() -> None:
    provider = FakeProvider([("echo", {"text": "1"}), ("echo", {"text": "2"}), "end"])
    memory = Memory(window=2)
    result = await Agent(provider, [echo], memory=memory).arun("task")
    last = provider.calls[2].messages
    assert last[0] == Message.user("task")
    assert len(last) == 3  # task + one tool-call/result pair
    assert len(result.messages) == 6  # the full history is still returned


async def test_script_exhaustion_surfaces_as_provider_error() -> None:
    provider = FakeProvider([("echo", {"text": "a"})])
    with pytest.raises(ScriptExhaustedError):
        await Agent(provider, [echo]).arun("x")


async def test_tool_timeout_option_applies_to_registry() -> None:
    provider = FakeProvider([("nap", {"seconds": 1.0}), "done"])
    started = time.monotonic()
    await Agent(provider, [nap], tool_timeout=0.05).arun("x")
    assert time.monotonic() - started < 0.5
    assert "timed out" in provider.calls[1].messages[-1].tool_results[0].content


async def test_system_prompt_is_optional() -> None:
    provider = FakeProvider(["ok"])
    await Agent(provider).arun("x")
    assert provider.calls[0].system is None
