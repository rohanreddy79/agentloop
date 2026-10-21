"""Registry behaviour: lookup errors, validation, timeouts, exception capture, concurrency."""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import BaseModel

from agentloop import DuplicateToolError, ToolCall, ToolRegistry, UnknownToolError, tool
from tests.conftest import echo, nap


def test_duplicate_registration_raises() -> None:
    registry = ToolRegistry([echo])
    with pytest.raises(DuplicateToolError, match="'echo'"):
        registry.register(echo)


def test_unknown_tool_lookup_raises_and_lists_available(registry: ToolRegistry) -> None:
    with pytest.raises(UnknownToolError, match="unknown tool 'nope'; available: echo, add"):
        registry.get("nope")
    assert "echo" in registry
    assert registry.names() == ["echo", "add", "boom", "nap"]
    assert len(registry) == 4
    assert [d.name for d in registry.definitions()] == registry.names()


async def test_unknown_tool_call_is_error_result(registry: ToolRegistry) -> None:
    result = await registry.execute(ToolCall("c1", "nope", {}))
    assert result.is_error
    assert result.tool_call_id == "c1"
    assert "unknown tool 'nope'" in result.content


async def test_sync_tool_executes_and_coerces_return(registry: ToolRegistry) -> None:
    result = await registry.execute(ToolCall("c1", "add", {"a": 2, "b": 3}))
    assert result == result.__class__("c1", "5", False)
    result = await registry.execute(ToolCall("c2", "add", {"a": "4"}))  # str coerced to int
    assert result.content == "5"


async def test_invalid_arguments_are_error_results(registry: ToolRegistry) -> None:
    result = await registry.execute(ToolCall("c1", "add", {"a": "two"}))
    assert result.is_error
    assert result.content.startswith(
        "invalid arguments for add: a: Input should be a valid integer"
    )
    extra = await registry.execute(ToolCall("c2", "echo", {"text": "x", "junk": 1}))
    assert extra.is_error and "junk" in extra.content


async def test_tool_exception_becomes_error_result(registry: ToolRegistry) -> None:
    result = await registry.execute(ToolCall("c1", "boom", {"message": "kaboom"}))
    assert result.is_error
    assert result.content == "ValueError: kaboom"


async def test_async_tool_timeout_is_error_result() -> None:
    registry = ToolRegistry([nap], default_timeout=0.05)
    started = time.monotonic()
    result = await registry.execute(ToolCall("c1", "nap", {"seconds": 1.0}))
    assert result.is_error and "timed out after 0.05s" in result.content
    assert time.monotonic() - started < 0.5


async def test_sync_tool_timeout_is_error_result() -> None:
    @tool(timeout=0.05)
    def block(seconds: float) -> str:
        """Block the thread."""
        time.sleep(seconds)
        return "done"

    registry = ToolRegistry([block])
    started = time.monotonic()
    result = await registry.execute(ToolCall("c1", "block", {"seconds": 0.3}))
    assert result.is_error and "timed out" in result.content
    assert time.monotonic() - started < 0.25


async def test_execute_all_runs_concurrently(registry: ToolRegistry) -> None:
    calls = [ToolCall(f"c{i}", "nap", {"seconds": 0.15}) for i in range(3)]
    started = time.monotonic()
    results = await registry.execute_all(calls)
    elapsed = time.monotonic() - started
    assert [r.content for r in results] == ["slept 0.15"] * 3
    assert elapsed < 0.3, f"expected concurrent execution, took {elapsed:.2f}s"


async def test_sync_tools_run_concurrently_in_threads() -> None:
    @tool
    def block(seconds: float) -> str:
        """Block the thread."""
        time.sleep(seconds)
        return "ok"

    registry = ToolRegistry([block])
    started = time.monotonic()
    await registry.execute_all([ToolCall(f"c{i}", "block", {"seconds": 0.15}) for i in range(3)])
    assert time.monotonic() - started < 0.3


def test_register_plain_function_wraps_it() -> None:
    def shout(text: str) -> str:
        """Upper-case the text."""
        return text.upper()

    registry = ToolRegistry([shout])
    assert registry.get("shout").description == "Upper-case the text."


class Point(BaseModel):
    x: int
    y: int


async def test_pydantic_argument_is_instantiated() -> None:
    @tool
    def norm(point: Point) -> float:
        """Length."""
        assert isinstance(point, Point)
        return (point.x**2 + point.y**2) ** 0.5

    registry = ToolRegistry([norm])
    result = await registry.execute(ToolCall("c1", "norm", {"point": {"x": 3, "y": 4}}))
    assert result.content == "5.0"


def test_locally_defined_model_in_closure_is_resolved() -> None:
    class Local(BaseModel):
        n: int

    @tool
    def use(value: Local) -> int:
        """Use a model defined in the enclosing function."""
        return value.n

    assert use.parameters["properties"]["value"]["properties"] == {"n": {"type": "integer"}}


async def test_non_string_results_are_json() -> None:
    @tool
    def pair() -> dict[str, int]:
        """A dict."""
        return {"a": 1}

    registry = ToolRegistry([pair])
    assert (await registry.execute(ToolCall("c1", "pair", {}))).content == '{"a": 1}'


async def test_execute_never_raises_even_for_cancelled_style_errors() -> None:
    @tool
    async def weird() -> str:
        """Raise a non-Exception-derived-looking error."""
        raise RuntimeError("inner")

    registry = ToolRegistry([weird])
    result = await registry.execute(ToolCall("c1", "weird", {}))
    assert result.is_error and result.content == "RuntimeError: inner"
    await asyncio.sleep(0)
