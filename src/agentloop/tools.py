"""Typed tools: the ``@tool`` decorator, :class:`Tool`, and :class:`ToolRegistry`.

Execution is where the safety guarantees live. ``ToolRegistry.execute`` validates arguments
against the schema, enforces a timeout, and converts *every* failure (unknown tool, bad
arguments, timeout, exception) into a :class:`ToolResult` with ``is_error=True``. The loop
therefore never crashes because a tool did; the model sees the error text and can recover.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, overload

from pydantic import BaseModel, ValidationError

from agentloop.errors import DuplicateToolError, ToolSignatureError, UnknownToolError
from agentloop.providers.base import ToolCall, ToolDefinition, ToolResult
from agentloop.schema import (
    build_params_model,
    format_validation_error,
    json_schema_for,
    parse_docstring,
)


@dataclass(frozen=True)
class Tool:
    """A callable plus the metadata the model needs to call it.

    ``timeout`` (seconds) overrides the registry default for this tool. Calling the ``Tool``
    object directly runs the underlying function unchanged, so decorated functions stay usable
    as plain Python.
    """

    name: str
    description: str
    fn: Callable[..., Any]
    params_model: type[BaseModel]
    parameters: dict[str, Any]
    is_async: bool
    timeout: float | None = None

    @property
    def definition(self) -> ToolDefinition:
        """The provider-facing definition."""
        return ToolDefinition(self.name, self.description, self.parameters)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Coerce ``arguments`` through the schema; raises ``pydantic.ValidationError``."""
        parsed = self.params_model.model_validate(arguments)
        return {field: getattr(parsed, field) for field in self.params_model.model_fields}

    async def execute(self, arguments: dict[str, Any], timeout: float | None = None) -> str:
        """Validate, run (sync tools in a worker thread) and stringify the result."""
        kwargs = self.validate(arguments)
        coro = self.fn(**kwargs) if self.is_async else asyncio.to_thread(self.fn, **kwargs)
        result = await asyncio.wait_for(coro, timeout=timeout or self.timeout)
        return _stringify(result)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    try:
        return json.dumps(value, default=str)
    except TypeError:
        return str(value)


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...
@overload
def tool(
    *, name: str | None = ..., description: str | None = ..., timeout: float | None = ...
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a function into a :class:`Tool`.

    The name defaults to the function name, the description to the docstring summary, and the
    parameter schema is derived from the type hints (``Args:`` lines become field descriptions).
    Works bare (``@tool``) or with options (``@tool(name="calc", timeout=5)``).
    """

    def wrap(func: Callable[..., Any], scope: dict[str, Any] | None) -> Tool:
        tool_name = name or func.__name__
        summary, _ = parse_docstring(func.__doc__)
        desc = description or summary
        if not desc:
            raise ToolSignatureError(f"tool {tool_name!r} needs a description or a docstring")
        model = build_params_model(func, tool_name, scope)
        return Tool(
            name=tool_name,
            description=desc,
            fn=func,
            params_model=model,
            parameters=json_schema_for(model),
            is_async=inspect.iscoroutinefunction(func),
            timeout=timeout,
        )

    if fn is not None:
        return wrap(fn, _defining_scope())

    def decorator(func: Callable[..., Any]) -> Tool:
        return wrap(func, _defining_scope())

    return decorator


def _defining_scope() -> dict[str, Any]:
    """Locals of the frame that applied the decorator (two frames up from here)."""
    frame = sys._getframe(2)
    try:
        return dict(frame.f_locals)
    finally:
        del frame


class ToolRegistry:
    """Named collection of tools with safe, concurrent execution."""

    def __init__(
        self, tools: Iterable[Tool | Callable[..., Any]] = (), *, default_timeout: float = 30.0
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self.default_timeout = default_timeout
        for item in tools:
            self.register(item)

    def register(self, item: Tool | Callable[..., Any]) -> Tool:
        """Add a tool (plain functions are wrapped with :func:`tool`)."""
        built = item if isinstance(item, Tool) else tool(item)
        if built.name in self._tools:
            raise DuplicateToolError(f"tool {built.name!r} is already registered")
        self._tools[built.name] = built
        return built

    def get(self, name: str) -> Tool:
        """Look up a tool; raises :class:`UnknownToolError`."""
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(
                f"unknown tool {name!r}; available: {', '.join(self.names()) or 'none'}"
            ) from None

    def names(self) -> list[str]:
        """Registered tool names in registration order."""
        return list(self._tools)

    def definitions(self) -> list[ToolDefinition]:
        """Provider-facing definitions for every tool."""
        return [t.definition for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    async def execute(self, call: ToolCall) -> ToolResult:
        """Run one call; never raises, failures come back as ``is_error`` results."""
        try:
            selected = self.get(call.name)
        except UnknownToolError as exc:
            return ToolResult(call.id, str(exc), is_error=True)
        timeout = selected.timeout or self.default_timeout
        try:
            content = await selected.execute(call.arguments, timeout=timeout)
        except ValidationError as exc:
            detail = format_validation_error(exc)
            return ToolResult(
                call.id, f"invalid arguments for {call.name}: {detail}", is_error=True
            )
        except TimeoutError:
            return ToolResult(call.id, f"{call.name} timed out after {timeout}s", is_error=True)
        except Exception as exc:
            return ToolResult(call.id, f"{type(exc).__name__}: {exc}", is_error=True)
        return ToolResult(call.id, content)

    async def execute_all(self, calls: Iterable[ToolCall]) -> list[ToolResult]:
        """Run several calls concurrently, preserving order."""
        return list(await asyncio.gather(*(self.execute(c) for c in calls)))
