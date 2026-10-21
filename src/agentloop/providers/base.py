"""Provider-agnostic data types and the :class:`Provider` protocol.

A provider turns a conversation (``messages``) plus tool definitions into a
:class:`ModelResponse`. The agent loop only ever sees these types; the Anthropic wire format
lives entirely in :mod:`agentloop.providers.anthropic`, which keeps the loop testable with
the fake providers and makes adding another backend a one-file job.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request from the model to run ``name`` with ``arguments``."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What the tool returned, keyed back to the call that produced it.

    ``is_error`` is how failures reach the model: the content carries the error text and the
    model decides what to do next (retry, use another tool, give up).
    """

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool as advertised to the model: name, description and a JSON-schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of the conversation.

    User turns carry ``text`` and/or ``tool_results``; assistant turns carry ``text`` and/or
    ``tool_calls``. A turn can hold several tool calls (parallel tool use) and the matching
    user turn holds all of their results.
    """

    role: Role
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()

    @classmethod
    def user(cls, text: str) -> Message:
        """A plain user message."""
        return cls(role="user", text=text)

    @classmethod
    def assistant(cls, text: str = "", tool_calls: Iterable[ToolCall] = ()) -> Message:
        """An assistant message with optional tool calls."""
        return cls(role="assistant", text=text, tool_calls=tuple(tool_calls))

    @classmethod
    def results(cls, results: Iterable[ToolResult], text: str = "") -> Message:
        """A user message carrying tool results (and optional trailing text)."""
        return cls(role="user", text=text, tool_results=tuple(results))

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form used by the tracer; empty fields are omitted."""
        out: dict[str, Any] = {"role": self.role}
        if self.text:
            out["text"] = self.text
        if self.tool_calls:
            out["tool_calls"] = [asdict(c) for c in self.tool_calls]
        if self.tool_results:
            out["tool_results"] = [asdict(r) for r in self.tool_results]
        return out


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for one provider call (or, summed, for a whole run)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Every token billed on this call, cached or not."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        """Plain dict including the derived total."""
        return {**asdict(self), "total_tokens": self.total_tokens}


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """What the model said: text, zero or more tool calls, usage and why it stopped.

    ``stop_reason`` mirrors the Messages API (``end_turn``, ``tool_use``, ``max_tokens``,
    ``refusal``, ...). The loop keys on ``tool_calls`` being empty, not on the stop reason,
    so a provider that reports stop reasons differently still works.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = Usage()
    stop_reason: str = "end_turn"
    model: str = ""

    def to_message(self) -> Message:
        """The assistant turn to append to the conversation."""
        return Message.assistant(self.text, self.tool_calls)


@runtime_checkable
class Provider(Protocol):
    """Anything that can complete a conversation.

    ``name`` identifies the backend in traces; ``model`` is used for cost lookup.
    """

    name: str
    model: str

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Return the model's next turn given the conversation so far."""
        ...
