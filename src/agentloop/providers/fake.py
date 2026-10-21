"""Offline providers.

:class:`FakeProvider` replays a script of responses and records every request it received,
which is what the loop tests use. :class:`RuleBasedFakeProvider` is a tiny deterministic
"model": it reads the task and the transcript so far and decides which tools to call next,
so the offline demo exercises the whole loop (parallel calls, a failing call, recovery, a
write, a final answer) with no network and no API key.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any

from agentloop.errors import ScriptExhaustedError
from agentloop.providers.base import (
    Message,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)

CallSpec = ToolCall | tuple[str, dict[str, Any]]
ScriptItem = ModelResponse | str | CallSpec | Sequence[CallSpec]


def text_response(
    text: str, *, usage: Usage | None = None, stop_reason: str = "end_turn"
) -> ModelResponse:
    """A text-only response (ends the loop)."""
    return ModelResponse(text=text, usage=usage or Usage(), stop_reason=stop_reason)


def tool_response(
    *calls: CallSpec, text: str = "", usage: Usage | None = None, prefix: str = "call"
) -> ModelResponse:
    """A response that calls one or more tools; ``("name", {args})`` tuples get ids assigned."""
    tool_calls = tuple(
        c
        if isinstance(c, ToolCall)
        else ToolCall(id=f"{prefix}_{i}", name=c[0], arguments=dict(c[1]))
        for i, c in enumerate(calls, start=1)
    )
    return ModelResponse(
        text=text, tool_calls=tool_calls, usage=usage or Usage(), stop_reason="tool_use"
    )


def estimate_usage(messages: Sequence[Message], response: ModelResponse) -> Usage:
    """Deterministic stand-in for token counts: roughly one token per four characters."""
    chars_in = sum(len(json.dumps(m.to_dict())) for m in messages)
    chars_out = len(response.text) + sum(
        len(c.name) + len(json.dumps(c.arguments)) for c in response.tool_calls
    )
    return Usage(input_tokens=max(1, chars_in // 4), output_tokens=max(1, chars_out // 4))


@dataclass(frozen=True, slots=True)
class ProviderCall:
    """One request a fake provider received (for assertions in tests)."""

    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    system: str | None
    max_tokens: int


class FakeProvider:
    """Replays ``script`` in order; raises :class:`ScriptExhaustedError` when it runs out."""

    name = "fake"

    def __init__(self, script: Iterable[ScriptItem], *, model: str = "fake-scripted") -> None:
        self.model = model
        self._script = [self._coerce(item, i) for i, item in enumerate(script, start=1)]
        self._index = 0
        self.calls: list[ProviderCall] = []

    @staticmethod
    def _coerce(item: ScriptItem, position: int) -> ModelResponse:
        if isinstance(item, ModelResponse):
            return item
        if isinstance(item, str):
            return text_response(item)
        single = isinstance(item, ToolCall) or (
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        )
        calls: Sequence[CallSpec] = [item] if single else item  # type: ignore[list-item]
        return tool_response(*calls, prefix=f"call{position}")

    @property
    def remaining(self) -> int:
        """Scripted responses not yet replayed."""
        return len(self._script) - self._index

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Record the request and return the next scripted response."""
        self.calls.append(ProviderCall(tuple(messages), tuple(tools), system, max_tokens))
        if self._index >= len(self._script):
            raise ScriptExhaustedError(
                f"fake provider script exhausted after {len(self._script)} responses"
            )
        scripted = self._script[self._index]
        self._index += 1
        usage = (
            scripted.usage if scripted.usage.total_tokens else estimate_usage(messages, scripted)
        )
        return replace(scripted, usage=usage, model=scripted.model or self.model)


# -- rule-based provider -----------------------------------------------------------------

_FILENAME = re.compile(r"(?<![\w/])[\w./-]*\w\.(?:txt|md|json|csv|log|yaml|yml|toml)\b")
_WRITE_TARGET = re.compile(
    r"\b(?:write|save|store)\b.*?\bto\s+(?:the\s+)?(?:file\s+)?([\w./-]+\.\w+)", re.I | re.S
)
_EXPR_CHARS = re.compile(r"[\d\s()+\-*/.%^]+")
_OPERATOR = re.compile(r"[-+*/%^]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class _Exchange:
    call: ToolCall
    result: ToolResult | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None and not self.result.is_error

    def arg(self, key: str) -> str:
        return str(self.call.arguments.get(key, ""))

    @property
    def stem(self) -> str:
        return PurePosixPath(self.arg("path")).stem


def find_expression(task: str) -> str | None:
    """The longest arithmetic-looking span in ``task`` (two numbers and an operator)."""
    best: str | None = None
    for span in _EXPR_CHARS.findall(task):
        candidate = span.strip().strip(".,")
        if (
            len(_NUMBER.findall(candidate)) >= 2
            and _OPERATOR.search(candidate)
            and candidate.count("(") == candidate.count(")")
            and (best is None or len(candidate) > len(best))
        ):
            best = candidate
    return best.replace("^", "**") if best else None


def find_files(task: str) -> list[str]:
    """File names mentioned in ``task``, in order, without duplicates."""
    seen: dict[str, None] = {}
    for name in _FILENAME.findall(task):
        seen.setdefault(name, None)
    return list(seen)


def find_write_target(task: str) -> str | None:
    """The file named after ``write ... to``, if any."""
    match = _WRITE_TARGET.search(task)
    return match.group(1) if match else None


def _transcript(messages: Sequence[Message]) -> tuple[str, list[_Exchange]]:
    task = ""
    exchanges: list[_Exchange] = []
    by_id: dict[str, _Exchange] = {}
    for message in messages:
        if message.role == "user" and not task and message.text and not message.tool_results:
            task = message.text
        for call in message.tool_calls:
            exchange = _Exchange(call)
            exchanges.append(exchange)
            by_id[call.id] = exchange
        for result in message.tool_results:
            if result.tool_call_id in by_id:
                by_id[result.tool_call_id].result = result
    return task, exchanges


def _called(exchanges: Sequence[_Exchange], name: str) -> bool:
    return any(ex.call.name == name for ex in exchanges)


def _closest(target: str, entries: Iterable[str]) -> str | None:
    stem = PurePosixPath(target).stem
    for entry in entries:
        entry = entry.strip()
        if entry and not entry.endswith("/") and PurePosixPath(entry).stem == stem:
            return entry
    return None


class RuleBasedFakeProvider:
    """Deterministic planner for small 'compute, read, write' tasks.

    Rules, in order: evaluate any arithmetic with ``calculator`` and read every file the task
    names (both in the same turn, so they run in parallel); if a read fails, ``list_dir`` and
    retry under the closest name; then ``write_file`` the gathered content to the ``write ...
    to`` target; finally answer in text. Unknown tasks get an honest "could not" answer.
    """

    name = "fake-rules"

    def __init__(self, *, model: str = "fake-rules") -> None:
        self.model = model

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Plan the next tool calls from the transcript, or answer."""
        task, exchanges = _transcript(messages)
        available = {t.name for t in tools}
        planned = self._plan(task, exchanges, available)
        if planned:
            start = len(exchanges) + 1
            calls = tuple(
                ToolCall(f"call_{start + i}", name, args) for i, (name, args) in enumerate(planned)
            )
            response = ModelResponse(tool_calls=calls, stop_reason="tool_use", model=self.model)
        else:
            response = ModelResponse(
                text=_answer(task, exchanges), stop_reason="end_turn", model=self.model
            )
        return replace(response, usage=estimate_usage(messages, response))

    def _plan(
        self, task: str, exchanges: list[_Exchange], available: set[str]
    ) -> list[tuple[str, dict[str, Any]]]:
        calls: list[tuple[str, dict[str, Any]]] = []
        expression = find_expression(task)
        write_target = find_write_target(task)
        reads = [f for f in find_files(task) if f != write_target]
        if expression and "calculator" in available and not _called(exchanges, "calculator"):
            calls.append(("calculator", {"expression": expression}))
        if "read_file" in available:
            for target in reads:
                calls.extend(_plan_read(target, exchanges, available))
        if calls:
            return calls
        if write_target and "write_file" in available and not _called(exchanges, "write_file"):
            return [("write_file", {"path": write_target, "content": _compose(exchanges)})]
        return []


def _plan_read(
    target: str, exchanges: list[_Exchange], available: set[str]
) -> list[tuple[str, dict[str, Any]]]:
    stem = PurePosixPath(target).stem
    attempts = [ex for ex in exchanges if ex.call.name == "read_file" and ex.stem == stem]
    if any(ex.ok for ex in attempts):
        return []
    if not attempts:
        return [("read_file", {"path": target})]
    listing = next((ex for ex in exchanges if ex.ok and ex.call.name == "list_dir"), None)
    if listing is None:
        if "list_dir" in available and not _called(exchanges, "list_dir"):
            return [("list_dir", {"path": "."})]
        return []
    assert listing.result is not None
    alternative = _closest(target, listing.result.content.splitlines())
    tried = {ex.arg("path") for ex in attempts}
    if alternative and alternative not in tried:
        return [("read_file", {"path": alternative})]
    return []


def _compose(exchanges: Sequence[_Exchange]) -> str:
    parts: list[str] = []
    for ex in exchanges:
        if not ex.ok:
            continue
        assert ex.result is not None
        if ex.call.name == "calculator":
            parts.append(f"{ex.arg('expression')} = {ex.result.content}")
        elif ex.call.name == "read_file":
            parts.append(f"--- {ex.arg('path')} ---\n{ex.result.content.rstrip()}")
    return "\n\n".join(parts) + "\n"


def _answer(task: str, exchanges: Sequence[_Exchange]) -> str:
    done: list[str] = []
    for ex in exchanges:
        if not ex.ok:
            continue
        assert ex.result is not None
        if ex.call.name == "calculator":
            done.append(f"{ex.arg('expression')} = {ex.result.content}")
        elif ex.call.name == "read_file":
            done.append(f"read {ex.arg('path')} ({len(ex.result.content)} chars)")
        elif ex.call.name == "write_file":
            done.append(f"wrote {ex.arg('path')}")
    for ex in exchanges:
        if ex.result is None or not ex.result.is_error:
            continue
        recovered = next(
            (o for o in exchanges if o.ok and o.call.name == ex.call.name and o.stem == ex.stem),
            None,
        )
        if recovered is not None:
            done.append(
                f"recovered from missing {ex.arg('path')} by listing the directory and "
                f"reading {recovered.arg('path')}"
            )
        else:
            done.append(f"{ex.call.name} failed: {ex.result.content.splitlines()[0]}")
    if not done:
        return "I could not make progress on this task with the available tools."
    return "Done: " + "; ".join(done) + "."
