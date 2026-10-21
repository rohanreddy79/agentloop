"""Provider backed by the official ``anthropic`` SDK (Messages API with tool use).

Wire mapping (SDK usage per the ``claude-api`` skill, 2026-10-21):

* assistant turns become ``text`` + ``tool_use`` content blocks (several ``tool_use`` blocks
  in one message is how the API expresses parallel tool calls);
* tool results become ``tool_result`` blocks in a single ``user`` message, with ``is_error``
  set when the tool failed;
* ``tool_use`` blocks in the response become :class:`ToolCall` objects; ``thinking`` blocks are
  ignored; ``usage`` (including cache counters) and ``stop_reason`` are passed through.

The SDK client is injectable so the mapping is unit-tested with a stand-in object and no key.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import anthropic

from agentloop.errors import ProviderError
from agentloop.providers.base import (
    Message,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    Usage,
)

# Default from the claude-api skill's model table (cached 2026-10-06). Override with the
# ``model`` argument or the AGENTLOOP_MODEL environment variable.
DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicProvider:
    """Completes conversations with ``client.messages.create``.

    ``effort`` (``low`` ... ``max``) is passed as ``output_config.effort`` when set; thinking is
    left at the model's default because current models manage it adaptively.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        effort: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model or os.environ.get("AGENTLOOP_MODEL") or DEFAULT_MODEL
        self.effort = effort
        if client is None:
            try:
                client = anthropic.AsyncAnthropic(
                    api_key=api_key, timeout=timeout, max_retries=max_retries
                )
            except anthropic.AnthropicError as exc:
                raise ProviderError(
                    "cannot create the Anthropic client; set ANTHROPIC_API_KEY or pass api_key"
                ) from exc
        self._client = client

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] = (),
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """One Messages API call."""
        request = build_request(self.model, messages, tools, system, max_tokens, self.effort)
        try:
            raw = await self._client.messages.create(**request)
        except anthropic.AnthropicError as exc:
            raise ProviderError(f"anthropic request failed: {exc}") from exc
        except TypeError as exc:  # SDK 2.x reports missing credentials this way, at call time
            raise ProviderError(
                f"anthropic request failed: {exc} (set ANTHROPIC_API_KEY or pass api_key)"
            ) from exc
        return parse_response(raw, default_model=self.model)


def build_request(
    model: str,
    messages: Sequence[Message],
    tools: Sequence[ToolDefinition],
    system: str | None,
    max_tokens: int,
    effort: str | None = None,
) -> dict[str, Any]:
    """Keyword arguments for ``messages.create``."""
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": to_wire_messages(messages),
    }
    if system:
        request["system"] = system
    if tools:
        request["tools"] = [to_wire_tool(t) for t in tools]
    if effort:
        request["output_config"] = {"effort": effort}
    return request


def to_wire_tool(tool: ToolDefinition) -> dict[str, Any]:
    """A Messages API tool definition."""
    return {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}


def to_wire_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Messages API ``messages`` array; empty turns are dropped."""
    wire = [_to_wire_message(m) for m in messages]
    return [m for m in wire if m is not None]


def _to_wire_message(message: Message) -> dict[str, Any] | None:
    if message.role == "assistant":
        blocks: list[dict[str, Any]] = []
        if message.text:
            blocks.append({"type": "text", "text": message.text})
        for call in message.tool_calls:
            blocks.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )
        return {"role": "assistant", "content": blocks} if blocks else None
    if not message.tool_results:
        return {"role": "user", "content": message.text} if message.text else None
    blocks = []
    for result in message.tool_results:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": result.tool_call_id,
            "content": result.content,
        }
        if result.is_error:
            block["is_error"] = True
        blocks.append(block)
    if message.text:
        blocks.append({"type": "text", "text": message.text})
    return {"role": "user", "content": blocks}


def parse_response(raw: Any, default_model: str = "") -> ModelResponse:
    """Map an SDK ``Message`` (or any object with the same attributes) to a ModelResponse."""
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in getattr(raw, "content", None) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            texts.append(block.text)
        elif kind == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))
    usage = getattr(raw, "usage", None)
    return ModelResponse(
        text="".join(texts),
        tool_calls=tuple(calls),
        usage=Usage(
            input_tokens=_count(usage, "input_tokens"),
            output_tokens=_count(usage, "output_tokens"),
            cache_read_tokens=_count(usage, "cache_read_input_tokens"),
            cache_write_tokens=_count(usage, "cache_creation_input_tokens"),
        ),
        stop_reason=getattr(raw, "stop_reason", None) or "unknown",
        model=getattr(raw, "model", None) or default_model,
    )


def _count(usage: Any, attr: str) -> int:
    return int(getattr(usage, attr, 0) or 0)
