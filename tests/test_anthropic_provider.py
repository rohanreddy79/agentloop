"""AnthropicProvider request/response mapping against a stand-in client (no network, no key)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from agentloop import (
    DEFAULT_MODEL,
    Agent,
    AnthropicProvider,
    Message,
    ProviderError,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)
from agentloop.builtin import calculator
from agentloop.providers.anthropic import to_wire_messages


class _Messages:
    def __init__(self, responses: list[Any], error: Exception | None = None) -> None:
        self.responses = list(responses)
        self.error = error
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


class FakeClient:
    """Just enough of ``anthropic.AsyncAnthropic`` for the provider."""

    def __init__(self, *responses: Any, error: Exception | None = None) -> None:
        self.messages = _Messages(list(responses), error)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use(block_id: str, name: str, **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=arguments)


def _response(
    *blocks: SimpleNamespace,
    stop_reason: str = "tool_use",
    model: str = "claude-opus-5-5",
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    usage = usage or SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=2,
    )
    return SimpleNamespace(content=list(blocks), usage=usage, stop_reason=stop_reason, model=model)


CALC_SCHEMA = {
    "type": "object",
    "properties": {"expression": {"type": "string"}},
    "required": ["expression"],
    "additionalProperties": False,
}


async def test_request_mapping() -> None:
    client = FakeClient(_response(_text("ok"), stop_reason="end_turn"))
    provider = AnthropicProvider("claude-opus-5-5", client=client, effort="low")
    messages = [
        Message.user("hi"),
        Message.assistant(
            "let me check",
            [
                ToolCall("toolu_1", "calculator", {"expression": "1+1"}),
                ToolCall("toolu_2", "read_file", {"path": "a"}),
            ],
        ),
        Message.results(
            [ToolResult("toolu_1", "2"), ToolResult("toolu_2", "no such file", is_error=True)]
        ),
    ]
    tools = [ToolDefinition("calculator", "Calc.", CALC_SCHEMA)]
    await provider.complete(messages, tools, system="sys", max_tokens=321)
    request = client.messages.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["max_tokens"] == 321
    assert request["system"] == "sys"
    assert request["output_config"] == {"effort": "low"}
    assert request["tools"] == [
        {"name": "calculator", "description": "Calc.", "input_schema": CALC_SCHEMA}
    ]
    assert request["messages"] == [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "let me check"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "calculator",
                    "input": {"expression": "1+1"},
                },
                {"type": "tool_use", "id": "toolu_2", "name": "read_file", "input": {"path": "a"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "2"},
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": "no such file",
                    "is_error": True,
                },
            ],
        },
    ]


async def test_optional_request_fields_are_omitted() -> None:
    client = FakeClient(_response(_text("ok"), stop_reason="end_turn"))
    await AnthropicProvider(client=client).complete([Message.user("hi")])
    request = client.messages.requests[0]
    assert set(request) == {"model", "max_tokens", "messages"}


async def test_response_mapping_with_parallel_tool_use_and_thinking() -> None:
    blocks = [
        SimpleNamespace(type="thinking", thinking="..."),
        _text("Let me "),
        _tool_use("toolu_a", "calculator", expression="2*3"),
        _tool_use("toolu_b", "read_file", path="x"),
        _text("now."),
    ]
    provider = AnthropicProvider(client=FakeClient(_response(*blocks)))
    response = await provider.complete([Message.user("hi")])
    assert response.text == "Let me now."
    assert response.tool_calls == (
        ToolCall("toolu_a", "calculator", {"expression": "2*3"}),
        ToolCall("toolu_b", "read_file", {"path": "x"}),
    )
    assert response.usage == Usage(10, 5, 0, 2)
    assert response.stop_reason == "tool_use"
    assert response.model == "claude-opus-5-5"


async def test_refusal_and_empty_content_pass_through() -> None:
    provider = AnthropicProvider(client=FakeClient(_response(stop_reason="refusal")))
    response = await provider.complete([Message.user("hi")])
    assert response.text == "" and response.tool_calls == ()
    assert response.stop_reason == "refusal"


def test_empty_turns_are_dropped_from_the_wire() -> None:
    assert to_wire_messages([Message.assistant(""), Message.user("")]) == []
    assert to_wire_messages([Message.results([ToolResult("t1", "r")], text="also")]) == [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "r"},
                {"type": "text", "text": "also"},
            ],
        }
    ]


def test_model_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTLOOP_MODEL", raising=False)
    assert AnthropicProvider(client=FakeClient()).model == DEFAULT_MODEL == "claude-opus-5-5"
    monkeypatch.setenv("AGENTLOOP_MODEL", "claude-sonnet-5-5")
    assert AnthropicProvider(client=FakeClient()).model == "claude-sonnet-5-5"
    assert AnthropicProvider("claude-haiku-5-5", client=FakeClient()).model == "claude-haiku-5-5"


def test_real_sdk_client_is_constructed_when_a_key_is_given() -> None:
    provider = AnthropicProvider(api_key="test-key-not-used")
    assert isinstance(provider._client, anthropic.AsyncAnthropic)


async def test_sdk_errors_become_provider_errors() -> None:
    client = FakeClient(error=anthropic.AnthropicError("rate limited"))
    with pytest.raises(ProviderError, match="anthropic request failed: rate limited"):
        await AnthropicProvider(client=client).complete([Message.user("x")])


async def test_missing_credentials_become_a_provider_error() -> None:
    sdk_message = "Could not resolve authentication method. Expected one of api_key, ..."
    client = FakeClient(error=TypeError(sdk_message))
    with pytest.raises(ProviderError, match="set ANTHROPIC_API_KEY or pass api_key"):
        await AnthropicProvider(client=client).complete([Message.user("x")])


async def test_agent_end_to_end_with_mocked_client() -> None:
    client = FakeClient(
        _response(_text("I'll compute it."), _tool_use("toolu_1", "calculator", expression="6*7")),
        _response(_text("42!"), stop_reason="end_turn"),
    )
    agent = Agent(AnthropicProvider(client=client), [calculator])
    result = await agent.arun("what is 6*7?")
    assert result.output == "42!"
    assert result.steps == 2
    assert result.usage == Usage(20, 10, 0, 4)
    assert result.cost_usd == pytest.approx((20 * 4 + 10 * 20 + 4 * 5) / 1e6)
    second = client.messages.requests[1]["messages"]
    assert second[-1]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"}
    ]
    assert second[-2]["content"][1]["type"] == "tool_use"
