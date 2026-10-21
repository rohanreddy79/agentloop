"""Structured output via the hidden final_answer tool, including the validation-retry path."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agentloop import (
    FINAL_ANSWER,
    Agent,
    DuplicateToolError,
    FakeProvider,
    StructuredOutputError,
    tool,
)
from tests.conftest import echo


class Person(BaseModel):
    name: str
    age: int


async def test_success_path_registers_hidden_tool_and_parses() -> None:
    provider = FakeProvider([(FINAL_ANSWER, {"name": "Ada", "age": 36})])
    result = await Agent(provider, [echo], system="sys").arun("extract", output_model=Person)
    assert result.parsed == Person(name="Ada", age=36)
    assert result.output == '{"name":"Ada","age":36}'
    assert result.steps == 1
    request = provider.calls[0]
    assert [t.name for t in request.tools] == ["echo", FINAL_ANSWER]
    final = request.tools[1]
    assert set(final.parameters["properties"]) == {"name", "age"}
    assert final.parameters["required"] == ["name", "age"]
    assert "title" not in str(final.parameters)
    assert request.system is not None
    assert request.system.startswith("sys\n\n") and "final_answer" in request.system
    assert result.messages[-1].tool_results[0].content == "final answer accepted"


async def test_validation_error_is_returned_and_retried() -> None:
    provider = FakeProvider(
        [
            (FINAL_ANSWER, {"name": "Ada", "age": "unknown"}),
            (FINAL_ANSWER, {"name": "Ada", "age": 36}),
        ]
    )
    result = await Agent(provider, [echo]).arun("x", output_model=Person)
    error = provider.calls[1].messages[-1].tool_results[0]
    assert error.is_error
    assert "age: Input should be a valid integer" in error.content
    assert "call final_answer again" in error.content
    assert result.parsed is not None and result.parsed.age == 36
    assert result.steps == 2


async def test_plain_text_answer_is_nudged_toward_the_tool() -> None:
    provider = FakeProvider(["Ada is 36.", (FINAL_ANSWER, {"name": "Ada", "age": 36})])
    result = await Agent(provider, []).arun("x", output_model=Person)
    nudge = provider.calls[1].messages[-1]
    assert nudge.role == "user" and FINAL_ANSWER in nudge.text
    assert result.parsed == Person(name="Ada", age=36)


async def test_retry_limit_raises() -> None:
    provider = FakeProvider([(FINAL_ANSWER, {"age": "x"})] * 5)
    agent = Agent(provider, [], max_output_retries=2)
    with pytest.raises(StructuredOutputError, match="after 2 retries"):
        await agent.arun("x", output_model=Person)
    assert len(provider.calls) == 3


async def test_other_tools_in_the_same_turn_still_run() -> None:
    provider = FakeProvider([[("echo", {"text": "hi"}), (FINAL_ANSWER, {"name": "A", "age": 1})]])
    result = await Agent(provider, [echo]).arun("x", output_model=Person)
    results = result.messages[-1].tool_results
    assert [r.content for r in results] == ["hi", "final answer accepted"]
    assert result.parsed is not None and result.parsed.age == 1


def test_reserved_tool_name_is_rejected() -> None:
    @tool
    def final_answer(x: int) -> int:
        """Clash."""
        return x

    with pytest.raises(DuplicateToolError, match="reserved"):
        Agent(FakeProvider([]), [final_answer]).run("x", output_model=Person)
