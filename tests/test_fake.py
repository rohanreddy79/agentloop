"""Fake providers: script coercion, request recording, and the rule-based planner."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentloop import Agent, Budget, FakeProvider, RuleBasedFakeProvider, text_response
from agentloop.builtin import calculator, make_fs_tools
from agentloop.providers.fake import find_expression, find_files, find_write_target

TASK = "Compute 17 * 23 + 5, read notes.txt, and write the result and the notes to result.txt."


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (TASK, "17 * 23 + 5"),
        ("What is (2 + 3) * 4?", "(2 + 3) * 4"),
        ("compute 2^10 please", "2**10"),
        ("read the 2 files", None),
        ("no numbers here", None),
    ],
)
def test_find_expression(task: str, expected: str | None) -> None:
    assert find_expression(task) == expected


def test_find_files_and_write_target() -> None:
    assert find_files(TASK) == ["notes.txt", "result.txt"]
    assert find_write_target(TASK) == "result.txt"
    assert find_write_target("Save a summary to the file out/summary.md") == "out/summary.md"
    assert find_write_target("Just read a.txt") is None
    assert find_files("Read data/x.json and y.csv twice: y.csv") == ["data/x.json", "y.csv"]


def _run_demo(root: Path) -> tuple[str, list[str], str]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "notes.md").write_text("some notes\n")
    agent = Agent(
        RuleBasedFakeProvider(), [calculator, *make_fs_tools(root)], budget=Budget(max_steps=8)
    )
    result = agent.run(TASK)
    names = [c.name for m in result.messages for c in m.tool_calls]
    return result.output, names, (root / "result.txt").read_text()


def test_rule_based_provider_solves_the_demo_deterministically(tmp_path: Path) -> None:
    first = _run_demo(tmp_path / "a")
    second = _run_demo(tmp_path / "b")
    assert first == second
    output, names, written = first
    assert names == ["calculator", "read_file", "list_dir", "read_file", "write_file"]
    assert written == "17 * 23 + 5 = 396\n\n--- notes.md ---\nsome notes\n"
    assert "17 * 23 + 5 = 396" in output
    assert "recovered from missing notes.txt" in output


def test_rule_based_provider_issues_parallel_calls_first(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("n\n")
    agent = Agent(RuleBasedFakeProvider(), [calculator, *make_fs_tools(tmp_path)])
    result = agent.run(TASK)
    first_turn = result.messages[1].tool_calls
    assert [c.name for c in first_turn] == ["calculator", "read_file"]
    assert result.steps == 3  # calc+read, write, answer
    assert result.output == "Done: 17 * 23 + 5 = 396; read notes.txt (2 chars); wrote result.txt."


def test_rule_based_provider_without_tools_answers_honestly() -> None:
    result = Agent(RuleBasedFakeProvider(), []).run("Compute 1 + 1")
    assert result.steps == 1
    assert result.output == "I could not make progress on this task with the available tools."


def test_rule_based_provider_gives_up_when_no_alternative_file(tmp_path: Path) -> None:
    agent = Agent(RuleBasedFakeProvider(), make_fs_tools(tmp_path))
    result = agent.run("Read missing.txt and summarise it.")
    names = [c.name for m in result.messages for c in m.tool_calls]
    assert names == ["read_file", "list_dir"]
    assert "read_file failed: FileNotFoundError: no such file: missing.txt" in result.output


def test_fake_provider_script_coercion() -> None:
    provider = FakeProvider(["a", ("t", {}), [("t", {}), ("u", {"k": 1})], text_response("z")])
    assert provider.remaining == 4
    multi = provider._script[2]
    assert [c.id for c in multi.tool_calls] == ["call3_1", "call3_2"]
    assert multi.tool_calls[1].arguments == {"k": 1}
    assert provider._script[1].tool_calls[0].id == "call2_1"
    assert provider._script[0].text == "a"


async def test_fake_provider_estimates_usage_and_records_requests() -> None:
    provider = FakeProvider(["answer"], model="fake-x")
    from agentloop import Message

    response = await provider.complete([Message.user("hello world")], system="s", max_tokens=9)
    assert response.model == "fake-x"
    assert response.usage.input_tokens > 0 and response.usage.output_tokens > 0
    assert provider.calls[0].system == "s" and provider.calls[0].max_tokens == 9
    assert provider.remaining == 0
