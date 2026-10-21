"""Memory windowing never splits a tool-call/result pair and summarises what it drops."""

from __future__ import annotations

import pytest

from agentloop import Memory, Message, ToolCall, ToolResult


def _history() -> list[Message]:
    msgs = [Message.user("task")]
    for i in range(3):
        call = ToolCall(f"c{i}", "echo", {"text": str(i)})
        msgs.append(Message.assistant("", [call]))
        msgs.append(Message.results([ToolResult(call.id, str(i))]))
    msgs.append(Message.assistant("final"))
    return msgs


def test_without_window_everything_is_returned() -> None:
    memory = Memory()
    memory.extend(_history())
    assert memory.messages() == _history()
    assert memory.history() == _history()
    assert len(memory) == 8


def test_window_keeps_task_and_tail() -> None:
    memory = Memory(window=3)
    memory.extend(_history())
    view = memory.messages()
    assert view[0] == Message.user("task")
    assert view[1:] == _history()[-3:]


def test_window_never_starts_on_tool_results() -> None:
    memory = Memory(window=2)
    memory.extend(_history()[:-1])  # ends with a results message
    view = memory.messages()
    # tail of 2 would be [results(c2)]... no: [assistant(c2), results(c2)] is fine; use 1
    memory_one = Memory(window=1)
    memory_one.extend(_history()[:-1])
    view_one = memory_one.messages()
    assert view_one[0] == Message.user("task")
    assert view_one[1].tool_calls and view_one[1].tool_calls[0].id == "c2"
    assert view_one[2].tool_results[0].tool_call_id == "c2"
    assert len(view) == 3


def test_summarize_hook_is_called_once_per_truncation_size() -> None:
    calls: list[int] = []

    def summarize(dropped: list[Message]) -> str:  # type: ignore[type-arg]
        calls.append(len(dropped))
        return f"{len(dropped)} messages dropped"

    memory = Memory(window=2, summarize=summarize)
    memory.extend(_history())
    view = memory.messages()
    memory.messages()
    # window=2 would start on a tool-result turn, so the cut moves back one: 4 dropped
    assert calls == [4]
    assert view[1] == Message.user("[Summary of earlier steps]\n4 messages dropped")
    assert view[2:] == _history()[-3:]
    memory.add(Message.user("more"))
    memory.messages()
    assert calls == [4, 6]


def test_small_histories_are_not_summarised() -> None:
    memory = Memory(window=5, summarize=lambda dropped: "never")
    memory.extend([Message.user("a"), Message.assistant("b")])
    assert memory.messages() == [Message.user("a"), Message.assistant("b")]


def test_invalid_window_and_clear() -> None:
    with pytest.raises(ValueError, match="window"):
        Memory(window=0)
    memory = Memory()
    memory.add(Message.user("x"))
    memory.clear()
    assert len(memory) == 0
