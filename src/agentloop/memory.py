"""Conversation memory: the message list the provider sees on each call.

The full history is always kept; ``window`` only limits the *view* handed to the provider.
The view always starts with the task message and never begins with a tool-result turn (its
matching tool-call turn is pulled in too), because the Messages API rejects a ``tool_result``
whose ``tool_use`` is missing. An optional ``summarize`` hook condenses the dropped middle
into one user message so the model keeps the gist of what it already did.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from agentloop.providers.base import Message

Summarizer = Callable[[Sequence[Message]], str]


class Memory:
    """Append-only history with an optional sliding window for the provider view."""

    def __init__(self, window: int | None = None, summarize: Summarizer | None = None) -> None:
        if window is not None and window < 1:
            raise ValueError("window must be >= 1 or None")
        self.window = window
        self.summarize = summarize
        self._messages: list[Message] = []
        self._summary_cache: tuple[int, str] | None = None

    def add(self, message: Message) -> None:
        """Append one message to the history."""
        self._messages.append(message)

    def extend(self, messages: Iterable[Message]) -> None:
        """Append several messages."""
        self._messages.extend(messages)

    def clear(self) -> None:
        """Forget everything (a new conversation)."""
        self._messages.clear()
        self._summary_cache = None

    def history(self) -> list[Message]:
        """The complete, unwindowed history."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def messages(self) -> list[Message]:
        """The provider view: full history, or task + summary + the last ``window`` turns."""
        total = len(self._messages)
        if self.window is None or total <= self.window + 1:
            return list(self._messages)
        start = total - self.window
        # Never cut between a tool-call turn and its results.
        while start > 1 and self._messages[start].tool_results:
            start -= 1
        head, dropped, tail = self._messages[0], self._messages[1:start], self._messages[start:]
        if not dropped or self.summarize is None:
            return [head, *tail]
        if self._summary_cache is None or self._summary_cache[0] != len(dropped):
            self._summary_cache = (len(dropped), self.summarize(dropped))
        summary = Message.user(f"[Summary of earlier steps]\n{self._summary_cache[1]}")
        return [head, summary, *tail]
