"""Shared test helpers: a few tiny tools and a registry fixture."""

from __future__ import annotations

import asyncio

import pytest

from agentloop import ToolRegistry, tool


@tool
def echo(text: str) -> str:
    """Return the text unchanged.

    Args:
        text: Anything.
    """
    return text


@tool
def add(a: int, b: int = 1) -> int:
    """Add two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


@tool
def boom(message: str = "boom") -> str:
    """Always fail.

    Args:
        message: Error text.
    """
    raise ValueError(message)


@tool
async def nap(seconds: float) -> str:
    """Sleep asynchronously then return how long it slept.

    Args:
        seconds: Duration.
    """
    await asyncio.sleep(seconds)
    return f"slept {seconds}"


@pytest.fixture
def registry() -> ToolRegistry:
    """Registry with echo, add, boom and nap."""
    return ToolRegistry([echo, add, boom, nap], default_timeout=5.0)
