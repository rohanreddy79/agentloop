"""Measure the runtime's own overhead, with no model latency involved.

    uv run python examples/bench_overhead.py

Three numbers: loop iterations per second with the scripted fake provider (one provider call,
one tool call and all bookkeeping per step, with and without tracing); wall time for N async
tools that each sleep 50 ms, executed by the registry concurrently versus one at a time; and
the time to build a tool (schema derivation) from a five-parameter function.
"""

from __future__ import annotations

import asyncio
import platform
import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from agentloop import Agent, Budget, FakeProvider, Memory, ToolCall, ToolRegistry, Tracer, tool


@tool
def echo(text: str) -> str:
    """Return the text.

    Args:
        text: Anything.
    """
    return text


@tool
async def nap(seconds: float) -> str:
    """Sleep.

    Args:
        seconds: Duration.
    """
    await asyncio.sleep(seconds)
    return "ok"


class Options(BaseModel):
    verbose: bool = False
    tags: list[str] = []


def bench_loop(steps: int, trace: Path | None) -> float:
    """Steps per second for a run of ``steps`` provider calls (window keeps the view small)."""
    script = [("echo", {"text": "x"})] * (steps - 1) + ["done"]
    agent = Agent(
        FakeProvider(script),
        [echo],
        budget=Budget(max_steps=steps + 1),
        memory=Memory(window=8),
        tracer=Tracer(trace) if trace else None,
    )
    started = time.perf_counter()
    result = agent.run("bench")
    elapsed = time.perf_counter() - started
    assert result.steps == steps
    return steps / elapsed


async def bench_parallel(n: int, seconds: float) -> tuple[float, float]:
    """(concurrent, sequential) wall seconds for ``n`` tools sleeping ``seconds`` each."""
    registry = ToolRegistry([nap])
    calls = [ToolCall(f"c{i}", "nap", {"seconds": seconds}) for i in range(n)]
    started = time.perf_counter()
    await registry.execute_all(calls)
    concurrent = time.perf_counter() - started
    started = time.perf_counter()
    for call in calls:
        await registry.execute(call)
    sequential = time.perf_counter() - started
    return concurrent, sequential


def bench_schema(n: int) -> float:
    """Microseconds to turn a five-parameter function into a Tool, averaged over ``n``."""

    def make() -> None:
        @tool
        def search(
            query: str,
            limit: int = 10,
            score: float | None = None,
            mode: Literal["fast", "deep"] = "fast",
            options: Options | None = None,
        ) -> str:
            """Search.

            Args:
                query: Text.
                limit: Max results.
                score: Minimum score.
                mode: Strategy.
                options: Extras.
            """
            return query

    started = time.perf_counter()
    for _ in range(n):
        make()
    return (time.perf_counter() - started) / n * 1e6


def main() -> None:
    print(f"python {platform.python_version()} on {platform.machine()} ({platform.system()})\n")
    with tempfile.TemporaryDirectory() as tmp:
        plain = bench_loop(300, None)
        traced = bench_loop(300, Path(tmp) / "trace.jsonl")
    concurrent, sequential = asyncio.run(bench_parallel(8, 0.05))
    schema_us = bench_schema(200)
    rows = [
        ("loop steps/s: fake provider + 1 tool call per step, no tracing", f"{plain:,.0f}"),
        ("loop steps/s: same, with JSONL tracing on", f"{traced:,.0f}"),
        ("8 async tools x 50 ms, concurrent (registry.execute_all)", f"{concurrent * 1000:.0f} ms"),
        ("8 async tools x 50 ms, one at a time", f"{sequential * 1000:.0f} ms"),
        ("build a 5-parameter tool (schema derivation), mean of 200", f"{schema_us:.0f} us"),
    ]
    print("| measurement | result |")
    print("|---|---|")
    for label, value in rows:
        print(f"| {label} | {value} |")


if __name__ == "__main__":
    main()
