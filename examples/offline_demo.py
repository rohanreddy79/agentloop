"""Offline demo: a multi-step task solved by the deterministic rule-based provider.

    uv run python examples/offline_demo.py [--root DIR] [--trace FILE]

The sandbox contains ``notes.md`` but the task asks for ``notes.txt``, so the first read fails
with an ``is_error`` tool result. The "model" recovers by listing the directory, reads the
right file, writes the combined result and answers. Five tool calls, no API key, no network.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from agentloop import Agent, Budget, RuleBasedFakeProvider, StepEvent, Tracer
from agentloop.builtin import calculator, make_fs_tools

TASK = "Compute 17 * 23 + 5, read notes.txt, and write the result and the notes to result.txt."
NOTES = "Q3 planning notes\n- ship agentloop 0.1\n- write the README\n"
SYSTEM = "You are a careful assistant. Use the tools when they help; answer concisely."


def print_step(event: StepEvent) -> None:
    """Print one line per loop iteration as it happens."""
    response = event.response
    if response.tool_calls:
        calls = ", ".join(f"{c.name}({_args(c.arguments)})" for c in response.tool_calls)
        outcome = ", ".join("error" if r.is_error else "ok" for r in event.tool_results)
        print(f"step {event.step}: {calls} -> {outcome}")
    else:
        print(f"step {event.step}: text answer")


def _args(arguments: dict[str, object]) -> str:
    rendered = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    return rendered if len(rendered) <= 60 else rendered[:57] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, help="sandbox directory (default: a temp dir)")
    parser.add_argument(
        "--trace", type=Path, default=Path("traces/offline_demo.jsonl"), help="trace file"
    )
    args = parser.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="agentloop-demo-"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "notes.md").write_text(NOTES, encoding="utf-8")
    (root / "result.txt").unlink(missing_ok=True)
    args.trace.unlink(missing_ok=True)

    agent = Agent(
        RuleBasedFakeProvider(),
        [calculator, *make_fs_tools(root)],
        system=SYSTEM,
        budget=Budget(max_steps=8),
        tracer=Tracer(args.trace),
        on_step=print_step,
    )
    print(f"task: {TASK}\n")
    result = agent.run(TASK)
    print(f"\nanswer: {result.output}")
    print(
        f"steps={result.steps} tool_calls={sum(len(m.tool_calls) for m in result.messages)} "
        f"tokens={result.usage.total_tokens} elapsed={result.elapsed_s:.3f}s"
    )
    print(f"trace: {result.trace_path}")
    print(f"\n{root / 'result.txt'}:\n{(root / 'result.txt').read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
