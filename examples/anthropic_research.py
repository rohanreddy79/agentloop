"""Live example (optional, needs a key): answer a question with http_get and the calculator.

    ANTHROPIC_API_KEY=sk-ant-... uv run python examples/anthropic_research.py "your question"

Not run by the tests or CI. The default question fetches one allowlisted page and does a bit
of arithmetic so the trace shows both tools; the run is capped at 8 steps and $0.50.
"""

from __future__ import annotations

import os
import sys

from agentloop import Agent, AnthropicProvider, Budget, StepEvent, Tracer
from agentloop.builtin import calculator, make_http_tool

DEFAULT_QUESTION = (
    "Fetch https://www.python.org/ and tell me the first headline you see, then use the "
    "calculator to compute 2 ** 20."
)


def show(event: StepEvent) -> None:
    """One line per step with cumulative cost."""
    calls = ", ".join(c.name for c in event.response.tool_calls) or "text"
    print(f"step {event.step}: {calls}  (${event.cost_usd:.4f} so far)", file=sys.stderr)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("set ANTHROPIC_API_KEY to run this example (see .env.example)", file=sys.stderr)
        sys.exit(1)
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    agent = Agent(
        AnthropicProvider(),  # model from AGENTLOOP_MODEL or the package default
        [calculator, make_http_tool(["python.org"])],
        system="You are a careful research assistant. Use the tools; answer in two sentences.",
        budget=Budget(max_steps=8, max_cost_usd=0.50),
        tracer=Tracer("traces/anthropic_research.jsonl"),
        on_step=show,
    )
    result = agent.run(question)
    print(result.output)
    print(
        f"\nsteps={result.steps} tokens={result.usage.total_tokens} "
        f"cost=${result.cost_usd:.4f} trace={result.trace_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
