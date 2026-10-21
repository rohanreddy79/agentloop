"""Structured output, offline: the model must answer through the hidden ``final_answer`` tool.

    uv run python examples/structured_extraction.py

The scripted provider's first ``final_answer`` call has two schema violations, so the loop
returns the validation message as an error tool result and the "model" retries correctly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentloop import FINAL_ANSWER, Agent, FakeProvider, StepEvent


class Contact(BaseModel):
    """A sales contact extracted from free text."""

    name: str
    email: str
    plan: Literal["free", "pro", "enterprise"]
    interests: list[str] = Field(default_factory=list)
    demo_requested: bool


TEXT = (
    "Jane Doe (jane@example.com) wants the Enterprise plan; she is interested in the API "
    "and SDKs and asked for a demo."
)

# What a real model might send: a wrong enum case and a string where a list is required,
# then a corrected call after seeing the validation errors.
SCRIPT = [
    (
        FINAL_ANSWER,
        {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "plan": "Enterprise",
            "interests": "API, SDKs",
            "demo_requested": True,
        },
    ),
    (
        FINAL_ANSWER,
        {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "plan": "enterprise",
            "interests": ["API", "SDKs"],
            "demo_requested": True,
        },
    ),
]


def show(event: StepEvent) -> None:
    """Print the final_answer arguments and how the loop judged them."""
    for call, result in zip(event.response.tool_calls, event.tool_results, strict=True):
        status = "rejected" if result.is_error else "accepted"
        print(f"step {event.step}: {call.name}({call.arguments}) -> {status}")
        if result.is_error:
            print(f"  {result.content}")


def main() -> None:
    agent = Agent(FakeProvider(SCRIPT), system="Extract the contact details.", on_step=show)
    result = agent.run(f"Extract the contact from: {TEXT}", output_model=Contact)
    print(f"\nparsed ({type(result.parsed).__name__}): {result.parsed!r}")
    print(f"steps={result.steps}")


if __name__ == "__main__":
    main()
