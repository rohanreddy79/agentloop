"""The agent loop: call the model, run its tool calls concurrently, append results, repeat.

The loop stops on the first text-only response, or raises :class:`BudgetExceeded` (carrying a
partial :class:`RunResult`) when a budget limit is hit. With ``output_model`` set, a hidden
``final_answer`` tool whose schema is the model's is advertised; the model's call is validated
and, on failure, the validation message is returned as an error tool result so it can retry.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from agentloop.errors import AgentLoopError, DuplicateToolError, StructuredOutputError
from agentloop.memory import Memory
from agentloop.pricing import estimate_cost
from agentloop.providers.base import (
    Message,
    ModelResponse,
    Provider,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)
from agentloop.schema import format_validation_error, json_schema_for
from agentloop.tools import Tool, ToolRegistry
from agentloop.tracing import Tracer

FINAL_ANSWER = "final_answer"
_FINAL_ANSWER_INSTRUCTION = (
    "When you have the final answer, call the `final_answer` tool with it; that call ends the "
    "task. If the tool reports a validation error, fix the fields and call it again. Do not "
    "give the final answer as plain text."
)
_NUDGE = "Call the `final_answer` tool with your answer to finish."


@dataclass(frozen=True, slots=True)
class Budget:
    """Hard limits for one run. ``None`` disables a limit; ``max_steps`` is always on.

    ``max_tokens`` counts every token billed across all provider calls (input + output +
    cache). ``max_cost_usd`` only trips for models in the price table.
    """

    max_steps: int = 10
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for traces."""
        return asdict(self)


@dataclass
class RunResult:
    """Outcome of a run (also attached to :class:`BudgetExceeded` as the partial result)."""

    output: str
    parsed: Any = None
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    stop_reason: str = ""
    elapsed_s: float = 0.0
    messages: list[Message] = field(default_factory=list)
    trace_path: Path | None = None
    run_id: str = ""


class BudgetExceeded(AgentLoopError):
    """A budget limit was hit; ``result`` holds what the run produced so far."""

    def __init__(self, limit: str, detail: str, result: RunResult) -> None:
        super().__init__(f"budget exceeded ({limit}): {detail}")
        self.limit = limit
        self.detail = detail
        self.result = result


@dataclass(frozen=True, slots=True)
class StepEvent:
    """Passed to the ``on_step`` hook after each provider call and its tool executions."""

    run_id: str
    step: int
    response: ModelResponse
    tool_results: tuple[ToolResult, ...]
    usage: Usage
    cost_usd: float
    elapsed_s: float


OnStep = Callable[[StepEvent], None]


@dataclass
class _RunState:
    run_id: str
    started: float
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    last_text: str = ""
    stop_reason: str = ""
    retries: int = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


class Agent:
    """Provider + tools + system prompt + budget, driven by :meth:`run` / :meth:`arun`.

    The agent keeps its :class:`Memory` between runs, so calling ``run`` twice continues the
    same conversation.
    """

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry | Iterable[Tool | Callable[..., Any]] = (),
        system: str = "",
        budget: Budget | None = None,
        memory: Memory | None = None,
        tracer: Tracer | None = None,
        *,
        max_tokens: int = 4096,
        on_step: OnStep | None = None,
        max_output_retries: int = 3,
        tool_timeout: float = 30.0,
    ) -> None:
        self.provider = provider
        self.registry = (
            tools
            if isinstance(tools, ToolRegistry)
            else ToolRegistry(tools, default_timeout=tool_timeout)
        )
        self.system = system
        self.budget = budget or Budget()
        self.memory = memory if memory is not None else Memory()
        self.tracer = tracer
        self.max_tokens = max_tokens
        self.on_step = on_step
        self.max_output_retries = max_output_retries

    def run(self, task: str, output_model: type[BaseModel] | None = None) -> RunResult:
        """Blocking entry point (use :meth:`arun` from inside an event loop)."""
        return asyncio.run(self.arun(task, output_model=output_model))

    async def arun(self, task: str, output_model: type[BaseModel] | None = None) -> RunResult:
        """Run ``task`` to completion; returns the parsed object in ``result.parsed`` if asked."""
        state = _RunState(run_id=uuid.uuid4().hex[:8], started=time.monotonic())
        tools, system = self._prepare(output_model)
        self.memory.add(Message.user(task))
        if self.tracer:
            self.tracer.run_start(
                state.run_id, task, self.provider.name, self.provider.model, self.budget.to_dict()
            )
        try:
            result = await self._loop(state, tools, system, output_model)
        except BudgetExceeded as exc:
            self._trace_end(state, "budget_exceeded", exc.result.output)
            raise
        except Exception:
            self._trace_end(state, "error", state.last_text)
            raise
        self._trace_end(state, "ok", result.output)
        return result

    # -- loop -----------------------------------------------------------------------------

    async def _loop(
        self,
        state: _RunState,
        tools: list[ToolDefinition],
        system: str,
        output_model: type[BaseModel] | None,
    ) -> RunResult:
        while True:
            self._check_before_call(state)
            state.steps += 1
            started = time.monotonic()
            response = await self.provider.complete(
                self.memory.messages(), tools, system or None, self.max_tokens
            )
            latency_ms = (time.monotonic() - started) * 1000
            state.usage += response.usage
            state.cost_usd += (
                estimate_cost(response.model or self.provider.model, response.usage) or 0.0
            )
            state.stop_reason = response.stop_reason
            if response.text:
                state.last_text = response.text

            assistant = response.to_message()
            self.memory.add(assistant)
            delta = [assistant]
            results: tuple[ToolResult, ...] = ()
            parsed: BaseModel | None = None
            invalid_final = False
            if response.tool_calls:
                results, parsed, invalid_final = await self._dispatch(
                    state, response.tool_calls, output_model
                )
                results_msg = Message.results(results)
                self.memory.add(results_msg)
                delta.append(results_msg)

            if self.tracer:
                self.tracer.step(state.run_id, state.steps, response, latency_ms, delta)
            if self.on_step:
                self.on_step(
                    StepEvent(
                        state.run_id,
                        state.steps,
                        response,
                        results,
                        state.usage,
                        state.cost_usd,
                        state.elapsed,
                    )
                )

            if parsed is not None:
                return self._finish(state, parsed.model_dump_json(), parsed)
            if not response.tool_calls:
                if output_model is None:
                    return self._finish(state, response.text, None)
                # Text instead of the required final_answer call: nudge and count a retry.
                invalid_final = True
                self.memory.add(Message.user(_NUDGE))
            if invalid_final:
                state.retries += 1
                if state.retries > self.max_output_retries:
                    raise StructuredOutputError(
                        f"no valid final_answer after {self.max_output_retries} retries"
                    )
            self._check_after_call(state)

    async def _dispatch(
        self,
        state: _RunState,
        calls: Sequence[ToolCall],
        output_model: type[BaseModel] | None,
    ) -> tuple[tuple[ToolResult, ...], BaseModel | None, bool]:
        """Run all calls concurrently; ``final_answer`` is validated instead of executed."""

        async def run_one(call: ToolCall) -> tuple[ToolResult, BaseModel | None]:
            started = time.monotonic()
            if output_model is not None and call.name == FINAL_ANSWER:
                result, parsed = _validate_final(call, output_model)
            else:
                result, parsed = await self.registry.execute(call), None
            if self.tracer:
                latency = (time.monotonic() - started) * 1000
                self.tracer.tool_call(state.run_id, state.steps, call, result, latency)
            return result, parsed

        outcomes = await asyncio.gather(*(run_one(c) for c in calls))
        results = tuple(result for result, _ in outcomes)
        parsed = next((p for _, p in outcomes if p is not None), None)
        invalid_final = any(
            r.is_error for (r, _), c in zip(outcomes, calls, strict=True) if c.name == FINAL_ANSWER
        )
        return results, parsed, invalid_final

    # -- helpers --------------------------------------------------------------------------

    def _prepare(self, output_model: type[BaseModel] | None) -> tuple[list[ToolDefinition], str]:
        tools = self.registry.definitions()
        system = self.system
        if output_model is not None:
            if FINAL_ANSWER in self.registry:
                raise DuplicateToolError(f"{FINAL_ANSWER!r} is reserved for structured output")
            tools.append(
                ToolDefinition(
                    FINAL_ANSWER,
                    "Submit the final answer. Call exactly once, when the task is complete.",
                    json_schema_for(output_model),
                )
            )
            system = f"{system}\n\n{_FINAL_ANSWER_INSTRUCTION}".strip()
        return tools, system

    def _check_before_call(self, state: _RunState) -> None:
        budget = self.budget
        if state.steps >= budget.max_steps:
            raise BudgetExceeded("max_steps", f"{state.steps} steps used", self._partial(state))
        if budget.max_seconds is not None and state.elapsed > budget.max_seconds:
            raise BudgetExceeded(
                "max_seconds", f"{state.elapsed:.2f}s elapsed", self._partial(state)
            )

    def _check_after_call(self, state: _RunState) -> None:
        budget = self.budget
        if budget.max_tokens is not None and state.usage.total_tokens > budget.max_tokens:
            raise BudgetExceeded(
                "max_tokens", f"{state.usage.total_tokens} tokens used", self._partial(state)
            )
        if budget.max_cost_usd is not None and state.cost_usd > budget.max_cost_usd:
            raise BudgetExceeded(
                "max_cost_usd", f"${state.cost_usd:.4f} spent", self._partial(state)
            )

    def _partial(self, state: _RunState) -> RunResult:
        result = self._finish(state, state.last_text, None)
        result.stop_reason = "budget_exceeded"
        return result

    def _finish(self, state: _RunState, output: str, parsed: Any) -> RunResult:
        return RunResult(
            output=output,
            parsed=parsed,
            steps=state.steps,
            usage=state.usage,
            cost_usd=state.cost_usd,
            stop_reason=state.stop_reason,
            elapsed_s=state.elapsed,
            messages=self.memory.history(),
            trace_path=self.tracer.path if self.tracer else None,
            run_id=state.run_id,
        )

    def _trace_end(self, state: _RunState, status: str, output: str) -> None:
        if self.tracer:
            self.tracer.run_end(
                state.run_id,
                status,
                state.steps,
                state.usage,
                state.cost_usd,
                state.elapsed,
                output,
            )


def _validate_final(call: ToolCall, model: type[BaseModel]) -> tuple[ToolResult, BaseModel | None]:
    try:
        parsed = model.model_validate(call.arguments)
    except ValidationError as exc:
        detail = format_validation_error(exc)
        return (
            ToolResult(
                call.id,
                f"final_answer rejected: {detail}. Fix the fields and call final_answer again.",
                is_error=True,
            ),
            None,
        )
    return ToolResult(call.id, "final answer accepted"), parsed
