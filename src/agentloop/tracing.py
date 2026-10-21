"""JSONL run traces and a terminal renderer for them.

One line per event so a trace can be tailed while a run is in progress and parsed with any
JSON tool. Events: ``run_start``, ``step`` (one provider call, with the messages it added),
``tool_call`` (one execution, with latency and error flag) and ``run_end``. Cost comes from
:mod:`agentloop.pricing`; unpriced models are flagged rather than priced at zero silently.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import IO, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentloop.pricing import estimate_cost
from agentloop.providers.base import Message, ModelResponse, ToolCall, ToolResult, Usage


class Tracer:
    """Writes trace events for one or more runs to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = self.path.open("a", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        """Write one event line (flushed immediately so partial runs are inspectable)."""
        record = {"event": event, "ts": round(time.time(), 3), **fields}
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def run_start(
        self, run_id: str, task: str, provider: str, model: str, budget: dict[str, Any]
    ) -> None:
        """Record the start of a run."""
        self.emit(
            "run_start", run_id=run_id, task=task, provider=provider, model=model, budget=budget
        )

    def step(
        self,
        run_id: str,
        step: int,
        response: ModelResponse,
        latency_ms: float,
        messages_delta: Sequence[Message],
    ) -> None:
        """Record one provider call and the messages it appended."""
        cost = estimate_cost(response.model, response.usage)
        self.emit(
            "step",
            run_id=run_id,
            step=step,
            latency_ms=round(latency_ms, 1),
            stop_reason=response.stop_reason,
            text=response.text,
            tool_calls=[asdict(c) for c in response.tool_calls],
            usage=response.usage.to_dict(),
            cost_usd=cost or 0.0,
            price_known=cost is not None,
            messages_delta=[m.to_dict() for m in messages_delta],
        )

    def tool_call(
        self, run_id: str, step: int, call: ToolCall, result: ToolResult, latency_ms: float
    ) -> None:
        """Record one tool execution."""
        self.emit(
            "tool_call",
            run_id=run_id,
            step=step,
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            result=result.content,
            is_error=result.is_error,
            latency_ms=round(latency_ms, 1),
        )

    def run_end(
        self,
        run_id: str,
        status: str,
        steps: int,
        usage: Usage,
        cost_usd: float,
        elapsed_s: float,
        output: str,
    ) -> None:
        """Record the end of a run (``status`` is ``ok``, ``budget_exceeded`` or ``error``)."""
        self.emit(
            "run_end",
            run_id=run_id,
            status=status,
            steps=steps,
            usage=usage.to_dict(),
            cost_usd=cost_usd,
            elapsed_s=round(elapsed_s, 3),
            output=output,
        )

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Parse a JSONL trace into a list of event dicts (blank lines ignored)."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def render_trace(events: Iterable[dict[str, Any]], console: Console | None = None) -> None:
    """Pretty-print trace events with rich (one panel + table per run)."""
    console = console or Console()
    runs: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        runs.setdefault(event.get("run_id", "?"), []).append(event)
    for run_id, run_events in runs.items():
        _render_run(run_id, run_events, console)


def _render_run(run_id: str, events: list[dict[str, Any]], console: Console) -> None:
    start = next((e for e in events if e["event"] == "run_start"), {})
    end = next((e for e in events if e["event"] == "run_end"), {})
    usage = end.get("usage", {})
    priced = all(e.get("price_known", True) for e in events if e["event"] == "step")
    cost = f"${end.get('cost_usd', 0.0):.4f}" + ("" if priced else " (model not in price table)")
    header = (
        f"[bold]task:[/bold] {start.get('task', '?')}\n"
        f"[bold]provider:[/bold] {start.get('provider', '?')} ({start.get('model', '?')})   "
        f"[bold]status:[/bold] {end.get('status', 'in progress')}   "
        f"[bold]steps:[/bold] {end.get('steps', '?')}   "
        f"[bold]tokens:[/bold] {usage.get('total_tokens', '?')}   "
        f"[bold]cost:[/bold] {cost}   "
        f"[bold]elapsed:[/bold] {end.get('elapsed_s', '?')}s"
    )
    console.print(Panel(header, title=f"run {run_id}", expand=False))

    table = Table(show_lines=False, expand=False)
    table.add_column("step", justify="right")
    table.add_column("event")
    table.add_column("detail", overflow="fold", max_width=90)
    table.add_column("ms", justify="right")
    for event in _model_rows_first(events):
        kind = event["event"]
        if kind == "step":
            calls = ", ".join(
                f"{c['name']}({_short_args(c['arguments'])})" for c in event["tool_calls"]
            )
            detail = f"-> {calls}" if calls else f"text: {_short(event['text'])}"
            detail += f"  [dim]({event['usage']['total_tokens']} tok, {event['stop_reason']})[/dim]"
            table.add_row(str(event["step"]), "model", detail, f"{event['latency_ms']:.0f}")
        elif kind == "tool_call":
            style = "red" if event["is_error"] else "green"
            label = "error" if event["is_error"] else "ok"
            detail = f"[{style}]{event['name']} {label}:[/{style}] {_short(event['result'])}"
            table.add_row(str(event["step"]), "tool", detail, f"{event['latency_ms']:.0f}")
    console.print(table)
    if end:
        console.print(Panel(_short(end.get("output", ""), 600), title="output", expand=False))


def _short(text: str, limit: int = 160) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _short_args(arguments: dict[str, Any]) -> str:
    return _short(", ".join(f"{k}={v!r}" for k, v in arguments.items()), 80)


def _model_rows_first(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool events are logged before their step event; show the model row first per step."""
    ordered = sorted(
        (e for e in events if e["event"] in ("step", "tool_call")),
        key=lambda e: (e["step"], 0 if e["event"] == "step" else 1),
    )
    return ordered
