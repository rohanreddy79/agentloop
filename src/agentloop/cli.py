"""Command-line interface: ``agentloop run ...`` and ``agentloop trace show <file>``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from agentloop.agent import Agent, Budget, BudgetExceeded
from agentloop.builtin import calculator, make_fs_tools, make_http_tool, make_python_exec_tool
from agentloop.errors import AgentLoopError
from agentloop.providers import AnthropicProvider, Provider, RuleBasedFakeProvider
from agentloop.tools import Tool
from agentloop.tracing import Tracer, read_trace, render_trace

app = typer.Typer(
    help="A minimal, fully tested LLM agent runtime.",
    no_args_is_help=True,
    add_completion=False,
)
trace_app = typer.Typer(help="Inspect JSONL run traces.", no_args_is_help=True)
app.add_typer(trace_app, name="trace")

console = Console()
err_console = Console(stderr=True)

DEFAULT_SYSTEM = "You are a careful assistant. Use the tools when they help; answer concisely."
KNOWN_TOOLS = ("calc", "fs", "http", "python")


def _make_provider(name: str, model: str | None) -> Provider:
    if name == "fake":
        return RuleBasedFakeProvider()
    if name == "anthropic":
        return AnthropicProvider(model)
    raise typer.BadParameter(f"unknown provider {name!r}; use fake or anthropic")


def _make_tools(spec: str, root: Path, allowed_hosts: str, allow_python: bool) -> list[Tool]:
    tools: list[Tool] = []
    for name in (part.strip() for part in spec.split(",") if part.strip()):
        if name == "calc":
            tools.append(calculator)
        elif name == "fs":
            tools.extend(make_fs_tools(root))
        elif name == "http":
            hosts = [h for h in allowed_hosts.split(",") if h.strip()]
            if not hosts:
                raise typer.BadParameter("--allowed-hosts is required with the http tool")
            tools.append(make_http_tool(hosts))
        elif name == "python":
            if not allow_python:
                raise typer.BadParameter("the python tool requires --allow-python")
            tools.append(make_python_exec_tool())
        else:
            raise typer.BadParameter(f"unknown tool {name!r}; known: {', '.join(KNOWN_TOOLS)}")
    return tools


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="What the agent should do.")],
    provider: Annotated[str, typer.Option(help="fake (offline) or anthropic.")] = "fake",
    tools: Annotated[str, typer.Option(help="Comma-separated: calc,fs,http,python.")] = "calc,fs",
    root: Annotated[Path, typer.Option(help="Sandbox root for the fs tools.")] = Path("sandbox"),
    model: Annotated[str | None, typer.Option(help="Model id for the anthropic provider.")] = None,
    system: Annotated[str, typer.Option(help="System prompt.")] = DEFAULT_SYSTEM,
    max_steps: Annotated[int, typer.Option(help="Provider calls allowed.")] = 10,
    max_tokens: Annotated[int | None, typer.Option(help="Total token budget.")] = None,
    max_cost: Annotated[float | None, typer.Option(help="USD budget (priced models).")] = None,
    max_seconds: Annotated[float | None, typer.Option(help="Wall-clock budget.")] = None,
    trace: Annotated[
        Path | None, typer.Option(help="Trace file (default traces/<ts>.jsonl).")
    ] = None,
    allowed_hosts: Annotated[str, typer.Option(help="Comma-separated hosts for http_get.")] = "",
    allow_python: Annotated[bool, typer.Option(help="Enable the python_exec tool.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """Run one task through the agent loop and print the answer."""
    trace_path = trace or Path("traces") / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    try:
        agent = Agent(
            _make_provider(provider, model),
            _make_tools(tools, root, allowed_hosts, allow_python),
            system=system,
            budget=Budget(max_steps, max_tokens, max_cost, max_seconds),
            tracer=Tracer(trace_path),
        )
        result = agent.run(task)
    except BudgetExceeded as exc:
        err_console.print(f"[yellow]{exc}[/yellow]")
        _print_result(exc.result.output, exc.result, json_output, status="budget_exceeded")
        raise typer.Exit(code=2) from None
    except AgentLoopError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from None
    _print_result(result.output, result, json_output, status="ok")


def _print_result(output: str, result: object, as_json: bool, status: str) -> None:
    from agentloop.agent import RunResult

    assert isinstance(result, RunResult)
    if as_json:
        console.print_json(
            json.dumps(
                {
                    "status": status,
                    "output": output,
                    "steps": result.steps,
                    "usage": result.usage.to_dict(),
                    "cost_usd": result.cost_usd,
                    "stop_reason": result.stop_reason,
                    "elapsed_s": round(result.elapsed_s, 3),
                    "trace": str(result.trace_path) if result.trace_path else None,
                }
            )
        )
        return
    console.print(output)
    err_console.print(
        f"[dim]{result.steps} steps · {result.usage.total_tokens} tokens · "
        f"${result.cost_usd:.4f} · {result.elapsed_s:.2f}s · trace: {result.trace_path}[/dim]"
    )


@trace_app.command("show")
def trace_show(
    file: Annotated[Path, typer.Argument(help="A JSONL trace written by a run.", exists=True)],
) -> None:
    """Pretty-print a trace: one header, one table of model/tool events, the output."""
    render_trace(read_trace(file), console)


if __name__ == "__main__":  # pragma: no cover
    app()
