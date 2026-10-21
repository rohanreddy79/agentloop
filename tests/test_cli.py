"""CLI: run with the fake provider, JSON output, trace show, and argument validation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from agentloop.cli import app

TASK = "Compute 17 * 23 + 5, read notes.txt, and write the result and the notes to result.txt."
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(text: str) -> str:
    """Strip ANSI styling and rich box drawing and collapse whitespace.

    GitHub Actions makes typer force a colored 80-column terminal, which wraps and styles
    error messages; assertions compare against the plain text instead.
    """
    text = _ANSI.sub("", text)
    text = re.sub(r"[\u2502\u256d\u256e\u2570\u256f\u2500]", " ", text)
    return " ".join(text.split())


runner = CliRunner()


def _sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "notes.md").write_text("hello\n")
    return root


def test_run_with_fake_provider_writes_trace(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    result = runner.invoke(app, ["run", TASK, "--root", str(root), "--trace", str(trace)])
    assert result.exit_code == 0, result.output
    assert "Done: 17 * 23 + 5 = 396" in result.output
    assert "5 steps" in result.output
    assert trace.exists()
    assert (root / "result.txt").read_text().startswith("17 * 23 + 5 = 396")


def test_run_json_output(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    result = runner.invoke(app, ["run", TASK, "--root", str(root), "--trace", str(trace), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok" and payload["steps"] == 5
    assert payload["usage"]["total_tokens"] > 0
    assert payload["trace"] == str(trace)


def test_trace_show_renders_a_run(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    assert (
        runner.invoke(app, ["run", TASK, "--root", str(root), "--trace", str(trace)]).exit_code == 0
    )
    result = runner.invoke(app, ["trace", "show", str(trace)])
    assert result.exit_code == 0, result.output
    assert "status: ok" in result.output
    assert "read_file error" in result.output
    assert "write_file ok" in result.output


def test_budget_exceeded_exits_with_code_2(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    result = runner.invoke(
        app, ["run", TASK, "--root", str(root), "--trace", str(trace), "--max-steps", "1"]
    )
    assert result.exit_code == 2
    assert "budget exceeded (max_steps)" in result.output


def test_argument_validation(tmp_path: Path) -> None:
    trace = str(tmp_path / "t.jsonl")
    assert runner.invoke(app, ["run", "x", "--provider", "nope", "--trace", trace]).exit_code != 0
    bad_tool = runner.invoke(app, ["run", "x", "--tools", "python", "--trace", trace])
    assert bad_tool.exit_code != 0 and "--allow-python" in _plain(bad_tool.output)
    no_hosts = runner.invoke(app, ["run", "x", "--tools", "http", "--trace", trace])
    assert no_hosts.exit_code != 0 and "--allowed-hosts" in _plain(no_hosts.output)
    unknown = runner.invoke(app, ["run", "x", "--tools", "laser", "--trace", trace])
    assert unknown.exit_code != 0 and "unknown tool 'laser'" in _plain(unknown.output)


def test_python_tool_can_be_enabled(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    result = runner.invoke(
        app,
        ["run", "Compute 2 + 2", "--tools", "calc,python", "--allow-python", "--trace", str(trace)],
    )
    assert result.exit_code == 0, result.output
    assert "2 + 2 = 4" in result.output
