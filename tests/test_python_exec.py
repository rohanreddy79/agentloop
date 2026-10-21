"""python_exec runs in a subprocess, reports exit codes, and is killed on timeout."""

from __future__ import annotations

import time

import pytest

from agentloop.builtin import make_python_exec_tool
from agentloop.errors import PythonExecTimeoutError


async def test_runs_code_and_captures_output() -> None:
    python_exec = make_python_exec_tool()
    out = await python_exec("print(1 + 2)\nimport sys; print('err', file=sys.stderr)")
    assert out.startswith("exit code: 0\nstdout:\n3\n")
    assert "stderr:\nerr" in out


async def test_nonzero_exit_code_reported() -> None:
    out = await make_python_exec_tool()("raise SystemExit(3)")
    assert out.startswith("exit code: 3")


async def test_timeout_kills_the_process() -> None:
    python_exec = make_python_exec_tool(timeout=0.5)
    started = time.monotonic()
    with pytest.raises(PythonExecTimeoutError, match=r"timed out after 0\.5s"):
        await python_exec("import time; time.sleep(10)")
    assert time.monotonic() - started < 3


async def test_output_is_truncated() -> None:
    out = await make_python_exec_tool(max_output_chars=10)("print('x' * 100)")
    assert "stdout:\nxxxxxxxxxx\n" in out
