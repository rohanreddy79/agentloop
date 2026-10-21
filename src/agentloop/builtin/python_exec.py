"""Opt-in ``python_exec``: run a snippet in a fresh ``python -I`` subprocess with a timeout.

This is a convenience, not a security boundary. The subprocess runs with the agent's own OS
permissions; isolated mode (``-I``) only keeps the agent's environment variables and working
directory off ``sys.path``. It is disabled unless explicitly requested (``--allow-python``).
"""

from __future__ import annotations

import asyncio
import sys

from agentloop.errors import PythonExecTimeoutError
from agentloop.tools import Tool, tool


def make_python_exec_tool(*, timeout: float = 10.0, max_output_chars: int = 10_000) -> Tool:
    """Build a ``python_exec`` tool that kills the interpreter after ``timeout`` seconds."""

    @tool(timeout=timeout + 5)
    async def python_exec(code: str) -> str:
        """Run Python source in a fresh interpreter and return exit code, stdout and stderr.

        Args:
            code: Python source to execute; print() whatever you need to see.
        """
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise PythonExecTimeoutError(f"python_exec timed out after {timeout}s") from None
        stdout = out.decode(errors="replace")[:max_output_chars]
        stderr = err.decode(errors="replace")[:max_output_chars]
        return f"exit code: {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"

    return python_exec
