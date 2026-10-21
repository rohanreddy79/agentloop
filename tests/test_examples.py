"""The example scripts run offline exactly as documented in the README."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _run(
    script: str, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )


def test_offline_demo(tmp_path: Path) -> None:
    trace = tmp_path / "demo.jsonl"
    proc = _run("offline_demo.py", "--root", str(tmp_path / "sandbox"), "--trace", str(trace))
    assert proc.returncode == 0, proc.stderr
    assert (
        "step 1: calculator(expression='17 * 23 + 5'), read_file(path='notes.txt') -> ok, error"
        in proc.stdout
    )
    assert "answer: Done: 17 * 23 + 5 = 396" in proc.stdout
    assert "steps=5 tool_calls=5" in proc.stdout
    assert trace.exists()


def test_structured_extraction() -> None:
    proc = _run("structured_extraction.py")
    assert proc.returncode == 0, proc.stderr
    assert "rejected" in proc.stdout
    assert "name='Jane Doe'" in proc.stdout
    assert "plan='enterprise'" in proc.stdout


def test_anthropic_example_is_gated_on_the_api_key() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    proc = _run("anthropic_research.py", env=env)
    assert proc.returncode == 1
    assert "ANTHROPIC_API_KEY" in proc.stderr
