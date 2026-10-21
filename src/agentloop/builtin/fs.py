"""Sandboxed file tools bound to a root directory.

Every path is resolved (symlinks followed) and must stay inside the root, so ``../x``,
absolute paths and symlinks pointing outside are all rejected with
:class:`PathOutsideRootError`. Errors carry the path as the model wrote it, not the absolute
location on disk.
"""

from __future__ import annotations

from pathlib import Path

from agentloop.errors import PathOutsideRootError
from agentloop.tools import Tool, tool


def resolve_in_root(root: Path, path: str) -> Path:
    """Absolute path for ``path`` under ``root``; raises if it escapes the root."""
    base = root.resolve()
    target = (base / path).resolve()
    if target != base and not target.is_relative_to(base):
        raise PathOutsideRootError(f"{path!r} is outside the sandbox root")
    return target


def make_fs_tools(
    root: str | Path, *, max_read_bytes: int = 1_000_000, max_write_bytes: int = 1_000_000
) -> list[Tool]:
    """``read_file``, ``write_file`` and ``list_dir`` bound to ``root`` (created if missing)."""
    base = Path(root).resolve()
    base.mkdir(parents=True, exist_ok=True)

    @tool
    def read_file(path: str) -> str:
        """Read a UTF-8 text file from the sandbox.

        Args:
            path: File path relative to the sandbox root.
        """
        target = resolve_in_root(base, path)
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        if target.stat().st_size > max_read_bytes:
            raise ValueError(f"{path} is larger than {max_read_bytes} bytes")
        return target.read_text(encoding="utf-8")

    @tool
    def write_file(path: str, content: str) -> str:
        """Write UTF-8 text to a file in the sandbox (overwrites; creates parent directories).

        Args:
            path: File path relative to the sandbox root.
            content: The full text to write.
        """
        target = resolve_in_root(base, path)
        data = content.encode("utf-8")
        if len(data) > max_write_bytes:
            raise ValueError(f"content is larger than {max_write_bytes} bytes")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return f"wrote {len(data)} bytes to {path}"

    @tool
    def list_dir(path: str = ".") -> str:
        """List a sandbox directory, one entry per line; directories end with '/'.

        Args:
            path: Directory path relative to the sandbox root.
        """
        target = resolve_in_root(base, path)
        if not target.is_dir():
            raise NotADirectoryError(f"no such directory: {path}")
        entries = sorted(target.iterdir(), key=lambda p: p.name)
        lines = [f"{p.name}/" if p.is_dir() else p.name for p in entries]
        return "\n".join(lines) if lines else "(empty)"

    return [read_file, write_file, list_dir]
