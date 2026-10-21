"""Sandboxed file tools: traversal and symlink escapes are rejected, errors stay relative."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentloop import ToolCall, ToolRegistry
from agentloop.builtin import make_fs_tools, resolve_in_root
from agentloop.errors import PathOutsideRootError


def _tools(root: Path) -> dict[str, object]:
    return {t.name: t for t in make_fs_tools(root)}


def test_read_write_list_round_trip(tmp_path: Path) -> None:
    tools = _tools(tmp_path)
    assert tools["write_file"]("sub/a.txt", "hello") == "wrote 5 bytes to sub/a.txt"
    assert tools["read_file"]("sub/a.txt") == "hello"
    assert tools["list_dir"](".") == "sub/"
    assert tools["list_dir"]("sub") == "a.txt"
    assert tools["list_dir"]("sub") == "a.txt"


@pytest.mark.parametrize("bad", ["../secret.txt", "/etc/passwd", "sub/../../x", "../../", ".."])
def test_path_traversal_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(PathOutsideRootError, match="outside the sandbox root"):
        resolve_in_root(tmp_path, bad)


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.txt").symlink_to(outside / "secret.txt")
    (root / "linkdir").symlink_to(outside)
    tools = _tools(root)
    with pytest.raises(PathOutsideRootError):
        tools["read_file"]("link.txt")
    with pytest.raises(PathOutsideRootError):
        tools["write_file"]("linkdir/new.txt", "x")
    with pytest.raises(PathOutsideRootError):
        tools["list_dir"]("linkdir")


def test_root_itself_is_allowed(tmp_path: Path) -> None:
    assert resolve_in_root(tmp_path, ".") == tmp_path.resolve()
    assert _tools(tmp_path)["list_dir"]() == "(empty)"


def test_missing_file_error_uses_relative_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as info:
        _tools(tmp_path)["read_file"]("nope.txt")
    assert str(info.value) == "no such file: nope.txt"


def test_size_caps(tmp_path: Path) -> None:
    tools = {t.name: t for t in make_fs_tools(tmp_path, max_read_bytes=4, max_write_bytes=4)}
    with pytest.raises(ValueError, match="larger than 4 bytes"):
        tools["write_file"]("big.txt", "12345")
    (tmp_path / "big.txt").write_text("12345")
    with pytest.raises(ValueError, match="larger than 4 bytes"):
        tools["read_file"]("big.txt")


def test_list_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError, match="no such directory: nope"):
        _tools(tmp_path)["list_dir"]("nope")


async def test_traversal_reaches_model_as_error_result(tmp_path: Path) -> None:
    registry = ToolRegistry(make_fs_tools(tmp_path))
    result = await registry.execute(ToolCall("c1", "read_file", {"path": "../x"}))
    assert result.is_error
    assert result.content == "PathOutsideRootError: '../x' is outside the sandbox root"
