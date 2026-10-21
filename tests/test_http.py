"""http_get: allowlist, size cap, scheme check and no redirects, all via a mock transport."""

from __future__ import annotations

import httpx
import pytest

from agentloop import ToolCall, ToolRegistry
from agentloop.builtin import make_http_tool
from agentloop.errors import HostNotAllowedError, ResponseTooLargeError


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/big":
        return httpx.Response(200, content=b"x" * 500)
    if request.url.path == "/redirect":
        return httpx.Response(302, headers={"location": "https://evil.example/"})
    return httpx.Response(200, text=f"hello from {request.url.host}")


def _tool(**kwargs: object) -> object:
    return make_http_tool(
        ["example.com", " Docs.Python.org "], transport=httpx.MockTransport(_handler), **kwargs
    )


async def test_get_allowed_host() -> None:
    http_get = _tool()
    assert await http_get("https://example.com/page") == "HTTP 200\nhello from example.com"


async def test_subdomain_and_case_insensitive_host() -> None:
    http_get = _tool()
    assert (await http_get("https://api.example.com/")).startswith("HTTP 200")
    assert (await http_get("https://DOCS.python.org/3/")).startswith("HTTP 200")


async def test_disallowed_host_rejected_before_any_request() -> None:
    http_get = _tool()
    with pytest.raises(HostNotAllowedError, match=r"'evil\.example' is not allowlisted"):
        await http_get("https://evil.example/")
    with pytest.raises(HostNotAllowedError):
        await http_get("https://example.com.evil.example/")


async def test_size_cap() -> None:
    http_get = _tool(max_bytes=100)
    with pytest.raises(ResponseTooLargeError, match="exceeds 100 bytes"):
        await http_get("https://example.com/big")


async def test_non_http_scheme_rejected() -> None:
    with pytest.raises(ValueError, match="not an http"):
        await _tool()("ftp://example.com/x")


async def test_redirects_are_not_followed() -> None:
    assert (await _tool()("https://example.com/redirect")).startswith("HTTP 302")


async def test_errors_reach_model_as_error_results() -> None:
    registry = ToolRegistry([_tool()])
    result = await registry.execute(ToolCall("c1", "http_get", {"url": "https://evil.example/"}))
    assert result.is_error and result.content.startswith("HostNotAllowedError")
