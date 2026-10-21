"""``http_get`` with a host allowlist, a byte cap, no redirects and a timeout.

Redirects are not followed on purpose: a redirect is the easy way for an allowlisted host to
send the agent somewhere that is not. The transport is injectable so tests use
``httpx.MockTransport`` and never touch the network.
"""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlparse

import httpx

from agentloop.errors import HostNotAllowedError, ResponseTooLargeError
from agentloop.tools import Tool, tool


def make_http_tool(
    allowed_hosts: Sequence[str],
    *,
    max_bytes: int = 100_000,
    timeout: float = 10.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Tool:
    """Build an ``http_get`` tool restricted to ``allowed_hosts`` (and their subdomains)."""
    allowed = tuple(h.strip().lower() for h in allowed_hosts if h.strip())

    def host_allowed(host: str) -> bool:
        host = host.lower()
        return any(host == h or host.endswith("." + h) for h in allowed)

    @tool(timeout=timeout + 5)
    async def http_get(url: str) -> str:
        """Fetch a URL with HTTP GET; returns the status code and the body as text.

        Args:
            url: Absolute http(s) URL on an allowlisted host.
        """
        parts = urlparse(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"not an http(s) URL: {url}")
        if not host_allowed(parts.hostname):
            permitted = ", ".join(allowed) or "none"
            raise HostNotAllowedError(
                f"host {parts.hostname!r} is not allowlisted (allowed: {permitted})"
            )
        async with (
            httpx.AsyncClient(
                transport=transport, timeout=timeout, follow_redirects=False
            ) as client,
            client.stream("GET", url) as response,
        ):
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise ResponseTooLargeError(f"response exceeds {max_bytes} bytes")
            status = response.status_code
        return f"HTTP {status}\n{body.decode('utf-8', errors='replace')}"

    return http_get
