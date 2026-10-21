"""Exception hierarchy. Everything agentloop raises derives from :class:`AgentLoopError`.

Tool *execution* failures are deliberately not exceptions at the loop level: the registry
converts them into ``is_error`` tool results so the model can recover. The classes below are
for programmer errors (bad tool signatures, duplicate names), provider failures, and budget
or structured-output exhaustion, which the caller must handle.
"""

from __future__ import annotations


class AgentLoopError(Exception):
    """Base class for all agentloop errors."""


class ToolError(AgentLoopError):
    """A tool could not be defined, found, or safely executed."""


class ToolSignatureError(ToolError):
    """A function cannot be turned into a tool (missing type hints, *args, ...)."""


class DuplicateToolError(ToolError):
    """Two tools with the same name were registered."""


class UnknownToolError(ToolError):
    """The model called a tool that is not in the registry."""


class PathOutsideRootError(ToolError):
    """A sandboxed file tool was asked for a path outside its root."""


class CalculatorError(ToolError):
    """The calculator was given something that is not plain arithmetic."""


class HostNotAllowedError(ToolError):
    """``http_get`` was asked for a host that is not on the allowlist."""


class ResponseTooLargeError(ToolError):
    """An HTTP response exceeded the configured size cap."""


class PythonExecTimeoutError(ToolError):
    """``python_exec`` did not finish within its timeout."""


class ProviderError(AgentLoopError):
    """A model provider failed (network, API error, malformed response)."""


class ScriptExhaustedError(ProviderError):
    """A :class:`FakeProvider` was asked for more responses than it was scripted with."""


class StructuredOutputError(AgentLoopError):
    """The model never produced a valid ``final_answer`` within the retry limit."""
