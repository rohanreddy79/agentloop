"""agentloop: a minimal, fully tested LLM agent runtime.

Public surface: :class:`Agent` and :class:`Budget` (the loop), :func:`tool` and
:class:`ToolRegistry` (typed tools), :class:`Memory`, :class:`Tracer`, the provider types,
and the built-in tools under :mod:`agentloop.builtin`.
"""

from agentloop.agent import FINAL_ANSWER, Agent, Budget, BudgetExceeded, RunResult, StepEvent
from agentloop.errors import (
    AgentLoopError,
    DuplicateToolError,
    ProviderError,
    ScriptExhaustedError,
    StructuredOutputError,
    ToolError,
    ToolSignatureError,
    UnknownToolError,
)
from agentloop.memory import Memory
from agentloop.providers import (
    DEFAULT_MODEL,
    AnthropicProvider,
    FakeProvider,
    Message,
    ModelResponse,
    Provider,
    RuleBasedFakeProvider,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
    text_response,
    tool_response,
)
from agentloop.tools import Tool, ToolRegistry, tool
from agentloop.tracing import Tracer, read_trace, render_trace

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODEL",
    "FINAL_ANSWER",
    "Agent",
    "AgentLoopError",
    "AnthropicProvider",
    "Budget",
    "BudgetExceeded",
    "DuplicateToolError",
    "FakeProvider",
    "Memory",
    "Message",
    "ModelResponse",
    "Provider",
    "ProviderError",
    "RuleBasedFakeProvider",
    "RunResult",
    "ScriptExhaustedError",
    "StepEvent",
    "StructuredOutputError",
    "Tool",
    "ToolCall",
    "ToolDefinition",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSignatureError",
    "Tracer",
    "UnknownToolError",
    "Usage",
    "__version__",
    "read_trace",
    "render_trace",
    "text_response",
    "tool",
    "tool_response",
]
