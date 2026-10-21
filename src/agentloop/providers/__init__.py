"""Model providers: the shared message types, the Anthropic provider, and offline fakes."""

from agentloop.providers.anthropic import DEFAULT_MODEL, AnthropicProvider
from agentloop.providers.base import (
    Message,
    ModelResponse,
    Provider,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)
from agentloop.providers.fake import (
    FakeProvider,
    RuleBasedFakeProvider,
    text_response,
    tool_response,
)

__all__ = [
    "DEFAULT_MODEL",
    "AnthropicProvider",
    "FakeProvider",
    "Message",
    "ModelResponse",
    "Provider",
    "RuleBasedFakeProvider",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "Usage",
    "text_response",
    "tool_response",
]
