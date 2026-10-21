"""Derive a JSON schema for a tool's parameters from its Python signature.

Rather than hand-mapping Python types to JSON schema, a synthetic pydantic model is built from
the signature (``create_model``) and pydantic emits the schema. That gives correct handling of
``Optional``, ``Literal``, ``list[T]``, nested ``BaseModel`` parameters and defaults for free,
and the same model is reused at call time to validate the arguments the model sent. The raw
pydantic schema is then cleaned: ``$ref``s are inlined (one self-contained object is easier for
a model to follow than ``$defs``) and ``title`` noise is removed.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from agentloop.errors import ToolSignatureError

_ARGS_HEADERS = ("Args:", "Arguments:", "Parameters:")
_PARAM_LINE = re.compile(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {param: description}).

    The summary is everything before the first section header (``Args:``, ``Returns:`` ...)
    with whitespace collapsed. Parameter lines look like ``name: text`` or ``name (type):
    text``; continuation lines are indented further.
    """
    if not doc:
        return "", {}
    summary_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = seen_section = False
    current: str | None = None
    for line in inspect.cleandoc(doc).splitlines():
        stripped = line.strip()
        if _is_section_header(stripped):
            in_args = stripped in _ARGS_HEADERS
            seen_section = True
            current = None
            continue
        if in_args:
            match = _PARAM_LINE.match(line)
            if match and not line.startswith(" " * 8):
                current = match.group(1)
                params[current] = match.group(2).strip()
            elif current and stripped:
                params[current] = f"{params[current]} {stripped}".strip()
        elif not seen_section:
            summary_lines.append(stripped)
    summary = " ".join(part for part in summary_lines if part).strip()
    return summary, params


def _is_section_header(stripped: str) -> bool:
    return len(stripped) > 1 and stripped.endswith(":") and " " not in stripped


def _type_hints(fn: Callable[..., Any], scope: dict[str, Any] | None) -> dict[str, Any]:
    """``get_type_hints`` that also sees the defining scope and the function's closure.

    With ``from __future__ import annotations`` a tool defined inside another function may
    annotate a parameter with a class that only exists in that enclosing scope; ``scope`` is
    that scope's locals as captured by the decorator.
    """
    localns: dict[str, Any] = dict(scope or {})
    closure = getattr(fn, "__closure__", None) or ()
    for var_name, cell in zip(fn.__code__.co_freevars, closure, strict=False):
        try:
            localns[var_name] = cell.cell_contents
        except ValueError:  # empty cell
            continue
    return get_type_hints(fn, localns=localns or None, include_extras=True)


def build_params_model(
    fn: Callable[..., Any], name: str, scope: dict[str, Any] | None = None
) -> type[BaseModel]:
    """Build a pydantic model whose fields mirror ``fn``'s parameters."""
    try:
        hints = _type_hints(fn, scope)
    except Exception as exc:  # unresolvable forward reference etc.
        raise ToolSignatureError(f"tool {name!r}: cannot resolve type hints ({exc})") from exc
    _, descriptions = parse_docstring(fn.__doc__)
    fields: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ToolSignatureError(f"tool {name!r}: *args/**kwargs are not supported")
        if param.name not in hints:
            raise ToolSignatureError(f"tool {name!r}: parameter {param.name!r} needs a type hint")
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (
            hints[param.name],
            Field(default, description=descriptions.get(param.name)),
        )
    return create_model(f"{name}_params", __config__=ConfigDict(extra="forbid"), **fields)


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """A self-contained JSON-schema object for ``model`` (refs inlined, titles stripped)."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})
    schema = _clean(raw, defs)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def _clean(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"].rsplit("/", 1)[-1]
            merged = {**defs[ref], **{k: v for k, v in node.items() if k != "$ref"}}
            return _clean(merged, defs)
        return {k: _clean(v, defs) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_clean(item, defs) for item in node]
    return node


def format_validation_error(exc: ValidationError) -> str:
    """One line per problem, e.g. ``age: Input should be a valid integer`` (no doc URLs)."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"{loc}: {err['msg']}")
    return "; ".join(lines)
