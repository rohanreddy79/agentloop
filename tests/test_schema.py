"""Schema generation from type hints and docstrings."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from agentloop import ToolSignatureError, tool
from agentloop.schema import format_validation_error, parse_docstring


class Address(BaseModel):
    city: str
    zip_code: str | None = None


def _no_keys(node: object, *keys: str) -> bool:
    if isinstance(node, dict):
        return all(k not in keys for k in node) and all(_no_keys(v, *keys) for v in node.values())
    if isinstance(node, list):
        return all(_no_keys(v, *keys) for v in node)
    return True


def test_str_and_int_with_default_and_docstring_descriptions() -> None:
    @tool
    def greet(name: str, times: int = 1) -> str:
        """Greet someone.

        Args:
            name: Who to greet.
            times: How many times.
        """
        return name * times

    schema = greet.parameters
    assert schema["type"] == "object"
    assert schema["properties"]["name"] == {"type": "string", "description": "Who to greet."}
    assert schema["properties"]["times"] == {
        "type": "integer",
        "default": 1,
        "description": "How many times.",
    }
    assert schema["required"] == ["name"]
    assert schema["additionalProperties"] is False
    assert greet.name == "greet"
    assert greet.description == "Greet someone."
    assert greet("ab", 2) == "abab"  # still a plain function


def test_float_and_bool() -> None:
    @tool
    def scale(factor: float, invert: bool = False) -> float:
        """Scale."""
        return -factor if invert else factor

    props = scale.parameters["properties"]
    assert props["factor"] == {"type": "number"}
    assert props["invert"] == {"type": "boolean", "default": False}


def test_list_of_int() -> None:
    @tool
    def total(values: list[int]) -> int:
        """Sum values."""
        return sum(values)

    assert total.parameters["properties"]["values"] == {
        "type": "array",
        "items": {"type": "integer"},
    }


def test_optional_str() -> None:
    @tool
    def find(query: str, lang: str | None = None) -> str:
        """Find."""
        return query

    lang = find.parameters["properties"]["lang"]
    assert lang["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert lang["default"] is None
    assert find.parameters["required"] == ["query"]


def test_literal_becomes_enum() -> None:
    @tool
    def convert(value: float, unit: Literal["c", "f"]) -> float:
        """Convert."""
        return value

    assert convert.parameters["properties"]["unit"] == {"type": "string", "enum": ["c", "f"]}


def test_pydantic_model_parameter_is_inlined() -> None:
    @tool
    def ship(address: Address, items: list[str]) -> str:
        """Ship."""
        return address.city

    address = ship.parameters["properties"]["address"]
    assert address["type"] == "object"
    assert address["properties"]["city"] == {"type": "string"}
    assert address["properties"]["zip_code"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert address["required"] == ["city"]
    assert _no_keys(ship.parameters, "$defs", "$ref", "title")


def test_no_parameters() -> None:
    @tool
    def now() -> str:
        """Current time."""
        return "noon"

    assert now.parameters == {"type": "object", "properties": {}, "additionalProperties": False}


def test_async_flag_and_explicit_name_description_timeout() -> None:
    @tool(name="sleepy", description="Sleeps.", timeout=2.5)
    async def _impl(seconds: float) -> None:
        return None

    assert _impl.name == "sleepy"
    assert _impl.description == "Sleeps."
    assert _impl.is_async is True
    assert _impl.timeout == 2.5


def test_missing_type_hint_rejected() -> None:
    with pytest.raises(ToolSignatureError, match="needs a type hint"):

        @tool
        def bad(x):  # type: ignore[no-untyped-def]
            """Bad."""
            return x


def test_varargs_rejected() -> None:
    with pytest.raises(ToolSignatureError, match=r"\*args"):

        @tool
        def bad(*args: int) -> int:
            """Bad."""
            return 0


def test_missing_description_rejected() -> None:
    with pytest.raises(ToolSignatureError, match="description"):

        @tool
        def bad(x: int) -> int:
            return x


def test_parse_docstring_multiline_param_and_other_sections() -> None:
    summary, params = parse_docstring(
        """Do a thing
        over two lines.

        Args:
            alpha (int): First value,
                continued here.
            beta: Second.

        Returns:
            Nothing.
        """
    )
    assert summary == "Do a thing over two lines."
    assert params == {"alpha": "First value, continued here.", "beta": "Second."}


def test_format_validation_error_is_compact() -> None:
    class M(BaseModel):
        age: int
        name: str

    with pytest.raises(ValidationError) as info:
        M.model_validate({"age": "old"})
    text = format_validation_error(info.value)
    assert "age: Input should be a valid integer" in text
    assert "name: Field required" in text
    assert "http" not in text
