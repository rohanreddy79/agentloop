"""The calculator evaluates arithmetic and rejects everything else before evaluation."""

from __future__ import annotations

import pytest

from agentloop import ToolCall, ToolRegistry
from agentloop.builtin import calculator, safe_eval
from agentloop.errors import CalculatorError


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("(17 * 23 + 5)", "396"),
        ("7 / 2", "3.5"),
        ("8 / 2", "4"),
        ("2 ** 10", "1024"),
        ("-3 + abs(-2)", "-1"),
        ("sqrt(16)", "4"),
        ("10 % 3", "1"),
        ("17 // 5", "3"),
        ("round(2.5555, 2)", "2.56"),
        ("max(1, 2, 3) + min(4, 5)", "7"),
        ("round(pi, 3)", "3.142"),
    ],
)
def test_evaluates_arithmetic(expression: str, expected: str) -> None:
    assert calculator(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "os.path",
        "x + 1",
        "[1, 2][0]",
        "(1).real",
        "lambda: 1",
        "1 if 2 else 3",
        "'a' * 3",
        "1 < 2",
        "print(1)",
        "max(1, key=2)",
        "1 and 2",
        "True + 1",
        "f'{1}'",
        "{1: 2}",
        "2 ** 100000",
        "1 << 3",
    ],
)
def test_rejects_non_arithmetic(expression: str) -> None:
    with pytest.raises(CalculatorError):
        safe_eval(expression)


def test_syntax_error_is_calculator_error() -> None:
    with pytest.raises(CalculatorError, match="not a valid expression"):
        safe_eval("2 +")


def test_division_by_zero_is_calculator_error() -> None:
    with pytest.raises(CalculatorError, match="division by zero"):
        safe_eval("1 / 0")


def test_huge_results_are_rejected() -> None:
    with pytest.raises(CalculatorError, match="too large"):
        safe_eval("(10 ** 1000) ** 100")


def test_math_domain_error_is_calculator_error() -> None:
    with pytest.raises(CalculatorError, match="math error"):
        safe_eval("sqrt(-1)")


async def test_errors_reach_the_model_as_error_results() -> None:
    registry = ToolRegistry([calculator])
    result = await registry.execute(ToolCall("c1", "calculator", {"expression": "1 / 0"}))
    assert result.is_error
    assert result.content == "CalculatorError: division by zero"
