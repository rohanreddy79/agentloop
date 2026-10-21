"""AST-safe arithmetic.

The expression is parsed with :mod:`ast` and walked explicitly: only numeric constants, the
arithmetic operators, unary signs, parentheses, two constants and a short list of math
functions are accepted. Anything else (names, attributes, subscripts, calls to other
functions, comprehensions, ...) raises :class:`CalculatorError` before anything is evaluated,
so ``__import__("os")`` and friends never run.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from agentloop.errors import CalculatorError
from agentloop.tools import tool

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 10_000
MAX_RESULT_BITS = 13_000  # ~4000 decimal digits, under Python's int-to-str limit


def safe_eval(expression: str) -> int | float:
    """Evaluate ``expression`` or raise :class:`CalculatorError`."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"not a valid expression: {exc.msg}") from None
    return _eval(tree.body)


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return node.value
        raise CalculatorError(f"only numbers are allowed, not {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BINARY.get(type(node.op))
        if op is None:
            raise CalculatorError(f"unsupported operator {type(node.op).__name__}")
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"exponent {right} exceeds the limit of {MAX_EXPONENT}")
        return _check(_apply(op, left, right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY.get(type(node.op))
        if op is None:
            raise CalculatorError(f"unsupported operator {type(node.op).__name__}")
        return op(_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise CalculatorError(f"unknown name {node.id!r}; only pi and e are defined")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords:
            names = ", ".join(_FUNCTIONS)
            raise CalculatorError(f"only these functions can be called: {names}")
        return _check(_apply(_FUNCTIONS[node.func.id], *(_eval(arg) for arg in node.args)))
    raise CalculatorError(f"unsupported syntax: {type(node).__name__}")


def _apply(fn: Callable[..., Any], *args: Any) -> int | float:
    try:
        return fn(*args)
    except ZeroDivisionError:
        raise CalculatorError("division by zero") from None
    except (OverflowError, ValueError) as exc:
        raise CalculatorError(f"math error: {exc}") from None


def _check(value: int | float) -> int | float:
    if isinstance(value, int) and value.bit_length() > MAX_RESULT_BITS:
        raise CalculatorError("result is too large")
    return value


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result as text.

    Supports + - * / // % ** and parentheses, the constants pi and e, and the functions abs,
    round, sqrt, floor, ceil, min, max, log, log10, exp, sin, cos and tan. Nothing else.

    Args:
        expression: The expression to evaluate, for example "(17 * 23 + 5) / 2".
    """
    value = safe_eval(expression)
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return str(value)
