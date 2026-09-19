"""Gramática acotada del catálogo macro, sin ejecución de código del catálogo."""

import ast
import math
from dataclasses import dataclass


class MissingCalculation(ValueError):
    """Ausencia numérica explicada, diferente de una fórmula inválida."""


def _offset(node: ast.AST) -> int:
    if isinstance(node, ast.Name) and node.id == "p":
        return 0
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and isinstance(node.left, ast.Name)
        and node.left.id == "p"
        and isinstance(node.right, ast.Constant)
        and type(node.right.value) is int
        and 1 <= node.right.value <= 52
    ):
        return node.right.value
    raise ValueError("Formula contains an invalid or unbounded period offset")


def _reference(node: ast.Subscript) -> tuple[str, tuple[int, ...]]:
    if not isinstance(node.value, ast.Name):
        raise ValueError("Formula reference must name a catalog indicator")
    index = node.slice
    if isinstance(index, ast.Slice):
        if index.step is not None or index.lower is None or index.upper is None:
            raise ValueError("Formula contains an invalid period window")
        first, last = _offset(index.lower), _offset(index.upper)
        if first < last:
            raise ValueError("Formula window is reversed")
        return node.value.id, tuple(range(last, first + 1))
    return node.value.id, (_offset(index),)


@dataclass(frozen=True)
class Formula:
    """AST validado y referencias necesarias para conservar su procedencia."""

    tree: ast.AST
    references: tuple[tuple[str, int], ...]
    growth: bool

    @classmethod
    def parse(cls, expression: str, dependencies: set[str], unit: str) -> "Formula":
        if not expression or len(expression) > 1024:
            raise ValueError("Formula is empty or too long")
        try:
            tree = ast.parse(expression.replace("^", "**"), mode="eval").body
        except (SyntaxError, RecursionError) as error:
            raise ValueError("Formula syntax is invalid") from error
        if sum(1 for _ in ast.walk(tree)) > 128:
            raise ValueError("Formula is too complex")
        references = set()

        def validate(node):
            if isinstance(node, ast.Constant):
                if type(node.value) not in {int, float} or not 0 <= node.value <= 100:
                    raise ValueError("Formula constant is outside the allowed range")
            elif isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Slice):
                    raise ValueError("Formula windows require mean")
                name, offsets = _reference(node)
                references.update((name, offset) for offset in offsets)
            elif isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
            ):
                if isinstance(node.op, ast.Pow) and (
                    not isinstance(node.right, ast.Constant)
                    or type(node.right.value) is not int
                    or not 1 <= node.right.value <= 4
                ):
                    raise ValueError("Formula power is outside the allowed range")
                validate(node.left)
                validate(node.right)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "mean"
                and len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Subscript)
                and isinstance(node.args[0].slice, ast.Slice)
            ):
                name, offsets = _reference(node.args[0])
                references.update((name, offset) for offset in offsets)
            else:
                raise ValueError("Formula uses a forbidden expression")

        validate(tree)
        if {name for name, _ in references} != dependencies or not references:
            raise ValueError("Formula references do not match input_ids")
        return cls(
            tree, tuple(sorted(references)), unit in {"percent_change", "percent_annualized"}
        )

    def calculate(self, values: dict[tuple[str, int], float]) -> float:
        def visit(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Subscript):
                name, offsets = _reference(node)
                return values[name, offsets[0]]
            if isinstance(node, ast.Call):
                name, offsets = _reference(node.args[0])
                return math.fsum(values[name, offset] / len(offsets) for offset in offsets)
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if self.growth and (left <= 0 or right <= 0):
                    raise MissingCalculation("nonpositive_growth_input")
                if right == 0:
                    raise MissingCalculation("zero_denominator")
                return left / right
            return left**right

        try:
            result = float(visit(self.tree))
        except OverflowError as error:
            raise MissingCalculation("nonfinite_calculation") from error
        if not math.isfinite(result):
            raise MissingCalculation("nonfinite_calculation")
        return result
