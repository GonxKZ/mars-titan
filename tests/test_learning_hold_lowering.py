"""El inventario de pruebas que rebajan la protección nombra pruebas que existen (#546)."""

import ast
from pathlib import Path

import pytest

from mars_titan.training.learning_hold import HOLD_ENV, learning_blocked
from tests.learning_hold_lowering import INVENTORY, lowers_learning_hold

ROOT = Path(__file__).resolve().parents[1]


def test_each_inventoried_test_exists_and_is_marked_with_its_own_key():
    for key in INVENTORY:
        path, _, name = key.partition("::")
        source = (ROOT / path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        if name:
            functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
            assert name in functions
        # La clave aparece entera en la marca de su archivo, que solo admite claves inventariadas.
        assert "lowers_learning_hold(" in source
        assert all(part in source for part in key.split("::"))


def test_the_mark_follows_the_session_hold_and_skips_with_a_blocking_one(monkeypatch, tmp_path):
    key = next(iter(INVENTORY))
    # La marca sigue a la protección de la sesión, sin rebajarla en ningún momento.
    mark = lowers_learning_hold(key).mark
    assert mark.args == (learning_blocked(),) and "#546" in mark.kwargs["reason"]
    blocking = tmp_path / "hold.json"
    blocking.write_text('{"training_allowed": false}', encoding="utf-8")
    monkeypatch.setenv(HOLD_ENV, str(blocking))
    assert lowers_learning_hold(key).mark.args == (True,)


def test_the_mark_refuses_a_test_outside_the_inventory():
    with pytest.raises(KeyError, match="no figura en el inventario"):
        lowers_learning_hold("tests/otra.py")
