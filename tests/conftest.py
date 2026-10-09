"""Configuración común de pytest."""

import json
from contextlib import contextmanager

import pytest

from mars_titan.training.learning_hold import HOLD_ENV, LearningHoldError, install_optimizer_guard
from tests.suite_support import python_shebang as _python_shebang


def _skip(reason: str) -> None:
    pytest.skip(reason)


# Mientras la protección local esté vigente, una prueba que intente un paso de optimizador
# de PyTorch se omite antes de modificar pesos. Sin protección, el gancho no se instala.
_GUARD = install_optimizer_guard(_skip)


@contextmanager
def _hold_as_skip():
    # Los puntos de entrada que ajustan modelos se detienen con LearningHoldError antes de
    # abrir fuentes o crear salidas. Bajo la protección, la prueba se omite con su motivo.
    try:
        yield
    except LearningHoldError as error:
        pytest.skip(str(error))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    with _hold_as_skip():
        return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    with _hold_as_skip():
        return (yield)


@pytest.fixture
def learning_hold(tmp_path_factory, monkeypatch):
    """Seleccionar una protección temporal: False bloquea, True permite y None la deja ausente."""

    def select(allowed):
        path = tmp_path_factory.mktemp("learning-hold") / "training-hold.json"
        if allowed is not None:
            path.write_text(json.dumps({"training_allowed": allowed}), encoding="utf-8")
        monkeypatch.setenv(HOLD_ENV, str(path))
        return path

    return select


@pytest.fixture
def learning_doubles(learning_hold):
    """Permitir un lanzador cuyo aprendizaje está sustituido por binarios o ejecutores simulados.

    Solo procede cuando la prueba no puede ejecutar ningún ajuste real. El gancho de PyTorch
    instalado al inicio de la sesión sigue omitiendo cualquier paso de optimizador.
    """
    return learning_hold(True)


@pytest.fixture(scope="session")
def python_shebang(tmp_path_factory):
    """Primera línea de los ejecutables falsos escritos en Python, válida con rutas con espacios."""
    return _python_shebang(tmp_path_factory.mktemp("interpreter"))
