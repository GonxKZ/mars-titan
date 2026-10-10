"""Protección del aprendizaje y optimizador sin actualizaciones para el postentrenamiento."""

import pytest
import torch

from mars_titan.training.learning_hold import learning_blocked
from tests.posttraining import recording_optimizer


@pytest.fixture(autouse=True)
def entry_points_admitted(request):
    """Admitir los puntos de entrada sin retirar la guarda global de pasos.

    El postentrenamiento solo ajusta parámetros mediante optimizadores de PyTorch y las pruebas
    de la compleción sustituyen la etapa tabular por dobles. Si la protección local está
    vigente, `tests/conftest.py` ya ha registrado el gancho que omite esos pasos. Aquí solo se
    sustituye la protección leída por los puntos de entrada, para que cada prueba llegue hasta
    ese paso. Las pruebas del bloqueo declaran su propia protección.
    """
    if learning_blocked():
        request.getfixturevalue("learning_doubles")


@pytest.fixture
def recorder(monkeypatch):
    """Sustituir AdamW por un optimizador que registra gradientes y nunca cambia pesos.

    Ver `recording_optimizer.install`. `on_step` permite a una prueba reaccionar después de
    un paso, por ejemplo para pedir una parada.
    """
    deterministic = torch.are_deterministic_algorithms_enabled()
    yield recording_optimizer.install(monkeypatch)
    torch.use_deterministic_algorithms(deterministic)
