"""Protección temporal del aprendizaje para las pruebas técnicas del postentrenamiento."""

import pytest

from mars_titan.training.learning_hold import learning_blocked


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
