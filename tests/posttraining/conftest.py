"""Protección temporal del aprendizaje para las pruebas técnicas del postentrenamiento."""

import json

import pytest

from mars_titan.training.learning_hold import HOLD_ENV, learning_blocked


@pytest.fixture(autouse=True)
def entry_points_admitted(tmp_path_factory, monkeypatch):
    """Admitir los puntos de entrada sin retirar la guarda global de pasos.

    Si la protección local está vigente, `tests/conftest.py` ya ha registrado el gancho
    que omite cualquier paso de un optimizador de PyTorch. Aquí solo se sustituye la
    protección leída por los puntos de entrada, para que cada prueba llegue hasta ese
    paso como antes. Las pruebas del bloqueo declaran su propia protección temporal.
    """
    if not learning_blocked():
        yield
        return
    path = tmp_path_factory.mktemp("hold") / "training-hold.json"
    path.write_text(json.dumps({"training_allowed": True}))
    monkeypatch.setenv(HOLD_ENV, str(path))
    yield
