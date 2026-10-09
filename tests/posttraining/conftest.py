"""Protección del aprendizaje y optimizador sin actualizaciones para el postentrenamiento."""

from types import SimpleNamespace

import pytest
import torch

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


@pytest.fixture
def recorder(monkeypatch):
    """Sustituir AdamW por un optimizador que registra gradientes y nunca cambia pesos.

    No hereda de `torch.optim.Optimizer`, así que el gancho global del bloqueo no lo
    intercepta. Cada paso exige pesos idénticos a los iniciales. `on_step` permite a una
    prueba reaccionar después de un paso, por ejemplo para pedir una parada.
    """
    record = SimpleNamespace(optimizers=[], on_step=None)

    class RecordingOptimizer:
        def __init__(self, parameters, lr, weight_decay):
            self.parameters = list(parameters)
            self.initial = [value.detach().clone() for value in self.parameters]
            self.lr, self.weight_decay, self.calls = lr, weight_decay, []
            record.optimizers.append(self)

        def zero_grad(self, set_to_none=True):
            assert set_to_none is True
            for value in self.parameters:
                value.grad = None

        @torch.no_grad()
        def step(self):
            current = self.parameters
            assert all(torch.equal(a, b) for a, b in zip(current, self.initial, strict=True))
            self.calls.append([None if p.grad is None else p.grad.clone() for p in current])
            if record.on_step is not None:
                record.on_step(self)

        def state_dict(self):
            return dict(kind="recording_without_updates", calls=len(self.calls))

        def load_state_dict(self, state):
            assert state["kind"] == "recording_without_updates"

    deterministic = torch.are_deterministic_algorithms_enabled()
    monkeypatch.setattr(torch.optim, "AdamW", RecordingOptimizer)
    yield record
    torch.use_deterministic_algorithms(deterministic)
