"""Optimizador del postentrenamiento que registra gradientes y nunca cambia pesos."""

from types import SimpleNamespace

import torch


def install(patch):
    """Sustituir `torch.optim.AdamW` con `patch` y devolver el registro de optimizadores.

    El sustituto no hereda de `torch.optim.Optimizer`, así que el gancho global del bloqueo
    no lo intercepta. Cada paso exige pesos idénticos a los iniciales y solo guarda los
    gradientes. Sirve a un fixture de módulo con `pytest.MonkeyPatch.context()` y al fixture
    `recorder` de cada prueba.
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

    patch.setattr(torch.optim, "AdamW", RecordingOptimizer)
    return record
