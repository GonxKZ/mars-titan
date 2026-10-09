"""Protección local del aprendizaje: impide pasos de optimizador mientras esté vigente."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

HOLD_ENV = "MARS_TITAN_TRAINING_HOLD"
DEFAULT_HOLD = Path.home() / ".local/state/mars-titan/training-hold-2000.json"


class LearningHoldError(Exception):
    """Ajuste rechazado por la protección local.

    No hereda de RuntimeError para que los lanzadores que capturan errores operativos
    (OSError, ValueError o RuntimeError) no lo conviertan en un fallo ordinario del intento.
    """


def hold_path() -> Path:
    return Path(os.environ.get(HOLD_ENV, DEFAULT_HOLD))


def learning_blocked(path: Path | None = None) -> bool:
    """Una protección presente bloquea salvo que declare `training_allowed` verdadero."""
    path = hold_path() if path is None else Path(path)
    if not path.exists():
        return False
    value = json.loads(path.read_text(encoding="utf-8")).get("training_allowed")
    if type(value) is not bool:
        raise ValueError("La protección del aprendizaje no declara training_allowed como booleano")
    return not value


def require_learning_allowed(action: str) -> None:
    """Detener un punto de entrada que ajusta parámetros antes de abrir fuentes o crear salidas."""
    path = hold_path()
    if learning_blocked(path):
        raise LearningHoldError(
            f"Bloqueo de aprendizaje vigente: {action} no se ejecuta mientras {path} "
            "no declare training_allowed verdadero"
        )


def install_optimizer_guard(on_block: Callable[[str], None], path: Path | None = None):
    """Registrar un gancho global previo a cada paso de optimizador de PyTorch.

    Devuelve el manejador del gancho, o None si el aprendizaje está permitido. El gancho se
    evalúa antes de modificar pesos o estados del optimizador. Los optimizadores nativos de
    LibTorch en C++ no pasan por este gancho.
    """
    if not learning_blocked(path):
        return None
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    def guard(optimizer, args, kwargs):
        on_block(
            f"Bloqueo de aprendizaje vigente: paso de {type(optimizer).__name__} omitido "
            "antes de modificar pesos"
        )

    return register_optimizer_step_pre_hook(guard)
