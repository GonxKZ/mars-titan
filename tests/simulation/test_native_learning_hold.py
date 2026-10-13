"""El entrenamiento nativo de PPO lee la misma protección del aprendizaje que Python.

Las fuentes no existen. Con la protección permitida el proceso falla al leerlas, antes de
construir el entorno o ejecutar un paso de optimizador. Ese caso abre la protección, así que
figura en el inventario de #546 y se omite mientras rija el bloqueo.
"""

import os
import subprocess
from pathlib import Path

import pytest

from mars_titan.training.learning_hold import HOLD_ENV
from tests.learning_hold_lowering import lowers_learning_hold

pytestmark = pytest.mark.native("MARS_TITAN_PPO_EXECUTABLE")

ROOT = Path(__file__).resolve().parents[2]
THIS = "tests/simulation/test_native_learning_hold.py"


@pytest.fixture
def binary():
    value = os.environ.get("MARS_TITAN_PPO_EXECUTABLE")
    if not value:
        pytest.skip(
            "Falta declarar MARS_TITAN_PPO_EXECUTABLE con un mars-titan-ppo compilado por "
            "`make native-build` (preset native-ppo-release). Esta prueba no usa la ruta "
            "por defecto"
        )
    return Path(value).resolve(strict=True)


def run(binary, hold, arguments):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    environment[HOLD_ENV] = str(hold)
    return subprocess.run(
        [str(binary), *arguments, "--device", "cpu", "--diagnostic"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def training(binary, hold, tmp_path):
    config = ROOT / "configs/simulation/native-ppo-diagnostic.json"
    arguments = ["--config", str(config), "--output", str(tmp_path / "out")]
    arguments += ["--train-tape", str(tmp_path / "train"), "--validation-tape", str(tmp_path / "v")]
    result = run(binary, hold, arguments)
    assert result.returncode == 1
    assert not (tmp_path / "out").exists()
    return "Bloqueo de aprendizaje vigente: el entrenamiento PPO nativo" in result.stderr


def test_training_stops_on_a_blocking_hold_before_opening_sources(binary, learning_hold, tmp_path):
    assert training(binary, learning_hold(False), tmp_path)


@lowers_learning_hold(f"{THIS}::test_training_passes_an_allowed_hold_and_fails_on_missing_sources")
def test_training_passes_an_allowed_hold_and_fails_on_missing_sources(
    binary, learning_hold, tmp_path
):
    assert not training(binary, learning_hold(True), tmp_path)


def test_frozen_audit_is_not_blocked(binary, learning_hold, tmp_path):
    config = ROOT / "configs/simulation/adaptive-ppo-diagnostic.json"
    arguments = ["--config", str(config), "--output", str(tmp_path / "out")]
    arguments += ["--audit-run", str(tmp_path / "run"), "--audit-tape", str(tmp_path / "audit")]
    result = run(binary, learning_hold(False), arguments)
    assert result.returncode == 1
    assert "Bloqueo de aprendizaje" not in result.stderr
    assert not (tmp_path / "out").exists()
