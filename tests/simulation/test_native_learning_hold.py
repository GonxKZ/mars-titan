"""El entrenamiento nativo de PPO lee la misma protección del aprendizaje que Python.

Las fuentes no existen. Con la protección permitida el proceso falla al leerlas, antes de
construir el entorno o ejecutar un paso de optimizador.
"""

import os
import subprocess
from pathlib import Path

import pytest

from mars_titan.training.learning_hold import HOLD_ENV

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def binary():
    value = os.environ.get("MARS_TITAN_PPO_EXECUTABLE")
    if not value:
        pytest.skip("Define MARS_TITAN_PPO_EXECUTABLE con un mars-titan-ppo compilado")
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


@pytest.mark.parametrize("allowed", [False, True])
def test_training_reads_the_hold_before_opening_sources(binary, learning_hold, tmp_path, allowed):
    config = ROOT / "configs/simulation/native-ppo-diagnostic.json"
    arguments = ["--config", str(config), "--output", str(tmp_path / "out")]
    arguments += ["--train-tape", str(tmp_path / "train"), "--validation-tape", str(tmp_path / "v")]
    result = run(binary, learning_hold(allowed), arguments)
    assert result.returncode == 1
    blocked = "Bloqueo de aprendizaje vigente: el entrenamiento PPO nativo" in result.stderr
    assert blocked is not allowed
    assert not (tmp_path / "out").exists()


def test_frozen_audit_is_not_blocked(binary, learning_hold, tmp_path):
    config = ROOT / "configs/simulation/adaptive-ppo-diagnostic.json"
    arguments = ["--config", str(config), "--output", str(tmp_path / "out")]
    arguments += ["--audit-run", str(tmp_path / "run"), "--audit-tape", str(tmp_path / "audit")]
    result = run(binary, learning_hold(False), arguments)
    assert result.returncode == 1
    assert "Bloqueo de aprendizaje" not in result.stderr
    assert not (tmp_path / "out").exists()
