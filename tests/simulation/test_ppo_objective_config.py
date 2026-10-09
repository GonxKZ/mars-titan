"""Validar objetivos antes de fuentes o tensores, sin ejecutar aprendizaje."""

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = json.loads((ROOT / "configs/simulation/adaptive-ppo-diagnostic.json").read_text())
PENALTY = {
    "schema_version": 1,
    "id": "ppo_kl_penalty_adaptive_v1",
    "target_kl": 0.01,
    "beta_initial": 1.0,
    "beta_min": 1e-6,
    "beta_max": 1e6,
}


@pytest.fixture
def executable():
    value = os.environ.get("MARS_TITAN_PPO_EXECUTABLE")
    if not value:
        pytest.skip("Define MARS_TITAN_PPO_EXECUTABLE con un mars-titan-ppo compilado")
    return Path(value).resolve(strict=True)


def parse_only(executable, tmp_path, objective, *, variant="ppo"):
    config = copy.deepcopy(BASE)
    config["policy_objective"] = objective
    config["agent"]["variant"] = variant
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run(
        [
            str(executable),
            "--config",
            str(path),
            "--output",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--diagnostic",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 1
    assert not (tmp_path / "out").exists()
    return result.stderr


@pytest.mark.parametrize(
    "objective",
    [
        {"schema_version": 1, "id": "ppo_clip_full_kl_v1"},
        {"schema_version": 1, "id": "ppo_clip_kl_epoch_stop_v1", "target_kl": 0.01},
        PENALTY,
    ],
)
def test_valid_objective_reaches_source_guard_without_loading_sources(
    executable, tmp_path, objective
):
    assert "Se necesitan fuentes acotadas" in parse_only(executable, tmp_path, objective)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("id", "ppo"),
        ("target_kl", True),
        ("target_kl", 0),
        ("beta_initial", 0),
        ("beta_min", 2),
        ("beta_max", 0.5),
        ("extra", 1),
    ],
)
def test_incompatible_objective_is_rejected_before_source_guard(executable, tmp_path, field, value):
    objective = {**PENALTY, field: value}
    assert "Se necesitan fuentes acotadas" not in parse_only(executable, tmp_path, objective)


def test_double_dqn_rejects_the_ppo_controller_before_sources(executable, tmp_path):
    assert "Se necesitan fuentes acotadas" not in parse_only(
        executable, tmp_path, PENALTY, variant="double_dqn"
    )
