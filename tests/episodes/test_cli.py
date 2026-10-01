"""La generación analítica no exige instalar ni cargar un motor de aprendizaje."""

import json
import subprocess
import sys


def test_analytic_cli_does_not_import_torch(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": {"assets": 2, "sessions": 16, "context": 8},
                "scenarios": {"no_signal": 0},
                "train_seeds": [42],
                "validation_seeds": [43],
                "final_test_opened": False,
            }
        )
    )
    output = tmp_path / "worlds"
    script = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("La generación analítica no debe importar PyTorch")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ["scripts/generate_episode_worlds.py", "--config", sys.argv[1], "--output", sys.argv[2]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config), str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "summary.json").read_text())["status"] == "completed"


def test_invalid_generator_publishes_failed_receipt(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": {"assets": 2, "sessions": 16, "context": 8},
                "scenarios": {"invalid": 0.2},
                "train_seeds": [42],
                "validation_seeds": [43],
                "final_test_opened": False,
            }
        )
    )
    output = tmp_path / "worlds"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_episode_worlds.py",
            "--config",
            str(config),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"
