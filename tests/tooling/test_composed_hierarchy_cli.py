"""Contratos del CLI de coste, comprobados sin importar bibliotecas de aprendizaje."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "benchmarks/composed_hierarchy_cuda.py"
LAUNCHER = """
import importlib.abc
import runpy
import sys
class RejectML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy'):
            raise RuntimeError('CLI_IMPORTED_ML: ' + fullname)
sys.meta_path.insert(0, RejectML())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def invoke(script, arguments, cwd):
    assert SCRIPT.is_file(), "Falta benchmarks/composed_hierarchy_cuda.py"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", LAUNCHER, str(script), *map(str, arguments)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert "CLI_IMPORTED_ML" not in result.stderr, result.stderr
    return result


@pytest.mark.parametrize("with_output", [False, True])
def test_default_cli_prints_the_fixed_protocol_without_a_repository_or_ml(tmp_path, with_output):
    output = tmp_path / "unused-output.json"
    result = invoke(SCRIPT, ["--output", output] if with_output else [], tmp_path)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["kind"] == "joint_hierarchy_resident_cost_v1"
    assert plan["controls"] == ["flat", "pre_only", "post_only", "both"]
    assert plan["rows"] == [4, 64, 256]
    assert plan["orders"] == [plan["controls"], list(reversed(plan["controls"]))]
    assert (plan["warmup"], plan["samples"]) == (1, 3)
    assert (plan["records"], plan["cohort_invocations"]) == (24, 96)
    assert plan["physical_prepare_invocations"] == 192
    assert (plan["predictor_max_batch"], plan["physical_block_rows"]) == (256, 64)
    assert (plan["internal_seconds"], plan["external_seconds"]) == (480, 510)
    assert plan["device"] == "cuda:0" and plan["dtype"] == "torch.float64"
    assert plan["optimizer_steps"] == 0
    assert plan["predictive_comparison"] is False
    assert plan["paired_parameters"] is False
    assert plan["tail_precision_claim"] is False
    assert not output.exists()
    assert "/home/" not in SCRIPT.read_text(encoding="utf-8")


@pytest.mark.parametrize("missing", ["--repository", "--output"])
def test_run_requires_both_explicit_paths_before_importing_ml(tmp_path, missing):
    output = tmp_path / "new-output.json"
    arguments = ["--run"]
    arguments += [] if missing == "--repository" else ["--repository", REPOSITORY]
    arguments += [] if missing == "--output" else ["--output", output]
    result = invoke(SCRIPT, arguments, tmp_path)
    assert result.returncode == 2, result.stderr
    assert missing in result.stderr
    assert "requiere" in result.stderr
    assert "unrecognized arguments" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_run_rejects_the_canonical_script_and_links_to_it_before_ml(tmp_path, symlink):
    output = tmp_path / "new-output.json"
    script = SCRIPT
    if symlink:
        script = tmp_path / "linked-probe.py"
        script.symlink_to(SCRIPT)
    result = invoke(script, ["--run", "--repository", REPOSITORY, "--output", output], tmp_path)
    assert result.returncode != 0
    assert "copia privada" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("fault", ["existing", "inside_repository", "missing_parent"])
def test_private_copy_rejects_invalid_output_before_ml(tmp_path, fault):
    assert SCRIPT.is_file(), "Falta benchmarks/composed_hierarchy_cuda.py"
    script = tmp_path / "copied-probe.py"
    shutil.copyfile(SCRIPT, script)
    output = tmp_path / "output.json"
    if fault == "existing":
        output.write_text("contenido previo", encoding="utf-8")
    elif fault == "inside_repository":
        output = REPOSITORY / "benchmarks" / f"unused-{tmp_path.name}.json"
    else:
        output = tmp_path / "missing-parent" / "output.json"
    result = invoke(script, ["--run", "--repository", REPOSITORY, "--output", output], tmp_path)
    assert result.returncode != 0
    assert "salida" in result.stderr
    if fault == "existing":
        assert output.read_text(encoding="utf-8") == "contenido previo"
    else:
        assert not output.exists()
    assert not (tmp_path / "missing-parent").exists()


def test_private_copy_rejects_a_missing_repository_before_ml(tmp_path):
    assert SCRIPT.is_file(), "Falta benchmarks/composed_hierarchy_cuda.py"
    script = tmp_path / "copied-probe.py"
    shutil.copyfile(SCRIPT, script)
    output = tmp_path / "new-output.json"
    repository = tmp_path / "missing-repository"
    result = invoke(script, ["--run", "--repository", repository, "--output", output], tmp_path)
    assert result.returncode != 0
    assert not repository.exists() and not output.exists()


def test_private_copy_must_match_the_public_source_before_ml(tmp_path):
    assert SCRIPT.is_file(), "Falta benchmarks/composed_hierarchy_cuda.py"
    script = tmp_path / "modified-copy.py"
    script.write_text(
        SCRIPT.read_text(encoding="utf-8") + "\n# Copia modificada.\n", encoding="utf-8"
    )
    output = tmp_path / "new-output.json"
    result = invoke(script, ["--run", "--repository", REPOSITORY, "--output", output], tmp_path)
    assert result.returncode != 0
    assert "fuente pública" in result.stderr
    assert not output.exists()
