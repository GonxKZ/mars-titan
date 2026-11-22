"""El guion de un origen en Colab da las órdenes con las que el relevo de la DGX lo prepara."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "colab_origin.sh"
SITE = "google-colab-prueba"
SEEN = (
    "MARS_TITAN_SITE MARS_TITAN_DGX_PYTHON MARS_TITAN_COLAB_PYTHON MARS_TITAN_EPISODIC_NATIVE "
    "MARS_TITAN_TRAIN_ARCHIVE MARS_TITAN_SHARED_INDICES MARS_TITAN_DIGEST_CACHE "
    "MARS_TITAN_DECODE_WORKERS PYTHONPATH CUBLAS_WORKSPACE_CONFIG"
)
# Un doble que anota su nombre, su carpeta, sus argumentos y las variables que le llegan.
DOUBLE = """#!/bin/bash
{ printf "%s|" "$(basename "$0")" "$PWD" "$@"
  for name in SEEN; do printf "%s=%s|" "$name" "${!name:-}"; done
  echo; } >> "LOG"
"""


def copy(tmp_path, launcher):
    """Una copia fija de mentira: su lanzador y su intérprete anotan lo que reciben."""
    repo, log = tmp_path / "repo", tmp_path / "calls.log"
    if repo.exists():
        return repo, tmp_path / "python", log
    (repo / "src/mars_titan").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "configs").mkdir()
    (repo / "configs/campaign.json").write_text("{}")
    (repo / "configs/recipe.json").write_text("{}")
    body = DOUBLE.replace("SEEN", SEEN).replace("LOG", str(log))
    for path in (repo / "scripts" / launcher, tmp_path / "python"):
        path.write_text(body)
        path.chmod(0o755)
    return repo, tmp_path / "python", log


def run(step, tmp_path, *extra, site=SITE, launcher="dgx_campaign.sh", origin="2013"):
    repo, python, log = copy(tmp_path, launcher)
    env = dict(os.environ, MARS_TITAN_COLAB_PYTHON=str(python))
    env.pop("MARS_TITAN_SITE", None)
    if site:
        env["MARS_TITAN_SITE"] = site
    command = ["bash", str(SCRIPT), step, "--repo", str(repo), "--estado", str(tmp_path / "state")]
    command += ["--vistas", "/vistas/US", "--declaracion", "configs/campaign.json"]
    command += ["--origen", origin, *extra]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    calls = [line.split("|") for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def common(tmp_path):
    state = str(tmp_path / "state")
    return ["--declaration", "configs/campaign.json", "--state", state, "--views", "/vistas/US"]


def arguments(call):
    """Los argumentos de una llamada, sin su nombre, su carpeta ni las variables anotadas."""
    names = tuple(f"{name}=" for name in SEEN.split())
    return [item for item in call[2:] if item and not item.startswith(names)]


def values(call):
    names = tuple(f"{name}=" for name in SEEN.split())
    return dict(item.split("=", 1) for item in call if item.startswith(names))


def test_the_cohorts_step_prepares_the_state_and_checks_the_machine(tmp_path):
    result, calls = run("cohortes", tmp_path, "--trabajadores", "6")
    assert result.returncode == 0, result.stderr
    assert [call[0] for call in calls] == ["dgx_campaign.sh", "dgx_campaign.sh"]
    assert all(call[1] == str(tmp_path / "repo") for call in calls)
    assert arguments(calls[0]) == ["prepare", *common(tmp_path)]
    expected = ["doctor", "--machine", "mars_titan", "--workers", "6", *common(tmp_path)]
    assert arguments(calls[1]) == expected
    # La copia fija de la v10 se lanza con su guion de la DGX y el entorno de esta máquina.
    seen = values(calls[0])
    assert seen["MARS_TITAN_SITE"] == SITE
    assert seen["MARS_TITAN_DGX_PYTHON"] == str(tmp_path / "python")
    assert "+ cd" in result.stdout and "prepare" in result.stdout


def test_the_inputs_step_gives_the_order_of_the_relay(tmp_path):
    result, calls = run("entradas", tmp_path, "--receta", "configs/recipe.json")
    assert result.returncode == 1 and "cohortes" in result.stderr and calls == []
    state = tmp_path / "state"
    (state / "cohorts/2013").mkdir(parents=True)
    (state / "cohorts/2013/manifest.json").write_text("{}")
    result, calls = run("entradas", tmp_path, "--receta", "configs/recipe.json")
    assert result.returncode == 0, result.stderr
    (call,) = calls
    assert call[0] == "python" and call[1] == str(tmp_path / "repo")
    assert arguments(call) == [
        "scripts/dgx_prepare_inputs.py",
        "origin",
        "--view",
        str(state / "cohorts/2013/manifest.json"),
        "--fold",
        "fold-008",
        "--recipe",
        "configs/recipe.json",
        "--archive",
    ]
    seen = values(call)
    assert seen["MARS_TITAN_TRAIN_ARCHIVE"] == str(state / "packed-inputs/2013")
    assert seen["MARS_TITAN_SHARED_INDICES"] == str(state / "indices")
    assert seen["MARS_TITAN_DIGEST_CACHE"] == str(state / "file-digests.json")
    assert seen["MARS_TITAN_DECODE_WORKERS"] == "4"
    assert seen["PYTHONPATH"] == str(tmp_path / "repo" / "src")
    assert seen["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    # Sin receta no hay copia empaquetada.
    result, _ = run("entradas", tmp_path)
    assert result.returncode == 1 and "--receta" in result.stderr


def test_the_permission_needs_a_note_and_passes_it_whole(tmp_path):
    result, calls = run("permiso", tmp_path)
    assert result.returncode == 1 and "--nota" in result.stderr and calls == []
    note = "Puerta de paridad en Colab, lanzada por el coordinador"
    result, calls = run("permiso", tmp_path, "--nota", note)
    assert result.returncode == 0, result.stderr
    assert arguments(calls[0]) == ["authorize", "--note", note, *common(tmp_path)]


def test_the_queue_repeats_a_flag_per_seed_and_tier(tmp_path):
    options = ["--semillas", "42 43 44", "--niveles", "core controls", "--trabajadores", "5"]
    options += ["--extra", "--no-preflight", "--nativo", "/n.so"]
    result, calls = run("lanza", tmp_path, *options)
    assert result.returncode == 0, result.stderr
    expected = ["run", "--machine", "mars_titan", "--workers", "5"]
    expected += ["--seed", "42", "--seed", "43", "--seed", "44"]
    expected += ["--tier", "core", "--tier", "controls"]
    expected += ["--origin", "2013", "--no-preflight", *common(tmp_path)]
    assert arguments(calls[0]) == expected
    assert values(calls[0])["MARS_TITAN_EPISODIC_NATIVE"] == "/n.so"
    result, calls = run("estado", tmp_path)
    assert arguments(calls[-1]) == ["status", "--machine", "mars_titan", *common(tmp_path)]


def test_a_rehearsal_prints_the_orders_and_runs_nothing(tmp_path):
    result, calls = run("lanza", tmp_path, "--ensayo")
    assert result.returncode == 0, result.stderr
    assert calls == []
    order = "./scripts/dgx_campaign.sh run --machine mars_titan --workers 4 --seed 42"
    assert order in result.stdout and f"MARS_TITAN_SITE={SITE}" in result.stdout


def test_without_the_site_label_nothing_runs(tmp_path):
    result, calls = run("lanza", tmp_path, site=None)
    assert result.returncode != 0 and "MARS_TITAN_SITE" in result.stderr and calls == []


def test_a_copy_with_the_colab_launcher_uses_it(tmp_path):
    result, calls = run("estado", tmp_path, launcher="colab_campaign.sh")
    assert result.returncode == 0, result.stderr
    assert calls[0][0] == "colab_campaign.sh"
    seen = values(calls[0])
    assert seen["MARS_TITAN_COLAB_PYTHON"] == str(tmp_path / "python")
    assert seen["MARS_TITAN_DGX_PYTHON"] == ""


@pytest.mark.parametrize(
    "options, message",
    [
        (["--trabajadores", "9"], "--trabajadores"),
        (["--trabajadores", "0"], "--trabajadores"),
        (["--raro", "1"], "desconocida"),
    ],
)
def test_wrong_options_are_refused(tmp_path, options, message):
    result, calls = run("estado", tmp_path, *options)
    assert result.returncode != 0 and message in result.stderr and calls == []


@pytest.mark.parametrize("origin", ["2004", "13", "20133"])
def test_a_wrong_origin_is_refused(tmp_path, origin):
    result, calls = run("estado", tmp_path, origin=origin)
    assert result.returncode == 1 and "--origen" in result.stderr and calls == []


def test_an_unknown_step_and_a_missing_declaration_are_refused(tmp_path):
    result, calls = run("entrena", tmp_path)
    assert result.returncode == 2 and "desconocido" in result.stderr and calls == []
    (tmp_path / "repo/configs/campaign.json").unlink()
    result, calls = run("estado", tmp_path)
    assert result.returncode == 1 and "declaración" in result.stderr and calls == []
