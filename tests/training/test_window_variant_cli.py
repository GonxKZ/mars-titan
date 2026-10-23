"""CLI de ventanas experimentales, con configuración real y ejecución sustituida."""

import importlib
import json
import os
import runpy
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_window_variant.py"


def cli():
    assert SCRIPT.is_file(), "Falta el CLI experimental de ventanas"
    return runpy.run_path(str(SCRIPT), run_name="window_variant_cli_test")["main"]


def recipe_document(tmp_path, **changes):
    document = dict(
        train_start="2015-01-01",
        train_end="2015-04-01",
        validation_start="2015-04-01",
        validation_end="2015-05-01",
        seed=77,
        hidden_size=32,
        batch_size=32,
        max_cuda_bytes=1024**3,
    )
    document.update(changes)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(document))
    return path


def backend():
    return importlib.import_module("mars_titan.training.window_variant_run")


def test_prepare_forwards_explicit_paths_and_a_real_recipe_without_training(
    tmp_path, monkeypatch, capsys
):
    main, module = cli(), backend()
    recipe = recipe_document(tmp_path)
    parent, catalog, output = (tmp_path / name for name in ("parent.json", "catalog.json", "view"))
    calls = []
    manifest = output / "views/fold-000/manifest.json"

    def prepare(*args):
        calls.append(args)
        return manifest

    def forbidden(*args, **kwargs):
        pytest.fail("Preparar la vista no puede ajustar un modelo")

    monkeypatch.setattr(module, "prepare_window_variant_view", prepare)
    monkeypatch.setattr(module, "run_window_variant", forbidden)
    arguments = [
        "prepare",
        "--parent",
        str(parent),
        "--catalog",
        str(catalog),
        "--recipe",
        str(recipe),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    assert len(calls) == 1
    supplied_parent, supplied_catalog, supplied_output, supplied_recipe = calls[0]
    assert (supplied_parent, supplied_catalog, supplied_output) == (parent, catalog, output)
    assert type(supplied_recipe) is module.WindowVariantRecipe
    assert (supplied_recipe.seed, supplied_recipe.hidden_size, supplied_recipe.batch_size) == (
        77,
        32,
        32,
    )
    assert supplied_recipe.validation_end == "2015-05-01"
    assert json.loads(capsys.readouterr().out) == dict(
        status="prepared", manifest=str(manifest), output=str(output)
    )
    assert not output.exists()


@pytest.mark.parametrize("status,code", [("completed", 0), ("paused", 3), ("partial", 3)])
def test_run_forwards_resume_budget_and_stop_after_the_learning_guard(
    tmp_path, monkeypatch, capsys, status, code
):
    main, module = cli(), backend()
    hold = importlib.import_module("mars_titan.training.learning_hold")
    resources = importlib.import_module("mars_titan.training.experiment_resources")
    checkpoints = importlib.import_module("mars_titan.training.checkpoints")
    recipe = recipe_document(tmp_path)
    manifest, output = tmp_path / "manifest.json", tmp_path / "run"
    events, stop = [], object()

    @contextmanager
    def lease(**options):
        assert options == dict(max_vram=1024**3)
        events.append("lease")
        yield

    @contextmanager
    def stopping():
        yield stop

    def run(supplied_manifest, supplied_output, supplied_recipe, *, resume, stop):
        assert (supplied_manifest, supplied_output) == (manifest, output)
        assert type(supplied_recipe) is module.WindowVariantRecipe
        assert supplied_recipe.seed == 77
        assert resume is True
        events.append(("run", stop))
        return dict(status=status, global_step=4)

    monkeypatch.setattr(hold, "require_learning_allowed", lambda *_: events.append("hold"))
    monkeypatch.setattr(resources, "GpuLease", lease)
    monkeypatch.setattr(checkpoints, "StopRequest", stopping)
    monkeypatch.setattr(module, "run_window_variant", run)
    arguments = [
        "run",
        "--manifest",
        str(manifest),
        "--recipe",
        str(recipe),
        "--output",
        str(output),
        "--resume",
    ]
    assert main(arguments) == code
    assert events == ["hold", "lease", ("run", stop)]
    assert json.loads(capsys.readouterr().out) == dict(
        status=status, global_step=4, output=str(output)
    )
    assert not output.exists()


def test_learning_hold_blocks_before_gpu_or_execution_and_is_not_modified(tmp_path, monkeypatch):
    main, module = cli(), backend()
    hold = importlib.import_module("mars_titan.training.learning_hold")
    resources = importlib.import_module("mars_titan.training.experiment_resources")
    guard = tmp_path / "hold.json"
    guard.write_text(json.dumps(dict(training_allowed=False)))
    monkeypatch.setenv(hold.HOLD_ENV, str(guard))
    before = guard.read_bytes()
    recipe, output = recipe_document(tmp_path), tmp_path / "run"

    def forbidden(*args, **kwargs):
        pytest.fail("El hold debe preceder a la reserva GPU y al ajuste")

    monkeypatch.setattr(resources, "GpuLease", forbidden)
    monkeypatch.setattr(module, "run_window_variant", forbidden)
    with pytest.raises(hold.LearningHoldError):
        main(
            [
                "run",
                "--manifest",
                str(tmp_path / "missing.json"),
                "--recipe",
                str(recipe),
                "--output",
                str(output),
            ]
        )
    assert guard.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("document", ["[]", '{"seed":NaN}', '{"seed":1,"seed":2}', " " * 65_537])
def test_invalid_or_oversized_json_is_rejected_without_output(tmp_path, document):
    main = cli()
    path, output = tmp_path / "recipe.json", tmp_path / "view"
    path.write_text(document)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "prepare",
                "--parent",
                str(tmp_path / "parent.json"),
                "--catalog",
                str(tmp_path / "catalog.json"),
                "--recipe",
                str(path),
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "changes",
    [
        dict(train_start="2013-01-01"),
        dict(validation_end="2024-05-01"),
        dict(batch_size=257),
        dict(final_test_access=True),
        dict(unknown_override=True),
    ],
)
def test_real_recipe_limits_reject_before_preparation_or_training(tmp_path, monkeypatch, changes):
    main, module = cli(), backend()
    path, output = recipe_document(tmp_path, **changes), tmp_path / "view"

    def forbidden(*args, **kwargs):
        pytest.fail("La receta inválida no puede llegar al backend")

    monkeypatch.setattr(module, "prepare_window_variant_view", forbidden)
    monkeypatch.setattr(module, "run_window_variant", forbidden)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "prepare",
                "--parent",
                str(tmp_path / "parent.json"),
                "--catalog",
                str(tmp_path / "catalog.json"),
                "--recipe",
                str(path),
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("arguments", [[], ["prepare"], ["run"], ["run", "--resume"]])
def test_subcommands_require_explicit_sources_recipe_and_output(arguments):
    with pytest.raises(SystemExit) as error:
        cli()(arguments)
    assert error.value.code == 2


def test_help_does_not_import_torch_or_create_an_output(tmp_path):
    cli()
    code = """
import importlib.abc
import runpy
import sys
class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('La ayuda no puede importar Torch')
sys.meta_path.insert(0, RejectTorch())
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "prepare" in completed.stdout and "run" in completed.stdout
    assert not list(tmp_path.iterdir())
