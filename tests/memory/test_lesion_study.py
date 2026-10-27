"""Declaración de B4 y entrada protegida a la ventana existente."""

import copy
import json
import runpy
from contextlib import contextmanager
from pathlib import Path

import pytest

from mars_titan.memory.lesion_protocol import LesionProtocol

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/lesion-protocol-v1.json"


def test_saved_configuration_reconstructs_the_original_protocol():
    from mars_titan.memory.lesion_study import load_study

    study, digest = load_study(CONFIG)
    assert study.protocol == LesionProtocol()
    assert study.analysis["block_length"] == 16
    assert study.analysis["sensitivity_block_lengths"] == [8, 32]
    assert study.analysis["replicates"] == 2000
    assert len(digest) == 64


def test_variant_fingerprint_belongs_to_the_bytes_read_even_if_the_file_changes(
    tmp_path, monkeypatch
):
    from mars_titan.data.storage import sha256
    from mars_titan.memory import lesion_study

    path = tmp_path / "variants.json"
    document = dict(
        kind="mature_correction_variants",
        default_enabled=False,
        final_test_access=False,
        arms=dict(fixture=dict(recipe="recipe.json", components=dict(value=1))),
    )
    path.write_text(json.dumps(document))
    expected = sha256(path)
    original = lesion_study.read_manifest

    def changed(*args, **kwargs):
        value, digest = original(*args, **kwargs)
        altered = copy.deepcopy(value)
        altered["arms"]["fixture"]["components"]["value"] = 2
        path.write_text(json.dumps(altered))
        return value, digest

    monkeypatch.setattr(lesion_study, "read_manifest", changed)
    recipe, components, digest = lesion_study.select_arm(path, "fixture")
    assert recipe == Path("recipe.json") and components == dict(value=1)
    assert digest == expected and digest != sha256(path)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(default_enabled=True),
        lambda d: d.update(final_test_access=True),
        lambda d: d.update(schema_version=True),
        lambda d: d.update(unrecognized=1),
        lambda d: d["protocol"].update(generator="another"),
        lambda d: d["analysis"].update(replicates=0),
        lambda d: d["analysis"].update(seed=True),
        lambda d: d["analysis"].update(confidence=1.0),
        lambda d: d["analysis"].update(market_weighting="rows"),
        lambda d: d["analysis"].update(sensitivity_block_lengths=[8, 8]),
        lambda d: d["analysis"].update(rank_ic_min_assets=2),
    ],
)
def test_invalid_declarations_do_not_change_the_saved_configuration(tmp_path, change):
    from mars_titan.memory.lesion_study import load_study

    original = json.loads(CONFIG.read_text())
    altered = copy.deepcopy(original)
    change(altered)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError):
        load_study(path)
    assert json.loads(CONFIG.read_text()) == original


def test_window_cli_checks_the_hold_before_reading_files_or_reserving_cuda(tmp_path, learning_hold):
    from mars_titan.training.learning_hold import LearningHoldError

    script = ROOT / "scripts/run_lesion_window.py"
    assert script.is_file(), "Falta la orden de B4"
    main = runpy.run_path(str(script), run_name="lesion_cli_test")["main"]
    args = []
    for name in ("view", "parent", "study", "lesions", "output"):
        args += [f"--{name}", str(tmp_path / name)]
    args += ["--arm", "bridge", "--search-case", "eta-0", "--seed", "0"]
    with pytest.raises(LearningHoldError):
        main(args)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("device,status,code", [("cpu", "completed", 0), ("cuda:0", "paused", 3)])
def test_cli_reuses_the_window_and_freezes_analysis_before_execution(
    tmp_path, monkeypatch, capsys, device, status, code
):
    from mars_titan.memory import lesion_study
    from mars_titan.training import experiment_resources, mars_titan_correction

    script = ROOT / "scripts/run_lesion_window.py"
    main = runpy.run_path(str(script), run_name="lesion_cli_test")["main"]
    output, events = tmp_path / "run", []

    @contextmanager
    def lease():
        events.append("lease")
        yield

    def run(view, parent, recipe, **options):
        assert (view, parent) == (tmp_path / "view", tmp_path / "parent")
        assert recipe == Path("configs/titans/mature-correction-historical-masked.json")
        assert options["lesions"] == LesionProtocol()
        assert options["device"] == device and options["seed"] == 7
        assert options["search_case"] == "eta-0"
        assert options["components"]["associative_memory"]["query_bridge"]["gain"] == 0.25
        frozen = json.loads((output / "lesion-study.json").read_text())
        assert frozen["analysis"]["replicates"] == 2000
        assert frozen["protocol"] == options["lesions"].identity()
        assert frozen["seed"] == 7 and frozen["arm"] == "bridge"
        events.append("run")
        return dict(status=status, run_id="fixture")

    monkeypatch.setitem(
        main.__globals__, "require_learning_allowed", lambda *_: events.append("hold")
    )
    monkeypatch.setattr(experiment_resources, "GpuLease", lease)
    monkeypatch.setattr(mars_titan_correction, "run_correction_window", run)
    args = [
        "--view",
        str(tmp_path / "view"),
        "--parent",
        str(tmp_path / "parent"),
        "--study",
        str(ROOT / "configs/experiments/area-query-bridge-v1.json"),
        "--lesions",
        str(CONFIG),
        "--output",
        str(output),
        "--arm",
        "bridge",
        "--search-case",
        "eta-0",
        "--seed",
        "7",
        "--device",
        device,
    ]
    assert main(args) == code
    assert events == ["hold", *(["lease"] if device == "cuda:0" else []), "run"]
    assert json.loads(capsys.readouterr().out)["status"] == status
    # Reanudar exige la misma declaración, también si cambia solo el análisis.
    study, digest = lesion_study.load_study(CONFIG)
    assert (output / "lesion-study.json").exists() and digest
    study.analysis["replicates"] = 100
    with pytest.raises(ValueError, match="declaración"):
        lesion_study.freeze_study(
            output,
            study=study,
            study_sha256=digest,
            arms_sha256="0" * 64,
            arm="bridge",
            seed=7,
            search_case="eta-0",
            recipe=ROOT / "configs/titans/mature-correction-historical-masked.json",
        )
