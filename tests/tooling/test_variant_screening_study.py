"""Motor de la criba de #746 con cohortes de fixture y ejecutores sustituidos.

Los ejecutores dobles escriben un `run.json` completado sin construir modelos. Ninguna
prueba crea un entrenador ni ejecuta pasos de optimizador.
"""

import json
from contextlib import nullcontext

import pytest
import torch

from benchmarks import variant_screening_study as study
from mars_titan.data.storage import atomic_json
from mars_titan.training import (
    experiment_resources,
    mars_titan_correction,
    mars_titan_walk_forward,
    screening_cohorts,
    spaced_window,
    titans_walk_forward,
    variant_screening,
)
from mars_titan.training.learning_hold import LearningHoldError
from tests.training.test_screening_cohorts import source


@pytest.fixture(scope="module")
def cohorts(tmp_path_factory):
    sources = tmp_path_factory.mktemp("sources")
    for origin, fold in screening_cohorts.ORIGIN_FOLDS.items():
        atomic_json(sources / fold / "manifest.json", source(origin))
    output = tmp_path_factory.mktemp("cohorts") / "work"
    screening_cohorts.prepare(sources, output)
    return output


@pytest.fixture(scope="module")
def prepared(cohorts, tmp_path_factory):
    work = tmp_path_factory.mktemp("screening") / "work"
    return work, study.prepare(cohorts, work)


class Lease:
    record = dict(fixture=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def check(self):
        pass


@pytest.fixture
def runners(monkeypatch, learning_doubles):
    """Ejecutores dobles que registran sus argumentos y completan su carpeta."""
    calls = []

    def runner(name):
        def run(*args, **options):
            calls.append((name, args, options))
            report = dict(status="completed", run_id=f"{name}-{len(calls)}")
            if name == "correction" and "regime_gate" in options:
                gate = options["regime_gate"]
                metrics = dict(regime_gate=dict(configuration=gate.identity(), mean_gate=0.375))
                report["predictions"] = dict(validation=dict(metrics=metrics))
            atomic_json(options["output"] / "run.json", report)
            return report

        return run

    monkeypatch.setattr(titans_walk_forward, "run_titans_window", runner("fit"))
    monkeypatch.setattr(mars_titan_correction, "run_correction_window", runner("correction"))
    monkeypatch.setattr(mars_titan_walk_forward, "run_mars_titan_window", runner("reader"))
    monkeypatch.setattr(spaced_window, "run_spaced_window", runner("spaced"))
    monkeypatch.setattr(experiment_resources, "GpuLease", lambda **_: Lease())
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(titans_walk_forward, "unfused_attention", nullcontext)
    return calls


def job(receipt, label, *, origin=2019, seed=42, case=None):
    for item in receipt["jobs"]:
        if (
            label in item["used_by"]
            and (item["origin"], item["seed"]) == (origin, seed)
            and (case is None or item["definition"].get("search_case") == case)
        ):
            return item
    raise KeyError(label)


def test_prepare_seals_the_views_recipes_and_jobs_of_the_protocol(prepared):
    work, receipt = prepared
    protocol = variant_screening.load_protocol()
    assert [{k: v for k, v in j.items() if k != "files"} for j in receipt["jobs"]] == (
        variant_screening.screening_jobs(protocol)
    )
    assert receipt["final_test_access"] is False and receipt["optimizer_steps"] == 0
    for year in screening_cohorts.ORIGIN_FOLDS:
        view = json.loads((work / f"origins/{year}/view-x1.json").read_text())
        walk = view["temporal_view"]["protocol"]
        assert walk["selection"]["max_epochs"] == protocol["fit"]["epochs"]
        assert walk["seeds"] == protocol["seeds"]
        assert len(view["assets"]) == 256
        assert json.loads((work / f"origins/{year}/protocol-x1.json").read_text()) == walk
    laplace = job(receipt, "F1/laplace_scale_head")
    recipe = json.loads((work / laplace["files"]["recipe"]).read_text())
    assert recipe["predictor"]["head"] == "scalar" and recipe["recipe"]["loss"] == "mae"
    assert recipe["recipe"]["epochs"] == protocol["fit"]["epochs"]
    assert recipe["walk_forward"]["fit_months"] == protocol["fit"]["fit_months"]
    reference = job(receipt, "references/quantile_reference")
    recipe = json.loads((work / reference["files"]["recipe"]).read_text())
    assert recipe["predictor"]["head"] == "quantile_head_v1"
    assert "recipe" not in job(receipt, "B1/area_memory")["files"]
    assert set(receipt["declarations"]) >= {
        "configs/experiments/redundant-area-memory-v1.json",
        "configs/titans/mature-correction-historical-masked.json",
    }


def test_recipe_changes_and_matched_updates_reach_the_fit_recipe():
    protocol = variant_screening.load_protocol()
    definition = dict(
        runner="fit",
        base_recipe="configs/titans/chronological-training-historical-sequence-v1.json",
        head=None,
        loss=None,
        predictor_changes={},
        recipe_changes=dict(weight_average=dict(decay=0.99)),
        updates_multiplier=3,
    )
    recipe = study.fit_recipe(protocol, definition)
    assert recipe["recipe"]["weight_average"] == dict(decay=0.99)
    assert recipe["recipe"]["epochs"] == 3 * protocol["fit"]["epochs"]
    view, _ = screening_cohorts.project_origin_cohort(source(2020), origin=2020)
    projected = study.project_view(view, protocol, 2020, 3)
    assert projected["temporal_view"]["protocol"]["selection"]["max_epochs"] == 36
    assert view["temporal_view"]["protocol"]["selection"]["max_epochs"] == 30


def test_a_view_of_another_origin_or_size_is_rejected():
    protocol = variant_screening.load_protocol()
    view, _ = screening_cohorts.project_origin_cohort(source(2019), origin=2019)
    assert study.project_view(view, protocol, 2019, 1)["assets"] == view["assets"]
    with pytest.raises(ValueError, match="cohorte"):
        study.project_view(view, protocol, 2020, 1)
    small, _ = screening_cohorts.project_origin_cohort(source(2019), origin=2019, size=64)
    with pytest.raises(ValueError, match="cohorte"):
        study.project_view(small, protocol, 2019, 1)


def test_prepare_needs_a_new_directory(prepared, cohorts):
    work, _ = prepared
    with pytest.raises(ValueError, match="nuevo"):
        study.prepare(cohorts, work)


def test_verification_rejects_a_changed_artifact_or_protocol(prepared, monkeypatch, tmp_path):
    work, receipt = prepared
    assert study.verify_prepared(work)["protocol_sha256"] == receipt["protocol_sha256"]
    original = variant_screening.load_protocol()
    changed = dict(original, seeds=[42, 43, 44, 45, 47])
    monkeypatch.setattr(variant_screening, "load_protocol", lambda root: changed)
    with pytest.raises(ValueError, match="protocolo"):
        study.verify_prepared(work)
    monkeypatch.undo()
    name = next(iter(receipt["artifacts"]))
    path = work / name
    saved = path.read_bytes()
    try:
        path.write_bytes(saved + b" ")
        with pytest.raises(ValueError, match="Ha cambiado"):
            study.verify_prepared(work)
    finally:
        path.write_bytes(saved)


def test_the_hold_is_checked_before_reading_the_preparation(prepared, learning_hold, monkeypatch):
    work, receipt = prepared
    learning_hold(False)
    monkeypatch.setattr(
        study, "verify_prepared", lambda *_: pytest.fail("Se leyó la criba bajo la protección")
    )
    with pytest.raises(LearningHoldError):
        study.run_job(work, receipt["jobs"][0]["id"])


def test_a_fit_runs_with_its_sealed_files_and_is_charged_once(prepared, runners):
    work, receipt = prepared
    item = job(receipt, "references/quantile_reference", seed=43)
    result = study.run_job(work, item["id"])
    assert result["status"] == "completed"
    ((name, args, options),) = runners
    assert name == "fit"
    assert args == (
        work / item["files"]["view"],
        work / item["files"]["protocol"],
        "fold-014",
        work / item["files"]["recipe"],
    )
    assert options["seed"] == 43 and options["search_case"] == "lr1e-3"
    assert options["predictor_changes"] is None and options["device"] == "cuda:0"
    assert options["output"] == work / "runs" / item["id"]
    attempts = [json.loads(p.read_text()) for p in (work / "attempts").glob("*.json")]
    (attempt,) = [a for a in attempts if a["job"] == item["id"]]
    assert attempt["closure_confirmed"] is True and attempt["status"] == "completed"
    assert attempt["optimizer_steps"] == 0
    assert study.run_job(work, item["id"])["status"] == "completed"
    assert len(runners) == 1


def test_a_correction_waits_for_its_parent_and_receives_its_case(prepared, runners):
    work, receipt = prepared
    item = job(receipt, "B1/area_memory", seed=44, case="eta25e-2")
    with pytest.raises(ValueError, match="necesita antes"):
        study.run_job(work, item["id"])
    assert runners == []
    study.run_job(work, item["parent"])
    study.run_job(work, item["id"])
    name, args, options = runners[-1]
    assert name == "correction"
    assert args[1] == work / "runs" / item["parent"]
    assert args[2] == study.ROOT / "configs/titans/mature-correction-historical-masked.json"
    assert options["search_case"] == "eta25e-2" and options["seed"] == 44
    assert options["components"] == item["definition"]["components"]
    assert "salience" not in options and "regime_gate" not in options


def test_the_matched_e4_control_uses_the_mean_gate_of_its_variant(prepared, runners):
    work, receipt = prepared
    control = job(receipt, "E4/matched_constant", seed=45, case="eta5e-2")
    variant = next(item for item in receipt["jobs"] if item["id"] == control["after"])
    study.run_job(work, control["parent"])
    with pytest.raises(ValueError, match="necesita antes"):
        study.run_job(work, control["id"])
    study.run_job(work, variant["id"])
    study.run_job(work, control["id"])
    gate = runners[-1][2]["regime_gate"]
    assert gate.mode == "constant" and gate.constant == 0.375
    assert runners[-2][2]["regime_gate"].mode == "cusum"


def test_readers_and_spaced_windows_receive_their_parent(prepared, runners):
    work, receipt = prepared
    reader = job(receipt, "references/reader_reference", seed=46)
    spaced = job(receipt, "E2/utility", seed=46)
    assert spaced["parent"] == reader["id"]
    study.run_job(work, reader["parent"])
    study.run_job(work, reader["id"])
    study.run_job(work, spaced["id"])
    names = [call[0] for call in runners]
    assert names == ["fit", "reader", "spaced"]
    _, args, options = runners[1]
    assert args[1] == work / "runs" / reader["parent"]
    assert options["components"] == {"episodic_bank": "m1"}
    _, args, options = runners[2]
    assert args[1] == work / "runs" / reader["id"]
    assert options["arm"] == "utility"


def test_status_lists_the_jobs_without_pending_requirements(prepared):
    work, receipt = prepared
    result = study.status(work)
    assert result["jobs"] == len(receipt["jobs"])
    assert result["completed"] <= result["jobs"]
    index = {item["id"]: item for item in receipt["jobs"]}
    for name in result["ready"]:
        for field in variant_screening.REQUIRES:
            required = index[name][field]
            assert required is None or (work / "runs" / required / "run.json").exists()


def test_plan_estimates_the_cost_from_the_cohorts(cohorts):
    result = study.plan(cohorts)
    assert result["jobs"] == result["cost"]["jobs"] > 0
    assert result["cost"]["estimate"] == "pilot_measurements_not_a_measured_screening_cost"
    assert {row["entry"] for row in result["pending"]} >= {"C2", "C3", "G1"}
