"""Motor de la criba de #746 con cohortes de fixture y ejecutores sustituidos.

Los ejecutores dobles escriben un `run.json` completado sin construir modelos. Ninguna
prueba crea un entrenador ni ejecuta pasos de optimizador.
"""

import json
from contextlib import nullcontext

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch

from benchmarks import variant_screening_study as study
from mars_titan.data.storage import atomic_json, sha256
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


def test_a_failed_job_is_charged_and_keeps_its_error(prepared, runners, monkeypatch):
    work, receipt = prepared
    item = job(receipt, "references/scalar_reference", origin=2021, seed=45)

    def broken(*args, **options):
        raise RuntimeError("fallo del ejecutor de fixture")

    monkeypatch.setattr(titans_walk_forward, "run_titans_window", broken)
    before = set((work / "attempts").glob("*.json"))
    with pytest.raises(RuntimeError, match="fallo del ejecutor"):
        study.run_job(work, item["id"])
    (path,) = set((work / "attempts").glob("*.json")) - before
    attempt = json.loads(path.read_text())
    assert attempt["status"] == "failed" and attempt["closure_confirmed"] is True
    assert attempt["error"] == "RuntimeError: fallo del ejecutor de fixture"
    assert attempt["charged_seconds"] == attempt["actual_seconds"] > 0
    assert study._completed(work, item["id"]) is None


def test_no_job_starts_once_the_budget_is_spent(prepared, runners, monkeypatch):
    work, receipt = prepared
    budget = receipt["protocol"]["budget"]["max_total_gpu_hours"] * 3600
    monkeypatch.setattr(study, "_charged", lambda _: budget)
    item = job(receipt, "references/scalar_reference", origin=2022, seed=46)
    before = set((work / "attempts").glob("*.json"))
    with pytest.raises(RuntimeError, match="presupuesto"):
        study.run_job(work, item["id"])
    assert runners == [] and set((work / "attempts").glob("*.json")) == before


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


def test_the_e3_profile_reaches_the_correction_as_its_salience_budget(prepared, runners):
    from mars_titan.memory.salience_budget import SalienceBudgetConfig

    work, receipt = prepared
    item = job(receipt, "E3/standardized_error", origin=2020, seed=42, case="eta5e-2")
    study.run_job(work, item["parent"])
    study.run_job(work, item["id"])
    options = runners[-1][2]
    assert options["salience"] == SalienceBudgetConfig(**item["definition"]["value"])
    assert options["search_case"] == "eta5e-2" and "regime_gate" not in options


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
    approximate = study.plan()
    assert approximate["jobs"] == result["jobs"]
    assert study.origin_rows()[2019] == study.DEFAULT_ROWS


def test_the_command_line_prints_the_plan_and_the_status(prepared, monkeypatch, capsys):
    work, receipt = prepared
    monkeypatch.setattr("sys.argv", ["variant_screening_study", "status", "--work", str(work)])
    study.main()
    assert json.loads(capsys.readouterr().out)["jobs"] == len(receipt["jobs"])
    monkeypatch.setattr("sys.argv", ["variant_screening_study", "plan"])
    study.main()
    assert json.loads(capsys.readouterr().out)["jobs"] == len(receipt["jobs"])


def test_the_analysis_checks_the_hold_before_reading_the_preparation(
    prepared, learning_hold, monkeypatch, tmp_path
):
    work, _ = prepared
    learning_hold(False)
    monkeypatch.setattr(
        study, "verify_prepared", lambda *_: pytest.fail("Se leyó la criba bajo la protección")
    )
    with pytest.raises(LearningHoldError):
        study.analyze(work, tmp_path / "analysis", tmp_path)
    assert not (tmp_path / "analysis").exists()


def test_the_analysis_needs_every_job_completed(prepared, runners, tmp_path):
    work, _ = prepared
    with pytest.raises(ValueError, match="Faltan .* trabajos por terminar"):
        study.analyze(work, tmp_path / "analysis", tmp_path)
    assert not (tmp_path / "analysis").exists()


DAY = 86_400_000_000
QUANTILES = ("quantile_0025", "quantile_0100", "quantile_0500", "quantile_0900", "quantile_0975")


def prediction_table(start, assets=("US/AAA", "US/BBB", "US/CCC")):
    rows = [(asset, start + day * DAY) for day in range(3) for asset in reversed(assets)]
    base = pa.array([float(i) for i in range(len(rows))])
    columns = dict(
        sample_id=[f"{asset}/{at}" for asset, at in rows],
        asset_id=[asset for asset, _ in rows],
        market=["US"] * len(rows),
        prediction_at=pa.array([at for _, at in rows], type=pa.timestamp("us", tz="UTC")),
        target=base,
        prediction=base,
    )
    columns.update({name: pa.array(base.to_numpy() + k) for k, name in enumerate(QUANTILES)})
    return pa.table(columns)


def test_run_panels_read_the_quantile_levels_and_check_every_file(tmp_path):
    folder = tmp_path / "2019-42-abc"
    folder.mkdir()
    predictions = {}
    for k, name in enumerate(("validation", "calibration", "evaluation")):
        path = folder / f"{name}-predictions.parquet"
        pq.write_table(prediction_table(1_546_000_000_000_000 + 10 * k * DAY), path)
        predictions[name] = dict(path=path.name, sha256=sha256(path))
    atomic_json(folder / "run.json", dict(status="completed", predictions=predictions))
    reads = {}
    panels = study.run_panels(folder, reads)
    assert panels["evaluation"].levels == (0.025, 0.1, 0.5, 0.9, 0.975)
    assert panels["evaluation"].quantiles.shape == (9, 5) and panels["validation"].rows == 9
    assert set(reads) == {f"2019-42-abc/{name}-predictions.parquet" for name in predictions}
    pq.write_table(
        prediction_table(1_546_000_000_000_000 + DAY), folder / "evaluation-predictions.parquet"
    )
    with pytest.raises(ValueError, match="Ha cambiado"):
        study.run_panels(folder, {})
    atomic_json(folder / "run.json", dict(status="failed", predictions=predictions))
    with pytest.raises(ValueError, match="no ha terminado"):
        study.run_panels(folder, {})


def test_the_maturity_of_the_view_follows_the_canonical_order_of_the_panels(monkeypatch, tmp_path):
    from mars_titan.evaluation import online_calibration
    from mars_titan.evaluation.forecast_panel import ForecastPanel

    view = tmp_path / "view.json"
    view.write_text("{}")
    tables = {}
    for k, name in enumerate(("validation", "calibration", "evaluation")):
        table = prediction_table(1_546_000_000_000_000 + 10 * k * DAY)
        # Cada fila madura en un instante propio, también dentro de la misma sesión.
        offsets = pa.array(DAY + 1_000 * np.arange(table.num_rows) + k)
        matures = pc.add(table["prediction_at"].cast(pa.int64()), offsets)
        tables[name] = table.select(
            ["asset_id", "market", "prediction_at", "target"]
        ).append_column("matures_at", matures)
    calls = []

    def maturity(path, policy, expected, partitions):
        calls.append((path, policy, expected, partitions))
        return tables

    monkeypatch.setattr(online_calibration, "view_maturity", maturity)
    cohorts, aligned = study.origin_maturity(view, "historical_masked_2000_v1")
    assert calls == [
        (
            view,
            "historical_masked_2000_v1",
            sha256(view),
            study.variant_screening_analysis.PARTITIONS,
        )
    ]
    for name, table in tables.items():
        sample = pc.binary_join_element_wise(
            table["asset_id"], table["prediction_at"].cast(pa.int64()).cast(pa.string()), "/"
        )
        panel = ForecastPanel.from_arrow(
            table.append_column("sample_id", sample).append_column("prediction", table["target"])
        )
        matures = dict(zip(sample.to_pylist(), table["matures_at"].to_pylist(), strict=True))
        assert cohorts[name] == panel.cohort_sha256
        assert aligned[name].tolist() == [matures[row] for row in panel.row_id.to_pylist()]
        assert aligned[name].tolist() != table["matures_at"].to_pylist()


def test_current_sectors_leave_missing_or_unknown_sectors_empty(tmp_path):
    (tmp_path / "sp500stock_data_description.csv").write_text(
        "stock_name,Sector\nAAA,Technology\nBBB,N/A\n", encoding="utf-8"
    )
    (tmp_path / "hs300stock_data_description.csv").write_bytes(
        "证券代码,申万一级行业\n600000.SH,银行\n".encode("gb18030")
    )
    sectors = study.current_sectors(tmp_path, ["US/AAA", "US/BBB", "US/ZZZ"])
    assert sectors == {"US/AAA": "US/Technology", "US/BBB": None, "US/ZZZ": None}


def test_the_analysis_writes_its_report_with_the_hashes_of_what_it_read(
    prepared, runners, monkeypatch, tmp_path
):
    work, receipt = prepared
    for name in ("sp500stock_data_description.csv", "hs300stock_data_description.csv"):
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(study, "_completed", lambda *_: dict(status="completed"))
    monkeypatch.setattr(study, "origin_maturity", lambda view, policy: ({}, {}))
    monkeypatch.setattr(study, "current_sectors", lambda path, assets: dict.fromkeys(assets))
    read = []

    def panels(folder, reads):
        read.append(folder)
        reads[f"{folder.name}/evaluation-predictions.parquet"] = "f" * 64
        return {}

    monkeypatch.setattr(study, "run_panels", panels)
    seen = {}

    def analyze(protocol, jobs, origins, root):
        seen.update(protocol=protocol, jobs=jobs, origins=origins)
        origins[2020].load("job-a")
        origins[2020].load("job-a")
        values, defined = np.array([0.1, 0.2, 0.3]), np.array([True, True, False])
        report = dict(decisions=dict(B1=dict(passes=False)), sessions=3)
        merged = {(("B1", "area_memory"), "session_ic"): (values, defined)}
        return report, dict(session_time=np.array([1, 2, 3]), merged=merged)

    monkeypatch.setattr(study.variant_screening_analysis, "analyze", analyze)
    result = study.analyze(work, tmp_path / "analysis", tmp_path)
    assert result["decisions"] == {"B1": False} and result["sessions"] == 3
    assert seen["protocol"] == receipt["protocol"] and seen["jobs"] == receipt["jobs"]
    assert sorted(seen["origins"]) == sorted(screening_cohorts.ORIGIN_FOLDS)
    assert all(len(origin.sectors) == 256 for origin in seen["origins"].values())
    assert read == [work / "runs" / "job-a"]
    report = json.loads((tmp_path / "analysis" / "analysis.json").read_text())
    assert report["final_test_access"] is False and report["optimizer_steps"] == 0
    assert report["prepared_sha256"] == sha256(work / "prepared.json")
    assert report["predictions_sha256"] == {"job-a/evaluation-predictions.parquet": "f" * 64}
    assert set(report["descriptions_sha256"]) == {
        "sp500stock_data_description.csv",
        "hs300stock_data_description.csv",
    }
    series = pq.read_table(tmp_path / "analysis" / "series.parquet")
    assert series["model"].to_pylist() == ["B1/area_memory"] * 3
    assert series["defined"].to_pylist() == [True, True, False]
    with pytest.raises(ValueError, match="directorio nuevo"):
        study.analyze(work, tmp_path / "analysis", tmp_path)


def test_the_queue_runs_pending_jobs_in_sealed_order_up_to_its_limit(prepared, runners):
    work, receipt = prepared
    pending = [job["id"] for job in receipt["jobs"] if study._completed(work, job["id"]) is None]
    result = study.run_queue(work, limit=3)
    assert result == dict(completed=pending[:3], stopped_at=None, status="queued_jobs_completed")
    assert [options["output"].name for _, _, options in runners] == pending[:3]
    with pytest.raises(ValueError, match="entero positivo"):
        study.run_queue(work, limit=0)


def test_the_queue_stops_at_a_job_that_does_not_finish(prepared, runners, monkeypatch):
    work, receipt = prepared
    paused = []

    def pause(*args, **options):
        paused.append(options["output"].name)
        return dict(status="paused")

    monkeypatch.setattr(titans_walk_forward, "run_titans_window", pause)
    result = study.run_queue(work)
    assert result["status"] == "paused" and result["stopped_at"] == paused[0]
    assert len(paused) == 1 and study._completed(work, paused[0]) is None


def test_the_queue_checks_the_hold_before_reading_the_preparation(
    prepared, learning_hold, monkeypatch
):
    work, _ = prepared
    learning_hold(False)
    monkeypatch.setattr(
        study, "verify_prepared", lambda *_: pytest.fail("Se leyó la criba bajo la protección")
    )
    with pytest.raises(LearningHoldError):
        study.run_queue(work)
