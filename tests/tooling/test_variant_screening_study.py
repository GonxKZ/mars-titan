"""Motor de la criba de #746 con cohortes de fixture y ejecutores sustituidos.

Los ejecutores dobles escriben un `run.json` completado sin construir modelos. Ninguna
prueba crea un entrenador ni ejecuta pasos de optimizador.
"""

import itertools
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch

from benchmarks import variant_screening_study as study
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.training import (
    common_control_windows,
    experiment_resources,
    mars_titan_correction,
    mars_titan_walk_forward,
    screening_cohorts,
    selection,
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
    monkeypatch.setattr(common_control_windows, "run_rule_window", runner("rule"))
    monkeypatch.setattr(common_control_windows, "run_tabular_window", runner("tabular"))
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


def test_the_common_controls_run_on_cpu_with_the_files_of_their_origin(
    prepared, runners, monkeypatch
):
    work, receipt = prepared
    protocol = receipt["protocol"]

    def forbidden(*_, **__):
        raise AssertionError("La regla y LightGBM no toman la GPU")

    monkeypatch.setattr(experiment_resources, "GpuLease", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    for label, runner in (("common/momentum_reversal", "rule"), ("common/lightgbm", "tabular")):
        (item,) = [j for j in receipt["jobs"] if j["used_by"] == [label] and j["origin"] == 2020]
        assert item["seed"] is None and item["seeds"] == protocol["seeds"]
        assert study.run_job(work, item["id"])["status"] == "completed"
        name, args, options = runners[-1]
        assert name == runner
        assert args == (work / item["files"]["view"], work / item["files"]["protocol"], "fold-015")
        assert options["output"] == work / "runs" / item["id"]
        assert options["config"] == item["definition"]["config"]
        assert options["stop"] is not None and options["stop"].config["max_job_seconds"] > 0
        if runner == "tabular":
            assert options["seeds"] == protocol["seeds"] and options["fit_months"] == 12
            assert options["radius"] == protocol["selection"]["smoothing_radius"]
            assert options["min_assets"] == protocol["contrast"]["primary"]["minimum_assets"]
        attempts = [json.loads(p.read_text()) for p in (work / "attempts").glob("*.json")]
        (attempt,) = [a for a in attempts if a["job"] == item["id"]]
        assert attempt["status"] == "completed" and "admission" not in attempt
    assert item["files"]["view"] == "origins/2020/view-x1.json" and "recipe" not in item["files"]


def write_run(folder, start):
    folder.mkdir(parents=True)
    predictions = {}
    for k, name in enumerate(("validation", "calibration", "evaluation")):
        path = folder / f"{name}-predictions.parquet"
        pq.write_table(prediction_table(start + 10 * k * DAY), path)
        predictions[name] = dict(path=path.name, sha256=sha256(path))
    atomic_json(folder / "run.json", dict(status="completed", predictions=predictions))
    return sha256(folder / "run.json")


def test_each_seed_of_lightgbm_is_read_from_its_folder_with_its_recorded_hash(tmp_path):
    job_id = "2019-abc"
    folder = tmp_path / "runs" / job_id
    seeds = {}
    for k, seed in enumerate((42, 43)):
        digest = write_run(folder / f"seed-{seed}", 1_546_000_000_000_000 + k * DAY)
        seeds[str(seed)] = dict(path=f"seed-{seed}/run.json", sha256=digest)
    atomic_json(folder / "run.json", dict(status="completed", seeds=seeds))
    reads = {}
    panels = study.job_panels(tmp_path, f"{job_id}/seed-43", reads)
    assert panels["evaluation"].rows == 9
    assert set(reads) == {
        f"{job_id}/seed-43/{n}-predictions.parquet"
        for n in study.variant_screening_analysis.PARTITIONS
    }
    with pytest.raises(ValueError, match="no emite seed-44"):
        study.job_panels(tmp_path, f"{job_id}/seed-44", {})
    record = folder / "seed-42" / "run.json"
    record.write_text(record.read_text() + " ")
    with pytest.raises(ValueError, match="Ha cambiado seed-42 de 2019-abc"):
        study.job_panels(tmp_path, f"{job_id}/seed-42", {})
    atomic_json(folder / "run.json", dict(status="paused", seeds=seeds))
    with pytest.raises(ValueError, match="El trabajo 2019-abc no ha terminado"):
        study.job_panels(tmp_path, f"{job_id}/seed-43", {})
    plain = tmp_path / "runs" / "2019-42-def"
    write_run(plain, 1_546_000_000_000_000)
    reads = {}
    study.job_panels(tmp_path, "2019-42-def", reads)
    assert set(reads) == {
        f"2019-42-def/{n}-predictions.parquet" for n in study.variant_screening_analysis.PARTITIONS
    }


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
    calls = []
    monkeypatch.setattr(study, "rule_check", lambda *args: calls.append(args) or dict(groups=0))
    command = ["variant_screening_study", "rule-check", "--work", str(work), "--output", "check"]
    monkeypatch.setattr("sys.argv", command)
    study.main()
    assert json.loads(capsys.readouterr().out) == dict(groups=0)
    assert calls == [(work, Path("check"))]


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
    finished = dict(status="completed", fit=dict(joint_stop_epoch=4), versions=dict(epochs=[3]))
    monkeypatch.setattr(study, "_completed", lambda *_: finished)
    monkeypatch.setattr(study, "_decision", lambda *_: dict(epoch=4, versions=[3]))
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
        origins[2021].load("job-a")
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
    # Un ajuste emitido con otra época o con otras versiones que las de su grupo se rechaza.
    for changed in (dict(epoch=5, versions=[3]), dict(epoch=4, versions=[6]), None):
        monkeypatch.setattr(study, "_decision", lambda *_, value=changed: value)
        with pytest.raises(ValueError, match="no emite la época común"):
            study.analyze(work, tmp_path / "again", tmp_path)
    assert not (tmp_path / "again").exists()


def test_the_queue_runs_pending_jobs_in_sealed_order_up_to_its_limit(prepared, runners):
    work, receipt = prepared
    pending = [job["id"] for job in receipt["jobs"] if study._completed(work, job["id"]) is None]
    result = study.run_queue(work, limit=3)
    assert result == dict(
        completed=pending[:3], stopped_at=None, status="limit_reached", executed=3
    )
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


EPOCHS = 12
# IC medio con máximo suavizado en la época 5. Cada semilla suma una constante pequeña.
PEAK = [0.05 - 0.01 * abs(epoch - 5) for epoch in range(1, EPOCHS + 1)]
# Tres sesiones al mes durante seis meses: la comprobación de la regla elige con los cuatro
# primeros y mide con los dos últimos.
SESSIONS = [
    int(np.datetime64(f"2019-{month:02d}-{day:02d}T21:00", "us").astype(np.int64))
    for month in range(1, 7)
    for day in (3, 10, 17)
]


def epoch_record(seed, epoch):
    value = PEAK[epoch - 1] + 0.001 * (seed - 42)
    return dict(
        session_mae=0.5,
        session_ic=value,
        session_ic_values=[value] * len(SESSIONS),
        session_ic_defined=[True] * len(SESSIONS),
        session_mae_values=[0.5 - 0.01 * epoch] * len(SESSIONS),
        session_time=SESSIONS,
        cohort_sha256="cohort",
    )


@pytest.fixture
def common_runners(monkeypatch, learning_doubles):
    """Ajuste doble que recorre su presupuesto, espera la época común y después termina."""
    calls = []

    def fit(*args, **options):
        calls.append(options)
        output = options["output"]
        if options["joint_epoch"] is None:
            history = [dict(epoch=0, validation=dict(session_mae=0.6))]
            history += [
                dict(epoch=epoch, validation=epoch_record(options["seed"], epoch))
                for epoch in range(1, EPOCHS + 1)
            ]
            atomic_json(
                output / "fit" / "run.json",
                dict(
                    status=selection.AWAIT,
                    history=history,
                    epoch_states={str(e): dict(sha256="0" * 64) for e in range(1, EPOCHS + 1)},
                    selection=dict(best_epoch=EPOCHS),
                ),
            )
            report = dict(status=selection.AWAIT, individual_stop_epoch=EPOCHS)
        else:
            report = dict(
                status="completed",
                fit=dict(best_epoch=options["joint_epoch"]),
                versions=options["versions"],
            )
        atomic_json(output / "run.json", report)
        return report

    monkeypatch.setattr(titans_walk_forward, "run_titans_window", fit)
    monkeypatch.setattr(experiment_resources, "GpuLease", lambda **_: Lease())
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(titans_walk_forward, "unfused_attention", nullcontext)
    return calls


def attempts(work):
    return len(list((work / "attempts").glob("*.json"))) if (work / "attempts").exists() else 0


def test_a_group_waits_for_its_seeds_and_resumes_with_the_common_epoch(
    prepared, common_runners, monkeypatch
):
    work, receipt = prepared
    members = [
        job(receipt, "F4/session_ic_objective", origin=2021, seed=seed)
        for seed in receipt["protocol"]["seeds"]
    ]
    group = members[0]["common_group"]
    assert {item["common_group"] for item in members} == {group}
    assert members[0]["versions"] == [3, 6, 9, 12] and not members[0]["epoch_predictions"]
    for position, item in enumerate(members):
        result = study.run_job(work, item["id"])
        assert result["status"] == selection.AWAIT
        options = common_runners[-1]
        assert options["stopping"]["stopping"] == "common_epoch"
        assert options["stopping"]["metric"] == "session_mae"
        assert options["epoch_scoring"] == dict(min_assets=3, predictions=False)
        assert options["joint_epoch"] is None and options["versions"] is None
        assert study._group_waits(work, receipt, group) is (position == len(members) - 1)
        if position == 0:
            with pytest.raises(ValueError, match="no espera la época común"):
                study.decide_group(work, receipt, group)
    # Repetir un trabajo que espera sin la época del grupo no llama al ejecutor ni se cobra.
    before, calls = attempts(work), len(common_runners)
    assert study.run_job(work, members[0]["id"])["status"] == selection.AWAIT
    assert (attempts(work), len(common_runners)) == (before, calls)
    assert study.status(work)["awaiting_common_epoch"] >= len(members)
    decision = study.decide_group(work, receipt, group)
    assert decision["epoch"] == 5 and decision["versions"] == [3, 6, 9, 12]
    assert decision["kind"] == study.GROUP_KIND and decision["final_test_access"] is False
    assert decision["mae_best_epochs"] == {item["id"]: EPOCHS for item in members}
    assert set(decision["members"]) == {item["id"] for item in members}
    # La decisión no se reescribe.
    assert study.decide_group(work, receipt, group) == decision
    resume = titans_walk_forward.run_titans_window
    monkeypatch.setattr(
        titans_walk_forward, "run_titans_window", lambda *_, **__: dict(status=selection.AWAIT)
    )
    with pytest.raises(ValueError, match="volvió a esperar"):
        study.run_job(work, members[0]["id"])
    monkeypatch.setattr(titans_walk_forward, "run_titans_window", resume)
    for item in members:
        result = study.run_job(work, item["id"])
        assert result["status"] == "completed" and result["fit"]["best_epoch"] == 5
        assert common_runners[-1]["joint_epoch"] == 5
        assert common_runners[-1]["versions"] == [3, 6, 9, 12]
    assert study.status(work)["decided_groups"] >= 1


def test_the_queue_releases_each_group_before_the_next_one_trains(
    prepared, common_runners, monkeypatch
):
    work, receipt = prepared
    seeds = receipt["protocol"]["seeds"]
    first = [job(receipt, "F5/auxiliary_horizons", origin=2020, seed=seed) for seed in seeds]
    second = [job(receipt, "F4/session_ic_objective", origin=2022, seed=seed) for seed in seeds]
    reduced = dict(receipt, jobs=first + second)
    monkeypatch.setattr(study, "verify_prepared", lambda _: reduced)
    # Cinco ajustes recorren su presupuesto y esperan, y el límite corta tras dos reanudaciones.
    result = study.run_queue(work, limit=7)
    ids = [item["id"] for item in first]
    assert result == dict(completed=ids[:2], stopped_at=None, status="limit_reached", executed=7)
    assert study._decision(work, first[0]["common_group"])["epoch"] == 5
    result = study.run_queue(work)
    assert result["status"] == "queued_jobs_completed" and result["pending"] == 0
    assert result["completed"] == ids[2:] + [item["id"] for item in second]
    # El primer grupo emite y libera sus estados antes de que el segundo empiece a ajustar.
    order = [(options["output"].name, options["joint_epoch"]) for options in common_runners]
    assert order == [(i, None) for i in ids] + [(i, 5) for i in ids] + [
        (item["id"], epoch) for epoch in (None, 5) for item in second
    ]


def test_the_group_decision_needs_the_same_validation_rows(prepared, common_runners):
    work, receipt = prepared
    members = [
        job(receipt, "F1/laplace_scale_head", origin=2022, seed=seed)
        for seed in receipt["protocol"]["seeds"]
    ]
    for item in members:
        study.run_job(work, item["id"])
    path = work / "runs" / members[-1]["id"] / "fit" / "run.json"
    report = json.loads(path.read_text())
    report["history"][3]["validation"]["cohort_sha256"] = "other"
    atomic_json(path, report)
    with pytest.raises(ValueError, match="mismas filas"):
        study.decide_group(work, receipt, members[0]["common_group"])
    assert study._decision(work, members[0]["common_group"]) is None


def write_epochs(root, job_id, predictions, target, times):
    """Historial de un trabajo con la predicción puntual de validación de cada época."""
    history = [dict(epoch=0, validation={})]
    for epoch, prediction in enumerate(predictions, 1):
        name = f"epochs/validation-epoch-{epoch:04d}.parquet"
        path = root / "runs" / job_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        assets = [f"US/A{i:02d}" for i in range(len(target) // len(times))]
        pq.write_table(
            pa.table(
                dict(
                    sample_id=[f"{a}/{t}" for t in times for a in assets],
                    market=["US"] * len(target),
                    prediction_at=pa.array(
                        [t for t in times for _ in assets], type=pa.timestamp("us", tz="UTC")
                    ),
                    target=target,
                    prediction=prediction,
                )
            ),
            path,
        )
        history.append(
            dict(
                epoch=epoch, validation=dict(epoch_predictions=dict(path=name, sha256=sha256(path)))
            )
        )
    atomic_json(root / "runs" / job_id / "fit" / "run.json", dict(history=history))


def test_a_seed_combination_chooses_its_epoch_with_its_own_ic(tmp_path):
    rng = np.random.default_rng(7)
    times = [1_546_380_000_000_000 + day * 86_400_000_000 for day in range(40)]
    assets = 24
    target = rng.standard_normal(len(times) * assets)
    common = rng.standard_normal(len(target))
    seeds = (42, 43, 44, 45)
    own = {seed: rng.standard_normal(len(target)) for seed in seeds}
    # En la época 2 el ruido es común a las semillas y en la 4 es independiente: cada semilla
    # sola ordena mejor en la 2, pero la media de cada terna ordena mejor en la 4.
    shape = {1: (1.5, 1.0), 2: (0.8, 1.0), 3: (1.5, 0.0), 4: (1.0, 0.0)}
    jobs = []
    for seed in seeds:
        predictions = [
            target + scale * (mix * common + np.sqrt(1 - mix**2) * own[seed])
            for scale, mix in (shape[epoch] for epoch in range(1, 5))
        ]
        job_id = f"2019-{seed}-source"
        write_epochs(tmp_path, job_id, predictions, target, times)
        jobs.append(dict(id=job_id, seed=seed, definition=dict(runner="fit")))
    protocol = dict(
        seeds=list(seeds),
        fit=dict(epochs=4),
        contrast=dict(primary=dict(minimum_assets=3)),
        selection=dict(smoothing_radius=0),
    )
    replicates = [list(item) for item in itertools.combinations(seeds, 3)]
    arm = dict(
        label="F2/seed_mean",
        definition=dict(runner="output", source="reference", replicates=replicates),
    )
    result = study.combination_epoch(tmp_path, jobs, arm, protocol)
    assert result["epoch"] == 4 and result["replicates"] == replicates
    assert result["model"] == "F2/seed_mean"
    assert max(range(4), key=lambda i: result["scores"][i]) == 3
    # La puntuación de cada época es la media de las ternas en las sesiones que todas definen.
    panels = {job["seed"]: study._epoch_panels(tmp_path, job, 4) for job in jobs}
    for epoch in range(4):
        items = []
        for replicate in replicates:
            total = panels[replicate[0]][epoch].prediction.copy()
            for seed in replicate[1:]:
                total += panels[seed][epoch].prediction
            panel = panels[replicate[0]][epoch]
            items.append(sd.session_correlations(panel, total / 3, min_assets=3))
        defined = np.logical_and.reduce([mask for _, mask in items])
        expected = np.mean([values for values, _ in items], axis=0)[defined].mean()
        assert result["scores"][epoch] == expected
    with pytest.raises(ValueError, match="todas sus semillas"):
        study.combination_epoch(tmp_path, jobs[:3], arm, protocol)
    with pytest.raises(ValueError, match="todas sus épocas"):
        study.combination_epoch(tmp_path, jobs, arm, dict(protocol, fit=dict(epochs=5)))
    (tmp_path / "runs" / jobs[0]["id"] / "epochs" / "validation-epoch-0002.parquet").write_bytes(
        b"PAR1 x PAR1"
    )
    with pytest.raises(ValueError, match="Ha cambiado"):
        study.combination_epoch(tmp_path, jobs, arm, protocol)


def test_a_group_adds_the_epoch_of_its_seed_combination_to_the_versions(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    times = [1_546_380_000_000_000 + day * 86_400_000_000 for day in range(30)]
    target = rng.standard_normal(len(times) * 12)
    common = rng.standard_normal(len(target))
    seeds = [42, 43, 44, 45, 46]
    own = {seed: rng.standard_normal(len(target)) for seed in seeds}
    # El mismo patrón que la prueba anterior: cada semilla sola ordena mejor en la época 2 y la
    # media de las tres semillas de la combinación ordena mejor en la 4.
    shape = {1: (1.5, 1.0), 2: (0.8, 1.0), 3: (1.5, 0.0), 4: (1.0, 0.0)}
    members = []
    for seed in seeds:
        job_id = f"2019-{seed}-reference"
        predictions = [
            target + scale * (mix * common + np.sqrt(1 - mix**2) * own[seed])
            for scale, mix in (shape[epoch] for epoch in range(1, 5))
        ]
        write_epochs(tmp_path, job_id, predictions, target, times)
        path = tmp_path / "runs" / job_id / "fit" / "run.json"
        report = json.loads(path.read_text())
        for entry in report["history"][1:]:
            record = entry["validation"]["epoch_predictions"]
            panel = ForecastPanel.from_arrow(pq.read_table(path.parents[1] / record["path"]))
            values, defined = sd.session_correlations(panel, panel.prediction, min_assets=3)
            entry["validation"].update(
                session_ic_values=values.tolist(),
                session_ic_defined=defined.tolist(),
                cohort_sha256=panel.cohort_sha256,
            )
        report.update(
            status=selection.AWAIT,
            epoch_states={str(epoch): dict(sha256="0" * 64) for epoch in range(1, 5)},
            selection=dict(best_epoch=4),
        )
        atomic_json(path, report)
        atomic_json(tmp_path / "runs" / job_id / "run.json", dict(status=selection.AWAIT))
        members.append(
            dict(
                id=job_id,
                seed=seed,
                common_group="2019-reference",
                definition=dict(runner="fit"),
                used_by=["F2/single_seed_reference"],
                versions=[3],
                epoch_predictions=True,
            )
        )
    arms = {
        ("F2", "single_seed_reference"): variant_screening.Arm(
            "F2", "single_seed_reference", dict(runner="fit")
        ),
        ("F2", "seed_mean"): variant_screening.Arm(
            "F2",
            "seed_mean",
            dict(
                runner="output",
                source="single_seed_reference",
                combine="seed_mean_v1",
                replicates=[list(item) for item in itertools.combinations(seeds, 3)],
            ),
        ),
    }
    monkeypatch.setattr(variant_screening, "all_arms", lambda *_: arms)
    protocol = dict(
        seeds=seeds,
        fit=dict(epochs=4),
        contrast=dict(primary=dict(minimum_assets=3)),
        selection=dict(smoothing_radius=0),
    )
    decision = study.decide_group(tmp_path, dict(protocol=protocol, jobs=members), "2019-reference")
    assert decision["epoch"] == 2
    [combination] = decision["combinations"]
    assert (combination["model"], combination["epoch"]) == ("F2/seed_mean", 4)
    assert len(combination["replicates"]) == 10
    # La época de la combinación se emite como una versión más del grupo.
    assert decision["versions"] == [3, 4]
    del arms[("F2", "single_seed_reference")]
    with pytest.raises(ValueError, match="no tiene fuente"):
        variant_screening.combination_sources(arms)


def window_table(times, assets, target, prediction, spread):
    """Tabla con el esquema de las predicciones de la ventana, con los cinco cuantiles."""
    rows = len(target)
    columns = dict(
        sample_id=[f"{a}/{at}" for at in times for a in assets],
        asset_id=[a for _ in times for a in assets],
        market=["US"] * rows,
        prediction_at=pa.array(
            [at for at in times for _ in assets], type=pa.timestamp("us", tz="UTC")
        ),
        target=pa.array(target, type=pa.float64()),
        prediction=pa.array(prediction, type=pa.float64()),
        zero=pa.array(np.zeros(rows), type=pa.float64()),
    )
    for offset, name in zip((-2, -1, 0, 1, 2), QUANTILE_COLUMNS, strict=True):
        columns[name] = pa.array(prediction + offset * spread, type=pa.float64())
    return pa.table(columns)


def test_each_replicate_of_the_seed_mean_reads_the_epoch_of_its_combination(tmp_path, monkeypatch):
    rng = np.random.default_rng(13)
    seeds = [42, 43, 44, 45, 46]
    assets = [f"US/A{i:02d}" for i in range(6)]
    times = {
        "calibration": [1_546_380_000_000_000 + day * 86_400_000_000 for day in range(5)],
        "evaluation": [1_547_380_000_000_000 + day * 86_400_000_000 for day in range(7)],
    }
    target = {name: rng.standard_normal(len(at) * len(assets)) for name, at in times.items()}
    emitted, jobs = {}, []
    for seed in seeds:
        job_id = f"2019-{seed}-reference"
        folder = tmp_path / "runs" / job_id
        records = {}
        for epoch, path in ((2, "predictions/{}.parquet"), (4, "versions/epoch-0004-{}.parquet")):
            records[epoch] = {}
            for name, at in times.items():
                point = rng.standard_normal(len(target[name]))
                emitted[seed, epoch, name] = point
                destination = folder / path.format(name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                spread = rng.random(len(point)) + 0.1
                pq.write_table(window_table(at, assets, target[name], point, spread), destination)
                records[epoch][name] = dict(path=path.format(name), sha256=sha256(destination))
        atomic_json(
            folder / "run.json",
            dict(
                status="completed",
                fit=dict(joint_stop_epoch=2),
                predictions=records[2],
                versions=dict(epochs=[4], predictions={"4": records[4]}),
            ),
        )
        jobs.append(
            dict(
                id=job_id,
                origin=2019,
                seed=seed,
                definition=dict(runner="fit"),
                used_by=["F2/single_seed_reference"],
                common_group="2019-reference",
            )
        )
    replicates = [list(item) for item in itertools.combinations(seeds, 3)]
    arms = {
        ("F2", "seed_mean"): variant_screening.Arm(
            "F2",
            "seed_mean",
            dict(
                runner="output",
                source="single_seed_reference",
                combine="seed_mean_v1",
                replicates=replicates,
            ),
        )
    }
    monkeypatch.setattr(variant_screening, "all_arms", lambda *_: arms)
    receipt = dict(protocol=dict(seeds=seeds), jobs=jobs)
    label = ("F2", "seed_mean")
    combine = study.combination_panels(tmp_path, receipt, 2019, {})
    with pytest.raises(ValueError, match="no ha decidido"):
        combine(label, (42, 43, 44))

    def decide(epoch, chosen=replicates):
        combination = dict(model="F2/seed_mean", epoch=epoch, replicates=chosen)
        atomic_json(
            tmp_path / "groups" / "2019-reference.json",
            dict(epoch=2, versions=[4], combinations=[combination]),
        )

    for epoch in (4, 2):
        decide(epoch)
        reads = {}
        panels = study.combination_panels(tmp_path, receipt, 2019, reads)(label, (43, 45, 46))
        for name in ("calibration", "evaluation"):
            # La media sigue el orden ascendente de semillas con la misma suma que seed_mean.
            total = emitted[43, epoch, name] + emitted[45, epoch, name] + emitted[46, epoch, name]
            assert np.array_equal(panels[name].prediction, total / 3)
            assert panels[name].levels == (0.025, 0.1, 0.5, 0.9, 0.975)
            assert np.all(np.diff(panels[name].quantiles, axis=1) >= 0)
        stem = "epoch-0004-{}" if epoch == 4 else "{}"
        assert set(reads) == {
            f"2019-{seed}-reference/{stem.format(name)}.parquet"
            for seed in (43, 45, 46)
            for name in times
        }
    decide(4, replicates[:5])
    with pytest.raises(ValueError, match="réplicas declaradas"):
        study.combination_panels(tmp_path, receipt, 2019, {})(label, (42, 43, 44))
    decide(3)
    with pytest.raises(ValueError, match="no emite la época 3"):
        study.combination_panels(tmp_path, receipt, 2019, {})(label, (42, 43, 44))
    decide(4)
    version = tmp_path / "runs" / "2019-44-reference" / "versions" / "epoch-0004-evaluation.parquet"
    version.write_bytes(b"PAR1 x PAR1")
    with pytest.raises(ValueError, match="Ha cambiado"):
        study.combination_panels(tmp_path, receipt, 2019, {})(label, (42, 43, 44))


def test_the_rule_check_reads_every_group_and_writes_its_report(
    prepared, common_runners, monkeypatch, tmp_path
):
    work, receipt = prepared
    members = [
        job(receipt, "F5/auxiliary_horizons", origin=2021, seed=seed)
        for seed in receipt["protocol"]["seeds"]
    ]
    monkeypatch.setattr(study, "verify_prepared", lambda _: dict(receipt, jobs=members))
    with pytest.raises(ValueError, match="no ha recorrido su presupuesto"):
        study.rule_check(work, tmp_path / "early")
    for item in members:
        study.run_job(work, item["id"])
    result = study.rule_check(work, tmp_path / "check")
    report = json.loads((tmp_path / "check" / "rule-check.json").read_text())
    assert result == dict(groups=1, summary=report["summary"])
    assert report["kind"] == study.RULE_CHECK_KIND and report["final_test_access"] is False
    assert report["optimizer_steps"] == 0
    assert set(report["reads"]) == {f"{item['id']}/fit/run.json" for item in members}
    assert list(report["groups"]) == [members[0]["common_group"]]
    (row,) = report["groups"].values()
    assert row["common_epoch"] == 5 and row["session_mae_best_epochs"] == [EPOCHS] * 5
    with pytest.raises(ValueError, match="directorio nuevo"):
        study.rule_check(work, tmp_path / "check")


def test_the_rule_check_checks_the_hold_before_reading_the_preparation(
    prepared, learning_hold, monkeypatch, tmp_path
):
    work, _ = prepared
    learning_hold(False)
    monkeypatch.setattr(
        study, "verify_prepared", lambda *_: pytest.fail("Se leyó la criba bajo la protección")
    )
    with pytest.raises(LearningHoldError):
        study.rule_check(work, tmp_path / "check")
    assert not (tmp_path / "check").exists()
