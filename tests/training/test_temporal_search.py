"""Búsqueda secuencial por ventanas, con reanudación y población identificada."""

import importlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from tests.training.test_temporal_corpus import inputs as inputs
from tests.training.test_temporal_corpus import prepare


def search_inputs(inputs, tmp_path):
    parent = json.loads(inputs[0].read_text())
    parent["context_sessions"] = 64
    samples = Path(parent["roots"]["samples"]) / "US/A0000/samples.parquet"
    rows = pq.read_table(samples).to_pylist()
    for row in rows:
        row["price_end_index"] = 63
    pq.write_table(pa.Table.from_pylist(rows), samples)
    original = Path(parent["roots"]["prepared"]) / "US/A0000/prices.parquet"
    table = pq.read_table(original)
    prices = table.to_pylist()
    from datetime import timedelta

    first = prices[0]
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(first, available_at=first["available_at"] + timedelta(days=i))
                for i in range(64)
            ],
            schema=table.schema,
        ),
        original,
    )
    parent["assets"][0].update(samples_sha256=sha256(samples), prices_sha256=sha256(original))
    inputs[0].write_text(json.dumps(parent))
    views = tmp_path / "views"
    prepare(inputs, views)
    # El estudio técnico usa solo la primera ventana, cuyo resto se comprueba por separado.
    config = tmp_path / "search.json"
    plan = json.loads(Path("configs/baselines/strict-temporal-search-us.json").read_text())
    plan.update(
        scope="development_snapshot",
        models=["rnn"],
        finalist_seeds=[42],
        max_epochs=2,
        patience=1,
        batch_size=2,
        posttraining_epochs=1,
        continuation_selection=dict(metric="session_mae", patience=1, min_delta=100.0),
    )
    config.write_text(json.dumps(plan))
    return config, views


def test_strict_search_runs_prespecified_candidates_and_selects_parent_for_paired_controls(
    inputs, tmp_path
):
    config, views = search_inputs(inputs, tmp_path)
    engine = importlib.import_module("mars_titan.training.reference_search")
    events = []
    report = engine.run_search(
        config,
        views / "fold-000/manifest.json",
        tmp_path / "study",
        progress=lambda status: events.append(status["completed_runs"]),
    )
    assert report["planned_runs"] == report["completed_runs"] == 4
    assert events[-1] == 4
    for run in report["runs"]:
        if run["stage"] == "posttraining":
            detail = json.loads((tmp_path / "study" / run["path"] / "run.json").read_text())
            assert detail["selection"]["best_epoch"] == 0
            assert len(detail["epochs"]) == 1


def orchestration(tmp_path, monkeypatch):
    from types import SimpleNamespace

    engine = importlib.import_module("mars_titan.training.temporal_search")
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "config.json"
    config.write_text("{}")
    views = tmp_path / "views"
    views.mkdir()
    (views / "report.json").write_text("{}")
    records = []
    for index in range(2):
        path = views / f"fold-{index:03}.json"
        path.write_text("{}")
        records.append(dict(id=f"fold-{index:03}", manifest=path))
    identity = dict(config_sha256=sha256(config), views_report_sha256=sha256(views / "report.json"))
    monkeypatch.setattr(engine, "_inputs", lambda c, v: (records, identity, 4))

    class Lease:
        def __enter__(self):
            return SimpleNamespace(record={"device": "prueba"}, check=lambda: None)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(engine, "GpuLease", Lease)
    return engine, config, views


def test_controller_updates_progress_and_resumes_each_fold(inputs, tmp_path, monkeypatch):
    engine, config, views = orchestration(tmp_path / "controller", monkeypatch)
    calls = []

    def study(config, manifest, folder, *, resume, progress):
        calls.append((folder.name, resume))
        folder.mkdir(exist_ok=True)
        result = dict(
            status="paused" if len(calls) == 1 else "completed",
            planned_runs=4,
            completed_runs=1 if len(calls) == 1 else 4,
        )
        progress(result)
        return result

    monkeypatch.setattr(engine, "run_search", study)
    output = tmp_path / "study"
    first = engine.run_temporal_search(config, views, output)
    assert first["status"] == "paused" and first["completed_runs"] == 1
    second = engine.run_temporal_search(config, views, output, resume=True)
    assert second["status"] == "completed" and second["completed_runs"] == 8
    assert calls == [("fold-000", False), ("fold-000", True), ("fold-001", False)]


def test_controller_rejects_changed_identity_without_overwriting_completed_evidence(
    tmp_path, monkeypatch
):
    engine, config, views = orchestration(tmp_path, monkeypatch)

    def study(config, manifest, folder, *, resume, progress):
        folder.mkdir(exist_ok=True)
        result = dict(status="completed", planned_runs=4, completed_runs=4)
        progress(result)
        return result

    monkeypatch.setattr(engine, "run_search", study)
    output = tmp_path / "study"
    engine.run_temporal_search(config, views, output)
    path = output / "summary.json"
    value = json.loads(path.read_text())
    value["identity"]["config_sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        engine.run_temporal_search(config, views, output, resume=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "field,value", [("planned_runs", 999), ("completed_runs", -1), ("status", "unknown")]
)
def test_controller_rejects_inconsistent_fold_progress(tmp_path, monkeypatch, field, value):
    engine, config, views = orchestration(tmp_path, monkeypatch)

    def study(config, manifest, folder, *, resume, progress):
        folder.mkdir(exist_ok=True)
        result = dict(status="completed", planned_runs=4, completed_runs=4)
        progress(result)
        return result

    monkeypatch.setattr(engine, "run_search", study)
    output = tmp_path / "study"
    engine.run_temporal_search(config, views, output)
    path = output / "summary.json"
    data = json.loads(path.read_text())
    data["folds"][0][field] = value
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        engine.run_temporal_search(config, views, output, resume=True)
    assert path.read_bytes() == before


def test_preflight_rejects_missing_provenance_before_reading_fold_files(tmp_path):
    engine = importlib.import_module("mars_titan.training.temporal_search")
    views = tmp_path / "views"
    views.mkdir()
    (views / "report.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                status="temporal_views_prepared",
                final_test_opened=False,
                folds=[dict(id="fold-000", has_all_partitions=True)],
            )
        )
    )
    with pytest.raises(ValueError, match="procedencia"):
        engine._inputs(Path("configs/baselines/strict-temporal-search-us.json"), views)


def test_controller_recovers_a_cut_before_first_summary(tmp_path, monkeypatch):
    engine, config, views = orchestration(tmp_path, monkeypatch)

    def study(config, manifest, folder, *, resume, progress):
        folder.mkdir(exist_ok=True)
        result = dict(status="completed", planned_runs=4, completed_runs=4)
        progress(result)
        return result

    monkeypatch.setattr(engine, "run_search", study)
    output = tmp_path / "study"
    output.mkdir()
    (output / ".lock").write_text("")
    report = engine.run_temporal_search(config, views, output, resume=True)
    assert report["status"] == "completed"


def test_controller_does_not_adopt_unknown_files_without_a_summary(tmp_path, monkeypatch):
    engine, config, views = orchestration(tmp_path, monkeypatch)
    output = tmp_path / "study"
    output.mkdir()
    (output / "unrelated.json").write_text("{}")
    with pytest.raises(ValueError):
        engine.run_temporal_search(config, views, output, resume=True)
    assert not (output / "summary.json").exists()


def metadata_views(tmp_path):
    from mars_titan.evaluation.splits import build_folds

    protocol = json.loads(Path("configs/evaluation/strict-macro-walk-forward.json").read_text())
    values = dict(parent_sha256="a" * 64, macro_sha256="b" * 64, admission_sha256="c" * 64)
    root = tmp_path / "views"
    report = dict(
        schema_version=1,
        status="temporal_views_prepared",
        final_test_opened=False,
        protocol_sha256="d" * 64,
        **values,
        folds=[],
    )
    counts = dict(train=10, validation=4, calibration=3, evaluation=2)
    for fold in build_folds(protocol):
        path = root / fold["id"] / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                dict(
                    temporal_view=dict(protocol=protocol, fold=fold, **values),
                    final_test_opened=False,
                    counts=counts,
                )
            )
        )
        report["folds"].append(
            dict(
                id=fold["id"], counts=counts, has_all_partitions=True, manifest_sha256=sha256(path)
            )
        )
    (root / "report.json").write_text(json.dumps(report))
    return root


def test_preflight_reconciles_all_windows_and_derives_total_budget(tmp_path):
    engine = importlib.import_module("mars_titan.training.temporal_search")
    views = metadata_views(tmp_path)
    records, identity, count = engine._inputs(
        Path("configs/baselines/strict-temporal-search-us.json"), views
    )
    assert len(records) == 4 and count == 40
    assert len(identity["manifests"]) == 4


@pytest.mark.parametrize(
    "fault",
    [
        "missing_fold",
        "duplicate",
        "changed_file",
        "different_protocol",
        "test_opened",
        "empty_partition",
    ],
)
def test_preflight_rejects_inconsistent_temporal_evidence(tmp_path, fault):
    engine = importlib.import_module("mars_titan.training.temporal_search")
    views = metadata_views(tmp_path)
    path = views / "report.json"
    report = json.loads(path.read_text())
    if fault == "missing_fold":
        report["folds"].pop()
    elif fault == "duplicate":
        report["folds"][0]["id"] = "fold-001"
    else:
        target = views / "fold-001/manifest.json"
        meta = json.loads(target.read_text())
        if fault == "changed_file":
            meta["unexpected"] = True
        elif fault == "different_protocol":
            meta["temporal_view"]["protocol"]["seeds"] = [44]
        elif fault == "test_opened":
            meta["final_test_opened"] = True
        else:
            meta["counts"]["evaluation"] = 0
            report["folds"][1]["counts"] = meta["counts"]
        target.write_text(json.dumps(meta))
        if fault != "changed_file":
            report["folds"][1]["manifest_sha256"] = sha256(target)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        engine._inputs(Path("configs/baselines/strict-temporal-search-us.json"), views)
