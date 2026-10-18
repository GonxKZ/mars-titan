"""Emparejamiento, semillas e interacciones de la comparación combinada."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks import combined_memory_analysis as analysis
from benchmarks.combined_memory_study import jobs
from benchmarks.memory_microstudy_analysis import score_table
from tests.tooling.test_memory_microstudy_analysis import table


def test_factorial_average_requires_every_declared_seed():
    series = score_table(table()).series("mae")
    inputs = {f"fold-018/cell0-{seed}": series for seed in (42, 43)}
    assert analysis.cell_means(inputs, "fold-018") == {}
    inputs["fold-018/cell0-44"] = series
    means = analysis.cell_means(inputs, "fold-018")
    assert list(means) == ["cell0-mean"]
    np.testing.assert_array_equal(means["cell0-mean"].values, series.values)
    assert len(means["cell0-mean"].values) == 2


def test_interactions_use_the_full_inclusion_exclusion_signs():
    assert analysis.factor_interaction(3) == {
        "cell0-mean": 1,
        "cell1-mean": -1,
        "cell2-mean": -1,
        "cell3-mean": 1,
    }
    coefficients = analysis.factor_interaction(7)
    assert len(coefficients) == 8 and sum(coefficients.values()) == 0
    assert coefficients["cell7-mean"] == 1 and coefficients["cell0-mean"] == -1
    effects = {f"cell{i}-mean": 10 + (i & 1) + 2 * bool(i & 2) + 3 * bool(i & 4) for i in range(8)}
    assert sum(coefficients[name] * value for name, value in effects.items()) == 0
    effects["cell7-mean"] -= 0.25
    assert sum(coefficients[name] * value for name, value in effects.items()) == -0.25


def test_average_factor_differences_separate_main_pair_and_triple_terms():
    values = {
        f"cell{i}-mean": 10.0 + bool(i & 1) + 2 * bool(i & 2) + 3 * bool(i & 4) + 0.5 * (i & 3 == 3)
        for i in range(8)
    }
    expected = {1: 1.25, 2: 2.25, 4: 3.0, 3: 0.5, 5: 0.0, 6: 0.0, 7: 0.0}
    for mask, value in expected.items():
        coefficients = analysis.averaged_factor_difference(mask)
        assert len(coefficients) == 8
        assert sum(coefficients.values()) == 0
        assert sum(coefficients[key] * v for key, v in values.items()) == value


@pytest.mark.parametrize("mask", [0, -1, 8, True, 1.0])
def test_average_factor_differences_reject_other_masks(mask):
    with pytest.raises(ValueError):
        analysis.averaged_factor_difference(mask)


def test_readers_use_the_matching_frozen_parent_and_cm_keeps_its_own_baseline():
    available = {j["id"]: None for j in jobs() if j["fold"] == "fold-018"}
    available["zero"] = None
    available.update({f"cell{i}-mean": None for i in range(8)})
    families = analysis.contrast_families(available, "fold-018")
    reader = "fold-018/cell7_m1_zero_refinement-42"
    assert families["readers"][reader + "_minus_parent"] == {
        reader: 1.0,
        "fold-018/cell7_frozen-42": -1.0,
    }
    assert families["cm"]["cm_interaction"] == {
        "fold-018/cm_v1_bcm-42": 1.0,
        "fold-018/cm_v1_bc-42": -1.0,
        "fold-018/cm_v1_bm-42": -1.0,
        "fold-018/cm_v1_b-42": 1.0,
    }
    assert families["factorial"]["joint_minus_without_1"] == {"cell7-mean": 1.0, "cell6-mean": -1.0}
    assert families["cm"]["cm_joint_minus_c"] == {
        "fold-018/cm_v1_bcm-42": 1.0,
        "fold-018/cm_v1_bc-42": -1.0,
    }
    assert all(0 < len(family) <= 64 for family in families.values())
    assert families["readers"]["fold-018/cell7_m2_zero_refinement-42_minus_m1"] == {
        "fold-018/cell7_m1_zero_refinement-42": -1.0,
        "fold-018/cell7_m2_zero_refinement-42": 1.0,
    }


def test_missing_parent_does_not_turn_into_an_unpaired_comparison():
    only = {"fold-018/cell0_m1_zero_refinement-42": None, "zero": None}
    families = analysis.contrast_families(only, "fold-018")
    assert "readers" not in families


@pytest.mark.parametrize("mask", [-1, 0, 1, 2, 4, 8, True])
def test_only_pair_and_triple_interactions_are_accepted(mask):
    with pytest.raises(ValueError):
        analysis.factor_interaction(mask)


def fixture_runs(root):
    schedule = [
        j for j in jobs() if j["fold"] == "fold-018" and j.get("bits") == 0 and j.get("factorial")
    ]
    source = table()
    times = np.array(["2023-01-03", "2023-01-03", "2023-01-04"], dtype="datetime64[us]")
    source = source.set_column(
        3, "prediction_at", pa.array(times, type=pa.timestamp("us", tz="UTC"))
    )
    source = source.append_column("zero", pa.array([0.0] * 3))
    for job in schedule:
        folder = root / "runs" / job["id"]
        folder.mkdir(parents=True)
        path = folder / "evaluation.parquet"
        pq.write_table(source, path)
        (folder / "run.json").write_text(
            json.dumps(
                dict(
                    status="completed",
                    final_test_opened=False,
                    predictions=dict(
                        evaluation=dict(path=path.name, sha256=analysis.sha256(path), rows=3)
                    ),
                )
            )
        )
    receipt = dict(jobs=schedule, cohorts={"fold-018": dict(counts=dict(evaluation=3))})
    (root / "prepared.json").write_text(json.dumps(receipt))
    (root / "measurements").mkdir()
    for i, job in enumerate(schedule):
        (root / f"measurements/{i}.json").write_text(
            json.dumps(
                dict(
                    job=job,
                    status="completed",
                    process_seconds=0.1,
                    prepared_sha256=analysis.sha256(root / "prepared.json"),
                )
            )
        )
    return receipt, source


def test_full_analysis_writes_sessions_and_keeps_seed_and_temporal_units_separate(
    tmp_path, monkeypatch
):
    work = tmp_path / "work"
    receipt, _ = fixture_runs(work)
    monkeypatch.setattr(analysis, "verify_prepared", lambda _: receipt)
    result = analysis.analyze(work, tmp_path / "analysis")
    models = result["windows"]["fold-018"]["models"]
    assert models["cell0-mean"]["seeds"] == [42, 43, 44]
    assert models["cell0-mean"]["sessions"] == 2
    assert models["cell0-mean"]["mae"] == 5.5
    assert models["zero"]["summary"]["point"]["mae"] == 5.5
    assert result["pending"] == []
    assert (tmp_path / "analysis/fold-018/zero-sessions.parquet").exists()
    with pytest.raises(FileExistsError):
        analysis.analyze(work, tmp_path / "analysis")


@pytest.mark.parametrize("fault", ["hash", "rows", "test", "year", "target"])
def test_result_integrity_rejects_drift_and_future_years(tmp_path, fault):
    work = tmp_path / "work"
    receipt, source = fixture_runs(work)
    job = receipt["jobs"][0]
    folder = work / "runs" / job["id"]
    manifest = folder / "run.json"
    report = json.loads(manifest.read_text())
    path = folder / "evaluation.parquet"
    cohort = score_table(source).cohort_sha256
    if fault == "hash":
        report["predictions"]["evaluation"]["sha256"] = "a" * 64
    elif fault == "rows":
        report["predictions"]["evaluation"]["rows"] = 4
    elif fault == "test":
        report["final_test_opened"] = True
    else:
        column = "prediction_at" if fault == "year" else "target"
        values = (
            pa.array(
                np.array(["2024-01-03"] * 3, dtype="datetime64[us]"),
                type=pa.timestamp("us", tz="UTC"),
            )
            if fault == "year"
            else pa.array([1.0, 2.0, 10.0])
        )
        altered = source.set_column(source.column_names.index(column), column, values)
        pq.write_table(altered, path)
        report["predictions"]["evaluation"]["sha256"] = analysis.sha256(path)
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        analysis.read_result(work, job, 3, None if fault == "year" else cohort)


def test_stratum_summary_covers_each_row_once_and_uses_session_mae():
    source = table().append_column("zero", pa.array([0.0] * 3))
    groups = {"long": ["A"], "short": ["B"], "absent": ["C"]}
    result = analysis.stratum_scores(source, groups)
    assert result["long"]["rows"] == 2 and result["long"]["mae"] == 5.5
    assert result["short"]["rows"] == 1 and result["short"]["mae"] == 1
    assert result["absent"] == {"rows": 0}
    assert sum(r["rows"] for r in result.values()) == len(source)
    with pytest.raises(ValueError):
        analysis.stratum_scores(source, {"long": ["A"]})
    with pytest.raises(ValueError):
        analysis.stratum_scores(source, {"long": ["A", "B"], "short": ["A"]})


def test_analysis_rejects_an_execution_bound_to_another_prepared_receipt(tmp_path, monkeypatch):
    work = tmp_path / "work"
    receipt, _ = fixture_runs(work)
    monkeypatch.setattr(analysis, "verify_prepared", lambda _: receipt)
    path = work / "measurements/0.json"
    record = json.loads(path.read_text())
    record["prepared_sha256"] = "a" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="preparación"):
        analysis.analyze(work, tmp_path / "out")


def test_completed_predictions_without_a_cost_receipt_are_rejected(tmp_path, monkeypatch):
    work = tmp_path / "work"
    receipt, _ = fixture_runs(work)
    monkeypatch.setattr(analysis, "verify_prepared", lambda _: receipt)
    (work / "measurements/0.json").unlink()
    with pytest.raises(ValueError, match="medición"):
        analysis.analyze(work, tmp_path / "out")


def test_audit_groups_are_bound_to_the_original_input_audit(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(dict(rows=[dict(symbol="A", folds={"fold-018": dict(stratum="n1f0_long")})]))
    )
    receipt = dict(audit_sha256=analysis.sha256(audit), cohorts={"fold-018": dict(assets=["A"])})
    assert analysis.audit_groups(audit, receipt) == {"fold-018": {"n1f0_long": ["US/A"]}}
    changed = json.loads(audit.read_text())
    changed["rows"][0]["folds"]["fold-018"]["stratum"] = "n0f0_long"
    audit.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="auditoría"):
        analysis.audit_groups(audit, receipt)


@pytest.mark.parametrize(
    "fault",
    [None, "optimizer_steps", "shared_parameters_unchanged", "checkpoint_sha256", "parent_sha256"],
)
def test_reference_is_bound_to_its_frozen_parent(tmp_path, fault):
    work = tmp_path / "work"
    receipt, source = fixture_runs(work)
    job = next(j for j in jobs() if j["id"] == "fold-018/cell0_frozen-42")
    parent = work / "runs" / job["parent"] / "run.json"
    parent_record = json.loads(parent.read_text())
    parent_record["checkpoint"] = dict(sha256="b" * 64)
    parent.write_text(json.dumps(parent_record))
    folder = work / "runs" / job["id"]
    folder.mkdir()
    path = folder / "evaluation.parquet"
    pq.write_table(source, path)
    record = dict(
        status="completed",
        final_test_opened=False,
        optimizer_steps=0,
        shared_parameters_unchanged=True,
        checkpoint_sha256="b" * 64,
        parent_sha256=analysis.sha256(parent),
        predictions=dict(evaluation=dict(path=path.name, sha256=analysis.sha256(path), rows=3)),
    )
    if fault:
        record[fault] = "incompatible"
    (folder / "run.json").write_text(json.dumps(record))
    if fault:
        with pytest.raises(ValueError, match="congelado"):
            analysis.read_result(work, job, 3, None)
    else:
        result, reason = analysis.read_result(work, job, 3, None)
        assert reason is None and result[0]["summary"]["point"]["mae"] == 5.5


def test_cli_passes_audit_and_paths_to_analysis(tmp_path, monkeypatch, capsys):
    import sys

    called = []

    def run(work, output, *, audit):
        called.append((work, output, audit))
        return dict(windows={}, pending=[])

    monkeypatch.setattr(analysis, "analyze", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analysis",
            "--work",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--audit",
            str(tmp_path / "audit.json"),
        ],
    )
    analysis.main()
    assert called == [(tmp_path, tmp_path / "out", tmp_path / "audit.json")]
    assert json.loads(capsys.readouterr().out) == dict(windows=0, pending=0)


def test_unfinished_jobs_are_reported_without_inventing_metrics(tmp_path, monkeypatch):
    work = tmp_path / "work"
    receipt, _ = fixture_runs(work)
    monkeypatch.setattr(analysis, "verify_prepared", lambda _: receipt)
    for job in receipt["jobs"]:
        path = work / "runs" / job["id"] / "run.json"
        path.write_text(json.dumps(dict(status="paused")))
    result = analysis.analyze(work, tmp_path / "out")
    assert result["windows"]["fold-018"]["status"] == "no_completed_jobs"
    assert len(result["pending"]) == 3
    assert all(r["reason"] == "paused" for r in result["pending"])
    job = receipt["jobs"][0]
    (work / "runs" / job["id"] / "run.json").unlink()
    assert analysis.read_result(work, job, 3, None) == (None, "Sin manifiesto")
