"""Comparaciones E2 con errores manuales conocidos. No son resultados de un modelo."""

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.learning_hold import LearningHoldError
from mars_titan.training.spaced_window import _digest
from tests.evaluation.spaced_run_fixture import manual_runs, rewrite

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source(tmp_path):
    return manual_runs(tmp_path / "manual-e2")


def analyze(source, tmp_path, monkeypatch, **kwargs):
    from mars_titan.evaluation import spaced_comparison as module

    assert all(json.loads(p.read_text())["technical_fixture"] for p in source["reports"].values())
    monkeypatch.setattr(module, "require_learning_allowed", lambda *_: None)
    return module.analyze_spaced_runs(
        source["reports"],
        source["study"],
        partition="evaluation",
        output=tmp_path / "comparison",
        **kwargs,
    )


def test_primary_difference_uses_sessions_and_keeps_its_relative_sign(
    source, tmp_path, monkeypatch
):
    report = analyze(source, tmp_path, monkeypatch)
    assert report["scores"]["baseline"]["point"]["mae"] == pytest.approx(0.025)
    assert report["scores"]["utility"]["point"]["mae"] == pytest.approx(0.0125)
    primary = report["primary"]["comparison"]["contrasts"][0]
    assert primary["name"] == "utility_minus_baseline"
    assert primary["estimate"] == pytest.approx(-0.0125)
    assert primary["relative_improvement_percent"] == pytest.approx(50.0)
    assert primary["sessions"] == 42
    assert report["primary"]["assessment"] == "menor_error_en_esta_ventana"
    assert (
        report["comparisons"]["mae"]["resampling"]["unit"]
        == "utc_calendar_day_with_all_sessions_and_assets"
    )
    assert report["comparisons"]["mae"]["multiplicity"]["family_size"] == 10
    assert report["scores"]["residual_zero"]["quantiles"] is None
    sessions = pq.read_table(tmp_path / "comparison/sessions.parquet")
    assert len(sessions) == 7 * 42
    assert sha256(tmp_path / "comparison/sessions.parquet") == report["sessions"]["sha256"]
    assert json.loads((tmp_path / "comparison/comparison.json").read_text()) == report


def test_hold_precedes_every_report_and_output_access(tmp_path, learning_hold):
    from mars_titan.evaluation.spaced_comparison import analyze_spaced_runs

    with pytest.raises(LearningHoldError):
        analyze_spaced_runs(
            {"baseline": tmp_path / "missing"},
            tmp_path / "missing-study",
            partition="evaluation",
            output=tmp_path / "comparison",
        )
    assert not (tmp_path / "comparison").exists()


@pytest.mark.parametrize("field", ["target", "sample_id", "market", "prediction_at"])
def test_equal_row_counts_do_not_allow_different_populations(source, tmp_path, monkeypatch, field):
    path = source["reports"]["utility"].parent / "evaluation-predictions.parquet"
    table = pq.read_table(path)
    values = table[field].to_pylist()
    if field == "target":
        values[0] += 0.0001
    elif field == "sample_id":
        values[0] += "/other"
    elif field == "market":
        values[0] = "CN" if values[0] == "US" else "US"
    else:
        values[0] = pa.scalar(
            int(table[field].cast(pa.int64())[0].as_py()) + 1, pa.timestamp("us", "UTC")
        ).as_py()
    table = table.set_column(
        table.column_names.index(field), field, pa.array(values, type=table[field].type)
    )
    rewrite(source, "utility", table=table)
    with pytest.raises(ValueError):
        analyze(source, tmp_path, monkeypatch)
    assert not (tmp_path / "comparison/comparison.json").exists()


@pytest.mark.parametrize(
    "problem",
    [
        "reader",
        "view",
        "indices",
        "phases",
        "code",
        "native",
        "reset",
        "policy",
        "run_id",
        "final",
        "status",
    ],
)
def test_incompatible_run_headers_are_rejected(source, tmp_path, monkeypatch, problem):
    path = source["reports"]["utility"]
    report = json.loads(path.read_text())
    identity = report["identity"]
    if problem == "reader":
        identity["selected"]["readout_sha256"] = "0" * 64
    elif problem == "view":
        identity["request"]["view_sha256"] = "0" * 64
        report["request"] = copy.deepcopy(identity["request"])
    elif problem == "indices":
        identity["indices"]["evaluation"] = "0" * 64
    elif problem == "phases":
        identity["phases"]["evaluation"]["decision_start"] += 1
    elif problem == "code":
        identity["request"]["code"]["manual_wrong_module"] = "0" * 64
        report["request"] = copy.deepcopy(identity["request"])
    elif problem == "native":
        identity["request"]["native_sha256"] = "0" * 64
        report["request"] = copy.deepcopy(identity["request"])
    elif problem == "reset":
        identity["state_reset"] = "carry_future_memory"
    elif problem == "policy":
        identity["retention"]["policy"] = "recent"
    elif problem == "final":
        report["final_test_opened"] = True
    elif problem == "status":
        report["status"] = "paused"
    report["run_id"] = "0" * 64 if problem == "run_id" else _digest(identity)
    rewrite(source, "utility", record=report)
    with pytest.raises(ValueError):
        analyze(source, tmp_path, monkeypatch)
    assert not (tmp_path / "comparison").exists()


@pytest.mark.parametrize(
    "problem",
    [
        "sha",
        "bytes",
        "rows",
        "session_metric",
        "stored_session",
        "future",
        "outside_phase",
        "missing_control",
        "duplicate_run",
        "quantile_subset",
    ],
)
def test_artifact_or_control_mismatch_never_publishes_a_comparison(
    source, tmp_path, monkeypatch, problem
):
    path = source["reports"]["utility"]
    report = json.loads(path.read_text())
    info = report["predictions"]["evaluation"]
    if problem in ("sha", "bytes", "rows"):
        key = {"sha": "sha256", "bytes": "bytes", "rows": "rows"}[problem]
        info[key] = "0" * 64 if problem == "sha" else info[key] + 1
    elif problem == "session_metric":
        info["metrics"]["session_mae"] += 0.01
    elif problem == "stored_session":
        file = path.parent / info["sessions"]["path"]
        table = pq.read_table(file)
        values = table["mae"].to_numpy().copy()
        values[0] += 0.001
        pq.write_table(
            table.set_column(table.column_names.index("mae"), "mae", pa.array(values)), file
        )
        info["sessions"]["sha256"] = sha256(file)
    elif problem in ("future", "outside_phase", "quantile_subset"):
        table = pq.read_table(path.parent / info["path"])
        if problem == "quantile_subset":
            table = table.append_column("quantile_0500", table["prediction"])
        else:
            values = table["prediction_at"].cast(pa.int64()).to_numpy().copy()
            values[0] = 1_704_067_200_000_000 if problem == "future" else values.min() - 1
            table = table.set_column(
                table.column_names.index("prediction_at"),
                "prediction_at",
                pa.array(values, pa.timestamp("us", "UTC")),
            )
        rewrite(source, "utility", table=table)
        report = json.loads(path.read_text())
    elif problem == "missing_control":
        source["reports"].pop("recent")
    else:
        source["reports"]["recent"] = source["reports"]["utility"]
    rewrite(source, "utility", record=report)
    with pytest.raises(ValueError):
        analyze(source, tmp_path, monkeypatch)
    assert not (tmp_path / "comparison/comparison.json").exists()


def test_changed_analysis_cannot_be_attached_retrospectively(source, tmp_path, monkeypatch):
    document = json.loads(source["study"].read_text())
    document["analysis"]["replicates"] += 1
    atomic_json(source["study"], document)
    with pytest.raises(ValueError, match="declaración|estudio"):
        analyze(source, tmp_path, monkeypatch)
    assert not (tmp_path / "comparison").exists()


def test_zero_base_never_fabricates_a_relative_percentage(source, tmp_path, monkeypatch):
    path = source["reports"]["baseline"]
    report = json.loads(path.read_text())
    for part, info in report["predictions"].items():
        table = pq.read_table(path.parent / info["path"])
        table = table.set_column(
            table.column_names.index("prediction"), "prediction", table["target"]
        )
        rewrite(source, "baseline", table=table, partition=part)
    result = analyze(source, tmp_path, monkeypatch)
    primary = result["primary"]["comparison"]["contrasts"][0]
    assert primary["base_estimate"] == 0
    assert primary["relative_improvement_percent"] is None
    assert result["primary"]["assessment"] == "mayor_error_en_esta_ventana"


@pytest.mark.parametrize("contract", ["standard", "regional"])
def test_existing_quantiles_are_kept_and_zero_has_no_fake_head(tmp_path, monkeypatch, contract):
    source = manual_runs(tmp_path / "quantile-fixture", contract=contract, quantiles=True)
    report = analyze(source, tmp_path, monkeypatch)
    assert report["scores"]["utility"]["quantiles"]["levels"] == [0.025, 0.1, 0.5, 0.9, 0.975]
    assert report["scores"]["residual_zero"]["quantiles"] is None
    assert "pinball" in report["comparisons"] and "sign_brier" in report["comparisons"]
    assert report["reader_contract"] == contract


def test_batch_size_and_mapping_order_do_not_change_the_comparisons(source, tmp_path, monkeypatch):
    from mars_titan.evaluation import spaced_comparison as module

    monkeypatch.setattr(module, "require_learning_allowed", lambda *_: None)
    first = module.analyze_spaced_runs(
        source["reports"],
        source["study"],
        partition="evaluation",
        output=tmp_path / "one",
        batch_size=1,
    )
    second = module.analyze_spaced_runs(
        dict(reversed(tuple(source["reports"].items()))),
        source["study"],
        partition="evaluation",
        output=tmp_path / "two",
        batch_size=31,
    )
    assert first["scores"] == second["scores"]
    assert first["comparisons"] == second["comparisons"]
    assert first["primary"] == second["primary"]
    assert pq.read_table(tmp_path / "one/sessions.parquet").equals(
        pq.read_table(tmp_path / "two/sessions.parquet")
    )


def test_source_change_during_scoring_prevents_publication(source, tmp_path, monkeypatch):
    from mars_titan.evaluation import spaced_comparison as module

    original = module._score_run

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with source["reports"]["baseline"].open("a") as stream:
            stream.write(" ")
        return result

    monkeypatch.setattr(module, "_score_run", changed)
    with pytest.raises(ValueError, match="cambi"):
        analyze(source, tmp_path, monkeypatch)
    assert not (tmp_path / "comparison").exists()


@pytest.mark.parametrize("partition", ["train", "test", "final_test", "unknown"])
def test_unavailable_partitions_are_rejected(source, tmp_path, monkeypatch, partition):
    from mars_titan.evaluation import spaced_comparison as module

    monkeypatch.setattr(module, "require_learning_allowed", lambda *_: None)
    with pytest.raises(ValueError):
        module.analyze_spaced_runs(
            source["reports"], source["study"], partition=partition, output=tmp_path / "comparison"
        )
    assert not (tmp_path / "comparison").exists()


def test_output_cannot_overwrite_or_sit_inside_a_source(source, tmp_path, monkeypatch):
    from mars_titan.evaluation import spaced_comparison as module

    monkeypatch.setattr(module, "require_learning_allowed", lambda *_: None)
    before = source["reports"]["baseline"].read_bytes()
    with pytest.raises(ValueError):
        module.analyze_spaced_runs(
            source["reports"],
            source["study"],
            partition="evaluation",
            output=source["reports"]["baseline"].parent / "comparison",
        )
    assert source["reports"]["baseline"].read_bytes() == before
    report = analyze(source, tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        analyze(source, tmp_path, monkeypatch)
    assert json.loads((tmp_path / "comparison/comparison.json").read_text()) == report


def test_cli_emits_the_real_comparison_artifacts(source, tmp_path, monkeypatch, capsys):
    from mars_titan.evaluation import spaced_comparison as module

    monkeypatch.setattr(module, "require_learning_allowed", lambda *_: None)
    path = ROOT / "scripts/analyze_spaced_windows.py"
    assert path.is_file(), "Falta la orden de comparación E2"
    main = runpy.run_path(str(path), run_name="spaced_comparison_fixture")["main"]
    args = [
        "--study",
        str(source["study"]),
        "--partition",
        "evaluation",
        "--output",
        str(tmp_path / "comparison"),
    ]
    for arm, report in source["reports"].items():
        args += ["--run", f"{arm}={report}"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert (tmp_path / "comparison/sessions.parquet").is_file()


def test_one_day_reports_insufficient_time_instead_of_a_fake_interval(tmp_path, monkeypatch):
    source = manual_runs(tmp_path / "one-day", days=1)
    report = analyze(source, tmp_path, monkeypatch)
    assert report["primary"]["assessment"] == "inconcluyente"
    primary = report["primary"]["comparison"]
    assert primary["resampling"]["reason"]
    assert primary["contrasts"][0]["interval"] is None


def test_market_weighting_uses_market_means_with_unequal_session_counts(tmp_path, monkeypatch):
    source = manual_runs(tmp_path / "sparse-cn", sparse_cn=True, weighting="market")
    report = analyze(source, tmp_path, monkeypatch)
    assert report["scores"]["baseline"]["sessions"] == 32
    assert report["scores"]["baseline"]["point"]["mae"] == pytest.approx(0.025)
    assert report["primary"]["comparison"]["contrasts"][0]["estimate"] == pytest.approx(-0.0125)


def test_identical_points_produce_an_interval_inside_the_declared_margin(
    source, tmp_path, monkeypatch
):
    baseline = source["reports"]["baseline"]
    report = json.loads(baseline.read_text())
    for part, info in report["predictions"].items():
        rewrite(
            source, "utility", table=pq.read_table(baseline.parent / info["path"]), partition=part
        )
    result = analyze(source, tmp_path, monkeypatch)
    assert result["primary"]["assessment"] == "intervalo_dentro_del_margen"
    assert result["primary"]["comparison"]["contrasts"][0]["estimate"] == 0


def test_bootstrap_does_not_consume_global_rng(source, tmp_path, monkeypatch):
    import random

    import torch

    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    analyze(source, tmp_path, monkeypatch)
    assert random.getstate() == python
    after = np.random.get_state()
    assert after[0] == numpy[0] and after[2:] == numpy[2:]
    np.testing.assert_array_equal(after[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), cpu)


@pytest.mark.parametrize(
    "entries", [["baseline=one", "baseline=two"], ["missing_separator"], ["unknown=one"]]
)
def test_cli_rejects_duplicate_or_unknown_controls_before_analysis(tmp_path, entries, monkeypatch):
    path = ROOT / "scripts/analyze_spaced_windows.py"
    assert path.is_file(), "Falta la orden de comparación E2"
    main = runpy.run_path(str(path), run_name="spaced_invalid_cli")["main"]
    monkeypatch.setitem(
        main.__globals__,
        "analyze_spaced_runs",
        lambda *_a, **_k: pytest.fail("El analizador recibió controles inválidos"),
    )
    args = [
        "--study",
        str(tmp_path / "missing"),
        "--partition",
        "evaluation",
        "--output",
        str(tmp_path / "comparison"),
    ]
    for entry in entries:
        args += ["--run", entry]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert not (tmp_path / "comparison").exists()


def test_selected_descriptor_does_not_accept_a_boolean_version(source, tmp_path, monkeypatch):
    for arm, path in source["reports"].items():
        record = json.loads(path.read_text())
        record["identity"]["selected"]["schema_version"] = True
        record["run_id"] = _digest(record["identity"])
        rewrite(source, arm, record=record)
    with pytest.raises(ValueError):
        analyze(source, tmp_path, monkeypatch)
    assert not (tmp_path / "comparison").exists()


def test_future_phase_metadata_is_refused_before_artifact_verification(
    source, tmp_path, monkeypatch
):
    from mars_titan.evaluation import spaced_sources

    for arm, path in source["reports"].items():
        record = json.loads(path.read_text())
        phase = record["identity"]["phases"]["evaluation"]
        phase["decision_end"] = phase["close_at"] = 1_704_067_200_000_001
        record["run_id"] = _digest(record["identity"])
        rewrite(source, arm, record=record)
    monkeypatch.setattr(
        spaced_sources,
        "_verify_outputs",
        lambda *_: pytest.fail("Abrió artefactos de una fase futura"),
    )
    with pytest.raises(ValueError, match="fronteras"):
        analyze(source, tmp_path, monkeypatch)
    assert not (tmp_path / "comparison").exists()


def test_index_identities_cannot_be_missing_in_all_controls(source, tmp_path, monkeypatch):
    for arm, path in source["reports"].items():
        record = json.loads(path.read_text())
        record["identity"]["indices"]["evaluation"] = None
        record["run_id"] = _digest(record["identity"])
        rewrite(source, arm, record=record)
    with pytest.raises(ValueError):
        analyze(source, tmp_path, monkeypatch)
    assert not (tmp_path / "comparison").exists()


def test_all_controls_outside_the_phase_are_rejected_even_with_a_shared_population(
    source, tmp_path, monkeypatch
):
    for arm, path in source["reports"].items():
        record = json.loads(path.read_text())
        table = pq.read_table(path.parent / record["predictions"]["evaluation"]["path"])
        values = table["prediction_at"].cast(pa.int64()).to_numpy().copy()
        values[0] = record["identity"]["phases"]["evaluation"]["decision_start"] - 1
        table = table.set_column(
            table.column_names.index("prediction_at"),
            "prediction_at",
            pa.array(values, pa.timestamp("us", "UTC")),
        )
        rewrite(source, arm, table=table)
    with pytest.raises(ValueError, match="fase"):
        analyze(source, tmp_path, monkeypatch)
    assert not (tmp_path / "comparison").exists()


def test_a_smaller_error_within_the_margin_is_not_called_a_relevant_improvement(
    source, tmp_path, monkeypatch
):
    baseline = source["reports"]["baseline"]
    record = json.loads(baseline.read_text())
    for part, info in record["predictions"].items():
        table = pq.read_table(baseline.parent / info["path"])
        values = table["prediction"].to_numpy() - 0.000005
        table = table.set_column(
            table.column_names.index("prediction"), "prediction", pa.array(values)
        )
        rewrite(source, "utility", table=table, partition=part)
    report = analyze(source, tmp_path, monkeypatch)
    assert report["primary"]["comparison"]["contrasts"][0]["estimate"] == pytest.approx(-0.000005)
    assert report["primary"]["assessment"] == "intervalo_dentro_del_margen"
