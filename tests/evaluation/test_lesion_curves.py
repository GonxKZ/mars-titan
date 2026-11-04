"""Curvas B4 con fixtures identificados. Ningún resultado es una evaluación financiera."""

import copy
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.memory.area_memory import AreaLayout
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.lesion_protocol import LesionProtocol

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source(tmp_path):
    protocol = LesionProtocol(seeds=(3,), fractions=((1, 2),))
    plan = protocol.bind(MatureCorrection(AssociativeMemoryConfig("proximal"), areas=AreaLayout()))
    rows = []
    for day in range(21):
        for market, size in (("US", 4), ("CN", 1 if day % 2 else 3)):
            for asset in range(size):
                target = (asset - 1) * 0.02 + day * 0.0001
                prediction = target + (0.01 if market == "CN" else 0.04)
                row = dict(
                    sample_id=f"{market}/{day}/{asset}",
                    market=market,
                    prediction_at=1_600_000_000_000_000 + day * 86_400_000_000,
                    target=target,
                    prediction=prediction,
                )
                for i, name in enumerate(plan.names):
                    row[f"lesion_{name}"] = prediction if i == 0 else target + 0.005 * i
                rows.append(row)
    table = pa.Table.from_pylist(rows).set_column(
        2, "prediction_at", pa.array([r["prediction_at"] for r in rows], pa.timestamp("us", "UTC"))
    )
    # El lector debe ordenar y reunir una sesión aunque cruce lotes y row groups.
    table = table.take(np.random.Generator(np.random.PCG64(9)).permutation(len(rows)))
    run = tmp_path / "run"
    run.mkdir()
    path = run / "evaluation-predictions.parquet"
    pq.write_table(table, path, row_group_size=7)
    config = json.loads((ROOT / "configs/experiments/lesion-protocol-v1.json").read_text())
    config["protocol"] = protocol.identity()
    config["analysis"].update(replicates=30, block_length=2, sensitivity_block_lengths=[1, 3])
    study = tmp_path / "study.json"
    study.write_text(json.dumps(config))
    record = dict(
        schema_version=1,
        kind="mars_titan_correction_window",
        status="completed",
        final_test_opened=False,
        request=dict(lesions=protocol.identity(), seed=7),
        predictions=dict(
            evaluation=dict(
                path=path.name,
                sha256=sha256(path),
                rows=len(table),
                bytes=path.stat().st_size,
                metrics=dict(lesions=dict(configuration=plan.identity())),
            )
        ),
    )
    report = run / "run.json"
    report.write_text(json.dumps(record))
    return dict(table=table, path=path, report=report, study=study, record=record, plan=plan)


def analyze(source, tmp_path, monkeypatch, **kwargs):
    from mars_titan.evaluation import lesion_curves as module

    monkeypatch.setattr(module, "require_learning_allowed", lambda *_: None)
    return module.analyze_lesion_run(
        source["report"],
        source["study"],
        partition="evaluation",
        output=tmp_path / "curves",
        **kwargs,
    )


def rewrite(source, *, table=None, record=None):
    record = copy.deepcopy(source["record"] if record is None else record)
    if table is not None:
        pq.write_table(table, source["path"], row_group_size=7)
        info = record["predictions"]["evaluation"]
        info.update(
            sha256=sha256(source["path"]), rows=len(table), bytes=source["path"].stat().st_size
        )
    source["report"].write_text(json.dumps(record))


@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_streaming_matches_session_scores_and_paired_delta(
    source, tmp_path, monkeypatch, batch_size
):
    report = analyze(source, tmp_path, monkeypatch, batch_size=batch_size)
    expected = score_sessions(ForecastPanel.from_arrow(source["table"])).summary()
    from mars_titan.evaluation.lesion_curves import POINT_ONLY

    expected["quantiles_reason"] = POINT_ONLY
    measured = dict(report["scores"]["intact"])
    measured.pop("cohort_sha256")
    expected.pop("cohort_sha256")
    assert measured == expected
    contrast = next(
        x for x in report["comparisons"]["mae"]["contrasts"] if x["name"] == "zero_minus_intact"
    )
    assert contrast["estimate"] == pytest.approx(0.005 - expected["point"]["mae"])
    assert contrast["relative_improvement_percent"] > 0
    assert report["comparisons"]["mae"]["resampling"]["unit"] == (
        "utc_calendar_day_with_all_sessions_and_assets"
    )
    assert report["comparisons"]["rank_ic"]["contrasts"]
    assert "residual_zero" in report["scores"]
    assert report["scores"]["residual_zero"]["point"]["rank_ic"] is None
    assert report["rows"] == len(source["table"])
    sessions = pq.read_table(tmp_path / "curves/sessions.parquet")
    assert len(sessions) == 42 * (len(source["plan"].names) + 1)
    assert sha256(tmp_path / "curves/sessions.parquet") == report["sessions"]["sha256"]
    assert json.loads((tmp_path / "curves/curves.json").read_text()) == report


@pytest.mark.parametrize("problem", ["future", "intact", "nan", "null", "duplicate", "extra"])
def test_bad_predictions_are_rejected_without_publishing_curves(
    source, tmp_path, monkeypatch, problem
):
    table = source["table"]
    if problem == "duplicate":
        table = pa.concat_tables([table, table.slice(0, 1)])
    elif problem == "extra":
        table = table.append_column("lesion_not_declared", pa.array(np.zeros(len(table))))
    else:
        name = "prediction_at" if problem == "future" else "lesion_intact"
        values = table[name].to_pylist()
        values[0] = {
            "future": pa.scalar(1_704_067_200_000_000, pa.timestamp("us", "UTC")).as_py(),
            "intact": 99.0,
            "nan": float("nan"),
            "null": None,
        }[problem]
        table = table.set_column(
            table.column_names.index(name), name, pa.array(values, type=table[name].type)
        )
    rewrite(source, table=table)
    with pytest.raises(ValueError):
        analyze(source, tmp_path, monkeypatch, batch_size=3)
    assert not (tmp_path / "curves/curves.json").exists()


@pytest.mark.parametrize("problem", ["sha", "rows", "paused", "final", "path", "plan"])
def test_report_integrity_precedes_scoring(source, tmp_path, monkeypatch, problem):
    record = copy.deepcopy(source["record"])
    info = record["predictions"]["evaluation"]
    if problem == "sha":
        info["sha256"] = "0" * 64
    elif problem == "rows":
        info["rows"] += 1
    elif problem == "paused":
        record["status"] = "paused"
    elif problem == "final":
        record["final_test_opened"] = True
    elif problem == "path":
        info["path"] = "../outside.parquet"
    else:
        info["metrics"]["lesions"]["configuration"]["cases"][0]["name"] = "invented"
    rewrite(source, record=record)
    with pytest.raises(ValueError):
        analyze(source, tmp_path, monkeypatch)
    assert not (tmp_path / "curves/curves.json").exists()


def test_hold_precedes_all_report_io_and_output(tmp_path, learning_hold):
    from mars_titan.evaluation.lesion_curves import analyze_lesion_run
    from mars_titan.training.learning_hold import LearningHoldError

    with pytest.raises(LearningHoldError):
        analyze_lesion_run(
            tmp_path / "missing",
            tmp_path / "missing-study",
            partition="evaluation",
            output=tmp_path / "curves",
        )
    assert not (tmp_path / "curves").exists()


def test_source_change_during_scoring_prevents_publication(source, tmp_path, monkeypatch):
    from mars_titan.evaluation import lesion_curves as module

    original = module._scores

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with source["path"].open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(module, "_scores", changed)
    with pytest.raises(ValueError, match="cambiado"):
        analyze(source, tmp_path, monkeypatch)
    assert not (tmp_path / "curves/curves.json").exists()


def test_zero_error_has_no_fabricated_relative_improvement(source, tmp_path, monkeypatch):
    table = source["table"]
    for name in ("prediction", "lesion_intact"):
        table = table.set_column(table.column_names.index(name), name, table["target"])
    rewrite(source, table=table)
    report = analyze(source, tmp_path, monkeypatch)
    contrast = report["comparisons"]["mae"]["contrasts"][0]
    assert contrast["base_estimate"] == 0
    assert contrast["relative_improvement_percent"] is None


def test_market_weighting_and_insufficient_periods_remain_explicit(source, tmp_path, monkeypatch):
    config = json.loads(source["study"].read_text())
    config["analysis"].update(
        market_weighting="market", block_length=32, sensitivity_block_lengths=[]
    )
    source["study"].write_text(json.dumps(config))
    report = analyze(source, tmp_path, monkeypatch)
    assert report["scores"]["intact"]["point"]["mae"] == pytest.approx(0.025)
    comparison = report["comparisons"]["mae"]
    assert comparison["resampling"]["reason"]
    assert all(c["interval"] is None for c in comparison["contrasts"])


def test_session_budget_stops_without_a_partial_completed_report(source, tmp_path, monkeypatch):
    from mars_titan.evaluation import lesion_curves as module

    monkeypatch.setattr(module, "MAX_SESSIONS", 1)
    with pytest.raises(ValueError, match="presupuesto de sesiones"):
        analyze(source, tmp_path, monkeypatch)
    assert not (tmp_path / "curves").exists()


def test_analysis_cli_writes_the_real_session_artifacts(source, tmp_path, monkeypatch, capsys):
    from mars_titan.evaluation import lesion_curves

    monkeypatch.setattr(lesion_curves, "require_learning_allowed", lambda *_: None)
    main = runpy.run_path(str(ROOT / "scripts/analyze_lesion_curves.py"), run_name="b4_cli")["main"]
    output = tmp_path / "cli-curves"
    assert (
        main(
            [
                "--report",
                str(source["report"]),
                "--study",
                str(source["study"]),
                "--partition",
                "evaluation",
                "--output",
                str(output),
                "--batch-size",
                "3",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == dict(
        status="completed", rows=len(source["table"]), output=str(output)
    )
    assert pq.read_table(output / "sessions.parquet").num_rows == 252


@pytest.mark.parametrize("changed", [False, True])
def test_frozen_analysis_is_traced_and_cannot_change_after_the_window(
    source, tmp_path, monkeypatch, changed
):
    from mars_titan.memory.lesion_study import freeze_study, load_study

    study, digest = load_study(source["study"])
    declared = tmp_path / "declared-run"
    freeze_study(
        declared,
        study=study,
        study_sha256=digest,
        arms_sha256="0" * 64,
        arm="fixture",
        seed=7,
        search_case="fixture",
        recipe=ROOT / "configs/titans/mature-correction-historical-masked.json",
    )
    for path in (source["report"], source["path"]):
        shutil.copy2(path, declared / path.name)
    source["report"] = declared / "run.json"
    if changed:
        config = json.loads(source["study"].read_text())
        config["analysis"]["replicates"] += 1
        source["study"].write_text(json.dumps(config))
        with pytest.raises(ValueError, match="declaración"):
            analyze(source, tmp_path, monkeypatch)
    else:
        report = analyze(source, tmp_path, monkeypatch)
        assert report["source"]["analysis_declaration_sha256"] == sha256(
            declared / "lesion-study.json"
        )


def test_carry_receipt_reuses_the_same_protocol_and_scores(source, tmp_path, monkeypatch):
    record = copy.deepcopy(source["record"])
    record["kind"] = "mars_titan_correction_carried_predictions"
    record["lesions"] = record["request"]["lesions"]
    record["anchor"] = dict(seed=record["request"]["seed"])
    del record["request"]
    rewrite(source, record=record)
    report = analyze(source, tmp_path, monkeypatch)
    assert report["source"]["seed"] == 7
    assert report["scores"]["intact"]["rows"] == len(source["table"])
