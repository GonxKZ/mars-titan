"""Distinguir el promedio por ventana del promedio de ejecuciones."""

import pytest

from mars_titan.posttraining.analysis import aggregate, analyse_completion


def test_aggregate_weights_folds_equally_without_pseudoreplicating_seeds():
    common = dict(stage="posttraining", family="gru", method="mae", partition="evaluation")
    rows = [
        dict(common, fold="fold-000", session_mae=1.0, delta_parent=-0.1),
        dict(common, fold="fold-001", session_mae=3.0, delta_parent=0.1),
        dict(common, fold="fold-001", session_mae=5.0, delta_parent=0.3),
    ]
    result = aggregate(rows)[0]
    assert result["mean_fold_session_mae"] == 2.5
    assert result["mean_fold_delta_parent"] == pytest.approx(0.05)
    assert result["folds"] == 2
    assert result["runs"] == 3


def test_analysis_refuses_an_unfinished_campaign(tmp_path):
    from mars_titan.data.storage import atomic_json

    atomic_json(
        tmp_path / "summary.json",
        dict(
            final_test_opened=False,
            stages=[dict(status="paused", completed_runs=1, planned_runs=2)],
        ),
    )
    with pytest.raises(ValueError, match="completas"):
        analyse_completion(tmp_path)
    assert not (tmp_path / "analysis.json").exists()


def test_completed_analysis_preserves_seed_metrics_and_artifact_provenance(tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from mars_titan.data.storage import atomic_json, sha256

    folder = tmp_path / "fold-000/evaluation"
    run_path = folder / "runs/posttraining/gru/run.json"
    run_path.parent.mkdir(parents=True)
    metrics = {
        "prediction": dict(samples=1, session_mae=0.2, session_mse=0.04),
        "parent": dict(session_mae=0.3),
        "zero": dict(session_mae=0.4),
    }
    predictions = {}
    for partition in ("calibration", "evaluation"):
        path = run_path.parent / f"{partition}.parquet"
        pq.write_table(pa.table(dict(target=[0.4], prediction=[0.2])), path)
        predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
    atomic_json(
        run_path,
        dict(
            status="completed",
            final_test_opened=False,
            case=dict(mode="mae", seed=43),
            job=dict(stage="posttraining", phase="posttraining"),
            family="gru",
            predictions=predictions,
            parent_comparison=True,
            primary="median",
            selection=dict(best_epoch=0),
        ),
    )
    atomic_json(
        folder / "summary.json",
        dict(
            status="completed",
            final_test_opened=False,
            runs={
                "posttraining/gru": dict(
                    path=str(run_path.relative_to(folder)), sha256=sha256(run_path)
                )
            },
        ),
    )
    stage = dict(
        fold="fold-000",
        stage="evaluation",
        status="completed",
        completed_runs=1,
        planned_runs=1,
        sha256=sha256(folder / "summary.json"),
    )
    atomic_json(
        tmp_path / "summary.json",
        dict(identity=dict(protocol="fixture"), final_test_opened=False, stages=[stage]),
    )
    report = analyse_completion(tmp_path)
    assert len(report["results"]) == 2
    assert all(row["seed"] == 43 and row["best_epoch"] == 0 for row in report["results"])
    assert all(row["delta_parent"] == pytest.approx(-0.1) for row in report["results"])
    assert json.loads((tmp_path / "analysis.json").read_text()) == report
    assert "0.20000000" in (tmp_path / "analysis.md").read_text()
    (run_path.parent / "evaluation.parquet").write_bytes(b"corrupto")
    with pytest.raises(ValueError, match="huella|artefacto"):
        analyse_completion(tmp_path)
