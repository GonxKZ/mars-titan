"""Contrastes declarados y límites del análisis, sin ajustes ni datos financieros."""

import importlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def api():
    return importlib.import_module("benchmarks.neural_plasticity_analysis")


def panel():
    return pa.table(
        dict(
            sample_id=["A0", "B0", "A1"],
            asset_id=["A", "B", "A"],
            market=["US"] * 3,
            prediction_at=pa.array([1, 1, 86_400_000_001], type=pa.timestamp("us", tz="UTC")),
            target=[1.0, 1.0, 10.0],
            prediction=[0.0, 0.0, 0.0],
            zero=[0.0, 0.0, 0.0],
        )
    )


def test_scores_use_sessions_and_reject_a_counterfeit_zero_control():
    first = api().checked_scores(panel())
    assert first.summary()["point"]["mae"] == 5.5
    assert first.summary()["row_weighted"]["mae"] == 4.0
    fake = panel().set_column(6, "zero", pa.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        api().checked_scores(fake)


def test_pairing_accepts_order_changes_but_rejects_changed_rows_and_targets():
    first = api().checked_scores(panel())
    permuted = api().checked_scores(panel().take([2, 0, 1]), first.cohort_sha256)
    np.testing.assert_array_equal(first.mae, permuted.mae)
    with pytest.raises(ValueError):
        api().checked_scores(panel().slice(1), first.cohort_sha256)
    changed = panel().set_column(4, "target", pa.array([2.0, 1.0, 10.0]))
    with pytest.raises(ValueError):
        api().checked_scores(changed, first.cohort_sha256)


def test_factorial_interaction_and_constant_controls_keep_their_direction():
    names = [
        "mac_original",
        "write_coherence",
        "fixed_write",
        "cascade_original",
        "cascade_write",
        "cascade_capture",
        "cascade_combined",
        "cascade_fixed_write",
        "cascade_fixed_capture",
        "cascade_fixed_both",
        "cascade_unity",
    ]
    contrasts = api().secondary_contrasts(names)
    assert contrasts["interaction"] == {
        "cascade_combined": 1,
        "cascade_write": -1,
        "cascade_capture": -1,
        "cascade_original": 1,
    }
    assert contrasts["write_coherence_minus_fixed_write"] == {
        "write_coherence": 1,
        "fixed_write": -1,
    }
    assert contrasts["cascade_combined_minus_cascade_fixed_both"] == {
        "cascade_combined": 1,
        "cascade_fixed_both": -1,
    }


@pytest.mark.parametrize(
    "interval,expected",
    [
        ([-0.03, -0.02], "menor_error_en_desarrollo"),
        ([-0.03, -0.000001], "inconcluyente_respecto_al_margen"),
        ([-0.01, 0.02], "inconcluyente"),
        ([0.01, 0.02], "mayor_error_en_desarrollo"),
        (None, "intervalo_no_disponible"),
    ],
)
def test_result_wording_cannot_promote_an_uncertain_or_tiny_effect(interval, expected):
    assert api().interpret(interval, 1e-5) == expected


def test_analysis_requires_the_learning_scope_before_opening_files(tmp_path, learning_hold):
    from mars_titan.training.learning_hold import LearningHoldError

    learning_hold(False)
    with pytest.raises(LearningHoldError):
        api().analyze(tmp_path / "absent", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def run_fixture(tmp_path):
    from mars_titan.data.storage import sha256
    from mars_titan.training.titans_walk_forward import _protocol
    from tests.training.test_titans_walk_forward import PROTOCOL

    module = api()
    protocol_path = tmp_path / module.FILES["protocol"]
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(json.dumps(PROTOCOL, indent=2))
    _, protocol_hash, _, fold = _protocol(protocol_path, "fold-000", 42)
    folder = tmp_path / "runs" / "fixture-42"
    folder.mkdir(parents=True)
    start = int(np.datetime64(fold["evaluation"][0], "us").astype(np.int64))
    table = panel().set_column(
        3,
        "prediction_at",
        pa.array([start, start, start + 86_400_000_000], type=pa.timestamp("us", tz="UTC")),
    )
    pq.write_table(table, folder / "predictions.parquet")
    (folder / "checkpoint.fixture").write_bytes(b"fixture without model weights")
    prepared = dict(
        artifacts={name: str(i) * 64 for i, name in enumerate(module.FILES.values())},
        code={"src/example.py": "e" * 64},
        counts=dict(evaluation=3),
        configuration=dict(fold="fold-000", window={k: v for k, v in fold.items() if k != "id"}),
    )
    prepared["artifacts"][module.FILES["protocol"]] = sha256(protocol_path)
    report = dict(
        status="completed",
        final_test_opened=False,
        request=dict(
            seed=42,
            window="fold-000",
            device="cuda:0",
            variant="mac_online",
            code={"example": "e" * 64},
            view_sha256=prepared["artifacts"][module.FILES["view"]],
            protocol_sha256=protocol_hash,
            recipe_sha256=prepared["artifacts"][module.FILES["recipe"]],
        ),
        checkpoint=dict(path="checkpoint.fixture", sha256=sha256(folder / "checkpoint.fixture")),
        predictions=dict(
            evaluation=dict(
                path="predictions.parquet", sha256=sha256(folder / "predictions.parquet")
            )
        ),
    )
    (folder / "run.json").write_text(json.dumps(report))
    return prepared, dict(id="fixture-42", seed=42, changes={}), report, folder


def test_run_reader_checks_the_confirmed_prediction_and_checkpoint(tmp_path):
    prepared, job, expected, _ = run_fixture(tmp_path)
    actual, table, digest = api().read_run(tmp_path, prepared, job)
    assert actual == expected and len(table) == 3 and len(digest) == 64


@pytest.mark.parametrize(
    "fault", ["paused", "predictions", "checkpoint", "code", "future", "missing", "escape"]
)
def test_incomplete_changed_or_future_results_are_rejected(tmp_path, fault):
    from mars_titan.data.storage import sha256

    prepared, job, report, folder = run_fixture(tmp_path)
    if fault == "paused":
        report["status"] = "paused"
    elif fault == "predictions":
        (folder / "predictions.parquet").write_bytes(b"changed fixture")
    elif fault == "checkpoint":
        (folder / "checkpoint.fixture").write_bytes(b"changed fixture")
    elif fault == "code":
        report["request"]["code"]["example"] = "f" * 64
    elif fault == "escape":
        outside = tmp_path / "outside.parquet"
        outside.write_bytes((folder / "predictions.parquet").read_bytes())
        report["predictions"]["evaluation"]["path"] = str(outside)
    else:
        table = pq.read_table(folder / "predictions.parquet")
        if fault == "future":
            table = table.set_column(
                3,
                "prediction_at",
                pa.array([1_704_067_200_000_000] * 3, type=pa.timestamp("us", tz="UTC")),
            )
        else:
            table = table.slice(1)
        pq.write_table(table, folder / "predictions.parquet")
        report["predictions"]["evaluation"]["sha256"] = sha256(folder / "predictions.parquet")
    (folder / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        api().read_run(tmp_path, prepared, job)


def cost_fixture(tmp_path):
    from mars_titan.data.storage import atomic_json, sha256

    atomic_json(tmp_path / "prepared.json", {})
    digest = sha256(tmp_path / "prepared.json")
    row = dict(
        id="current",
        job=dict(id="model-42"),
        closed=True,
        status="completed",
        global_hold_unchanged=True,
        final_test_opened=False,
        wrapper_seconds=2.0,
        prepared_sha256=digest,
        source_commit="a" * 40,
        internal=dict(
            closure_confirmed=True,
            actual_seconds=1.0,
            optimizer_steps=1,
            peak_rss_bytes=1024,
            peak_cuda_allocated_bytes=0,
            peak_cuda_reserved_bytes=0,
        ),
    )
    atomic_json(tmp_path / "launches/current.json", row)
    old = dict(row, id="previous", status="failed", wrapper_seconds=0.5)
    atomic_json(tmp_path / "previous.json", old)
    plan = dict(
        prepared_sha256=digest,
        prior_receipts=[
            dict(path=str(tmp_path / "previous.json"), sha256=sha256(tmp_path / "previous.json"))
        ],
    )
    atomic_json(tmp_path / "execution-plan.json", plan)
    return dict(jobs=[dict(id="model-42")]), row, plan


def test_cost_includes_failed_attempts_without_double_counting_internal_time(tmp_path):
    prepared, _, _ = cost_fixture(tmp_path)
    costs = api().execution_cost(tmp_path, prepared)
    assert costs["total_wrapper_seconds"] == 2.5
    assert len(costs["attempts"]) == 2


def test_cost_keeps_completed_historical_diagnostics_without_counting_them_as_current_runs(
    tmp_path,
):
    from mars_titan.data.storage import atomic_json, sha256

    prepared, row, plan = cost_fixture(tmp_path)
    historical = dict(row, id="previous", prepared_sha256="f" * 64, wrapper_seconds=3.0)
    atomic_json(tmp_path / "previous.json", historical)
    plan["prior_receipts"][0]["sha256"] = sha256(tmp_path / "previous.json")
    atomic_json(tmp_path / "execution-plan.json", plan)
    costs = api().execution_cost(tmp_path, prepared)
    assert costs["total_wrapper_seconds"] == 5.0
    (tmp_path / "launches/current.json").unlink()
    with pytest.raises(ValueError):
        api().execution_cost(tmp_path, prepared)


def test_unity_control_requires_identical_predictions_and_shared_tensors():
    import torch

    payload = dict(
        identity=dict(initial_parameters_sha256="a" * 64),
        state=dict(
            model={"weights": torch.tensor([1.0, 2.0]), "_extra_state": {"variant": "base"}}
        ),
    )
    other = dict(
        identity=payload["identity"],
        state=dict(
            model={"weights": torch.tensor([1.0, 2.0]), "_extra_state": {"variant": "unit"}}
        ),
    )
    result = api().unity_control(panel(), panel().take([2, 0, 1]), payload, other)
    assert result["passed"] and result["shared_tensors"] == 1
    changed = panel().set_column(5, "prediction", pa.array([2.0, 0.0, 0.0]))
    # El mismo MAE no acredita que las predicciones sean iguales.
    assert api().checked_scores(changed).mae.tolist() == api().checked_scores(panel()).mae.tolist()
    assert not api().unity_control(panel(), changed, payload, other)["passed"]
    other["state"]["model"]["weights"][0] = 3
    assert not api().unity_control(panel(), panel(), payload, other)["passed"]
    other["state"]["model"]["weights"] = payload["state"]["model"]["weights"].clone()
    other["identity"] = dict(initial_parameters_sha256="b" * 64)
    assert not api().unity_control(panel(), panel(), payload, other)["passed"]


@pytest.mark.parametrize("fault", ["unclosed", "duplicate", "missing_model"])
def test_cost_does_not_hide_open_receipts_duplicates_or_incomplete_runs(tmp_path, fault):
    from mars_titan.data.storage import atomic_json

    prepared, row, plan = cost_fixture(tmp_path)
    if fault == "unclosed":
        row["closed"] = False
        atomic_json(tmp_path / "launches/current.json", row)
    elif fault == "duplicate":
        plan["prior_receipts"].append(dict(plan["prior_receipts"][0]))
        atomic_json(tmp_path / "execution-plan.json", plan)
    else:
        prepared["jobs"].append(dict(id="missing-42"))
    with pytest.raises(ValueError):
        api().execution_cost(tmp_path, prepared)
