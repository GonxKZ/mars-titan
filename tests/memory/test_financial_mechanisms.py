"""Publicación financiera de mecanismos sobre fixtures, sin ajustes de parámetros."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mechanism_consumer import calibration
from test_native_episode_backend import native as native

from mars_titan.memory.financial_mechanisms import MechanismBinding
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.mechanism_history import HistoryCapacityError, MechanismHistoryConfig
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.quantile_head import LEVELS, QUANTILE_HEAD
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.mechanism_catalog import MechanismAction
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_same(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_same(left, right)
    else:
        assert actual == expected


def session(
    native, output, data, *, admission="m1", resume=False, phase=None, block_rows=64, history=None
):
    return FinancialSession(
        output,
        native=native,
        consumer=data[3],
        codec=data[2],
        prefixes=data[4],
        retention=RetentionConfig(capacity=4, frontier=2, new_candidates=4),
        phase=phase
        or FinancialPhase("validation", moment(125), moment(126), moment(130), moment(131)),
        admission=admission,
        world="mechanism_fixture",
        fold="0",
        resume=resume,
        block_rows=block_rows,
        mechanism_history=history,
    )


def engines(source, device, *, online=False, max_rows=32768):
    model = (
        FinancialPredictor(
            FinancialConfig(
                source["spec"],
                variant="mac_online",
                hidden_size=32,
                head=QUANTILE_HEAD,
                session_context="market_memory_v1" if online else None,
            ),
            device=device,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    if online:
        values = model.state_dict()
        values["session_context.output"] = torch.eye(32, device=device, dtype=torch.float64) * 0.2
        model.load_state_dict(values)
    mechanism = FrozenMechanismConsumer(
        model,
        calibration=calibration(max_rows=max_rows) if online else None,
        action=MechanismAction("parent" if online else "off", "k1", "online" if online else "none"),
    )
    data = (None, source["spec"], source["codec"], mechanism, source["prefixes"])
    return data


@pytest.mark.parametrize("device", DEVICES)
def test_disabled_session_matches_issued_points_and_next_update_of_the_previous_consumer(
    native, four_flow_source, tmp_path, device
):
    data = engines(four_flow_source, device)
    original = (*data[:3], FrozenFinancialConsumer(data[3].predictor), data[4])
    with (
        session(native, tmp_path / "old", original, admission="m0") as old,
        session(native, tmp_path / "new", data, admission="m0") as new,
    ):
        for index, kind in ((125, "warmup"), (126, "decision"), (127, "decision")):
            before = old.step(four_flow_source["batches"][index], [], kind=kind)
            after = new.step(four_flow_source["batches"][index], [], kind=kind)
            assert [(p.asset, p.decision_at, p.value) for p in after.predictions] == [
                (p.asset, p.decision_at, p.value) for p in before.predictions
            ]
        event = new.mechanism_event(2)
        assert event["generation"] == 2 and event["kind"] == "decision"
        assert event["payload"]["work"]["core_observations"] == 4
        assert all(row["requested"] == "off/k1/none" for row in event["payload"]["rows"])
        for row in event["payload"]["rows"]:
            assert row["point"] == row["quantiles"][2] == row["raw_quantiles"][2]


@pytest.mark.parametrize("device", DEVICES)
def test_online_history_keeps_emitted_quantiles_across_partial_labels_pruning_and_recovery(
    native, four_flow_source, tmp_path, device
):
    data = engines(four_flow_source, device, online=True)
    engine = data[3]
    output = tmp_path / "online"
    with session(native, output, data) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        issued = run.step(four_flow_source["batches"][126], [])
        assert len(issued.predictions) == 4 and not issued.excluded
        first = run.mechanism_event(2)
        partial = [
            native.Feedback(prediction.id, 0, moment(127), 0.125)
            for prediction in issued.predictions[:2]
        ]
        later = run.step(four_flow_source["batches"][127], partial)
        auxiliary = engine.restore_state(run.mechanism_state())
        assert auxiliary.calibration.pending_rows == 8
        assert len(auxiliary.calibration.unresolved_keys()) == 6
        complete = [
            native.Feedback(prediction.id, 0, moment(128), 0.125)
            for prediction in (*issued.predictions[2:], *later.predictions)
        ]
        run.step([], complete, kind="settlement", cutoff=moment(128))
        state = engine.restore_state(run.mechanism_state())
        assert state.calibration.pending_rows == 0
        assert run.diagnostics()["pending"] == 0
        run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        terminal = run.mechanism_state()
        assert terminal["terminal"]["status"] == "phase_closed"
        assert terminal["terminal"]["censored_rows"] == 0
        assert run.mechanism_event(2) == first
        assert not (output / "checkpoint-2.json").exists()
        record = native.read_blob(str(output / "record-2.json"), 64 * 1024**2)
        assert first["native_record_sha256"] == hashlib.sha256(record).hexdigest()
        expected = {row["asset"]: row["value"] for row in json.loads(record)["predictions"]}
        for row in first["payload"]["rows"]:
            assert row["point"] == expected[row["flow_id"]]
            assert np.isfinite(row["quantiles"]).all()
        confirmed = run.snapshot()
    with session(native, output, data, resume=True) as restored:
        assert restored.snapshot() == confirmed
        assert_same(restored.mechanism_state(), terminal)
        assert restored.mechanism_event(2) == first
        with pytest.raises(ValueError):
            restored.mechanism_event(6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "point", ["record_written", "checkpoint_written", "before_commit", "committed"]
)
def test_cut_recovery_preserves_the_next_emission_and_single_feedback(
    native, four_flow_source, tmp_path, device, point
):
    data = engines(four_flow_source, device, online=True)
    with session(native, tmp_path / "reference", data) as reference:
        reference.step(four_flow_source["batches"][125], [], kind="warmup")
        first = reference.step(four_flow_source["batches"][126], [])
        partial = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions[:2]]
        second = reference.step(four_flow_source["batches"][127], partial)
        checkpoint = reference.snapshot()
        checkpoint_state = reference.mechanism_state()
        checkpoint_trace = reference.mechanism_event(3)
        remaining = [
            native.Feedback(p.id, 0, moment(128), 0.125)
            for p in (*first.predictions[2:], *second.predictions)
        ]
        reference.step(four_flow_source["batches"][128], remaining)
        following = reference.snapshot()
        following_state = reference.mechanism_state()
        following_trace = reference.mechanism_event(4)
    with session(native, tmp_path / "cut", data) as interrupted:
        interrupted.step(four_flow_source["batches"][125], [], kind="warmup")
        interrupted.step(four_flow_source["batches"][126], [])

        def fail(boundary):
            if boundary == getattr(native.Boundary, point):
                raise RuntimeError("Corte técnico del archivo de mecanismos")

        with pytest.raises(RuntimeError, match="Corte técnico"):
            interrupted.step(four_flow_source["batches"][127], partial, fault=fail)
    with session(native, tmp_path / "cut", data, resume=True) as restored:
        if point != "committed":
            restored.step(four_flow_source["batches"][127], partial)
        assert restored.snapshot() == checkpoint
        assert_same(restored.mechanism_state(), checkpoint_state)
        assert restored.mechanism_event(3) == checkpoint_trace
        restored.step(four_flow_source["batches"][128], remaining)
        assert restored.snapshot() == following
        assert_same(restored.mechanism_state(), following_state)
        assert restored.mechanism_event(4) == following_trace
        assert restored.snapshot()["applied"] == 8


@pytest.mark.parametrize("device", DEVICES)
def test_partition_and_asset_order_keep_one_context_write_and_a_common_snapshot(
    native, four_flow_source, tmp_path, device
):
    data = engines(four_flow_source, device, online=True)
    outputs = []
    for block in (1, 4):
        with session(native, tmp_path / str(block), data, block_rows=block) as run:
            run.step(list(reversed(four_flow_source["batches"][125])), [], kind="warmup")
            run.step(list(reversed(four_flow_source["batches"][126])), [])
            event = run.mechanism_event(2)["payload"]
            assert event["work"]["context_updates_proposed"] == 1
            assert event["work"]["core_observations"] == 4
            state = run.mechanism_state()
            assert state["context"]["committed_at"].count(moment(126)) == 1
            outputs.append(event)
    assert [row["flow_id"] for row in outputs[0]["rows"]] == [
        row["flow_id"] for row in outputs[1]["rows"]
    ]
    np.testing.assert_allclose(
        [row["quantiles"] for row in outputs[0]["rows"]],
        [row["quantiles"] for row in outputs[1]["rows"]],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("with_labels", [False, True])
def test_reserved_boundary_closes_with_an_explicit_exclusive_feedback_watermark(
    native, four_flow_source, tmp_path, with_labels
):
    data = engines(four_flow_source, "cpu", online=True)
    phase = FinancialPhase("validation", moment(125), moment(126), moment(130), FINAL_TEST_US)
    with session(native, tmp_path / "phase", data, phase=phase) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        issued = run.step(four_flow_source["batches"][126], [])
        labels = (
            [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions]
            if with_labels
            else []
        )
        closed = run.step([], labels, kind="settlement", cutoff=FINAL_TEST_US, close_phase=True)
        assert len(closed.finalized) == (0 if with_labels else 4)
        assert len(closed.applied) == (4 if with_labels else 0)
        state = run.mechanism_state()
        assert state["cutoff"] == state["terminal"]["closed_at"] == FINAL_TEST_US
        assert state["terminal"]["censored_rows"] == (0 if with_labels else 4)
        event = run.mechanism_event(3)
        assert event["cutoff"] == FINAL_TEST_US
        assert event["payload"]["feedback_watermark"] == FINAL_TEST_US - 1
        if with_labels:
            assert set(event["payload"]["received_prediction_ids"]) == {
                p.id for p in issued.predictions
            }
            received = event["payload"]["settlements"][0]["labels"]
            assert len(received) == 4 and {row["available_at"] for row in received} == {moment(127)}
        else:
            assert event["payload"]["received_prediction_ids"] == []
            assert event["payload"]["settlements"] == []
        before = run.snapshot()
    with session(native, tmp_path / "phase", data, phase=phase, resume=True) as restored:
        assert restored.snapshot() == before
        assert_same(restored.mechanism_state(), state)
        assert restored.mechanism_event(2)["kind"] == "decision"
    with session(native, tmp_path / "new-phase", data, phase=phase) as fresh:
        initial = fresh.mechanism_state()
        assert initial["cutoff"] is None and initial["calibration"]["cohorts"] == []
        assert all(value == -1 for value in initial["context"]["committed_at"])


def test_archive_exhaustion_aborts_without_changing_the_last_confirmed_generation(
    native, four_flow_source, tmp_path
):
    data = engines(four_flow_source, "cpu", online=True)
    budget = MechanismHistoryConfig(max_chunks=1, max_events_per_chunk=1)
    output = tmp_path / "bounded"
    with session(native, output, data, history=budget) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        before = run.snapshot()
        head = (output / "latest.json").read_bytes()
        with pytest.raises(HistoryCapacityError):
            run.step(four_flow_source["batches"][126], [])
        assert (output / "latest.json").read_bytes() == head
    with session(native, output, data, history=budget, resume=True) as restored:
        assert restored.snapshot() == before
        assert restored.mechanism_event(1)["kind"] == "warmup"
        with pytest.raises(ValueError):
            restored.mechanism_event(2)


@pytest.mark.parametrize("compression", ["none", "zlib1"])
def test_history_corruption_rejects_recovery_without_replacing_the_head(
    native, four_flow_source, tmp_path, compression
):
    data = engines(four_flow_source, "cpu", online=True)
    budget = MechanismHistoryConfig(compression=compression)
    output = tmp_path / "damaged"
    with session(native, output, data, history=budget) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])
        bundle = run._bundle(run.snapshot()["state"])
        block = bundle["mechanism_history"]["chunks"][0]["reference"]
        first = run.mechanism_event(1)
        assert first["kind"] == "warmup"
    head = (output / "latest.json").read_bytes()
    path = output / "artifacts" / block["name"]
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError):
        session(native, output, data, history=budget, resume=True)
    assert (output / "latest.json").read_bytes() == head


@pytest.mark.parametrize(
    "field",
    [
        "consumer_id",
        "auxiliary_sha256",
        "context_id",
        "parameter_id",
        "snapshot_sha256",
        "levels",
        "calibration_record",
        "received_prediction_ids",
        "feedback_watermark",
        "extra",
    ],
)
def test_proposed_history_must_match_all_control_fields_and_native_feedback(
    native, four_flow_source, tmp_path, field
):
    data = engines(four_flow_source, "cpu", online=True)
    with session(native, tmp_path / field, data) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])
        bundle = run._bundle(run.snapshot()["state"])
        first, second = run.mechanism_event(1), run.mechanism_event(2)
        forged = copy.deepcopy(second)
        changes = dict(
            levels=[],
            calibration_record=None,
            received_prediction_ids=["d" * 64],
            feedback_watermark=second["cutoff"] + 1,
            extra=True,
        )
        forged["payload"][field] = changes.get(field, "d" * 64)
        history = run._binding.history
        manifest = history.append(history.empty(), first)
        manifest = history.append(manifest, forged)
        proposed = dict(bundle, mechanism_history=manifest)
        names = ("flow_id", "sample_id", "prediction_at", "input_available_at", "input_sha256")
        observations = [{name: row[name] for name in names} for row in second["payload"]["rows"]]
        with pytest.raises(ValueError):
            run._binding.verify(run, proposed, observations)


def test_ordered_finite_extreme_quantiles_do_not_need_overflowing_differences():
    maximum = np.finfo(np.float64).max
    values = [-maximum, maximum, maximum, maximum, maximum]
    control = dict(
        levels=list(LEVELS),
        rows=[
            dict(
                quantiles=values,
                raw_quantiles=values,
                k1_quantiles=values,
                point=maximum,
                calibration="none",
            )
        ],
    )
    with np.errstate(over="raise"):
        MechanismBinding._check_quantiles(control)


def test_online_capacity_fallback_keeps_the_financial_labels_outside_pt2(
    native, four_flow_source, tmp_path
):
    data = engines(four_flow_source, "cpu", online=True, max_rows=1)
    with session(native, tmp_path / "capacity", data) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        issued = run.step(four_flow_source["batches"][126], [])
        emitted = run.mechanism_event(2)
        assert len(emitted["payload"]["rows"]) == 4
        for row in emitted["payload"]["rows"]:
            assert row["reasons"] == ["pending_calibration_budget"]
            assert row["emitted_route"] == "k1" and row["calibration"] == "none"
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        event = run.mechanism_event(3)
        assert event["payload"]["received_prediction_ids"] == []
        assert run.snapshot()["applied"] == 4 and run.diagnostics()["admitted"] == 4
        assert len(run.retained_episodes()) == 4
        assert run.mechanism_state()["calibration"]["cohorts"] == []


def test_phase_with_no_observations_can_close_and_recover_an_empty_terminal(
    native, four_flow_source, tmp_path
):
    data = engines(four_flow_source, "cpu", online=True)
    output = tmp_path / "empty"
    with session(native, output, data) as run:
        run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        expected = run.mechanism_state()
        assert expected["terminal"]["censored_rows"] == 0
        assert run.mechanism_event(1)["payload"]["rows"] == []
        assert run.snapshot()["observed"] == run.snapshot()["issued"] == 0
    with session(native, output, data, resume=True) as restored:
        assert_same(restored.mechanism_state(), expected)


@pytest.mark.parametrize(
    "field", ["point", "quantiles", "raw_quantiles", "k1_quantiles", "calibration_record"]
)
def test_trace_validation_rejects_changed_points_quantiles_and_emitted_corrections(
    native, four_flow_source, tmp_path, field
):
    data = engines(four_flow_source, "cpu", online=True)
    with session(native, tmp_path / field, data) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])
        event = run.mechanism_event(2)
        record = json.loads(run._executor.record_json(2))
        control = copy.deepcopy(event["payload"])
        control["observations"] = 4
        if field == "point":
            control["rows"][0]["point"] += 1
        elif field == "calibration_record":
            control["calibration_record"]["groups"]["US"]["intervals"]["0.8"]["correction"] += 1
        elif field == "k1_quantiles":
            control["rows"][0][field][0] = float("nan")
        else:
            control["rows"][0][field][0] -= 1
        with pytest.raises(ValueError):
            run._binding._check_record(control, record, cutoff=event["cutoff"])
