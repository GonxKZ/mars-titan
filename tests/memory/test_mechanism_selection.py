"""Selección automática acotada sobre entradas admitidas, sin ajuste de coeficientes."""

import copy
import json

import pytest
import torch
from test_episodic_readout import CONTEXT
from test_financial_adapter import raw_batch
from test_financial_mechanisms import assert_same, engines, session
from test_financial_session import moment
from test_financial_session_controls import flow_source
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import KINDS
from test_mechanism_consumer import calibration, consumer
from test_mechanism_observations import AT, extract, observation_rows
from test_mechanism_policy import policy
from test_native_episode_backend import native as native

from mars_titan.models.titans.mechanism_catalog import MechanismAction
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def selected_data(source, device="cpu"):
    data = engines(source, device, online=True)
    parent = data[3]
    selector = policy(device=device)
    selector.support_max.fill_(100)
    # El contexto anterior distingue mercados. Las alternativas sin lector son inadmisibles.
    selector.point_coefficients.zero_()
    selector.interval_coefficients.zero_()
    selector.point_coefficients[13, 6] = 2
    selector.interval_coefficients[-1, 20] = 1
    selected = FrozenMechanismConsumer(
        parent.predictor,
        calibration=parent.initial_state().calibration,
        policy=selector,
    )
    return (*data[:3], selected, data[4])


@pytest.mark.parametrize("device", DEVICES)
def test_automatic_selection_is_recorded_once_per_market_before_mac(
    native, four_flow_source, tmp_path, device
):
    data = selected_data(four_flow_source, device)
    with session(native, tmp_path / "selected", data) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        assert run.mechanism_event(1)["payload"]["selection"] is None
        issued = run.step(four_flow_source["batches"][126], [])
        payload = run.mechanism_event(2)["payload"]
        trace = payload["selection"]
        assert trace["kind"] == "pre_mac_mechanism_selection_v1"
        assert trace["observation"]["observation"]["cutoff"] == moment(126)
        assert trace["observation"]["observation"]["previous_cutoff"] == moment(125)
        assert trace["observation"]["observation"]["groups"][0]["values"][13] == 0
        assert trace["dtype"] == "torch.float64"
        assert len(trace["groups"]) == 1
        chosen = trace["groups"][0]
        assert chosen["market"] == "US" and chosen["action"] == "parent/k1/online"
        assert chosen["action_id"] == 20
        assert len(chosen["scores"]) == len(chosen["available"]) == 36
        assert chosen["available"][3] is False
        assert chosen["point_gains"][18:21] == [0.0, 0.0, 0.0]
        assert chosen["interval_gains"][20] == 1
        assert len(chosen["features"]) == 21
        assert all(row["requested"] == chosen["action"] for row in payload["rows"])
        assert all("selection" not in row for row in payload["rows"])
        assert payload["work"]["mac_online_observations"] == 4
        assert len(issued.predictions) == 4
        identity = data[3].identity()
        assert identity["fixed_controls"] is False
        assert identity["new_learned_selector"] is False
        assert identity["selector"]["coefficient_origin"] == "supplied_not_verified_as_fitted"
        assert len(json.dumps(trace)) < 64 * 1024
        run.step(four_flow_source["batches"][127], [])
        following = run.mechanism_event(3)["payload"]["selection"]
        assert following["observation"]["observation"]["groups"][0]["values"][13] == 1
        assert following["groups"][0]["point_gains"][18:21] == [2.0, 2.0, 2.0]


def test_policy_and_explicit_action_are_unambiguous(four_flow_source):
    parent = engines(four_flow_source, "cpu")[3]
    with pytest.raises(ValueError, match="acción|selector"):
        FrozenMechanismConsumer(parent.predictor, action=MechanismAction(), policy=policy())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "boundary", ["record_written", "checkpoint_written", "before_commit", "committed"]
)
def test_recovery_reproduces_the_next_choice_prediction_and_feedback(
    native, four_flow_source, tmp_path, device, boundary
):
    data = selected_data(four_flow_source, device)
    with session(native, tmp_path / "reference", data) as reference:
        reference.step(four_flow_source["batches"][125], [], kind="warmup")
        first = reference.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions]
        reference.step(four_flow_source["batches"][127], labels)
        before = reference.snapshot(), reference.mechanism_state(), reference.mechanism_event(3)
        import math

        observed = before[2]["payload"]["selection"]["observation"]["observation"]
        assert observed["groups"][0]["values"][16] == math.log1p(4)
        reference.step(four_flow_source["batches"][128], [])
        after = reference.snapshot(), reference.mechanism_state(), reference.mechanism_event(4)
    with session(native, tmp_path / "cut", data) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])

        def fail(point):
            if point == getattr(native.Boundary, boundary):
                raise RuntimeError("Corte técnico del selector")

        with pytest.raises(RuntimeError, match="Corte técnico del selector"):
            run.step(four_flow_source["batches"][127], labels, fault=fail)
    with session(native, tmp_path / "cut", data, resume=True) as restored:
        if boundary != "committed":
            restored.step(four_flow_source["batches"][127], labels)
        from test_financial_mechanisms import assert_same

        assert_same(
            (restored.snapshot(), restored.mechanism_state(), restored.mechanism_event(3)), before
        )
        restored.step(four_flow_source["batches"][128], [])
        assert_same(
            (restored.snapshot(), restored.mechanism_state(), restored.mechanism_event(4)), after
        )
        assert restored.snapshot()["applied"] == 4


@pytest.mark.parametrize(
    "field",
    [
        "features",
        "standardized",
        "scores",
        "point_gains",
        "interval_gains",
        "action",
        "action_id",
        "available",
        "supported",
        "availability",
        "market",
        "extra",
    ],
)
def test_resealed_control_cannot_hide_changed_selection(native, four_flow_source, tmp_path, field):
    data = selected_data(four_flow_source)
    with session(native, tmp_path / field, data) as run:
        run.step(four_flow_source["batches"][126], [])
        event = run.mechanism_event(1)
        control = copy.deepcopy(event["payload"])
        control["observations"] = 4
        group = control["selection"]["groups"][0]
        if field in {"features", "standardized", "scores", "point_gains", "interval_gains"}:
            group[field][0] += 0.25
        elif field == "available":
            group[field][3] = True
        elif field == "availability":
            group[field]["reader"] = True
        else:
            group[field] = {
                "action": "off/k1/none",
                "action_id": 0,
                "supported": False,
                "market": "CN",
                "extra": True,
            }[field]
        record = json.loads(run._executor.record_json(1))
        with pytest.raises(ValueError):
            run._binding._check_record(control, record, cutoff=event["cutoff"])


@pytest.mark.parametrize(
    "field",
    [
        "context_id",
        "snapshot_sha256",
        "rows_sha256",
        "consumer_id",
        "producer_sha256",
        "sha256",
        "dtype",
        "policy_sha256",
    ],
)
def test_foreign_observation_or_identity_is_rejected(native, four_flow_source, tmp_path, field):
    data = selected_data(four_flow_source)
    with session(native, tmp_path / field, data) as run:
        run.step(four_flow_source["batches"][126], [])
        event = run.mechanism_event(1)
        control = copy.deepcopy(event["payload"])
        control["observations"] = 4
        trace = control["selection"]
        if field in {"dtype", "policy_sha256"}:
            trace[field] = "torch.float32" if field == "dtype" else "d" * 64
        elif field == "sha256":
            trace["observation"][field] = "d" * 64
        else:
            from mars_titan.models.titans.mechanism_observation import MechanismObservation

            body = trace["observation"]["observation"]
            body[field] = "d" * 64
            trace["observation"] = MechanismObservation(body).record()
        with pytest.raises(ValueError):
            run._binding._check_record(
                control, json.loads(run._executor.record_json(1)), cutoff=event["cutoff"]
            )


@pytest.mark.parametrize(
    "field", ["buffer", "data", "mode", "method", "scalar", "hook", "reference"]
)
def test_policy_is_frozen_across_cohorts(four_flow_source, field):
    engine = selected_data(four_flow_source)[3]
    if field == "buffer":
        engine.policy.costs[1] = 1
    elif field == "data":
        engine.policy.costs.data[1] = 1
    elif field == "mode":
        engine.policy.train()
    elif field == "method":
        engine.policy.forward = lambda *_: None
    elif field == "scalar":
        engine.policy.min_gain = 0.25
    elif field == "hook":
        engine.policy.register_forward_pre_hook(lambda *_: None)
    else:
        engine.policy = policy()
    with pytest.raises(ValueError):
        engine.verify()


@pytest.mark.parametrize("device", DEVICES)
def test_none_and_baseline_policy_keep_points_states_and_no_current_feedback_features(
    native, four_flow_source, tmp_path, device
):
    data = selected_data(four_flow_source, device)
    selector = data[3].policy
    selector.point_coefficients.zero_()
    selector.interval_coefficients.zero_()
    parent = engines(four_flow_source, device, online=True)[3]
    # Reidentificar tras cambiar coeficientes, sin compartir una identidad ya congelada.
    selected = FrozenMechanismConsumer(
        parent.predictor, calibration=parent.initial_state().calibration, policy=selector
    )
    fixed = FrozenMechanismConsumer(
        parent.predictor, calibration=parent.initial_state().calibration
    )
    traces = []
    for name, engine in (("selected", selected), ("fixed", fixed)):
        variant = (*data[:3], engine, data[4])
        with session(native, tmp_path / name, variant) as run:
            run.step(four_flow_source["batches"][125], [], kind="warmup")
            result = run.step(four_flow_source["batches"][126], [])
            traces.append(
                (
                    [(p.asset, p.value) for p in result.predictions],
                    run.mechanism_event(2)["payload"],
                )
            )
    assert traces[0][0] == traces[1][0]
    assert traces[0][1]["rows"] == traces[1][1]["rows"]
    assert traces[0][1]["work"] == traces[1][1]["work"]
    assert traces[0][1]["selection"]["groups"][0]["action_id"] == 0
    assert "selection" not in traces[1][1]


@pytest.mark.parametrize("device", DEVICES)
def test_two_markets_use_distinct_observations_and_mask_before_argmax(device):
    parent = consumer(device)
    selector = policy(device=device)
    selector.point_coefficients.zero_()
    selector.interval_coefficients.zero_()
    selector.point_coefficients[2, 0] = 0  # Referencia incremental cero.
    selector.point_coefficients[-1, 1] = (
        1e6  # No hay lector, aunque esta ruta tenga mayor utilidad.
    )
    selector.point_coefficients[-1, 6] = 2  # Tampoco hay contexto en este padre.
    engine = FrozenMechanismConsumer(parent.predictor, policy=selector)
    raw = raw_batch()
    raw["sample_ids"][0] = f"CN/000001.SZ/{AT}"
    raw["market"][0] = "CN"
    rows = observation_rows(engine, raw)
    packet = extract(engine, rows)
    event = engine.begin(
        engine.initial_state(),
        flow_ids=tuple(r.flow_id for r in rows),
        cutoff=AT,
        context_id=packet.record()["observation"]["context_id"],
        matures_at=(AT + 10,) * 2,
        observation=packet,
    )
    trace = json.loads(event.selection_json)
    assert [g["market"] for g in trace["groups"]] == ["CN", "US"]
    assert all(g["action_id"] == 0 for g in trace["groups"])
    assert trace["groups"][0]["features"][2] == 1
    assert trace["groups"][1]["features"][2] == 0
    assert all(g["available"][3] is False and g["scores"][3] == 1e6 for g in trace["groups"])


@pytest.mark.parametrize("device", DEVICES)
def test_auto_k4_matches_the_fixed_route_without_repeating_mac(device):
    parent = consumer(device, reader=True)
    selector = policy(device=device)
    selector.point_coefficients.zero_()
    selector.interval_coefficients.zero_()
    selector.point_coefficients[-1, 4] = 2  # off/k4cr con las tres calibraciones.
    automatic = FrozenMechanismConsumer(parent.predictor, readout=parent.readout, policy=selector)
    fixed = FrozenMechanismConsumer(
        parent.predictor, readout=parent.readout, action=MechanismAction(route="k4cr")
    )
    observed = []
    for engine in (automatic, fixed):
        rows = observation_rows(engine)
        options = dict(observation=extract(engine, rows)) if engine.policy is not None else {}
        event = engine.begin(
            engine.initial_state(),
            flow_ids=tuple(r.flow_id for r in rows),
            cutoff=AT,
            context_id=CONTEXT,
            matures_at=(AT + 10,) * 2,
            **options,
        )
        from mars_titan.models.titans.financial_inputs import DecisionBatch

        batch = DecisionBatch.from_corpus(
            raw_batch(), engine.predictor.config.inputs, device=device, dtype=torch.float64
        )
        prepared = engine.prepare(batch, engine.predictor.initial_state(batch.flow_ids), event)
        emission = engine.finish(prepared.event)
        observed.append((emission, prepared.next_state))
        assert all(row["reader_steps"] == 4 for row in emission.rows)
        assert prepared.next_state.mac.memory.steps.tolist() == [1, 1]
    assert observed[0][0].point_predictions.tolist() == observed[1][0].point_predictions.tolist()
    assert torch.equal(observed[0][1].mac.memory.weights[0], observed[1][1].mac.memory.weights[0])


@pytest.mark.parametrize("device", DEVICES)
def test_per_market_choices_follow_their_own_prefix(device):
    parent = consumer(device, kind=KINDS[0])
    selector = policy(device=device)
    selector.point_coefficients.zero_()
    selector.interval_coefficients.zero_()
    selector.point_coefficients[2, 6] = 2  # Presencia de noticias.
    engine = FrozenMechanismConsumer(parent.predictor, policy=selector)
    raw = raw_batch()
    raw["sample_ids"][0] = f"CN/000001.SZ/{AT}"
    raw["market"][0] = "CN"
    rows = observation_rows(engine, raw)
    packet = extract(engine, rows)
    event = engine.begin(
        engine.initial_state(),
        flow_ids=tuple(r.flow_id for r in rows),
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + 10,) * 2,
        observation=packet,
    )
    assert [(market, plan.applied.key) for market, plan in event.plans] == [
        ("CN", "parent/k1/none"),
        ("US", "off/k1/none"),
    ]


@pytest.mark.parametrize("device", DEVICES)
def test_input_cast_and_actual_standardization_are_archived(device):
    parent = consumer(device)
    selector = policy(dtype=torch.float32, device=device)
    selector.mean.fill_(0.5)
    selector.scale.fill_(2)
    engine = FrozenMechanismConsumer(parent.predictor, policy=selector)
    rows = observation_rows(engine)
    packet = extract(engine, rows)
    event = engine.begin(
        engine.initial_state(),
        flow_ids=tuple(r.flow_id for r in rows),
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + 10,) * 2,
        observation=packet,
    )
    trace = json.loads(event.selection_json)
    actual = torch.tensor(
        packet.record()["observation"]["groups"][0]["values"], dtype=torch.float32
    )
    assert trace["dtype"] == "torch.float32"
    assert trace["groups"][0]["features"] == actual.tolist()
    assert trace["groups"][0]["standardized"] == ((actual - 0.5) / 2).tolist()


@pytest.mark.parametrize("bad", [None, 7, [], {"market": "US"}])
def test_malformed_group_is_a_contract_error(native, four_flow_source, tmp_path, bad):
    data = selected_data(four_flow_source)
    with session(native, tmp_path / "malformed", data) as run:
        run.step(four_flow_source["batches"][126], [])
        event = run.mechanism_event(1)
        control = copy.deepcopy(event["payload"])
        control["observations"] = 4
        control["selection"]["groups"] = [bad]
        with pytest.raises(ValueError):
            run._binding._check_record(
                control, json.loads(run._executor.record_json(1)), cutoff=event["cutoff"]
            )


@pytest.mark.parametrize("device", DEVICES)
def test_partition_and_input_order_preserve_the_same_choice(
    native, four_flow_source, tmp_path, device
):
    data = selected_data(four_flow_source, device)
    records = []
    for size in (1, 4):
        with session(native, tmp_path / str(size), data, block_rows=size) as run:
            run.step(list(reversed(four_flow_source["batches"][125])), [], kind="warmup")
            run.step(list(reversed(four_flow_source["batches"][126])), [])
            records.append(run.mechanism_event(2)["payload"])
    left, right = (copy.deepcopy(record["selection"]) for record in records)
    first, second = (trace.pop("observation") for trace in (left, right))
    assert left == right
    # El contrato de sesión incluye el lote físico. Su identidad debe cambiar,
    # aunque el vector consumido y la decisión sean iguales.
    assert first["sha256"] != second["sha256"]
    bodies = first["observation"], second["observation"]
    assert {key for key in bodies[0] if bodies[0][key] != bodies[1][key]} == {"context_id"}
    assert records[0]["work"] == records[1]["work"]
    import numpy as np

    np.testing.assert_allclose(
        [r["quantiles"] for r in records[0]["rows"]],
        [r["quantiles"] for r in records[1]["rows"]],
        rtol=1e-12,
        atol=1e-12,
    )


def test_incompatible_selector_cannot_resume_or_replace_the_head(
    native, four_flow_source, tmp_path
):
    data = selected_data(four_flow_source)
    output = tmp_path / "frozen-policy"
    with session(native, output, data) as run:
        run.step(four_flow_source["batches"][126], [])
    head = (output / "latest.json").read_bytes()
    changed = policy()
    engine = FrozenMechanismConsumer(
        data[3].predictor, calibration=data[3].initial_state().calibration, policy=changed
    )
    other = (*data[:3], engine, data[4])
    with pytest.raises(ValueError):
        session(native, output, other, resume=True)
    assert (output / "latest.json").read_bytes() == head


@pytest.mark.parametrize("device", DEVICES)
def test_selected_online_byte_fallback_publishes_k1_and_recovers_without_pending_labels(
    native, tmp_path, device
):
    source = flow_source(tmp_path / "source", flows=tuple(f"US/T{i:04d}" for i in range(8)))
    data = selected_data(source, device)
    engine = FrozenMechanismConsumer(
        data[3].predictor, calibration=calibration(max_bytes=4096), policy=data[3].policy
    )
    data = (*data[:3], engine, data[4])
    output = tmp_path / "bounded-online"
    with session(native, output, data) as run:
        issued = run.step(source["batches"][126], [])
        event = run.mechanism_event(1)
        payload = event["payload"]
        assert payload["selection"]["groups"][0]["action"] == "parent/k1/online"
        assert len(issued.predictions) == 8
        assert payload["work"]["online_failures"] == 1
        assert payload["work"]["online_selected_proposals"] == 0
        for row in payload["rows"]:
            assert row["requested"] == "parent/k1/online"
            assert row["reasons"] == ["pending_calibration_bytes"]
            assert row["emitted_route"] == "k1" and row["calibration"] == "none"
            assert row["point"] == row["k1_point"]
        state = run.mechanism_state()
        assert state["calibration"]["cohorts"] == []
    with session(native, output, data, resume=True) as restored:
        assert restored.mechanism_event(1) == event
        assert_same(restored.mechanism_state(), state)


@pytest.mark.parametrize("device", DEVICES)
def test_selected_k4_online_byte_fallback_keeps_execution_and_emission_distinct(
    native, tmp_path, device
):
    from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig

    source = flow_source(tmp_path / "source", flows=tuple(f"US/T{i:04d}" for i in range(8)))
    data = selected_data(source, device)
    selector = policy(device=device)
    selector.point_coefficients.zero_()
    selector.interval_coefficients.zero_()
    selector.point_coefficients[-1, 9] = 2
    selector.interval_coefficients[-1, 29] = 1
    reader = (
        EpisodicReadout(
            EpisodicReadoutConfig(source["codec"].fingerprint(), hidden_size=32),
            device=device,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    engine = FrozenMechanismConsumer(
        data[3].predictor,
        readout=reader,
        calibration=calibration(max_bytes=4096),
        policy=selector,
    )
    data = (*data[:3], engine, data[4])
    with session(native, tmp_path / "k4", data) as run:
        run.step(source["batches"][126], [])
        payload = run.mechanism_event(1)["payload"]
        assert payload["selection"]["groups"][0]["action"] == "parent/k4cc/online"
        assert payload["work"]["mac_online_observations"] == 8
        for row in payload["rows"]:
            assert row["executed_route"] == "k4cc"
            assert row["emitted_route"] == "k1"
            assert row["reader_steps"] == 4 and row["discarded_reader_steps"] == 3
            assert row["point"] == row["k1_point"]
            assert row["quantiles"] == row["raw_quantiles"] == row["k1_quantiles"]


@pytest.mark.parametrize(
    "field", ["reason", "k1", "emitted_route", "discarded", "failed", "selected"]
)
def test_online_fallback_does_not_allow_forged_emission_or_work(native, tmp_path, field):
    source = flow_source(tmp_path / "source", flows=tuple(f"US/T{i:04d}" for i in range(8)))
    data = selected_data(source)
    engine = FrozenMechanismConsumer(
        data[3].predictor, calibration=calibration(max_bytes=4096), policy=data[3].policy
    )
    data = (*data[:3], engine, data[4])
    with session(native, tmp_path / "forged", data) as run:
        run.step(source["batches"][126], [])
        event = run.mechanism_event(1)
        control = copy.deepcopy(event["payload"])
        control["observations"] = 8
        if field == "reason":
            control["rows"][0]["reasons"] = ["unavailable_readout"]
        elif field == "k1":
            control["rows"][0]["k1_point"] += 1
        elif field == "emitted_route":
            control["rows"][0]["emitted_route"] = "k2c"
        elif field == "discarded":
            control["rows"][0]["discarded_reader_steps"] = 1
        else:
            control["work"][
                "online_failures" if field == "failed" else "online_selected_proposals"
            ] = 0 if field == "failed" else 1
        with pytest.raises(ValueError):
            run._binding._check_record(
                control, json.loads(run._executor.record_json(1)), cutoff=event["cutoff"]
            )
