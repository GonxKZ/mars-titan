"""Composición S2 con acciones nativas, sin ajustes ni métricas científicas."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_failure_session import book, open_session, policy
from test_financial_mechanisms import engines
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.memory.episodic_session import EpisodicSession
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.models.titans.mechanism_catalog import MechanismAction
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


def mechanism_policy(source, engine, path):
    return replace(policy(source, engine, path), action_source="mechanism_trace_v1")


@pytest.mark.parametrize("online", [False, True])
def test_confidence_preserves_mechanism_points_quantiles_and_mature_errors(
    native, four_flow_source, tmp_path, online
):
    source = four_flow_source
    engine = engines(source, "cpu", online=online)[3]
    config = mechanism_policy(source, engine, tmp_path)
    phase = FinancialPhase("validation", moment(125), moment(126), moment(200), moment(201))
    previous_left, previous_right = [], []
    with (
        open_session(native, source, engine, tmp_path / "off", None, phase=phase) as off,
        open_session(native, source, engine, tmp_path / "on", config, phase=phase) as on,
    ):
        for at in (125, 126, 127, 128):
            left_labels = [
                native.Feedback(p.id, 0, moment(at), p.value + 1.0) for p in previous_left
            ]
            right_labels = [
                native.Feedback(p.id, 0, moment(at), p.value + 1.0) for p in previous_right
            ]
            left = off.step(
                source["batches"][at], left_labels, kind="warmup" if at == 125 else "decision"
            )
            right = on.step(
                list(reversed(source["batches"][at])),
                right_labels,
                kind="warmup" if at == 125 else "decision",
            )
            assert [(p.asset, p.value) for p in right.predictions] == [
                (p.asset, p.value) for p in left.predictions
            ]
            previous_left, previous_right = left.predictions, right.predictions
            if at == 125:
                assert book(on).counts()["issued"] == 0
                continue
            a = off.mechanism_event(at - 124)["payload"]["rows"]
            b = on.mechanism_event(at - 124)["payload"]["rows"]
            assert [(r["flow_id"], r["quantiles"], r["raw_quantiles"]) for r in a] == [
                (r["flow_id"], r["quantiles"], r["raw_quantiles"]) for r in b
            ]
            assert all(r["point"] == r["quantiles"][2] for r in b)
            if at == 128:
                bundle = on._bundle(on.snapshot()["state"])
                signal = on._read(bundle["control"], "control")["failure_memory"]
                assert signal["eligible"] == [4] * 4
                assert all(value < 1 for value in signal["confidence"])
        mature = book(on)
        assert mature.counts()["resolved"] == 8
        assert all(r.target - r.emission.prediction == pytest.approx(1.0) for r in mature._retained)
        assert all(r.emission.context.action.startswith("mechanism/") for r in mature._retained)
        saved = mature.export()
        expected = off.step(source["batches"][130], [])
        expected_quantiles = [r["quantiles"] for r in off.mechanism_event(5)["payload"]["rows"]]
    with open_session(
        native, source, engine, tmp_path / "on", config, phase=phase, resume=True
    ) as run:
        assert book(run).export() == saved
        resumed = run.step(source["batches"][130], [])
        assert [(p.asset, p.value) for p in resumed.predictions] == [
            (p.asset, p.value) for p in expected.predictions
        ]
        assert [
            r["quantiles"] for r in run.mechanism_event(5)["payload"]["rows"]
        ] == expected_quantiles
        engine.verify(strong=True)


def test_failure_context_records_applied_fallback_instead_of_the_requested_action(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    predictor = engines(source, "cpu")[3].predictor
    engine = FrozenMechanismConsumer(predictor, action=MechanismAction("parent", "k4rc", "online"))
    config = mechanism_policy(source, engine, tmp_path)
    with open_session(native, source, engine, tmp_path / "fallback", config) as run:
        run.step(source["batches"][125], [])
        traces = run.mechanism_event(1)["payload"]["rows"]
        assert all(row["requested"] == "parent/k4rc/online" for row in traces)
        assert {r.context.action for r in book(run)._pending.values()} == {
            "mechanism/off/k1/none/emitted/k1"
        }


@pytest.mark.parametrize("mode", ["shrink_point", "constant_shrink"])
def test_mechanism_attenuation_is_rejected_before_publication(
    native, four_flow_source, tmp_path, mode
):
    source = four_flow_source
    engine = engines(source, "cpu")[3]
    config = replace(mechanism_policy(source, engine, tmp_path), mode=mode)
    with pytest.raises(ValueError, match="cuantiles|calibración"):
        open_session(native, source, engine, tmp_path / "rejected", config)
    assert not (tmp_path / "rejected" / "latest.json").exists()


def trace_fixture(source, native):
    rows = EpisodicSession._rows(
        SimpleNamespace(codec=source["codec"], native=native), source["batches"][125]
    )
    records = [
        row.metadata()
        | dict(
            point=1.0,
            context="off",
            executed_route="k1",
            emitted_route="k1",
            calibration="none",
            requested="parent/k4rc/online",
        )
        for row in rows
    ]
    return rows, [1.0] * len(rows), dict(rows=records)


@pytest.mark.parametrize("field,value", [("flow_id", []), ("point", True)])
def test_action_trace_rejects_wrong_field_types(native, four_flow_source, field, value):
    from mars_titan.memory.failure_actions import emission_actions

    rows, points, control = trace_fixture(four_flow_source, native)
    control["rows"][0][field] = value
    with pytest.raises(ValueError):
        emission_actions(rows, points, control)


def test_action_trace_rejects_matching_infinite_points(native, four_flow_source):
    from mars_titan.memory.failure_actions import emission_actions

    rows, points, control = trace_fixture(four_flow_source, native)
    control["rows"][0]["point"] = points[0] = float("inf")
    with pytest.raises(ValueError):
        emission_actions(rows, points, control)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_id", "other-sample"),
        ("prediction_at", moment(126)),
        ("input_available_at", moment(126)),
        ("input_sha256", "f" * 64),
        ("point", 2.0),
        ("context", "unknown"),
        ("executed_route", "k99"),
        ("emitted_route", "k99"),
        ("calibration", "unknown"),
    ],
)
def test_action_trace_rejects_other_inputs_points_or_unknown_actions(
    native, four_flow_source, field, value
):
    from mars_titan.memory.failure_actions import emission_actions

    rows, points, control = trace_fixture(four_flow_source, native)
    control["rows"][0][field] = value
    with pytest.raises(ValueError):
        emission_actions(rows, points, control)


@pytest.mark.parametrize(
    "bad", ["none", "missing", "not_list", "not_dict", "short", "duplicate", "foreign", "points"]
)
def test_action_trace_rejects_missing_duplicate_or_foreign_rows(native, four_flow_source, bad):
    from mars_titan.memory.failure_actions import emission_actions

    rows, points, control = trace_fixture(four_flow_source, native)
    if bad == "none":
        control = None
    elif bad == "missing":
        control = {}
    elif bad == "not_list":
        control["rows"] = tuple(control["rows"])
    elif bad == "not_dict":
        control["rows"][0] = None
    elif bad == "short":
        control["rows"].pop()
    elif bad == "duplicate":
        control["rows"][1] = control["rows"][0]
    elif bad == "points":
        points.pop()
    else:
        control["rows"][0]["flow_id"] = "US/OTHER"
    with pytest.raises(ValueError):
        emission_actions(rows, points, control)


def test_action_trace_uses_identity_instead_of_row_order(native, four_flow_source):
    from mars_titan.memory.failure_actions import emission_actions

    rows, points, control = trace_fixture(four_flow_source, native)
    control["rows"][0]["context"] = "parent"
    control["rows"].reverse()
    assert emission_actions(rows, points, control) == [
        "mechanism/parent/k1/none/emitted/k1",
        "mechanism/off/k1/none/emitted/k1",
        "mechanism/off/k1/none/emitted/k1",
        "mechanism/off/k1/none/emitted/k1",
    ]


@pytest.mark.parametrize("mechanism", [False, True])
def test_action_source_must_match_the_real_binding(native, four_flow_source, tmp_path, mechanism):
    from test_failure_session import consumer

    source = four_flow_source
    engine = engines(source, "cpu")[3] if mechanism else consumer(source)
    config = policy(source, engine, tmp_path)
    if not mechanism:
        config = replace(config, action_source="mechanism_trace_v1")
    with pytest.raises(ValueError, match="procedencia de acción"):
        open_session(native, source, engine, tmp_path / "wrong-source", config)
    assert not (tmp_path / "wrong-source" / "latest.json").exists()


@pytest.mark.parametrize("action_source", [None, True, "unknown", []])
def test_action_source_configuration_is_closed(four_flow_source, tmp_path, action_source):
    engine = engines(four_flow_source, "cpu")[3]
    with pytest.raises(ValueError, match="procedencia de acción"):
        replace(policy(four_flow_source, engine, tmp_path), action_source=action_source)


def test_failure_before_commit_preserves_mechanism_trace_and_book(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    engine = engines(source, "cpu", online=True)[3]
    config = mechanism_policy(source, engine, tmp_path)
    phase = FinancialPhase("validation", moment(125), moment(126), moment(200), moment(201))
    path = tmp_path / "interrupted"
    with open_session(native, source, engine, path, config, phase=phase) as run:
        run.step(source["batches"][125], [], kind="warmup")
        event = run.step(source["batches"][126], [])
        before, saved_book = run.snapshot(), book(run).export()
        saved_trace = run.mechanism_event(2)
        labels = [native.Feedback(p.id, 0, moment(127), p.value + 1) for p in event.predictions]

        def fail(boundary):
            if boundary == native.Boundary.before_commit:
                raise RuntimeError("Corte técnico del contexto de acciones")

        with pytest.raises(RuntimeError, match="Corte técnico"):
            run.step(source["batches"][127], labels, fault=fail)
    with open_session(native, source, engine, path, config, phase=phase, resume=True) as run:
        assert run.snapshot() == before
        assert book(run).export() == saved_book
        assert run.mechanism_event(2) == saved_trace
        replay = run.step(source["batches"][127], labels)
        assert book(run).counts()["admitted"] == 4
        points = [(p.asset, p.value) for p in replay.predictions]
    with open_session(native, source, engine, tmp_path / "reference", config, phase=phase) as run:
        run.step(source["batches"][125], [], kind="warmup")
        event = run.step(source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), p.value + 1) for p in event.predictions]
        result = run.step(source["batches"][127], labels)
        assert [(p.asset, p.value) for p in result.predictions] == points


def test_action_trace_keeps_executed_and_emitted_routes_separate(native, four_flow_source):
    from mars_titan.memory.failure_actions import emission_actions

    rows, points, control = trace_fixture(four_flow_source, native)
    control["rows"][0].update(executed_route="k4rc", emitted_route="k2c")
    assert emission_actions(rows, points, control)[0] == "mechanism/off/k4rc/none/emitted/k2c"


def test_empty_action_trace_has_no_emissions():
    from mars_titan.memory.failure_actions import emission_actions

    assert emission_actions([], [], dict(rows=[])) == []


def test_failure_context_uses_the_action_of_the_actual_selector(native, four_flow_source, tmp_path):
    from test_mechanism_selection import selected_data

    source = four_flow_source
    engine = selected_data(source)[3]
    config = mechanism_policy(source, engine, tmp_path)
    phase = FinancialPhase("validation", moment(125), moment(126), moment(200), moment(201))
    with open_session(native, source, engine, tmp_path / "selected", config, phase=phase) as run:
        run.step(source["batches"][125], [], kind="warmup")
        run.step(source["batches"][126], [])
        trace = run.mechanism_event(2)["payload"]["selection"]
        assert trace["groups"][0]["action"] == "parent/k1/online"
        assert {r.context.action for r in book(run)._pending.values()} == {
            "mechanism/parent/k1/online/emitted/k1"
        }
        engine.verify(strong=True)


def test_failure_book_and_alternative_collection_mature_the_same_actual_prediction(
    native, four_flow_source, tmp_path
):
    from mars_titan.memory.financial_session import FinancialSession
    from mars_titan.memory.native_mechanism_collection import MechanismCollectionConfig
    from mars_titan.memory.retention_bank import RetentionConfig

    source = four_flow_source
    engine = engines(source, "cpu")[3]
    config = mechanism_policy(source, engine, tmp_path)
    with FinancialSession(
        tmp_path / "collection",
        native=native,
        consumer=engine,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=RetentionConfig(capacity=4, frontier=2, new_candidates=4),
        phase=FinancialPhase("validation", moment(125), moment(126), moment(200), moment(201)),
        world="failure_collection_fixture",
        fold="0",
        admission="m1",
        block_rows=2,
        mechanism_collection=MechanismCollectionConfig(),
        failure_memory=config,
    ) as run:
        run.step(source["batches"][125], [], kind="warmup")
        issued = run.step(source["batches"][126], [])
        assert run.mechanism_collection(2)["supervision"]["outcomes"] == [None] * 4
        labels = [native.Feedback(p.id, 0, moment(127), p.value + 1) for p in issued.predictions]
        run.step(source["batches"][127], labels)
        assert all(v is not None for v in run.mechanism_collection(2)["supervision"]["outcomes"])
        assert book(run).counts()["resolved"] == 4
        assert all(
            r.target - r.emission.prediction == pytest.approx(1) for r in book(run)._retained
        )
        engine.verify(strong=True)


def test_restore_rejects_changed_failure_action_source(
    native, four_flow_source, tmp_path, monkeypatch
):
    from pathlib import Path

    from mars_titan.memory import failure_actions

    source = four_flow_source
    engine = engines(source, "cpu")[3]
    config = mechanism_policy(source, engine, tmp_path)
    path = tmp_path / "code-version"
    with open_session(native, source, engine, path, config) as run:
        run.step(source["batches"][125], [])
        saved = run.snapshot()
    replacement = tmp_path / "failure_actions.py"
    replacement.write_bytes(Path(failure_actions.__file__).read_bytes())
    monkeypatch.setattr(failure_actions, "__file__", str(replacement))
    with open_session(native, source, engine, path, config, resume=True) as run:
        assert run.snapshot() == saved
    with replacement.open("ab") as stream:
        stream.write(b"\n# Otra revision del enlace de acciones.\n")
    with pytest.raises(ValueError):
        open_session(native, source, engine, path, config, resume=True)


def test_changing_action_source_is_rejected_before_the_next_generation(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    engine = engines(source, "cpu")[3]
    config = mechanism_policy(source, engine, tmp_path)
    with open_session(native, source, engine, tmp_path / "runtime-source", config) as run:
        run.step(source["batches"][125], [])
        before = run.snapshot()
        try:
            object.__setattr__(config, "action_source", "fixed_predictor/no_external_tool")
            with pytest.raises(ValueError, match="configuración"):
                run.step(source["batches"][126], [])
        finally:
            object.__setattr__(config, "action_source", "mechanism_trace_v1")
        assert run.snapshot() == before
