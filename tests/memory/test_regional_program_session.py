"""Reservas regionales en la sesión nativa, con pesos y etiquetas de fixture fijos."""

import hashlib
import inspect
import json
import sys
from contextlib import contextmanager
from copy import deepcopy

import pytest
import torch
from test_financial_mechanisms import assert_same
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mechanism_consumer import calibration
from test_native_episode_backend import native as native

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.native_mechanism_collection import MechanismCollectionConfig
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))
RESOURCES = (
    "core_observations",
    "context_queries",
    "selector_decisions",
    "local_node_evaluations",
    "summary_evaluations",
    "mac_observations",
    "reader_refinements",
    "reader_searches",
    "context_commits",
)


def require_session_api():
    parameters = inspect.signature(FinancialSession).parameters
    assert "regional_work_limits" in parameters, "Falta la reserva durable de FinancialSession"
    assert "regional_max_attempts" in parameters, "Falta el límite de intentos de FinancialSession"
    assert parameters["regional_work_limits"].default is None
    assert parameters["regional_max_attempts"].default == 2


def limits():
    return dict.fromkeys(RESOURCES, 1_000_000)


def engines(source, device, *, active=True, online=True, variant="mac_online", calibrated=False):
    program = RegionalExecutionConfig(program="summary_interleaved_v1") if active else None
    model = (
        FinancialPredictor(
            FinancialConfig(
                source["spec"],
                variant=variant,
                hidden_size=32,
                head=QUANTILE_HEAD,
                session_context="market_memory_v1" if online else None,
                regional_fusion=RegionalFusionConfig(topology="ring"),
                regional_execution=program,
            ),
            device=device,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    if online:
        weights = model.state_dict()
        weights["session_context.output"] = torch.eye(32, device=device, dtype=torch.float64) * 0.2
        model.load_state_dict(weights)
    catalog = MechanismCatalog(regional_execution=program) if active else MechanismCatalog()
    consumer = FrozenMechanismConsumer(
        model,
        catalog=catalog,
        action=MechanismAction(
            "parent" if online else "off", "k1", "online" if calibrated else "none"
        ),
        calibration=calibration() if calibrated else None,
    )
    return None, source["spec"], source["codec"], consumer, source["prefixes"]


def session(native, output, data, *, resume=False, block_rows=2, collection=False, **options):
    return FinancialSession(
        output,
        native=native,
        consumer=data[3],
        codec=data[2],
        prefixes=data[4],
        retention=RetentionConfig(capacity=4, frontier=2, new_candidates=4),
        phase=FinancialPhase("validation", moment(125), moment(126), moment(130), moment(131)),
        admission="m1",
        world="regional_program_fixture",
        fold="0",
        resume=resume,
        block_rows=block_rows,
        mechanism_collection=MechanismCollectionConfig() if collection else None,
        **options,
    )


def work_index(output):
    return json.loads((output / "regional-work.json").read_text())


def attempt(output):
    return work_index(output)["cohorts"][-1]["attempts"][-1]


def assert_reference(output, reference):
    payload = (output / "artifacts" / reference["name"]).read_bytes()
    assert len(payload) == reference["bytes"]
    assert hashlib.sha256(payload).hexdigest() == reference["sha256"]


def artifact_bytes(output):
    return {
        path.name: path.read_bytes() for path in (output / "artifacts").iterdir() if path.is_file()
    }


def predictions(result):
    return [(row.id, row.asset, row.decision_at, row.value) for row in result.predictions]


@contextmanager
def observed_calls(consumer):
    """Observar llamadas durante la reapertura sin sustituir métodos identificados."""
    model = consumer.predictor
    watched = {
        FinancialPredictor.prepare.__code__: "predictor_prepare",
        FrozenMechanismConsumer.prepare.__code__: "consumer_prepare",
        FrozenMechanismConsumer.feedback.__code__: "feedback",
    }
    if model.mac is not None:
        watched[type(model.mac).forward.__code__] = "mac"
    if model.session_context is not None:
        watched[type(model.session_context).commit.__code__] = "context_commit"
    counts = dict.fromkeys(watched.values(), 0)

    def observe(frame, event, argument):
        if event == "call" and frame.f_code in watched:
            counts[watched[frame.f_code]] += 1

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("bad", ["absent", "missing", "boolean", "unknown"])
def test_active_program_requires_explicit_exact_resource_limits(
    native, four_flow_source, tmp_path, bad
):
    require_session_api()
    data = engines(four_flow_source, "cpu")
    supplied = limits()
    if bad == "absent":
        supplied = None
    elif bad == "missing":
        supplied.pop("core_observations")
    elif bad == "boolean":
        supplied["core_observations"] = False
    else:
        supplied["invented_work"] = 1
    output = tmp_path / bad
    with pytest.raises(ValueError):
        session(native, output, data, regional_work_limits=supplied)
    assert not (output / "latest.json").exists()


@pytest.mark.parametrize("device", DEVICES)
def test_reservation_precedes_fault_and_abandoned_reference_survives_recovery_and_prune(
    native, four_flow_source, tmp_path, device
):
    require_session_api()
    data = engines(four_flow_source, device)
    output = tmp_path / "attempts"
    captured = {}
    captured_payload = {}
    options = dict(regional_work_limits=limits(), collection=True)
    with session(native, output, data, **options) as run:
        previous_head = (output / "latest.json").read_bytes()

        def fail(boundary):
            if boundary == native.Boundary.before_predictions:
                current = work_index(output)["cohorts"][-1]
                captured.update(deepcopy(current["attempts"][-1]))
                assert captured["status"] == "reserved" and captured["observed_units"] is None
                assert captured["units"]["core_observations"] == 12
                assert captured["units"]["local_node_evaluations"] == 240
                assert current["charged_units"] == captured["units"]
                assert_reference(output, captured["reference"])
                captured_payload.update(
                    run.artifacts.read(
                        captured["reference"],
                        identity=run.contract_id,
                        kind="regional_work_reservation",
                    )
                )
                assert (output / "latest.json").read_bytes() == previous_head
                raise RuntimeError("Corte regional anterior a cualquier predicción")

        with pytest.raises(RuntimeError, match="anterior a cualquier predicción"):
            run.step(four_flow_source["batches"][126], [], fault=fail)
    assert (output / "latest.json").read_bytes() == previous_head
    with session(native, output, data, resume=True, **options) as run:
        first = attempt(output)
        assert first["reservation_id"] == captured["reservation_id"]
        assert first["status"] == "abandoned" and first["observed_units"] is None
        assert_reference(output, captured["reference"])
        issued = run.step(four_flow_source["batches"][126], [])
        assert len(issued.predictions) == 4
        cohort = work_index(output)["cohorts"][0]
        assert [row["status"] for row in cohort["attempts"]] == ["abandoned", "confirmed"]
        assert cohort["charged_units"] == {
            name: 2 * value for name, value in captured["units"].items()
        }
        for index in (127, 128):
            run.step(four_flow_source["batches"][index], [])
        run.step([], [], kind="settlement", cutoff=moment(130))
        assert not (output / "checkpoint-1.json").exists()
        archived = run.mechanism_event(1)["payload"]["regional_work"]
        assert archived["receipts"][0] == captured_payload
        assert archived["cohort"]["attempts"][0]["reference"] == captured["reference"]
        assert archived["cohort"]["attempts"][0]["status"] == "abandoned"
        assert archived["cohort"]["charged_units"] == {
            name: 2 * value for name, value in captured["units"].items()
        }
        assert run.snapshot()["observed"] == 12
        assert sum(row["observed_steps"] for row in run.fast_state()["references"].values()) == 12
        expected = run.snapshot()
    with session(native, output, data, resume=True, **options) as restored:
        assert restored.snapshot() == expected
        assert (
            restored.mechanism_event(1)["payload"]["regional_work"]["receipts"][0]
            == captured_payload
        )
        for cohort in work_index(output)["cohorts"]:
            for receipt in cohort["attempts"]:
                assert_reference(output, receipt["reference"])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("blocked_by", ["quota", "attempts"])
def test_an_abandoned_attempt_does_not_refund_quota_on_reopen(
    native, four_flow_source, tmp_path, device, blocked_by
):
    require_session_api()
    data = engines(four_flow_source, device)
    output = tmp_path / "spent"
    bounded = limits()
    if blocked_by == "quota":
        bounded["core_observations"] = 12
    options = dict(
        regional_work_limits=bounded,
        regional_max_attempts=2 if blocked_by == "quota" else 1,
        collection=True,
    )
    with session(native, output, data, **options) as run:
        previous_head = (output / "latest.json").read_bytes()

        def fail(boundary):
            if boundary == native.Boundary.before_predictions:
                assert attempt(output)["units"]["core_observations"] == 12
                raise RuntimeError("Cargo regional reservado sin emisión")

        with pytest.raises(RuntimeError, match="sin emisión"):
            run.step(four_flow_source["batches"][126], [], fault=fail)
    with session(native, output, data, resume=True, **options) as restored:
        before = (output / "regional-work.json").read_bytes()
        assert attempt(output)["status"] == "abandoned"
        called = []
        with pytest.raises(ValueError):
            restored.step(four_flow_source["batches"][126], [], fault=called.append)
        assert not called
        assert (output / "regional-work.json").read_bytes() == before
    assert (output / "latest.json").read_bytes() == previous_head


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("block_rows", [1, 4])
def test_the_collection_and_emission_budget_covers_the_whole_cohort_before_first_block(
    native, four_flow_source, tmp_path, device, block_rows
):
    require_session_api()
    data = engines(four_flow_source, device)
    output = tmp_path / "global"
    bounded = limits()
    bounded["core_observations"] = 11
    with session(
        native,
        output,
        data,
        block_rows=block_rows,
        collection=True,
        regional_work_limits=bounded,
    ) as run:
        previous_head = (output / "latest.json").read_bytes()
        before = (output / "regional-work.json").read_bytes()
        called = []
        with pytest.raises(ValueError):
            run.step(four_flow_source["batches"][126], [], fault=called.append)
        assert not called
        assert (output / "latest.json").read_bytes() == previous_head
        assert (output / "regional-work.json").read_bytes() == before


@pytest.mark.parametrize(
    "variant,online,collection,core,mac",
    [
        ("mac_online", True, True, 12, 12),
        ("mac_disabled", False, True, 8, 8),
        ("transformer_direct", False, True, 8, 0),
        ("mac_online", False, False, 4, 4),
    ],
)
def test_reservation_counts_core_calls_and_actual_mac_presence(
    native, four_flow_source, tmp_path, variant, online, collection, core, mac
):
    require_session_api()
    data = engines(four_flow_source, "cpu", online=online, variant=variant)
    output = tmp_path / "bound"
    with session(native, output, data, collection=collection, regional_work_limits=limits()) as run:

        def fail(boundary):
            if boundary == native.Boundary.before_predictions:
                units = attempt(output)["units"]
                assert set(units) == set(RESOURCES)
                assert units["core_observations"] == core
                assert units["mac_observations"] == mac
                assert units["selector_decisions"] == core
                assert units["local_node_evaluations"] == 20 * core
                assert units["summary_evaluations"] == 2 * core
                assert units["reader_refinements"] == units["reader_searches"] == 0
                assert units["context_commits"] == int(online)
                if online:
                    assert 0 < units["context_queries"] <= core
                else:
                    assert units["context_queries"] == 0
                raise RuntimeError("Cota comprobada antes de ejecutar el núcleo")

        with pytest.raises(RuntimeError, match="antes de ejecutar el núcleo"):
            run.step(four_flow_source["batches"][126], [], fault=fail)


@pytest.mark.parametrize("device", DEVICES)
def test_a_native_commit_recovers_without_replaying_mac_context_or_feedback(
    native, four_flow_source, tmp_path, device
):
    require_session_api()
    data = engines(four_flow_source, device, calibrated=True)
    options = dict(collection=True, regional_work_limits=limits())
    with session(native, tmp_path / "reference", data, **options) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        first = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(row.id, 0, moment(127), 0.125) for row in first.predictions]
        second = run.step(four_flow_source["batches"][127], labels)
        expected, expected_fast = run.snapshot(), run.fast_state()
        expected_state, expected_event = run.mechanism_state(), run.mechanism_event(3)
        later_labels = [
            native.Feedback(row.id, 0, moment(128), 0.125) for row in second.predictions
        ]
        later = run.step(four_flow_source["batches"][128], later_labels)
        following, following_state = run.snapshot(), run.mechanism_state()
    output = tmp_path / "committed"
    with session(native, output, data, **options) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])

        def fail(boundary):
            if boundary == native.Boundary.committed:
                assert attempt(output)["status"] == "reserved"
                assert json.loads((output / "latest.json").read_text())["generation"] == 3
                raise RuntimeError("Corte tras commit nativo y antes del cierre regional")

        with pytest.raises(RuntimeError, match="antes del cierre regional"):
            run.step(four_flow_source["batches"][127], labels, fault=fail)
    with observed_calls(data[3]) as calls:
        restored = session(native, output, data, resume=True, **options)
    with restored:
        assert all(count == 0 for count in calls.values()), calls
        assert restored.snapshot() == expected
        assert_same(restored.fast_state(), expected_fast)
        assert_same(restored.mechanism_state(), expected_state)
        assert restored.mechanism_event(3) == expected_event
        assert attempt(output)["status"] == "confirmed"
        cohort = work_index(output)["cohorts"][-1]
        assert len(cohort["attempts"]) == 1
        actual = restored.step(four_flow_source["batches"][128], later_labels)
        assert predictions(actual) == predictions(later)
        assert restored.snapshot() == following
        assert_same(restored.mechanism_state(), following_state)
        assert restored.snapshot()["applied"] == 8


@pytest.mark.parametrize("device", DEVICES)
def test_settlement_at_the_decision_cutoff_has_its_own_zero_predictive_reservation(
    native, four_flow_source, tmp_path, device
):
    require_session_api()
    data = engines(four_flow_source, device)
    output = tmp_path / "same_cutoff"
    options = dict(regional_work_limits=limits())
    with session(native, output, data, **options) as run:
        first = run.step(four_flow_source["batches"][126], [])
        run.step(four_flow_source["batches"][127], [])
        before_fast = run.fast_state()
        previous = deepcopy(attempt(output))
        labels = [native.Feedback(row.id, 0, moment(127), 0.125) for row in first.predictions]
        result = run.step([], labels, kind="settlement", cutoff=moment(127))
        assert len(result.applied) == 4 and not result.predictions
        current = attempt(output)
        assert current["reservation_id"] != previous["reservation_id"]
        assert current["ordinal"] == 1 and current["status"] == "confirmed"
        assert current["units"] == dict.fromkeys(RESOURCES, 0)
        assert_same(run.fast_state(), before_fast)
        bundle = run._bundle(run.snapshot()["state"])
        control = run._read(bundle["control"], "control")
        link = run._regional_work.archive_payload(current["reservation_id"])
        assert set(link) == {"reservation_id", "reference", "cohort", "receipts"}
        assert bundle["regional_work"] == control["regional_work"] == link
        expected = run.snapshot()
        before_index = (output / "regional-work.json").read_bytes()
        with pytest.raises(ValueError):
            run.step([], [], kind="settlement", cutoff=moment(127))
        assert (output / "regional-work.json").read_bytes() == before_index
    with session(native, output, data, resume=True, **options) as restored:
        assert restored.snapshot() == expected


@pytest.mark.parametrize("damage", ["index", "receipt"])
def test_resume_rejects_a_missing_regional_root_before_pruning_other_artifacts(
    native, four_flow_source, tmp_path, damage
):
    require_session_api()
    data = engines(four_flow_source, "cpu")
    output = tmp_path / damage
    options = dict(regional_work_limits=limits())
    with session(native, output, data, **options) as run:
        run.step(four_flow_source["batches"][126], [])
        reference = attempt(output)["reference"]
    damaged = (
        output / "regional-work.json"
        if damage == "index"
        else output / "artifacts" / reference["name"]
    )
    damaged.unlink()
    previous_head, previous_artifacts = (
        (output / "latest.json").read_bytes(),
        artifact_bytes(output),
    )
    with pytest.raises(ValueError):
        session(native, output, data, resume=True, **options)
    assert (output / "latest.json").read_bytes() == previous_head
    assert artifact_bytes(output) == previous_artifacts
    assert not damaged.exists()


@pytest.mark.parametrize("change", ["limits", "attempts"])
def test_resume_cannot_reset_the_regional_limits_or_attempt_cap(
    native, four_flow_source, tmp_path, change
):
    require_session_api()
    data = engines(four_flow_source, "cpu")
    output = tmp_path / change
    options = dict(regional_work_limits=limits(), regional_max_attempts=2)
    with session(native, output, data, **options) as run:
        run.step(four_flow_source["batches"][126], [])
    before_index, before_head = (
        (output / "regional-work.json").read_bytes(),
        (output / "latest.json").read_bytes(),
    )
    changed = deepcopy(options)
    if change == "limits":
        changed["regional_work_limits"]["core_observations"] += 1
    else:
        changed["regional_max_attempts"] = 3
    with pytest.raises(ValueError):
        session(native, output, data, resume=True, **changed)
    assert (output / "regional-work.json").read_bytes() == before_index
    assert (output / "latest.json").read_bytes() == before_head


@pytest.mark.parametrize("device", DEVICES)
def test_none_preserves_legacy_bundles_predictions_and_next_state_without_a_work_index(
    native, four_flow_source, tmp_path, device
):
    require_session_api()
    data = engines(four_flow_source, device, active=False)
    results = []
    for name, options in (
        ("implicit", {}),
        ("explicit", dict(regional_work_limits=None, regional_max_attempts=2)),
    ):
        output = tmp_path / name
        with session(native, output, data, **options) as run:
            emitted = []
            for index in (126, 127):
                emitted.append(predictions(run.step(four_flow_source["batches"][index], [])))
            bundle = run._bundle(run.snapshot()["state"])
            control = run._read(bundle["control"], "control")
            assert "regional_work" not in bundle and "regional_work" not in control
            assert not (output / "regional-work.json").exists()
            results.append((emitted, run.snapshot(), run.fast_state(), run.mechanism_state()))
        with session(native, output, data, resume=True, **options) as restored:
            assert restored.snapshot() == results[-1][1]
            assert not (output / "regional-work.json").exists()
    assert_same(results[0], results[1])
