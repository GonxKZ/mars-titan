"""Ventanas fuente auditables, sin construir regresiones ni levantar el bloqueo."""

import importlib
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_financial_mechanisms import engines
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native
from test_native_mechanism_collection import session

from mars_titan.training.learning_hold import LearningHoldError
from mars_titan.training.mechanism_fit_contract import MechanismFitConfig

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def api():
    try:
        return importlib.import_module("mars_titan.training.mechanism_fit_window")
    except ModuleNotFoundError:
        pytest.fail("Falta el contrato de ventanas para las fuentes de ajuste")


def config(parent="a" * 64, **changes):
    options = dict(
        parent_consumer_id=parent,
        parent_selected_before=moment(125),
        fit_start=moment(126),
        fit_end=moment(128),
        ridge=0.01,
        interval_weight=0.0,
        min_gain=0.0,
        market_weights=(("US", 1.0),),
        costs=(0.0,) * 36,
        min_cohorts=2,
        source_mode="native_windows",
    )
    return MechanismFitConfig(**dict(options, **changes))


def first_two(run, native, batches):
    first = run.step(batches[126], [])
    labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions[:2]]
    second = run.step(batches[127], labels)
    return first, second


def test_window_protocol_has_a_new_identity_and_preserves_the_closed_default():
    chosen = config()
    closed = replace(chosen, source_mode="closed_phases")
    assert chosen.identity()["kind"] == "windowed_native_cohort_mechanism_fit_v1"
    assert closed.identity()["kind"] == "closed_native_cohort_mechanism_fit_v1"
    assert chosen.fingerprint() != closed.fingerprint()
    assert "source_mode" not in closed.identity()
    with pytest.raises(ValueError):
        replace(chosen, source_mode="automatic")


@pytest.mark.parametrize("device", DEVICES)
def test_audit_reads_a_past_window_of_an_open_or_later_closed_session(
    native, four_flow_source, tmp_path, device
):
    data = engines(four_flow_source, device)
    with session(native, tmp_path / "source", data) as run:
        first, second = first_two(run, native, four_flow_source["batches"])
        request = api().MechanismFitWindow(run, (1, 2), through_generation=2)
        protocol = config(data[3].model_id)
        snapshot = run.snapshot()
        rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone() if device != "cpu" else None
        audit = api().audit_fit_windows((request,), config=protocol)
        assert audit["observed_cohorts"] == 2
        assert audit["mature_rows"] == 2
        assert audit["pending_rows"] == 6
        assert audit["excluded_rows"] == audit["censored_rows"] == 0
        assert audit["sources"][0]["before"] == moment(128)
        assert run.snapshot() == snapshot and torch.equal(torch.random.get_rng_state(), rng)
        if cuda_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
        labels = [native.Feedback(p.id, 0, moment(127), 99.0) for p in first.predictions[2:]]
        labels += [native.Feedback(p.id, 0, moment(128), -0.25) for p in second.predictions[:2]]
        run.step([], labels, kind="settlement", cutoff=moment(129))
        run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        assert api().audit_fit_windows((request,), config=protocol) == audit
        closed = api().audit_fit_windows(
            (api().MechanismFitWindow(run, (1, 2), through_generation=4),),
            config=replace(protocol, fit_end=moment(132)),
        )
        assert closed["mature_rows"] == 6
        assert closed["censored_rows"] == 2
        assert closed["pending_rows"] == closed["excluded_rows"] == 0
    with session(native, tmp_path / "source", data, resume=True) as restored:
        assert (
            api().audit_fit_windows(
                (api().MechanismFitWindow(restored, (1, 2), through_generation=2),), config=protocol
            )
            == audit
        )


def test_fit_still_blocks_before_window_or_protocol_inspection(learning_hold):
    from mars_titan.training.mechanism_policy_fit import _cohorts, _window_cohorts, fit_policy

    learning_hold(False)
    seen = []

    class Forbidden:
        def __iter__(self):
            seen.append(True)
            raise AssertionError("Se inspeccionó una ventana bajo el bloqueo")

    for method in (_cohorts, _window_cohorts, fit_policy):
        with pytest.raises(LearningHoldError):
            method(Forbidden(), config=None, device="invalid")
    assert seen == []


def test_closed_protocol_keeps_its_captured_identity():
    from tests.training.test_mechanism_policy_fit import configuration

    assert (
        configuration().fingerprint()
        == "738cf57a55e5550754c330e3dd45a0f7a5854db142aea9c5dfe966dca017bca3"
    )


@pytest.mark.parametrize(
    "generations,through",
    [([1], 1), ((), 1), ((True,), 1), ((0,), 1), ((2, 1), 2), ((1, 1), 1), ((2,), 1), ((1,), 257)],
)
def test_invalid_window_request_is_rejected_without_reading(
    native, four_flow_source, tmp_path, monkeypatch, generations, through
):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "invalid", data) as run:

        def forbidden(*args):
            raise AssertionError("La petición abrió un archivo antes de validarse")

        with monkeypatch.context() as guard:
            guard.setattr(native, "read_blob", forbidden)
            with pytest.raises(ValueError):
                api().MechanismFitWindow(run, generations, through_generation=through)


def test_nonoverlapping_requests_share_global_byte_and_cohort_limits(
    native, four_flow_source, tmp_path
):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "budgets", data) as run:
        first_two(run, native, four_flow_source["batches"])
        protocol = config(data[3].model_id)
        requests = tuple(
            api().MechanismFitWindow(run, (generation,), through_generation=2)
            for generation in (1, 2)
        )
        singles = [api().audit_fit_windows((request,), config=protocol) for request in requests]
        full = api().audit_fit_windows(requests, config=protocol)
        total = sum(item["source_bytes"] for item in singles)
        assert full["source_bytes"] == total
        assert full["observed_cohorts"] == 2
        assert api().audit_fit_windows(requests, config=protocol, max_bytes=total) == dict(
            full, sources=[dict(s, budget_bytes=total) for s in full["sources"]]
        )
        head = run.snapshot()
        with pytest.raises((ValueError, RuntimeError), match="bytes|size|limit|límite|budget"):
            api().audit_fit_windows(requests, config=protocol, max_bytes=total - 1)
        with pytest.raises(ValueError, match="acotadas"):
            api().audit_fit_windows(requests, config=replace(protocol, max_sources=1))
        with pytest.raises(ValueError, match="cohorte"):
            api().audit_fit_windows((requests[0], requests[0]), config=protocol)
        assert run.snapshot() == head
        run.step(four_flow_source["batches"][128], [])
        all_three = tuple(
            api().MechanismFitWindow(run, (g,), through_generation=3) for g in (1, 2, 3)
        )
        with pytest.raises(ValueError, match="cohortes acumuladas"):
            api().audit_fit_windows(
                all_three, config=replace(protocol, fit_end=moment(130), max_cohorts=2)
            )


def test_exact_reception_boundary_wrong_parent_and_earlier_decision_are_rejected(
    native, four_flow_source, tmp_path
):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "boundaries", data) as run:
        first_two(run, native, four_flow_source["batches"])
        run.step([], [], kind="settlement", cutoff=moment(129))
        request = api().MechanismFitWindow(run, (1, 2), through_generation=3)
        protocol = config(data[3].model_id, fit_end=moment(129))
        with pytest.raises(ValueError, match="futuro"):
            api().audit_fit_windows((request,), config=protocol)
        earlier = api().MechanismFitWindow(run, (1, 2), through_generation=2)
        with pytest.raises(ValueError, match="tramo"):
            api().audit_fit_windows((earlier,), config=replace(protocol, fit_start=moment(127)))
        with pytest.raises(ValueError, match="padre"):
            api().audit_fit_windows(
                (earlier,), config=replace(protocol, parent_consumer_id="b" * 64)
            )
        with pytest.raises(ValueError, match="declara ventanas"):
            api().audit_fit_windows(
                (earlier,), config=replace(protocol, source_mode="closed_phases")
            )


def test_source_head_change_after_reading_is_not_accepted(
    native, four_flow_source, tmp_path, monkeypatch
):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "changed", data) as run:
        first_two(run, native, four_flow_source["batches"])
        module = api()
        read = module._read

        def advance(*args, **options):
            result = read(*args, **options)
            run.step([], [], kind="settlement", cutoff=moment(129))
            return result

        monkeypatch.setattr(module, "_read", advance)
        with pytest.raises(ValueError, match="cambió durante"):
            module.audit_fit_windows(
                (module.MechanismFitWindow(run, (1, 2), through_generation=2),),
                config=config(data[3].model_id),
            )


def test_replaced_session_callback_is_rejected_before_calling_it(
    native, four_flow_source, tmp_path, monkeypatch
):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "callback", data) as run:
        first_two(run, native, four_flow_source["batches"])
        request = api().MechanismFitWindow(run, (1, 2), through_generation=2)
        touched = []

        def forbidden():
            touched.append(True)
            raise AssertionError("Se invocó un callback sustituido antes de comprobarlo")

        with monkeypatch.context() as guard:
            guard.setattr(run, "snapshot", forbidden)
            with pytest.raises(ValueError, match="callbacks|métodos"):
                api().audit_fit_windows((request,), config=config(data[3].model_id))
        assert touched == []


def test_cohort_transport_preserves_signed_gains_absences_and_market_weighting():
    from mars_titan.training.mechanism_policy_fit import _append_cohorts, _finish_cohorts

    protocol = config(market_weights=(("CN", 0.25), ("US", 0.75)))
    used, keys, budgets = [], set(), [123, 0]
    for at, market, gain in ((3, "US", -0.2), (1, "CN", 0.3), (2, "US", -0.1)):
        observation = dict(
            cutoff=protocol.fit_start + at, groups=[dict(market=market, values=[0.0] * 21)]
        )
        metric = [
            dict(
                mature_rows=2,
                excluded_rows=1,
                point_gains=[0.0, gain] + [None] * 10,
                interval_gains=[0.0] + [None] * 35,
            )
        ]
        _append_cohorts(observation, metric, "c" * 64, protocol, used, keys, budgets)
    rows, weights, provenance = _finish_cohorts(used, [], budgets, protocol)
    assert [r["point"][1] for r in rows] == [0.3, -0.1, -0.2]
    assert all(r["point"][2:] == [None] * 10 for r in rows)
    assert all(r["interval"][1:] == [None] * 35 for r in rows)
    assert weights == [0.25, 0.375, 0.375]
    assert provenance["source_bytes"] == 123
    assert provenance["cohorts_per_market"] == {"CN": 1, "US": 2}


def test_no_label_cohort_is_counted_without_becoming_a_zero_gain_row():
    from mars_titan.training.mechanism_policy_fit import _append_cohorts

    used, keys, budgets = [], set(), [0, 0]
    protocol = config()
    observation = dict(cutoff=protocol.fit_start, groups=[dict(market="US", values=[0.0] * 21)])
    metrics = [dict(mature_rows=0)]
    _append_cohorts(observation, metrics, "c" * 64, protocol, used, keys, budgets)
    assert used == [] and budgets == [0, 1]
    assert keys == {("US", protocol.fit_start)}


def test_exclusions_remain_absent_targets_in_window_audit(native, four_flow_source, tmp_path):
    from mars_titan.memory.financial_session import FinancialPhase
    from mars_titan.models.titans.financial_inputs import validated_cpu_batch

    data = engines(four_flow_source, "cpu")
    early = []
    for batch in four_flow_source["batches"][126]:
        early.append(
            validated_cpu_batch(
                dict(
                    inputs={name: np.zeros_like(value) for name, value in batch.inputs.items()},
                    presence=batch.presence.copy(),
                    sample_ids=[f"{flow}/{moment(64)}" for flow in batch.flow_ids],
                    prediction_at=np.array([moment(64)], dtype="datetime64[us]"),
                    input_available_at=np.array([moment(64)], dtype="datetime64[us]"),
                ),
                four_flow_source["spec"],
            )
        )
    phase = FinancialPhase("validation", moment(63), moment(64), moment(130), moment(131))
    with session(native, tmp_path / "excluded", data, phase=phase) as run:
        emitted = run.step(early, [])
        assert len(emitted.excluded) == 4
        request = api().MechanismFitWindow(run, (1,), through_generation=1)
        result = api().audit_fit_windows(
            (request,),
            config=config(
                data[3].model_id, parent_selected_before=moment(63), fit_start=moment(64)
            ),
        )
        assert result["mature_rows"] == result["pending_rows"] == result["censored_rows"] == 0
        assert result["excluded_rows"] == 4


@pytest.mark.parametrize("maximum", [0, True, 1024**3 + 1])
def test_invalid_audit_limit_is_rejected_before_reading(
    native, four_flow_source, tmp_path, monkeypatch, maximum
):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "limit", data) as run:
        request = api().MechanismFitWindow(run, (1,), through_generation=1)

        def forbidden(*args):
            raise AssertionError("Se abrió un archivo con un presupuesto inválido")

        with monkeypatch.context() as guard:
            guard.setattr(native, "read_blob", forbidden)
            with pytest.raises(ValueError):
                api().audit_fit_windows(
                    (request,), config=config(data[3].model_id), max_bytes=maximum
                )
