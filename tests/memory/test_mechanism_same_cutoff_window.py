"""Regresión del prefijo nativo con decisión y recepción al mismo corte."""

import copy

import pytest
import torch
from test_financial_mechanisms import engines
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mechanism_fit_window import config as fit_config
from test_native_episode_backend import native as native
from test_native_mechanism_collection import session

from mars_titan.training.mechanism_fit_window import MechanismFitWindow, iter_fit_windows

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def read_prefix(run, interface, *, through, before):
    """Usar las tres APIs reales y exponer sus resultados sin recalcularlos."""
    if interface == "single":
        result = run.mechanism_collection_at(1, through_generation=through, before=before)
        return (
            result,
            [
                (
                    1,
                    result["supervision"],
                    result["censored"],
                    result["as_of"]["resolutions"],
                )
            ],
            result["as_of"],
        )
    if interface == "window":
        result = run.mechanism_collections_at((1, 2), through_generation=through, before=before)
        return (
            result,
            [
                (item["generation"], item["supervision"], item["censored"], item["resolutions"])
                for item in result["collections"]
            ],
            result["as_of"],
        )
    request = MechanismFitWindow(run, (1, 2), through_generation=through)
    result = tuple(
        iter_fit_windows((request,), config=fit_config(run.consumer.model_id, fit_end=before))
    )
    assert len(result) == 1
    collections, provenance = result[0]
    return (
        result,
        [
            (
                metadata["generation"],
                supervision.record(),
                metadata["censored"],
                metadata["resolutions"],
            )
            for _, supervision, metadata in collections
        ],
        provenance,
    )


def immutable_read(run, interface, *, through, before, device):
    head = run.snapshot()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device != "cpu" else None
    try:
        return read_prefix(run, interface, through=through, before=before)
    finally:
        assert run.snapshot() == head
        assert torch.equal(torch.random.get_rng_state(), cpu_rng)
        if cuda_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)


@pytest.mark.parametrize("interface", ("single", "window", "fit"))
@pytest.mark.parametrize("device", DEVICES)
def test_same_cutoff_settlement_is_readable_without_backdating_late_receipts(
    native, four_flow_source, tmp_path, interface, device
):
    data = engines(four_flow_source, device)
    output = tmp_path / "same_cutoff_window"
    with session(native, output, data) as run:
        first = run.step(four_flow_source["batches"][126], [])
        second = run.step(four_flow_source["batches"][127], [])
        assert len(first.predictions) == len(second.predictions) == 4
        early = [
            native.Feedback(prediction.id, 0, moment(127), 0.125)
            for prediction in first.predictions[:2]
        ]
        received = run.step([], early, kind="settlement", cutoff=moment(127))
        assert len(received.applied) == 2 and not received.predictions
        assert run.snapshot()["cursor"] == 3

        with pytest.raises(ValueError, match="corte|futuro|anterior"):
            immutable_read(run, interface, through=3, before=moment(127), device=device)
        prior = immutable_read(run, interface, through=2, before=moment(127) + 1, device=device)
        assert prior[1][0][1]["outcomes"] == [None] * 4
        assert [r["generation"] for r in prior[2]["native_records"]] == [1, 2]
        same_clock = immutable_read(
            run, interface, through=3, before=moment(127) + 1, device=device
        )
        assert [
            target["value"] if target else None for target in same_clock[1][0][1]["outcomes"]
        ] == [0.125, 0.125, None, None]
        assert same_clock[2]["before"] == moment(127) + 1
        prefix = immutable_read(run, interface, through=3, before=moment(128), device=device)
        _, collections, provenance = prefix
        assert [item[0] for item in collections] == ([1] if interface == "single" else [1, 2])
        assert [record["generation"] for record in provenance["native_records"]] == [1, 2, 3]
        assert [record["cutoff"] for record in provenance["native_records"]] == [
            moment(126),
            moment(127),
            moment(127),
        ]
        assert provenance["before"] == moment(128)
        first_supervision = collections[0][1]
        assert first_supervision["clock"] == moment(127)
        assert [
            target["value"] if target else None for target in first_supervision["outcomes"]
        ] == [0.125, 0.125, None, None]
        assert [target["available_at"] for target in first_supervision["outcomes"][:2]] == [
            moment(127),
            moment(127),
        ]
        assert collections[0][2] == []
        assert collections[0][3] == [
            {
                "generation": 3,
                "row_indices": [0, 1],
                "native_record_sha256": provenance["native_records"][2]["sha256"],
            }
        ]
        if interface != "single":
            assert collections[1][1]["outcomes"] == [None] * 4
            assert collections[1][3] == []

        late = [
            native.Feedback(prediction.id, 0, moment(127), 99.0)
            for prediction in first.predictions[2:]
        ] + [
            native.Feedback(prediction.id, 0, moment(128), -0.25)
            for prediction in second.predictions[:2]
        ]
        run.step([], late, kind="settlement", cutoff=moment(129))
        assert (
            immutable_read(run, interface, through=3, before=moment(128), device=device) == prefix
        )
        with pytest.raises(ValueError, match="corte|futuro|anterior"):
            immutable_read(run, interface, through=4, before=moment(129), device=device)
        complete = immutable_read(run, interface, through=4, before=moment(130), device=device)
        _, complete_collections, complete_provenance = complete
        assert [target["value"] for target in complete_collections[0][1]["outcomes"]] == [
            0.125,
            0.125,
            99.0,
            99.0,
        ]
        assert complete_provenance["native_records"][-1]["cutoff"] == moment(129)
        if interface != "single":
            assert [
                target["value"] if target else None
                for target in complete_collections[1][1]["outcomes"]
            ] == [-0.25, -0.25, None, None]
        run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        assert (
            immutable_read(run, interface, through=3, before=moment(128), device=device) == prefix
        )
    with session(native, output, data, resume=True) as restored:
        assert (
            immutable_read(restored, interface, through=3, before=moment(128), device=device)
            == prefix
        )


@pytest.mark.parametrize("previous", ("warmup", "decision", "settlement"))
@pytest.mark.parametrize("current", ("warmup", "decision", "settlement"))
@pytest.mark.parametrize("at", (9, 10, 11, 12))
def test_clock_order_matches_the_native_event_rule(previous, current, at):
    from mars_titan.memory import mechanism_collection_asof

    check = getattr(mechanism_collection_asof, "_check_event_order", None)
    assert callable(check), "Falta el control de orden de los eventos del prefijo"
    event = dict(cutoff=at, kind=current)
    # Los pares permitidos proceden del contrato nativo, sin repetir su fórmula.
    equal_pairs = {("warmup", "settlement"), ("decision", "settlement")}
    valid = at == 11 or (at == 10 and (previous, current) in equal_pairs)
    if valid:
        check(10, previous, event, 12)
    else:
        with pytest.raises(ValueError, match="corte|orden|futuro|registro"):
            check(10, previous, event, 12)


@pytest.mark.parametrize("changed_kind", ("warmup", "settlement"))
def test_historical_event_kind_must_match_the_native_record(
    native, four_flow_source, tmp_path, monkeypatch, changed_kind
):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "inconsistent_kind", data) as run:
        run.step(four_flow_source["batches"][126], [])
        run.step(four_flow_source["batches"][127], [])
        history = run._binding.history
        original = history.read_event

        def changed(manifest, generation):
            event = copy.deepcopy(original(manifest, generation))
            if generation == 2:
                event["kind"] = changed_kind
            return event

        with monkeypatch.context() as patch:
            patch.setattr(history, "read_event", changed)
            with pytest.raises(ValueError, match="tipo|nativo|evento|registro"):
                immutable_read(run, "single", through=2, before=moment(128), device="cpu")
