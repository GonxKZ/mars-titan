"""Recepción madura y pérdidas condicionales, sin ejecutar un ajuste del selector."""

import importlib

import pytest
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import KINDS
from test_mechanism_collection import DEVICES, collect
from test_mechanism_consumer import calibration, consumer
from test_mechanism_observations import AT, DAY

from mars_titan.calibration.cohort_feedback import CohortOutcome


def api():
    try:
        return importlib.import_module("mars_titan.memory.mechanism_supervision")
    except ModuleNotFoundError:
        pytest.fail("Falta la recepción de resultados maduros de las alternativas")


def outcome(packet, index, *, at=AT + DAY, value=2.0, status="label"):
    row = packet.payload()["rows"][index]
    return CohortOutcome(
        row["flow_id"].split("/")[0],
        AT,
        row["sample_id"],
        at,
        value=value,
        status=status,
        evidence_sha256=None if status == "label" else "e" * 64,
    )


def test_no_target_or_fake_loss_exists_until_its_result_matures():
    packet = collect(consumer())
    state = api().MechanismSupervision.start(packet)
    pending = state.metrics(device="cpu")[0]
    assert pending["mature_rows"] == 0 and pending["pending_rows"] == 2
    assert all(value is None for value in pending["point_losses"])
    assert all(value is None for value in pending["interval_losses"])
    with pytest.raises(ValueError):
        state.receive((outcome(packet, 0, at=AT + DAY - 1),), at=AT + DAY - 1)
    mature = state.receive((outcome(packet, 0),), at=AT + DAY)
    metric = mature.metrics(device="cpu")[0]
    assert metric["mature_rows"] == 1 and metric["pending_rows"] == 1
    assert metric["point_losses"][0] is not None
    assert metric["point_gains"][0] == 0.0
    assert all(value is None for value in metric["point_losses"][1:])
    assert state.metrics(device="cpu")[0] == pending
    with pytest.raises(ValueError):
        mature.receive((outcome(packet, 0),), at=AT + DAY)


@pytest.mark.parametrize("device", DEVICES)
def test_hand_checked_point_and_interval_losses_keep_signed_gains(device):
    from mars_titan.memory.mechanism_collection_record import CollectedMechanisms

    body = collect(consumer(kind=KINDS[0], reader=True, calibrator=calibration())).payload()
    candidates = body["groups"][0]["candidates"]
    # Cuantiles escritos a mano para comprobar la fórmula, no resultados de un modelo.
    for item in candidates:
        item["quantiles"] = [[-2.0, -1.0, 0.0, 1.0, 2.0]] * 2
    for index in range(3, 6):
        candidates[index]["quantiles"] = [[-3.0, -2.0, 1.0, 2.0, 3.0]] * 2
    for index in range(6, 9):
        candidates[index]["quantiles"] = [[-5.0, -4.0, -2.0, 0.0, 1.0]] * 2
    packet = CollectedMechanisms(body)
    state = api().MechanismSupervision.start(packet)
    state = state.receive(tuple(outcome(packet, i) for i in (0, 1)), at=AT + DAY)
    metric = state.metrics(device=device)[0]
    assert metric["point_losses"][:3] == [2.0, 1.0, 4.0]
    assert metric["point_gains"][:3] == [0.0, 1.0, -2.0]
    assert metric["interval_losses"][0] == 8.0
    assert metric["interval_losses"][3] == 5.0
    assert metric["interval_gains"][3] == 3.0
    assert metric["coverage"][0] == [0.0, 1.0]
    assert metric["widths"][0] == [2.0, 4.0]
    assert metric["unit"] == "market_decision_cohort_mean_over_mature_rows"


def test_exclusion_does_not_erase_another_assets_valid_loss_or_become_zero():
    packet = collect(consumer())
    state = api().MechanismSupervision.start(packet)
    state = state.receive(
        (outcome(packet, 0, status="excluded", value=None), outcome(packet, 1)), at=AT + DAY
    )
    metric = state.metrics(device="cpu")[0]
    assert metric["excluded_rows"] == 1 and metric["mature_rows"] == 1
    assert metric["pending_rows"] == 0
    assert metric["point_losses"][0] is not None
    assert metric["mature_sample_ids"] == [packet.payload()["rows"][1]["sample_id"]]


def test_partial_recovery_reproduces_the_next_resolution_and_rejects_another_collection():
    packet = collect(consumer())
    partial = api().MechanismSupervision.start(packet).receive((outcome(packet, 0),), at=AT + DAY)
    restored = api().MechanismSupervision.restore(packet, partial.record())
    result = partial.receive((outcome(packet, 1),), at=AT + DAY + 1)
    recovered = restored.receive((outcome(packet, 1),), at=AT + DAY + 1)
    assert recovered.record() == result.record()
    assert recovered.metrics(device="cpu") == result.metrics(device="cpu")
    foreign = dict(partial.record(), collection_sha256="d" * 64)
    with pytest.raises(ValueError):
        api().MechanismSupervision.restore(packet, foreign)


@pytest.mark.parametrize(
    "field", ["future", "group", "prediction", "sample", "nan", "exclusion", "clock"]
)
def test_invalid_or_future_feedback_cannot_change_the_original_state(field):
    from dataclasses import replace

    packet = collect(consumer())
    state = api().MechanismSupervision.start(packet)
    received = outcome(packet, 0)
    if field == "future":
        received = replace(received, available_at=AT + DAY + 1)
    elif field == "group":
        received = replace(received, group="CN")
    elif field == "prediction":
        received = replace(received, prediction_at=AT - DAY)
    elif field == "sample":
        received = replace(received, sample_id="foreign")
    elif field == "nan":
        received = replace(received, value=float("nan"))
    elif field == "exclusion":
        received = replace(received, status="excluded", value=None, evidence_sha256=None)
    else:
        state = state.receive((received,), at=AT + DAY)
        received = outcome(packet, 1)
    original = state.record()
    with pytest.raises(ValueError):
        state.receive((received,), at=AT + DAY - 1 if field == "clock" else AT + DAY)
    assert state.record() == original


def test_online_return_has_no_requested_interval_loss_but_keeps_valid_point_losses():
    from test_mechanism_collection import full_collection, many_rows

    engine = consumer(kind=KINDS[0], reader=True, calibrator=calibration(max_bytes=4096))
    packet = full_collection(engine, many_rows(engine, 8))
    state = (
        api()
        .MechanismSupervision.start(packet)
        .receive(tuple(outcome(packet, i) for i in range(8)), at=AT + DAY)
    )
    metric = state.metrics(device="cpu")[0]
    assert all(value is not None for value in metric["point_losses"])
    assert all(metric["interval_losses"][i] is None for i in range(2, 36, 3))
    assert all(metric["interval_gains"][i] is None for i in range(2, 36, 3))


@pytest.mark.parametrize("device", DEVICES)
def test_finite_extreme_means_do_not_overflow_their_reductions(device):
    import sys

    from mars_titan.memory.mechanism_collection_record import CollectedMechanisms

    body = collect(consumer(kind=KINDS[0], reader=True, calibrator=calibration())).payload()
    maximum = sys.float_info.max
    for candidate in body["groups"][0]["candidates"]:
        candidate["quantiles"] = [[0.0, 0.0, maximum, maximum, maximum]] * 2
    packet = CollectedMechanisms(body)
    state = (
        api()
        .MechanismSupervision.start(packet)
        .receive(tuple(outcome(packet, i, value=0.0) for i in (0, 1)), at=AT + DAY)
    )
    metric = state.metrics(device=device)[0]
    assert metric["point_losses"][0] == maximum
    assert metric["interval_losses"][0] == maximum
    assert metric["widths"][0] == [maximum, maximum]
