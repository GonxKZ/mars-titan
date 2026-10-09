"""Conectar M2 con emisiones originales y fixtures de etiquetas manuales."""

import copy

import numpy as np
import pytest
import torch
from test_financial_session import moment
from test_financial_session_controls import (
    FLOWS,
    capture_control,
    matured_record,
    paired_consumers,
    tensor_leaves,
)
from test_financial_session_controls import (
    four_flow_source as four_flow_source,
)
from test_financial_session_controls import (
    no_target_estimation as no_target_estimation,
)
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.write_policy import MatureErrorConfig
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer


def open_session(
    native, source, consumer, output, *, admission="m2", block_rows=4, resume=False, retention=None
):
    if admission == "m0":
        consumer = FrozenFinancialConsumer(consumer.predictor)
    if retention is None:
        retention = (
            MatureErrorConfig(capacity=4)
            if admission == "m2"
            else RetentionConfig(capacity=4, frontier=1, new_candidates=2)
        )
    return FinancialSession(
        output,
        native=native,
        consumer=consumer,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=retention,
        phase=FinancialPhase("validation", moment(125), moment(125), moment(200), moment(201)),
        world="manual_m2",
        fold="0",
        admission=admission,
        block_rows=block_rows,
        resume=resume,
    )


def labels(native, predictions, index):
    return [
        native.Feedback(p.id, 0, moment(index), 0.125 * (int(p.asset[-1]) + 1)) for p in predictions
    ]


def test_m2_uses_original_mature_error_without_filtering_observations(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    scores = {}
    with open_session(native, source, consumer, tmp_path / "m2") as session:
        previous = []
        for index in (125, 126, 127, 128):
            result = session.step(source["batches"][index], labels(native, previous, index))
            assert len(result.predictions) == 4
            for item in result.applied:
                scores[len(scores) + 1] = abs(item.label.value - item.prediction.value)
            bundle = session._bundle(session.snapshot()["state"])
            bank, episodes = session._bank(bundle["bank"])
            assert bank.seen == len(scores)
            if scores:
                expected = min(scores, key=lambda identity: (-scores[identity], identity))
                assert bank.index_ids()["selective"] == (expected,)
                assert bank.selective_scores == {expected: scores[expected]}
                assert bank.index_ids()["recent"] == (len(scores),)
                assert set(episodes) <= set(scores)
                assert session.diagnostics()["physical_slots"] == 4
                assert session.diagnostics()["retained"] <= 4
            previous = result.predictions
        final = session.snapshot()
        assert final["issued"] == 16 and final["applied"] == 12
    with open_session(native, source, consumer, tmp_path / "m2", resume=True) as restored:
        assert restored.snapshot() == final


def test_recovery_rejects_a_score_different_from_the_emitted_prediction(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    with open_session(native, source, consumer, tmp_path / "m2") as session:
        first = session.step(source["batches"][125], [])
        session.step(source["batches"][126], labels(native, first.predictions, 126))
        latest = (tmp_path / "m2/latest.json").read_bytes()
        bundle = session._bundle(session.snapshot()["state"])
        payload = copy.deepcopy(session._read(bundle["bank"], "bank"))
        key = next(iter(payload["snapshot"]["scores"]))
        payload["snapshot"]["scores"][key] += 1.0
        reference = session._stage(payload, "bank")
        with pytest.raises(ValueError, match="error|puntuación"):
            session._bank(reference)
        assert (tmp_path / "m2/latest.json").read_bytes() == latest


def test_recovery_rejects_a_selective_index_worse_than_another_retained_episode(
    native, four_flow_source, tmp_path
):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    with open_session(native, source, consumer, tmp_path / "m2") as session:
        first = session.step(source["batches"][125], [])
        second = session.step(source["batches"][126], labels(native, first.predictions, 126))
        bundle = session._bundle(session.snapshot()["state"])
        bank, episodes = session._bank(bundle["bank"])
        worst = min(episodes, key=lambda key: (abs(episodes[key]["error"]), -key))
        assert (worst,) != bank.index_ids()["selective"]
        incoming = [
            matured_record(native, source, outcome, identifier)
            for identifier, outcome in enumerate(second.applied, 1)
        ]
        replacement = bank._new()
        replacement._indices["selective"].retain_batch(incoming, [worst], moment(126))
        payload = copy.deepcopy(session._read(bundle["bank"], "bank"))
        snapshot = payload["snapshot"]
        snapshot["indices"]["selective"] = replacement._indices["selective"].snapshot_bytes()
        snapshot["scores"] = {worst: abs(episodes[worst]["error"])}
        receipt = snapshot["receipt"]
        receipt["index_ids"]["selective"] = [worst]
        union = sorted({key for values in receipt["index_ids"].values() for key in values})
        receipt["retained_ids"] = receipt["new_unique_ids"] = union
        receipt["unique_episodes"] = len(union)
        receipt["duplicate_slots"] = receipt["physical_slots"] - len(union)
        receipt["native_archive_bytes"]["selective"] = len(snapshot["indices"]["selective"])
        payload["episodes"] = {key: episodes[key] for key in union}
        reference = session._stage(payload, "bank")
        with pytest.raises(ValueError, match="error|puntuación|selectivo"):
            session._bank(reference)


@pytest.mark.parametrize("admission", ["m0", "m1"])
def test_previous_admission_routes_keep_their_own_recovery(
    native, four_flow_source, tmp_path, admission
):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    with open_session(
        native, source, consumer, tmp_path / admission, admission=admission
    ) as session:
        first = session.step(source["batches"][125], [])
        session.step(source["batches"][126], labels(native, first.predictions, 126))
        final = session.snapshot()
        assert session.diagnostics()["admitted"] == (4 if admission == "m1" else 0)
    with open_session(
        native, source, consumer, tmp_path / admission, admission=admission, resume=True
    ) as restored:
        assert restored.snapshot() == final


def m2_trajectory(native, source, consumer, output, *, block_rows, reverse=False, recover=False):
    mode = consumer.predictor.local_control.config.mode
    run = open_session(native, source, consumer, output, block_rows=block_rows)
    previous, points, indices, scores, receipts = [], {}, [], [], []
    controls, stores = [], []
    try:
        for index in (125, 126, 127, 128, 129, 130):
            batches = [] if index == 129 else source["batches"][index]
            feedback = labels(native, previous, index)
            if reverse:
                batches, feedback = list(reversed(batches)), list(reversed(feedback))
            options = dict(kind="settlement" if index == 129 else "decision", cutoff=moment(index))
            if recover and index == 128:
                before = run.snapshot()
                latest = (output / "latest.json").read_bytes()

                def fail(boundary):
                    if boundary == native.Boundary.before_commit:
                        raise RuntimeError("Corte M2 antes de publicar")

                with pytest.raises(RuntimeError, match="Corte M2"):
                    run.step(batches, feedback, fault=fail, **options)
                assert (output / "latest.json").read_bytes() == latest
                run.close()
                run = open_session(
                    native, source, consumer, output, block_rows=block_rows, resume=True
                )
                assert run.snapshot() == before
            result = run.step(batches, feedback, **options)
            previous = result.predictions
            for item in previous:
                points[(item.asset, item.decision_at)] = item.value
            bundle = run._bundle(run.snapshot()["state"])
            bank, episodes = run._bank(bundle["bank"])
            indices.append(bank.index_ids())
            scores.append(bank.selective_scores)
            receipts.append(bank.receipt)
            assert len(episodes) == bank.size <= 4
            assert bank.seen == run.snapshot()["applied"]
            controls.append(capture_control(run, mode, 4 if batches else 0))
            stores.append(
                dict(
                    bank=run._read(bundle["bank"], "bank"),
                    pending=run._read(bundle["pending"], "pending"),
                )
            )
        final = run.snapshot()
        assert (final["issued"], final["applied"], run.diagnostics()["pending"]) == (20, 16, 4)
        fast = run._fast_store.gather(run.fast_state(), FLOWS)
        assert fast.observed_steps.tolist() == [5] * 4
        assert fast.mac.memory.steps.tolist() == [5] * 4
        tensors = tensor_leaves(consumer.predictor.export_state_cpu(fast))
    finally:
        run.close()
    with open_session(
        native, source, consumer, output, block_rows=block_rows, resume=True
    ) as restored:
        assert restored.snapshot() == final
    return dict(
        points=points,
        indices=indices,
        scores=scores,
        receipts=receipts,
        controls=controls,
        stores=stores,
        fast=tensors,
        final=final,
    )


def test_m2_preserves_order_partition_and_atomic_recovery(native, four_flow_source, tmp_path):
    source = four_flow_source
    consumer = paired_consumers(source)["disabled"]
    rng = torch.random.get_rng_state().clone()
    reference = m2_trajectory(native, source, consumer, tmp_path / "reference", block_rows=4)
    permuted = m2_trajectory(
        native, source, consumer, tmp_path / "permuted", block_rows=1, reverse=True
    )
    recovered = m2_trajectory(
        native, source, consumer, tmp_path / "recovered", block_rows=4, recover=True
    )
    for other, tolerance in ((permuted, 1e-12), (recovered, 0)):
        assert reference["points"].keys() == other["points"].keys()
        keys = sorted(reference["points"])
        np.testing.assert_allclose(
            [reference["points"][key] for key in keys],
            [other["points"][key] for key in keys],
            rtol=tolerance,
            atol=tolerance,
        )
        assert reference["indices"] == other["indices"]
        assert reference["receipts"] == other["receipts"]
        for expected, actual in zip(reference["scores"], other["scores"], strict=True):
            assert expected == pytest.approx(actual, rel=tolerance, abs=tolerance)
        for key, expected in reference["fast"].items():
            torch.testing.assert_close(expected, other["fast"][key], rtol=tolerance, atol=tolerance)
    assert reference["final"] == recovered["final"]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(parameter.grad is None for parameter in consumer.predictor.parameters())


@pytest.mark.parametrize(
    "admission,retention",
    [
        ("m0", MatureErrorConfig(capacity=4)),
        ("m1", MatureErrorConfig(capacity=4)),
        ("m2", RetentionConfig(capacity=4, frontier=1, new_candidates=2)),
        ("m3", MatureErrorConfig(capacity=4)),
    ],
)
def test_incompatible_write_contracts_fail_before_creating_a_session(
    native, four_flow_source, tmp_path, admission, retention
):
    consumer = paired_consumers(four_flow_source)["disabled"]
    with pytest.raises(ValueError, match="admisión"):
        open_session(
            native,
            four_flow_source,
            consumer,
            tmp_path / "invalid",
            admission=admission,
            retention=retention,
        )
    assert not (tmp_path / "invalid/latest.json").exists()
