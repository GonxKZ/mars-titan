"""Ciclo temporal real con banco nativo y una preparación escalar sintética."""

import json
import os

import numpy as np
import pytest
import torch
from test_episodic_codec import raw_batch, selected, specification

from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.episodic_session import EpisodicSession, SessionPreparation
from mars_titan.memory.native_backend import load_native
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.titans.financial_inputs import validated_cpu_batch


@pytest.fixture
def native():
    path = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    if not path:
        pytest.skip("Falta el enlace nativo CPU compilado")
    return load_native(path)


def inputs(day, split=False):
    raw = raw_batch(2)
    raw["prediction_at"] += np.timedelta64(day, "us")
    raw["input_available_at"] += np.timedelta64(day, "us")
    raw["sample_ids"] = [
        f"US/A{i:03d}/{int(raw['prediction_at'][i].astype('int64'))}" for i in range(2)
    ]
    blocks = [selected(raw, [1]), selected(raw, [0])] if split else [raw]
    return [validated_cpu_batch(block, specification()) for block in blocks]


def preparer(trace):
    def prepare(rows, fast, bank, context_id, batch_rows):
        count = 0 if fast is None else fast["count"]
        trace.append((context_id, [len(bank.query(row)) for row in rows]))
        return SessionPreparation(
            {row.flow_id: float(row.inputs["prices"][0, 0]) + count for row in rows},
            dict(count=count + 1),
        )

    return prepare


def session(native, output, trace, resume=False):
    return EpisodicSession(
        output,
        native=native,
        codec=FrozenEpisodeCodec(specification()),
        retention=RetentionConfig(policy="anchored", capacity=4, frontier=2, new_candidates=4),
        model_id="d" * 64,
        task="residual",
        horizon=1,
        world="fixture",
        partition="train",
        fold="0",
        prepare=preparer(trace),
        resume=resume,
    )


def feedback(native, predictions, at):
    return [native.Feedback(p.id, 0, at, 3.0) for p in reversed(predictions)]


def test_all_predictions_see_one_snapshot_and_errors_use_the_emitted_prediction(native, tmp_path):
    trace = []
    with session(native, tmp_path / "run", trace) as run:
        first = run.step(inputs(0), [], batch_rows=1)
        second = run.step(
            inputs(1),
            feedback(native, first.predictions, inputs(1)[0].prediction_at[0]),
            batch_rows=2,
        )
        assert trace[0][1] == trace[1][1] == [0, 0]
        assert run.diagnostics()["pending"] == 2
        assert run.diagnostics()["admitted"] == 2
        episodes = run.retained_episodes()
        for episode in episodes:
            original = next(p for p in first.predictions if p.id == episode["prediction_id"])
            assert episode["error"] == 3.0 - original.value
            assert episode["issued_prediction"] == original.value
        run.step(inputs(2), [], batch_rows=1)
        assert trace[-1][1] == [2, 2]
        assert all(p.value != first.predictions[i].value for i, p in enumerate(second.predictions))


def test_physical_input_batches_keep_identical_state_and_context(native, tmp_path):
    traces = [[], []]
    with (
        session(native, tmp_path / "first", traces[0]) as first,
        session(native, tmp_path / "second", traces[1]) as second,
    ):
        a = first.step(inputs(0), [], batch_rows=1)
        b = second.step(inputs(0, split=True), [], batch_rows=2)
        at = inputs(1)[0].prediction_at[0]
        first.step(inputs(1), feedback(native, a.predictions, at), batch_rows=2)
        second.step(inputs(1, split=True), feedback(native, b.predictions, at), batch_rows=1)
        assert first.snapshot() == second.snapshot()
        assert traces[0] == traces[1]


@pytest.mark.parametrize("boundary", ["record_written", "before_commit", "committed"])
def test_session_recovery_never_admits_an_unpublished_proposal(native, tmp_path, boundary):
    baseline = tmp_path / "baseline"
    with session(native, baseline, []) as reference:
        issued = reference.step(inputs(0), [], batch_rows=1)
        labels = feedback(native, issued.predictions, inputs(1)[0].prediction_at[0])
        reference.step(inputs(1), labels, batch_rows=1)
        expected = reference.snapshot()
    output = tmp_path / "interrupted"
    with session(native, output, []) as run:
        run.step(inputs(0), [], batch_rows=2)

        def fail(point):
            if point == getattr(native.Boundary, boundary):
                raise RuntimeError("Corte de prueba")

        with pytest.raises(RuntimeError, match="Corte de prueba"):
            run.step(inputs(1), labels, batch_rows=2, fault=fail)
    with session(native, output, [], resume=True) as resumed:
        if boundary != "committed":
            assert resumed.diagnostics()["admitted"] == 0
            resumed.step(inputs(1), labels, batch_rows=1)
        assert resumed.snapshot() == expected
        assert resumed.diagnostics()["admitted"] == 2


def test_recovery_rejects_changed_referenced_artifacts(native, tmp_path):
    output = tmp_path / "run"
    with session(native, output, []) as run:
        run.step(inputs(0), [], batch_rows=1)
        state = run.snapshot()["state"]
    (output / "artifacts" / state["bundle"]["name"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        session(native, output, [], resume=True)
    assert json.loads((output / "latest.json").read_text())["generation"] == 1


@pytest.mark.parametrize("field", ["horizon", "prediction_at", "label"])
def test_episode_envelope_rejects_numeric_type_aliases(native, tmp_path, field):
    with session(native, tmp_path / "run", []) as run:
        issued = run.step(inputs(0), [])
        run.step(inputs(1), feedback(native, issued.predictions, inputs(1)[0].prediction_at[0]))
        bundle = run._bundle(run.snapshot()["state"])
        payload = run._read(bundle["bank"], "bank")
        first = payload["episodes"][min(payload["episodes"])]
        first[field] = (
            True
            if field == "horizon"
            else float(first[field])
            if field == "prediction_at"
            else int(first[field])
        )
        changed = run._stage(payload, "bank")
        with pytest.raises(ValueError):
            run._bank(changed)


@pytest.mark.parametrize("alteration", ["next_microsecond", "leading_zero", "extra_segment"])
def test_episode_sample_id_matches_its_exact_decision_timestamp(native, tmp_path, alteration):
    with session(native, tmp_path / "run", []) as run:
        issued = run.step(inputs(0), [])
        run.step(inputs(1), feedback(native, issued.predictions, inputs(1)[0].prediction_at[0]))
        reference = run._bundle(run.snapshot()["state"])["bank"]
        payload = run._read(reference, "bank")
        first = payload["episodes"][min(payload["episodes"])]
        at = first["prediction_at"]
        suffix = (
            str(at + 1)
            if alteration == "next_microsecond"
            else f"0{at}"
            if alteration == "leading_zero"
            else f"{at}/extra"
        )
        first["sample_id"] = f"{first['flow_id']}/{suffix}"
        changed = run._stage(payload, "bank")
        with pytest.raises(ValueError):
            run._bank(changed)
        run._bank(reference)


def test_complete_predictor_inputs_are_referenced_and_checked_on_recovery(native, tmp_path):
    output = tmp_path / "run"
    with session(native, output, []) as run:
        run.step(inputs(0), [])
        bundle = run._bundle(run.snapshot()["state"])
        references = bundle["inputs"]
        assert len(references) == 1
        full = run._read(references[0], "inputs")
        assert set(full["inputs"]) == {"prices", "news", "charts", "fundamentals", "macro"}
        expected = inputs(0)[0]
        for name, values in expected.inputs.items():
            np.testing.assert_array_equal(full["inputs"][name].numpy(), values)
        np.testing.assert_array_equal(full["presence"].numpy(), expected.presence)
    (output / "artifacts" / references[0]["name"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        session(native, output, [], resume=True)


def test_prepared_executor_accepts_the_financial_flow_alphabet(native, tmp_path):
    raw = raw_batch(2)
    at = int(raw["prediction_at"][0].astype("int64"))
    raw["sample_ids"] = [f"US/^INDEX/{at}", f"US/CAD=X/{at}"]
    batch = validated_cpu_batch(raw, specification())
    output = tmp_path / "run"
    with session(native, output, []) as run:
        assert len(run.step([batch], []).predictions) == 2
    with session(native, output, [], resume=True) as resumed:
        assert resumed.diagnostics()["pending"] == 2


def test_cpu_storage_and_recovery_ignore_the_default_device(native, tmp_path):
    batches = inputs(0), inputs(1)
    snapshots, episodes, traces = [], [], []
    for device in ("cpu", "meta"):
        output = tmp_path / device
        trace = []
        with torch.device(device):
            with session(native, output, trace) as run:
                issued = run.step(batches[0], [], batch_rows=1)
                labels = feedback(native, issued.predictions, batches[1][0].prediction_at[0])
                run.step(batches[1], labels, batch_rows=2)
                assert run.diagnostics() == dict(cursor=2, admitted=2, retained=2, pending=2)
                snapshots.append(run.snapshot())
                episodes.append(run.retained_episodes())
            with session(native, output, [], resume=True) as recovered:
                assert recovered.snapshot() == snapshots[-1]
                assert recovered.retained_episodes() == episodes[-1]
                bundle = recovered._bundle(recovered.snapshot()["state"])
                pending = recovered._read(bundle["pending"], "pending")
                assert pending["key_inputs"].device.type == pending["values"].device.type == "cpu"
                for reference in bundle["inputs"]:
                    source = recovered._read(reference, "inputs")
                    assert source["presence"].device.type == "cpu"
                    assert all(value.device.type == "cpu" for value in source["inputs"].values())
        traces.append(trace)
    assert snapshots[0] == snapshots[1]
    assert episodes[0] == episodes[1]
    assert traces[0] == traces[1]
