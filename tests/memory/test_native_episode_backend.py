"""Enlace real al banco y al ejecutor, con episodios y estados escalares sintéticos."""

import hashlib
import json
import os

import numpy as np
import pytest

from mars_titan.memory.native_backend import load_native


@pytest.fixture
def native():
    path = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    if not path:
        pytest.skip("Falta el enlace nativo CPU compilado para esta comprobación")
    return load_native(path)


def bank(native, capacity=2):
    scope = native.MemoryScope()
    scope.world, scope.partition, scope.fold = "fixture", "train", "fold0"
    scope.representation = "fixed64"
    return native.EpisodicMemory(scope, 13, capacity)


def record(native, identity, first=3.0, second=4.0):
    result = native.MemoryRecord()
    result.id = identity
    result.decision_at = 2 * identity
    result.available_at = result.decision_at
    result.maturity_at = result.decision_at + 1
    result.key = [first, second] + [0.0] * 62
    result.value = [float(identity)] * 64
    result.label, result.label_valid = 0.25 * identity, True
    return result


def test_bank_retention_normalization_and_recovery_use_native_code(native):
    first = bank(native)
    first.write(record(native, 1), 3)
    first.write(record(native, 2), 5)
    first.retain_batch([record(native, 3), record(native, 4, 0, 2)], [1, 4], 9)
    retained = first.retained_records()
    assert [r.id for r in retained] == [1, 4]
    np.testing.assert_array_equal(retained[0].key, native.normalize_key(record(native, 1).key))
    assert first.seen == 4
    restored = bank(native)
    restored.restore_bytes(first.snapshot_bytes())
    assert [r.id for r in restored.retained_records()] == [1, 4]
    assert [n.record.id for n in restored.query([1.0] + [0.0] * 63, 9)] == [1, 4]
    assert restored.query([1.0] + [0.0] * 63, 2) == []
    with pytest.raises(ValueError):
        restored.retain_batch([record(native, 5)], [1, 20], 11)
    assert [r.id for r in restored.retained_records()] == [1, 4]


def test_batch_validation_precedes_selection_and_does_not_change_memory(native):
    first = bank(native)
    ready = first.validate_batch([record(native, 1)], 3)
    assert ready[0].key == native.normalize_key(record(native, 1).key)
    assert first.seen == first.size == 0
    with pytest.raises(ValueError):
        first.validate_batch([record(native, 1)], 2)


def test_artifact_files_use_checked_atomic_writes_and_reject_corruption(native, tmp_path):
    value = b"opaque-state-without-publication"
    reference = native.seal_blob(str(tmp_path / "artifacts"), 1024, value)
    assert reference["sha256"] == hashlib.sha256(value).hexdigest()
    assert reference["bytes"] == len(value)
    assert native.seal_blob(str(tmp_path / "artifacts"), 1024, value) == reference
    path = tmp_path / "artifacts" / reference["name"]
    assert native.read_blob(str(path), 1024) == value
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        native.seal_blob(str(path.parent), 1024, value)
    linked = tmp_path / "linked"
    linked.symlink_to(path)
    with pytest.raises(ValueError):
        native.read_blob(str(linked), 1024)


def definition(native):
    result = native.Definition()
    identity = native.Identity()
    for name, value in zip(
        ("source_sha256", "view_sha256", "representation_sha256", "model_sha256"),
        "abcd",
        strict=True,
    ):
        setattr(identity, name, value * 64)
    result.identity = identity
    result.tasks = [native.Task("return", 1), native.Task("event", 2)]
    result.initial_state_json = json.dumps(dict(fast=0, labels=0))
    result.prediction_mode = native.PredictionMode.prepared
    return result


def cohort(native, cursor):
    result = native.Cohort()
    result.cursor, result.cutoff = cursor, 10 * (cursor + 1)
    result.observations = [native.Observation("US/B", 1, [2]), native.Observation("US/A", 1, [1])]
    return result


def callbacks():
    def prepare(rows, tasks, cutoff, before_json, batch_rows):
        before = json.loads(before_json)
        assert [r.asset for r in rows] == ["US/A", "US/B"]
        assert cutoff > 0 and batch_rows > 0
        values = [r.features[0] + before["fast"] for r in rows for _ in tasks]
        return values, json.dumps(dict(fast=before["fast"] + 1, labels=before["labels"]))

    def update(proposal_json, labels):
        state = json.loads(proposal_json)
        state["labels"] += len(labels)
        return json.dumps(state)

    return prepare, update


def test_native_executor_confirms_the_prepared_proposal_with_mature_labels(native, tmp_path):
    prepare, update = callbacks()
    run = native.Executor(str(tmp_path / "run"), definition(native), prepare, update, False)
    first = run.step(cohort(native, 0), [], 1)
    assert len(first.predictions) == 4
    label = native.Feedback(first.predictions[0].id, 0, 20, 7.0)
    second = run.step(cohort(native, 1), [label], 2)
    assert [p.value for p in second.predictions] == [2.0, 2.0, 3.0, 3.0]
    assert json.loads(run.snapshot_json())["state"] == dict(fast=2, labels=1)
    run.close()
    restored = native.Executor(str(tmp_path / "run"), definition(native), prepare, update, True)
    assert restored.cursor == 2
    assert len(restored.pending()) == 7
    assert json.loads(restored.snapshot_json())["state"] == dict(fast=2, labels=1)
    restored.close()


@pytest.mark.parametrize("boundary", ["record_written", "before_commit", "committed"])
def test_python_callback_failure_recovers_only_the_confirmed_generation(native, tmp_path, boundary):
    prepare, update = callbacks()
    output = str(tmp_path / boundary)
    run = native.Executor(output, definition(native), prepare, update, False)

    def fail(point):
        if point == getattr(native.Boundary, boundary):
            raise RuntimeError("Corte de prueba")

    with pytest.raises(RuntimeError, match="Corte de prueba"):
        run.step(cohort(native, 0), [], 1, fail)
    run.close()
    restored = native.Executor(output, definition(native), prepare, update, True)
    if boundary != "committed":
        assert restored.cursor == 0
        restored.step(cohort(native, 0), [], 2)
    assert restored.cursor == 1
    assert json.loads(restored.snapshot_json())["state"] == dict(fast=1, labels=0)
    restored.close()
