"""Referencias rápidas vivas, selección y compactación CPU exactas."""

import copy
import importlib

import pytest
import test_native_episode_backend as backend_fixtures
import torch
from test_financial_adapter import setup

from mars_titan.memory.session_artifacts import SessionArtifacts
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

native = backend_fixtures.native


def create(native, tmp_path, **options):
    model, batch = setup()
    model.eval().requires_grad_(False)
    consumer = FrozenFinancialConsumer(model)
    artifacts = SessionArtifacts(native, tmp_path / "artifacts", max_files=64)
    api = importlib.import_module("mars_titan.memory.financial_state_artifacts")
    store = api.FinancialStateArtifacts(consumer, artifacts, identity="e" * 64, **options)
    return store, model, batch


def advanced(store, model, batch):
    empty = store.empty()
    initial = model.initial_state(batch.flow_ids)
    state = store.gather(empty, batch.flow_ids, initial_state=initial)
    prepared = model.prepare(batch, state)
    manifest = store.replace(empty, state, prepared.next_state)
    return manifest, prepared.next_state


def test_unknown_flows_require_explicit_initial_state(native, tmp_path):
    store, model, batch = create(native, tmp_path)
    with pytest.raises(ValueError):
        store.gather(store.empty(), batch.flow_ids)
    manifest, expected = advanced(store, model, batch)
    state = store.gather(manifest, batch.flow_ids)
    for actual, original in zip(state.mac.memory.weights, expected.mac.memory.weights, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        assert actual.data_ptr() != original.data_ptr()
    assert state.last_sample_ids == expected.last_sample_ids


def test_references_match_rows_identity_and_cursors(native, tmp_path):
    store, model, batch = create(native, tmp_path)
    manifest, _ = advanced(store, model, batch)
    for field, value in (
        ("row", 1),
        ("observed_steps", 2),
        ("last_prediction_at", 0),
        ("block_id", "a" * 64),
        ("parameter_id", "a" * 64),
    ):
        changed = copy.deepcopy(manifest)
        changed["references"][batch.flow_ids[0]][field] = value
        with pytest.raises(ValueError):
            store.verify(changed)


def test_compaction_is_cpu_storage_only_and_next_prediction_is_exact(native, tmp_path):
    store, model, batch = create(native, tmp_path)
    manifest = store.empty()
    for index in range(2):
        one = batch.select([index])
        previous = model.initial_state(one.flow_ids)
        result = model.prepare(one, previous)
        manifest = store.replace(manifest, previous, result.next_state)
    assert len(manifest["blocks"]) == 2
    before = store.gather(manifest, batch.flow_ids)
    packed = store.compact(manifest)
    after = store.gather(packed, batch.flow_ids)
    assert len(packed["blocks"]) == 1
    for a, b in zip(
        before.mac.memory.weights + before.mac.memory.momentum,
        after.mac.memory.weights + after.mac.memory.momentum,
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.data_ptr() != b.data_ptr()
    assert torch.equal(before.observed_steps, after.observed_steps)
    assert before.last_prediction_at == after.last_prediction_at
    from test_financial_adapter import raw_batch

    from mars_titan.models.titans.financial_inputs import DecisionBatch

    following = DecisionBatch.from_corpus(
        raw_batch(at=batch.prediction_at[0] + 1), model.config.inputs, dtype=torch.float64
    )
    torch.testing.assert_close(
        model.prepare(following, before).point_predictions,
        model.prepare(following, after).point_predictions,
        rtol=0,
        atol=0,
    )
    assert len(store.live_references(manifest)) == 2
    assert len(store.live_references(packed)) == 1


def test_missing_payload_cannot_silently_initialize_a_flow(native, tmp_path):
    store, model, batch = create(native, tmp_path)
    manifest, _ = advanced(store, model, batch)
    reference = next(iter(manifest["blocks"].values()))
    (store.artifacts.directory / reference["name"]).unlink()
    with pytest.raises((ValueError, RuntimeError)):
        store.gather(manifest, batch.flow_ids)


def test_unique_source_budget_is_checked_before_payload_loading(native, tmp_path, monkeypatch):
    store, model, batch = create(native, tmp_path)
    manifest, _ = advanced(store, model, batch)
    strict, _, _ = create(native, tmp_path / "other", max_source_bytes=1)
    monkeypatch.setattr(
        strict.artifacts, "read", lambda *a, **k: pytest.fail("Leyó antes del límite")
    )
    with pytest.raises(ValueError):
        strict.gather(manifest, batch.flow_ids)


def test_strict_session_roundtrip_does_not_require_historical_presence(native, tmp_path):
    from test_financial_adapter import raw_batch, specification

    from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
    from mars_titan.memory.episodic_session import EpisodicSession, SessionPreparation
    from mars_titan.memory.retention_bank import RetentionConfig
    from mars_titan.models.titans.financial_inputs import validated_cpu_batch

    spec = specification(historical=False)
    raw = raw_batch(absent=False)
    raw.pop("presence")
    batch = validated_cpu_batch(raw, spec)
    with EpisodicSession(
        tmp_path / "strict",
        native=native,
        codec=FrozenEpisodeCodec(spec),
        retention=RetentionConfig(),
        model_id="a" * 64,
        task="residual",
        horizon=1,
        world="fixture",
        partition="train",
        fold="0",
        prepare=lambda rows, *args: SessionPreparation({r.flow_id: 1.0 for r in rows}, {}),
    ) as run:
        assert len(run.step([batch], []).predictions) == 2
        bundle = run._bundle(run.snapshot()["state"])
        payload = run._read(bundle["inputs"][0], "inputs")
        payload["presence"][0, 1] = False
        reference = run._stage(payload, "inputs")
        with pytest.raises(ValueError):
            run._input_rows(reference)
