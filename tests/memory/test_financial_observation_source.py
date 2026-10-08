"""Orden causal de observaciones y maduraciones desde las filas del corpus."""

import importlib
import json

import numpy as np
import pytest
import test_native_episode_backend as backend_fixtures
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.test_historical_corpus_inputs import supervised

native = backend_fixtures.native


def source(tmp_path):
    dataset = CorpusDataset(supervised(tmp_path), input_policy=HISTORICAL_MASKED)
    api = importlib.import_module("mars_titan.memory.financial_observations")
    phase = FinancialPhase(
        "train",
        946_684_800_000_000,
        946_684_800_000_000,
        1_672_531_200_000_000,
        1_672_531_200_000_000,
    )
    manifest = api.prepare_observation_index(dataset, tmp_path / "index", phase=phase)
    return api, dataset, api.FinancialObservationSource(dataset, manifest), phase


def test_index_preserves_unlabeled_rows_and_adds_only_declared_closure(tmp_path):
    _, _, stream, phase = source(tmp_path)
    events = list(stream.events())
    assert sum(sum(len(b["sample_ids"]) for b in e.inputs) for e in events) == 2
    assert [e.at for e in events] == sorted({e.at for e in events})
    assert events[-1].at == phase.close_at and events[-1].close_phase
    assert events[-1].inputs == () and events[-1].labels == ()
    assert all(not {"target", "reason"} & set(b) for e in events for b in e.inputs)
    assert any(e.inputs == () and e.labels for e in events)


def test_index_reuses_confirmed_artifact_and_rejects_source_change(tmp_path):
    api, dataset, stream, phase = source(tmp_path)
    before = stream.path.read_bytes()
    result = api.prepare_observation_index(dataset, tmp_path / "index", phase=phase, resume=True)
    assert result == stream.path and result.read_bytes() == before
    path = dataset._file(dataset.assets[0], "samples")
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        list(stream.events())


def test_index_summary_is_verified_before_the_first_event(tmp_path):
    api, dataset, stream, _ = source(tmp_path)
    metadata = json.loads(stream.path.read_text())
    metadata["groups"][0][1] += 1
    stream.path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        api.FinancialObservationSource(dataset, stream.path)


def test_completed_cursor_does_not_decode_historical_modalities(tmp_path, monkeypatch):
    _, _, stream, _ = source(tmp_path)
    monkeypatch.setattr(stream, "_input", lambda *a: pytest.fail("Volvió a decodificar el prefijo"))
    assert list(stream.events(start_cursor=len(stream.metadata["groups"]))) == []


def test_inputs_loaded_by_position_equal_the_unfiltered_reader(tmp_path):
    _, dataset, stream, phase = source(tmp_path)
    expected = {
        b["sample_ids"][0]: b
        for b in dataset.observation_batches(
            start=phase.warmup_start, end=phase.decision_end, batch_size=1
        )
    }
    for event in stream.events():
        for batch in event.inputs:
            for row, identity in enumerate(batch["sample_ids"]):
                for name, value in batch["inputs"].items():
                    np.testing.assert_array_equal(value[row], expected[identity]["inputs"][name][0])


def test_index_drives_real_frozen_consumer_and_recovery(native, tmp_path):
    from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
    from mars_titan.memory.financial_session import FinancialSession
    from mars_titan.memory.retention_bank import RetentionConfig
    from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
    from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
    from mars_titan.training.prefix_eligibility import PrefixTargetVerifier

    api, dataset, stream, phase = source(tmp_path)
    specification = stream.specification()
    codec = FrozenEpisodeCodec(specification)
    model = FinancialPredictor(
        FinancialConfig(specification, variant="mac_online", hidden_size=32), dtype=torch.float64
    )
    model.eval().requires_grad_(False)
    readout = EpisodicReadout(
        EpisodicReadoutConfig(codec.fingerprint(), hidden_size=32), dtype=torch.float64
    )
    readout.eval().requires_grad_(False)
    consumer = FrozenFinancialConsumer(model, readout=readout)
    options = dict(
        native=native,
        consumer=consumer,
        codec=codec,
        prefixes=PrefixTargetVerifier(dataset, source_manifest=tmp_path / "materialized.json"),
        retention=RetentionConfig(),
        phase=phase,
        admission="m1",
        world="fixture",
        fold="0",
    )
    with FinancialSession(tmp_path / "run", **options) as session:
        result = api.run_observation_source(stream, session)
        assert result["admitted"] == 1 and result["pending"] == 0
        snapshot = session.snapshot()
        assert snapshot["observed"] == snapshot["issued"] == 2
        assert snapshot["applied"] == snapshot["finalized"] == 1
    with FinancialSession(tmp_path / "run", **options, resume=True) as restored:
        assert api.run_observation_source(stream, restored) == result
        assert restored.snapshot() == snapshot
