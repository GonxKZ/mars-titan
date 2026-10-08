"""Recorrido financiero técnico con observaciones sin etiqueta y estados recuperables."""

import importlib

import numpy as np
import pandas as pd
import pytest
import test_native_episode_backend as backend_fixtures
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.temporal import MarketClock
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import FinancialInputSpec, validated_cpu_batch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.training.corpus_inputs import CorpusDataset, _price_contexts
from mars_titan.training.prefix_eligibility import PrefixTargetVerifier
from tests.training.test_historical_corpus_inputs import supervised

native = backend_fixtures.native


def moment(index):
    clock = MarketClock("US", "2021-01-01", "2023-12-31")
    return int(pd.Timestamp(clock.decisions[index]).value // 1000)


def resources(tmp_path, *, k=None, variant="mac_online"):
    path = supervised(tmp_path)
    dataset = CorpusDataset(path, input_policy=HISTORICAL_MASKED)
    spec = FinancialInputSpec(
        source_sha256=dataset.identity,
        view_sha256="b" * 64,
        representation=dataset.manifest["representation"],
        dimensions=dict(prices=5, news=2, charts=1, fundamentals=3, macro=3),
        input_policy=HISTORICAL_MASKED,
    )
    model = FinancialPredictor(
        FinancialConfig(spec, variant=variant, hidden_size=32), dtype=torch.float64
    )
    model.eval().requires_grad_(False)
    codec = FrozenEpisodeCodec(spec)
    readout = (
        None
        if k is None
        else EpisodicReadout(
            EpisodicReadoutConfig(codec.fingerprint(), hidden_size=32, refinements=k),
            dtype=torch.float64,
        )
    )
    if readout is not None:
        readout.eval().requires_grad_(False)
    engine = FrozenFinancialConsumer(model, readout=readout)
    prefix = PrefixTargetVerifier(dataset, source_manifest=tmp_path / "materialized.json")
    return dataset, spec, codec, engine, prefix


def inputs(data, index):
    dataset, spec, *_ = data
    prices, _ = dataset._prices(dataset.assets[0])
    at = moment(index)
    raw = dict(
        inputs=dict(
            prices=_price_contexts(prices, np.array([index]), 64),
            news=np.zeros((1, 2), np.float32),
            charts=np.array([[3.0]], np.float32),
            fundamentals=np.zeros((1, 3), np.float32),
            macro=np.zeros((1, 3), np.float32),
        ),
        presence=np.array([[True, False, True, False, False]]),
        sample_ids=[f"US/AAA/{at}"],
        prediction_at=np.array([at], dtype="datetime64[us]"),
        input_available_at=np.array([at], dtype="datetime64[us]"),
        reason=object(),
        target=object(),
    )
    return [validated_cpu_batch(raw, spec)]


def session(native, output, data, *, resume=False, admission="m1", **options):
    api = importlib.import_module("mars_titan.memory.financial_session")
    _, _, codec, engine, prefix = data
    phase = api.FinancialPhase("validation", moment(63), moment(64), moment(200), moment(201))
    return api.FinancialSession(
        output,
        native=native,
        consumer=engine,
        codec=codec,
        prefixes=prefix,
        retention=RetentionConfig(policy="reservoir", capacity=4, frontier=2, new_candidates=4),
        phase=phase,
        admission=admission,
        world="fixture",
        fold="0",
        resume=resume,
        **options,
    )


@pytest.mark.parametrize("k", [None, 1, 2, 4])
def test_unlabeled_observations_prefix_exclusion_settlement_and_closure(native, tmp_path, k):
    data = resources(tmp_path, k=k)
    with session(native, tmp_path / "run", data) as run:
        warm = run.step(inputs(data, 63), [], kind="warmup")
        assert not warm.predictions and run.diagnostics()["pending"] == 0
        early = run.step(inputs(data, 64), [])
        assert len(early.predictions) == len(early.excluded) == 1
        assert run.diagnostics()["pending"] == run.diagnostics()["admitted"] == 0
        issued = run.step(inputs(data, 125), [])
        fast_before = run.fast_state()
        run.step(
            [],
            [native.Feedback(issued.predictions[0].id, 0, moment(126), 0.5)],
            kind="settlement",
            cutoff=moment(126),
        )
        assert run.fast_state() == fast_before
        run.step(inputs(data, 127), [])
        run.step(inputs(data, 128), [])
        assert run.diagnostics()["pending"] == 2
        assert run.diagnostics()["admitted"] == 1
        closed = run.step([], [], kind="settlement", cutoff=moment(201), close_phase=True)
        assert len(closed.finalized) == 2 and not closed.applied
        final = run.snapshot()
        assert (
            final["observed"],
            final["issued"],
            final["applied"],
            final["excluded"],
            final["finalized"],
        ) == (5, 4, 1, 1, 2)
        episode = run.retained_episodes()[0]
        assert episode["issued_prediction"] == issued.predictions[0].value
        assert episode["error"] == 0.5 - issued.predictions[0].value
    with session(native, tmp_path / "run", data, resume=True) as restored:
        assert restored.snapshot() == final
        assert restored.diagnostics()["pending"] == 0


@pytest.mark.parametrize("point", ["record_written", "before_commit", "committed"])
def test_recovery_preserves_fast_references_and_single_admission(native, tmp_path, point):
    data = resources(tmp_path)
    with session(native, tmp_path / "reference", data) as reference:
        first = reference.step(inputs(data, 125), [])
        labels = [native.Feedback(first.predictions[0].id, 0, moment(126), 0.5)]
        reference.step(inputs(data, 126), labels)
        expected = reference.snapshot()
    with session(native, tmp_path / "run", data) as run:
        run.step(inputs(data, 125), [])

        def fail(boundary):
            if boundary == getattr(native.Boundary, point):
                raise RuntimeError("Corte de fixture financiero")

        with pytest.raises(RuntimeError, match="Corte de fixture"):
            run.step(inputs(data, 126), labels, fault=fail)
    with session(native, tmp_path / "run", data, resume=True) as restored:
        if point != "committed":
            restored.step(inputs(data, 126), labels)
        assert restored.snapshot() == expected
        assert restored.diagnostics()["admitted"] == 1


def test_m0_preserves_emission_but_never_admits_an_episode(native, tmp_path):
    data = resources(tmp_path)
    with session(native, tmp_path / "run", data, admission="m0") as run:
        first = run.step(inputs(data, 125), [])
        run.step(inputs(data, 126), [native.Feedback(first.predictions[0].id, 0, moment(126), 0.5)])
        assert run.snapshot()["applied"] == 1 and not run.retained_episodes()
    for unsupported in ("m2", "m3"):
        with pytest.raises(ValueError):
            session(native, tmp_path / unsupported, data, admission=unsupported)


def test_m0_requires_an_explicitly_disabled_bank_read(native, tmp_path):
    data = resources(tmp_path, k=1)
    with pytest.raises(ValueError):
        session(native, tmp_path / "run", data, admission="m0")


@pytest.mark.parametrize("partition", ["train", "validation", "calibration", "evaluation"])
def test_each_financial_phase_recovers_its_separate_v2_bank(native, tmp_path, partition):
    api = importlib.import_module("mars_titan.memory.financial_session")
    data = resources(tmp_path)
    phase = api.FinancialPhase(partition, moment(63), moment(64), moment(200), moment(201))
    options = dict(
        native=native,
        consumer=data[3],
        codec=data[2],
        prefixes=data[4],
        retention=RetentionConfig(capacity=4, frontier=2, new_candidates=4),
        phase=phase,
        admission="m1",
        world="fixture",
        fold="0",
    )
    with api.FinancialSession(tmp_path / "run", **options) as run:
        emitted = run.step(inputs(data, 125), [])
        run.step(
            [],
            [native.Feedback(emitted.predictions[0].id, 0, moment(126), 0.5)],
            kind="settlement",
            cutoff=moment(126),
        )
        expected = run.snapshot()
        assert run.diagnostics()["admitted"] == 1
    with api.FinancialSession(tmp_path / "run", resume=True, **options) as restored:
        assert restored.snapshot() == expected


def test_pending_inputs_compact_across_dates_without_inventing_expiry(native, tmp_path):
    data = resources(tmp_path)
    with session(native, tmp_path / "run", data, max_input_blocks=2) as run:
        for index in range(125, 131):
            run.step(inputs(data, index), [])
        assert run.diagnostics()["pending"] == 6
        state = run.snapshot()
    with session(native, tmp_path / "run", data, max_input_blocks=2, resume=True) as restored:
        assert restored.snapshot() == state
        assert restored.diagnostics()["pending"] == 6


def test_parameter_data_change_cannot_publish_another_generation(native, tmp_path):
    data = resources(tmp_path)
    with session(native, tmp_path / "run", data) as run:
        run.step(inputs(data, 125), [])
        latest = (tmp_path / "run/latest.json").read_bytes()
        data[3].predictor.head.weight.data.add_(1)
        with pytest.raises(ValueError):
            run.step(inputs(data, 126), [])
        assert (tmp_path / "run/latest.json").read_bytes() == latest


def test_corrupting_nested_fast_payload_before_commit_keeps_previous_generation(native, tmp_path):
    data = resources(tmp_path)
    with session(native, tmp_path / "run", data) as run:
        run.step(inputs(data, 125), [])
        latest = (tmp_path / "run/latest.json").read_bytes()

        def corrupt(boundary):
            if boundary == native.Boundary.before_commit:
                manifest = run._read(run._proposal["fast"], "fast")
                reference = next(iter(manifest["blocks"].values()))
                (run.artifacts.directory / reference["name"]).write_bytes(b"changed")

        with pytest.raises((ValueError, RuntimeError)):
            run.step(inputs(data, 126), [], fault=corrupt)
        assert (tmp_path / "run/latest.json").read_bytes() == latest


def test_replacing_preparation_callback_cannot_keep_the_model_identity(native, tmp_path):
    data = resources(tmp_path)
    with session(native, tmp_path / "run", data) as run:
        original = run._prepare_event

        def changed(*args):
            values, state = original(*args)
            return [value + 1 for value in values], state

        run._prepare_event = changed
        latest = (tmp_path / "run/latest.json").read_bytes()
        with pytest.raises(ValueError):
            run.step(inputs(data, 125), [])
        assert (tmp_path / "run/latest.json").read_bytes() == latest
