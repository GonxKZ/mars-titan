"""Paridad de la validación cronológica con la sesión financiera congelada y su ejecutor nativo."""

import test_native_episode_backend as backend_fixtures
import torch
from test_financial_observation_source import source

from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import validated_cpu_batch
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from tests.training.test_financial_run import RecordingOptimizer

native = backend_fixtures.native


def test_validation_pass_matches_the_frozen_session(native, tmp_path):
    from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
    from mars_titan.memory.financial_session import FinancialSession
    from mars_titan.memory.retention_bank import RetentionConfig
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
    from mars_titan.training.prefix_eligibility import PrefixTargetVerifier

    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        api, dataset, stream, phase = source(tmp_path)
        specification = stream.specification()
        later = FinancialPhase(
            "validation",
            phase.warmup_start,
            1_672_531_200_000_000,
            1_704_067_200_000_000,
            1_704_067_200_000_000,
        )
        validation = api.FinancialObservationSource(
            dataset, api.prepare_observation_index(dataset, tmp_path / "later", phase=later)
        )

        def model():
            config = FinancialConfig(specification, variant="mac_online", hidden_size=32)
            return FinancialPredictor(config, dtype=torch.float64)

        engine = ChronologicalTrainer(
            model(),
            ChronologicalRecipe(
                truncation=2,
                epochs=1,
                block_rows=1,
                selection=dict(metric="session_mae", patience=2, min_delta=0.0),
            ),
            train=stream,
            validation=validation,
            output=tmp_path / "trainer",
            optimizer_factory=RecordingOptimizer,
            audit=True,
        )
        engine.evaluate(stream)
        issued = {(e[2], e[3]): e[4] for e in engine.audit if e[0] == "prediction"}
        frozen = model()
        frozen.eval().requires_grad_(False)
        codec = FrozenEpisodeCodec(specification)
        options = dict(
            native=native,
            consumer=FrozenFinancialConsumer(frozen),
            codec=codec,
            prefixes=PrefixTargetVerifier(dataset, source_manifest=tmp_path / "materialized.json"),
            retention=RetentionConfig(),
            phase=phase,
            admission="m0",
            world="fixture",
            fold="0",
        )
        expected, pending = {}, {}
        with FinancialSession(tmp_path / "session", **options) as session:
            for event in stream.events():
                feedback = [
                    session.native.Feedback(pending.pop((flow, at)), 0, event.at, label)
                    for flow, at, label in event.labels
                ]
                batches = [validated_cpu_batch(raw, specification) for raw in event.inputs]
                kind = (
                    ("warmup" if event.at < phase.decision_start else "decision")
                    if batches
                    else "settlement"
                )
                result = session.step(
                    batches, feedback, kind=kind, cutoff=event.at, close_phase=event.close_phase
                )
                for prediction in result.predictions:
                    pending[prediction.asset, prediction.decision_at] = prediction.id
                    expected[prediction.asset, prediction.decision_at] = prediction.value
        assert len(expected) >= 2 and issued == expected
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
