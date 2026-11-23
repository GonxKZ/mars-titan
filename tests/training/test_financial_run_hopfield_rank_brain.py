"""Recorrido cronológico con RSHRB (`hopfield_rank_brain`)."""

from collections import Counter

import pytest
import torch

from mars_titan.models.titans.hopfield_rank_brain import (
    ATTRACTIVE_ONLY,
    REPULSIVE,
    REPULSIVE_ONLY,
    SHUFFLED_PEERS,
    HopfieldRankBrainConfig,
)
from tests.training.test_financial_run import entries
from tests.training.test_financial_run_session_objective import (
    ASSETS,
    explicit_fastpath,  # noqa: F401
    predictor,
    records,
    shared,  # noqa: F401
    trainer,
)

PLAIN = dict(session_rank_inputs=None, session_ic_objective=None)


def rshrb(sources, mode=REPULSIVE):
    return predictor(
        sources,
        **PLAIN,
        hopfield_rank_brain=HopfieldRankBrainConfig(
            prototypes=8, steps=2, mode=mode, position="after_mac"
        ),
    )


@pytest.mark.parametrize("mode", [REPULSIVE, ATTRACTIVE_ONLY, REPULSIVE_ONLY, SHUFFLED_PEERS])
def test_chronological_run_replays_whole_sessions_and_reaches_rshrb_gate(shared, tmp_path, mode):  # noqa: F811
    engine = trainer(shared, tmp_path / mode, model=rshrb(shared, mode))
    prepared = []
    original = engine.predictor.prepare

    def spy(batch, state, **options):
        prepared.append((len(set(batch.prediction_at)), options.get("session_complete")))
        return original(batch, state, **options)

    engine.predictor.prepare = spy
    before = engine.predictor._parameter_id
    report = engine.run()
    assert report["status"] == "completed"
    assert prepared and all(count == 1 and whole is True for count, whole in prepared)
    labels = Counter(at for _, _, _, at, _ in entries(engine.audit, "label", "train"))
    assert max(labels.values()) == ASSETS
    gradients = records(engine)
    assert len(gradients) > 2
    first = gradients[0]
    # La puerta arranca en cero sobre W_out no nula: recibe gradiente desde la primera pasada.
    assert first["hopfield_rank_brain.gate"].abs().sum() > 0
    assert torch.isfinite(first["hopfield_rank_brain.gate"]).all()
    assert all(record["head.weight"].abs().sum() > 0 for record in gradients)
    assert engine.predictor._parameter_id == before


def test_split_session_is_refused_and_accumulation_matches_direct_gradients(shared, tmp_path):  # noqa: F811
    split = trainer(shared, tmp_path / "split", model=rshrb(shared), block_rows=2)
    with pytest.raises(ValueError, match="sesión completa"):
        split.run()

    direct = trainer(shared, tmp_path / "direct", model=rshrb(shared))
    direct.run()
    replayed = trainer(shared, tmp_path / "replayed", model=rshrb(shared), accumulation_rows=8)
    replayed.run()
    left, right = records(direct), records(replayed)
    assert len(left) == len(right) > 2
    for first, second in zip(left, right, strict=True):
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)


def test_checkpoint_and_resume_produce_bitwise_identical_predictions(shared, tmp_path):  # noqa: F811
    from tests.training.test_financial_run import StopAtStep

    full = trainer(shared, tmp_path / "full", model=rshrb(shared), checkpoint_updates=1)
    report_full = full.run()
    assert report_full["status"] == "completed"

    # Interrumpir tras dos actualizaciones y reanudar desde el punto de control guardado.
    partial = trainer(shared, tmp_path / "resumed", model=rshrb(shared), checkpoint_updates=1)
    stop = StopAtStep(2)
    stop.trainer = partial
    paused = partial.run(stop=stop)
    assert paused["status"] == "paused"

    resumed = trainer(shared, tmp_path / "resumed", model=rshrb(shared), checkpoint_updates=1)
    report_resumed = resumed.run(resume=True)
    assert report_resumed["status"] == "completed"
    assert report_resumed["best_checkpoint"] == report_full["best_checkpoint"]

    for name, value in full.predictor.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(resumed.predictor.state_dict()[name], value, rtol=0, atol=0)

    rows_full, rows_resumed = [], []
    metrics_full = full.evaluate(shared["validation"], rows=rows_full)
    metrics_resumed = resumed.evaluate(shared["validation"], rows=rows_resumed)
    assert metrics_resumed == metrics_full
    assert rows_resumed == rows_full and len(rows_full) > 0
