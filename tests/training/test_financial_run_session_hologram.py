"""Recorrido cronológico con interferómetro holográfico de sesión (`session_hologram`)."""

from collections import Counter

import pytest
import torch

from mars_titan.models.titans.session_hologram import SessionHologramConfig
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


def hsi(sources, mode="interferometric"):
    return predictor(
        sources,
        **PLAIN,
        session_hologram=SessionHologramConfig(mode=mode, position="after_mac"),
    )


@pytest.mark.parametrize(
    "mode", ["interferometric", "in_phase_only", "self_only", "shuffled_peers"]
)
def test_chronological_run_replays_whole_sessions_and_reaches_hsi_gate(shared, tmp_path, mode):  # noqa: F811
    engine = trainer(shared, tmp_path / mode, model=hsi(shared, mode))
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
    assert first["session_hologram.gate"].abs().sum() > 0
    assert torch.isfinite(first["session_hologram.gate"]).all()
    assert all(record["head.weight"].abs().sum() > 0 for record in gradients)
    assert engine.predictor._parameter_id == before


def test_split_session_is_refused_and_accumulation_matches_direct_gradients(shared, tmp_path):  # noqa: F811
    split = trainer(shared, tmp_path / "split", model=hsi(shared), block_rows=2)
    with pytest.raises(ValueError, match="sesión completa"):
        split.run()

    direct = trainer(shared, tmp_path / "direct", model=hsi(shared))
    direct.run()
    replayed = trainer(shared, tmp_path / "replayed", model=hsi(shared), accumulation_rows=8)
    replayed.run()
    left, right = records(direct), records(replayed)
    assert len(left) == len(right) > 2
    for first, second in zip(left, right, strict=True):
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)
