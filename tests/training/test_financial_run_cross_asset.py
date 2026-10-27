"""Recorrido cronológico con atención entre activos de la misma sesión (C5 de #751), sin pasos.

Reutiliza el corpus y el entrenador de las pruebas por sesión de #749. El optimizador
registrador guarda los gradientes y no cambia pesos.
"""

from collections import Counter

import pytest
import torch

from mars_titan.models.titans.cross_asset_attention import CrossAssetAttentionConfig
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


def c5(sources, mode="same_session"):
    return predictor(sources, **PLAIN, cross_asset_attention=CrossAssetAttentionConfig(mode=mode))


@pytest.mark.parametrize("mode", ["same_session", "self_only"])
def test_each_update_replays_whole_sessions_and_reaches_c5(shared, tmp_path, mode):  # noqa: F811
    engine = trainer(shared, tmp_path / mode, model=c5(shared, mode))
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
    # W_o empieza en cero: en la primera actualización solo ella recibe gradiente de C5.
    first = gradients[0]
    assert first["cross_asset_attention.out.weight"].abs().sum() > 0
    assert first["cross_asset_attention.qkv.weight"].abs().sum() == 0
    assert all(record["head.weight"].abs().sum() > 0 for record in gradients)
    assert engine.predictor._parameter_id == before


def test_an_instant_split_into_blocks_is_refused(shared, tmp_path):  # noqa: F811
    engine = trainer(shared, tmp_path / "split", model=c5(shared), block_rows=2)
    with pytest.raises(ValueError, match="sesión completa"):
        engine.run()


def test_replay_by_flow_blocks_reproduces_the_direct_gradients(shared, tmp_path):  # noqa: F811
    direct = trainer(shared, tmp_path / "direct", model=c5(shared))
    direct.run()
    replayed = trainer(shared, tmp_path / "replayed", model=c5(shared), accumulation_rows=8)
    replayed.run()
    left, right = records(direct), records(replayed)
    assert len(left) == len(right) > 2
    for first, second in zip(left, right, strict=True):
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)
    split = trainer(shared, tmp_path / "rows", model=c5(shared), accumulation_rows=2)
    with pytest.raises(ValueError, match="repetir todos los flujos"):
        split.run()
