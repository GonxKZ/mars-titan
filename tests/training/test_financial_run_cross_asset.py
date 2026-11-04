"""Recorrido cronológico con atención entre activos de la misma sesión (C5 de #751), sin pasos.

Reutiliza el corpus y el entrenador de las pruebas por sesión de #749. El optimizador
registrador guarda los gradientes y no cambia pesos.
"""

import hashlib
import json
from collections import Counter
from dataclasses import asdict

import pytest
import torch

from mars_titan.models.titans.cross_asset_attention import CrossAssetAttentionConfig
from mars_titan.training import mars_titan_walk_forward
from tests.models.titans.test_financial_adapter import specification
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


def test_the_frozen_loader_refuses_c5_before_build_or_read(tmp_path, monkeypatch):
    document = dict(predictor=dict(hidden_size=32, dtype="float64"))
    changes = dict(cross_asset_attention=asdict(CrossAssetAttentionConfig()))
    report = dict(request=dict(predictor_changes=changes))

    def forbidden(*_args, **_options):
        pytest.fail("Se construyó o leyó un padre con C5 sin registro congelado")

    monkeypatch.setattr(mars_titan_walk_forward, "FinancialPredictor", forbidden)
    monkeypatch.setattr(mars_titan_walk_forward, "read_manifest", forbidden)
    with pytest.raises(ValueError, match="atención entre activos"):
        mars_titan_walk_forward._frozen_parent(
            document, specification(), 42, "cpu", tmp_path, report
        )


@pytest.mark.parametrize("in_recipe", (False, True))
def test_a_c5_parent_is_refused_before_checkpoint_verification(tmp_path, monkeypatch, in_recipe):
    view = tmp_path / "view.json"
    view.write_text("{}\n")
    declared = asdict(CrossAssetAttentionConfig())
    predictor = dict(hidden_size=32, dtype="float64")
    request = dict(
        variant="mac_online", seed=42, view_sha256=hashlib.sha256(view.read_bytes()).hexdigest()
    )
    if in_recipe:
        predictor["cross_asset_attention"] = declared
    else:
        request["predictor_changes"] = dict(cross_asset_attention=declared)
    report = dict(
        kind=mars_titan_walk_forward.TITANS_KIND,
        status="completed",
        final_test_opened=False,
        checkpoint={},
        request=request,
        identity=dict(recipe=dict(predictor=predictor)),
    )
    (tmp_path / "run.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        mars_titan_walk_forward,
        "_verify",
        lambda *_a: pytest.fail("Se verificó el checkpoint de un padre con C5 sin registrar"),
    )
    with pytest.raises(ValueError, match="atención entre activos"):
        mars_titan_walk_forward._parent(tmp_path, view, 42)
