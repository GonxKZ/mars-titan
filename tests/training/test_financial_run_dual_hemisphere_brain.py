"""Recorrido cronológico con cerebro de doble hemisferio ortogonal (`dual_hemisphere_brain`)."""

from collections import Counter

import pytest
import torch

from mars_titan.models.titans.dual_hemisphere_brain import (
    ORTHOGONAL,
    SELF_ONLY,
    SHUFFLED_PEERS,
    UNORTHOGONALIZED,
    DualHemisphereBrainConfig,
)
from tests.training.chronological_fixture import decision
from tests.training.test_financial_run import StopAtStep, entries
from tests.training.test_financial_run_session_objective import (
    ASSETS,
    explicit_fastpath,  # noqa: F401
    predictor,
    records,
    shared,  # noqa: F401
    streams,
    trainer,
)

PLAIN = dict(session_rank_inputs=None, session_ic_objective=None)


def codhb(sources, mode=ORTHOGONAL):
    return predictor(
        sources,
        **PLAIN,
        dual_hemisphere_brain=DualHemisphereBrainConfig(dim=16, ridge=1e-4, mode=mode),
    )


@pytest.mark.parametrize("mode", [ORTHOGONAL, UNORTHOGONALIZED, SELF_ONLY, SHUFFLED_PEERS])
def test_chronological_run_replays_whole_sessions_and_reaches_codhb_gate(shared, tmp_path, mode):  # noqa: F811
    engine = trainer(shared, tmp_path / mode, model=codhb(shared, mode))
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
    # La puerta arranca en cero sobre W_out no nula: recibe gradiente desde el primer paso.
    assert first["dual_hemisphere_brain.gate"].abs().sum() > 0
    assert torch.isfinite(first["dual_hemisphere_brain.gate"]).all()
    assert all(record["head.weight"].abs().sum() > 0 for record in gradients)
    assert engine.predictor._parameter_id == before


def test_split_session_is_refused_and_accumulation_matches_direct_gradients(shared, tmp_path):  # noqa: F811
    split = trainer(shared, tmp_path / "split", model=codhb(shared), block_rows=2)
    with pytest.raises(ValueError, match="sesión completa"):
        split.run()

    direct = trainer(shared, tmp_path / "direct", model=codhb(shared))
    direct.run()
    replayed = trainer(shared, tmp_path / "replayed", model=codhb(shared), accumulation_rows=8)
    replayed.run()
    left, right = records(direct), records(replayed)
    assert len(left) == len(right) > 2
    for first, second in zip(left, right, strict=True):
        for name, value in first.items():
            torch.testing.assert_close(second[name], value, rtol=1e-9, atol=1e-12)


def test_future_suffix_does_not_change_past_predictions_or_gradients(tmp_path):
    """Prueba de no fuga temporal: perturbar cualquier dato posterior al corte no cambia ni un
    bit las predicciones ni los gradientes anteriores al corte."""
    cut = decision("2022-12-09")
    engines = []
    for name, perturb in (("base", None), ("future", cut)):
        sources = streams(tmp_path / name, perturb_after=perturb)
        engine = trainer(sources, tmp_path / name / "run", model=codhb(sources))
        engine.run()
        engines.append(engine)
    base, future = engines

    def past(engine):
        return [e for e in entries(engine.audit, "prediction", "train") if e[3] <= cut]

    assert past(base) and past(base) == past(future)
    assert entries(base.audit, "prediction", "train") != entries(
        future.audit, "prediction", "train"
    )
    steps = [u[2] for u in entries(base.audit, "update") if u[1] <= cut]
    assert steps
    for left, right in zip(records(base)[: len(steps)], records(future), strict=False):
        for key, value in left.items():
            torch.testing.assert_close(right[key], value, rtol=0, atol=0)


def test_checkpoint_resume_matches_uninterrupted_run_and_validation_is_repeatable(shared, tmp_path):  # noqa: F811
    """Interrumpir y reanudar desde checkpoint reproduce el recorrido continuo bit a bit, y
    la evaluación tras recargar parámetros coincide exactamente entre fases."""
    options = dict(truncation=2, epochs=2, checkpoint_updates=2, block_rows=8)

    def make(name):
        model = codhb(shared, ORTHOGONAL)
        with torch.no_grad():
            model.dual_hemisphere_brain.gate.fill_(0.08)
        model._seal_parameters()
        return trainer(shared, tmp_path / name, model=model, **options)

    continuous = make("continuous")
    expected = continuous.run()
    assert expected["status"] == "completed"

    first = make("interrupted")
    stop = StopAtStep(3)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"

    second = make("interrupted")
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = records(first) + records(second)
    expected_grads = records(continuous)
    assert len(joined) == len(expected_grads)
    for left, right in zip(joined, expected_grads, strict=True):
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    assert resumed["history"] == expected["history"]

    restored_model = codhb(shared)
    restored_model.load_state_dict(continuous.predictor.state_dict())
    eval_engine = trainer(shared, tmp_path / "eval", model=restored_model)
    rows_a, rows_b = [], []
    metrics_a = continuous.evaluate(shared["validation"], rows=rows_a)
    metrics_b = eval_engine.evaluate(shared["validation"], rows=rows_b)
    assert metrics_a == metrics_b
    assert rows_a == rows_b and len(rows_a) > 0
