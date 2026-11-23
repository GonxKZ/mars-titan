"""Lo que hace de verdad el entrenador al reducir la tasa y al emitir un promedio.

Las pruebas del autor comprueban el factor de la tasa, el registro de la época restaurada y la
reanudación exacta. Aquí se comprueba el estado: los pesos y el optimizador con los que arranca
la época siguiente a una reducción son los de la mejor época, la tasa anotada en cada época es
la que la entrenó y, con promedio y suavizado, la puntuación guardada es la cruda.
"""

import copy

import pytest
import torch

from mars_titan.training.learning_hold import HOLD_ENV
from tests.learning_hold_lowering import allowed_hold
from tests.training.test_financial_run import corpus
from tests.training.test_financial_run_adaptive_stopping import (
    PLATEAU_SELECTION,
    make_trainer,
    scripted_validation,
)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("adaptive-restore"))[1]


@pytest.fixture(autouse=True)
def allow_local_training(tmp_path, monkeypatch):
    hold = allowed_hold(tmp_path / "allow-training.json")
    monkeypatch.setenv(HOLD_ENV, str(hold))
    return hold


def _tensors(module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if torch.is_tensor(value)
    }


def _same(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_same(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_same(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _spied(engine, monkeypatch):
    """Guarda pesos y optimizador al empezar el ajuste de cada época."""
    seen = {}
    real = engine._train_pass

    def spy(run, cursor, stop, save):
        if cursor["stage"] == "start":
            seen[cursor["epoch"] + 1] = (
                _tensors(engine.predictor),
                copy.deepcopy(engine.optimizer.state_dict()),
            )
        return real(run, cursor, stop, save)

    monkeypatch.setattr(engine, "_train_pass", spy)
    return seen


def test_a_learning_rate_decay_restores_the_weights_and_the_optimizer_of_the_best_epoch(
    shared, tmp_path, monkeypatch
):
    # Época 1 la mejor; las 2 y 3 no mejoran y se reduce la tasa; las 4 y 5 tampoco: para.
    curve = [0.90, 0.30, 0.40, 0.45, 0.35, 0.38]
    selection = dict(
        PLATEAU_SELECTION, patience=2, learning_rate_decay={"factor": 0.25, "reductions": 1}
    )
    engine = make_trainer(
        shared, tmp_path / "decay", selection=selection, epochs=10, learning_rate=1e-2
    )
    seen = _spied(engine, monkeypatch)
    monkeypatch.setattr(engine, "evaluate", scripted_validation(curve))
    report = engine.run()

    assert report["status"] == "completed"
    assert report["best_epoch"] == 1
    assert [item["at_epoch"] for item in report["learning_rate_reductions"]] == [3]
    assert sorted(seen) == [1, 2, 3, 4, 5]

    best_weights, best_optimizer = seen[2]  # con lo que arrancó la época 2: el final de la 1
    drifted_weights, drifted_optimizer = seen[3]
    restored_weights, restored_optimizer = seen[4]
    # El ajuste mueve los pesos y el optimizador: sin eso la prueba no distinguiría nada.
    assert not _same(best_weights, drifted_weights)
    assert not _same(best_optimizer["state"], drifted_optimizer["state"])
    # La época 4 arranca con los pesos de la mejor época, bit a bit, y no con los de la 3.
    assert _same(restored_weights, best_weights)
    # Y con los momentos y el contador del optimizador de la mejor época.
    assert _same(restored_optimizer["state"], best_optimizer["state"])
    # Solo cambia la tasa, por el factor exacto, en todos los grupos.
    for before, after in zip(
        best_optimizer["param_groups"], restored_optimizer["param_groups"], strict=True
    ):
        assert after["lr"] == pytest.approx(0.25 * before["lr"], rel=0, abs=0)
        assert {k: v for k, v in after.items() if k != "lr"} == {
            k: v for k, v in before.items() if k != "lr"
        }

    # El informe anota en cada época la tasa con la que se ajustó esa época.
    rates = {row["epoch"]: row["learning_rate"] for row in report["stopping_report"]["curve"]}
    assert rates[0] is None
    assert [rates[epoch] for epoch in (1, 2, 3)] == [pytest.approx(1e-2)] * 3
    assert [rates[epoch] for epoch in (4, 5)] == [pytest.approx(2.5e-3)] * 2


def test_with_average_and_smoothing_the_saved_score_is_the_raw_one_of_the_best_epoch(
    shared, tmp_path, monkeypatch
):
    # Las medias centradas eligen la época 3 (0,30, 0,40 y 0,45); la segunda del promedio es la 2.
    curve = [0.90, 0.50, 0.30, 0.40, 0.45, 0.50]
    selection = dict(PLATEAU_SELECTION, patience=2, smoothing=3, emit={"average": 2})
    engine = make_trainer(
        shared, tmp_path / "average", selection=selection, epochs=10, learning_rate=1e-2
    )
    monkeypatch.setattr(engine, "evaluate", scripted_validation(curve))
    report = engine.run()

    assert report["status"] == "completed"
    assert report["best_epoch"] == 3
    assert report["averaged_epochs"] == [3, 2]
    assert report["best_score"] == pytest.approx(0.40)
    assert report["best_smoothed_score"] == pytest.approx((0.30 + 0.40 + 0.45) / 3)
