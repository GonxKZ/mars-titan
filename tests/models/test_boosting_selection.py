"""Parada por sesiones y recuperación con referencias pequeñas de XGBoost CPU."""

import importlib

import numpy as np
import pytest


def module():
    return importlib.import_module("mars_titan.models.baselines.boosting_selection")


def policy(**changes):
    return dict(schema_version=1, minimum_rounds=3, patience_rounds=2, min_delta=0.01) | changes


def test_minimum_delays_stopping_but_keeps_earlier_best_and_late_improvement():
    selection = module().BoostingSelection(policy(), 8)
    for count, score in enumerate([1.0, 1.1, 0.8, 0.81, 0.82], 1):
        selection.observe(count, score)
        assert selection.state["stop_reason"] == ("validation_plateau" if count == 5 else None)
    assert selection.state["selected_round"] == 3
    assert selection.state["rounds_without_improvement"] == 2
    early = module().BoostingSelection(policy(), 8)
    for count, score in enumerate([0.5, 0.8, 0.9, 1.0, 1.1], 1):
        early.observe(count, score)
    assert early.state["selected_round"] == 1


def test_budget_with_improvement_is_not_convergence_and_small_best_is_selected():
    selection = module().BoostingSelection(policy(), 4)
    for count, score in enumerate([1.0, 0.8, 0.7, 0.695], 1):
        selection.observe(count, score)
    assert selection.state["stop_reason"] == "budget_exhausted"
    assert selection.state["selected_round"] == 4
    assert selection.state["best_session_mae"] == 0.695
    assert selection.state["rounds_without_improvement"] == 1


def test_resume_restores_patience_and_rejects_incomplete_or_nonfinite_evaluation():
    selection = module().BoostingSelection(policy(), 8)
    for count, score in enumerate([1.0, 0.8, 0.7, 0.9], 1):
        selection.observe(count, score)
    restored = module().BoostingSelection(policy(), 8, state=selection.state)
    before = dict(restored.state)
    for count, score in [(6, 0.6), (5, float("nan")), (5, -0.1)]:
        with pytest.raises(ValueError):
            restored.observe(count, score)
        assert restored.state == before
    restored.observe(5, 0.9)
    assert restored.state["stop_reason"] == "validation_plateau"
    assert restored.state["selected_round"] == 3


@pytest.mark.parametrize(
    "change",
    [
        {"minimum_rounds": 0},
        {"minimum_rounds": True},
        {"patience_rounds": 0},
        {"min_delta": float("inf")},
        {"schema_version": 2},
    ],
)
def test_invalid_policy_is_rejected_before_any_round(change):
    with pytest.raises(ValueError):
        module().BoostingSelection(policy(**change), 8)


@pytest.mark.parametrize(
    "change",
    [
        {"completed_rounds": 20},
        {"selected_round": 0},
        {"rounds_without_improvement": 3},
        {"stop_reason": "validation_plateau"},
        {"best_session_mae": 2.0},
    ],
)
def test_corrupt_recovery_cannot_change_rounds_or_stopping(change):
    selection = module().BoostingSelection(policy(), 8)
    selection.observe(1, 1.0)
    with pytest.raises(ValueError):
        module().BoostingSelection(policy(), 8, state=selection.state | change)


def test_session_metric_keeps_markets_separate_and_requires_complete_population():
    class Predictor:
        def predict(self, x):
            return x[:, 0]

    def blocks():
        yield np.array([[1.0], [3.0]]), np.zeros(2), ["US", "US"], np.array([1, 1])
        yield np.array([[9.0], [5.0]]), np.zeros(2), ["CN", "US"], np.array([1, 2])

    score = module().session_validation(Predictor(), blocks, expected_rows=4)
    assert score == pytest.approx(16 / 3)
    assert score != pytest.approx(4.5)
    with pytest.raises(ValueError, match="población"):
        module().session_validation(Predictor(), blocks, expected_rows=5)


def test_cpu_callback_stops_and_continues_from_confirmed_booster():
    xgb = pytest.importorskip("xgboost")
    x = np.arange(16, dtype=np.float32).reshape(8, 2)
    data = xgb.DMatrix(x, label=x[:, 0], nthread=1)
    parameters = dict(device="cpu", nthread=1, tree_method="hist", max_depth=1)
    scores = [1.0, 0.8, 0.7, 0.9, 1.0]
    snapshots = []

    def callback(selection, interrupt=False):
        def confirm(model, state):
            snapshots.append((model.copy(), dict(state)))
            if interrupt and model.num_boosted_rounds() == 4:
                raise InterruptedError("Corte tras una evaluación completa")

        return module().selection_callback(
            xgb, selection, lambda model: scores[model.num_boosted_rounds() - 1], confirm
        )

    selection = module().BoostingSelection(policy(), 8)
    with pytest.raises(InterruptedError):
        xgb.train(parameters, data, 8, callbacks=[callback(selection, True)])
    parent, state = snapshots[-1]
    recovered = module().BoostingSelection(policy(), 8, state=state)
    resumed = xgb.train(parameters, data, 4, xgb_model=parent, callbacks=[callback(recovered)])
    uninterrupted = module().BoostingSelection(policy(), 8)
    complete = xgb.train(parameters, data, 8, callbacks=[callback(uninterrupted)])
    assert resumed.num_boosted_rounds() == complete.num_boosted_rounds() == 5
    assert recovered.state == uninterrupted.state
    np.testing.assert_array_equal(resumed[:3].predict(data), complete[:3].predict(data))
    assert not np.array_equal(resumed[:3].predict(data), resumed.predict(data))


def test_partial_validation_cannot_advance_a_round_or_publish_a_model():
    xgb = pytest.importorskip("xgboost")
    selection = module().BoostingSelection(policy(), 8)
    before = dict(selection.state)

    def interrupted(model):
        raise InterruptedError("Validación incompleta")

    def unexpected(*args):
        pytest.fail("No se puede confirmar una evaluación incompleta")

    callback = module().selection_callback(xgb, selection, interrupted, unexpected)
    model = xgb.train(
        dict(device="cpu", nthread=1),
        xgb.DMatrix(np.array([[0.0], [1.0]]), label=[0.0, 1.0], nthread=1),
        1,
    )
    with pytest.raises(InterruptedError):
        callback.after_iteration(model, 0, {})
    assert selection.state == before


def test_callback_can_pause_after_confirmation_without_raising_inside_xgboost():
    xgb = pytest.importorskip("xgboost")
    control = module().BoostingSelection(policy(), 8)
    data = xgb.DMatrix(np.arange(8, dtype=np.float32).reshape(8, 1), label=np.arange(8), nthread=1)
    callback = module().selection_callback(
        xgb,
        control,
        lambda model: 1.0,
        lambda model, state: state["completed_rounds"] == 3,
    )
    model = xgb.train(dict(device="cpu", nthread=1), data, 8, callbacks=[callback])
    assert model.num_boosted_rounds() == control.state["completed_rounds"] == 3
    assert control.state["stop_reason"] is None


@pytest.mark.parametrize("mode", ["continue", "pause", "mismatch"])
def test_replay_verifies_prefix_without_repeating_selection_or_confirmation(mode):
    xgb = pytest.importorskip("xgboost")
    data = xgb.DMatrix(np.arange(8, dtype=np.float32).reshape(8, 1), label=np.arange(8), nthread=1)
    params = dict(device="cpu", nthread=1, max_depth=1)
    parent = xgb.train(params, data, 3)
    control = module().BoostingSelection(policy(), 8)
    for count in range(1, 4):
        control.observe(count, 1.0)
    confirmed = []
    callback = module().selection_callback(
        xgb,
        control,
        lambda model: 1.0,
        lambda model, state: confirmed.append(state["completed_rounds"]),
        replay_model=parent,
        stop_requested=lambda: mode == "pause" and callback.replayed_rounds == 2,
    )
    if mode == "mismatch":
        with pytest.raises(ValueError, match="prefijo"):
            xgb.train(dict(params, learning_rate=0.1), data, 8, callbacks=[callback])
        assert confirmed == []
    else:
        model = xgb.train(params, data, 8, callbacks=[callback])
        assert model.num_boosted_rounds() == (2 if mode == "pause" else 5)
        assert confirmed == ([] if mode == "pause" else [4, 5])
        assert control.state["completed_rounds"] == (3 if mode == "pause" else 5)


def test_external_resume_rejects_changed_selection_before_loading_cuda(tmp_path, monkeypatch):
    from mars_titan.models.baselines import external_boosting as external

    selector = module().BoostingSelection(policy(), 8)
    selector.observe(1, 1.0)
    parent = external.ExternalBoostingModel(
        object(), 1, 1, dict(selection=selector.state, selection_policy=policy())
    )

    def unexpected():
        pytest.fail("Una recuperación inválida no debe inicializar CUDA")

    monkeypatch.setattr(external, "_libraries", unexpected)
    with pytest.raises(ValueError, match="selección"):
        external.fit_external_boosting(
            lambda: iter(()),
            tmp_path / "pages",
            expected_rows=1,
            rounds=8,
            resume=parent,
            selection=policy(min_delta=0.02),
            validation_factory=lambda: iter(()),
            validation_rows=1,
        )
