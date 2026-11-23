"""Pruebas TDD de la regla de parada adaptativa en `selection.py`.

Cubre validación, identidad canónica (valores por defecto sin cambio), suavizado por media
móvil centrada, reducción de tasa de aprendizaje con reinicio de paciencia, selección de las
k mejores épocas para promedio de pesos, retención mínima de puntos de control e informe de
diagnóstico por ajuste.
"""

import copy
import math

import pytest

from mars_titan.training.selection import (
    CHRONOLOGICAL_MODES,
    COMMON_EPOCH,
    DECAY_LR,
    FINISH,
    FIXED_BUDGET,
    JOINT_PLATEAU,
    SESSION_MAE,
    SESSION_RANK_IC,
    VALIDATION_PLATEAU,
    advance_selection,
    apply_lr_decay,
    build_stopping_report,
    campaign_rule,
    canonical_rule,
    canonical_selection,
    epoch_decision,
    initial_selection,
    retained_epochs,
    selected_emit_epochs,
    smoothed_series,
    validate_selection,
)

MAE = dict(metric=SESSION_MAE, patience=3, min_delta=0.0)
RANKED = dict(metric=SESSION_RANK_IC, min_assets=30, patience=3, min_delta=0.0)


def loss(ic):
    return (1.0 - ic) / 2.0


def run_curve(scores, options, *, max_epochs, initial=None):
    """Recorrer una curva sintética de pérdidas de validación paso a paso."""
    state = None if initial is None else initial_selection(initial, options)
    decays = []
    history = []
    if initial is not None:
        history.append(
            dict(
                epoch=0,
                global_step=0,
                train=None,
                validation=dict(session_mae=initial, session_rank_ic=1.0 - 2.0 * initial),
            )
        )
    for epoch, score in enumerate(scores, 1):
        state = advance_selection(state, score, epoch, options)
        history.append(
            dict(
                epoch=epoch,
                global_step=epoch * 10,
                train=dict(
                    session_mae=max(0.0, score - 0.05),
                    session_mse=(max(0.0, score - 0.05)) ** 2,
                    mean_loss=max(0.0, score - 0.05),
                    session_rank_ic=min(1.0, 1.0 - 2.0 * score + 0.02),
                ),
                validation=dict(
                    session_mae=score,
                    session_mse=score**2,
                    session_rank_ic=1.0 - 2.0 * score,
                ),
            )
        )
        decision = epoch_decision(state, options, max_epochs)
        if decision == DECAY_LR:
            decays.append(dict(at_epoch=epoch, restored_epoch=state["best_epoch"]))
            state = apply_lr_decay(state, options)
            continue
        if decision == FINISH:
            return state, epoch, decays, history
    return state, len(scores), decays, history


def test_validation_accepts_valid_adaptive_options_and_rejects_invalid_ones():
    validate_selection(dict(RANKED, smoothing=1), adaptive=True)
    validate_selection(dict(RANKED, smoothing=3), adaptive=True)
    validate_selection(dict(RANKED, smoothing=5, emit="best"), adaptive=True)
    validate_selection(dict(RANKED, emit={"average": 1}), adaptive=True)
    validate_selection(dict(RANKED, emit={"average": 3}), adaptive=True)
    validate_selection(
        dict(
            RANKED,
            stopping=VALIDATION_PLATEAU,
            smoothing=3,
            learning_rate_decay={"factor": 0.5, "reductions": 2},
            emit={"average": 3},
            initial_state="reported",
        ),
        epochs=48,
        modes=CHRONOLOGICAL_MODES,
        initial_reported=True,
        adaptive=True,
    )
    # Suavizado y promedio se admiten también con fixed_budget, pero no con common_epoch.
    validate_selection(
        dict(MAE, stopping=FIXED_BUDGET, smoothing=3, emit={"average": 2}),
        epochs=6,
        adaptive=True,
    )

    invalid_smoothing = [0, -1, 2, 4, 3.0, True, "3", None, 1001]
    for value in invalid_smoothing:
        with pytest.raises(ValueError, match="smoothing|suavizado"):
            validate_selection(dict(RANKED, smoothing=value), adaptive=True)

    invalid_decay = [
        None,
        "half",
        {},
        {"factor": 0.5},
        {"reductions": 2},
        {"factor": 0.5, "reductions": 2, "extra": 1},
        {"factor": 0.0, "reductions": 2},
        {"factor": 1.0, "reductions": 2},
        {"factor": -0.5, "reductions": 2},
        {"factor": math.nan, "reductions": 2},
        {"factor": math.inf, "reductions": 2},
        {"factor": True, "reductions": 2},
        {"factor": 0.5, "reductions": 0},
        {"factor": 0.5, "reductions": -1},
        {"factor": 0.5, "reductions": True},
        {"factor": 0.5, "reductions": 2.0},
        {"factor": 0.5, "reductions": 101},
    ]
    for value in invalid_decay:
        with pytest.raises(ValueError, match="learning_rate_decay|reducción"):
            validate_selection(dict(RANKED, learning_rate_decay=value), adaptive=True)

    invalid_emit = [
        None,
        "average",
        3,
        {},
        {"average": 0},
        {"average": -2},
        {"average": True},
        {"average": 2.0},
        {"average": 65},
        {"average": 2, "extra": 1},
    ]
    for value in invalid_emit:
        with pytest.raises(ValueError, match="emit|emisión"):
            validate_selection(dict(RANKED, emit=value), adaptive=True)

    # learning_rate_decay solo aplica a validation_plateau.
    for mode in (FIXED_BUDGET, JOINT_PLATEAU, COMMON_EPOCH):
        base = MAE if mode == COMMON_EPOCH else RANKED
        with pytest.raises(ValueError, match="validation_plateau|meseta"):
            validate_selection(
                dict(base, stopping=mode, learning_rate_decay={"factor": 0.5, "reductions": 1}),
                modes=CHRONOLOGICAL_MODES,
                adaptive=True,
            )

    # common_epoch no se combina con smoothing > 1 ni emit promedio > 1.
    with pytest.raises(ValueError, match="época común"):
        validate_selection(
            dict(MAE, stopping=COMMON_EPOCH, smoothing=3),
            modes=CHRONOLOGICAL_MODES,
            adaptive=True,
        )
    with pytest.raises(ValueError, match="época común"):
        validate_selection(
            dict(MAE, stopping=COMMON_EPOCH, emit={"average": 2}),
            modes=CHRONOLOGICAL_MODES,
            adaptive=True,
        )

    # k + radio de suavizado no puede superar el presupuesto de puntos de control fijados (63).
    with pytest.raises(ValueError, match="fijados|límite"):
        validate_selection(dict(RANKED, smoothing=63, emit={"average": 33}), adaptive=True)

    # Por defecto (`adaptive=False`), entrenadores sin soporte adaptativo rechazan estas opciones.
    protocol = dict(MAE, stopping=FIXED_BUDGET, max_epochs=10)
    for option in (
        dict(smoothing=3),
        dict(learning_rate_decay={"factor": 0.5, "reductions": 1}),
        dict(emit={"average": 2}),
    ):
        with pytest.raises(ValueError):
            validate_selection(dict(RANKED, **option))
        with pytest.raises(ValueError):
            campaign_rule(protocol, dict(RANKED, max_epochs=10, **option))


def test_canonical_identity_ignores_default_values_and_changes_on_non_default():
    base_sel = dict(RANKED, stopping=VALIDATION_PLATEAU, initial_state="reported")
    base_rule = dict(base_sel, max_epochs=48)

    assert canonical_selection(base_sel) == base_sel
    assert canonical_selection(dict(base_sel, smoothing=1)) == base_sel
    assert canonical_selection(dict(base_sel, emit="best")) == base_sel
    assert canonical_selection(dict(base_sel, emit={"average": 1})) == base_sel
    assert canonical_selection(dict(base_sel, smoothing=1, emit="best")) == base_sel
    assert canonical_selection(dict(base_sel, smoothing=1, emit={"average": 1})) == base_sel

    assert canonical_rule(base_rule) == base_rule
    assert canonical_rule(dict(base_rule, smoothing=1, emit="best")) == base_rule
    assert canonical_rule(dict(base_rule, emit={"average": 1})) == base_rule

    protocol = dict(MAE, stopping=FIXED_BUDGET, max_epochs=30)
    assert (
        campaign_rule(
            protocol,
            dict(base_rule, smoothing=1, emit="best"),
            modes=CHRONOLOGICAL_MODES,
            initial_reported=True,
            adaptive=True,
        )
        == base_rule
    )
    assert (
        campaign_rule(
            protocol,
            dict(base_rule, emit={"average": 1}),
            modes=CHRONOLOGICAL_MODES,
            initial_reported=True,
            adaptive=True,
        )
        == base_rule
    )

    # Cualquier valor no por defecto cambia la identidad canónica.
    with_smooth = canonical_selection(dict(base_sel, smoothing=3))
    with_decay = canonical_selection(
        dict(base_sel, learning_rate_decay={"factor": 0.5, "reductions": 2})
    )
    with_emit = canonical_selection(dict(base_sel, emit={"average": 3}))
    assert with_smooth != base_sel and with_smooth["smoothing"] == 3
    assert with_decay != base_sel and with_decay["learning_rate_decay"] == {
        "factor": 0.5,
        "reductions": 2,
    }
    assert with_emit != base_sel and with_emit["emit"] == {"average": 3}
    assert len({str(x) for x in (base_sel, with_smooth, with_decay, with_emit)}) == 4


def test_absent_options_preserve_exact_legacy_selection_state():
    state = initial_selection(0.4, MAE)
    assert set(state) == {
        "last_epoch",
        "best_epoch",
        "best_score",
        "stale_epochs",
        "should_stop",
        "last_improved",
    }
    for epoch, value in enumerate((0.3, 0.35, 0.36, 0.37), 1):
        state = advance_selection(state, value, epoch, MAE)
        assert set(state) == {
            "last_epoch",
            "best_epoch",
            "best_score",
            "stale_epochs",
            "should_stop",
            "last_improved",
        }
    assert state["best_epoch"] == 1 and state["should_stop"] is True


def test_centered_moving_average_matches_manual_computation_including_boundaries():
    values = [0.50, 0.40, 0.48, 0.35, 0.30]
    assert smoothed_series(values, 1) == values
    assert smoothed_series(values, 3) == pytest.approx(
        [
            (0.50 + 0.40) / 2,
            (0.50 + 0.40 + 0.48) / 3,
            (0.40 + 0.48 + 0.35) / 3,
            (0.48 + 0.35 + 0.30) / 3,
            (0.35 + 0.30) / 2,
        ],
        rel=0,
        abs=1e-15,
    )
    assert smoothed_series(values, 5) == pytest.approx(
        [
            (0.50 + 0.40 + 0.48) / 3,
            (0.50 + 0.40 + 0.48 + 0.35) / 4,
            (0.50 + 0.40 + 0.48 + 0.35 + 0.30) / 5,
            (0.40 + 0.48 + 0.35 + 0.30) / 4,
            (0.48 + 0.35 + 0.30) / 3,
        ],
        rel=0,
        abs=1e-15,
    )


def test_noisy_late_peak_stops_early_without_smoothing_and_reaches_peak_with_smoothing():
    # Curva sintética de IC con ruido (pico espurio en época 2, máximo real en época 9).
    ic_curve = [0.010, 0.028, 0.015, 0.022, 0.026, 0.032, 0.036, 0.042, 0.048, 0.045, 0.040, 0.035]
    losses = [loss(ic) for ic in ic_curve]
    plain_opt = dict(RANKED, stopping=VALIDATION_PLATEAU, patience=3, initial_state="reported")
    smooth_opt = dict(plain_opt, smoothing=3)

    plain_state, plain_stop, _, _ = run_curve(losses, plain_opt, max_epochs=20, initial=0.5)
    smooth_state, smooth_stop, _, _ = run_curve(losses, smooth_opt, max_epochs=20, initial=0.5)

    # La regla actual se corta en la época 5 atrapada por el pico de ruido en la época 2.
    assert plain_stop == 5 and plain_state["best_epoch"] == 2
    # La regla suavizada atraviesa la racha ruidosa, alcanza el máximo real (época 9) y para en 12.
    assert smooth_stop == 12 and smooth_state["best_epoch"] == 9
    assert smooth_state["best_raw_score"] == pytest.approx(loss(0.048), abs=1e-15)
    assert smooth_state["best_score"] == pytest.approx(
        (loss(0.042) + loss(0.048) + loss(0.045)) / 3, abs=1e-15
    )


@pytest.mark.parametrize("window", [3, 5])
def test_strictly_worsening_curve_stops_equally_early_with_or_without_smoothing(window):
    worsening = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    plain_opt = dict(MAE, stopping=VALIDATION_PLATEAU, patience=3, initial_state="reported")
    smooth_opt = dict(plain_opt, smoothing=window)

    plain_state, plain_stop, _, _ = run_curve(worsening, plain_opt, max_epochs=10, initial=0.9)
    smooth_state, smooth_stop, _, _ = run_curve(worsening, smooth_opt, max_epochs=10, initial=0.9)

    assert plain_stop == smooth_stop == 4
    assert plain_state["best_epoch"] == smooth_state["best_epoch"] == 1


def test_strictly_improving_curve_does_not_stop_early_with_large_smoothing_and_small_patience():
    improving = [0.50, 0.40, 0.30, 0.20, 0.10]
    options = dict(
        MAE, stopping=VALIDATION_PLATEAU, patience=1, smoothing=5, initial_state="reported"
    )
    state, stop_epoch, _, _ = run_curve(improving, options, max_epochs=5, initial=0.9)
    assert stop_epoch == 5 and state["best_epoch"] == 5 and state["stale_epochs"] == 0


def test_smoothing_can_retroactively_promote_earlier_epoch_when_tail_degrades():
    # En época 3 la cola incompleta favorece ep 3; al llegar ep 4 mala, ep 2 vuelve a ganar.
    losses = [0.50, 0.20, 0.22, 0.60, 0.65, 0.70]
    options = dict(
        MAE, stopping=VALIDATION_PLATEAU, patience=3, smoothing=3, initial_state="reported"
    )
    state = initial_selection(0.9, options)
    bests = []
    for epoch, score in enumerate(losses, 1):
        state = advance_selection(state, score, epoch, options)
        bests.append(state["best_epoch"])
        if epoch_decision(state, options, 10) == FINISH:
            break
    # En ep 3: sm = [0.35, 0.3067, 0.21] -> mejor es ep 3.
    # En ep 4: sm = [0.35, 0.3067, 0.34, 0.41] -> mejor vuelve a ser ep 2!
    assert bests[2] == 3
    assert bests[3] == 2
    assert state["best_epoch"] == 2
    assert state["should_stop"] is True


def test_learning_rate_decay_requests_reductions_and_stops_only_after_n_reductions():
    # Paciencia 2, 2 reducciones:
    # Fase 0: ep1=0.5, ep2=0.3 (best=2), ep3=0.4 (s=1), ep4=0.45 (s=2 -> reducción 1, restaura 2)
    # Fase 1: ep5=0.25 (best=5), ep6=0.28 (s=1), ep7=0.29 (s=2 -> reducción 2, restaura 5)
    # Fase 2: ep8=0.30 (s=1), ep9=0.31 (s=2 -> parada definitiva, best=5)
    scores = [0.50, 0.30, 0.40, 0.45, 0.25, 0.28, 0.29, 0.30, 0.31, 0.10]
    options = dict(
        MAE,
        stopping=VALIDATION_PLATEAU,
        patience=2,
        learning_rate_decay={"factor": 0.5, "reductions": 2},
        initial_state="reported",
    )
    state, stop_epoch, decays, _ = run_curve(scores, options, max_epochs=20, initial=0.9)
    assert decays == [
        dict(at_epoch=4, restored_epoch=2),
        dict(at_epoch=7, restored_epoch=5),
    ]
    assert stop_epoch == 9
    assert state["best_epoch"] == 5
    assert state["best_score"] == 0.25
    assert state["reductions_done"] == 2
    assert state["should_stop"] is True


def test_learning_rate_decay_finishes_at_max_epochs_without_extra_reduction():
    scores = [0.50, 0.60, 0.70]
    options = dict(
        MAE,
        stopping=VALIDATION_PLATEAU,
        patience=2,
        learning_rate_decay={"factor": 0.5, "reductions": 2},
        initial_state="reported",
    )
    state, stop_epoch, decays, _ = run_curve(scores, options, max_epochs=3, initial=0.9)
    assert stop_epoch == 3
    assert decays == []
    assert epoch_decision(state, options, 3) == FINISH


def test_smoothing_and_learning_rate_decay_isolate_smoothing_across_rollbacks():
    # Fase 0: ep1=0.45, ep2=0.30, ep3=0.32, ep4=0.50, ep5=0.60 -> con smoothing=3:
    # sm = [0.375, 0.3567, 0.3733, 0.4733, 0.55] -> mejor es ep 2, meseta en ep 5.
    # Reducción 1 restaura ep 2; Fase 1 arranca sin contaminarse por ep5=0.60.
    scores = [0.45, 0.30, 0.32, 0.50, 0.60, 0.28, 0.26, 0.35, 0.40, 0.45]
    options = dict(
        MAE,
        stopping=VALIDATION_PLATEAU,
        patience=3,
        smoothing=3,
        learning_rate_decay={"factor": 0.5, "reductions": 1},
        initial_state="reported",
    )
    state, stop_epoch, decays, _ = run_curve(scores, options, max_epochs=20, initial=0.9)
    assert len(decays) == 1
    assert decays[0] == dict(at_epoch=5, restored_epoch=2)
    # En fase 1 (ep 6=0.28, ep 7=0.26, ep 8=0.35, ep 9=0.40):
    # sin contaminación de ep 5=0.60, sm[ep6] = (0.28+0.26)/2 = 0.27 < 0.3567 -> mejor es ep 6,
    # y con paciencia 3 para en ep 9 (6 + 3).
    assert state["best_epoch"] == 6
    assert state["best_score"] == pytest.approx(0.27, abs=1e-15)
    assert stop_epoch == 9
    assert state["should_stop"] is True


def test_emit_average_selects_top_k_epochs_and_k1_matches_best():
    scores = [0.40, 0.20, 0.35, 0.25, 0.30]
    for smoothing in (1, 3):
        opt_best = dict(MAE, stopping=FIXED_BUDGET, smoothing=smoothing, initial_state="reported")
        opt_k1 = dict(opt_best, emit={"average": 1})
        opt_k3 = dict(opt_best, emit={"average": 3})

        st_best, _, _, _ = run_curve(scores, opt_best, max_epochs=5, initial=0.9)
        st_k1, _, _, _ = run_curve(scores, opt_k1, max_epochs=5, initial=0.9)
        st_k3, _, _, _ = run_curve(scores, opt_k3, max_epochs=5, initial=0.9)

        assert selected_emit_epochs(st_best, opt_best) == [st_best["best_epoch"]]
        assert selected_emit_epochs(st_k1, opt_k1) == [st_best["best_epoch"]]
        top3 = selected_emit_epochs(st_k3, opt_k3)
        assert len(top3) == 3 and top3[0] == st_k3["best_epoch"]
        if smoothing == 1:
            assert top3 == [2, 4, 5]
        else:
            sm = smoothed_series(scores, 3)
            expected = sorted(range(1, 6), key=lambda e: (sm[e - 1], scores[e - 1], e))[:3]
            assert top3 == expected


def test_retained_epochs_stays_bounded_and_always_keeps_best_topk_and_tail():
    options = dict(
        MAE,
        stopping=FIXED_BUDGET,
        patience=5,
        smoothing=5,
        emit={"average": 3},
        initial_state="reported",
    )
    state = initial_selection(0.9, options)
    # 80 épocas (> límite de 64 fijados en checkpoints.py): retención siempre acotada.
    for epoch in range(1, 81):
        score = 0.5 - 0.002 * epoch + (0.03 if epoch % 2 else -0.02)
        state = advance_selection(state, score, epoch, options)
        keep = retained_epochs(state, options)
        # Máximo: k (3) + radio (2) + best (1) = 6, muy por debajo de 64.
        assert len(keep) <= 6
        assert state["best_epoch"] in keep
        # Todas las épocas no finalizadas de la cola [epoch - r + 1, epoch] se conservan.
        for tail in range(max(1, epoch - 1), epoch + 1):
            assert tail in keep
    # Al terminar, todas las épocas del promedio final estuvieron siempre retenidas.
    final_top3 = selected_emit_epochs(state, options)
    assert set(final_top3) <= set(retained_epochs(state, options))


def test_stopping_report_contains_full_curve_and_explicit_diagnosis():
    # Con smoothing=3, el máximo suavizado está en época 4 (mean(0.040, 0.038, 0.035)=0.03767),
    # así que con paciencia 3 la meseta se agota en la época 7 (3 épocas después del máximo).
    ic_curve = [0.010, 0.030, 0.040, 0.038, 0.035, 0.032, 0.028]
    losses = [loss(ic) for ic in ic_curve]
    options = dict(
        RANKED,
        stopping=VALIDATION_PLATEAU,
        patience=3,
        smoothing=3,
        emit={"average": 2},
        initial_state="reported",
    )
    state, stop_epoch, decays, history = run_curve(losses, options, max_epochs=12, initial=0.5)
    report = build_stopping_report(
        history,
        state,
        options,
        epochs=12,
        stopped_early=stop_epoch < 12,
        emitted_epochs=selected_emit_epochs(state, options),
        lr_decays=decays,
    )
    assert report["schema_version"] == 1
    assert report["kind"] == "fit_stopping_report_v1"
    assert report["metric"] == SESSION_RANK_IC
    assert report["higher_is_better"] is True
    assert len(report["curve"]) == len(history)
    assert [row["epoch"] for row in report["curve"]] == list(range(len(history)))

    diag = report["diagnosis"]
    assert diag["stopped_by_plateau"] is True
    assert diag["stopped_early"] is True
    assert diag["stop_reason"] == VALIDATION_PLATEAU
    assert diag["final_epoch"] == stop_epoch
    assert diag["best_epoch"] == state["best_epoch"]
    assert diag["epochs_after_best"] == stop_epoch - state["best_epoch"]
    best_entry = history[state["best_epoch"]]
    expected_train_ic = best_entry["train"]["session_rank_ic"]
    expected_val_ic = best_entry["validation"]["session_rank_ic"]
    assert diag["best_epoch_train_metric"] == pytest.approx(expected_train_ic)
    assert diag["best_epoch_validation_metric"] == pytest.approx(expected_val_ic)
    assert diag["best_epoch_train_validation_difference"] == pytest.approx(
        expected_train_ic - expected_val_ic
    )
    assert diag["best_epoch_generalization_gap"] == pytest.approx(
        expected_train_ic - expected_val_ic
    )
    assert report["emitted"]["mode"] == "average"
    assert report["emitted"]["epochs"] == selected_emit_epochs(state, options)
    assert report["emitted"]["memories_and_online_state_epoch"] == state["best_epoch"]


def test_stopping_report_for_mae_and_max_epochs_has_correct_signs_and_reason():
    # Para MAE (menor es mejor), si train_mae=0.10 y val_mae=0.15, la brecha de generalización
    # (sobreajuste) es val - train = +0.05 y la diferencia train - val es -0.05.
    scores = [0.20, 0.15]
    options = dict(MAE, stopping=VALIDATION_PLATEAU, patience=3, initial_state="reported")
    state, stop_epoch, decays, history = run_curve(scores, options, max_epochs=2, initial=0.30)
    report = build_stopping_report(
        copy.deepcopy(history),
        state,
        options,
        epochs=2,
        stopped_early=False,
        emitted_epochs=[state["best_epoch"]],
        lr_decays=decays,
    )
    diag = report["diagnosis"]
    assert diag["stopped_by_plateau"] is False
    assert diag["stopped_early"] is False
    assert diag["stop_reason"] == "max_epochs"
    assert diag["best_epoch"] == 2
    assert diag["epochs_after_best"] == 0
    assert diag["best_epoch_train_validation_difference"] == pytest.approx(-0.05)
    assert diag["best_epoch_generalization_gap"] == pytest.approx(+0.05)
    assert report["emitted"]["mode"] == "best"
    assert report["emitted"]["epochs"] == [2]
