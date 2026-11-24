"""Huecos de las pruebas de la parada adaptativa que encontraron las mutaciones del coordinador.

Cada prueba mata un mutante que sobrevivía a las pruebas del autor:
- al abrir una fase tras reducir la tasa, la mejor puntuación anterior sigue siendo la vara;
- en las dos primeras épocas de una fase, la referencia sube cuando la media mejora;
- la mejor época sigue retenida aunque otra posterior tenga mejor media sin superar el margen;
- el informe convierte la puntuación suavizada al IC con la misma regla que la cruda.
"""

import pytest

from mars_titan.training.selection import (
    DECAY_LR,
    FINISH,
    SESSION_MAE,
    SESSION_RANK_IC,
    advance_selection,
    apply_lr_decay,
    build_stopping_report,
    epoch_decision,
    retained_epochs,
)

MAE = dict(metric=SESSION_MAE, min_delta=0.0)


def walk(scores, options, *, max_epochs=64):
    """Recorre la curva y devuelve el estado tras cada época y las reducciones pedidas."""
    state, states, decays = None, [], []
    for epoch, score in enumerate(scores, 1):
        state = advance_selection(state, score, epoch, options)
        states.append(state)
        decision = epoch_decision(state, options, max_epochs)
        if decision == DECAY_LR:
            decays.append(epoch)
            state = apply_lr_decay(state, options)
        elif decision == FINISH:
            break
    return state, states, decays


def test_a_worse_phase_after_a_decay_does_not_take_the_best_epoch_with_smoothing():
    options = dict(
        MAE, patience=2, smoothing=3, learning_rate_decay={"factor": 0.5, "reductions": 1}
    )
    # Fase 1: la media centrada más baja es la de la época 3 (0,30, 0,40 y 0,45). Tras dos
    # épocas sin mejora se reduce la tasa y la fase nueva es peor en todas sus épocas.
    state, states, decays = walk([0.50, 0.30, 0.40, 0.45, 0.50, 0.60, 0.62], options)
    assert decays == [5]
    before = states[4]
    assert before["best_epoch"] == 3
    assert before["best_score"] == pytest.approx((0.30 + 0.40 + 0.45) / 3)
    # Ninguna época de la fase nueva supera a la mejor anterior: la elegida no cambia.
    assert states[5]["best_epoch"] == 3
    assert states[5]["best_score"] == pytest.approx(before["best_score"])
    assert states[5]["best_raw_score"] == pytest.approx(0.40)
    assert state["best_epoch"] == 3
    assert state["best_score"] == pytest.approx(before["best_score"])
    assert state["should_stop"] is True


def test_the_early_reference_of_a_phase_rises_when_its_mean_improves():
    options = dict(
        MAE, patience=3, smoothing=3, learning_rate_decay={"factor": 0.5, "reductions": 1}
    )
    # Tras la reducción (época 6) la primera época de la fase mejora a la anterior (0,20) y la
    # segunda deja la media de la fase en 0,25: peor que la referencia recién subida, aunque
    # mejor que la mejor de la fase anterior. Cuenta como época sin mejora.
    state, states, decays = walk([0.50, 0.30, 0.40, 0.45, 0.50, 0.55, 0.20, 0.30], options)
    assert decays == [6]
    first, second = states[6], states[7]
    assert first["last_improved"] is True
    assert first["stale_epochs"] == 0
    assert first["early_ref_score"] == pytest.approx(0.20)
    assert second["last_improved"] is False
    assert second["stale_epochs"] == 1
    assert second["early_ref_score"] == pytest.approx(0.20)
    assert second["best_epoch"] == 7


def test_the_best_epoch_stays_retained_when_a_later_mean_is_lower_within_the_margin():
    options = dict(metric=SESSION_MAE, patience=5, min_delta=0.05, smoothing=3)
    # La época 4 es la elegida (su media mejora a la primera en más del margen). Las medias de
    # las épocas 6 y 7 son más bajas, pero no la superan en el margen: la elegida sigue siendo
    # la 4 y su estado no se puede soltar.
    state, _, decays = walk([0.50, 0.40, 0.40, 0.40, 0.38, 0.38, 0.38, 0.38], options)
    assert decays == []
    assert state["best_epoch"] == 4
    kept = retained_epochs(state, options)
    assert 4 in kept
    # También siguen la de mejor media ya cerrada y la última, que aún puede cambiar.
    assert 6 in kept and 8 in kept


def test_the_report_converts_the_smoothed_score_to_the_ic_like_the_raw_one():
    options = dict(metric=SESSION_RANK_IC, min_assets=30, patience=3, min_delta=0.0, smoothing=3)
    ics = [0.02, 0.05, 0.08, 0.06, 0.04, 0.03, 0.02]
    history = [
        dict(
            epoch=0,
            global_step=0,
            train=None,
            validation=dict(session_mae=0.5, session_rank_ic=0.0),
        )
    ]
    state = None
    for epoch, ic in enumerate(ics, 1):
        score = (1.0 - ic) / 2.0
        state = advance_selection(state, score, epoch, options)
        history.append(
            dict(
                epoch=epoch,
                global_step=10 * epoch,
                train=dict(
                    session_mae=0.4, session_mse=0.16, mean_loss=0.4, session_rank_ic=ic + 0.01
                ),
                validation=dict(session_mae=0.5, session_mse=0.25, session_rank_ic=ic),
            )
        )
        if epoch_decision(state, options, 64) == FINISH:
            break
    report = build_stopping_report(history, state, options, epochs=64, stopped_early=True)
    rows = {row["epoch"]: row for row in report["curve"]}
    # Una época interior: la media centrada de los tres IC de validación.
    for epoch in (2, 3, 4):
        expected = sum(ics[epoch - 2 : epoch + 1]) / 3
        assert rows[epoch]["validation_smoothed_metric"] == pytest.approx(expected, abs=1e-12)
        assert rows[epoch]["validation_smoothed_metric"] == pytest.approx(
            1.0 - 2.0 * rows[epoch]["validation_smoothed_score"], abs=1e-12
        )
    best = report["diagnosis"]["best_epoch"]
    assert report["diagnosis"]["best_epoch_validation_smoothed_metric"] == pytest.approx(
        rows[best]["validation_smoothed_metric"], abs=1e-12
    )
