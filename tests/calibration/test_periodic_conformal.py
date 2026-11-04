"""Recalibración periódica de la CQR con filas maduras, la alternativa trivial de PT2.

Los paneles son sintéticos y no proceden de ningún modelo. Se comprueba que el primer bloque
es la CQR estática, que cada reajuste usa exactamente las filas maduras de su ventana, que
una etiqueta posterior al reajuste no cambia nada emitido y que los mercados no se mezclan.
"""

import numpy as np
import pytest

from mars_titan.calibration import conformal_quantiles as cqr
from mars_titan.calibration.periodic_conformal import DAY_MICROSECONDS, replay_periodic_refit
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.models.quantile_head import LEVELS

NOMINALS = (0.8, 0.95)
BASE = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
START = 1_600_000_000_000_000  # septiembre de 2020 en microsegundos UTC
# Siete activos por sesión: tres sesiones dan 21 filas, el mínimo para el orden del 95 %.
ASSETS = 7


def part(groups, days, *, delay=1, scale=1.0, seed=0, first=0):
    """Filas de varios mercados con `ASSETS` activos por sesión diaria y madurez fija.

    Cada mercado tiene su propio generador, así que sus filas no dependen de los demás.
    """
    names, moments, targets = [], [], []
    for name in groups:
        rng = np.random.default_rng([seed, *map(ord, name)])
        targets.append(np.round(rng.normal(0, scale, days * ASSETS), 6))
        for day in range(first, first + days):
            names += [name] * ASSETS
            moments += [START + day * DAY_MICROSECONDS] * ASSETS
    rows = len(names)
    target = np.concatenate(targets)
    prediction_at = np.array(moments, dtype=np.int64)
    return dict(
        target=target,
        quantiles=np.tile(BASE, (rows, 1)),
        groups=np.array(names),
        prediction_at=prediction_at,
        matures_at=prediction_at + delay * DAY_MICROSECONDS,
    )


def fitted(calibration, min_rows=12):
    return cqr.fit_conformal_quantiles(
        calibration["target"],
        calibration["quantiles"],
        calibration["groups"],
        levels=LEVELS,
        nominals=NOMINALS,
        min_rows=min_rows,
    )


def window(groups=("US",), *, every=3, lookback=8, scale=2.0, delay=1, min_rows=12):
    calibration = part(groups, 4, seed=1)
    evaluation = part(groups, 9, scale=scale, seed=2, first=6, delay=delay)
    record = fitted(calibration, min_rows)
    emitted, refits = replay_periodic_refit(
        record, calibration, evaluation, every_sessions=every, lookback_days=lookback
    )
    return calibration, evaluation, record, emitted, refits


def pooled(calibration, evaluation, name, at, lookback):
    """Filas maduras de un mercado en la ventana del reajuste, elegidas a mano."""
    keys = ("target", "quantiles", "groups", "prediction_at", "matures_at")
    joined = {key: np.concatenate([calibration[key], evaluation[key]]) for key in keys}
    selected = (
        (joined["groups"] == name)
        & (joined["matures_at"] <= at)
        & (joined["prediction_at"] >= at - lookback * DAY_MICROSECONDS)
    )
    return {key: value[selected] for key, value in joined.items()}


def test_first_block_is_the_static_cqr_and_later_blocks_refit_with_mature_rows():
    calibration, evaluation, record, emitted, refits = window()
    sessions = np.unique(evaluation["prediction_at"])
    first = np.isin(evaluation["prediction_at"], sessions[:3])
    static, _ = cqr.apply_conformal_quantiles(
        record, evaluation["quantiles"][first], evaluation["groups"][first]
    )
    assert np.array_equal(emitted[first], static)
    assert [item["at"] for item in refits] == [int(sessions[3]), int(sessions[6])]
    for block, item in zip((sessions[3:6], sessions[6:9]), refits, strict=True):
        pool = pooled(calibration, evaluation, "US", item["at"], 8)
        assert item["rows"] == len(pool["target"]) and item["used"] is True
        expected_record = fitted(pool)
        assert item["calibrator_sha256"] == cqr.calibrator_sha256(expected_record)
        members = np.isin(evaluation["prediction_at"], block)
        expected, _ = cqr.apply_conformal_quantiles(
            expected_record, evaluation["quantiles"][members], evaluation["groups"][members]
        )
        assert np.array_equal(emitted[members], expected)
    # La primera ventana de reajuste incluye filas de calibración.
    assert refits[0]["rows"] == 6 * ASSETS
    # Con objetivos más dispersos en la evaluación, el reajuste ensancha el intervalo del 80 %.
    width = emitted[:, 3] - emitted[:, 1]
    assert width[~first].min() > width[first].max()


def test_a_label_that_matures_after_a_refit_never_changes_what_was_emitted():
    calibration, evaluation, record, emitted, refits = window()
    last = refits[-1]["at"]
    later = evaluation["matures_at"] > last
    changed = dict(evaluation, target=np.where(later, 1e6, evaluation["target"]))
    again, again_refits = replay_periodic_refit(
        record, calibration, changed, every_sessions=3, lookback_days=8
    )
    assert np.array_equal(again, emitted) and again_refits == refits
    # Una etiqueta que madura justo en el instante del reajuste sí entra en él.
    exact = evaluation["matures_at"] == last
    assert exact.any()
    moved = dict(evaluation, target=np.where(exact, 1e6, evaluation["target"]))
    _, moved_refits = replay_periodic_refit(
        record, calibration, moved, every_sessions=3, lookback_days=8
    )
    assert moved_refits[-1]["calibrator_sha256"] != refits[-1]["calibrator_sha256"]


def test_rows_outside_the_lookback_do_not_enter_the_refit():
    calibration, evaluation, record, emitted, refits = window(lookback=3)
    assert [item["rows"] for item in refits] == [3 * ASSETS, 3 * ASSETS]
    assert all(item["used"] for item in refits)
    # Con tres días de ventana, la calibración ya no entra en ningún reajuste. El registro
    # estático es el mismo, así que cambiar sus objetivos no cambia nada emitido.
    changed = dict(calibration, target=calibration["target"] * 100)
    again, again_refits = replay_periodic_refit(
        record, changed, evaluation, every_sessions=3, lookback_days=3
    )
    assert again_refits == refits and np.array_equal(again, emitted)


def test_a_pool_below_the_minimum_keeps_the_previous_correction():
    calibration, evaluation, record, emitted, refits = window(min_rows=12, lookback=1)
    # Con un día de ventana y madurez de un día solo entra una sesión de siete filas.
    assert all(item["rows"] == ASSETS and item["used"] is False for item in refits)
    static, _ = cqr.apply_conformal_quantiles(record, evaluation["quantiles"], evaluation["groups"])
    assert np.array_equal(emitted, static)
    assert {item["calibrator_sha256"] for item in refits} == {cqr.calibrator_sha256(record)}


def test_markets_follow_their_own_sessions_and_never_share_rows():
    calibration, evaluation, record, emitted, refits = window(("CN", "US"))
    alone = window(("US",))
    us = evaluation["groups"] == "US"
    assert np.array_equal(emitted[us], alone[3])
    assert [item for item in refits if item["group"] == "US"] == alone[4]
    changed = dict(evaluation, target=np.where(~us, -evaluation["target"], evaluation["target"]))
    again, _ = replay_periodic_refit(
        record, calibration, changed, every_sessions=3, lookback_days=8
    )
    assert np.array_equal(again[us], emitted[us])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(every_sessions=0), "número entero positivo de sesiones"),
        (dict(lookback_days=0), "número entero positivo de días"),
        (dict(overlap=True), "debe preceder"),
        (dict(early=True), "madurez posterior"),
        (dict(sealed=True), "2024"),
        (dict(market="JP"), "no define una corrección inicial"),
    ],
)
def test_declarations_and_rows_that_break_the_contract_are_rejected(change, message):
    calibration, evaluation = part(("US",), 4, seed=1), part(("US",), 9, seed=2, first=6)
    record = fitted(calibration)
    options = dict(every_sessions=3, lookback_days=8)
    if "overlap" in change:
        calibration = part(("US",), 7, seed=1, first=3)
    elif "early" in change:
        evaluation = dict(evaluation, matures_at=evaluation["prediction_at"])
    elif "sealed" in change:
        # Una sola etiqueta en el test reservado basta para rechazar el tramo (#748).
        matures = evaluation["matures_at"].copy()
        matures[-1] = FINAL_TEST_START_US
        evaluation = dict(evaluation, matures_at=matures)
    elif "market" in change:
        evaluation = dict(evaluation, groups=np.full(len(evaluation["groups"]), "JP"))
    else:
        options.update(change)
    with pytest.raises(ValueError, match=message):
        replay_periodic_refit(record, calibration, evaluation, **options)
