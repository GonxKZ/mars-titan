"""Calibración conformal en línea (PT2) con ejemplos a mano y secuencias explícitas.

Las secuencias no ajustan ningún modelo. Comprueban la ecuación de actualización, la cota
de cobertura a largo plazo con retraso, la causalidad, el aislamiento entre mercados y la
recuperación del estado. El componente no tiene parámetros entrenables ni gradientes.
"""

import hashlib
import json

import numpy as np
import pytest

from mars_titan.calibration import online_conformal
from mars_titan.calibration.conformal_quantiles import (
    apply_conformal_quantiles,
    calibrator_sha256,
    fit_conformal_quantiles,
)
from mars_titan.calibration.online_conformal import OnlineConformal, replay_online_conformal
from mars_titan.models.quantile_head import LEVELS

NOMINALS = (0.8, 0.95)
BASE = [-2.0, -1.0, 0.0, 1.0, 2.0]
# Huella de la CQR estática sobre el ejemplo de calibración, capturada en 42e7dbca antes de
# añadir PT2. Si cambia, la ruta sin el brazo ya no es la misma.
STATIC_FIXTURE_SHA256 = "913d30aca54efeea9af1f6640707f2474a2b6b0fefe6360d2b708f94e424d4aa"


def calibration_rows(groups=("US",)):
    # Diecinueve filas por mercado con y = −9, …, 9 y cuantiles fijos. Con la puntuación
    # CQR, Q80 = 8 − 1 = 7 y Q95 = 9 − 2 = 7, y la mayor |E| es 8 y 7 respectivamente.
    target = np.tile(np.arange(-9.0, 10.0), len(groups))
    quantiles = np.tile(BASE, (len(target), 1))
    names = np.repeat(np.array(groups), 19)
    return target, quantiles, names


def start(rate_fraction=0.25, groups=("US",), **options):
    target, quantiles, names = calibration_rows(groups)
    return OnlineConformal.start(
        target,
        quantiles,
        names,
        levels=LEVELS,
        nominals=NOMINALS,
        min_rows=1,
        rate_fraction=rate_fraction,
        **options,
    )


def cohort(rows):
    return np.tile(BASE, (rows, 1))


def static_fixture_digest():
    target, quantiles, names = calibration_rows(("CN", "US"))
    record = fit_conformal_quantiles(
        target, quantiles, names, levels=LEVELS, nominals=NOMINALS, min_rows=1
    )
    rows = np.array([[-2.0, -1.0, 0.5, 1.0, 2.0], [-9.0, -8.0, 0.0, 8.0, 9.0]])
    calibrated, adjusted = apply_conformal_quantiles(record, rows, ["US", "CN"])
    digest = hashlib.sha256(calibrator_sha256(record).encode())
    digest.update(calibrated.tobytes())
    digest.update(json.dumps(adjusted, sort_keys=True).encode())
    return digest.hexdigest()


def panel(markets=("US",), sessions=60, assets=8, horizon=1, seed=5, shift=None):
    """Panel explícito con cuantiles base fijos y objetivos con un cambio de escala."""
    rng = np.random.default_rng(seed)
    rows = []
    for market_index, market in enumerate(markets):
        for session in range(sessions):
            scale = 1.0 if shift is None or session < shift else 4.0
            scale *= 1 + market_index
            for _ in range(assets):
                rows.append((market, session + 1, rng.normal(0, scale)))
    groups = np.array([row[0] for row in rows])
    at = np.array([row[1] for row in rows], dtype=np.int64)
    target = np.array([row[2] for row in rows])
    quantiles = np.tile([-2.0, -1.3, 0.0, 1.3, 2.0], (len(rows), 1))
    return quantiles, groups, at, target, at + horizon


def test_static_cqr_path_is_unchanged():
    assert static_fixture_digest() == STATIC_FIXTURE_SHA256


def test_start_takes_the_static_correction_and_the_declared_rate_scale():
    calibrator = start()
    assert calibrator.corrections("US") == {"0.8": 7.0, "0.95": 7.0}
    target, quantiles, names = calibration_rows()
    record = fit_conformal_quantiles(
        target, quantiles, names, levels=LEVELS, nominals=NOMINALS, min_rows=1
    )
    payload = calibrator.export()
    assert payload["header"]["static_calibrator_sha256"] == calibrator_sha256(record)
    intervals = payload["markets"]["US"]["intervals"]
    # γ = κ · max|E|: 0,25 · 8 y 0,25 · 7. El objetivo a es exacto, 0,2 y 0,05.
    assert (intervals["0.8"]["scale"], intervals["0.8"]["rate"]) == (8.0, 2.0)
    assert (intervals["0.95"]["scale"], intervals["0.95"]["rate"]) == (7.0, 1.75)
    assert intervals["0.8"]["target_miscoverage"] == 0.2
    assert intervals["0.95"]["target_miscoverage"] == 0.05


def test_rate_scale_uses_the_absolute_score_when_intervals_are_too_wide():
    # Intervalos de calibración demasiado anchos: todas las puntuaciones son negativas.
    calibrator = OnlineConformal.start(
        np.zeros(20),
        np.tile([-4.0, -3.0, 0.0, 3.0, 4.0], (20, 1)),
        np.array(["US"] * 20),
        levels=LEVELS,
        nominals=NOMINALS,
        min_rows=1,
        rate_fraction=0.5,
    )
    intervals = calibrator.export()["markets"]["US"]["intervals"]
    assert calibrator.corrections("US") == {"0.8": -3.0, "0.95": -4.0}
    assert (intervals["0.8"]["rate"], intervals["0.95"]["rate"]) == (1.5, 2.0)


def test_one_matured_cohort_moves_each_correction_by_its_equation():
    calibrator = start()
    calibrated, emission = calibrator.emit("US", 1, cohort(5), 2)
    assert emission["corrections"] == {"0.8": 7.0, "0.95": 7.0}
    assert calibrated[0].tolist() == [-9.0, -8.0, 0.0, 8.0, 9.0]
    # E80 = |y| − 1 = −1, 7,5, 9, 2 y 7. La última iguala a Q = 7 y cuenta como cubierta.
    # E95 = |y| − 2 = −2, 6,5, 8, 1 y 6. Solo la tercera supera Q = 7.
    update = calibrator.mature("US", 1, np.array([0.0, 8.5, -10.0, 3.0, 8.0]), at=2)
    assert update["intervals"]["0.8"]["miscoverage"] == 2 / 5
    assert update["intervals"]["0.95"]["miscoverage"] == 1 / 5
    assert calibrator.corrections("US") == {
        "0.8": 7.0 + 2.0 * (2 / 5 - 0.2),
        "0.95": 7.0 + 1.75 * (1 / 5 - 0.05),
    }
    # La siguiente emisión usa la corrección nueva y la mediana sigue intacta.
    calibrated, emission = calibrator.emit("US", 2, cohort(1), 3)
    q80, q95 = emission["corrections"]["0.8"], emission["corrections"]["0.95"]
    assert calibrated[0].tolist() == [-2.0 - q95, -1.0 - q80, 0.0, 1.0 + q80, 2.0 + q95]


def test_errors_use_the_correction_emitted_with_the_cohort():
    calibrator = start()
    calibrator.emit("US", 1, cohort(1), 3)
    calibrator.emit("US", 2, cohort(1), 4)
    calibrator.mature("US", 1, np.array([0.0]), at=3)
    # Q80 baja de 7 a 7 + 2 · (0 − 0,2) = 6,6 antes de madurar la segunda cohorte.
    assert calibrator.corrections("US")["0.8"] == 7.0 + 2.0 * (0.0 - 0.2)
    # Con y = 8,3, E80 = 7,3 supera la corrección emitida (7) aunque no la vigente con
    # una tasa mayor. El error cuenta frente a la emitida.
    update = calibrator.mature("US", 2, np.array([8.3]), at=4)
    assert update["intervals"]["0.8"]["emitted"] == 7.0
    assert update["intervals"]["0.8"]["miscoverage"] == 1.0
    # E95 = 6,3 no supera la emitida (7).
    assert update["intervals"]["0.95"]["miscoverage"] == 0.0


def test_late_correction_differs_from_the_correction_in_force():
    # Con la corrección vigente más alta que la emitida, el error solo existe frente a la
    # emitida. Este caso distingue las dos lecturas en el sentido contrario al anterior.
    calibrator = start()
    calibrator.emit("US", 1, cohort(1), 3)
    calibrator.emit("US", 2, cohort(1), 4)
    calibrator.mature("US", 1, np.array([20.0]), at=3)
    assert calibrator.corrections("US")["0.8"] == 7.0 + 2.0 * (1.0 - 0.2)
    update = calibrator.mature("US", 2, np.array([8.3]), at=4)
    assert update["intervals"]["0.8"]["miscoverage"] == 1.0
    assert update["intervals"]["0.8"]["previous"] == 7.0 + 2.0 * 0.8


def test_zero_rate_reproduces_the_static_cqr_bit_for_bit():
    quantiles, groups, at, target, matures = panel(("CN", "US"), sessions=30, horizon=2)
    quantiles = quantiles + np.random.default_rng(9).normal(0, 0.1, (len(groups), 1))
    calibration = calibration_rows(("CN", "US"))
    calibrator = OnlineConformal.start(
        *calibration, levels=LEVELS, nominals=NOMINALS, min_rows=1, rate_fraction=0
    )
    emitted, emissions, updates = replay_online_conformal(
        calibrator, quantiles, groups, at, target, matures
    )
    record = fit_conformal_quantiles(*calibration, levels=LEVELS, nominals=NOMINALS, min_rows=1)
    static, _ = apply_conformal_quantiles(record, quantiles, groups)
    assert emitted.tobytes() == static.tobytes()
    assert len(emissions) == 60 and len(updates) == 56
    assert calibrator.corrections("US") == {"0.8": 7.0, "0.95": 7.0}


@pytest.mark.parametrize("horizon", [1, 2, 4])
@pytest.mark.parametrize("shift", [None, 30])
def test_long_run_coverage_gap_respects_the_delayed_bound(horizon, shift):
    quantiles, groups, at, target, matures = panel(sessions=400, horizon=horizon, shift=shift)
    calibrator = start(rate_fraction=0.05)
    emitted, emissions, updates = replay_online_conformal(
        calibrator, quantiles, groups, at, target, matures
    )
    payload = calibrator.export()["markets"]["US"]
    matured = payload["matured"]
    # Con madurez a h sesiones quedan h − 1 cohortes pendientes en cada emisión y las h
    # últimas no maduran dentro del panel.
    delay = horizon - 1
    assert matured == 400 - horizon
    calibration_target, calibration_quantiles, _ = calibration_rows()
    for nominal, lower, upper in ((0.8, 1, 3), (0.95, 0, 4)):
        key = f"{nominal:g}"
        interval = payload["intervals"][key]
        rate, a = interval["rate"], interval["target_miscoverage"]
        scores = np.r_[
            np.maximum(quantiles[:, lower] - target, target - quantiles[:, upper]),
            np.maximum(
                calibration_quantiles[:, lower] - calibration_target,
                calibration_target - calibration_quantiles[:, upper],
            ),
        ]
        bound = np.abs(scores).max()
        gap = calibrator.coverage_gap("US")[key]
        assert abs(gap) <= (2 * bound + rate * (1 + delay)) / (rate * matured)
        # La corrección vigente y todas las emitidas permanecen en la banda del lema.
        trajectory = [emission["corrections"][key] for emission in emissions]
        trajectory.append(interval["correction"])
        low, high = -bound - rate * (1 + delay) * a, bound + rate * (1 + delay) * (1 - a)
        assert low <= min(trajectory) and max(trajectory) <= high
        # La cobertura de los intervalos emitidos no es menor que la de la puntuación, porque
        # la regla de orden solo ensancha.
        covered = (emitted[:, lower] <= target) & (target <= emitted[:, upper])
        errors = [update["intervals"][key]["miscoverage"] for update in updates]
        assert 1 - covered[: len(updates) * 8].mean() <= np.mean(errors) + 1e-12
    assert np.array_equal(emitted[:, 2], quantiles[:, 2])
    assert np.all(np.diff(emitted, axis=1) >= 0)


def test_tracking_reacts_to_a_scale_shift_that_static_cqr_cannot_follow():
    # Con la escala del objetivo multiplicada por cuatro a mitad del panel, la CQR estática
    # queda infracubierta y el seguimiento vuelve cerca del nominal en la segunda mitad.
    quantiles, groups, at, target, matures = panel(sessions=400, assets=16, shift=200)
    # La calibración procede del primer régimen, con la misma escala unitaria.
    calibration = panel(sessions=100, assets=16, seed=6)
    calibrator = OnlineConformal.start(
        calibration[3],
        calibration[0],
        calibration[1],
        levels=LEVELS,
        nominals=NOMINALS,
        min_rows=1,
        rate_fraction=0.05,
    )
    static = calibrator.corrections("US")["0.8"]
    emitted, _, _ = replay_online_conformal(calibrator, quantiles, groups, at, target, matures)
    late = at > 300
    static_cover = np.abs(target[late]) <= 1.3 + static
    online_cover = (emitted[late, 1] <= target[late]) & (target[late] <= emitted[late, 3])
    assert static_cover.mean() < 0.3
    # Con una tasa fija la corrección oscila alrededor del cuantil y la cobertura de cien
    # sesiones no es exacta. Basta con que su desviación sea menos de un cuarto de la estática.
    assert abs(online_cover.mean() - 0.8) < 0.25 * abs(static_cover.mean() - 0.8)


def test_future_and_immature_labels_do_not_change_earlier_emissions():
    quantiles, groups, at, target, matures = panel(sessions=40, horizon=3)
    reference = replay_online_conformal(start(), quantiles, groups, at, target, matures)[0]
    changed = target.copy()
    changed[at >= 20] *= 50
    later = quantiles.copy()
    later[at >= 20] *= 3
    # La cohorte de la sesión 20 madura en la 23. Con otras etiquetas desde la sesión 20,
    # nada cambia hasta la 22. Con otros cuantiles cambian sus propias filas, pero ninguna
    # emisión anterior.
    for labels, inputs, unchanged in ((changed, quantiles, 22), (target, later, 19)):
        emitted = replay_online_conformal(start(), inputs, groups, at, labels, matures)[0]
        assert emitted[at <= unchanged].tobytes() == reference[at <= unchanged].tobytes()
        assert not np.array_equal(emitted[at >= 24], reference[at >= 24])


def test_a_cohort_matures_only_when_all_its_labels_are_available():
    calibrator = start()
    quantiles = cohort(4)
    groups = np.array(["US"] * 4)
    at = np.array([1, 1, 2, 2])
    # La primera cohorte tiene una fila con etiqueta disponible en 2 y otra en 3.
    matures = np.array([2, 3, 3, 3])
    target = np.array([20.0, 20.0, 0.0, 0.0])
    emitted, emissions, updates = replay_online_conformal(
        calibrator, quantiles, groups, at, target, matures
    )
    assert emissions[1]["corrections"] == {"0.8": 7.0, "0.95": 7.0}
    assert updates == [] and calibrator.pending("US") == [(1, 3), (2, 3)]


def test_labels_available_at_the_prediction_instant_are_used():
    quantiles, groups, at, target, matures = panel(sessions=3, assets=2, horizon=1)
    calibrator = start()
    _, emissions, updates = replay_online_conformal(
        calibrator, quantiles, groups, at, target, matures
    )
    assert [update["at"] for update in updates] == [2, 3]
    assert emissions[1]["corrections"] == {
        key: update["correction"] for key, update in updates[0]["intervals"].items()
    }


def test_markets_and_their_interleaving_are_isolated():
    quantiles, groups, at, target, matures = panel(("CN", "US"), sessions=50, horizon=2)
    joint = replay_online_conformal(
        start(groups=("CN", "US")), quantiles, groups, at, target, matures
    )[0]
    us = groups == "US"
    alone = replay_online_conformal(
        start(groups=("CN", "US")), quantiles[us], groups[us], at[us], target[us], matures[us]
    )[0]
    assert joint[us].tobytes() == alone.tobytes()
    changed = target.copy()
    changed[~us] *= 100
    other = replay_online_conformal(
        start(groups=("CN", "US")), quantiles, groups, at, changed, matures
    )[0]
    assert other[us].tobytes() == joint[us].tobytes()
    assert not np.array_equal(other[~us], joint[~us])
    # Emitir CN antes o después de US en cada sesión no cambia ninguna salida.
    calibrator = start(groups=("CN", "US"))
    reverse = np.empty_like(joint)
    for session in range(1, 51):
        for market in ("US", "CN"):
            for cohort_at, ready in calibrator.pending(market):
                if ready <= session:
                    rows = (groups == market) & (at == cohort_at)
                    calibrator.mature(market, cohort_at, target[rows], at=session)
            rows = (groups == market) & (at == session)
            reverse[rows] = calibrator.emit(
                market, session, quantiles[rows], int(matures[rows].max())
            )[0]
    assert reverse.tobytes() == joint.tobytes()


def run_stream(calibrator, steps, *, restore_at=None):
    rng = np.random.default_rng(17)
    outputs = []
    for step in range(1, steps + 1):
        if restore_at == step:
            calibrator = OnlineConformal.restore(calibrator.export())
        targets = rng.normal(0, 6, 3)
        if step > 2:
            outputs.append(calibrator.mature("US", step - 2, targets, at=step)["intervals"])
        outputs.append(calibrator.emit("US", step, cohort(3) * (1 + step / 10), step + 2)[0])
    return calibrator, outputs


def test_recovery_from_the_exported_state_is_exact():
    uninterrupted, expected = run_stream(start(), 12)
    resumed, outputs = run_stream(start(), 12, restore_at=7)
    assert resumed.state_sha256() == uninterrupted.state_sha256()
    for left, right in zip(outputs, expected, strict=True):
        if isinstance(left, np.ndarray):
            assert left.tobytes() == right.tobytes()
        else:
            assert left == right


def test_exported_state_is_a_copy_and_tampering_is_rejected():
    calibrator, _ = run_stream(start(), 5)
    payload = calibrator.export()
    digest = payload["state_sha256"]
    calibrator.mature("US", 4, np.zeros(3), at=6)
    assert payload["state_sha256"] == digest and len(payload["markets"]["US"]["pending"]) == 2
    assert OnlineConformal.restore(payload).state_sha256() == digest

    def tampered(change):
        copy = OnlineConformal.restore(payload).export()
        change(copy)
        return copy

    def correction(copy):
        copy["markets"]["US"]["intervals"]["0.8"]["correction"] += 1e-9

    def quantiles(copy):
        copy["markets"]["US"]["pending"][0]["quantiles"] = (
            copy["markets"]["US"]["pending"][0]["quantiles"] + 1.0
        )

    for change, message in ((correction, "huella"), (quantiles, "cuantiles")):
        with pytest.raises(ValueError, match=message):
            OnlineConformal.restore(tampered(change))

    def reorder(copy):
        pending = copy["markets"]["US"]["pending"]
        pending.reverse()
        copy["markets"]["US"]["last_emitted"] = pending[-1]["prediction_at"]
        copy["state_sha256"] = online_conformal._state_sha256(copy)

    with pytest.raises(ValueError, match="orden"):
        OnlineConformal.restore(tampered(reorder))

    def method(copy):
        copy["header"]["method"] = "otro"

    with pytest.raises(ValueError, match="otro método"):
        OnlineConformal.restore(tampered(method))


def rejected(calibrator, operation, message):
    before = calibrator.state_sha256()
    with pytest.raises(ValueError, match=message):
        operation(calibrator)
    assert calibrator.state_sha256() == before


def test_invalid_operations_are_rejected_without_changing_the_state():
    calibrator = start(max_pending=2)
    calibrator.emit("US", 10, cohort(2), 12)
    calibrator.emit("US", 11, cohort(2), 13)
    cases = [
        (lambda c: c.mature("US", 10, np.zeros(2), at=11), "no es madura"),
        (lambda c: c.mature("US", 11, np.zeros(2), at=13), "fuera de orden"),
        (lambda c: c.mature("US", 9, np.zeros(2), at=13), "ya maduró"),
        (lambda c: c.mature("US", 10, np.zeros(3), at=12), "una etiqueta real"),
        (lambda c: c.mature("US", 10, np.array([0.0, np.nan]), at=12), "NaN"),
        (lambda c: c.mature("CN", 10, np.zeros(2), at=12), "no tiene calibración"),
        (lambda c: c.emit("US", 12, cohort(2), 14), "máximo de cohortes"),
        (lambda c: c.emit("US", 11, cohort(2), 14), "orden creciente"),
        (lambda c: c.emit("US", 12, cohort(2), 12), "después de la predicción"),
        (lambda c: c.emit("US", 1_704_067_200_000_000, cohort(1), 1_704_067_200_000_001), "fuera"),
        (lambda c: c.emit("US", 12.0, cohort(1), 14), "entero"),
    ]
    for operation, message in cases:
        rejected(calibrator, operation, message)
    calibrator.mature("US", 10, np.zeros(2), at=12)
    calibrator.mature("US", 11, np.zeros(2), at=13)
    rejected(calibrator, lambda c: c.mature("US", 11, np.zeros(2), at=14), "pendientes")
    crossed = [[0.0, -1.0, 0.0, 1.0, 2.0]]
    rejected(calibrator, lambda c: c.emit("US", 14, crossed, 16), "cruzados")
    # Tras aplicar una actualización en 13, una emisión anterior usaría información futura.
    calibrator.emit("US", 13, cohort(1), 20)
    rejected(calibrator, lambda c: c.mature("US", 13, np.zeros(1), at=12), "no es madura")


def test_operations_follow_the_market_clock():
    calibrator = start()
    calibrator.emit("US", 1, cohort(1), 2)
    calibrator.mature("US", 1, np.zeros(1), at=5)
    # Una emisión en 3 usaría una actualización aplicada en 5.
    rejected(calibrator, lambda c: c.emit("US", 3, cohort(1), 9), "anterior a una actualización")
    calibrator.emit("US", 6, cohort(1), 7)
    calibrator.emit("US", 9, cohort(1), 10)
    # La etiqueta de la cohorte de 6 es madura en 7, pero el mercado ya emitió en 9.
    rejected(calibrator, lambda c: c.mature("US", 6, np.zeros(1), at=8), "retroceder")


@pytest.mark.parametrize(
    "options,message",
    [
        (dict(rate_fraction=-0.1), "κ"),
        (dict(rate_fraction=1.5), "κ"),
        (dict(rate_fraction=float("nan")), "κ"),
        (dict(max_pending=0), "pendientes"),
        (dict(max_pending=65), "pendientes"),
    ],
)
def test_invalid_declarations_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        start(**options)


def test_undefined_static_corrections_cannot_start_the_tracking():
    target, quantiles, names = calibration_rows()
    with pytest.raises(ValueError, match="corrección inicial"):
        OnlineConformal.start(
            target,
            quantiles,
            names,
            levels=LEVELS,
            nominals=NOMINALS,
            min_rows=50,
            rate_fraction=0.1,
        )
