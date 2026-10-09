"""Calibración CQR común con ejemplos a mano y datos sintéticos intercambiables."""

import json

import numpy as np
import pytest

from mars_titan.calibration.conformal_quantiles import (
    apply_conformal_quantiles,
    calibrator_sha256,
    fit_conformal_quantiles,
    undefined_groups,
)
from mars_titan.models.quantile_head import LEVELS

NOMINALS = (0.8, 0.95)


def fit(target, quantiles, groups=None, *, min_rows=1):
    target = np.asarray(target, dtype=np.float64)
    groups = np.array(["US"] * len(target)) if groups is None else np.asarray(groups)
    return fit_conformal_quantiles(
        target,
        np.asarray(quantiles, dtype=np.float64),
        groups,
        levels=LEVELS,
        nominals=NOMINALS,
        min_rows=min_rows,
    )


def test_corrections_are_the_declared_order_statistics_by_hand():
    # q0,1 = −1 y q0,9 = 1. Puntuaciones E = max(−1 − y, y − 1) de los nueve objetivos:
    # −1, −0,5, −0,5, 0,5, 1, 2, −0,8, 0,2 y 3. Ordenadas, la octava es 2.
    target = [0, 0.5, -0.5, 1.5, -2, 3, 0.2, -1.2, 4]
    record = fit(target, [[-2, -1, 0, 1, 2]] * 9)
    narrow = record["groups"]["US"]["intervals"]["0.8"]
    assert narrow["order_statistic"] == 8  # ceil(10 · 0,8)
    assert narrow["correction"] == 2.0
    assert narrow["raw_calibration_coverage"] == pytest.approx(4 / 9)
    assert narrow["calibration_coverage"] == pytest.approx(8 / 9)
    # Para el 95 % el orden ceil(10 · 0,95) = 10 supera las nueve filas.
    wide = record["groups"]["US"]["intervals"]["0.95"]
    assert wide["order_statistic"] == 10 and wide["correction"] is None
    assert "supera" in wide["reason"] and wide["calibration_coverage"] is None
    assert undefined_groups(record, ["US"]) == ["US"]
    with pytest.raises(ValueError, match="No hay corrección definida"):
        apply_conformal_quantiles(record, [[-2.0, -1.0, 0.0, 1.0, 2.0]], ["US"])


def test_overconfident_intervals_widen_and_order_is_preserved():
    # Diecinueve filas: el orden ceil(20 · 0,95) = 19 toma la mayor puntuación.
    target = np.arange(-9.0, 10.0)
    record = fit(target, [[-2, -1, 0, 1, 2]] * 19)
    intervals = record["groups"]["US"]["intervals"]
    assert intervals["0.95"]["correction"] == 9 - 2  # max(−2 − y, y − 2) con |y| = 9
    assert intervals["0.8"]["correction"] == 8 - 1  # ceil(20 · 0,8) = 16: |y| = 8
    calibrated, adjusted = apply_conformal_quantiles(record, [[-2.0, -1.0, 0.5, 1.0, 2.0]], ["US"])
    assert calibrated.tolist() == [[-9.0, -8.0, 0.5, 8.0, 9.0]]
    assert adjusted == {"0.8": 0, "0.95": 0}


def test_negative_corrections_shrink_but_never_cross_the_median_or_inner_interval():
    # Calibración con intervalos amplios y objetivos nulos: E80 = −3 y E95 = −4.
    record = fit([0.0] * 20, [[-4, -3, 0, 3, 4]] * 20)
    intervals = record["groups"]["US"]["intervals"]
    assert (intervals["0.8"]["correction"], intervals["0.95"]["correction"]) == (-3.0, -4.0)
    narrow = [[-2.0, -1.0, 0.0, 1.0, 2.0], [-9.0, -8.0, 0.0, 8.0, 9.0]]
    calibrated, adjusted = apply_conformal_quantiles(record, narrow, ["US", "US"])
    # Fila 1: −1 + 3 = 2 cruzaría la mediana y se queda en 0. El 95 % no baja del 80 %.
    assert calibrated[0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    # Fila 2: −8 + 3 = −5 y −9 + 4 = −5 respetan el orden sin ajuste.
    assert calibrated[1].tolist() == [-5.0, -5.0, 0.0, 5.0, 5.0]
    assert adjusted == {"0.8": 1, "0.95": 1}
    assert np.all(np.diff(calibrated, axis=1) >= 0)


def test_each_market_receives_its_own_correction_and_minimum():
    rng = np.random.default_rng(3)
    target = np.r_[rng.normal(0, 1, 400), rng.normal(0, 5, 400)]
    groups = np.array(["US"] * 400 + ["CN"] * 400)
    base = np.array([-1.96, -1.28, 0.0, 1.28, 1.96])
    record = fit(target, np.tile(base, (800, 1)), groups, min_rows=300)
    us = record["groups"]["US"]["intervals"]["0.8"]["correction"]
    cn = record["groups"]["CN"]["intervals"]["0.8"]["correction"]
    assert abs(us) < 0.3 < cn
    assert record["groups"]["CN"]["rows"] == record["groups"]["US"]["rows"] == 400
    small = fit(target, np.tile(base, (800, 1)), groups, min_rows=401)
    entry = small["groups"]["US"]["intervals"]["0.8"]
    assert entry["correction"] is None and "mínimo" in entry["reason"]


@pytest.mark.parametrize("nominal", NOMINALS)
def test_synthetic_coverage_reaches_the_nominal_level(nominal):
    rng = np.random.default_rng(20261009)
    # Modelo sobreconfiado: emite cuantiles de N(0, 1) y el objetivo sigue N(0, 2²).
    base = np.array([-1.959964, -1.281552, 0.0, 1.281552, 1.959964])
    lower, upper = {0.8: (1, 3), 0.95: (0, 4)}[nominal]

    def sample(rows):
        center = rng.normal(0, 0.5, rows)
        return center + rng.normal(0, 2, rows), center[:, None] + base

    calibration_target, calibration_quantiles = sample(2000)
    groups = np.array(["US"] * 2000)
    record = fit(calibration_target, calibration_quantiles, groups, min_rows=1000)
    entry = record["groups"]["US"]["intervals"][f"{nominal:g}"]
    assert entry["raw_calibration_coverage"] < nominal - 0.25
    # En calibración, al menos ceil(2001 · nominal) puntuaciones quedan bajo la corrección.
    assert entry["calibration_coverage"] >= nominal
    assert entry["calibration_coverage"] * 2000 >= entry["order_statistic"]
    target, quantiles = sample(20000)
    calibrated, _ = apply_conformal_quantiles(record, quantiles, np.array(["US"] * 20000))
    covered = (calibrated[:, lower] <= target) & (target <= calibrated[:, upper])
    raw = (quantiles[:, lower] <= target) & (target <= quantiles[:, upper])
    assert raw.mean() < nominal - 0.25
    # Error típico de unos 0,009 por la calibración con 2.000 filas y 0,003 por la muestra.
    assert abs(covered.mean() - nominal) < 0.03


def test_record_is_serializable_frozen_and_does_not_depend_on_later_rows():
    target = np.linspace(-3, 3, 50)
    quantiles = np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], (50, 1))
    record = fit(target, quantiles)
    digest = calibrator_sha256(record)
    assert calibrator_sha256(json.loads(json.dumps(record))) == digest
    first, _ = apply_conformal_quantiles(record, quantiles[:3], ["US"] * 3)
    apply_conformal_quantiles(record, quantiles * 7, ["US"] * 50)
    again, _ = apply_conformal_quantiles(record, quantiles[:3], ["US"] * 3)
    assert calibrator_sha256(record) == digest and np.array_equal(first, again)


@pytest.mark.parametrize(
    "change,message",
    [
        (dict(quantiles=[[0, -1, 0, 1, 2]]), "cruzados"),
        (dict(target=[float("nan")]), "NaN"),
        (dict(levels=(0.1, 0.9)), "mediana"),
        (dict(nominals=(0.8,)), "pares centrales"),
        (dict(levels=(0.025, 0.1, 0.5, 0.9, 0.95)), "pares centrales"),
        (dict(min_rows=0), "mínimo de filas"),
        (dict(groups=[""]), "no vacíos"),
    ],
)
def test_invalid_inputs_fail_before_fitting(change, message):
    options = dict(
        target=[0.0],
        quantiles=[[-2.0, -1.0, 0.0, 1.0, 2.0]],
        groups=["US"],
        levels=LEVELS,
        nominals=NOMINALS,
        min_rows=1,
    )
    options.update(change)
    with pytest.raises(ValueError, match=message):
        fit_conformal_quantiles(
            np.asarray(options.pop("target"), dtype=np.float64),
            np.asarray(options.pop("quantiles"), dtype=np.float64),
            np.asarray(options.pop("groups")),
            **options,
        )


def test_apply_rejects_a_record_from_another_method_or_unknown_group():
    record = fit(np.zeros(20), [[-1.0, -0.5, 0.0, 0.5, 1.0]] * 20)
    with pytest.raises(ValueError, match="No hay corrección"):
        apply_conformal_quantiles(record, [[-1.0, -0.5, 0.0, 0.5, 1.0]], ["CN"])
    with pytest.raises(ValueError, match="método"):
        apply_conformal_quantiles(
            dict(record, method="other"), [[-1.0, -0.5, 0.0, 0.5, 1.0]], ["US"]
        )
