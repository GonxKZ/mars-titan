"""Medidas de seguimiento de la campaña de comparación (#836) sobre tablas escritas a mano.

Ninguna prueba ajusta un modelo: las tablas de predicción se escriben aquí con valores cuyo
resultado se calcula aparte.
"""

import numpy as np
import pytest

from mars_titan.training import comparison_scores as scores
from tests.training.comparison_tables import EVALUATION, declared
from tests.training.comparison_tables import write_tables as tables


def session_mean(sessions, error):
    return np.mean(
        [np.mean([error(value, target) for value, target in pairs]) for _, pairs in sessions]
    )


def test_table_scores_compare_the_model_and_its_calibrated_prediction_with_zero(tmp_path):
    measured = scores.table_scores(tables(tmp_path), magnitude=True)
    assert (measured["sessions"], measured["rows"]) == (2, 6)
    assert measured["scale"] == pytest.approx(0.5)
    assert measured["zero"] == pytest.approx(session_mean(EVALUATION, lambda _, y: abs(y)))
    assert measured["raw"] == pytest.approx(session_mean(EVALUATION, lambda x, y: abs(y - x)))
    assert measured["calibrated"] == pytest.approx(
        session_mean(EVALUATION, lambda x, y: abs(y - 0.5 * x))
    )
    # Valores escritos a mano, para que un cambio de definición no pase inadvertido.
    assert measured["zero"] == pytest.approx(0.04 / 3)
    assert measured["raw"] == pytest.approx((0.07 / 3 + 0.02 / 3) / 2)
    assert measured["calibrated"] == pytest.approx((0.045 / 3 + 0.02 / 3) / 2)
    pearson = np.mean([np.corrcoef(*zip(*pairs, strict=True))[0, 1] for _, pairs in EVALUATION])
    assert measured["pearson_ic"] == pytest.approx(pearson)
    # En la primera sesión los rangos difieren en un intercambio (Spearman 0,5) y en la
    # segunda coinciden.
    assert measured["rank_ic"] == pytest.approx((0.5 + 1.0) / 2)
    assert len(measured["cohort_sha256"]) == 64


def test_a_model_that_predicts_an_order_has_no_raw_error_but_keeps_the_rest(tmp_path):
    # La misma ordenación con cien veces la amplitud: λ lo compensa y el IC no cambia.
    reference = scores.table_scores(tables(tmp_path / "a"), magnitude=True)
    measured = scores.table_scores(tables(tmp_path / "b", scale=100.0), magnitude=False)
    assert measured["raw"] is None
    assert measured["scale"] == pytest.approx(0.005)
    assert measured["calibrated"] == pytest.approx(reference["calibrated"])
    assert measured["pearson_ic"] == pytest.approx(reference["pearson_ic"])
    assert measured["zero"] == reference["zero"]
    assert measured["cohort_sha256"] == reference["cohort_sha256"]


def test_a_prediction_of_zero_is_scored_as_the_zero_prediction_without_an_order(tmp_path):
    measured = scores.table_scores(tables(tmp_path, scale=0.0), magnitude=True)
    assert measured["scale"] is None
    assert measured["raw"] == measured["zero"] == measured["calibrated"]
    assert measured["pearson_ic"] is None and measured["rank_ic"] is None


def test_the_scale_comes_from_the_calibration_and_never_from_the_evaluation(tmp_path):
    # En evaluación el residuo es el doble de la predicción. El λ de calibración sigue en 0,5
    # y no se recorta hacia el valor que convendría con información futura.
    later = [("2019-02-01", [(0.01, 0.02), (-0.02, -0.04), (0.03, 0.06)])]
    measured = scores.table_scores(tables(tmp_path, evaluation=later), magnitude=True)
    assert measured["scale"] == pytest.approx(0.5)
    assert measured["calibrated"] == pytest.approx(session_mean(later, lambda x, y: abs(y - x / 2)))
    assert measured["calibrated"] > measured["raw"]


def test_the_calibration_must_end_before_the_evaluation_starts(tmp_path):
    early = [("2019-01-02", [(0.01, 0.02), (-0.02, -0.04), (0.03, 0.06)])]
    with pytest.raises(ValueError, match="anterior a la evaluación"):
        scores.table_scores(tables(tmp_path, evaluation=early), magnitude=True)


def test_prediction_tables_follow_the_report_of_each_runner(tmp_path):
    paths = tables(tmp_path / "fit")
    report = dict(predictions=declared())
    assert scores.prediction_tables(tmp_path / "fit", report) == [paths]
    # El control tabular deja las tablas de cada semilla junto a su informe, en orden.
    seeds = {seed: tables(tmp_path / "tabular" / f"seed-{seed}") for seed in ("43", "42")}
    report = dict(seeds={seed: dict(path=f"seed-{seed}/run.json") for seed in seeds})
    assert scores.prediction_tables(tmp_path / "tabular", report) == [seeds["42"], seeds["43"]]
    seeds["43"]["calibration"].unlink()
    with pytest.raises(ValueError, match="no conserva sus tablas"):
        scores.prediction_tables(tmp_path / "tabular", report)
    with pytest.raises(ValueError, match="no conserva sus tablas"):
        scores.prediction_tables(tmp_path / "empty", dict(status="completed"))
    # Un informe que no declara alguna partición tampoco sirve.
    partial = dict(predictions=dict(evaluation=dict(path="evaluation-predictions.parquet")))
    with pytest.raises(ValueError, match="no conserva sus tablas"):
        scores.prediction_tables(tmp_path / "fit", partial)


def test_job_scores_average_the_seeds_of_a_job_over_the_same_rows(tmp_path):
    for seed, scale in (("42", 1.0), ("43", 3.0)):
        tables(tmp_path / f"seed-{seed}", scale=scale)
    report = dict(seeds={seed: dict(path=f"seed-{seed}/run.json") for seed in ("42", "43")})
    one = scores.table_scores(tables(tmp_path / "one"), magnitude=False)
    measured = scores.job_scores(tmp_path, report, magnitude=False)
    assert (measured["seeds"], measured["sessions"], measured["rows"]) == (2, 2, 6)
    assert measured["raw"] is None
    assert measured["scale"] == pytest.approx((0.5 + 0.5 / 3) / 2)
    assert measured["calibrated"] == pytest.approx(one["calibrated"])
    assert measured["pearson_ic"] == pytest.approx(one["pearson_ic"])
    assert measured["zero"] == one["zero"] and measured["cohort_sha256"] == one["cohort_sha256"]
    single = dict(predictions=declared("seed-42"))
    assert scores.job_scores(tmp_path, single, magnitude=True)["seeds"] == 1
    # Una semilla con otras filas no es comparable con las demás.
    other = [("2019-02-01", [(0.02, 0.03), (-0.04, -0.01), (0.01, 0.5)])]
    tables(tmp_path / "seed-43", evaluation=other)
    with pytest.raises(ValueError, match="no predicen las mismas filas"):
        scores.job_scores(tmp_path, report, magnitude=False)
