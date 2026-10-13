"""Abstención con su cobertura: curva de dirección, cobertura de llamadas y marcas (#442).

Los paneles se escriben a mano con valores conocidos. No hay modelos ajustados ni pasos de
optimizador.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from mars_titan.evaluation import abstention_coverage as abstention
from mars_titan.evaluation import forecast_scores
from mars_titan.evaluation.forecast_scores import SELECTIVE_COVERAGES, score_sessions
from mars_titan.integrity import independent_scores, score_recheck
from tests.evaluation.test_forecast_scores import build, random_panel


def coverage_index(value):
    return SELECTIVE_COVERAGES.index(value)


def test_selective_hits_rank_by_absolute_prediction_and_count_abstentions_as_misses():
    # Sesión 0: cinco filas juzgables y un objetivo nulo con la predicción más fuerte.
    # Por |predicción|: 3,0 (acierto), 2,0 (fallo), 1,0 y -1,0 empatan y decide la fila
    # canónica (r0003 acierta antes que r0004), y la abstención va la última.
    target = [1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 2.0, -2.0]
    prediction = [3.0, 2.0, 0.0, 1.0, -1.0, 9.0, 0.0, 0.0]
    scores = score_sessions(build(target, prediction, [0, 0, 0, 0, 0, 0, 1, 1]))
    hits, retained = scores.selective_hits, scores.selective_rows()
    assert scores.direction_eligible.tolist() == [5, 2]
    expected = {0.1: (1, 1), 0.2: (1, 1), 0.4: (1, 2), 0.6: (2, 3), 0.8: (2, 4), 1.0: (2, 5)}
    for coverage, (found, rows) in expected.items():
        assert (hits[0, coverage_index(coverage)], retained[0, coverage_index(coverage)]) == (
            found,
            rows,
        ), coverage
    # La sesión que siempre se abstiene no acierta a ninguna cobertura.
    assert hits[1].tolist() == [0] * len(SELECTIVE_COVERAGES)
    # Con c = 1 la curva es la exactitud de dirección.
    assert hits[:, -1].tolist() == scores.direction_hits.tolist()
    series = scores.series("direction_accuracy@1")
    assert np.array_equal(series.values, scores.series("direction_accuracy").values)
    assert series.loss is False


def test_a_session_without_judged_rows_has_no_selective_accuracy():
    scores = score_sessions(build([0.0, 0.0, 1.0], [1.0, -1.0, 1.0], [0, 0, 1]))
    assert scores.selective_rows()[0].tolist() == [0] * len(SELECTIVE_COVERAGES)
    series = scores.series(f"{forecast_scores.SELECTIVE_ACCURACY}0.5")
    assert series.defined.tolist() == [False, True] and series.values[1] == 1.0
    with pytest.raises(ValueError, match="no calculada"):
        scores.series(f"{forecast_scores.SELECTIVE_ACCURACY}0.55")


def test_the_curve_counts_rows_and_hits_and_survives_tables_and_windows():
    rng = np.random.default_rng(5)
    first, second = (score_sessions(random_panel(rng, quantiles=False)) for _ in range(2))
    curve = first.summary()["selective_direction"]
    assert curve["rule"] == forecast_scores.SELECTIVE_RULE
    assert [point["coverage"] for point in curve["curve"]] == list(SELECTIVE_COVERAGES)
    for index, point in enumerate(curve["curve"]):
        assert point["retained_rows"] == int(first.selective_rows()[:, index].sum())
        assert point["hits"] == int(first.selective_hits[:, index].sum())
    table = first.to_table()
    assert np.array_equal(
        np.vstack(table.column("selective_hits").to_pylist()), first.selective_hits
    )
    # Al unir ventanas, cada sesión conserva sus aciertos en el orden canónico (mercado,
    # instante) de la unión.
    later = dataclasses.replace(second, session_time=second.session_time + 10**12)
    joined = forecast_scores.SessionScores.concatenate([first, later])
    order = np.lexsort(
        (
            np.concatenate([first.session_time, later.session_time]),
            np.concatenate([first.session_market, later.session_market]),
        )
    )
    assert len(set(first.session_market)) == 2
    assert np.array_equal(
        joined.selective_hits, np.vstack([first.selective_hits, second.selective_hits])[order]
    )


def test_the_independent_implementation_gives_the_same_hits_with_ties():
    rng = np.random.default_rng(11)
    rows = 400
    sessions = rng.integers(0, 8, rows)
    target = np.round(rng.normal(size=rows), 0)
    # Predicciones redondeadas para forzar empates y abstenciones.
    prediction = np.round(rng.normal(size=rows), 0)
    scores = score_sessions(build(target, prediction, sessions))
    # Los mismos instantes que `build`, en microsegundos. La identidad de fila de la
    # comparación es mercado/activo/instante y aquí el activo es la fila del panel, así que
    # los empates se resuelven por el mismo orden.
    moments = np.datetime64("2023-03-01T20:00:00", "us") + sessions.astype("timedelta64[D]")
    frame = pd.DataFrame(
        dict(
            asset_id=[f"r{i:04d}" for i in range(rows)],
            market="US",
            prediction_at=moments.astype(np.int64),
            target=target,
            prediction=prediction,
        )
    )
    # Otro orden de lectura: los empates no pueden depender de él.
    frame = frame.iloc[rng.permutation(rows)].reset_index(drop=True)
    independent = independent_scores.session_scores(frame)
    assert np.array_equal(
        np.vstack(independent["selective_hits"].to_numpy()), scores.selective_hits
    )
    # Unas sesiones publicadas con otros aciertos no pasan el cotejo.
    published = independent.copy()
    published["selective_hits"] = [
        hits + (i == 3) for i, hits in enumerate(published["selective_hits"])
    ]
    findings = score_recheck.compare_sessions(independent, published)
    assert findings["columns"]["selective_hits"] == dict(mismatched_sessions=1)
    assert not findings["passed"]


def test_low_call_coverage_marks_only_the_contrasts_that_involve_that_model():
    entry = dict(
        contrasts=[
            dict(name="gru-zero", coefficients=dict(gru=1.0, zero=-1.0)),
            dict(name="titans-gru", coefficients=dict(titans=1.0, gru=-1.0, ridge=0.0)),
            dict(name="ridge-gru", coefficients=dict(ridge=1.0, gru=-1.0)),
        ]
    )
    coverage = dict(gru=0.9, titans=0.5, ridge=0.1, zero=0.0)
    abstention.mark(entry, "up_precision", coverage)
    marks = [(row["eligible"], row["ineligible_reason"]) for row in entry["contrasts"]]
    assert marks == [
        (False, "Cobertura de llamadas inferior a 0.5 en zero"),
        (True, None),
        (False, "Cobertura de llamadas inferior a 0.5 en ridge"),
    ]
    assert entry["contrasts"][1]["call_coverage"] == dict(gru=0.9, titans=0.5)
    # Las métricas que juzgan todas las filas no cambian.
    other = dict(contrasts=[dict(name="ridge-gru", coefficients=dict(ridge=1.0, gru=-1.0))])
    assert "eligible" not in abstention.mark(other, "direction_accuracy", coverage)["contrasts"][0]
    assert abstention.call_coverage(0, 0) is None and abstention.call_coverage(3, 4) == 0.75
    with pytest.raises(ValueError, match="no pueden superar"):
        abstention.call_coverage(5, 4)
