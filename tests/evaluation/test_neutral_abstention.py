"""Abstención neutral A2 y sus controles con la misma cobertura (#748).

Los paneles se escriben a mano o con un generador de semilla fija y solo fijan formas y
casos límite. No hay modelos ajustados ni pasos de optimizador.
"""

import json

import numpy as np
import pytest

from mars_titan.evaluation import neutral_abstention as na
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.evaluation.output_stages import OUTPUT_STAGES, checked_output_stages

LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)
DAY = 86_400_000_000
START = 1_672_617_600_000_000  # 2023-01-02, antes del test reservado
CONFIG = na.DEFAULT | dict(min_rows=10)


def panel(median, spread, sessions, *, target=None, markets=None, start=START):
    """Panel en orden canónico: mercado y sesión no decrecientes e identidades crecientes."""
    median = np.asarray(median, dtype=np.float64)
    spread = np.asarray(spread, dtype=np.float64)
    rows = len(median)
    target = np.zeros(rows) if target is None else np.asarray(target, dtype=np.float64)
    markets = np.full(rows, "US") if markets is None else np.asarray(markets)
    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    return ForecastPanel.from_columns(
        np.array([f"r{i:05d}" for i in range(rows)]),
        markets,
        start + np.asarray(sessions, dtype=np.int64) * DAY,
        target,
        median,
        quantiles=median[:, None] + spread[:, None] * offsets,
        levels=LEVELS,
    )


def fitted(spreads=None, config=CONFIG):
    """CQR con objetivos nulos y q0,1 = −s, q0,9 = s: la puntuación de 0,8 vale −s."""
    spreads = np.arange(1.0, 21.0) if spreads is None else np.asarray(spreads)
    sessions = np.arange(len(spreads))
    calibration = panel(np.zeros(len(spreads)), spreads, sessions, start=START - 400 * DAY)
    record = na.fit(calibration, calibration.prediction_at + DAY, config)
    return record, int(calibration.prediction_at.max() + DAY)


def random_case(rng, rows=400, days=20):
    sessions = np.sort(rng.integers(0, days, rows))
    median = rng.normal(0, 1, rows)
    spread = rng.uniform(0.2, 2.0, rows)
    target = 0.3 * median + rng.normal(0, 1, rows)
    return panel(median, spread, sessions, target=target)


def test_the_stage_emits_the_median_only_when_the_calibrated_interval_excludes_zero():
    record, _ = fitted()
    # Con n = 20, el orden de 0,8 es ceil(21 · 0,8) = 17 y Q = −4: el intervalo
    # [m − 5, m + 5] queda en [m − 1, m + 1].
    assert record["calibrator"]["groups"]["US"]["intervals"]["0.8"]["correction"] == -4.0
    median = [2.0, -2.0, 0.5, -1.0, 1.0, 1.5]
    evaluation = panel(median, np.full(6, 5.0), [0, 0, 0, 1, 1, 1])
    prediction, abstained = na.apply(record, evaluation)
    # Los extremos que tocan el cero (m = ±1) no lo excluyen.
    assert prediction.tolist() == [2.0, -2.0, 0.0, 0.0, 0.0, 1.5]
    assert abstained.tolist() == [False, False, True, True, True, False]


def test_the_wider_declared_interval_uses_its_own_correction():
    record, _ = fitted(config=CONFIG | dict(nominal=0.95))
    # Puntuación −2s y orden ceil(21 · 0,95) = 20: Q = −2 y [m − 10, m + 10] queda en
    # [m − 8, m + 8].
    assert record["calibrator"]["groups"]["US"]["intervals"]["0.95"]["correction"] == -2.0
    evaluation = panel([8.5, 8.0, -9.0], np.full(3, 5.0), [0, 0, 0])
    assert na.apply(record, evaluation)[0].tolist() == [8.5, 0.0, -9.0]


def test_the_calls_are_the_sign_commitments_of_the_calibrated_panel():
    rng = np.random.default_rng(3)
    calibration = random_case(rng)
    matures = calibration.prediction_at + DAY
    record = na.fit(calibration, matures, CONFIG)
    evaluation = random_case(np.random.default_rng(4))
    shifted = panel(
        evaluation.prediction,
        (evaluation.quantiles[:, 3] - evaluation.quantiles[:, 2]),
        (evaluation.prediction_at - START) // DAY + 30,
        target=evaluation.target,
    )
    prediction, abstained = na.apply(record, shifted)
    calibrated = shifted.quantiles.copy()
    groups = np.asarray(shifted.markets)[shifted.market]
    calibrated, _ = na.cqr.apply_conformal_quantiles(record["calibrator"], calibrated, groups)
    scores = score_sessions(
        ForecastPanel.from_columns(
            shifted.row_id,
            groups,
            shifted.prediction_at,
            shifted.target,
            shifted.prediction,
            quantiles=calibrated,
            levels=LEVELS,
        )
    )
    index = [nominal for nominal, *_ in scores.intervals].index(0.8)
    calls = np.bincount(shifted.session[~abstained], minlength=shifted.sessions)
    assert np.array_equal(calls, scores.commitments[:, index])
    # La predicción emitida conserva el signo de la mediana.
    assert np.all(np.sign(prediction[~abstained]) == np.sign(shifted.prediction[~abstained]))


def test_all_or_no_calls_reproduce_the_median_or_zero_bit_for_bit():
    record, _ = fitted()
    median = np.array([3.0, -7.25, 1.0000000001, -2.5])
    every = panel(median, np.full(4, 4.0), [0, 0, 1, 1])
    prediction, abstained = na.apply(record, every)
    assert prediction.tobytes() == every.prediction.tobytes() and not abstained.any()
    none = panel(np.array([0.5, -0.5]), np.full(2, 5.0), [0, 1])
    prediction, abstained = na.apply(record, none)
    assert prediction.tolist() == [0.0, 0.0] and abstained.all()


def test_without_a_declared_stage_the_output_is_the_model_output():
    assert checked_output_stages(None) is None
    evaluation = random_case(np.random.default_rng(8))
    before = evaluation.prediction.tobytes(), evaluation.quantiles.tobytes()
    record, _ = fitted()
    na.emit(record, evaluation, stream=0, volatility=np.zeros(400), disagreement=np.zeros(400))
    assert (evaluation.prediction.tobytes(), evaluation.quantiles.tobytes()) == before


def test_the_registry_validates_each_declared_stage():
    declared = {na.IDENTITY: dict(na.DEFAULT)}
    assert checked_output_stages(declared) == declared
    assert list(OUTPUT_STAGES) == [na.IDENTITY]
    for wrong in ({}, {"scale_shrink_v1": {}}, [na.IDENTITY], {na.IDENTITY: {"nominal": 0.8}}):
        with pytest.raises(ValueError):
            checked_output_stages(wrong)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nominal", 1),
        ("nominal", 1.0),
        ("min_rows", True),
        ("min_rows", 0),
        ("random_seed", -1),
        ("random_seed", 2**32),
        ("volatility_window", 0),
        ("volatility_min_labels", 61),
    ],
)
def test_the_configuration_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        na.checked_config(na.DEFAULT | {field: value})


def test_the_declared_interval_must_be_emitted():
    with pytest.raises(ValueError, match="no emite el intervalo"):
        fitted(config=CONFIG | dict(nominal=0.5))


def test_a_calibration_label_that_matures_after_an_emission_is_rejected():
    record, fitted_until = fitted()
    # La emisión en el mismo instante en que madura la última etiqueta es válida.
    on_time = panel([2.0], [5.0], [0], start=fitted_until)
    assert na.apply(record, on_time)[0].tolist() == [2.0]
    early = panel([2.0], [5.0], [0], start=fitted_until - 1)
    with pytest.raises(ValueError, match="maduran después"):
        na.apply(record, early)


def test_a_label_must_mature_after_its_prediction_and_before_2024():
    calibration = panel(np.zeros(20), np.arange(1.0, 21.0), np.arange(20), start=START - 400 * DAY)
    with pytest.raises(ValueError, match="después de su predicción"):
        na.fit(calibration, calibration.prediction_at, CONFIG)
    with pytest.raises(ValueError, match="2024"):
        na.fit(calibration, np.full(20, na.FINAL_TEST_START_US), CONFIG)


def test_the_emission_never_reads_the_evaluation_targets():
    record, _ = fitted()
    rng = np.random.default_rng(12)
    first = random_case(rng)
    other = panel(
        first.prediction,
        first.quantiles[:, 3] - first.quantiles[:, 2],
        (first.prediction_at - START) // DAY,
        target=-first.target + 3.0,
    )
    left, right = na.apply(record, first), na.apply(record, other)
    assert left[0].tobytes() == right[0].tobytes() and np.array_equal(left[1], right[1])


def test_the_record_survives_json_and_rejects_changes():
    record, _ = fitted()
    evaluation = random_case(np.random.default_rng(5))
    restored = json.loads(json.dumps(record))
    assert na.stage_sha256(restored) == na.stage_sha256(record)
    assert na.apply(restored, evaluation)[0].tobytes() == na.apply(record, evaluation)[0].tobytes()
    changed = json.loads(json.dumps(record))
    changed["calibrator"]["groups"]["US"]["intervals"]["0.8"]["correction"] = -3.0
    with pytest.raises(ValueError, match="huella"):
        na.apply(changed, evaluation)
    for field, value in (("fitted_until", na.FINAL_TEST_START_US), ("identity", "other")):
        with pytest.raises(ValueError, match="contrato"):
            na.apply(restored | {field: value}, evaluation)
    other = na.fit(*_calibration(spreads=np.arange(2.0, 22.0)), CONFIG)
    assert na.stage_sha256(other) != na.stage_sha256(record)


def _calibration(spreads):
    calibration = panel(
        np.zeros(len(spreads)), spreads, np.arange(len(spreads)), start=START - 400 * DAY
    )
    return calibration, calibration.prediction_at + DAY


def test_the_point_prediction_must_be_the_median():
    record, _ = fitted()
    evaluation = random_case(np.random.default_rng(6))
    moved = ForecastPanel.from_columns(
        evaluation.row_id,
        np.asarray(evaluation.markets)[evaluation.market],
        evaluation.prediction_at,
        evaluation.target,
        evaluation.prediction + 1e-9,
        quantiles=evaluation.quantiles,
        levels=LEVELS,
    )
    with pytest.raises(ValueError, match="mediana"):
        na.apply(record, moved)


def test_matched_controls_abstain_on_as_many_rows_of_each_session():
    record, _ = fitted()
    evaluation = random_case(np.random.default_rng(9))
    _, abstained = na.apply(record, evaluation)
    rng = np.random.default_rng(10)
    volatility = np.where(rng.random(400) < 0.1, np.nan, rng.random(400))
    outputs, summary = na.emit(
        record, evaluation, stream=2, volatility=volatility, disagreement=rng.random(400)
    )
    expected = np.bincount(evaluation.session[abstained], minlength=evaluation.sessions)
    for name in (na.IDENTITY, *na.CONTROLS):
        zero = outputs[name] == 0
        assert np.array_equal(
            np.bincount(evaluation.session[zero], minlength=evaluation.sessions), expected
        ), name
        # Donde no se abstiene, cada control emite la mediana sin cambios.
        assert np.array_equal(outputs[name][~zero], evaluation.prediction[~zero]), name
    assert summary["abstained_rows"] == int(abstained.sum())
    assert summary["undefined_volatility_rows"] == int(np.isnan(volatility).sum())
    assert len(set(summary["call_coverage"].values())) == 1


def test_each_control_ranks_rows_by_its_own_score():
    sessions = [0, 0, 0, 0, 1, 1, 1]
    evaluation = panel([0.3, -0.1, 0.2, -0.4, 0.5, 0.5, -0.05], np.full(7, 1.0), sessions)
    abstained = np.array([True, True, False, False, True, False, False])
    small = na.matched_abstention(evaluation, abstained, -np.abs(evaluation.prediction))
    assert small.tolist() == [False, True, True, False, False, False, True]
    # Empate entre las filas 4 y 5: decide la posición canónica.
    tied = na.matched_abstention(evaluation, abstained, np.r_[np.zeros(4), 1.0, 1.0, 0.0])
    assert tied.tolist()[4:] == [True, False, False]
    # Una puntuación no definida queda detrás de las definidas, aunque estas sean negativas.
    score = np.array([np.nan, -3.0, np.nan, -5.0, np.nan, np.nan, np.nan])
    assert na.matched_abstention(evaluation, abstained, score).tolist() == [
        False,
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    with pytest.raises(ValueError, match="infinita"):
        na.matched_abstention(evaluation, abstained, np.r_[np.inf, np.zeros(6)])


def test_the_random_control_is_reproducible_per_stream_and_uniform_within_sessions():
    record, _ = fitted()
    days = 2000
    evaluation = panel(np.zeros(4 * days), np.ones(4 * days), np.repeat(np.arange(days), 4))
    abstained = np.tile([True, False, False, False], days)
    scores = [
        na.control_scores(record, evaluation, stream=stream, volatility=0, disagreement=0)
        for stream in (0, 0, 1)
    ]
    assert np.array_equal(scores[0]["random"], scores[1]["random"])
    assert not np.array_equal(scores[0]["random"], scores[2]["random"])
    chosen = na.matched_abstention(evaluation, abstained, scores[0]["random"])
    frequency = chosen.reshape(days, 4).mean(axis=0)
    # Cada posición se elige con probabilidad 1/4. Con 2.000 sesiones la desviación típica
    # de la frecuencia es 0,0097, así que 0,05 son más de cinco desviaciones.
    assert np.all(np.abs(frequency - 0.25) < 0.05)
    with pytest.raises(ValueError, match="semilla del modelo"):
        na.control_scores(record, evaluation, stream=-1, volatility=0, disagreement=0)


def naive_volatility(assets, prediction_at, history, window, min_labels):
    result = []
    for asset, at in zip(assets, prediction_at, strict=True):
        labels = sorted(
            (matures, target)
            for name, target, matures in zip(*history.values(), strict=True)
            if name == asset and matures <= at
        )[-window:]
        result.append(
            np.sqrt(np.mean([t * t for _, t in labels])) if len(labels) >= min_labels else np.nan
        )
    return np.array(result)


def test_past_volatility_matches_a_direct_reference():
    rng = np.random.default_rng(14)
    names = np.array(["US/A", "US/B", "US/C"])
    labels = 300
    history = dict(
        asset_id=names[rng.integers(0, 3, labels)],
        target=rng.normal(0, 0.02, labels),
        matures_at=START - 500 * DAY + rng.permutation(400)[:labels] * DAY,
    )
    assets = np.r_[names[rng.integers(0, 3, 80)], ["US/Z"] * 3]
    at = START - 500 * DAY + rng.integers(0, 420, len(assets)) * DAY
    value = na.past_volatility(assets, at, history, window=7, min_labels=3)
    expected = naive_volatility(assets, at, history, 7, 3)
    assert np.array_equal(np.isnan(value), np.isnan(expected))
    assert np.allclose(value[~np.isnan(value)], expected[~np.isnan(expected)], rtol=1e-12)
    assert np.isnan(value[-3:]).all()


def test_past_volatility_only_reads_labels_matured_at_emission():
    history = dict(
        asset_id=["US/A"] * 4,
        target=[3.0, 4.0, 100.0, 1e6],
        matures_at=[START, START + DAY, START + 2 * DAY, START + 3 * DAY],
    )
    at = np.array([START + DAY, START + 2 * DAY])
    value = na.past_volatility(["US/A", "US/A"], at, history, window=2, min_labels=2)
    # En t = 1 cuentan 3 y 4. En t = 2 la etiqueta que madura justo en t ya cuenta.
    assert value[0] == pytest.approx(np.sqrt(12.5)) and value[1] == pytest.approx(
        np.sqrt((16 + 10_000) / 2)
    )
    # Cambiar una etiqueta que todavía no ha madurado no cambia nada.
    future = history | dict(target=[3.0, 4.0, 100.0, -5.0])
    assert np.array_equal(
        na.past_volatility(["US/A", "US/A"], at, future, window=2, min_labels=2), value
    )
    with pytest.raises(ValueError, match="2024"):
        na.past_volatility(
            ["US/A"],
            [START],
            history | dict(matures_at=[START, START, START, na.FINAL_TEST_START_US]),
            window=2,
            min_labels=1,
        )


def test_seed_disagreement_needs_the_same_rows_in_every_seed():
    rng = np.random.default_rng(15)
    first = random_case(rng)
    second = panel(
        first.prediction * 2.0,
        first.quantiles[:, 3] - first.quantiles[:, 2],
        (first.prediction_at - START) // DAY,
        target=first.target,
    )
    expected = np.abs(first.prediction - second.prediction) / 2
    assert np.allclose(na.seed_disagreement([first, second]), expected, rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="dos semillas"):
        na.seed_disagreement([first])
    with pytest.raises(ValueError, match="mismas filas"):
        na.seed_disagreement([first, random_case(np.random.default_rng(16))])


def test_past_volatility_rejects_two_labels_of_an_asset_with_the_same_maturity():
    history = dict(
        asset_id=["US/A", "US/B", "US/A"], target=[1.0, 2.0, 3.0], matures_at=[START] * 3
    )
    with pytest.raises(ValueError, match="mismo instante"):
        na.past_volatility(["US/A"], [START], history, window=2, min_labels=1)


def test_the_small_prediction_control_keeps_the_largest_medians():
    record, _ = fitted()
    evaluation = panel([0.3, -0.1, 0.2, -0.4], np.full(4, 1.0), [0, 0, 0, 0])
    scores = na.control_scores(record, evaluation, stream=0, volatility=0, disagreement=0)
    abstained = np.array([True, True, False, False])
    chosen = na.matched_abstention(evaluation, abstained, scores["small_prediction"])
    assert chosen.tolist() == [False, True, True, False]
