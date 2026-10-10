"""Componentes y escalas de M3 con oráculos independientes, sin modelos ni ajustes."""

import math
import statistics
from dataclasses import asdict, replace

import numpy as np
import pytest

from mars_titan.memory import write_scores as ws
from tests.training.chronological_fixture import US_2023, decision
from tests.training.test_financial_run import corpus


def window(closes):
    """Ventana [64, 5] de log-precios relativos al primer cierre, como `_price_contexts`."""
    closes = np.asarray(closes, dtype=np.float64)
    result = np.zeros((len(closes), 5), dtype=np.float32)
    result[:, :4] = np.log(closes / closes[0])[:, None]
    return result


def fundamentals(ages, observed):
    """Bloque [valores, máscaras, log1p(edad)] con relleno cero en los conceptos ausentes."""
    width = len(ages)
    values = np.zeros(3 * width, dtype=np.float32)
    for index, (age, seen) in enumerate(zip(ages, observed, strict=True)):
        if seen:
            values[index], values[width + index] = 0.5, 1.0
            values[2 * width + index] = np.float32(math.log1p(age))
    return values


def presence(news=False, filings=False):
    return np.array([True, news, True, filings, False])


def features(rows):
    prices = np.stack([row[0] for row in rows])
    flags = np.stack([row[1] for row in rows])
    blocks = np.stack([row[2] for row in rows])
    return ws.decision_features(prices, flags, blocks)


def scalers(**changes):
    values = dict(
        source_sha256="a" * 64,
        dataset_sha256="b" * 64,
        decision_start=10,
        decision_end=20,
        decisions=8,
        labels=8,
        filing_decisions=4,
        news_decisions=2,
        error_median=0.02,
        anomaly_median=1.5,
        filing_age_median=40.0,
        news_share=0.25,
    )
    return ws.WriteScalers(**(values | changes))


def oracle_anomaly(closes):
    """Último log-rendimiento frente a la MAD de los anteriores, con `statistics`."""
    log = [float(np.float32(math.log(c / closes[0]))) for c in closes]
    returns = [b - a for a, b in zip(log, log[1:], strict=False)]
    past, last = returns[:-1], returns[-1]
    center = statistics.median(past)
    spread = statistics.median(abs(value - center) for value in past)
    return abs(last) / max(spread, ws.MAD_FLOOR)


def test_anomaly_compares_the_last_return_with_the_previous_ones_only():
    generator = np.random.default_rng(5)
    for _ in range(20):
        closes = 50 * np.exp(np.cumsum(generator.normal(0, 0.02, 64)))
        (row,) = features([(window(closes), presence(), fundamentals([1.0], [False]))])
        assert row.anomaly == pytest.approx(oracle_anomaly(closes), rel=1e-12, abs=0)
    calm = 50 * np.exp(np.cumsum(np.r_[0.0, np.tile([0.01, -0.01], 31), 0.2]))
    (shock,) = features([(window(calm), presence(), fundamentals([1.0], [False]))])
    # Con la última sesión dentro de la MAD el choque quedaría diluido.
    assert shock.anomaly == pytest.approx(0.2 / 0.01, rel=1e-4)


def test_flat_windows_use_the_floor_without_infinite_values():
    flat = np.full(64, 20.0)
    jump = flat.copy()
    jump[-1] = 21.0
    rows = features([(window(v), presence(), fundamentals([1.0], [False])) for v in (flat, jump)])
    assert rows[0].anomaly == 0.0
    assert math.isfinite(rows[1].anomaly) and rows[1].anomaly > 1e9


def test_each_row_is_independent_of_its_batch():
    generator = np.random.default_rng(11)
    rows = []
    for index in range(7):
        closes = 30 * np.exp(np.cumsum(generator.normal(0, 0.03, 64)))
        ages = generator.uniform(0, 300, 3)
        rows.append(
            (
                window(closes),
                presence(news=index % 2 == 0, filings=index % 3 != 0),
                fundamentals(ages, [index % 3 != 0, False, index % 3 != 0]),
            )
        )
    together = features(rows)
    alone = [features([row])[0] for row in rows]
    assert together == alone
    assert features(rows[::-1]) == together[::-1]


def test_filing_age_is_the_most_recent_observed_concept_and_absence_is_unknown():
    closes = np.linspace(10, 11, 64)
    block = fundamentals([400.0, 3.0, 30.0], [True, False, True])
    (row,) = features([(window(closes), presence(filings=True), block)])
    # El concepto no observado tiene edad cero de relleno y no cuenta como publicación.
    assert row.filing_age_days == pytest.approx(30.0, rel=1e-6)
    (missing,) = features([(window(closes), presence(news=True), fundamentals([0.0], [False]))])
    assert missing.filing_age_days is None and missing.news is True
    with pytest.raises(ValueError, match="no tiene conceptos"):
        features([(window(closes), presence(filings=True), fundamentals([1.0], [False]))])


@pytest.mark.parametrize(
    "change",
    [
        lambda p, f, b: (p[:, :, :4], f, b),
        lambda p, f, b: (p, f.astype(np.int8), b),
        lambda p, f, b: (p, np.array([[False, False, True, False, False]]), b),
        lambda p, f, b: (p, f, b[:, :-1]),
    ],
)
def test_inputs_outside_the_masked_contract_are_rejected(change):
    prices = window(np.linspace(10, 11, 64))[None]
    flags, block = presence()[None], fundamentals([1.0], [False])[None]
    with pytest.raises(ValueError):
        ws.decision_features(*change(prices, flags, block))


def test_normalization_maps_the_training_median_to_one_half_and_stays_bounded():
    value = scalers()
    e, a, r, mask = ws.normalized(value, [0.02, 1.5, 40.0, False])
    assert (e, a, r, mask) == (0.5, 0.5, 0.5, 1)
    errors = [0.0, 1e-9, 0.01, 0.02, 0.5, 1e6]
    normalized = [ws.normalized(value, [x, 0.0, None, False])[0] for x in errors]
    assert normalized == sorted(set(normalized)) and normalized[0] == 0.0
    assert all(0 <= x < 1 for x in normalized)
    fresh, stale = (ws.normalized(value, [0.0, 0.0, age, False])[2] for age in (0.0, 4000.0))
    assert fresh == 1.0 and 0 < stale < 0.01


def test_missing_relevance_is_neutral_and_news_never_lowers_it():
    value = scalers()
    unknown = ws.normalized(value, [0.0, 0.0, None, False])
    assert unknown[2:] == (ws.NEUTRAL, 0)
    news = ws.normalized(value, [0.0, 0.0, None, True])
    assert news[2:] == (1 - 0.25 / 2, 2)
    for age in (0.0, 10.0, 40.0, 4000.0):
        alone = ws.normalized(value, [0.0, 0.0, age, False])[2]
        both = ws.normalized(value, [0.0, 0.0, age, True])
        assert both[2] == max(alone, 1 - 0.25 / 2) >= alone and both[3] == 3
    # Sin escala contable de entrenamiento la antigüedad no se interpreta.
    blind = scalers(filing_decisions=0, filing_age_median=None)
    assert ws.normalized(blind, [0.0, 0.0, 5.0, False])[2:] == (ws.NEUTRAL, 0)


def test_score_combines_the_three_components_in_a_fixed_order():
    assert ws.score((1.0, 0.0, 0.0), 0.2, 0.9, 0.7) == 0.2
    assert ws.score((0.0, 0.0, 1.0), 0.2, 0.9, 0.7) == 0.7
    assert ws.score(ws.WEIGHTS, 0.3, 0.6, 0.9) == pytest.approx(0.6, abs=1e-15)
    assert sum(ws.WEIGHTS) == 1.0


@pytest.mark.parametrize(
    "changes",
    [
        dict(error_median=0.0),
        dict(anomaly_median=-1.0),
        dict(anomaly_median=float("inf")),
        dict(error_median=1),
        dict(filing_age_median=None),
        dict(filing_decisions=0),
        dict(news_share=0.5),
        dict(news_decisions=9),
        dict(decision_end=10),
        dict(source_sha256="A" * 64),
        dict(sample_capacity=0),
    ],
)
def test_scalers_reject_values_that_are_not_positive_training_medians(changes):
    with pytest.raises(ValueError):
        scalers(**changes)


def test_scalers_round_trip_with_their_exact_types_and_fingerprint():
    value = scalers()
    assert ws.WriteScalers.from_fields(asdict(value)) == value
    assert value.fingerprint() != replace(value, error_median=0.021).fingerprint()
    with pytest.raises(ValueError):
        ws.WriteScalers.from_fields(asdict(value) | dict(filing_age_median=40))
    with pytest.raises(ValueError, match="campos"):
        ws.WriteScalers.from_fields(asdict(value) | dict(extra=1))
    with pytest.raises(ValueError):
        ws.WriteScalers.from_fields(None)


def test_sample_is_exact_below_capacity_and_uniform_above_it():
    small = ws._Sample(16, 19, 0)
    small.extend([5.0, 1.0, 3.0])
    small.extend([2.0])
    assert small.median() == 2.5 and small.seen == 4
    assert ws._Sample(4, 19, 0).median() is None
    counts = np.zeros(200)
    for seed in range(300):
        sample = ws._Sample(20, seed, 0)
        for start in range(0, 200, 37):
            sample.extend(np.arange(start, min(start + 37, 200), dtype=np.float64))
        assert sample.seen == 200 and sample.size == 20
        counts[sample.values.astype(int)] += 1
    # Cada valor aparece con probabilidad 20/200. Se admite una desviación de cinco sigmas.
    expected = 300 * 0.1
    assert np.abs(counts - expected).max() < 5 * math.sqrt(expected * 0.9)
    left, right = ws._Sample(8, 3, 1), ws._Sample(8, 3, 1)
    for sample in (left, right):
        sample.extend(np.arange(100.0))
    assert np.array_equal(left.values, right.values)


@pytest.fixture(scope="module")
def streams(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("write-scalers"))[1]


def test_scalers_use_only_the_training_tramo(streams, tmp_path):
    fitted = ws.fit_write_scalers(streams["train"], block_rows=2)
    assert fitted.source_sha256 == streams["train"].identity
    assert (fitted.decision_start, fitted.decision_end) == (
        streams["train"].phase.decision_start,
        streams["train"].phase.decision_end,
    )
    assert fitted.decisions > 0 and fitted.labels > 0 and 0 < fitted.news_share < 1
    # Bloques físicos distintos no cambian las escalas.
    assert ws.fit_write_scalers(streams["train"], block_rows=5) == fitted
    # Cambiar entradas, precios y etiquetas posteriores al tramo no cambia ninguna escala.
    _, later = corpus(tmp_path / "later", perturb_after=US_2023)
    assert ws.fit_write_scalers(later["train"], block_rows=2) == replace(
        fitted,
        source_sha256=later["train"].identity,
        dataset_sha256=later["train"].dataset.identity,
    )
    # Control: el mismo cambio dentro del tramo sí las cambia.
    _, inside = corpus(tmp_path / "inside", perturb_after=decision("2022-12-01"))
    changed = ws.fit_write_scalers(inside["train"], block_rows=2)
    assert (changed.error_median, changed.anomaly_median) != (
        fitted.error_median,
        fitted.anomaly_median,
    )
    with pytest.raises(ValueError, match="entrenamiento"):
        ws.fit_write_scalers(streams["validation"], block_rows=2)


def test_scaler_medians_repeat_an_independent_pass(streams):
    """Las medianas repiten un recorrido directo de las mismas decisiones y etiquetas."""
    from mars_titan.models.titans.financial_inputs import validated_cpu_batch

    source = streams["train"]
    specification, phase = source.specification(), source.phase
    errors, anomalies, ages, news, decisions = [], [], [], 0, 0
    for event in source.events():
        errors += [abs(value) for _, _, value in event.labels]
        if event.inputs and event.at >= phase.decision_start:
            for raw in event.inputs:
                for row in ws.batch_features(validated_cpu_batch(raw, specification)):
                    anomalies.append(row.anomaly)
                    decisions += 1
                    news += row.news
                    if row.filing_age_days is not None:
                        ages.append(row.filing_age_days)
    fitted = ws.fit_write_scalers(source, block_rows=3)
    assert fitted.error_median == statistics.median(errors)
    assert fitted.anomaly_median == statistics.median(anomalies)
    assert fitted.filing_age_median == statistics.median(ages)
    assert (fitted.decisions, fitted.news_decisions) == (decisions, news)
