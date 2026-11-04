"""Conformal ponderado por contextos parecidos A3 y sus controles (#748).

Los paneles son sintéticos con semilla fija o escritos a mano. No hay modelos ajustados ni
pasos de optimizador.
"""

import json
import math

import numpy as np
import pytest

from mars_titan.calibration import regime_weighted_conformal as rwc
from mars_titan.evaluation import neutral_abstention as na
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.output_stages import OUTPUT_STAGES, checked_output_stages

LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)
OFFSETS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
# Pares centrales del más estrecho al más ancho: (cobertura, columna baja, columna alta).
PAIRS = ((0.8, 1, 3), (0.95, 0, 4))
DAY = 86_400_000_000
START = 1_577_836_800_000_000  # 2020-01-01
CONFIG = rwc.DEFAULT | dict(min_rows=40, lookback_days=60, regime_days=10)


def make_panel(rng, start, sessions, assets=20, markets=("US",), spread=None, tails=False):
    """Panel diario con la mediana como predicción y etiquetas que maduran al día siguiente."""
    rows = sessions * assets * len(markets)
    market = np.repeat(np.array(markets), sessions * assets)
    session = np.tile(np.repeat(np.arange(sessions), assets), len(markets))
    at = start + session * DAY
    ids = np.array(
        [f"{m}/A{i % assets:03d}/{t}" for i, (m, t) in enumerate(zip(market, at, strict=True))]
    )
    volatility = 0.02 * (1 + (session % 10 >= 5))
    target = rng.normal(0, 1, rows) * volatility
    median = rng.normal(0, 0.005, rows)
    spread = rng.uniform(0.005, 0.03, rows) if spread is None else spread(rows)
    offsets = np.tile(OFFSETS, (rows, 1))
    if tails:
        # Colas con forma propia en cada fila, para que la anchura del intervalo de 0,95 no
        # sea proporcional a la del de 0,8.
        tail = rng.uniform(1.2, 3.0, rows)
        offsets[:, 0], offsets[:, 4] = -tail, tail
    panel = ForecastPanel.from_columns(
        ids,
        market,
        at,
        target,
        median,
        quantiles=median[:, None] + spread[:, None] * offsets,
        levels=LEVELS,
    )
    return panel, panel.prediction_at + DAY


def history(*parts):
    panels = [panel for panel, _ in parts]
    return dict(
        market=np.concatenate([np.asarray(p.markets)[p.market] for p in panels]),
        target=np.concatenate([p.target for p in panels]),
        matures_at=np.concatenate([matures for _, matures in parts]),
    )


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(3)
    validation = make_panel(rng, START, 15)
    calibration = make_panel(rng, START + 20 * DAY, 20)
    evaluation = make_panel(rng, START + 45 * DAY, 20)
    labels = history(validation, calibration, evaluation)
    record = rwc.fit(*calibration, labels, CONFIG)
    return dict(calibration=calibration, evaluation=evaluation, history=labels, record=record)


def reference(record, calibration, evaluation, labels, weighting):
    """Misma etapa con bucles explícitos de Python, fila a fila, sin las funciones del módulo."""
    config, scale = record["config"], record["scale"]

    def rows(panel, matures):
        out = []
        for i in range(panel.rows):
            market = panel.markets[panel.market[i]]
            at = int(panel.prediction_at[i])
            window = [
                y
                for m, y, t in zip(
                    labels["market"], labels["target"], labels["matures_at"], strict=True
                )
                if m == market and at - config["regime_days"] * DAY < t <= at
            ]
            q, y = panel.quantiles[i], panel.target[i]
            width = max(q[3] - q[1], 1e-12)
            if window:
                regime = math.log(max(math.sqrt(sum(v * v for v in window) / len(window)), 1e-12))
                context = [
                    (math.log(width) - scale["mean"][0]) / scale["deviation"][0],
                    (regime - scale["mean"][1]) / scale["deviation"][1],
                ]
            else:
                context = None
            scores = [max(q[lo] - y, y - q[hi]) for _, lo, hi in PAIRS]
            out.append((market, at, int(matures[i]), context, scores, q))
        return out

    pool = rows(*calibration) + rows(*evaluation)
    result = []
    for market, at, _, context, _, q in rows(*evaluation):
        past = [
            (key, scores)
            for m, emitted, matured, key, scores, _ in pool
            if m == market
            and key is not None
            and matured <= at
            and emitted >= at - config["lookback_days"] * DAY
        ]
        if weighting == "uniform":
            weights = [1.0 / len(past)] * len(past)
        else:
            h = config["bandwidth"]
            logits = [
                -((context[0] - k[0]) ** 2 + (context[1] - k[1]) ** 2) / (2 * h * h)
                for k, _ in past
            ]
            top = max(logits)
            raw = [math.exp(v - top) for v in logits]
            weights = [v / sum(raw) for v in raw]
        out = list(q)
        inner_low = inner_high = q[2]
        for k, (nominal, lo, hi) in enumerate(PAIRS):
            mass, correction = 0.0, None
            for score, weight in sorted(
                ((s[k], w) for (_, s), w in zip(past, weights, strict=True)),
                key=lambda pair: pair[0],
            ):
                mass += weight
                if mass >= nominal - 1e-9:
                    correction = score
                    break
            out[lo] = min(q[lo] - correction, inner_low)
            out[hi] = max(q[hi] + correction, inner_high)
            inner_low, inner_high = out[lo], out[hi]
        result.append(out)
    return np.array(result)


@pytest.fixture(scope="module")
def small():
    """Dos mercados, cuatro activos y ventanas de pocos días, donde los bordes cuentan."""
    rng = np.random.default_rng(11)
    calibration = make_panel(rng, START, 8, assets=4, markets=("CN", "US"), tails=True)
    evaluation = make_panel(rng, START + 8 * DAY, 6, assets=4, markets=("CN", "US"), tails=True)
    return dict(
        calibration=calibration, evaluation=evaluation, history=history(calibration, evaluation)
    )


@pytest.mark.parametrize(
    ("weighting", "lookback", "past_rows"),
    [
        # Tres sesiones de cuatro activos por mercado: la emitida justo tres días antes entra
        # y la que madura en el mismo instante de la emisión también.
        ("context", 3, dict(min=12, max=12)),
        ("uniform", 3, dict(min=12, max=12)),
        # Con ocho días entra la primera sesión de calibración, cuyo contexto no está
        # definido porque aún no había madurado ninguna etiqueta: se descarta.
        ("context", 8, dict(min=28, max=32)),
    ],
)
def test_the_replay_matches_a_row_by_row_reference(small, weighting, lookback, past_rows):
    config = rwc.DEFAULT | dict(
        min_rows=past_rows["min"], lookback_days=lookback, regime_days=2, bandwidth=0.7
    )
    parts = small["calibration"], small["evaluation"], small["history"]
    record = rwc.fit(*small["calibration"], small["history"], config)
    calibrated, summary = rwc.replay(record, *parts, weighting=weighting)
    expected = reference(record, *parts, weighting)
    np.testing.assert_allclose(calibrated, expected, rtol=0, atol=1e-15)
    assert summary["past_rows"] == past_rows
    assert summary["sessions"] == 12
    # El mínimo se cumple justo con las filas disponibles y una más ya no basta.
    with pytest.raises(ValueError, match="errores maduros"):
        rwc.replay(
            record | dict(config=config | dict(min_rows=past_rows["min"] + 1)),
            *parts,
            weighting=weighting,
        )


def test_rows_without_a_defined_context_are_rejected_in_every_weighting(small):
    config = rwc.DEFAULT | dict(min_rows=4, lookback_days=3, regime_days=2)
    calibration, evaluation = small["calibration"], small["evaluation"]
    # Sin etiquetas de la evaluación, a partir de su tercera sesión no queda ninguna madura
    # en los dos días previos y el régimen del mercado no está definido.
    labels = history(calibration)
    record = rwc.fit(*calibration, labels, config)
    for weighting in rwc.WEIGHTINGS:
        with pytest.raises(ValueError, match="contexto definido"):
            rwc.replay(record, calibration, evaluation, labels, weighting=weighting)
    late = {name: np.asarray(values) for name, values in labels.items()}
    late["matures_at"] = np.full(len(late["target"]), START + 20 * DAY)
    with pytest.raises(ValueError, match="suficientes"):
        rwc.fit(*calibration, late, config)


def test_evaluating_rows_in_blocks_does_not_change_the_result(small, monkeypatch):
    config = rwc.DEFAULT | dict(min_rows=12, lookback_days=3, regime_days=2, bandwidth=0.7)
    parts = small["calibration"], small["evaluation"], small["history"]
    record = rwc.fit(*small["calibration"], small["history"], config)
    whole, _ = rwc.replay(record, *parts, weighting="context")
    # Con 12 filas pasadas, 30 elementos dejan bloques de dos filas evaluadas.
    monkeypatch.setattr(rwc, "MAX_BLOCK", 30)
    blocks, _ = rwc.replay(record, *parts, weighting="context")
    assert blocks.tobytes() == whole.tobytes()


def test_far_contexts_put_the_weight_on_the_nearest_past_row(case):
    # Con una escala mínima las distancias estandarizadas son enormes y la exponencial sin
    # restar el máximo de cada fila daría cero en todos los pesos.
    record = case["record"] | dict(
        scale=dict(mean=case["record"]["scale"]["mean"], deviation=[1e-6, 1e-6])
    )
    calibrated, summary = rwc.replay(
        record, case["calibration"], case["evaluation"], case["history"], weighting="context"
    )
    assert np.isfinite(calibrated).all()
    assert summary["effective_rows"]["min"] == pytest.approx(1.0)


def test_the_weighted_quantile_matches_a_hand_computation():
    scores = np.array([4.0, 1.0, 3.0, 2.0])
    # Una columna de pesos por fila evaluada.
    weights = np.array([[0.1, 0.4, 0.1, 0.4], [0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]]).T
    # Ordenadas: 1 (0,4), 2 (0,4), 3 (0,1), 4 (0,1): la masa 0,8 se alcanza en 2.
    # Uniformes: 0,25, 0,5, 0,75, 1 alcanzan 0,8 en 4. Con 0,1, 0,1, 0,1, 0,7 también en 4.
    # Con 0,5 la primera fila llega en 2 (0,4 no basta) y la uniforme justo en 2.
    assert rwc.weighted_quantile(scores, weights, 0.8).tolist() == [2.0, 4.0, 4.0]
    assert rwc.weighted_quantile(scores, weights, 0.5).tolist() == [2.0, 2.0, 4.0]
    with pytest.raises(ValueError, match="no alcanzan"):
        rwc.weighted_quantile(scores, weights * 0.5, 0.8)


def test_uniform_weights_give_the_exact_order_statistic():
    rng = np.random.default_rng(1)
    for count in (5, 19, 20, 99, 100, 1000, 4097, 30_000, 30_001):
        scores = rng.normal(size=count)
        weights = np.full((count, 1), 1.0 / count)
        for level in (0.8, 0.95):
            order = math.ceil(round(count * level, 9))
            expected = np.sort(scores)[order - 1]
            assert rwc.weighted_quantile(scores, weights, level)[0] == expected, (count, level)


def test_the_market_regime_reads_only_labels_matured_in_the_window():
    labels = dict(
        market=["US"] * 4 + ["CN"],
        target=[3.0, 4.0, 100.0, -1e6, 7.0],
        matures_at=[START, START + DAY, START + 2 * DAY, START + 9 * DAY, START + DAY],
    )
    at = np.array([START + DAY, START + 2 * DAY, START + 5 * DAY, START])
    value = rwc.market_regime(["US", "US", "US", "CN"], at, labels, days=2)
    # En t = 1 cuentan 3 y 4. En t = 2 ya no cuenta la de t = 0 y sí la que madura en 2.
    assert value[0] == pytest.approx(np.sqrt(12.5))
    assert value[1] == pytest.approx(np.sqrt((16 + 10_000) / 2))
    assert np.isnan(value[2]) and np.isnan(value[3])
    extreme = rwc.market_regime(["US"], [START], labels | dict(target=[1e300, 0, 0, 0, 0]), days=1)
    assert extreme[0] == 1e300


def test_the_scale_uses_only_calibration_contexts_and_labels_already_matured(case):
    panel, matures = case["calibration"]
    record = case["record"]
    features = rwc.contexts(panel, case["history"], days=CONFIG["regime_days"])
    defined = ~np.isnan(features).any(axis=1)
    assert record["scale"]["mean"] == features[defined].mean(axis=0).tolist()
    assert record["scale"]["deviation"] == features[defined].std(axis=0).tolist()
    assert record["fitted_until"] == int(matures.max())
    assert record["nominals"] == [0.8, 0.95]
    # Quitar las etiquetas que maduran después de la última emisión de calibración no cambia
    # nada: el ajuste no las ha leído.
    labels = case["history"]
    known = labels["matures_at"] <= panel.prediction_at.max()
    trimmed = {name: np.asarray(values)[known] for name, values in labels.items()}
    assert rwc.fit(*case["calibration"], trimmed, CONFIG) == record
    flat = make_panel(np.random.default_rng(5), START + 20 * DAY, 20, spread=np.ones)
    with pytest.raises(ValueError, match="constante"):
        rwc.fit(*flat, case["history"], CONFIG)


def test_an_interval_too_wide_collapses_on_the_median_by_the_order_rule():
    rng = np.random.default_rng(7)
    calibration = make_panel(rng, START, 20, spread=lambda rows: rng.uniform(0.9, 1.1, rows))
    # Una sola sesión evaluada, para que el conjunto pasado sea solo la calibración.
    evaluation = make_panel(
        rng, START + 20 * DAY, 1, spread=lambda rows: np.where(np.arange(rows) % 2, 0.5, 2.0)
    )
    labels = history(calibration, evaluation)
    record = rwc.fit(*calibration, labels, CONFIG)
    calibrated, summary = rwc.replay(record, calibration, evaluation, labels, weighting="context")
    panel = evaluation[0]
    narrow = np.arange(panel.rows) % 2 == 1
    # Los errores pasados rondan −1 y −2, así que la corrección encoge cada intervalo en algo
    # más de lo que mide su mitad en las filas estrechas: la regla de orden lo lleva a la mediana.
    assert np.all(calibrated[narrow] == panel.prediction[narrow, None])
    assert summary["order_adjusted_rows"] == {"0.8": int(narrow.sum()), "0.95": int(narrow.sum())}
    assert np.all(calibrated[~narrow, 1] < panel.prediction[~narrow])
    assert np.all(np.diff(calibrated, axis=1) >= 0)


def test_identical_contexts_reproduce_the_uniform_weighting_bit_for_bit(case):
    record = case["record"]
    calibrated, _ = rwc.replay(
        record | dict(scale=dict(mean=[0.0, 0.0], deviation=[1e300, 1e300])),
        case["calibration"],
        case["evaluation"],
        case["history"],
        weighting="context",
    )
    uniform, _ = rwc.replay(
        record, case["calibration"], case["evaluation"], case["history"], weighting="uniform"
    )
    assert calibrated.tobytes() == uniform.tobytes()


def test_the_three_weightings_share_past_rows_and_shuffling_keeps_each_weight_set(case):
    results = {
        name: rwc.replay(
            case["record"], case["calibration"], case["evaluation"], case["history"], weighting=name
        )
        for name in rwc.WEIGHTINGS
    }
    sizes = {name: summary["past_rows"] for name, (_, summary) in results.items()}
    assert len({json.dumps(item, sort_keys=True) for item in sizes.values()}) == 1
    assert results["uniform"][1]["effective_rows"]["min"] == sizes["uniform"]["min"]
    context, shuffled = results["context"][1], results["shuffled"][1]
    # Barajar los contextos conserva el conjunto de pesos de cada fila y su tamaño efectivo.
    assert shuffled["effective_rows"] == pytest.approx(context["effective_rows"], rel=1e-9)
    assert 1 < context["effective_rows"]["min"] < context["effective_rows"]["mean"]
    assert context["effective_rows"]["mean"] < sizes["context"]["max"]
    assert not np.array_equal(results["context"][0], results["shuffled"][0])
    for calibrated, _ in results.values():
        assert np.all(np.diff(calibrated, axis=1) >= 0)
        assert np.array_equal(calibrated[:, 2], case["evaluation"][0].prediction)
    again, _ = rwc.replay(
        case["record"],
        case["calibration"],
        case["evaluation"],
        case["history"],
        weighting="shuffled",
    )
    assert again.tobytes() == results["shuffled"][0].tobytes()
    other, _ = rwc.replay(
        case["record"] | dict(config=case["record"]["config"] | dict(seed=1)),
        case["calibration"],
        case["evaluation"],
        case["history"],
        weighting="shuffled",
    )
    assert not np.array_equal(other, results["shuffled"][0])


def test_the_replay_never_reads_labels_before_they_mature(case):
    panel, matures = case["evaluation"]
    cut = int(panel.prediction_at.min()) + 10 * DAY
    future = matures > cut
    changed = ForecastPanel.from_columns(
        panel.row_id,
        np.asarray(panel.markets)[panel.market],
        panel.prediction_at,
        np.where(future, panel.target * -50.0 + 1.0, panel.target),
        panel.prediction,
        quantiles=panel.quantiles,
        levels=LEVELS,
    )
    labels = case["history"]
    count = len(labels["target"]) - panel.rows
    altered = labels | dict(target=np.r_[labels["target"][:count], changed.target])
    early = panel.prediction_at <= cut
    for weighting in rwc.WEIGHTINGS:
        left, _ = rwc.replay(
            case["record"], case["calibration"], (panel, matures), labels, weighting=weighting
        )
        right, _ = rwc.replay(
            case["record"], case["calibration"], (changed, matures), altered, weighting=weighting
        )
        assert left[early].tobytes() == right[early].tobytes(), weighting
        assert not np.array_equal(left[~early], right[~early]), weighting


def test_the_calibrated_interval_feeds_the_a2_abstention(case):
    calibrated, _ = rwc.replay(
        case["record"],
        case["calibration"],
        case["evaluation"],
        case["history"],
        weighting="context",
    )
    panel = case["evaluation"][0]
    # Desplazar los intervalos hace que una parte de las filas excluya el cero.
    shifted = calibrated + 0.04
    emitted, abstained = na.emit_calibrated(panel, shifted, 0.8)
    claim = (shifted[:, 1] > 0) | (shifted[:, 3] < 0)
    assert 0 < claim.sum() < panel.rows
    assert np.array_equal(abstained, ~claim)
    assert np.array_equal(emitted, np.where(claim, panel.prediction, 0.0))
    with pytest.raises(ValueError, match="forma"):
        na.emit_calibrated(panel, shifted[:, :3], 0.8)


def test_the_stage_rejects_emissions_before_g_matures_and_2024_labels(case):
    record = case["record"]
    with pytest.raises(ValueError, match="maduran después"):
        rwc.replay(
            record, case["calibration"], case["calibration"], case["history"], weighting="uniform"
        )
    labels = case["history"] | dict(
        matures_at=np.r_[case["history"]["matures_at"][:-1], rwc.FINAL_TEST_START_US]
    )
    with pytest.raises(ValueError, match="2024"):
        rwc.replay(record, case["calibration"], case["evaluation"], labels, weighting="uniform")
    with pytest.raises(ValueError, match="errores maduros"):
        rwc.replay(
            record | dict(config=record["config"] | dict(min_rows=100_000)),
            case["calibration"],
            case["evaluation"],
            case["history"],
            weighting="uniform",
        )
    with pytest.raises(ValueError, match="ponderación"):
        rwc.replay(
            record, case["calibration"], case["evaluation"], case["history"], weighting="none"
        )
    panel, matures = case["evaluation"]
    with pytest.raises(ValueError, match="madurar después"):
        rwc.replay(
            record,
            case["calibration"],
            (panel, panel.prediction_at),
            case["history"],
            weighting="uniform",
        )


def test_the_record_survives_json_and_the_stage_is_registered(case):
    record = case["record"]
    restored = json.loads(json.dumps(record))
    left, _ = rwc.replay(
        record, case["calibration"], case["evaluation"], case["history"], weighting="context"
    )
    right, _ = rwc.replay(
        restored, case["calibration"], case["evaluation"], case["history"], weighting="context"
    )
    assert left.tobytes() == right.tobytes()
    assert rwc.stage_sha256(restored) == rwc.stage_sha256(record)
    with pytest.raises(ValueError, match="escala"):
        rwc.checked_record(record | dict(scale=dict(mean=[0.0, 0.0], deviation=[0.0, 1.0])))
    with pytest.raises(ValueError, match="contrato"):
        rwc.checked_record(record | dict(fitted_until=rwc.FINAL_TEST_START_US))
    assert rwc.IDENTITY in OUTPUT_STAGES
    assert checked_output_stages({rwc.IDENTITY: dict(rwc.DEFAULT)}) == {rwc.IDENTITY: rwc.DEFAULT}
    for wrong in (
        rwc.DEFAULT | dict(bandwidth=1),
        rwc.DEFAULT | dict(lookback_days=0),
        rwc.DEFAULT | dict(regime_days=True),
        rwc.DEFAULT | dict(min_rows=0),
        rwc.DEFAULT | dict(seed=-1),
        rwc.DEFAULT | dict(extra=1),
    ):
        with pytest.raises(ValueError):
            rwc.checked_config(wrong)
