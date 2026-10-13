"""Regla kalman de B6 en la ventana walk-forward (PT3, #455), sin ajustar parámetros.

La estimación de σ² y ϱ y la corrección solo hacen inferencia, así que estas pruebas usan un
Titans-MAC `mac_online` con su inicialización aleatoria y no rebajan la protección del
aprendizaje. La ventana completa parte de un padre ajustado y está en
`test_mars_titan_correction.py`, con su entrada en el inventario de #546.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory.cohort_noise import METHOD, estimate_cohort_noise
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.memory.mars_titan_variant import check_components, load_declaration
from mars_titan.training import mars_titan_correction as mc
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import ChronologicalInference, load_recipe
from mars_titan.training.walk_forward_phases import micros
from tests.training.test_titans_walk_forward import recipe, views

DECLARED = Path("configs/titans/kalman-correction-historical-masked.json")
PROXIMAL_RECIPE = Path("configs/titans/mature-correction-historical-masked.json")
KALMAN = {
    "associative_memory": {"rule": "kalman", "key": "codec", "cohort_correlation": "estimated"}
}
RHO0 = {"associative_memory": {"rule": "kalman", "key": "codec", "cohort_correlation": "zero"}}
PROXIMAL = {"associative_memory": {"rule": "proximal", "key": "codec"}}
NOISE = dict(observation_noise=4.0, cohort_correlation=0.05)
CAP = 0.95


def kalman_values(case="kappa1e-5"):
    return mc.case_values(mc.load_correction_recipe(DECLARED), case)


# Receta y combinación, sin datos


def test_declared_recipe_searches_two_process_ratios_and_fixes_the_rest():
    document = mc.load_correction_recipe(DECLARED)
    assert document["recipe"] == dict(prior_ratio=0.1, correlation_cap=CAP, estimation_months=3)
    assert [kalman_values(name)["process_ratio"] for name in ("kappa1e-6", "kappa1e-5")] == [
        1e-6,
        1e-5,
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda d: d["walk_forward"]["search_cases"]["kappa1e-6"].update(process_ratio=-1e-6),
            "κ ≥ 0",
        ),
        (lambda d: d["recipe"].update(prior_ratio=0.0), "p0/σ² > 0"),
        (lambda d: d["recipe"].update(correlation_cap=1.0), "tope de ϱ"),
        (lambda d: d["recipe"].update(estimation_months=0), "meses de estimación"),
        (lambda d: d["recipe"].update(estimation_months=13), "meses de estimación"),
        (lambda d: d["recipe"].update(estimation_months=3.0), "meses de estimación"),
        (lambda d: d["recipe"].update(correlation_cap=1), "tope de ϱ"),
        (lambda d: d["recipe"].pop("estimation_months"), "valores fijos de kalman"),
        (lambda d: d["recipe"].update(rate=0.25), "valores fijos de kalman"),
        (
            lambda d: d["walk_forward"].update(
                search_cases={
                    "a": {"process_ratio": 1e-6, "prior_ratio": 0.1},
                    "b": {"process_ratio": 1e-5, "prior_ratio": 0.1},
                }
            ),
            "valores fijos de kalman",
        ),
        (lambda d: d.update(recipe_name="mars_titan_mature_correction_v1"), "η y λ"),
    ],
)
def test_kalman_recipe_rejects_impossible_or_misplaced_values(tmp_path, change, message):
    document = json.loads(DECLARED.read_text())
    change(document)
    path = tmp_path / "kalman.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        mc.load_correction_recipe(path)


def test_kalman_arm_scales_prior_and_process_noise_with_the_estimated_variance():
    estimated = mc.arm_components(KALMAN, kalman_values(), NOISE)
    assert estimated == {
        "associative_memory": dict(
            rule="kalman",
            key="codec",
            process_noise=1e-5 * 4.0,
            observation_noise=4.0,
            cohort_correlation=0.05,
            prior_variance=0.1 * 4.0,
            huber_threshold=None,
        )
    }
    # La alternativa trivial comparte σ², p0 y q y solo anula ϱ.
    zero = mc.arm_components(RHO0, kalman_values(), NOISE)
    assert zero["associative_memory"] == dict(
        estimated["associative_memory"], cohort_correlation=0.0
    )
    for combination in (estimated, zero):
        checked, correction = check_components(load_declaration(), combination)
        assert set(checked) == {"associative_memory"} and correction.memory.rule == "kalman"
    assert mc.kalman_arm(KALMAN) and mc.kalman_arm(RHO0) and not mc.kalman_arm(PROXIMAL)


@pytest.mark.parametrize(
    ("components", "values", "noise", "message"),
    [
        (KALMAN, "kalman", None, "σ² y la ϱ de la ventana"),
        (KALMAN, "proximal", NOISE, "receta de sus varianzas"),
        (PROXIMAL, "kalman", None, "receta de η y λ"),
        (PROXIMAL, "proximal", NOISE, "sin varianzas estimadas"),
        (
            {"associative_memory": {"rule": "kalman", "key": "codec"}},
            "kalman",
            NOISE,
            "también su correlación de cohorte",
        ),
        (
            {"associative_memory": dict(KALMAN["associative_memory"], cohort_correlation="half")},
            "kalman",
            NOISE,
            "estimada o nula",
        ),
        (
            {"associative_memory": dict(PROXIMAL["associative_memory"], cohort_correlation="zero")},
            "proximal",
            None,
            "regla y su clave",
        ),
        (KALMAN, "kalman", dict(observation_noise=4.0), "σ² y la ϱ de la ventana"),
    ],
)
def test_kalman_and_proximal_arms_only_take_their_own_values(components, values, noise, message):
    declared = dict(
        kalman=kalman_values(),
        proximal=mc.case_values(mc.load_correction_recipe(PROXIMAL_RECIPE), "eta25e-2"),
    )
    with pytest.raises(ValueError, match=message):
        mc.arm_components(components, declared[values], noise)


def test_proximal_arm_keeps_its_previous_combination():
    values = mc.case_values(mc.load_correction_recipe(PROXIMAL_RECIPE), "eta25e-2")
    assert mc.arm_components(PROXIMAL, values) == {
        "associative_memory": dict(rule="proximal", key="codec", rate=0.25, forgetting=0.01)
    }


def test_estimation_tramo_closes_the_fit_and_keeps_its_warmup_inside_the_window():
    fold = dict(train=["2000-01-01", "2023-09-01"])
    assert mc.estimation_phase(fold, 3, 1) == FinancialPhase(
        "train",
        micros("2023-05-01"),
        micros("2023-06-01"),
        micros("2023-09-01"),
        micros("2023-09-01"),
    )
    # El calentamiento no empieza antes del origen de la ventana.
    short = dict(train=["2023-07-15", "2023-09-01"])
    assert mc.estimation_phase(short, 1, 3).warmup_start == micros("2023-07-15")
    with pytest.raises(ValueError, match="no cabe"):
        mc.estimation_phase(short, 2, 0)
    for months in (0, 13, 3.0):
        with pytest.raises(ValueError, match="meses de estimación"):
            mc.estimation_phase(fold, months, 1)


# Estimación y corrección sobre la vista técnica, con un padre sin ajustar


@pytest.fixture(scope="module")
def tail(tmp_path_factory):
    root = tmp_path_factory.mktemp("kalman-correction")
    view, _ = views(root / "base")
    chronological, document = load_recipe(recipe(root))
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    _, _, _, fold = wf._protocol(wf.view_protocol(view), "fold-000", 42)
    months = wf.walk_forward_options(document)["warmup_months"]
    phases = dict(
        estimation=mc.estimation_phase(fold, 3, months),
        validation=wf.window_phases(fold, months)["validation"],
    )
    sources = wf._sources(dataset, phases, root / "indices")
    specification = sources["estimation"].specification()
    with wf.unfused_attention():
        predictor, _ = wf._predictor(document, specification, "mac_online", 42, "cpu")
    predictor = predictor.eval().requires_grad_(False)
    return dict(
        root=root,
        dataset=dataset,
        fold=fold,
        sources=sources,
        specification=specification,
        predictor=predictor,
        chronological=wf.case_recipe(document, None),
    )


def core_rows(tail, partition):
    """Filas de la inferencia del núcleo sin corrección: decisión, flujo, emitida y etiqueta."""
    rows = []
    with wf.unfused_attention():
        metrics = ChronologicalInference(tail["predictor"], tail["chronological"]).evaluate(
            tail["sources"][partition], rows=rows
        )
    return rows, metrics


def test_window_noise_uses_label_minus_core_residuals_of_each_market_session(tail):
    with wf.unfused_attention():
        noise = mc.window_cohort_noise(
            tail["predictor"], tail["chronological"], tail["sources"]["estimation"], CAP
        )
    rows, metrics = core_rows(tail, "estimation")
    values = np.array([value - issued for _, _, issued, value, _ in rows])
    markets = np.array([flow.split("/", 1)[0] for flow, *_ in rows])
    sessions = np.array([decision for _, decision, *_ in rows], dtype=np.int64)
    assert noise == dict(
        estimate_cohort_noise(values, markets, sessions, correlation_cap=CAP),
        core_session_mae=metrics["session_mae"],
    )
    assert noise["method"] == METHOD and noise["markets"] == ["US"]
    assert noise["rows"] == len(rows) and noise["sessions"] == len(set(sessions.tolist())) > 1
    # Solo decisiones del tramo, cuyas etiquetas maduran antes del final del ajuste.
    phase = tail["sources"]["estimation"].phase
    assert (
        phase.close_at == micros(tail["fold"]["train"][1]) <= micros(tail["fold"]["validation"][0])
    )
    assert all(phase.decision_start <= moment < phase.decision_end for moment in sessions.tolist())


def test_residual_rows_keep_the_market_of_each_flow():
    # La vista de las pruebas solo tiene EE. UU. Con dos mercados en la misma sesión, cada fila
    # debe conservar el suyo para que el estimador no mezcle sus cohortes.
    rows = mc._Residuals()
    for flow, decision, issued, value in (
        ("US/AAA", 1, 0.5, 1.0),
        ("US/BBB", 1, 0.0, 0.25),
        ("CN/600000", 1, 0.25, -0.5),
        ("CN/000001", 1, 0.0, -0.25),
        ("US/AAA", 2, 0.0, 0.75),
        ("US/BBB", 2, 0.5, 0.5),
        ("CN/600000", 2, 1.0, 1.5),
        ("CN/000001", 2, 0.0, 0.25),
    ):
        rows.append((flow, decision, issued, value, None))
    assert rows.values == [0.5, 0.25, -0.75, -0.25, 0.75, 0.0, 0.5, 0.25]
    assert rows.markets == ["US", "US", "CN", "CN", "US", "US", "CN", "CN"]
    assert rows.sessions == [1, 1, 1, 1, 2, 2, 2, 2]
    estimate = estimate_cohort_noise(
        rows.values, rows.markets, np.asarray(rows.sessions, dtype=np.int64), correlation_cap=CAP
    )
    # Cuatro sesiones, una por mercado e instante. Sin el mercado serían dos.
    assert estimate["markets"] == ["CN", "US"] and estimate["sessions"] == 4


def test_window_records_its_estimation_tramo_and_a_resumed_window_reuses_it(tail, monkeypatch):
    calls, original = [], mc.window_cohort_noise

    def counted(predictor, chronological, source, cap):
        calls.append(source.phase)
        return original(predictor, chronological, source, cap)

    monkeypatch.setattr(mc, "window_cohort_noise", counted)
    core = (tail["predictor"], tail["chronological"])
    root = tail["root"] / "indices"

    def noise(stored=None):
        with wf.unfused_attention():
            return mc.window_noise(
                tail["fold"], kalman_values(), 1, tail["dataset"], root, core, stored
            )

    first = noise()
    phase = mc.estimation_phase(tail["fold"], 3, 1)
    assert calls == [phase] and first["phase"] == asdict(phase)
    assert first["index"] == tail["sources"]["estimation"].identity
    # La reanudación lee la identidad de run.json, así que se compara tras pasar por JSON.
    stored = json.loads(json.dumps(first))
    assert noise(stored) is stored and len(calls) == 1
    # Con otro índice, otro tramo o un registro incompleto vuelve a estimar con el mismo
    # resultado, que ya no coincide con lo guardado.
    for other in (
        dict(stored, index="otra"),
        dict(stored, phase=dict(stored["phase"], decision_start=0)),
        {key: value for key, value in stored.items() if key != "estimate"},
        dict(stored, extra=1),
        None,
    ):
        assert noise(other) == first
    assert len(calls) == 6


def test_window_noise_refuses_a_measured_tramo(tail):
    with pytest.raises(ValueError, match="tramo de ajuste"):
        mc.window_cohort_noise(
            tail["predictor"], tail["chronological"], tail["sources"]["validation"], CAP
        )


def corrected(tail, noise, components=KALMAN):
    """Predicciones de validación con la regla kalman y las varianzas dadas."""
    _, correction = check_components(
        load_declaration(), mc.arm_components(components, kalman_values(), noise)
    )
    with wf.unfused_attention():
        inference = mc.CorrectionInference(
            tail["predictor"],
            tail["chronological"],
            correction,
            FrozenEpisodeCodec(tail["specification"]),
        )
        rows = wf.PredictionRows(inference.quantiles)
        inference.predict(tail["sources"]["validation"], rows)
    return pa.concat_tables(rows.finish()).sort_by("sample_id"), inference.memory


def core_table(tail):
    """Predicciones de validación del núcleo sin corrección."""
    with wf.unfused_attention():
        inference = ChronologicalInference(tail["predictor"], tail["chronological"])
        rows = wf.PredictionRows(inference.quantiles)
        inference.predict(tail["sources"]["validation"], rows)
    return pa.concat_tables(rows.finish()).sort_by("sample_id")


def test_the_mean_correction_does_not_depend_on_the_estimated_variance(tail):
    one, small = corrected(tail, dict(observation_noise=1.0, cohort_correlation=0.3))
    seven, large = corrected(tail, dict(observation_noise=7.0, cohort_correlation=0.3))
    assert small.writes == large.writes > 0
    # q y p0 escalan con σ², así que P escala con σ² y la ganancia P⁺b no cambia.
    torch.testing.assert_close(large.covariance, 7.0 * small.covariance, rtol=1e-9, atol=0.0)
    torch.testing.assert_close(large.matrix, small.matrix, rtol=1e-9, atol=1e-15)
    for left, right in zip(
        one["prediction"].to_pylist(), seven["prediction"].to_pylist(), strict=True
    ):
        assert left == pytest.approx(right, rel=1e-9, abs=1e-15)
    # Las primeras decisiones se emiten con la media previa, A = 0, y las siguientes corrigen
    # el núcleo.
    first = min(one["prediction_at"].to_pylist())
    pairs = list(zip(one.to_pylist(), core_table(tail).to_pylist(), strict=True))
    assert all(
        mine["prediction"] == other["prediction"]
        for mine, other in pairs
        if mine["prediction_at"] == first
    )
    assert len({mine["prediction"] - other["prediction"] for mine, other in pairs}) > 1


def test_cohort_correlation_changes_the_correction_and_zero_is_its_alternative(tail):
    noise = dict(observation_noise=1.0, cohort_correlation=0.3)
    estimated, correlated = corrected(tail, noise)
    zero, independent = corrected(tail, noise, RHO0)
    explicit, _ = corrected(tail, dict(noise, cohort_correlation=0.0))
    assert zero.equals(explicit)
    assert not estimated["prediction"].equals(zero["prediction"])
    assert not torch.equal(correlated.covariance, independent.covariance)
