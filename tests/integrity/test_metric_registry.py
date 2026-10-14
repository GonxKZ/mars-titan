"""Registro previo de las cifras de la comparación walk-forward, comprobado al publicar (#442).

Reutiliza el estudio de juguete de `tests/evaluation/test_walk_forward_comparison.py`, cuyas
predicciones se escriben con valores conocidos. Cada mutación cambia una cifra publicada o el
registro y comprueba que la comprobación lo detecta. No hay modelos ajustados ni pasos de
optimizador.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.integrity import metric_registry, score_recheck
from mars_titan.models.quantile_head import LEVELS
from tests.evaluation.test_comparison_sources import save
from tests.evaluation.test_walk_forward_comparison import Study
from tests.integrity.test_independent_scores import random_panel

ROOT = metric_registry.ROOT
BOTH = "US+CN"


@pytest.fixture(scope="module")
def registry():
    return metric_registry.load_registry()


def publish(root, weighting):
    study = Study(root / "study", scope=BOTH)
    study.config["metrics"]["market_weighting"] = weighting
    save(study.config_path, study.config)
    output = root / "result"
    walk.write_walk_forward(study.config_path, study.sources_path, BOTH, output)
    report = json.loads((output / "comparison.json").read_text())
    config = walk.resolve_config(study.config_path)
    sources = walk.load_sources(study.sources_path, config, BOTH)
    raw = score_recheck.published_raw(output, report)
    return SimpleNamespace(report=report, config=config, sources=sources, raw=raw)


@pytest.fixture(scope="module", params=["session", "market"])
def published(request, tmp_path_factory):
    return publish(tmp_path_factory.mktemp(f"registry-{request.param}"), request.param)


def seeds(published, arm):
    return list(published.report["arms"][arm]["seeds"])


def view(published, arm, name=BOTH, part="summary"):
    seed = seeds(published, arm)[0]
    entry = published.report["arms"][arm]["seeds"][seed]["overall"][name]
    frame = published.raw[published.raw["arm"] == arm]
    if seed != "deterministic":
        frame = frame[frame["seed"] == int(seed)]
    if name != BOTH:
        frame = frame[frame["market"] == name]
    return copy.deepcopy(entry[part]), frame


def check(registry, published, summary, frame, quantiles):
    return metric_registry.check_summary(
        registry,
        summary,
        frame,
        weighting=published.config["metrics"]["market_weighting"],
        quantiles=quantiles,
        rtol=score_recheck.RTOL,
        atol=score_recheck.ATOL,
    )


def test_every_published_figure_is_registered_and_rebuilt_from_its_sessions(registry, published):
    views, failures = score_recheck.check_summaries(
        walk.scope_config(published.config, BOTH),
        published.sources,
        published.raw,
        published.report,
        registry,
    )
    assert failures == 0, json.dumps(views, indent=1)[:3000]
    # Ridge (1 semilla), gru y titans (2 semillas) y el control cero, con tres vistas cada uno.
    assert len(views) == 6 * 3
    for entry in views:
        assert entry["passed"] and not entry["calibrated_unregistered"]
        # Con cuantiles se reconstruyen niveles, intervalos y la probabilidad de subida.
        quantiles = entry["arm"] in ("gru", "titans")
        assert entry["checked"] >= (180 if quantiles else 95)
        # Solo quedan sin recalcular los compromisos de signo, los periodos y las identidades.
        assert entry["not_recomputed"] == (16 if quantiles else 5)


@pytest.mark.parametrize("weighting", ["session", "market"])
@pytest.mark.parametrize("quantile", [True, False])
def test_irregular_sessions_follow_the_registered_aggregations(registry, quantile, weighting):
    # En el estudio todas las sesiones tienen las mismas filas. Aquí tienen de 1 a 8, con
    # ceros, empates y dos mercados, así que las medias por fila y las proporciones agrupadas
    # dejan de coincidir con las medias por sesión.
    parts = random_panel(np.random.default_rng(20261013), quantile=quantile)
    columns = ("sample_id", "market", "prediction_at", "target", "prediction")
    joined = {key: np.concatenate([np.asarray(part[key]) for part in parts]) for key in columns}
    panel = ForecastPanel.from_columns(
        *(joined[key] for key in columns),
        quantiles=np.vstack([part["quantiles"] for part in parts]) if quantile else None,
        levels=LEVELS if quantile else None,
        markets=("US", "CN"),
    )
    scores = score_sessions(panel)
    summary, table = scores.summary(market_weighting=weighting), scores.to_table().to_pandas()
    result = metric_registry.check_summary(
        registry,
        summary,
        table,
        weighting=weighting,
        quantiles=quantile,
        rtol=score_recheck.RTOL,
        atol=score_recheck.ATOL,
    )
    assert result["passed"], result
    calls, eligible = table["direction_calls"], table["direction_eligible"]
    session_coverage = (calls / eligible.clip(lower=1))[eligible > 0].mean()
    assert summary["direction"]["call_coverage"] != pytest.approx(session_coverage)
    assert summary["row_weighted"]["mae"] != pytest.approx(table["mae"].mean())


def test_a_view_without_defined_sessions_publishes_null_not_zero(registry, published):
    # El control cero nunca llama, así que la exactitud condicionada no está definida.
    summary, frame = view(published, "zero")
    assert summary["point"]["conditional_direction_accuracy"] is None
    assert summary["point"]["up_precision"] is None
    assert check(registry, published, summary, frame, False)["passed"]
    summary["point"]["conditional_direction_accuracy"] = 0.0
    result = check(registry, published, summary, frame, False)
    assert [entry["metric"] for entry in result["mismatched"]] == [
        "point.conditional_direction_accuracy"
    ]


def bump(value):
    return value + 1 if isinstance(value, int) else value * 1.01 + 0.001


@pytest.mark.parametrize(
    "path, label",
    [
        (("point", "rmse"), "point.rmse"),
        (("point", "direction_accuracy_percent"), "point.direction_accuracy_percent"),
        (("point", "up_recall"), "point.up_recall"),
        (("by_market", "CN", "mae"), "by_market.*.mae ['CN']"),
        (("row_weighted", "mae"), "row_weighted.mae"),
        (("direction", "call_coverage"), "direction.call_coverage"),
        (("direction", "zero_target_rows"), "direction.zero_target_rows"),
        (("rank_ic", "sessions", "defined"), "rank_ic.sessions.defined"),
        (
            ("selective_direction", "curve", 4, "accuracy"),
            "selective_direction.curve[].accuracy ['0.5']",
        ),
        (
            ("selective_direction", "curve", 9, "retained_rows"),
            "selective_direction.curve[].retained_rows ['1']",
        ),
        (("quantiles", "pinball", "0.9"), "quantiles.pinball.* ['0.9']"),
        (("quantiles", "mean_pinball"), "quantiles.mean_pinball"),
        (("quantiles", "level_frequency_gap", "0.5"), "quantiles.level_frequency_gap.* ['0.5']"),
        (
            ("quantiles", "intervals", 0, "coverage_gap"),
            "quantiles.intervals[].coverage_gap ['0.8']",
        ),
        (
            ("quantiles", "intervals", 1, "lower_level"),
            "quantiles.intervals[].lower_level ['0.95']",
        ),
        (
            ("quantiles", "intervals", 1, "interval_score"),
            "quantiles.intervals[].interval_score ['0.95']",
        ),
        (
            ("quantiles", "calibration", "level_max_absolute_error"),
            "quantiles.calibration.level_max_absolute_error",
        ),
        (("quantiles", "sign_probability", "brier"), "quantiles.sign_probability.brier"),
        (("quantiles", "sign_probability", "ece"), "quantiles.sign_probability.ece"),
        (
            ("quantiles", "sign_probability", "reliability", 3, "rows"),
            "quantiles.sign_probability.reliability[].rows ['3']",
        ),
    ],
)
def test_a_changed_published_figure_is_detected(registry, published, path, label):
    summary, frame = view(published, "titans")
    *parents, last = path
    node = summary
    for key in parents:
        node = node[key]
    assert node[last] is not None
    node[last] = bump(node[last])
    result = check(registry, published, summary, frame, True)
    assert not result["passed"]
    assert [entry["metric"] for entry in result["mismatched"]] == [label]


@pytest.mark.parametrize(
    "change, label",
    [
        (lambda s: s.update(market_weighting="other"), "market_weighting"),
        (lambda s: s["direction"].update(convention="other"), "direction.convention"),
        (lambda s: s["quantiles"].update(levels=s["quantiles"]["levels"][1:]), "quantiles.levels"),
        (
            lambda s: s["quantiles"]["calibration"].update(undercovered_intervals=[0.5]),
            "quantiles.calibration.undercovered_intervals",
        ),
    ],
)
def test_a_changed_convention_or_list_is_detected(registry, published, change, label):
    summary, frame = view(published, "titans")
    change(summary)
    result = check(registry, published, summary, frame, True)
    assert [entry["metric"] for entry in result["mismatched"]] == [label]


def test_unregistered_and_missing_figures_fail(registry, published):
    summary, frame = view(published, "titans", "US")
    summary["point"]["hit_rate"] = 0.6
    summary["quantiles"]["pinball"]["0.3"] = 0.01
    del summary["point"]["rmse"]
    result = check(registry, published, summary, frame, True)
    assert result["unregistered"] == ["point.hit_rate", "quantiles.pinball.* ['0.3']"]
    assert result["missing"] == ["point.rmse"]
    assert result["failures"] == 3 and not result["mismatched"]
    # Un modelo declarado con cuantiles cuyas sesiones no los guardan se rechaza.
    summary, frame = view(published, "ridge")
    with pytest.raises(ValueError, match="no guardan sus cuantiles"):
        check(registry, published, summary, frame, True)


def test_a_calibrated_figure_outside_the_registry_fails_the_recheck(registry, published):
    report = copy.deepcopy(published.report)
    seed = seeds(published, "gru")[0]
    calibrated = report["arms"]["gru"]["seeds"][seed]["overall"]["CN"]["calibrated"]
    assert metric_registry.registered(registry, calibrated) == []
    calibrated["quantiles"]["sharpness"] = 0.2
    calibrated["by_market"]["CN"]["hit_rate"] = 0.6
    views, failures = score_recheck.check_summaries(
        walk.scope_config(published.config, BOTH),
        published.sources,
        published.raw,
        report,
        registry,
    )
    assert failures == 2
    (wrong,) = [entry for entry in views if not entry["passed"]]
    assert (wrong["arm"], wrong["view"]) == ("gru", "CN")
    assert wrong["calibrated_unregistered"] == [
        "by_market.*.hit_rate ['CN']",
        "quantiles.sharpness",
    ]


def mutate(path, change):
    document = json.loads(metric_registry.DEFAULT.read_text())
    change(document)
    path.write_text(json.dumps(document))
    return path


def rule(name, **values):
    return lambda d: d["metrics"][name]["rule"].update(values)


def field(name, **values):
    return lambda d: d["metrics"][name].update(values)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.update(final_test_opened=True), "abre la reserva"),
        (lambda d: d.update(status="draft"), "abre la reserva"),
        (lambda d: d["aggregation"]["weightings"].reverse(), "las ponderaciones"),
        (lambda d: d["aggregation"].update(seeds="best_seed"), "la agregación de semillas"),
        (lambda d: d["session"].update(ineligible_rows="dropped"), "las filas excluidas"),
        (lambda d: d["metrics"].pop("rank_ic.sessions.constant_target"), "los estados"),
        (lambda d: d["selective_coverages"].pop(), "las coberturas"),
        (lambda d: d.update(sign_bins=20), "los intervalos de probabilidad"),
        (rule("direction.convention", value="zero_is_hit"), "la convención de dirección"),
        (rule("selective_direction.rule", value="random"), "la regla de la abstención"),
        (rule("quantiles.sign_probability.rule", value="normal"), "la probabilidad de subida"),
        (field("point.mae", unit="bps"), "unidad y sentido"),
        (field("point.mae", better="best"), "unidad y sentido"),
        (field("point.mae", definition=""), "unidad y sentido"),
        (rule("point.mae", kind="median"), "no es conocida"),
        (lambda d: d["metrics"]["point.mae"]["rule"].pop("column"), "parámetros"),
        (rule("point.mae", weight="samples"), "parámetros"),
        (rule("quantiles.intervals[].lower_level", side="middle"), "parámetros"),
        (field("point.mae", doc="Sección inventada"), "documentación"),
        (rule("point.rmse", of="point.mse2"), "sin registrar"),
        (field("point.mae", better="higher"), "sentido de la serie mae"),
        (lambda d: d["series"].pop("mse"), "La serie mse"),
        (lambda d: d["series"].update(mae="point.mae2"), "apunta a una cifra"),
    ],
)
def test_a_registry_that_differs_from_the_code_is_rejected(tmp_path, change, message):
    path = mutate(tmp_path / "registry.json", change)
    with pytest.raises(ValueError, match=message):
        metric_registry.load_registry(path, root=ROOT)


def test_the_declared_registry_covers_every_series_and_documents_each_figure(registry):
    assert registry["sha256"] and registry["status"] == walk.DECLARED
    for name in walk.SERIES_METRICS:
        assert any(name.startswith(prefix) for prefix in registry["series"])
    # Cada cifra con regla de agregación cita su sección, salvo recuentos e identidades.
    for name, entry in registry["metrics"].items():
        if entry["better"] != "none":
            assert "doc" in entry, name
