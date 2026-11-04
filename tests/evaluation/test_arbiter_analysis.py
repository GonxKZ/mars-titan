"""Etapa del análisis de los árbitros en línea de #789 con paneles sintéticos, sin PyTorch.

Las predicciones se generan en la prueba. La referencia B tiene señal en su época común y
una versión, la de la época 12, con mucha más. Las demás versiones y el resto de modelos solo
tienen ruido. No se construye ningún modelo ni se ejecutan pasos de optimizador.
"""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from test_variant_screening_analysis import ASSETS, SyntheticOrigin, make_panel, protocol_of

from mars_titan.evaluation import arbiter_analysis as stage
from mars_titan.evaluation import online_arbiters as arbiters
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation import variant_screening_analysis as analysis
from mars_titan.evaluation.forecast_panel import DAY_MICROSECONDS, ForecastPanel
from mars_titan.training import variant_screening

ROOT = Path(__file__).resolve().parents[2]
REAL_DECLARATION = json.loads(
    (ROOT / "configs/experiments/online-arbiters-v1.json").read_text(encoding="utf-8")
)
# F1 y F4 comparten el trabajo de su control de error absoluto.
ENTRIES = ("A1", "B1", "F1", "F4")
# El catálogo reducido no tiene F2 ni F3. La comparación con las medias de pesos se comprueba
# con un modelo de B1, que elige caso en validación, y con el control compartido de F1.
AVERAGING = ["B1/area_memory", "F1/absolute_error_control"]
DECLARATION = copy.deepcopy(REAL_DECLARATION)
DECLARATION["stages"]["version_arbiter_v1"]["analysis"]["time_averaging"]["compared_with"] = (
    AVERAGING
)
REFERENCE = "references/quantile_reference"
COMMON = 6
STRONG = 12


class VersionedOrigin(SyntheticOrigin):
    """Origen sintético que también emite las versiones de sus ajustes."""

    def __init__(self, year, jobs, epochs, *, shift=0.0, drop=None, extra=None):
        super().__init__(year, jobs)
        self.epochs, self.shift, self.drop, self.extra = epochs, shift, drop, extra
        self.version_calls = []

    def version(self, job, epoch, name):
        digest = hashlib.sha256(f"{job['id']}/{epoch}/{name}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        target, noise = self.target[name], rng.standard_normal(len(self.target[name]))
        if REFERENCE in job["used_by"] and epoch == STRONG:
            return 0.9 * target + 0.3 * noise
        return noise

    def versions(self, job_id):
        self.version_calls.append(job_id)
        job, result = self.jobs[job_id], {}
        for epoch in (*self.epochs, *([self.extra] if self.extra else [])):
            if epoch in (COMMON, self.drop):
                continue
            result[epoch] = {
                name: make_panel(
                    self.times[name],
                    self.target[name] + self.shift,
                    self.version(job, epoch, name),
                )
                for name in ("calibration", "evaluation")
            }
        return result

    def origin(self):
        return replace(super().origin(), versions=self.versions)


def screening(entries=ENTRIES, **options):
    protocol = protocol_of(entries)
    # Dos semillas bastan para comprobar la media de semillas y acortan la prueba.
    protocol["seeds"] = protocol["seeds"][:2]
    jobs = variant_screening.screening_jobs(protocol)
    epochs = protocol["selection"]["versions"]["epochs"]
    sources = {
        o["year"]: VersionedOrigin(o["year"], jobs, epochs, **options) for o in protocol["origins"]
    }
    return protocol, jobs, sources


@pytest.fixture(scope="module")
def analyzed():
    protocol, jobs, sources = screening()
    origins = {year: source.origin() for year, source in sources.items()}
    report, arrays = stage.analyze(protocol, DECLARATION, jobs, origins)
    return protocol, jobs, sources, report, arrays


def two_markets(start, sessions, *, seed, declared=("US", "CN")):
    """Panel con dos mercados de cinco activos y `sessions` días desde `start`.

    `declared` son los mercados que declara el panel, que pueden ser más que los que tienen filas.
    """
    rng = np.random.default_rng(seed)
    days = start + DAY_MICROSECONDS * np.arange(sessions)
    times = np.repeat(days, 10)
    markets = np.tile(np.repeat(np.array(["US", "CN"]), 5), sessions)
    assets = np.tile(np.arange(10), sessions)
    ids = [
        f"{market}/A{asset}/{at}" for market, asset, at in zip(markets, assets, times, strict=True)
    ]
    target = rng.standard_normal(len(times))
    # Las filas entran desordenadas, así que el panel las ordena.
    order = rng.permutation(len(times))
    return ForecastPanel.from_columns(
        pa.array(ids).take(pa.array(order)),
        markets[order],
        times[order],
        target[order],
        rng.standard_normal(len(times))[order],
        markets=declared,
    )


def test_the_joint_panel_keeps_every_row_and_says_where_it_comes_from():
    calibration = two_markets(1_600_000_000_000_000, 6, seed=1)
    evaluation = two_markets(1_600_000_000_000_000 + 6 * DAY_MICROSECONDS, 4, seed=2)
    panel, positions = stage.joined(calibration, evaluation)
    assert panel.rows == calibration.rows + evaluation.rows == 100
    assert sorted(positions.tolist()) == list(range(100))
    ids = pa.concat_arrays([calibration.row_id, evaluation.row_id])
    assert ids.take(pa.array(positions)).equals(panel.row_id)
    for name in ("target", "prediction_at", "market"):
        joined = np.concatenate([getattr(calibration, name), getattr(evaluation, name)])
        assert np.array_equal(joined[positions], getattr(panel, name))
    # Cada mercado recorre su calibración y después su S.
    assert np.all(np.diff(panel.session_market) >= 0)
    for code in (0, 1):
        assert np.all(np.diff(panel.session_time[panel.session_market == code]) > 0)
    with pytest.raises(ValueError, match="anterior"):
        stage.joined(evaluation, calibration)
    start = 1_600_000_000_000_000 + 6 * DAY_MICROSECONDS
    other = two_markets(start, 4, seed=2, declared=("US", "CN", "HK"))
    with pytest.raises(ValueError, match="mismos mercados"):
        stage.joined(calibration, other)


def market_options(count, *, seed):
    """Opciones de dos mercados con las mismas filas. Solo la primera sigue al objetivo."""
    start = 1_600_000_000_000_000
    calibration = two_markets(start, 30, seed=seed)
    evaluation = two_markets(start + 30 * DAY_MICROSECONDS, 20, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    options = []
    for k in range(count):
        weight = 1.0 if k == 0 else 0.0
        own = [
            replace(part, prediction=weight * part.target + rng.standard_normal(part.rows))
            for part in (calibration, evaluation)
        ]
        options.append(analysis.Emission(own[1], own[1].prediction, own[0]))
    maturity = {
        name: part.prediction_at + 3 * DAY_MICROSECONDS
        for name, part in (("calibration", calibration), ("evaluation", evaluation))
    }
    return options, maturity


def test_each_market_crosses_from_its_calibration_into_its_s():
    options, maturity = market_options(3, seed=11)
    options_of = arbiters.stage_options(DECLARATION, arbiters.VERSION_ARBITER)
    calibration, evaluation = options[0].calibration, options[0].panel
    panel, positions = stage.joined(calibration, evaluation)
    # Con dos mercados el panel conjunto no sigue el orden de la calibración seguida de S.
    assert not np.array_equal(positions, np.arange(panel.rows))
    columns = np.column_stack(
        [np.concatenate([o.calibration.prediction, o.prediction])[positions] for o in options]
    )
    matures = np.concatenate([maturity["calibration"], maturity["evaluation"]])[positions]
    joint = zip(panel.session_market.tolist(), panel.session_time.tolist(), strict=True)
    joint = {key: position for position, key in enumerate(joint)}
    own = zip(evaluation.session_market.tolist(), evaluation.session_time.tolist(), strict=True)
    sessions_of_s = [joint[key] for key in own]
    outputs = {}
    for mode, seed in (("online", None), ("shuffled", options_of["seed"])):
        direct = arbiters.online_mixture(
            panel, columns, matures, window=21, min_assets=3, margin=0.0, seed=seed
        )
        source = np.empty(panel.rows)
        source[positions] = direct["prediction"]
        output, abstained = stage.mixed(options, maturity, options_of, mode)
        np.testing.assert_array_equal(output.prediction, source[calibration.rows :])
        np.testing.assert_array_equal(output.calibration.prediction, source[: calibration.rows])
        assert np.array_equal(abstained, direct["abstained"][sessions_of_s])
        # Las primeras sesiones de cada mercado se abstienen y las de S ya no.
        assert direct["abstained"].any() and not abstained.any()
        outputs[mode] = output.prediction
    assert not np.array_equal(outputs["online"], outputs["shuffled"])
    uniform, abstained = stage.mixed(options, maturity, options_of, "uniform")
    expected = arbiters.uniform_mixture(
        evaluation, np.column_stack([o.prediction for o in options])
    )
    np.testing.assert_array_equal(uniform.prediction, expected)
    assert not abstained.any()
    foreign, _ = market_options(2, seed=12)
    with pytest.raises(ValueError, match="mismas filas"):
        stage.mixed([options[0], foreign[1]], maturity, options_of, "online")


def emissions(source, job):
    """Emisiones de la época común y de las versiones de un trabajo sintético."""
    panels = source.load(job["id"])
    result = [
        analysis.Emission(
            panels["evaluation"], panels["evaluation"].prediction, panels["calibration"]
        )
    ]
    for item in source.versions(job["id"]).values():
        result.append(
            analysis.Emission(
                item["evaluation"], item["evaluation"].prediction, item["calibration"]
            )
        )
    return result


@pytest.fixture(scope="module")
def reference_options():
    protocol, jobs, sources = screening()
    source = sources[2019]
    job = next(job for job in jobs if job["origin"] == 2019 and REFERENCE in job["used_by"])
    options = emissions(source, job)
    return source, options


def test_the_uniform_mixture_is_the_one_of_each_partition(reference_options):
    source, options = reference_options
    output, abstained = stage.mixed(options, source.origin().maturity, None, "uniform")
    assert not abstained.any() and abstained.shape == (options[0].panel.sessions,)
    evaluation = np.column_stack([option.prediction for option in options])
    calibration = np.column_stack([option.calibration.prediction for option in options])
    expected = arbiters.uniform_mixture(options[0].panel, evaluation)
    np.testing.assert_array_equal(output.prediction, expected)
    expected = arbiters.uniform_mixture(options[0].calibration, calibration)
    np.testing.assert_array_equal(output.calibration.prediction, expected)


def test_the_online_mixture_crosses_from_the_calibration_into_s(reference_options):
    source, options = reference_options
    maturity = source.origin().maturity
    options_of = arbiters.stage_options(DECLARATION, arbiters.VERSION_ARBITER)
    output, abstained = stage.mixed(options, maturity, options_of, "online")
    calibration, evaluation = options[0].calibration, options[0].panel
    panel, positions = stage.joined(calibration, evaluation)
    columns = np.column_stack(
        [np.concatenate([o.calibration.prediction, o.prediction])[positions] for o in options]
    )
    matures = np.concatenate([maturity["calibration"], maturity["evaluation"]])[positions]
    direct = arbiters.online_mixture(panel, columns, matures, window=21, min_assets=3, margin=0.0)
    source_order = np.empty(panel.rows)
    source_order[positions] = direct["prediction"]
    np.testing.assert_array_equal(output.prediction, source_order[calibration.rows :])
    np.testing.assert_array_equal(output.calibration.prediction, source_order[: calibration.rows])
    # Las sesiones de S son las últimas del panel conjunto, que solo tiene el mercado US.
    assert np.array_equal(abstained, direct["abstained"][-evaluation.sessions :])
    # El estado llega a S con los resultados de la calibración. Sin ella, S empieza de cero
    # y se abstiene mientras no madure ninguna sesión.
    alone = arbiters.online_mixture(
        evaluation,
        np.column_stack([o.prediction for o in options]),
        maturity["evaluation"],
        window=21,
        min_assets=3,
        margin=0.0,
    )
    assert alone["abstained"][0] and not abstained[0]
    # La versión con más señal, la última, acaba con la mayor parte del peso.
    assert direct["mixtures"][-1].argmax() == len(options) - 1
    assert direct["mixtures"][-1].max() > 0.5


def test_the_calibration_output_fixes_the_lambda_of_s(reference_options):
    source, options = reference_options
    options_of = arbiters.stage_options(DECLARATION, arbiters.VERSION_ARBITER)
    output, _ = stage.mixed(options, source.origin().maturity, options_of, "online")
    factors = sd.market_shrinkage(output.calibration, output.calibration.prediction)
    expected = sd.session_mae(
        output.panel, sd.apply_shrinkage(output.panel, output.prediction, factors)
    )
    values, defined = analysis.session_metric(output, "session_mae_calibrated", min_assets=3)
    np.testing.assert_array_equal(values, expected)
    assert defined.all()


def test_an_abstained_session_that_scores_counts_as_ic_zero():
    times = np.repeat(1_600_000_000_000_000 + DAY_MICROSECONDS * np.arange(4), 16)
    rng = np.random.default_rng(3)
    target = rng.standard_normal(64)
    target[48:] = 1.0
    prediction = rng.standard_normal(64)
    prediction[:16] = 0.0
    prediction[48:] = 0.0
    panel = make_panel(times, target, prediction)
    emission = analysis.Emission(panel, prediction, None)
    abstained = np.array([True, False, False, True])
    values, defined = stage.stage_metric(emission, abstained, "session_ic", min_assets=3)
    plain, plain_defined = sd.session_correlations(panel, prediction, min_assets=3)
    # La primera sesión se abstiene y puntúa. La última tiene el objetivo constante y no.
    assert values[0] == 0.0 and defined[0] and not plain_defined[0]
    assert not defined[3]
    assert np.array_equal(values[1:3], plain[1:3]) and defined[1:3].all()
    raw, raw_defined = stage.stage_metric(emission, abstained, "session_mae_raw", min_assets=3)
    assert np.array_equal(raw, sd.session_mae(panel, prediction)) and raw_defined.all()
    assert stage.scorable(panel, 17).sum() == 0


def test_the_combination_mixes_the_models_with_their_own_calibration():
    protocol = protocol_of(ENTRIES)
    arms = variant_screening.all_arms(protocol)
    compared = stage.compared_models(protocol, arms)
    runners = {arms[label].runner for label in compared}
    assert runners <= set(stage.OWN_RUNNERS) and {"fit", "correction"} <= runners
    used = {
        row[side]
        for row in variant_screening.contrasts(protocol)
        for side in ("variant", "control")
    }
    left_out = sorted("/".join(label) for label in used - set(compared))
    # Fuera quedan las etapas de A1, la referencia textual de B1 y el control cero.
    assert all(
        label == "common/zero" or arms[tuple(label.split("/", 1))].runner in ("output", "reference")
        for label in left_out
    )
    assert "common/zero" in left_out and "B1/zero_memory" in left_out


def test_the_analysis_recovers_the_version_with_signal(analyzed):
    protocol, jobs, sources, report, arrays = analyzed
    decisions = report["decisions"]
    assert sorted(decisions) == sorted(protocol["selection"]["versions"]["models"])
    contrasts = {row["name"]: row for row in report["contrasts"]}
    primary = contrasts[f"version_arbiter/{REFERENCE}/arbiter-common_epoch"]
    assert primary["role"] == "primary" and primary["metric"] == "session_ic"
    members = {m["name"]: m for m in report["declared_family"]["16"]["members"]}
    # La familia propia solo reúne el contraste principal de cada modelo con versiones.
    assert list(members) == [
        f"version_arbiter/{model}/arbiter-common_epoch"
        for model in protocol["selection"]["versions"]["models"]
    ]
    assert all(row["contrast"] in members for row in decisions.values())
    assert members[primary["name"]]["estimate"] > 0.1
    assert decisions[REFERENCE]["primary_passes"]
    # La media uniforme diluye la versión con señal y el árbitro la supera.
    required = report["required_contrasts"][f"version_arbiter/{REFERENCE}/arbiter-uniform_mixture"]
    assert required["estimate"] > 0 and required["passes_marginal"]
    assert set(decisions[REFERENCE]["required"]) == {
        f"version_arbiter/{REFERENCE}/arbiter-uniform_mixture",
        f"version_arbiter/{REFERENCE}/arbiter-shuffled_outcomes",
    }
    # Sin señal en ninguna opción, F1 no pasa.
    assert not decisions["F1/laplace_scale_head"]["passes"]
    for label, value in report["coverage"].items():
        assert 0.0 <= value <= 1.0, label
    assert report["coverage"][f"version_arbiter/{REFERENCE}/common_epoch"] == 1.0
    assert report["model_combination"]["name"] == "model_combination/all/combination-simple_mean"
    # Cada origen y semilla pidió las versiones de sus ajustes versionados.
    assert all(source.version_calls for source in sources.values())
    assert report["sessions"] == 4 * 40 and report["stages"]["version_arbiter"]["seed"] == 20261028


def test_a_model_passes_only_with_its_primary_and_every_required_contrast():
    protocol = protocol_of(ENTRIES)
    rng = np.random.default_rng(5)
    sessions = 400

    def row(name, role, model, mean):
        return dict(
            name=name,
            role=role,
            model=model,
            metric="session_ic",
            direction="greater",
            values=mean + 0.01 * rng.standard_normal(sessions),
            defined=np.ones(sessions, dtype=bool),
            variant=("version_arbiter", model, "arbiter"),
            control=("version_arbiter", model, name),
        )

    rows = [
        row("a-common", "primary", "a", 0.05),
        row("a-uniform", "required", "a", 0.05),
        row("a-shuffled", "required", "a", -0.05),
        row("b-common", "primary", "b", 0.05),
        row("b-uniform", "required", "b", 0.05),
        row("b-shuffled", "required", "b", 0.05),
    ]
    averaging = [row("a-ema", "reported", "a", 0.0)]
    combination = row("combination", "reported", "all", 0.0)
    report = stage.families(protocol, rows, averaging, combination, np.arange(sessions))
    assert [m["name"] for m in report["declared_family"]["16"]["members"]] == [
        "a-common",
        "b-common",
    ]
    assert sorted(report["declared_family"]) == ["16", "32", "8"]
    first, second = report["decisions"]["a"], report["decisions"]["b"]
    # Un contraste exigido desfavorable impide pasar aunque pase el principal.
    assert first["primary_passes"] and not first["passes"]
    assert first["required"] == {"a-uniform": True, "a-shuffled": False}
    assert second["passes"] and second["required"] == {"b-uniform": True, "b-shuffled": True}
    assert set(first["robustness"]) == {"blocks_16", "blocks_8", "blocks_32", "holm"}
    assert [m["name"] for m in report["time_averaging"]["members"]] == ["a-ema"]
    assert report["model_combination"]["name"] == "combination"
    assert [row["name"] for row in report["contrasts"]][-2:] == ["a-ema", "combination"]


def test_each_series_is_the_seed_mean_of_the_stage_outputs(analyzed):
    protocol, jobs, sources, report, arrays = analyzed
    index = analysis.job_index(jobs)
    options_of = arbiters.stage_options(DECLARATION, arbiters.VERSION_ARBITER)
    parts = []
    for year in sorted(sources):
        source, items = sources[year], []
        for seed in protocol["seeds"]:
            job = index[(tuple(REFERENCE.split("/")), year, seed)][0]
            options = emissions(source, job)
            output, abstained = stage.mixed(options, source.origin().maturity, options_of, "online")
            items.append(stage.stage_metric(output, abstained, "session_ic", min_assets=3))
        parts.append(analysis.seed_average(items))
    values, defined = arrays["merged"][(("version_arbiter", REFERENCE, "arbiter"), "session_ic")]
    assert np.array_equal(values, np.concatenate([v for v, _ in parts]))
    assert np.array_equal(defined, np.concatenate([d for _, d in parts]))
    # El informe resume cada serie con su media en las sesiones definidas.
    assert report["models"][f"version_arbiter/{REFERENCE}/arbiter"]["session_ic"] == dict(
        mean=float(values[defined].mean()), defined_sessions=int(defined.sum())
    )


def test_the_summary_of_each_output_only_averages_its_defined_sessions():
    arbiter, emission = (
        ("version_arbiter", "B/x", "arbiter"),
        ("time_averaging", "F2/y", "emission"),
    )
    merged = {
        (arbiter, "session_ic"): (np.array([1.0, 3.0, 9.0]), np.array([True, True, False])),
        (arbiter, "session_mae"): (np.zeros(3), np.zeros(3, dtype=bool)),
        (emission, "session_ic"): (np.array([2.0, 5.0, 0.0]), np.array([True, False, False])),
    }
    means = stage._means(merged)
    # Las salidas van por orden y cada una conserva el orden de sus métricas.
    assert list(means) == ["time_averaging/F2/y/emission", "version_arbiter/B/x/arbiter"]
    assert means["version_arbiter/B/x/arbiter"] == {
        "session_ic": dict(mean=2.0, defined_sessions=2),
        "session_mae": dict(mean=None, defined_sessions=0),
    }
    assert means["time_averaging/F2/y/emission"] == {
        "session_ic": dict(mean=2.0, defined_sessions=1)
    }


def test_the_arbiter_is_compared_apart_with_the_time_averaging_models(analyzed):
    protocol, jobs, sources, report, arrays = analyzed
    names = [f"version_arbiter/{REFERENCE}/arbiter-{name}" for name in AVERAGING]
    family = report["time_averaging"]
    assert [member["name"] for member in family["members"]] == names
    # Las comparaciones no entran en la familia declarada ni en ninguna decisión.
    declared = {member["name"] for member in report["declared_family"]["16"]["members"]}
    assert declared.isdisjoint(names) and set(report["required_contrasts"]).isdisjoint(names)
    assert all(row["contrast"] not in names for row in report["decisions"].values())
    # Solo la versión de la época 12 tiene señal, así que el árbitro supera a los dos modelos.
    assert all(member["estimate"] > 0.1 for member in family["members"])
    arms, index = variant_screening.all_arms(protocol), analysis.job_index(jobs)
    rows = {row["name"]: row for row in arrays["contrasts"]}
    arbiter = arrays["merged"][(("version_arbiter", REFERENCE, "arbiter"), "session_ic")]
    for name, contrast in zip(AVERAGING, names, strict=True):
        label, parts = tuple(name.split("/", 1)), []
        for year in sorted(sources):
            origin, items = sources[year].origin(), []
            for seed in protocol["seeds"]:
                emission = analysis.emission(protocol, arms, index, origin, label, seed, [])
                items.append(analysis.session_metric(emission, "session_ic", min_assets=3))
            parts.append(analysis.seed_average(items))
        values, defined = arrays["merged"][(("time_averaging", name, "emission"), "session_ic")]
        # La serie del modelo es la misma media de semillas que usa la criba.
        assert np.array_equal(values, np.concatenate([v for v, _ in parts]))
        assert np.array_equal(defined, np.concatenate([d for _, d in parts]))
        row = rows[contrast]
        both = arbiter[1] & defined
        assert row["role"] == "reported" and np.array_equal(row["defined"], both)
        np.testing.assert_array_equal(row["values"][both], (arbiter[0] - values)[both])


def test_the_declared_time_averaging_models_have_their_own_predictions():
    protocol = variant_screening.load_protocol()
    averaging = arbiters.stage_options(REAL_DECLARATION, arbiters.VERSION_ARBITER)["time_averaging"]
    assert averaging["arbiter_of"] == REFERENCE
    assert REFERENCE in protocol["selection"]["versions"]["models"]
    compared = stage.compared_models(protocol, variant_screening.all_arms(protocol))
    assert all(tuple(name.split("/", 1)) in compared for name in averaging["compared_with"])
    assert {name.split("/")[0] for name in averaging["compared_with"]} == {"F2", "F3"}


def test_shared_jobs_are_one_option_of_the_combination(analyzed):
    protocol, jobs, sources, report, arrays = analyzed
    index = analysis.job_index(jobs)
    compared = stage.compared_models(protocol, variant_screening.all_arms(protocol))
    shared = 0
    for key, members in report["combination_members"].items():
        year, seed = (int(part) for part in key.split("/"))
        origin, expected = sources[year].origin(), {}
        for label in compared:
            job = analysis.chosen_job(protocol, origin, index, label, seed, [])
            expected.setdefault(job["id"], "/".join(label))
        assert members == list(expected.values())
        shared += len(compared) - len(members)
    # En el catálogo reducido hay modelos que comparten trabajo y cuentan una vez.
    assert shared > 0


def test_the_combination_includes_the_common_models_with_their_own_predictions():
    protocol = protocol_of(ENTRIES)
    compared = stage.compared_models(protocol, variant_screening.all_arms(protocol))
    assert {("common", "momentum_reversal"), ("common", "lightgbm")} <= set(compared)


def test_a_combination_reads_continuations_only_after_the_first_origin():
    protocol, jobs, sources = screening(("A1", "F3"))
    arms = variant_screening.all_arms(protocol)
    index = analysis.job_index(jobs)
    compared = stage.compared_models(protocol, arms)
    options = arbiters.stage_options(REAL_DECLARATION, arbiters.MODEL_COMBINATION)
    first = protocol["origins"][0]["year"]
    for year in sorted(sources):
        origin = sources[year].origin()
        members = {}
        emitted = list(
            stage._combination_emissions(
                protocol, arms, index, origin, protocol["seeds"][0], [], options, compared, members
            )
        )
        assert len(emitted) == 2 and all(item[1].panel.rows for item in emitted)
        continuation = [
            label
            for label in members[f"{year}/{protocol['seeds'][0]}"]
            if "continues" in arms[tuple(label.split("/", 1))].definition
        ]
        assert bool(continuation) is (year != first)


def test_a_missing_first_origin_is_undefined_in_a_time_averaging_comparison():
    protocol, jobs, sources = screening(("A1", "F3"))
    declaration = copy.deepcopy(REAL_DECLARATION)
    declaration["stages"][arbiters.VERSION_ARBITER]["analysis"]["time_averaging"][
        "compared_with"
    ] = ["F3/continuation_control"]
    stages = stage._checked_stages(protocol, declaration)
    arms = variant_screening.all_arms(protocol)
    origins = {year: source.origin() for year, source in sources.items()}
    times, merged, read = stage._series(
        protocol,
        arms,
        analysis.job_index(jobs),
        origins,
        stages,
        stage.compared_models(protocol, arms),
    )
    first = min(origins)
    count = len(sources[first].times["evaluation"]) // len(ASSETS)
    values, defined = merged[
        ((stage.TIME_AVERAGING, "F3/continuation_control", "emission"), "session_ic")
    ]
    assert len(values) == len(times)
    assert not defined[:count].any() and not values[:count].any()
    assert defined[count:].all()
    assert read["coverage"]["time_averaging/F3/continuation_control/emission"] == 0.75
    assert all(
        not sources[first].jobs[name.partition("/")[0]]["continues"]
        for name in sources[first].loads
    )


def test_versioned_continuations_keep_the_first_origin_explicitly_unavailable():
    protocol, jobs, sources = screening(("A1", "F3"))
    protocol["selection"]["versions"]["models"] = [REFERENCE, "F3/continuation_control"]
    jobs = variant_screening.screening_jobs(protocol)
    epochs = protocol["selection"]["versions"]["epochs"]
    sources = {year: VersionedOrigin(year, jobs, epochs) for year in sources}
    arms = variant_screening.all_arms(protocol)
    origins = {year: source.origin() for year, source in sources.items()}
    declaration = copy.deepcopy(REAL_DECLARATION)
    declaration["stages"][arbiters.VERSION_ARBITER]["analysis"]["time_averaging"][
        "compared_with"
    ] = ["F3/schedule_free"]
    times, merged, read = stage._series(
        protocol,
        arms,
        analysis.job_index(jobs),
        origins,
        stage._checked_stages(protocol, declaration),
        stage.compared_models(protocol, arms),
    )
    count = len(sources[min(sources)].times["evaluation"]) // len(ASSETS)
    for mode in ("common_epoch", "arbiter", "uniform_mixture", "shuffled_outcomes"):
        label = (stage.VERSIONS, "F3/continuation_control", mode)
        values, defined = merged[label, "session_ic"]
        assert len(values) == len(times)
        assert not defined[:count].any() and not values[:count].any()
        assert defined[count:].any()
        assert 0 <= read["coverage"]["/".join(label)] <= 0.75


def test_missing_or_foreign_versions_are_refused():
    for options in (dict(drop=3), dict(extra=5)):
        protocol, jobs, sources = screening(**options)
        origins = {year: source.origin() for year, source in sources.items()}
        with pytest.raises(ValueError, match="no emite las versiones declaradas"):
            stage.analyze(protocol, DECLARATION, jobs, origins)
    protocol, jobs, sources = screening(shift=1.0)
    origins = {year: source.origin() for year, source in sources.items()}
    with pytest.raises(ValueError, match="no predice las filas de calibration"):
        stage.analyze(protocol, DECLARATION, jobs, origins)
    protocol, jobs, sources = screening()
    origins = {year: replace(source.origin(), versions=None) for year, source in sources.items()}
    with pytest.raises(ValueError, match="no recibe las versiones"):
        stage.analyze(protocol, DECLARATION, jobs, origins)


def test_the_stages_must_score_like_the_screening():
    protocol, jobs, sources = screening()
    origins = {year: source.origin() for year, source in sources.items()}
    changed = json.loads(json.dumps(DECLARATION))
    changed["stages"]["model_combination_v1"]["min_assets"] = 4
    with pytest.raises(ValueError, match="activos mínimos de la criba"):
        stage.analyze(protocol, changed, jobs, origins)
    # El catálogo reducido no tiene F2 ni F3, así que la declaración real no se puede analizar.
    with pytest.raises(ValueError, match="medias de pesos"):
        stage.analyze(protocol, REAL_DECLARATION, jobs, origins)
    changed = json.loads(json.dumps(DECLARATION))
    changed["stages"]["version_arbiter_v1"]["analysis"]["time_averaging"]["arbiter_of"] = (
        "B1/area_memory"
    )
    with pytest.raises(ValueError, match="medias de pesos"):
        stage.analyze(protocol, changed, jobs, origins)
    other = copy.deepcopy(protocol)
    other["contrast"]["primary"]["metric"] = "session_rank_ic"
    with pytest.raises(ValueError, match="métrica principal"):
        stage.analyze(other, DECLARATION, jobs, origins)
    with pytest.raises(ValueError, match="cuatro orígenes"):
        stage.analyze(protocol, DECLARATION, jobs, {2019: origins[2019]})
    with pytest.raises(ValueError, match="Modo de mezcla"):
        stage.mixed([], origins[2019].maturity, None, "best")
