"""Informe de la criba: IC por sesión, contrastes pareados, combinaciones y comparaciones."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "dgx_screening_report", ROOT / "scripts/dgx_screening_report.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)
STATES = importlib.util.spec_from_file_location(
    "screening_states", Path(__file__).with_name("screening_states.py")
)
states = importlib.util.module_from_spec(STATES)
STATES.loader.exec_module(states)
SESSIONS, write_model = states.SESSIONS, states.write_model


@pytest.fixture
def state(tmp_path):
    folder = tmp_path / "estado"
    write_model(folder, "control", 2013, 0.15, seed=1)
    write_model(folder, "mejor", 2013, 0.45, seed=2)
    write_model(folder, "otra_vista", 2013, 0.15, seed=3)
    write_model(folder, "ruido", 2013, 0.0, seed=4)
    pending = dict(
        status="running", model="a_medias", tier="core", year=2013, seed=42, attempts=[{}]
    )
    (folder / "queue/a_medias--2013--seed-42.json").write_text(json.dumps(pending))
    return folder


def spearman(left, right):
    order = lambda values: np.argsort(np.argsort(values)).astype(float)  # noqa: E731
    return float(np.corrcoef(order(left), order(right))[0, 1])


def _ranks_group_by_group(values):
    """La definición recorrida grupo a grupo, con la que se compara la versión de una pasada."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends, strict=True):
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2
    return ranks


def test_the_ranks_in_one_pass_equal_the_group_by_group_definition_bit_for_bit():
    rng = np.random.default_rng(7)
    cases = [
        np.empty(0),
        np.array([3.0]),
        np.array([2.0, 2.0]),
        np.array([np.nan, 1.0, np.nan, 1.0, 0.0]),
        np.array([np.inf, -np.inf, 0.0, np.inf]),
    ]
    for size in (2, 5, 30, 417, 4193):
        missing = rng.normal(size=size)
        missing[rng.random(size) < 0.1] = np.nan
        cases += [
            rng.normal(size=size),
            np.round(rng.normal(size=size), 1),
            rng.integers(0, 3, size=size).astype(np.float64),
            np.zeros(size),
            missing,
            rng.normal(size=size).astype(np.float32),
            np.where(rng.random(size) < 0.5, 0.0, -0.0),
        ]
    for values in cases:
        ranks, expected = report.average_ranks(values), _ranks_group_by_group(values)
        assert ranks.dtype == expected.dtype == np.float64
        assert np.array_equal(ranks, expected, equal_nan=True)
    assert report.average_ranks(np.zeros(4)).tolist() == [2.5, 2.5, 2.5, 2.5]


def test_the_ranks_average_ties_and_the_session_ic_is_the_spearman_of_each_session(state):
    assert report.average_ranks(np.array([2.0, 1.0, 2.0, 5.0])).tolist() == [2.5, 1.0, 2.5, 4.0]
    data = report.load_split(state / "runs/mejor/2013/seed-42", "validation")
    values = report.session_ic(data)
    assert len(values) == SESSIONS["validation"]
    market, instant, start, end = data["sessions"][3]
    expected = spearman(data["prediction"][start:end], data["target"][start:end])
    assert values[(market, instant)] == pytest.approx(expected, abs=1e-12)
    # Una sesión con menos de 30 activos no entra en la media.
    short = dict(data, sessions=[(market, instant, start, start + 29)])
    assert report.session_ic(short) == {}


def test_the_paired_contrast_separates_a_better_model_from_noise(state):
    controls = dict(controls=dict(mejor="control", ruido="control"), groups=dict(mejor="grupo"))
    document = report.build(state, [2013], 42, controls["controls"], controls["groups"])
    origin = document["origins"][0]
    assert origin["completed"] == ["control", "mejor", "otra_vista", "ruido"]
    rows = {row["model"]: row for row in origin["contrasts"]}
    assert rows["mejor"]["group"] == "grupo" and rows["mejor"]["control"] == "control"
    assert rows["mejor"]["decision"]["sessions"] == 60 and rows["mejor"]["decision"]["t"] > 4
    assert rows["mejor"]["calibration"]["sessions"] == 20
    assert rows["mejor"]["validation"]["sessions"] == 40
    assert (
        rows["mejor"]["evaluation"]["sessions"] == 60 and rows["mejor"]["evaluation"]["mean"] > 0.1
    )
    assert rows["ruido"]["decision"]["mean"] < 0 and rows["ruido"]["decision"]["t"] < -2
    model = origin["models"]["control"]
    assert model["hours"] == 1.0 and model["selected_validation_ic"] == pytest.approx(0.04)
    # Un trabajo reanudado suma el tiempo de todos sus intentos.
    path = state / "queue/control--2013--seed-42.json"
    record = json.loads(path.read_text())
    record["attempts"] = [dict(seconds=1800.0, status="interrupted"), dict(seconds=5400.0)]
    path.write_text(json.dumps(record))
    again = report.build(state, [2013], 42, {}, {})
    assert again["origins"][0]["models"]["control"]["hours"] == 2.0
    assert model["epoch_std"] == pytest.approx(np.std([0.02, 0.03, 0.04], ddof=1))
    assert model["ic"]["evaluation"]["sessions"] == 60 and model["ic_by_market"] == {}


def test_the_forward_selection_only_adds_members_that_raise_the_decision_ic(state):
    document = report.build(state, [2013], 42, {}, {})
    steps = document["origins"][0]["ensembles"]
    assert steps[0]["members"] == ["mejor"]
    scores = [step["decision_ic"] for step in steps]
    assert scores == sorted(scores) and len(set(scores)) == len(scores)
    assert "ruido" not in steps[-1]["members"]
    # Dos vistas ruidosas e independientes del mismo objetivo se combinan mejor que cada una.
    pair = report.blend(
        [
            report.model_report(state, json.loads(path.read_text()), 2013, 42)
            for path in (
                state / "queue/control--2013--seed-42.json",
                state / "queue/otra_vista--2013--seed-42.json",
            )
        ],
        "evaluation",
    )
    alone = document["origins"][0]["models"]["control"]["ic"]["evaluation"]["mean"]
    assert report.summary(pair.values())["mean"] > alone


def test_several_origins_pool_their_sessions_in_the_aggregate_contrast(state):
    write_model(state, "control", 2014, 0.15, seed=11)
    write_model(state, "mejor", 2014, 0.45, seed=12)
    document = report.build(state, [2013, 2014], 42, dict(mejor="control", ruido="control"), {})
    rows = {row["model"]: row for row in document["aggregate"]}
    assert rows["mejor"]["origins"] == [2013, 2014] and rows["mejor"]["decision"]["sessions"] == 120
    assert rows["mejor"]["mean_of_origins"] == pytest.approx(np.mean(rows["mejor"]["origin_means"]))
    assert rows["ruido"]["origins"] == [2013] and rows["ruido"]["decision"]["sessions"] == 60
    single = report.build(state, [2013], 42, dict(mejor="control"), {})
    assert single["aggregate"] == []


def test_a_reference_state_completes_the_models_and_a_comparison_pairs_the_common_sessions(
    tmp_path,
):
    joint, alone = tmp_path / "conjunta", tmp_path / "us"
    write_model(joint, "control", 2013, 0.3, seed=21, markets=("US", "CN"))
    write_model(alone, "control", 2013, 0.1, seed=22)
    write_model(alone, "solo_en_us", 2013, 0.2, seed=23)
    document = report.build(joint, [2013], 42, {}, {}, references=[alone], compare=alone)
    origin = document["origins"][0]
    assert sorted(origin["models"]) == ["control", "solo_en_us"]
    assert origin["models"]["solo_en_us"]["state"] == str(alone)
    by_market = origin["models"]["control"]["ic_by_market"]
    assert sorted(by_market) == ["CN", "US"] and by_market["CN"]["evaluation"]["sessions"] == 60
    assert origin["models"]["control"]["ic"]["evaluation"]["sessions"] == 120
    (row,) = document["comparison"]["rows"]
    # Solo son comunes las sesiones de EE. UU.: 60 de decisión y 60 de evaluación.
    assert row["model"] == "control" and row["decision"]["sessions"] == 60
    assert row["evaluation"]["sessions"] == 60 and row["evaluation"]["mean"] > 0


def test_the_command_writes_both_reports(state, tmp_path):
    controls = tmp_path / "controles.json"
    controls.write_text(json.dumps(dict(controls=dict(mejor="control"), groups={})))
    output = tmp_path / "informe"
    arguments = ["--state", str(state), "--year", "2013", "--controls", str(controls)]
    assert report.main([*arguments, "--output", str(output)]) == 0
    document = json.loads((output / "report.json").read_text())
    assert document["kind"] == "dgx_screening_report" and document["decision_splits"] == [
        "validation",
        "calibration",
    ]
    # Con un solo tramo declarado, la decisión usa solo sus sesiones.
    single = tmp_path / "solo-calibracion"
    assert (
        report.main([*arguments, "--decision-split", "calibration", "--output", str(single)]) == 0
    )
    alone = json.loads((single / "report.json").read_text())
    assert alone["decision_splits"] == ["calibration"]
    pooled_ic = document["origins"][0]["models"]["mejor"]["decision_ic"]
    assert alone["origins"][0]["models"]["mejor"]["decision_ic"] != pooled_ic
    text = (output / "report.md").read_text()
    assert "## Origen 2013: 4 modelos" in text and "| `mejor` | `control` |" in text
    assert "### Combinaciones por media de rangos del origen 2013" in text


def test_the_known_offset_of_the_label_is_measured_alone_and_with_each_model():
    random = np.random.default_rng(7)
    sessions, assets = 40, 60
    alpha = random.normal(0, 1.0, sessions * assets)
    skill = random.normal(0, 1.0, sessions * assets)
    free = 0.3 * skill + random.normal(0, 1.0, sessions * assets)
    # La etiqueta resta alfa; el modelo anticipa la parte libre y no sabe nada de alfa.
    data = dict(
        target=free - alpha,
        prediction=skill.copy(),
        sessions=[("US", s, s * assets, (s + 1) * assets) for s in range(sessions)],
    )
    alone = report.summary(report.offset_sessions(data, alpha, 0.0, free=False).values())
    model = report.summary(
        report.offset_sessions(data, alpha, 1.0, free=False, prediction=skill).values()
    )
    both = report.summary(
        report.offset_sessions(data, alpha, 0.3, free=False, prediction=skill).values()
    )
    clean = report.summary(
        report.offset_sessions(data, alpha, 1.0, free=True, prediction=skill).values()
    )
    assert alone["sessions"] == sessions and alone["mean"] > 0.5
    # El modelo solo acierta poco con la etiqueta oficial; combinado supera al desplazamiento.
    assert 0 < model["mean"] < alone["mean"] < both["mean"]
    # Sin alfa en la etiqueta se ve su habilidad real.
    assert clean["mean"] > model["mean"]
    # Las filas sin alfa conocido no cuentan, y una sesión con menos de 30 se omite.
    partial = alpha.copy()
    partial[: assets - 10] = np.nan
    fewer = report.offset_sessions(data, partial, 0.0, free=False)
    assert len(fewer) == sessions - 1


def test_the_prediction_for_the_label_without_alpha_is_the_prediction_plus_alpha():
    random = np.random.default_rng(11)
    sessions, assets = 30, 90
    count = sessions * assets
    alpha = random.normal(0, 1.0, count)
    skill = random.normal(0, 1.0, count)
    free = 0.5 * alpha + 0.4 * skill + random.normal(0, 1.0, count)
    liquidity = random.normal(0, 1.0, count)
    # Un modelo calibrado en unidades de retorno predice la etiqueta: lo operable menos alfa.
    prediction = 0.5 * alpha + 0.4 * skill - alpha
    data = dict(
        target=free - alpha,
        prediction=prediction,
        sessions=[("US", s, s * assets, (s + 1) * assets) for s in range(sessions)],
    )

    def mean(**options):
        return report.summary(report.implied_sessions(data, alpha, **options).values())["mean"]

    label = mean(free=False, prediction=prediction)
    implied = mean(free=True, prediction=prediction)
    as_is = report.summary(
        report.offset_sessions(data, alpha, 1.0, free=True, prediction=prediction).values()
    )["mean"]
    # Sumar alfa devuelve la predicción de lo operable; sin sumarlo, el modelo parece no saber.
    assert label > 0.3 and implied > 0.4 and implied > as_is + 0.3
    # Los desplazamientos solos: -alfa con la etiqueta y +alfa con la etiqueta sin alfa.
    assert mean(free=False) > 0.2 and mean(free=True) > 0.2
    # El tercio más líquido usa un tercio de las filas de cada sesión.
    top = report.implied_sessions(data, alpha, liquidity, free=False, prediction=prediction)
    assert len(top) == sessions and abs(np.mean(list(top.values())) - label) < 0.1
    few = dict(data, sessions=[("US", 0, 0, 60)])
    assert report.implied_sessions(few, alpha, liquidity, free=False, prediction=prediction) == {}
    unknown = alpha.copy()
    unknown[: assets - 10] = np.nan
    assert len(report.implied_sessions(data, unknown, free=True, prediction=prediction)) == 29


def test_the_scale_of_the_decision_splits_recovers_the_skill_of_an_overdispersed_model():
    random = np.random.default_rng(5)
    sessions, assets = 40, 90
    count = sessions * assets

    def split():
        alpha = random.normal(0, 1.0, count)
        skill = random.normal(0, 1.0, count)
        free = 0.5 * alpha + 0.4 * skill + random.normal(0, 1.0, count)
        calibrated = 0.5 * alpha + 0.4 * skill - alpha
        data = dict(
            target=free - alpha,
            prediction=calibrated,
            sessions=[("US", s, s * assets, (s + 1) * assets) for s in range(sessions)],
        )
        return data, alpha

    splits, alphas = {}, {}
    for name in report.SPLITS:
        splits[name], alphas[name] = split()

    def free(data, alpha, factor):
        values = report.implied_sessions(
            data, alpha, free=True, prediction=factor * data["prediction"]
        )
        return report.summary(values.values())["mean"]

    # Con la predicción calibrada, el mejor factor es el suyo y no cambia casi nada.
    own = report.best_scale(splits, alphas)
    assert 0.7 <= own <= 1.5
    honest = free(splits["evaluation"], alphas["evaluation"], 1.0)
    assert free(splits["evaluation"], alphas["evaluation"], own) == pytest.approx(honest, abs=0.02)
    # Doce veces más dispersa: sumar alfa ya no devuelve lo operable, y el factor lo recupera.
    wide = {name: dict(data, prediction=12 * data["prediction"]) for name, data in splits.items()}
    scale = report.best_scale(wide, alphas)
    assert scale == pytest.approx(own / 12, rel=0.5) and scale in report.SCALES
    collapsed = free(wide["evaluation"], alphas["evaluation"], 1.0)
    recovered = free(wide["evaluation"], alphas["evaluation"], scale)
    assert collapsed < honest - 0.2 and recovered == pytest.approx(honest, abs=0.03)
    # El factor solo mira validación y calibración: cambiar la evaluación no lo mueve.
    changed = dict(
        wide, evaluation=dict(wide["evaluation"], prediction=-wide["evaluation"]["prediction"])
    )
    assert report.best_scale(changed, alphas) == scale
    # La dispersión delata la escala: la calibrada queda por debajo de uno.
    narrow = report.dispersion(splits["calibration"], alphas["calibration"])
    assert narrow < 1 and report.dispersion(
        wide["calibration"], alphas["calibration"]
    ) == pytest.approx(12 * narrow)
    partial = alphas["calibration"].copy()
    partial[: assets * (sessions - 1) + assets - 10] = np.nan
    assert report.dispersion(splits["calibration"], partial) is None


def test_the_newey_west_t_discounts_the_dependence_between_sessions():
    random = np.random.default_rng(3)
    independent = 0.05 + random.normal(0, 0.1, 400)
    plain = independent.mean() / (independent.std(ddof=1) / np.sqrt(len(independent)))
    assert report.newey_west_t(independent) == pytest.approx(plain, rel=0.25)
    # Una serie con memoria tiene menos información que su número de sesiones.
    noise = random.normal(0, 0.1, 400)
    persistent = np.empty(400)
    persistent[0] = noise[0]
    for index in range(1, 400):
        persistent[index] = 0.8 * persistent[index - 1] + noise[index]
    persistent += 0.05
    naive = persistent.mean() / (persistent.std(ddof=1) / np.sqrt(len(persistent)))
    assert abs(report.newey_west_t(persistent)) < 0.6 * abs(naive)
    assert report.newey_west_t(independent[:1]) is None
    assert report.newey_west_t(independent[:20]) is not None
    assert report.summary(independent)["t_nw"] == report.newey_west_t(independent)
    assert report.summary([])["t_nw"] is None


def test_the_inference_writes_the_sessions_and_tests_every_model_against_the_reference(
    state, tmp_path
):
    output = tmp_path / "informe"
    arguments = ["--state", str(state), "--year", "2013", "--output", str(output)]
    assert report.main([*arguments, "--inference", "control", "--bootstrap", "300"]) == 0
    lines = (output / "sessions.csv").read_text().splitlines()
    # Las sesiones de evaluación de los modelos que comparten las de la referencia.
    assert (
        lines[0] == "session,control,mejor,otra_vista,ruido"
        and len(lines) == 1 + states.SESSIONS["evaluation"]
    )
    assert lines[1].startswith("2013:US:")
    document = json.loads((output / "report.json").read_text())
    inference = document["inference"]
    assert (
        inference["metadata"]["benchmark"] == "control" and inference["metadata"]["n_models"] == 4
    )
    assert inference["metadata"]["n_sessions"] == states.SESSIONS["evaluation"]
    for name, item in inference["models"].items():
        mean = document["origins"][0]["models"][name]["ic"]["evaluation"]["mean"]
        assert item["mean"] == pytest.approx(mean, abs=1e-12)
    # El modelo con más señal supera a la referencia aunque se hayan probado todos.
    assert inference["spa"]["best_candidate"] == "mejor" and inference["spa"]["p_consistent"] < 0.05
    assert "mejor" in inference["mcs"]["included"] and "ruido" not in inference["mcs"]["included"]
    assert inference["models"]["mejor"]["diebold_mariano"]["p_value"] < 0.05
    text = (output / "report.md").read_text()
    assert "## Inferencia entre modelos frente a `control`" in text and "(fuera)" in text
    # La misma tabla de sesiones da la estabilidad de cada modelo.
    stability = document["stability"]
    assert set(stability) == set(inference["models"])
    values = {
        name: [float(line.split(",")[column]) for line in lines[1:]]
        for column, name in enumerate(lines[0].split(",")[1:], start=1)
    }
    for name, scopes in stability.items():
        item = scopes["all"]
        assert set(scopes) == {"all"} and item["sessions"] == states.SESSIONS["evaluation"]
        assert item["mean"] == pytest.approx(inference["models"][name]["mean"], abs=1e-12)
        positive = sum(value > 0 for value in values[name]) / len(values[name])
        assert item["positive_fraction"] == pytest.approx(positive)
        accumulated, peak, depth = 0.0, 0.0, 0.0
        for value in values[name]:
            accumulated += value
            peak = max(peak, accumulated)
            depth = max(depth, peak - accumulated)
        assert item["max_drawdown"]["depth"] == pytest.approx(depth, abs=1e-12)
        assert sum(month["sessions"] for month in item["monthly"]) == item["sessions"]
        assert all(month["month"].startswith("2012-") for month in item["monthly"])
        assert not any(month["month"] == "1970-01" for month in item["monthly"])
    assert stability["mejor"]["all"]["icir"] > stability["ruido"]["all"]["icir"]
    assert "## Estabilidad del IC de evaluación" in text and "| `mejor` | todos |" in text
    # Sin la opción no se escribe nada más, y una referencia que no está no rompe el informe.
    plain = tmp_path / "sin"
    assert report.main(["--state", str(state), "--year", "2013", "--output", str(plain)]) == 0
    assert not (plain / "sessions.csv").exists()
    assert json.loads((plain / "report.json").read_text())["inference"] is None
    assert json.loads((plain / "report.json").read_text())["stability"] is None
    absent = tmp_path / "ausente"
    assert report.main([*arguments[:4], "--output", str(absent), "--inference", "no_esta"]) == 0
    assert json.loads((absent / "report.json").read_text())["inference"] is None


def test_paired_uses_hac_t_smaller_than_iid_under_ar1_autocorrelation():
    """Bajo autocorrelación AR(1) phi=0.8, paired()['t'] (HAC) es estrictamente menor que t_iid."""
    rng = np.random.default_rng(19)
    n = 200
    noise = rng.normal(0.0, 0.05, size=n)
    diff = np.empty(n, dtype=np.float64)
    diff[0] = noise[0]
    for i in range(1, n):
        diff[i] = 0.8 * diff[i - 1] + noise[i]
    diff += 0.04
    base = {("US", i): 0.02 for i in range(n)}
    cand = {("US", i): 0.02 + float(diff[i]) for i in range(n)}

    res = report.paired(cand, base)
    assert res["t"] is not None and res["t_iid"] is not None
    assert res["se_hac"] is not None and res["hac_lags"] > 0
    assert res["t"] < res["t_iid"]
    assert res["se_hac"] > res["se_iid"]


def test_offset_reports_dropped_market_when_has_market_model_zero(tmp_path):
    """En US+CN cuando CN tiene has_market_model=0, offset_coverage y Markdown lo avisan."""
    folder = tmp_path / "joint"
    write_model(folder, "modelo_a", 2013, 0.3, seed=1, markets=("US", "CN"))
    records = report.completed(folder, 2013, 42)
    rep = report.model_report(folder, records["modelo_a"], 2013, 42)
    reports = {"modelo_a": rep}

    class StubProvider:
        def rows(self, run, split):
            data = rep["_splits"][split]
            n_rows = len(data["target"])
            alpha = np.full(n_rows, np.nan, dtype=np.float64)
            vol = np.full(n_rows, 10.0, dtype=np.float64)
            rng = np.random.default_rng(7)
            for market, _, start, end in data["sessions"]:
                if market == "US":
                    alpha[start:end] = rng.normal(0.0, 0.01, size=end - start)
            return alpha, vol

    known = report.offset_report(reports, StubProvider(), 2013, 42)
    assert known is not None
    cov = known["offset_coverage"]
    assert cov["dropped_markets"] == ["CN"]
    assert cov["covered_markets"] == ["US"]
    assert cov["by_market"]["CN"]["known_alpha_sessions"] == 0
    assert cov["by_market"]["CN"]["total_sessions"] == SESSIONS["evaluation"]
    assert cov["by_market"]["US"]["known_alpha_sessions"] == SESSIONS["evaluation"]

    document = dict(
        schema_version=1,
        kind="dgx_screening_report",
        state=str(folder),
        references=[],
        seed=42,
        min_assets=report.MIN_ASSETS,
        decision_splits=["calibration"],
        origins=[
            dict(
                year=2013,
                completed=["modelo_a"],
                missing_splits=[],
                models={"modelo_a": {k: v for k, v in rep.items() if not k.startswith("_")}},
                contrasts=[],
                ensembles=[],
                offset=known,
                offset_coverage=cov,
            )
        ],
        aggregate=[],
        comparison=None,
        inference=None,
        dropped_from_inference=[],
        stability=None,
    )
    md = report.markdown(document)
    assert "AVISO" in md and "`CN`" in md and "has_market_model == 0" in md


def test_best_scale_fits_calibration_only_not_validation():
    """Perturbar validation sin tocar calibration deja best_scale exactamente igual."""
    rng = np.random.default_rng(17)
    sessions, assets = 30, 60
    count = sessions * assets

    def make_split():
        alpha = rng.normal(0.0, 1.0, count)
        skill = rng.normal(0.0, 1.0, count)
        free = 0.5 * alpha + 0.4 * skill + rng.normal(0.0, 1.0, count)
        pred = 10.0 * (0.5 * alpha + 0.4 * skill - alpha)
        return (
            dict(
                target=free - alpha,
                prediction=pred,
                sessions=[("US", s, s * assets, (s + 1) * assets) for s in range(sessions)],
            ),
            alpha,
        )

    splits, alphas = {}, {}
    for name in report.SPLITS:
        splits[name], alphas[name] = make_split()

    base_scale = report.best_scale(splits, alphas)
    perturbed = dict(
        splits,
        validation=dict(
            splits["validation"], prediction=-1000.0 * splits["validation"]["prediction"]
        ),
    )
    assert report.best_scale(perturbed, alphas) == base_scale


def test_dropped_from_inference_explicitly_reported(state, tmp_path):
    """Un modelo al que le falta 1 sesión aparece en dropped_from_inference, nunca desaparece."""
    import pyarrow.parquet as pq

    path = state / "runs/ruido/2013/seed-42/evaluation-predictions.parquet"
    table = pq.read_table(path)
    first_ts = table.column("prediction_at")[0].as_py()
    mask = table.column("prediction_at").to_pylist()
    keep_indices = [idx for idx, val in enumerate(mask) if val != first_ts]
    pq.write_table(table.take(keep_indices), path)

    output = tmp_path / "drop_check"
    assert (
        report.main(
            [
                "--state",
                str(state),
                "--year",
                "2013",
                "--output",
                str(output),
                "--inference",
                "control",
                "--bootstrap",
                "200",
            ]
        )
        == 0
    )
    doc = json.loads((output / "report.json").read_text())
    dropped = doc["dropped_from_inference"]
    assert any(item["model"] == "ruido" and item["missing_sessions"] == 1 for item in dropped)
    assert "ruido" not in doc["inference"]["models"]
    md = (output / "report.md").read_text()
    assert "`ruido`" in md and "excluidos de la inferencia" in md


def test_the_implied_probability_matches_the_row_by_row_interpolation():
    """La versión vectorizada da lo mismo, bit a bit, que la interpolación fila a fila."""
    levels = np.asarray(report.QUANTILE_LEVELS, dtype=np.float64)

    def reference(row):
        if 0.0 <= row[0]:
            below = levels[0]
        elif 0.0 >= row[-1]:
            below = levels[-1]
        else:
            high = int(np.searchsorted(row, 0.0, side="right"))
            high = min(max(high, 1), len(levels) - 1)
            low = high - 1
            width = row[high] - row[low]
            if width <= 0.0:
                below = 0.5 * (levels[low] + levels[high])
            else:
                below = levels[low] + (0.0 - row[low]) / width * (levels[high] - levels[low])
        return float(np.clip(1.0 - below, 0.025, 0.975))

    rng = np.random.default_rng(23)
    rows = np.sort(rng.normal(0.0, 0.02, size=(4000, 5)) + rng.normal(0.0, 0.03, size=(4000, 1)))
    # Casos de borde: todo por encima, todo por debajo, ceros exactos y empates.
    edge = np.array(
        [
            [0.0, 0.01, 0.02, 0.03, 0.04],
            [-0.04, -0.03, -0.02, -0.01, 0.0],
            [-0.02, 0.0, 0.0, 0.01, 0.02],
            [-0.02, -0.01, -0.01, -0.01, 0.03],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.01, -0.01, 0.02, 0.02, 0.02],
        ]
    )
    rows = np.vstack([rows, edge])
    expected = np.array([reference(row) for row in rows])
    actual = report._implied_up_probability(rows)
    np.testing.assert_array_equal(actual, expected)
    assert report._implied_up_probability(np.empty((0, 5))).shape == (0,)


def test_a_state_without_validation_is_read_when_validation_does_not_decide(tmp_path):
    """Los órganos de memoria emiten calibración y evaluación: cuentan si decide la calibración."""
    state, organs = tmp_path / "estado", tmp_path / "organos"
    write_model(state, "control", 2013, 0.15, seed=1)
    write_model(organs, "memoria", 2013, 0.3, seed=2)
    (organs / "runs/memoria/2013/seed-42/validation-predictions.parquet").unlink()
    controls = dict(memoria="control")
    pooled = report.build(state, [2013], 42, controls, {}, references=[organs])
    assert pooled["origins"][0]["missing_splits"] == [
        dict(model="memoria", state=str(organs), missing=["validation"])
    ]
    single = report.build(
        state, [2013], 42, controls, {}, references=[organs], decision_splits=("calibration",)
    )
    origin = single["origins"][0]
    assert sorted(origin["models"]) == ["control", "memoria"] and origin["missing_splits"] == []
    ic = origin["models"]["memoria"]["ic"]
    assert ic["validation"]["sessions"] == 0
    assert ic["evaluation"]["sessions"] == SESSIONS["evaluation"]
    (row,) = origin["contrasts"]
    assert row["validation"]["sessions"] == 0
    assert row["decision"]["sessions"] == SESSIONS["calibration"] and row["evaluation"]["mean"] > 0
    assert origin["ensembles"][0]["members"] == ["memoria"]
    assert "| `memoria` | `control` |" in report.markdown(single)
    # El desplazamiento conocido mide solo los tramos que hay.
    reports = {
        name: report.model_report(folder, record, 2013, 42, decision_splits=("calibration",))
        for folder in (state, organs)
        for name, record in report.completed(folder, 2013, 42).items()
    }

    class Provider:
        def rows(self, run, split):
            rows = len(reports[run.parts[-3]]["_splits"][split]["target"])
            return np.random.default_rng(rows).normal(0, 0.01, rows), np.full(rows, 10.0)

    known = report.offset_report(reports, Provider(), 2013, 42)
    assert set(known["models"]["memoria"]["free"]) == {"calibration", "evaluation"}
    assert set(known["models"]["control"]["free"]) == set(report.SPLITS)
