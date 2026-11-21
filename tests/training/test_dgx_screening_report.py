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
