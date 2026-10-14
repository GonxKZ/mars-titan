"""Cobertura de la varianza predictiva y D3 de B6 kalman en la comparación (PT3, #455).

Las predicciones son sintéticas y no salen de ningún modelo ajustado. En el estudio de
`test_walk_forward_comparison`, `titans` hace de modelo kalman con una varianza y una
predicción retrasada escritas en la prueba y `gru` de control sin corrección. No hay pasos de
optimizador.
"""

import copy
import json

import numpy as np
import pytest

from mars_titan.evaluation import kalman_diagnostics as kd
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation import window_aggregates
from tests.evaluation import test_walk_forward_comparison as base
from tests.evaluation.test_walk_forward_predictive_ability import long_decisions
from tests.evaluation.test_walk_forward_retention import session_mae

JOINT = base.ROOT / "configs/evaluation/historical-masked-2000-joint-comparison.json"
DECLARATION = base.ROOT / "configs/evaluation/kalman-associative-comparison.json"
SECTION = json.loads(JOINT.read_text())[kd.FIELD]
VARIANCE = 0.36


def test_the_campaign_section_follows_the_pt3_declaration():
    declared = json.loads(DECLARATION.read_text())
    config = json.loads(JOINT.read_text())
    assert kd.declaration(SECTION, config["arms"]) is SECTION
    models = declared["campaign"]["models"]
    assert SECTION["arms"] == {
        models["b6_kalman"]: models["titans_mac_online"],
        models["b6_kalman_rho0"]: models["titans_mac_online"],
    }
    assert SECTION["assessment_years"] == declared["metric"]["assessment_years"]
    rule = " ".join(declared["decision_rule"]["keep_if"])
    assert "coverage_of_correction_interval in [0.85, 0.95]" in rule
    assert "at least half of the gain with one extra session of delay" in rule
    assert SECTION["decision"] == dict(coverage=[0.85, 0.95], min_retained_gain=0.5)
    assert SECTION["z"] == 1.645


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s.update(status="computed"), "contrato"),
        (lambda s: s["arms"].update(mars_titan_b6_kalman="nadie"), "control"),
        (lambda s: s["arms"].update(titans_mac_online="titans_mac_online"), "control"),
        (lambda s: s.update(z=1), "z debe"),
        (lambda s: s.update(assessment_years=[2014, 2024]), "anterior a 2024"),
        (lambda s: s["decision"].update(coverage=[0.95, 0.85]), "cobertura"),
        (lambda s: s["decision"].update(min_retained_gain=0.0), "ganancia"),
    ],
)
def test_declaration_rejects_any_change_to_its_contract(edit, message):
    changed = copy.deepcopy(SECTION)
    edit(changed)
    with pytest.raises(ValueError, match=message):
        kd.declaration(changed, json.loads(JOINT.read_text())["arms"])


def delayed(values):
    """Predicción retrasada conocida: a medio camino entre la del modelo y la mitad del objetivo."""
    return 0.5 * values["prediction"] + 0.25 * np.asarray(values["target"])


def add_columns(arm, seed, fold, partition, values):
    if arm == "titans":
        values[kd.VARIANCE] = np.full(len(values["target"]), VARIANCE)
        values[kd.DELAYED] = delayed(values)


def declare(study):
    study.config[kd.FIELD] = dict(
        copy.deepcopy(SECTION), arms=dict(titans="gru"), assessment_years=[2023, 2023]
    )
    study.changes.append(add_columns)
    study.publish()
    return study


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "decisions", long_decisions)
        without = base.Study(tmp_path_factory.mktemp("plain")).run()[0]
        study = declare(base.Study(tmp_path_factory.mktemp("kalman")))
        report, _ = study.run()
        expected = {arm: session_mae(study, arm) for arm in ("gru", "titans")}
        hand = by_hand(study)
    return study, report, without, expected, hand


def by_hand(study):
    """Cobertura de las filas y MAE por sesión de la predicción retrasada, sin el panel."""
    covered, rows, sessions = 0, 0, {}
    for seed in base.ARMS["titans"]["seeds"]:
        for fold in study.folds:
            values = study.table("titans", seed, fold, "evaluation").to_pydict()
            late = delayed(dict(values, prediction=np.asarray(values["prediction"])))
            for market, moment, y, p, d in zip(
                values["market"],
                values["prediction_at"],
                values["target"],
                values["prediction"],
                late,
                strict=True,
            ):
                covered += abs(y - p) <= 1.645 * np.sqrt(VARIANCE)
                rows += 1
                sessions.setdefault((market, moment), {}).setdefault(seed, []).append(abs(d - y))
    late = {
        key: np.mean([np.mean(errors) for errors in seeds.values()])
        for key, seeds in sessions.items()
    }
    return dict(coverage=covered / rows, rows=rows, delayed=late)


def test_coverage_and_retained_gain_match_the_row_by_row_computation(runs):
    _, report, _, expected, hand = runs
    result = report[kd.FIELD]["arms"]["titans"]
    assert result["control"] == "gru" and result["seeds"] == [42, 43]
    for view in ("US+CN", "US", "CN"):
        markets = view.split("+")
        keys = sorted(key for key in expected["gru"] if key[0] in markets)
        control = np.array([expected["gru"][key] for key in keys])
        model = np.array([expected["titans"][key] for key in keys])
        late = np.array([hand["delayed"][key] for key in keys])
        found = result["views"][view]
        gains = found["label_delay"]
        assert gains["sessions"] == len(keys)
        assert gains["gain"] == pytest.approx((control - model).mean(), rel=1e-12)
        assert gains["delayed_gain"] == pytest.approx((control - late).mean(), rel=1e-12)
        if gains["gain"] > 0:
            assert gains["retained"] == pytest.approx(
                (control - late).mean() / (control - model).mean(), rel=1e-12
            )
        assert found["coverage"]["interval"][0] <= found["coverage"]["estimate"]
        assert found["coverage"]["estimate"] <= found["coverage"]["interval"][1]
        assert set(found["decision"]) == {"coverage_in_range", "label_delay"}
    joint = result["views"]["US+CN"]["coverage"]
    assert joint["rows"] == hand["rows"]
    assert joint["estimate"] == pytest.approx(hand["coverage"], rel=1e-12)
    assert list(joint["by_year"]) == ["2023"]


def test_the_rest_of_the_report_does_not_change(runs):
    _, report, without, _, _ = runs
    for key in ("contrasts", "arms", "interval_calibration", "sign_reliability"):
        assert report[key] == without[key]
    assert set(report["analysis_source_sha256"]) - set(without["analysis_source_sha256"]) == {
        "evaluation/kalman_diagnostics.py"
    }


def test_window_aggregates_give_the_same_section_as_reading_the_rows(runs, tmp_path):
    study, report, _, _, _ = runs
    config = walk.load_config(study.config_path)
    sources = walk.load_sources(study.sources_path, config, study.scope)
    folder = tmp_path / "aggregates"
    for window in sources["windows"]:
        window_aggregates.write(folder, config, sources, window)
    from_aggregates, _ = walk.evaluate_walk_forward(
        study.config_path, study.sources_path, study.scope, aggregates=folder
    )
    assert from_aggregates[kd.FIELD] == report[kd.FIELD]


def test_a_kalman_model_without_its_columns_is_rejected(tmp_path):
    study = base.Study(tmp_path)
    study.config[kd.FIELD] = dict(
        copy.deepcopy(SECTION), arms=dict(titans="gru"), assessment_years=[2023, 2023]
    )
    study.publish()
    with pytest.raises(ValueError, match="Faltan columnas"):
        study.run()


@pytest.mark.parametrize(
    ("gain", "late", "retained", "expected"),
    [
        (-0.1, 0.1, None, "no_gain_to_retain"),
        (0.2, -0.01, -0.05, "gain_disappears"),
        (0.2, 0.1, 0.5, "keeps_the_declared_share"),
        (0.2, 0.09, 0.45, "keeps_less_than_the_declared_share"),
    ],
)
def test_the_decision_applies_the_declared_thresholds(gain, late, retained, expected):
    decision = kd._decision(
        SECTION,
        dict(estimate=0.9),
        dict(gain=gain, delayed_gain=late, retained=retained),
    )
    assert decision == dict(coverage_in_range=True, label_delay=expected)
    outside = kd._decision(
        SECTION, dict(estimate=0.96), dict(gain=gain, delayed_gain=late, retained=retained)
    )
    assert outside["coverage_in_range"] is False
