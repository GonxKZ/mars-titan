"""Plan declarado de la campaña con máscaras: variantes A y B, recuentos y reglas comunes.

Estas pruebas no leen vistas ni datos, no reservan la GPU y no ajustan ningún modelo.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from mars_titan.data.storage import atomic_json
from mars_titan.training import campaign_plan as plan
from mars_titan.training.reference_design import design_cases

BASELINES = Path("configs/baselines")
EVALUATION = Path("configs/evaluation")
CAMPAIGNS = {
    variant: BASELINES / f"historical-masked-campaign-{variant.lower()}.json" for variant in "AB"
}
NEURAL_ARMS = ("rnn", "lstm", "gru", "dlinear", "transformer_compact")
# Recuentos derivados a mano: por ventana reentrenada, cada brazo neuronal ajusta dos
# candidatos con la semilla 42 y dos finalistas, Ridge tres alfas y XGBoost doce
# configuraciones con la semilla 42 y dos finalistas. Cada traslado es uno por semilla.
EXPECTED = {
    "A": dict(
        windows=dict(US=(19, 0), CN=(13, 0), JOINT=(13, 0)),
        training=1665,
        prediction=0,
    ),
    "B": dict(
        windows=dict(US=(7, 12), CN=(5, 8), JOINT=(5, 8)),
        training=629,
        prediction=532,
    ),
}


def loaded(variant):
    return plan.load_campaign(CAMPAIGNS[variant])


def evaluation_year(campaign, scope, window):
    fold = campaign["comparison_config"]["resolved_scopes"][scope]["windows"][window]
    return date.fromisoformat(fold["evaluation"][0]).year


@pytest.mark.parametrize("variant", ["A", "B"])
def test_declared_variants_count_jobs_per_scope_arm_and_seed(variant):
    campaign = loaded(variant)
    counts = plan.count_jobs(campaign)
    expected = EXPECTED[variant]
    assert counts["training_jobs"] == expected["training"]
    assert counts["prediction_jobs"] == expected["prediction"]
    for scope, key in (("US", "US"), ("CN", "CN"), ("US+CN", "JOINT")):
        trained, carried = expected["windows"][key]
        record = counts["scopes"][scope]
        assert len(record["retrained_windows"]) == trained
        assert record["carried_windows"] == carried
        assert record["windows"] == trained + carried
        for arm in NEURAL_ARMS:
            assert record["arms"][arm] == {
                "42": dict(fit=2 * trained, carry=carried),
                "43": dict(fit=trained, carry=carried),
                "44": dict(fit=trained, carry=carried),
            }
        assert record["arms"]["ridge"] == {"42": dict(fit=3 * trained, carry=carried)}
        assert record["arms"]["xgboost"] == {
            "42": dict(fit=12 * trained, carry=carried),
            "43": dict(fit=trained, carry=carried),
            "44": dict(fit=trained, carry=carried),
        }
        assert record["training_jobs"] == trained * (5 * 4 + 3 + 14)
        assert record["prediction_jobs"] == carried * (5 * 3 + 1 + 3)
    # Los límites declarados coinciden con el plan: cualquier ampliación exige cambiarlos.
    assert campaign["limits"] == dict(
        max_training_jobs=expected["training"], max_prediction_jobs=expected["prediction"]
    )


def test_variant_b_retrains_every_three_years_and_joint_anchors_are_us_anchors():
    campaign = loaded("B")
    counts = plan.count_jobs(campaign)["scopes"]
    years = {
        scope: [evaluation_year(campaign, scope, w) for w in counts[scope]["retrained_windows"]]
        for scope in ("US", "CN", "US+CN")
    }
    assert years["US"] == [2005, 2008, 2011, 2014, 2017, 2020, 2023]
    assert years["CN"] == years["US+CN"] == [2011, 2014, 2017, 2020, 2023]
    assert set(years["US+CN"]) <= set(years["US"])


def test_variant_b_never_carries_a_model_past_its_information():
    campaign = loaded("B")
    jobs = plan.plan_campaign(campaign)
    by_id = {job["id"]: job for job in jobs}
    for job in jobs:
        windows = campaign["comparison_config"]["resolved_scopes"][job["scope"]]["windows"]
        if job["kind"] == plan.FIT:
            assert job["anchor"] == job["window"]
            assert all(by_id[dep]["window"] == job["window"] for dep in job["depends"])
            continue
        anchor, window = windows[job["anchor"]], windows[job["window"]]
        # El ancla deja de aprender al final de su validación, antes de esta calibración.
        assert anchor["validation"][1] <= window["calibration"][0]
        assert anchor["evaluation"][0] < window["evaluation"][0]
        assert (
            1
            <= evaluation_year(campaign, job["scope"], job["window"])
            - evaluation_year(campaign, job["scope"], job["anchor"])
            <= 2
        )
        stages = {by_id[dep]["stage"] for dep in job["depends"]}
        assert all(by_id[dep]["window"] == job["anchor"] for dep in job["depends"])
        assert stages == ({"search"} if job["seed"] == 42 else {"finalist"})
        assert all(by_id[dep]["seed"] == job["seed"] for dep in job["depends"] if job["seed"] != 42)
    # El plan es topológico: cada dependencia aparece antes que el trabajo que la usa.
    position = {job["id"]: index for index, job in enumerate(jobs)}
    assert all(position[dep] < position[job["id"]] for job in jobs for dep in job["depends"])


def test_schedule_rejects_an_anchor_that_would_see_the_carried_calibration():
    # La primera ventana valida hasta noviembre y la segunda calibraría desde octubre.
    folds = [
        dict(
            id="fold-000",
            validation=["2004-04-01", "2005-11-01"],
            calibration=["2005-11-01", "2006-01-01"],
        ),
        dict(
            id="fold-001",
            validation=["2005-04-01", "2005-10-01"],
            calibration=["2005-10-01", "2006-01-01"],
        ),
    ]
    with pytest.raises(ValueError, match="información posterior"):
        plan.schedule(folds, 2)
    assert [row["trained"] for row in plan.schedule(folds, 1)] == [True, True]


def test_neural_cases_apply_the_protocol_rule_and_the_common_pinball():
    campaign = loaded("A")
    protocol = json.loads((EVALUATION / "historical-masked-us-walk-forward-v2.json").read_text())
    rule = protocol["selection"]
    assert campaign["rule"] == rule
    for arm, kind in campaign["neural"]["arms"].items():
        design = design_cases(
            [kind], seed=42, epochs=30, patience=5, min_delta=1e-5, stopping="fixed_budget"
        )
        cases = dict(campaign["neural"]["candidates"][arm])
        assert list(cases) == [f"{kind}-00", f"{kind}-10"]
        for index in (0, 10):
            case = cases[f"{kind}-{index:02d}"]
            assert case["selection"] == {k: v for k, v in rule.items() if k != "max_epochs"}
            assert case["epochs"] == rule["max_epochs"] == 30
            assert case["selection"]["min_delta"] == 1e-5
            assert case["head"] == "quantile_head_v1" and case["loss"] == "pinball"
            # Solo cambian la cabeza y la pérdida frente al caso del diseño.
            original = design[index]["case"]
            assert {k: v for k, v in case.items() if k not in {"head", "loss"}} == {
                k: v for k, v in original.items() if k != "loss"
            }


def test_campaign_agrees_with_the_historical_reference_plan_and_the_head_control():
    campaign = json.loads(CAMPAIGNS["A"].read_text())
    reference = json.loads((BASELINES / "historical-masked-reference-search-us.json").read_text())
    control = json.loads((BASELINES / "quantile-head-control-us.json").read_text())
    neural = campaign["neural"]
    assert list(neural["arms"].values()) == reference["models"]
    for key in ("case_indices", "batch_size", "context_sessions", "checkpoint_seconds"):
        assert neural[key] == reference[key]
    assert neural["search_seed"] == reference["search_seed"]
    assert neural["prediction_retention"] == reference["prediction_retention"]
    assert (
        neural["case_indices"] == control["case_indices"] and neural["head"] == control["arms"][1]
    )
    assert (
        json.loads(CAMPAIGNS["B"].read_text())
        | dict(
            name=campaign["name"], variant="A", retrain_every_months=12, limits=campaign["limits"]
        )
        == campaign
    )


def test_constants_repeat_the_runner_names_without_importing_torch():
    from mars_titan.training.reference_run import HELDOUT_FULL_TRAIN_SESSIONS

    assert plan.HELDOUT_RETENTION == HELDOUT_FULL_TRAIN_SESSIONS


def write_variant(tmp_path, base="A", **changes):
    value = json.loads(CAMPAIGNS[base].read_text())
    value["comparison"] = str((EVALUATION / "historical-masked-2000-comparison.json").resolve())
    value["tabular"]["config"] = str((BASELINES / "tabular-historical-masked.json").resolve())
    for key, item in changes.items():
        if isinstance(item, dict) and isinstance(value.get(key), dict):
            value[key] = value[key] | item
        else:
            value[key] = item
    path = tmp_path / "campaign.json"
    atomic_json(path, value)
    return path


@pytest.mark.parametrize(
    ("variant", "limits", "message"),
    [
        ("A", dict(max_training_jobs=1664), "1665 trabajos.*max_training_jobs=1664"),
        ("B", dict(max_prediction_jobs=531), "532 trabajos.*max_prediction_jobs=531"),
        ("B", dict(max_training_jobs=628), "629 trabajos.*max_training_jobs=628"),
    ],
)
def test_declared_limits_reject_a_plan_over_budget(tmp_path, variant, limits, message):
    path = write_variant(tmp_path, variant, limits=limits)
    with pytest.raises(ValueError, match=message):
        plan.check_campaign(path)


@pytest.mark.parametrize(
    "changes",
    [
        dict(variant="B"),
        dict(retrain_every_months=18),
        dict(retrain_every_months=0),
        dict(status="executed"),
        dict(final_test_opened=True),
        dict(scopes=["US", "US"]),
        dict(scopes=["EU"]),
        dict(neural=dict(head="scalar_l1")),
        dict(neural=dict(prediction_retention="full_train_validation_v1")),
        dict(neural=dict(search_seed=7)),
        dict(neural=dict(arms=dict(rnn="rnn"))),
        dict(neural=dict(case_indices=[10, 0])),
        dict(tabular=dict(cpu_workers=0)),
        dict(tabular=dict(arms=dict(ridge="ridge"))),
    ],
)
def test_campaign_contract_rejects_changes_that_alter_the_design(tmp_path, changes):
    with pytest.raises(ValueError):
        plan.load_campaign(write_variant(tmp_path, "A", **changes))


def test_tabular_configuration_must_declare_the_campaign_policy(tmp_path):
    strict = json.loads((BASELINES / "tabular-convergence-us.json").read_text())
    atomic_json(tmp_path / "strict.json", strict)
    with pytest.raises(ValueError, match="política"):
        plan.load_campaign(
            write_variant(tmp_path, tabular=dict(config=str(tmp_path / "strict.json")))
        )


def test_protocols_with_different_stopping_rules_are_rejected(tmp_path):
    comparison = json.loads((EVALUATION / "historical-masked-2000-comparison.json").read_text())
    protocols = {}
    for name in ("us", "cn"):
        value = json.loads(
            (EVALUATION / f"historical-masked-{name}-walk-forward-v2.json").read_text()
        )
        if name == "cn":
            value["selection"] = value["selection"] | dict(patience=6)
        protocols[name] = tmp_path / f"{name}.json"
        atomic_json(protocols[name], value)
    comparison["scopes"]["US"]["protocols"] = {"US": "us.json"}
    comparison["scopes"]["CN"]["protocols"] = {"CN": "cn.json"}
    comparison["scopes"].pop("US+CN")
    atomic_json(tmp_path / "comparison.json", comparison)
    path = write_variant(
        tmp_path, comparison=str(tmp_path / "comparison.json"), scopes=["US", "CN"]
    )
    with pytest.raises(ValueError, match="misma regla de parada"):
        plan.load_campaign(path)
    single = write_variant(tmp_path, comparison=str(tmp_path / "comparison.json"), scopes=["US"])
    assert plan.load_campaign(single)["rule"]["patience"] == 5


def test_pending_families_and_later_stages_are_declared_not_planned():
    report = plan.check_campaign(CAMPAIGNS["B"])
    pending = report["pending_families"]
    assert set(pending) == {"episodic_gru", "titans_mac", "mars_titan", "cm_v1"}
    assert pending["titans_mac"]["arms"] == [
        "titans_transformer_direct",
        "titans_mac_disabled",
        "titans_mac_frozen",
        "titans_mac_online",
    ]
    planned = {job["arm"] for job in plan.plan_campaign(loaded("B"))}
    assert planned == {*NEURAL_ARMS, "ridge", "xgboost"}
    assert not planned & {arm for entry in pending.values() for arm in entry["arms"]}
    stage = report["later_stages"]["posttraining_adapter_matrix"]
    assert Path(stage["config"]).is_file()
    assert stage["pending"] == [
        "ejecución desde la cola",
        "conexión con las ventanas walk-forward",
        "objetivo pinball para padres con cuantiles",
    ]
    assert report["scientific_training_started"] is False and report["final_test_opened"] is False
