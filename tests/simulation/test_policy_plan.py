"""Declaración y plan de la etapa de políticas, sin leer datos ni ajustar políticas.

Los recuentos se fijan con las configuraciones del repositorio. Las mutaciones de la
declaración deben fallar antes de planificar.
"""

import copy
import importlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mars_titan.data.storage import atomic_json
from mars_titan.simulation import campaign_stage, policy_plan
from mars_titan.training import campaign_plan as plan

CONFIGS = Path("configs/simulation")
STAGES = {v: CONFIGS / f"historical-masked-rl-stage-{v.lower()}.json" for v in "AB"}
POLICIES = CONFIGS / "historical-masked-rl-policies.json"
LEARNED = ("klpo_terminal", "ppo_clip_full_kl", "ppo_kl_penalty_adaptive")
LEARNED += ("ppo_clip_kl_epoch_stop", "double_dqn")
REFERENCES = ("cash", "hold_initial", "rebalance_50")
# Ventanas de política por ámbito: las del protocolo menos las cuatro primeras, que
# aportan los tres años de ajuste y el de validación de la primera política.
WINDOWS = {"US": 15, "CN": 9}
ANCHORS = dict(A={"US": 15, "CN": 9}, B={"US": 5, "CN": 3})


def us(day):
    return int(np.datetime64(day, "us").astype(np.int64))


@pytest.mark.parametrize(("variant", "fits", "carries"), [("A", 720, 0), ("B", 240, 480)])
def test_repository_stages_count_every_window_predictor_arm_seed_and_reference(
    variant, fits, carries
):
    result = campaign_stage.check_stage(STAGES[variant])
    counts = result["counts"]
    assert (counts["training_jobs"], counts["carried_jobs"]) == (fits, carries)
    assert counts["reference_jobs"] == 144 and counts["evaluation_jobs"] == carries + 144
    assert counts["evaluation_episodes"] == (fits + carries + 144) * 3
    stage = campaign_stage.load_stage(STAGES[variant])
    assert stage["limits"] == dict(max_training_jobs=fits, max_evaluation_jobs=carries + 144)
    for scope, total in WINDOWS.items():
        entry = counts["scopes"][scope]
        anchors = ANCHORS[variant][scope]
        assert len(entry["windows"]) == total and len(entry["anchors"]) == anchors
        assert entry["carried_windows"] == total - anchors
        # Dos predictores, cinco brazos aprendidos con tres semillas y tres referencias.
        assert entry["training_jobs"] == anchors * 2 * 5 * 3
        assert entry["carried_jobs"] == (total - anchors) * 2 * 5 * 3
        assert entry["reference_jobs"] == total * 2 * 3
        assert list(entry["arms"]) == [*LEARNED, *REFERENCES]
        for arm in LEARNED:
            fit = dict(fit=anchors * 2) if anchors else {}
            carry = dict(carry=(total - anchors) * 2) if total > anchors else {}
            assert entry["arms"][arm] == {str(s): fit | carry for s in (42, 43, 44)}
        for arm in REFERENCES:
            assert entry["arms"][arm] == {"none": dict(reference=total * 2)}
    assert result["contrasts"]["primary"] == "klpo_terminal"
    assert result["selection"]["metric"] == "ruin_count_then_mean_liquidated_log_growth"
    assert result["seeds"] == [42, 43, 44] and result["budget"]["transitions"] == 262144
    assert result["scientific_training_started"] is False and result["final_test_opened"] is False


def test_first_policy_windows_follow_three_training_years_and_one_validation_year():
    stage = campaign_stage.load_stage(STAGES["B"])
    us_rows = policy_plan.scope_windows(stage, "US")
    assert us_rows[0] == dict(
        window="fold-004",
        train=["fold-000", "fold-001", "fold-002"],
        validation="fold-003",
        anchor="fold-004",
        trained=True,
    )
    assert [row["window"] for row in us_rows if row["trained"]] == [
        "fold-004",
        "fold-007",
        "fold-010",
        "fold-013",
        "fold-016",
    ]
    assert {row["anchor"] for row in us_rows[1:3]} == {"fold-004"}
    cn = policy_plan.scope_windows(stage, "CN")
    assert cn[-1]["window"] == "fold-012" and cn[-1]["anchor"] == "fold-010"


@pytest.mark.parametrize("variant", "AB")
def test_every_job_ends_its_information_before_its_evaluation_and_never_reaches_2024(variant):
    stage = campaign_stage.load_stage(STAGES[variant])
    jobs = policy_plan.plan_stage(stage)
    resolved = stage["campaign"]["comparison_config"]["resolved_scopes"]
    for job in jobs:
        folds = resolved[job["scope"]]["windows"]
        segments = [folds[name]["evaluation"] for name in (*job["train"], job["validation"])]
        evaluated = folds[job["window"]]["evaluation"]
        assert all(us(a[1]) <= us(b[0]) for a, b in zip(segments, segments[1:], strict=False))
        # La política termina de aprender al final de su validación, antes de evaluar.
        assert us(segments[-1][1]) <= us(evaluated[0]) and us(evaluated[1]) <= us("2024-01-01")
        assert job["anchor"] == job["window"] or job["kind"] != "fit"


def test_klpo_is_planned_first_and_each_carry_depends_on_the_fit_of_its_anchor():
    stage = campaign_stage.load_stage(STAGES["B"])
    jobs = policy_plan.plan_stage(stage)
    by_id = {job["id"]: index for index, job in enumerate(jobs)}
    groups = {}
    for job in jobs:
        groups.setdefault((job["scope"], job["window"], job["predictor"]), []).append(job)
    for group in groups.values():
        assert group[0]["arm"] == "klpo_terminal"
        assert [job["arm"] for job in group[-3:]] == list(REFERENCES)
    for job in jobs:
        if job["kind"] == "carry":
            (anchor,) = job["depends"]
            fitted = jobs[by_id[anchor]]
            assert by_id[anchor] < by_id[job["id"]]
            assert (fitted["kind"], fitted["window"], fitted["arm"], fitted["seed"]) == (
                "fit",
                job["anchor"],
                job["arm"],
                job["seed"],
            )
            assert (fitted["train"], fitted["validation"]) == (job["train"], job["validation"])
        else:
            assert job["depends"] == []
    assert Counter(job["engine"] for job in jobs if job["kind"] != "reference") == {
        "native_klpo": 144,
        "native_ppo": 576,
    }


def mutated(tmp_path, change_policies=None, change_stage=None):
    policies = json.loads(POLICIES.read_text())
    stage = json.loads(STAGES["A"].read_text())
    if change_policies:
        change_policies(policies)
    stage.update(
        campaign=str(Path("configs/baselines/historical-masked-campaign-a.json").resolve()),
        policies=str(tmp_path / "policies.json"),
    )
    if change_stage:
        change_stage(stage)
    atomic_json(tmp_path / "policies.json", policies)
    atomic_json(tmp_path / "stage.json", stage)
    return tmp_path / "stage.json"


def _policy(name, **values):
    return lambda v: v["policies"][name].update(values)


INVALID_POLICIES = {
    "seeds": lambda v: v.update(seeds=[42, 43]),
    "predictor_mae": lambda v: v["selection"].update(metric="session_mae"),
    "selection_on_evaluation": lambda v: v["selection"].update(partition="evaluation"),
    "early_stopping": lambda v: v["selection"].update(early_stopping=True),
    "primary_not_klpo": lambda v: v["contrasts"].update(primary="double_dqn"),
    "klpo_not_first": lambda v: v.update(policies=dict(reversed(list(v["policies"].items())))),
    "missing_control": lambda v: v["contrasts"]["controls"].pop(),
    "missing_reference": lambda v: v.update(references=["cash", "hold_initial"]),
    "unknown_reference": lambda v: v.update(references=["cash", "hold_initial", "oracle"]),
    "dqn_with_ppo_objective": _policy(
        "double_dqn", policy_objective={"schema_version": 1, "id": "ppo_clip_full_kl_v1"}
    ),
    "unknown_ppo_objective": _policy(
        "ppo_clip_full_kl", policy_objective={"schema_version": 1, "id": "ppo"}
    ),
    "missing_target_kl": _policy(
        "ppo_clip_kl_epoch_stop",
        policy_objective={"schema_version": 1, "id": "ppo_clip_kl_epoch_stop_v1"},
    ),
    "beta_outside_limits": lambda v: v["policies"]["ppo_kl_penalty_adaptive"][
        "policy_objective"
    ].update(beta_initial=1e7),
    "klpo_other_controller": _policy("klpo_terminal", controller="ppo_epochs"),
    "unknown_engine": _policy("double_dqn", engine="python"),
    "cost_outside_grid": lambda v: v["environment"].update(cost_bps=5),
    "no_ruin_penalty": lambda v: v["environment"].update(ruin_penalty=0),
    "lag_float": lambda v: v["environment"].update(dividend_payment_lag_sessions=1.5),
    "budget_float": lambda v: v["budget"].update(transitions=1.5e5),
    "rollout_over_budget": lambda v: v["budget"].update(rollout_transitions=10**6),
    "no_training_years": lambda v: v.update(train_windows=0),
    "unknown_universe_rule": lambda v: v["universe"].update(rule="best_in_evaluation"),
    "universe_too_large": lambda v: v["universe"].update(max_assets=5000),
    "predictor_without_producer": lambda v: v["predictor"].update(arms=["gru_episodic"]),
    "predictor_seed": lambda v: v["predictor"].update(seed=7),
    "test_opened": lambda v: v.update(final_test_opened=True),
    "extra_field": lambda v: v.update(extra=1),
    "status": lambda v: v.update(status="executed"),
}
INVALID_STAGES = {
    "unknown_scope": lambda v: v.update(scopes=["EU"]),
    "unordered_scopes": lambda v: v.update(scopes=["CN", "US"]),
    "limit": lambda v: v["limits"].update(max_training_jobs=719),
    "evaluation_limit": lambda v: v["limits"].update(max_evaluation_jobs=143),
    "test_opened": lambda v: v.update(final_test_opened=True),
}


# Motivo esperado de cada rechazo, para que una mutación no pase por otra comprobación.
REASONS = {
    "seeds": "semillas",
    "predictor_mae": "criterio de cartera",
    "selection_on_evaluation": "criterio de cartera",
    "early_stopping": "presupuesto fijo",
    "primary_not_klpo": "KLPO es el brazo principal",
    "klpo_not_first": "va primero",
    "missing_control": "con todos los demás",
    "missing_reference": "tres referencias",
    "unknown_reference": "tres referencias",
    "dqn_with_ppo_objective": "sin objetivo PPO",
    "unknown_ppo_objective": "objetivo PPO identificado",
    "missing_target_kl": "objetivo PPO identificado",
    "beta_outside_limits": "beta inicial",
    "klpo_other_controller": "controlador KLPO",
    "unknown_engine": "motor y una variante",
    "cost_outside_grid": "costes de evaluación",
    "no_ruin_penalty": "costes de evaluación",
    "lag_float": "costes de evaluación",
    "budget_float": "presupuesto de transiciones",
    "rollout_over_budget": "presupuesto de transiciones",
    "no_training_years": "ventanas de ajuste",
    "unknown_universe_rule": "universo",
    "universe_too_large": "universo",
    "predictor_without_producer": "productor",
    "predictor_seed": "semilla declarada",
    "test_opened": "contrato",
    "extra_field": "contrato",
    "status": "contrato",
    "unknown_scope": "ámbitos",
    "unordered_scopes": "ámbitos",
    "limit": "max_training_jobs=719",
    "evaluation_limit": "max_evaluation_jobs=143",
}


def test_the_unchanged_declaration_is_accepted(tmp_path):
    path = mutated(tmp_path)
    assert campaign_stage.check_stage(path)["counts"]["training_jobs"] == 720


@pytest.mark.parametrize("name", sorted(INVALID_POLICIES))
def test_policies_reject_inconsistent_declarations(tmp_path, name):
    with pytest.raises(ValueError, match=REASONS[name]):
        campaign_stage.check_stage(mutated(tmp_path, change_policies=INVALID_POLICIES[name]))


@pytest.mark.parametrize("name", sorted(INVALID_STAGES))
def test_stage_rejects_inconsistent_declarations(tmp_path, name):
    with pytest.raises(ValueError, match=REASONS[name]):
        campaign_stage.check_stage(mutated(tmp_path, change_stage=INVALID_STAGES[name]))


def test_selection_never_uses_the_error_of_the_predictor():
    assert all("mae" not in metric for metric in policy_plan.SELECTION_METRICS)
    policies = json.loads(POLICIES.read_text())
    for metric in ("session_mae", "mae", "validation_mae"):
        candidate = copy.deepcopy(policies)
        candidate["selection"]["metric"] = metric
        assert candidate["selection"]["metric"] not in policy_plan.SELECTION_METRICS


def test_later_stage_registers_both_variants_with_their_pending_capabilities():
    report = plan.check_campaign(Path("configs/baselines/historical-masked-campaign-b.json"))
    declared = report["later_stages"]["rl_policy_comparison"]
    assert declared == plan.LATER_STAGES["rl_policy_comparison"]
    assert Path(declared["config"]).resolve() == POLICIES.resolve()
    assert declared["issue"] == 137 and set(declared["stages"]) == set(plan.VARIANTS)
    assert set(declared["pending"]) <= set(campaign_stage.CAPABILITIES)
    # Las piezas sin sonda siguen pendientes hasta que exista su ejecutor.
    unprobed = {k for k, v in campaign_stage.CAPABILITIES.items() if v["probe"] is None}
    assert unprobed <= set(declared["pending"])
    module, _, function = declared["entry"].partition(":")
    assert getattr(importlib.import_module(module), function) is campaign_stage.run_stage
    for variant, path in declared["stages"].items():
        stage = campaign_stage.load_stage(path)
        assert stage["campaign"]["variant"] == variant
        assert stage["policies"]["path"] == str(POLICIES.resolve())
