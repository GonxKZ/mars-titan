"""Declaración y plan de la etapa de políticas, sin leer datos ni ajustar políticas.

Los recuentos se fijan por nivel con las configuraciones del repositorio. Las mutaciones
de la declaración deben fallar antes de planificar.
"""

import copy
import importlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.data.storage import atomic_json
from mars_titan.simulation import campaign_stage, native_policy_runs, policy_plan
from mars_titan.training import campaign_plan as plan

CONFIGS = Path("configs/simulation")
STAGES = {v: CONFIGS / f"historical-masked-rl-stage-{v.lower()}.json" for v in "AB"}
POLICIES = CONFIGS / "historical-masked-rl-policies.json"
ALGORITHM_ARMS = ("ppo_clip_full_kl", "ppo_kl_penalty_adaptive", "ppo_clip_kl_epoch_stop")
ALGORITHM_ARMS += ("double_dqn",)
REFERENCES = ("cash", "hold_initial", "rebalance_50", "equal_weight_monthly", "market_index")
# El índice de mercado solo se planifica donde la edición tiene su instrumento (SPY).
INDEXED = {"US": True, "CN": False}
COSTS = 4
# Brazos con productor en las campañas A y B, en el orden de su configuración.
PRODUCERS = ["rnn", "lstm", "gru", "dlinear", "transformer_compact", "ridge", "xgboost"]
PRODUCERS += ["titans_transformer_direct", "titans_mac_disabled", "titans_mac_frozen"]
PRODUCERS += ["titans_mac_online"]
COMPARED = ["transformer_compact", "titans_mac_online"]
# Ajustes, traslados y referencias por nivel en todos los ámbitos.
LEVELS = dict(
    A=dict(all_predictors=(792, 0, 1221), algorithms=(576, 0, 0)),
    B=dict(all_predictors=(264, 528, 1221), algorithms=(192, 384, 0)),
)
# Ventanas de política por ámbito: las del protocolo menos las cuatro primeras, que
# aportan los tres años de ajuste y el de validación de la primera política.
WINDOWS = {"US": 15, "CN": 9}
ANCHORS = dict(A={"US": 15, "CN": 9}, B={"US": 5, "CN": 3})


def us(day):
    return int(np.datetime64(day, "us").astype(np.int64))


@pytest.mark.parametrize("variant", "AB")
def test_repository_stages_count_every_level_window_predictor_arm_seed_and_reference(variant):
    result = campaign_stage.check_stage(STAGES[variant])
    counts = result["counts"]
    fits = sum(value[0] for value in LEVELS[variant].values())
    carries = sum(value[1] for value in LEVELS[variant].values())
    assert (counts["training_jobs"], counts["carried_jobs"]) == (fits, carries)
    # Cinco referencias en las 15 ventanas de EE. UU. y cuatro en las 9 de China.
    assert counts["reference_jobs"] == (15 * 5 + 9 * 4) * 11 == 1221
    assert counts["evaluation_jobs"] == carries + 1221
    assert counts["evaluation_episodes"] == (fits + carries + 1221) * COSTS == 10356
    for level, (fit, carry, reference) in LEVELS[variant].items():
        entry = counts["levels"][level]
        assert (entry["training_jobs"], entry["carried_jobs"]) == (fit, carry)
        assert entry["reference_jobs"] == reference
        assert entry["evaluation_episodes"] == (fit + carry + reference) * COSTS
    assert counts["levels"]["all_predictors"]["predictors"] == PRODUCERS
    assert counts["levels"]["algorithms"]["predictors"] == COMPARED
    stage = campaign_stage.load_stage(STAGES[variant])
    assert stage["limits"] == dict(max_training_jobs=fits, max_evaluation_jobs=carries + 1221)
    for scope, total in WINDOWS.items():
        entry = counts["scopes"][scope]
        anchors = ANCHORS[variant][scope]
        assert len(entry["windows"]) == total and len(entry["anchors"]) == anchors
        assert entry["carried_windows"] == total - anchors
        # KLPO y las referencias sobre los 11 predictores y cuatro políticas más sobre dos.
        assert entry["training_jobs"] == anchors * (11 + 2 * 4) * 3
        assert entry["carried_jobs"] == (total - anchors) * (11 + 2 * 4) * 3
        references = [r for r in REFERENCES if INDEXED[scope] or r != "market_index"]
        assert entry["reference_jobs"] == total * 11 * len(references)
        assert set(entry["arms"]) == {"klpo_terminal", *ALGORITHM_ARMS, *references}
        for arm, predictors in (("klpo_terminal", 11), *((arm, 2) for arm in ALGORITHM_ARMS)):
            fit = dict(fit=anchors * predictors) if anchors else {}
            carry = dict(carry=(total - anchors) * predictors) if total > anchors else {}
            assert entry["arms"][arm] == {str(s): fit | carry for s in (42, 43, 44)}
        for arm in references:
            assert entry["arms"][arm] == {"none": dict(reference=total * 11)}
    assert result["contrasts"]["primary"] == "klpo_terminal"
    assert result["universe_predictor"] == "transformer_compact"
    assert result["selection"]["metric"] == "ruin_count_then_mean_liquidated_log_growth"
    assert result["seeds"] == [42, 43, 44] and result["budget"]["transitions"] == 262144
    assert result["scientific_training_started"] is False and result["final_test_opened"] is False


@pytest.mark.parametrize("variant", "AB")
def test_levels_resolve_every_producer_of_the_campaign(variant):
    stage = campaign_stage.load_stage(STAGES[variant])
    assert stage["predictors"] == [spec["arm"] for spec in plan._arm_specs(stage["campaign"])]
    assert stage["predictors"] == PRODUCERS
    jobs = policy_plan.plan_stage(stage)
    for job in jobs:
        if job["level"] == "algorithms":
            assert job["predictor"] in COMPARED and job["arm"] in ALGORITHM_ARMS
        else:
            assert job["arm"] in ("klpo_terminal", *REFERENCES)
    covered = {job["predictor"] for job in jobs if job["arm"] == "klpo_terminal"}
    assert covered == {job["predictor"] for job in jobs if job["arm"] == "cash"} == set(PRODUCERS)


def campaign_with_candidate(tmp_path, variant):
    """Campaña del repositorio con la sección de la GRU candidata, solo para planificar."""
    folder = Path("configs/baselines").resolve()
    campaign = json.loads(
        (folder / f"historical-masked-campaign-{variant.lower()}.json").read_text()
    )
    campaign["comparison"] = str((folder / campaign["comparison"]).resolve())
    campaign["tabular"]["config"] = str((folder / campaign["tabular"]["config"]).resolve())
    campaign["titans_mac"]["recipe"] = str((folder / campaign["titans_mac"]["recipe"]).resolve())
    recipe = Path("configs/candidate/chronological-training.json").resolve()
    principal = json.loads(recipe.read_text())["principal"]
    campaign["episodic_gru"] = dict(
        recipe=str(recipe), arms={"gru_episodic": principal}, search_seed=42
    )
    campaign["limits"]["max_training_jobs"] = 100_000
    campaign["limits"]["max_prediction_jobs"] = 100_000
    atomic_json(tmp_path / "campaign.json", campaign)
    return tmp_path / "campaign.json"


def test_a_producer_registered_in_the_campaign_enters_the_first_level_without_changes(tmp_path):
    def with_candidate(stage):
        stage["campaign"] = str(campaign_with_candidate(tmp_path, "A"))
        stage["limits"] = dict(max_training_jobs=100_000, max_evaluation_jobs=100_000)

    result = campaign_stage.check_stage(mutated(tmp_path, change_stage=with_candidate))
    levels = result["counts"]["levels"]
    assert levels["all_predictors"]["predictors"] == [
        *PRODUCERS[:7],
        "gru_episodic",
        *PRODUCERS[7:],
    ]
    # La candidata añade 24 ventanas por tres semillas de KLPO y sus referencias: cinco en
    # las 15 ventanas de EE. UU. y cuatro en las 9 de China, sin instrumento del índice.
    assert levels["all_predictors"]["training_jobs"] == 792 + 24 * 3
    assert levels["all_predictors"]["reference_jobs"] == 1221 + 15 * 5 + 9 * 4
    assert levels["algorithms"]["training_jobs"] == 576


def test_every_first_level_fit_shares_the_budget_and_the_portfolio_criterion():
    stage = campaign_stage.load_stage(STAGES["A"])
    jobs = [job for job in policy_plan.plan_stage(stage) if job["kind"] == "fit"]
    # Tres cintas anuales de ajuste: KLPO declara las oleadas completas que caben en el
    # presupuesto común y no puede superarlo.
    tapes = SimpleNamespace(failure=None, unfit=None, train=(range(253),) * 3)
    budget = stage["policies"]["budget"]["transitions"]
    environments = stage["policies"]["budget"]["environments"]
    waves, wave = native_policy_runs.klpo_waves(tapes.train, environments, budget)
    assert 0 < waves * wave <= budget
    selection = dict(metric="ruin_count_then_mean_liquidated_log_growth", partition="validation")
    equity = dict(basis="close_valuation", close_times=[1, 2], nav=[1.0, 1.0])
    records = [
        dict(cost_bps=cost, status="completed", reason=None, steps=1, net_return=0.0)
        | dict(liquidated_net_return=0.0, max_drawdown=0.0, equity=equity)
        for cost in stage["policies"]["evaluation_costs_bps"]
    ]
    for predictor in PRODUCERS:
        job = next(j for j in jobs if j["predictor"] == predictor and j["arm"] == "klpo_terminal")
        report = dict(status="completed", transitions=waves * wave, waves=waves, updates=1)
        report.update(selection=selection, policy=dict(id=job["id"], sha256="a" * 64))
        report.update(evaluation=records)
        campaign_stage.check_report(stage, job, report, tapes)
        for change in (
            dict(waves=waves - 1),
            dict(transitions=budget + 1),
            dict(selection=dict(selection, metric="mae")),
        ):
            with pytest.raises(ValueError, match="presupuesto, el criterio de cartera"):
                campaign_stage.check_report(stage, job, dict(report, **change), tapes)


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
    for (scope, _, predictor), group in groups.items():
        # KLPO primero, las políticas de algoritmos si el predictor se compara y las referencias.
        learned = ["klpo_terminal", *(ALGORITHM_ARMS if predictor in COMPARED else ())]
        expected = [arm for arm in learned for _ in (42, 43, 44)]
        references = [r for r in REFERENCES if INDEXED[scope] or r != "market_index"]
        assert [job["arm"] for job in group] == [*expected, *references]
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
        "native_klpo": 264 + 528,
        "native_ppo": 192 + 384,
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


def _klpo_second(value):
    # Double DQN pasa delante de KLPO y los controles siguen ese orden declarado.
    items = list(value["policies"].items())
    value["policies"] = dict([items[-1], *items[:-1]])
    value["contrasts"]["controls"] = [*list(value["policies"])[1:], *value["references"]]


INVALID_POLICIES = {
    "seeds": lambda v: v.update(seeds=[42, 43]),
    "predictor_mae": lambda v: v["selection"].update(metric="session_mae"),
    "selection_on_evaluation": lambda v: v["selection"].update(partition="evaluation"),
    "early_stopping": lambda v: v["selection"].update(early_stopping=True),
    "primary_not_klpo": lambda v: v["contrasts"].update(primary="double_dqn"),
    "klpo_not_first": lambda v: v.update(policies=dict(reversed(list(v["policies"].items())))),
    "klpo_second_with_matching_controls": _klpo_second,
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
    "cost_outside_grid": lambda v: v["environment"].update(cost_bps=15),
    "unordered_costs": lambda v: v.update(evaluation_costs_bps=[0, 10, 5, 20]),
    "too_many_costs": lambda v: v.update(evaluation_costs_bps=list(range(17))),
    "old_schema": lambda v: v.update(schema_version=1),
    "index_path": lambda v: v["market_index"].update(US="../SPY"),
    "index_unknown_market": lambda v: v["market_index"].update(EU="STOXX"),
    "report_cost_outside_grid": lambda v: v["report"].update(primary_cost_bps=25),
    "report_block_in_sensitivity": lambda v: v["report"].update(
        block_length_sensitivity=[5, 21, 63]
    ),
    "report_few_replicates": lambda v: v["report"].update(replicates=10),
    "report_integer_confidence": lambda v: v["report"].update(confidence=1),
    "report_unknown_benchmark": lambda v: v["report"]["benchmarks"].update(CN="oracle"),
    "report_benchmark_other_market": lambda v: v["report"]["benchmarks"].update(
        US="csi300_price_index_v1"
    ),
    "report_benchmark_and_index": lambda v: v["market_index"].update(CN="SPY"),
    "survival_primary": lambda v: v["survival_sensitivity"].update(role="primary"),
    "survival_positive_exit": lambda v: v["survival_sensitivity"].update(exit_returns=[0.1]),
    "survival_integer_exit": lambda v: v["survival_sensitivity"].update(exit_returns=[0]),
    "survival_other_rule": lambda v: v["survival_sensitivity"].update(applies_to="all_assets"),
    "no_ruin_penalty": lambda v: v["environment"].update(ruin_penalty=0),
    "lag_float": lambda v: v["environment"].update(dividend_payment_lag_sessions=1.5),
    "budget_float": lambda v: v["budget"].update(transitions=1.5e5),
    "rollout_over_budget": lambda v: v["budget"].update(rollout_transitions=10**6),
    "no_training_years": lambda v: v.update(train_windows=0),
    "unknown_universe_rule": lambda v: v["universe"].update(rule="best_in_evaluation"),
    "universe_too_large": lambda v: v["universe"].update(max_assets=5000),
    "predictor_without_producer": lambda v: v["levels"]["algorithms"].update(
        predictors=["gru_episodic"]
    ),
    "predictor_seed": lambda v: v["predictor"].update(seed=7),
    "fixed_first_level": lambda v: v["levels"]["all_predictors"].update(predictors=COMPARED),
    "first_level_without_klpo": lambda v: v["levels"]["all_predictors"]["arms"].pop(0),
    "klpo_in_algorithms": lambda v: v["levels"]["algorithms"]["arms"].insert(0, "klpo_terminal"),
    "algorithm_missing": lambda v: v["levels"]["algorithms"]["arms"].pop(),
    "no_compared_predictor": lambda v: v["levels"]["algorithms"].update(predictors=[]),
    "repeated_predictor": lambda v: v["levels"]["algorithms"].update(predictors=["ridge", "ridge"]),
    "missing_level": lambda v: v["levels"].pop("algorithms"),
    "old_predictor_list": lambda v: v["predictor"].update(arms=COMPARED),
    "test_opened": lambda v: v.update(final_test_opened=True),
    "extra_field": lambda v: v.update(extra=1),
    "status": lambda v: v.update(status="executed"),
}
INVALID_STAGES = {
    "unknown_scope": lambda v: v.update(scopes=["EU"]),
    "unordered_scopes": lambda v: v.update(scopes=["CN", "US"]),
    "limit": lambda v: v["limits"].update(max_training_jobs=1367),
    "evaluation_limit": lambda v: v["limits"].update(max_evaluation_jobs=1220),
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
    "klpo_second_with_matching_controls": "va primero",
    "missing_control": "con todos los demás",
    "missing_reference": "cinco referencias",
    "unknown_reference": "cinco referencias",
    "dqn_with_ppo_objective": "sin objetivo PPO",
    "unknown_ppo_objective": "objetivo PPO identificado",
    "missing_target_kl": "objetivo PPO identificado",
    "beta_outside_limits": "beta inicial",
    "klpo_other_controller": "controlador KLPO",
    "unknown_engine": "motor y una variante",
    "cost_outside_grid": "costes de evaluación",
    "unordered_costs": "costes de evaluación",
    "too_many_costs": "costes de evaluación",
    "old_schema": "contrato",
    "index_path": "instrumento de la edición",
    "index_unknown_market": "instrumento de la edición",
    "report_cost_outside_grid": "coste principal",
    "report_block_in_sensitivity": "coste principal",
    "report_few_replicates": "coste principal",
    "report_integer_confidence": "coste principal",
    "report_unknown_benchmark": "coste principal",
    "report_benchmark_other_market": "coste principal",
    "report_benchmark_and_index": "coste principal",
    "survival_primary": "supervivencia es secundaria",
    "survival_positive_exit": "supervivencia es secundaria",
    "survival_integer_exit": "supervivencia es secundaria",
    "survival_other_rule": "supervivencia es secundaria",
    "no_ruin_penalty": "costes de evaluación",
    "lag_float": "costes de evaluación",
    "budget_float": "presupuesto de transiciones",
    "rollout_over_budget": "presupuesto de transiciones",
    "no_training_years": "ventanas de ajuste",
    "unknown_universe_rule": "universo",
    "universe_too_large": "universo",
    "predictor_without_producer": "productor",
    "predictor_seed": "semilla declarada",
    "fixed_first_level": "nivel completo",
    "first_level_without_klpo": "nivel completo",
    "klpo_in_algorithms": "nivel completo",
    "algorithm_missing": "nivel completo",
    "no_compared_predictor": "nivel completo",
    "repeated_predictor": "nivel completo",
    "missing_level": "nivel completo",
    "old_predictor_list": "predictor, el universo",
    "test_opened": "contrato",
    "extra_field": "contrato",
    "status": "contrato",
    "unknown_scope": "ámbitos",
    "unordered_scopes": "ámbitos",
    "limit": "max_training_jobs=1367",
    "evaluation_limit": "max_evaluation_jobs=1220",
}


def test_the_unchanged_declaration_is_accepted(tmp_path):
    path = mutated(tmp_path)
    assert campaign_stage.check_stage(path)["counts"]["training_jobs"] == 1368


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


def test_the_declared_capital_buys_an_a_share_lot_in_every_selected_asset():
    from mars_titan.simulation.environment import ACTIONS
    from mars_titan.simulation.market_rules import china_a_share_instrument

    policies = json.loads(POLICIES.read_text())
    # El entorno reparte la exposición a partes iguales en el cuartil superior de puntuaciones
    # positivas. Con la menor exposición positiva y el universo completo, cada activo debe
    # recibir al menos un lote de 100 acciones A a 50 CNY, o China apenas podría operar.
    exposure = min(level for level in ACTIONS if level)
    selected = -(-policies["universe"]["max_assets"] // 4)
    per_asset = policies["environment"]["capital"] * exposure / selected
    lot = china_a_share_instrument("CN/600000.SS").lot
    assert per_asset >= lot * 50
    # Con el capital anterior de 10 000 no llegaba a un lote ni a 1 CNY por acción.
    assert 10_000 * exposure / selected < lot


def test_later_stage_registers_both_variants_with_their_pending_capabilities():
    report = plan.check_campaign(Path("configs/baselines/historical-masked-campaign-b.json"))
    declared = report["later_stages"]["rl_policy_comparison"]
    assert declared == plan.LATER_STAGES["rl_policy_comparison"]
    assert Path(declared["config"]).resolve() == POLICIES.resolve()
    assert declared["issue"] == 137 and set(declared["stages"]) == set(plan.VARIANTS)
    assert set(declared["pending"]) <= set(campaign_stage.CAPABILITIES)
    # Pendientes son justo las piezas sin sonda, que aún no existen. Una capacidad con sonda
    # ya está implementada y se comprueba con el motor instalado, como las reglas A (#406).
    unprobed = {k for k, v in campaign_stage.CAPABILITIES.items() if v["probe"] is None}
    assert set(declared["pending"]) == unprobed
    assert len(declared["pending"]) == len(unprobed)
    module, _, function = declared["entry"].partition(":")
    assert getattr(importlib.import_module(module), function) is campaign_stage.run_stage
    for variant, path in declared["stages"].items():
        stage = campaign_stage.load_stage(path)
        assert stage["campaign"]["variant"] == variant
        assert stage["policies"]["path"] == str(POLICIES.resolve())
