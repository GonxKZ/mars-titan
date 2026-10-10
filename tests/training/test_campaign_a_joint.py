"""Campaña A v2: modelo conjunto US+CN, controles separados y semillas del caso elegido.

Las pruebas del plan no leen vistas ni datos. Las de extremo a extremo preparan vistas sobre
el corpus técnico y sustituyen los ejecutores por dobles que escriben predicciones nulas con
las filas exactas de cada vista. Ningún modelo se ajusta, no se aplican pasos de optimizador
y la GPU no se usa.
"""

import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.evaluation.splits import build_folds, eligible_folds
from mars_titan.training import campaign_budget as budget
from mars_titan.training import campaign_data_policy as data_policy
from mars_titan.training import campaign_numerics as numerics
from mars_titan.training import campaign_plan as plan
from mars_titan.training import campaign_schedule as order
from mars_titan.training import masked_campaign as engine
from mars_titan.training.joint_temporal_corpus import prepare_joint_temporal_corpus
from tests.training.test_masked_campaign import Recorder, doubles, run
from tests.training.test_walk_forward_v2_views import expected, fixture, observed

CONFIGS = Path("configs")
EVALUATION = CONFIGS / "evaluation"
CAMPAIGN = CONFIGS / "baselines/historical-masked-campaign-a-v2.json"
COMPARISON = EVALUATION / "historical-masked-2000-joint-comparison.json"
JOINT_CN = EVALUATION / "historical-masked-joint-cn-walk-forward-v3.json"
US_V2 = EVALUATION / "historical-masked-us-walk-forward-v2.json"
CN_V2 = EVALUATION / "historical-masked-cn-walk-forward-v2.json"
JOINT_STOP = CONFIGS / "baselines/historical-masked-campaign-a-joint-stop.json"
CONTROLS = ("transformer_compact", "titans_mac_online", "mars_titan_m1")
PAIRED = ("rnn", "lstm", "gru", "dlinear", "transformer_compact", "titans_transformer_direct")


def campaign():
    return plan.load_campaign(CAMPAIGN)


def edited(tmp_path, change, *, comparison_change=None):
    """Copia de la campaña v2 con un cambio, con rutas absolutas a sus dependencias."""
    folder = tmp_path / "configs"
    (folder / "baselines").mkdir(parents=True)
    (folder / "evaluation").mkdir(parents=True, exist_ok=True)
    declared = json.loads(COMPARISON.read_text())
    for scope in declared["scopes"].values():
        scope["protocols"] = {
            market: str((EVALUATION / name).resolve())
            for market, name in scope["protocols"].items()
        }
    declared["joint_design"]["market_eligibility"] = {
        market: str((EVALUATION / name).resolve())
        for market, name in declared["joint_design"]["market_eligibility"].items()
    }
    if comparison_change:
        comparison_change(declared)
    atomic_json(folder / "evaluation/comparison.json", declared)
    value = json.loads(CAMPAIGN.read_text())
    value["comparison"] = "../evaluation/comparison.json"
    for section, key in (
        ("tabular", "config"),
        ("titans_mac", "recipe"),
        ("episodic_gru", "recipe"),
        ("mars_titan", "recipe"),
        ("cm_v1", "declaration"),
    ):
        value[section][key] = str((CAMPAIGN.parent / value[section][key]).resolve())
    change(value)
    atomic_json(folder / "baselines/campaign.json", value)
    return folder / "baselines/campaign.json"


# Plan sin datos


def test_v2_plans_the_joint_model_for_every_arm_and_three_separate_controls():
    loaded = campaign()
    counts = plan.count_jobs(loaded)
    assert (counts["training_jobs"], counts["prediction_jobs"]) == (2341, 0)
    joint, us, cn = (counts["scopes"][scope] for scope in ("US+CN", "US", "CN"))
    assert (joint["windows"], us["windows"], cn["windows"]) == (19, 19, 13)
    assert (joint["training_jobs"], us["training_jobs"], cn["training_jobs"]) == (1957, 228, 156)
    # El control en línea solo se evalúa, y por tanto solo se ajusta, en el ámbito conjunto.
    assert set(us["arms"]) == set(cn["arms"]) == set(CONTROLS)
    assert plan.scope_arms(loaded, "US") == plan.scope_arms(loaded, "CN") == list(CONTROLS)
    # Cinco tasas con la semilla 42 y la elegida repetida con 43 y 44 en cada ventana.
    assert joint["arms"]["transformer_compact_online"] == {
        "42": dict(fit=0, carry=0, online=5 * 19),
        "43": dict(fit=0, carry=0, online=19),
        "44": dict(fit=0, carry=0, online=19),
    }
    for record, windows in ((joint, 19), (us, 19), (cn, 13)):
        for arm in record["arms"]:
            if arm in {"ridge", "xgboost", "transformer_compact_online"}:
                continue
            assert record["arms"][arm] == {
                "42": dict(fit=2 * windows, carry=0),
                "43": dict(fit=windows, carry=0),
                "44": dict(fit=windows, carry=0),
            }, arm
    # Los tabulares son deterministas: sus casos solo se ajustan con la semilla de búsqueda.
    assert joint["arms"]["ridge"] == {"42": dict(fit=3 * 19, carry=0)}
    assert joint["arms"]["xgboost"] == {"42": dict(fit=12 * 19, carry=0)}
    assert {"cm_v1_core_b", "cm_v1_core_c"} <= set(joint["arms"])
    assert loaded["limits"]["max_training_jobs"] == counts["training_jobs"]


def test_extra_seeds_repeat_only_the_selected_case_after_every_search_of_the_scope():
    loaded = campaign()
    jobs = plan.plan_campaign(loaded)
    position = {job["id"]: index for index, job in enumerate(jobs)}
    groups = {}
    for job in jobs:
        groups.setdefault((job["scope"], job["window"], job["arm"]), []).append(job)
        # Orden topológico: ninguna dependencia aparece después del trabajo.
        assert all(position[dep] < position[job["id"]] for dep in job["depends"]), job["id"]
        # Ninguna dependencia cruza de ámbito.
        assert all(dep.split("/")[0] == job["scope"] for dep in job["depends"]), job["id"]
    for (scope, window, arm), members in groups.items():
        searches = sorted(job["id"] for job in members if job["stage"] == "search")
        finalists = [job for job in members if job["stage"] == "finalist"]
        assert all(job["seed"] == 42 for job in members if job["stage"] == "search")
        assert sorted(job["seed"] for job in finalists) == (
            [] if arm in {"ridge", "xgboost"} else [43, 44]
        )
        for job in finalists:
            assert set(searches) <= set(job["depends"]) and job["case"] is None
            if job.get("parent"):
                assert f"{scope}/{window}/{job['parent']}/finalist-s{job['seed']}" in job["depends"]
    # El control MARS-TITAN separado parte del Titans-MAC separado de su propio ámbito.
    finalist = next(j for j in jobs if j["id"] == "US/fold-003/mars_titan_m1/finalist-s43")
    assert "US/fold-003/titans_mac_online/finalist-s43" in finalist["depends"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda v: v["seed_policy"].update(deterministic_arms=["ridge"]), "política de semillas"),
        (lambda v: v["seed_policy"].update(selected_case_seeds=[42, 43]), "repiten la de búsqueda"),
        (lambda v: v["seed_policy"].update(selected_case_seeds=[43]), "política de semillas"),
        (lambda v: v["seed_policy"].update(deterministic_arms=["ridge", "zero"]), "productor"),
        (lambda v: v["seed_policy"].update(search_seed=41), "política de semillas"),
        (lambda v: v.update(stopping={"mode": "joint"}), "modo de parada"),
        (lambda v: v.update(stopping={"mode": "early_stop"}), "modo de parada"),
        (lambda v: v.update(early_stop=json.loads(JOINT_STOP.read_text())["early_stop"]), "modo"),
        (lambda v: v["memory_options"]["titans_mac"].update(accumulation_rows=128), "receta"),
        (lambda v: v["memory_options"].pop("cm_v1"), "opciones de memoria"),
        (lambda v: v["memory_options"]["episodic_gru"].pop("recompute"), "opciones de memoria"),
        (lambda v: v.pop("seed_policy"), "contrato"),
        (lambda v: v.update(execution={"order": "random"}), "orden de ejecución"),
        (lambda v: v.pop("execution"), "contrato"),
        (lambda v: v["numerics"].update(cudnn_allow_tf32=True), "FP32 estricto"),
        (lambda v: v["numerics"].update(float32_matmul_precision="high"), "FP32 estricto"),
        (lambda v: v["numerics"].update(cuda_matmul_allow_tf32=0), "FP32 estricto"),
        (lambda v: v.pop("numerics"), "contrato"),
        (lambda v: v.update(data_policy="real_and_synthetic"), "real_edition_only"),
        (lambda v: v.pop("data_policy"), "contrato"),
    ],
    ids=[
        "xgboost_not_declared_deterministic",
        "repeat_includes_search_seed",
        "missing_repeat_seed",
        "unknown_deterministic_arm",
        "other_search_seed",
        "unknown_stopping_mode",
        "early_stop_mode_without_section",
        "early_stop_section_with_protocol_mode",
        "memory_value_differs_from_recipe",
        "missing_family_options",
        "missing_option",
        "v2_without_seed_policy",
        "unknown_execution_order",
        "v2_without_execution_order",
        "cudnn_tf32_allowed",
        "matmul_precision_high",
        "matmul_flag_not_boolean",
        "v2_without_numerics",
        "other_data_policy",
        "v2_without_data_policy",
    ],
)
def test_v2_rejects_declarations_that_break_the_seed_stopping_or_memory_rules(
    tmp_path, change, message
):
    with pytest.raises(ValueError, match=message):
        plan.load_campaign(edited(tmp_path, change))


def test_the_joint_stop_groups_keep_every_paired_contrast_inside_one_scope_and_group(tmp_path):
    def joint_stop(value):
        value.update(
            stopping={"mode": plan.EARLY_STOP},
            early_stop=json.loads(JOINT_STOP.read_text())["early_stop"],
        )

    loaded = plan.load_campaign(edited(tmp_path, joint_stop))
    # Los contrastes del diseño conjunto, también los del control en línea, siguen dentro
    # de los grupos de la parada conjunta (la carga los comprueba al planificar).
    jobs = plan.plan_campaign(loaded)
    by_id = {job["id"]: job for job in jobs}
    membership = loaded["early_stop"]["membership"]
    finals = [job for job in jobs if job.get("phase") == plan.JOINT]
    assert finals and plan.count_jobs(loaded, jobs)["training_jobs"] == len(
        [job for job in plan.plan_campaign(campaign()) if job["kind"] == plan.FIT]
    )
    for final in finals:
        members = [by_id[plateau] for plateau in final["joint_group"]]
        assert {(job["scope"], job["window"], job["seed"]) for job in members} == {
            (final["scope"], final["window"], final["seed"])
        }
        assert {membership[job["arm"]] for job in members} == {membership[final["arm"]]}
    # En un mercado solo paran juntos los controles separados de un mismo grupo.
    us = {
        tuple(sorted(by_id[plateau]["arm"] for plateau in final["joint_group"]))
        for final in finals
        if final["scope"] == "US" and final["stage"] == "search"
    }
    assert us == {("titans_mac_online", "transformer_compact"), ("mars_titan_m1",)}


def test_memory_options_block_the_launch_until_they_match_the_recipe(tmp_path):
    blockers = plan.launch_blockers(campaign())
    memory = [reason for reason in blockers if "medida de memoria" in reason]
    assert len(memory) == 4 and all("pendiente" in reason for reason in blockers)

    # Titans-MAC y los núcleos de CM-v1 comparten la receta cronológica, que ya declara la
    # acumulación medida en cuda:0. Un valor igual al de la receta deja de bloquear.
    recipe = json.loads(
        (CONFIGS / "titans/chronological-training-historical-masked.json").read_text()
    )
    rows = recipe["recipe"]["accumulation_rows"]

    def fixed(value):
        value["memory_options"]["titans_mac"]["accumulation_rows"] = rows
        value["memory_options"]["cm_v1"]["accumulation_rows"] = rows

    loaded = plan.load_campaign(edited(tmp_path, fixed))
    # La regla del control en línea ya tiene sus valores y no bloquea.
    assert [reason.split(".")[0] for reason in plan.launch_blockers(loaded)] == [
        "episodic_gru",
        "episodic_gru",
    ]
    report = plan.check_campaign(CAMPAIGN)
    assert report["launch_blockers"] == blockers
    assert report["scope_arms"]["US"] == list(CONTROLS)
    assert report["stopping"] == {"mode": "protocol"}
    assert (
        plan.check_campaign(CONFIGS / "baselines/historical-masked-campaign-a.json")[
            "launch_blockers"
        ]
        == []
    )


def test_run_refuses_a_campaign_with_pending_memory_options_before_reading_views(tmp_path):
    with pytest.raises(ValueError, match="no se puede lanzar"):
        engine.run_campaign(CAMPAIGN, {}, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_separate_control_needs_its_parent_in_the_same_scope(tmp_path):
    def only_reader(declared):
        declared["joint_design"]["separate_controls"] = ["mars_titan_m1"]

    path = edited(tmp_path, lambda value: None, comparison_change=only_reader)
    with pytest.raises(ValueError, match="faltan padres: mars_titan_m1 sin titans_mac_online"):
        plan.load_campaign(path)


# Orden ventana a ventana


def test_campaign_windows_join_the_scopes_with_identical_partitions():
    rows = order.campaign_windows(campaign())
    assert [row["id"] for row in rows] == [f"fold-{i:03d}" for i in range(19)]
    assert all(row["scopes"] == {"US+CN": row["id"], "US": row["id"]} for row in rows[:6])
    assert rows[6]["scopes"] == {"US+CN": "fold-006", "US": "fold-006", "CN": "fold-000"}
    assert rows[18]["scopes"]["CN"] == "fold-012"
    with pytest.raises(ValueError, match="no es una ventana"):
        order.window_pairs(campaign(), "fold-019")


def test_v2_plan_finishes_each_window_of_every_scope_before_the_next():
    value = campaign()
    jobs = plan.plan_campaign(value)
    position = {
        pair: index
        for index, row in enumerate(order.campaign_windows(value))
        for pair in row["scopes"].items()
    }
    keys = [
        (position[job["scope"], job["window"]], order.PHASES.index(order.base_phase(job)))
        for job in jobs
    ]
    assert keys == sorted(keys)
    seen = set()
    for job in jobs:
        assert set(job["depends"]) <= seen
        seen.add(job["id"])
    # La versión 1 conserva su orden por ámbitos.
    v1 = plan.load_campaign(CONFIGS / "baselines/historical-masked-campaign-a.json")
    assert plan.execution_order(v1) == "by_scope"
    scopes = [job["scope"] for job in plan.plan_campaign(v1)]
    assert scopes == sorted(scopes, key=v1["scopes"].index)


def adapter_jobs(adapters):
    """Trabajos de la etapa de adaptadores v2 con las selecciones de su cadena."""
    stage = adapters.load_stage(plan.LATER_STAGES["posttraining_adapter_matrix"]["joint_stage"])
    jobs = adapters.plan_stage(stage)
    return jobs + adapters.plan_chain(stage, jobs)


def test_window_schedule_orders_every_stage_of_the_window_and_counts_all_jobs():
    from mars_titan.posttraining import campaign_stage as adapters
    from mars_titan.simulation import policy_plan
    from mars_titan.training import campaign_chain
    from mars_titan.training import modality_ablation_stage as ablation

    value = campaign()
    jobs = plan.plan_campaign(value)
    stages = dict(
        adapters=adapter_jobs(adapters),
        ablation=ablation.plan_stage(
            ablation.load_stage(plan.LATER_STAGES["modality_ablation"]["joint_stage"])
        ),
    )
    stage = policy_plan.load_stage(plan.LATER_STAGES["rl_policy_comparison"]["joint_stage"])
    stages["rl"] = policy_plan.plan_stage(stage)
    # La etapa de adaptadores da cadena a todos los predictores de las políticas, también la
    # trivial de Ridge y XGBoost, así que cada selección que leen está en el plan.
    chains = {job["id"] for job in stages["adapters"]}
    reads = {d for job in stages["rl"] for d in job["depends"] if campaign_chain.CHAIN_SUFFIX in d}
    assert reads and reads <= chains
    assert {job["base_arm"] for job in stages["adapters"]} == set(stage["predictors"])
    schedule = order.window_schedule(value, jobs, stages)
    assert [row["window"] for row in schedule] == [f"fold-{i:03d}" for i in range(19)]
    assert [entry["phase"] for entry in schedule[0]["phases"]] == list(order.PHASES)
    totals = Counter()
    for row in schedule:
        totals.update({entry["phase"]: len(entry["jobs"]) for entry in row["phases"]})
    assert totals["base_search"] + totals["selected_case_seeds"] == 2341
    # Adaptadores: 6.588 ajustes y 1.116 padres congelados, y 1.178 selecciones de la cadena
    # en su propia fase.
    assert (totals["adapters"], totals["chain"]) == (6588 + 1116, 1178)
    assert (totals["online"], totals["ablation"], totals["rl"]) == (133, 3534, 2160 + 2442)
    window = schedule[6]
    selection = window["phases"][order.PHASES.index("selection")]
    assert "CN/fold-000/mars_titan_m1" in selection["decisions"]
    assert window["phases"][order.PHASES.index("comparison")]["scopes"] == window["scopes"]


def test_window_schedule_and_order_reject_dependencies_on_later_work():
    value = campaign()
    jobs = [dict(job) for job in plan.plan_campaign(value)]
    first = next(job for job in jobs if job["window"] == "fold-000")
    later = next(job for job in jobs if job["window"] == "fold-001")
    first["depends"] = [later["id"]]
    with pytest.raises(ValueError, match="fase posterior"):
        order.window_schedule(value, jobs)
    with pytest.raises(ValueError, match="antes que una de sus dependencias"):
        order.order_by_window(value, jobs)
    with pytest.raises(ValueError, match="aparece dos veces"):
        order.window_schedule(value, jobs[:1], dict(ablation=jobs[:1]))


@pytest.mark.parametrize("module", ["posttraining", "simulation", "ablation"])
def test_later_stages_confirm_only_the_base_of_the_window_they_run(module):
    from mars_titan.posttraining import campaign_stage as adapters
    from mars_titan.simulation import campaign_stage as policies
    from mars_titan.training import modality_ablation_stage as ablation

    value = campaign()
    stage = dict(scopes=["US+CN"], families={"gru": "neural"}, predictors=["gru"], arms=["gru"])
    stage["campaign"] = value
    confirm = dict(posttraining=adapters, simulation=policies, ablation=ablation)[module]
    seen = []
    base = SimpleNamespace(
        receipts={},
        resolve=lambda job: (None, None, None),
        job_identity=lambda job, case, sources: job["id"],
        confirmed=lambda job, identity: seen.append(job) or {"id": identity},
    )
    pairs = order.window_pairs(value, "fold-006")
    confirm._base_receipts(base, value, stage, pairs)
    assert seen and all((job["scope"], job["window"]) == ("US+CN", "fold-006") for job in seen)


def test_a_staged_window_runs_its_jobs_and_confirms_the_base_of_the_parent_window():
    from mars_titan.posttraining import campaign_stage as adapters

    stage = adapters.load_stage(plan.LATER_STAGES["posttraining_adapter_matrix"]["joint_stage"])
    jobs = adapters.plan_stage(stage)
    chains = adapters.plan_chain(stage, jobs)
    selected, chosen, pairs = adapters.window_plan(stage["campaign"], jobs, chains, "fold-006")
    assert selected and chosen
    assert {(job["scope"], job["window"]) for job in selected + chosen} == {("US+CN", "fold-006")}
    # Los trabajos parten del estado elegido en fold-005, que es un trabajo de la base.
    assert pairs == {("US+CN", "fold-006"), ("US+CN", "fold-005")}
    assert len(adapters.ordered_jobs(selected, chosen)) == len(selected) + len(chosen)
    first = adapters.window_plan(stage["campaign"], jobs, chains, "fold-000")
    assert not first[0] and first[1] and first[2] == {("US+CN", "fold-000")}


def test_window_jobs_follow_dependencies_through_every_level():
    def job(name, window, depends=()):
        return dict(id=name, scope="US+CN", window=window, depends=list(depends))

    chain = [job("c", "fold-000"), job("b", "fold-000", ["c"]), job("a", "fold-001", ["b"])]
    selected = order.window_jobs(campaign(), chain, "fold-001")
    assert [item["id"] for item in selected] == ["c", "b", "a"]
    with pytest.raises(ValueError, match="no está en el plan"):
        order.window_jobs(campaign(), chain[1:], "fold-001")


def test_a_carried_window_keeps_its_anchor_and_the_dependencies_it_needs():
    variant_b = plan.load_campaign(CONFIGS / "baselines/historical-masked-campaign-b.json")
    jobs = plan.plan_campaign(variant_b)
    # En B, fold-001 de US se traslada desde el ancla fold-000.
    selected = order.window_jobs(variant_b, jobs, "fold-001")
    windows = {(job["scope"], job["window"]) for job in selected}
    assert ("US", "fold-001") in windows and ("US", "fold-000") in windows
    carried = [job for job in selected if job["window"] == "fold-001"]
    assert carried and all(job["kind"] == "carry" for job in carried)
    stage_jobs, needed = order.stage_window(
        variant_b,
        [dict(id="US/fold-001/x", scope="US", window="fold-001", anchor="fold-000", depends=[])],
        "fold-001",
    )
    assert needed == {("US", "fold-001"), ("US", "fold-000")} and len(stage_jobs) == 1


def test_later_stages_of_v2_read_the_joint_model_in_each_market():
    from mars_titan.posttraining import campaign_stage as adapters
    from mars_titan.simulation import policy_plan
    from mars_titan.training import modality_ablation_stage as ablation

    adapter = adapters.load_stage(plan.LATER_STAGES["posttraining_adapter_matrix"]["joint_stage"])
    assert adapter["scopes"] == ["US+CN"]
    counts = adapters.count_stage(adapter)
    # 20 brazos neuronales con tres semillas y los dos tabulares con una: 62 cadenas por
    # ventana. Cada ventana con padre congela esos 62 padres y todas eligen su cadena.
    assert (counts["training_jobs"], counts["prediction_jobs"], counts["selection_jobs"]) == (
        6588,
        62 * 18,
        62 * 19,
    )
    assert set(adapter["arms"]) == set(
        policy_plan.load_stage(plan.LATER_STAGES["rl_policy_comparison"]["joint_stage"])[
            "predictors"
        ]
    )
    masked = ablation.load_stage(plan.LATER_STAGES["modality_ablation"]["joint_stage"])
    # 20 brazos neuronales con tres semillas y dos tabulares con una, por tres variantes.
    assert ablation.count_stage(masked)["prediction_jobs"] == (20 * 3 + 2) * 3 * 19
    stage = policy_plan.load_stage(plan.LATER_STAGES["rl_policy_comparison"]["joint_stage"])
    assert len(stage["predictors"]) == 22 and stage["scopes"] == ["US+CN"]
    us = policy_plan.scope_windows(stage, "US+CN", "US")
    cn = policy_plan.scope_windows(stage, "US+CN", "CN")
    assert [row["window"] for row in us] == [f"fold-{i:03d}" for i in range(4, 19)]
    # China empieza con su propia historia mínima: la primera ventana elegible es fold-006.
    assert [row["window"] for row in cn] == [f"fold-{i:03d}" for i in range(10, 19)]
    assert cn[0]["train"] == ["fold-006", "fold-007", "fold-008"]
    counts = policy_plan.count_stage(stage)
    # Referencias: cuatro por ventana y predictor, y el índice de mercado solo en US.
    assert (counts["training_jobs"], counts["evaluation_jobs"]) == (2160, 22 * (15 * 5 + 9 * 4))
    assert counts["scopes"]["US+CN"]["markets"]["CN"][0] == "fold-010"
    jobs = policy_plan.plan_stage(stage)
    assert Counter(job["market"] for job in jobs if job["kind"] == "fit") == {
        "US": 15 * 30 * 3,
        "CN": 9 * 30 * 3,
    }


# Comparación sin datos


def test_joint_comparison_counts_china_only_with_its_minimum_history():
    config = comparison.load_config(COMPARISON)
    joint = config["resolved_scopes"]["US+CN"]
    assert joint["eligible"]["US"] == [f"fold-{i:03d}" for i in range(19)]
    assert joint["eligible"]["CN"] == [f"fold-{i:03d}" for i in range(6, 19)]
    first = joint["windows"]["fold-006"]
    assert first["evaluation"] == ["2011-01-01", "2012-01-01"]
    assert set(joint["arms"]) == set(config["arms"]) and joint["borrowed"] == {}
    for scope, offset in (("US", 0), ("CN", 6)):
        resolved = config["resolved_scopes"][scope]
        assert resolved["joint_windows"] == {
            window: f"fold-{int(window[5:]) + offset:03d}" for window in resolved["windows"]
        }
        assert resolved["borrowed"] == {f"{arm}_joint": arm for arm in CONTROLS}
        assert set(resolved["arms"]) == {"zero", *CONTROLS, *resolved["borrowed"]}
        family = resolved["families"]["joint_vs_separate"]
        # Diferencia conjunto menos separado: el separado es la base.
        assert family["transformer_compact_joint-transformer_compact"] == {
            "transformer_compact": -1.0,
            "transformer_compact_joint": 1.0,
        }
    assert config["arms"]["xgboost"]["seeds"] == [42] and config["arms"]["ridge"]["seeds"] == [42]


def test_eligibility_keeps_only_folds_with_identical_partitions():
    us = json.loads(US_V2.read_text())
    joint_cn = json.loads(JOINT_CN.read_text())
    cn = json.loads(CN_V2.read_text())
    assert {k: v for k, v in joint_cn.items() if k != "market"} == {
        k: v for k, v in us.items() if k != "market"
    }
    assert eligible_folds(joint_cn, cn) == [f"fold-{i:03d}" for i in range(6, 19)]
    assert eligible_folds(us, us) == [fold["id"] for fold in build_folds(us)]
    with pytest.raises(ValueError, match="mismo mercado"):
        eligible_folds(joint_cn, us)
    with pytest.raises(ValueError, match="Ninguna ventana"):
        eligible_folds(joint_cn, cn | dict(validation_months=5))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["joint_design"].update(joint_suffix=""), "alias"),
        (lambda d: d["joint_design"].update(separate_controls=["zero"]), "controles separados"),
        # Sin controles solo vale cuando no hay ámbitos de un mercado que contrastar.
        (lambda d: d["joint_design"].update(separate_controls=[]), "controles separados"),
        (lambda d: d["joint_design"].update(contrast_family="levels"), "contrato"),
        (lambda d: d["joint_design"].update(joint_scope="US"), "varios mercados"),
        (lambda d: d["joint_design"].update(ineligible_rows="dropped"), "contrato"),
        (
            lambda d: d["scopes"]["CN"]["protocols"].update(CN=str(JOINT_CN.resolve())),
            "ventana conjunta elegible",
        ),
        (lambda d: d.pop("joint_design"), "contrato"),
    ],
    ids=[
        "alias_equals_arm",
        "zero_control",
        "no_controls_with_single_scopes",
        "family_collision",
        "single_market_joint",
        "other_row_rule",
        "ineligible_pairs",
        "v4_without_design",
    ],
)
def test_joint_design_rejects_ambiguous_declarations(tmp_path, change, message):
    declared = json.loads(COMPARISON.read_text())
    for scope in declared["scopes"].values():
        scope["protocols"] = {
            market: str((EVALUATION / name).resolve())
            for market, name in scope["protocols"].items()
        }
    declared["joint_design"]["market_eligibility"] = {"CN": str(CN_V2.resolve())}
    change(declared)
    atomic_json(tmp_path / "comparison.json", declared)
    with pytest.raises(ValueError, match=message):
        comparison.load_config(tmp_path / "comparison.json")


# Vistas y ejecución con dobles


def joint_v3_protocols():
    """Protocolos del ámbito conjunto v3 y protocolo de elegibilidad de China."""
    return dict(US=US_V2, CN=JOINT_CN), dict(CN=CN_V2)


def test_joint_views_leave_china_empty_only_where_it_does_not_count(tmp_path):
    data = fixture(tmp_path / "data", ("US", "CN"))
    protocols, eligibility = joint_v3_protocols()
    sources = {market: dict(protocol=path) for market, path in protocols.items()}
    options = dict(input_policy=HISTORICAL_MASKED, recover_annual_boundaries=True)
    # Sin elegibilidad declarada, un tramo chino vacío sigue rechazándose.
    with pytest.raises(ValueError, match="sin filas en CN"):
        prepare_joint_temporal_corpus(data.parent, sources, tmp_path / "plain", **options)
    # Una elegibilidad que declara elegible una ventana sin China también se rechaza.
    with pytest.raises(ValueError, match="sin filas en CN"):
        prepare_joint_temporal_corpus(
            data.parent, sources, tmp_path / "wrong", eligibility=dict(CN=JOINT_CN), **options
        )
    output = tmp_path / "views"
    report = prepare_joint_temporal_corpus(
        data.parent, sources, output, eligibility=eligibility, **options
    )
    folds = report["folds"]
    assert len(folds) == 19 and all(row["has_all_partitions"] for row in folds)
    assert report["market_eligibility"]["CN"]["folds"] == [f"fold-{i:03d}" for i in range(6, 19)]
    assert [row["eligible_markets"] for row in folds[:6]] == [["US"]] * 6
    assert all(row["eligible_markets"] == ["US", "CN"] for row in folds[6:])
    assert folds[0]["market_counts"]["CN"] == dict.fromkeys(
        ("train", "validation", "calibration", "evaluation"), 0
    )
    # Los dos mercados conservan exactamente las filas de su protocolo, también China en el ajuste.
    for market, path in protocols.items():
        counts, _ = expected(market, data.days[market], json.loads(path.read_text()))
        for row in folds:
            local = Counter({key: n for key, n in counts.items() if key[0] == row["id"]})
            assert observed(output, row["id"], market) == local
    assert folds[3]["market_counts"]["CN"]["train"] > 0
    markets = json.loads((output / "markets/CN/report.json").read_text())
    assert markets["empty_folds_allowed"] == [f"fold-{i:03d}" for i in range(6)]


def reduced(folder, *, controls=("gru",), arms=("gru", "ridge")):
    """Comparación v4 y campaña v2 reducidas con protocolos y recetas versionados."""
    folder.mkdir(parents=True, exist_ok=True)
    declared = json.loads(COMPARISON.read_text())
    for scope in declared["scopes"].values():
        scope["protocols"] = {
            market: str((EVALUATION / name).resolve())
            for market, name in scope["protocols"].items()
        }
    declared["joint_design"].update(
        market_eligibility={"CN": str(CN_V2.resolve())}, separate_controls=list(controls)
    )
    declared["arms"] = {k: v for k, v in declared["arms"].items() if k in {"zero", *arms}}
    declared["comparison"].update(
        replicates=20,
        families=dict(references_vs_zero=dict(kind="delta", base="zero", variants=list(arms))),
    )
    atomic_json(folder / "comparison.json", declared)
    tabular = json.loads((CONFIGS / "baselines/tabular-historical-masked-v2.json").read_text())
    tabular.update(ridge_alphas=[1.0], depths=[3], bins=[64], rates=[0.1])
    atomic_json(folder / "tabular.json", tabular)
    value = json.loads(CAMPAIGN.read_text())
    for section in ("titans_mac", "episodic_gru", "mars_titan", "cm_v1", "online_controls"):
        value.pop(section)
    value.update(
        comparison="comparison.json",
        memory_options={},
        limits=dict(max_training_jobs=10_000, max_prediction_jobs=0),
        seed_policy=dict(value["seed_policy"], deterministic_arms=["ridge"]),
    )
    value["neural"]["arms"] = {arm: arm for arm in arms if arm != "ridge"}
    value["tabular"].update(config="tabular.json", arms={"ridge": "ridge"})
    atomic_json(folder / "campaign.json", value)
    return folder / "campaign.json", folder / "comparison.json"


@pytest.fixture(scope="module")
def joint_campaign(tmp_path_factory, learning_doubles_module):
    root = tmp_path_factory.mktemp("campaign-a-v2")
    data = fixture(root / "data", ("US", "CN"))
    campaign_path, comparison_path = reduced(root / "config")
    checked = engine.prepare_views(campaign_path, data.parent, root / "views")
    views = {scope: root / "views" / scope for scope in checked}
    output = root / "out"
    summary = run(campaign_path, views, output, Recorder())
    return SimpleNamespace(
        root=root,
        parent=data.parent,
        campaign=campaign_path,
        comparison=comparison_path,
        views=views,
        checked=checked,
        output=output,
        summary=summary,
    )


@pytest.fixture(scope="module")
def learning_doubles_module(tmp_path_factory):
    from mars_titan.training.learning_hold import HOLD_ENV

    path = tmp_path_factory.mktemp("hold") / "training-hold.json"
    path.write_text(json.dumps({"training_allowed": True}), encoding="utf-8")
    patch = pytest.MonkeyPatch()
    patch.setenv(HOLD_ENV, str(path))
    yield path
    patch.undo()


@pytest.fixture(autouse=True)
def _doubles(learning_doubles):
    return learning_doubles


@pytest.fixture(autouse=True)
def _restore_numerics():
    """La campaña fija la precisión del proceso. Las demás pruebas conservan la suya."""
    before = numerics.current()
    yield
    numerics.apply(before)


def test_prepare_and_views_accept_the_joint_scope_with_its_eligibility(joint_campaign):
    report = json.loads((joint_campaign.views["US+CN"] / "report.json").read_text())
    assert report["market_eligibility"]["CN"]["folds"][0] == "fold-006"
    assert (
        engine.main(
            ["views", "--campaign", str(joint_campaign.campaign)]
            + [f"--views={scope}={path}" for scope, path in joint_campaign.views.items()]
        )
        == 0
    )
    # La vista US del ámbito conjunto tiene exactamente las filas de la vista US.
    for window in ("fold-000", "fold-018"):
        joint = observed(joint_campaign.views["US+CN"], window, "US")
        alone = observed(joint_campaign.views["US"], window, "US")
        assert joint == alone


def test_joint_campaign_runs_every_scope_and_publishes_receipts_only_with_rows(joint_campaign):
    assert joint_campaign.summary["status"] == "completed"
    # US+CN: 4 de gru y 1 de ridge en 19 ventanas. US y CN: 4 de gru en 19 y 13 ventanas.
    assert joint_campaign.summary["planned"] == dict(
        training_jobs=5 * 19 + 4 * 19 + 4 * 13, prediction_jobs=0
    )
    windows = joint_campaign.output / "windows"
    seed = windows / "US+CN/fold-000/gru/seed-42"
    assert (seed / "US.json").is_file() and not (seed / "CN.json").exists()
    assert (windows / "US+CN/fold-006/gru/seed-43/CN.json").is_file()
    assert (windows / "CN/fold-000/gru/seed-44/CN.json").is_file()


def test_a_window_run_executes_that_window_of_every_scope_and_reuses_its_receipts(
    joint_campaign, tmp_path
):
    def window(name, recorder):
        return engine.run_campaign(
            joint_campaign.campaign,
            joint_campaign.views,
            tmp_path / "out",
            executors=doubles(recorder),
            lease=nullcontext,
            stop=SimpleNamespace(requested=False),
            window=name,
        )

    recorder = Recorder()
    summary = window("fold-006", recorder)
    assert summary["status"] == "completed" and summary["window"] == "fold-006"
    ids = [call["id"] for call in recorder.calls]
    expected = {("US+CN", "fold-006"), ("US", "fold-006"), ("CN", "fold-000")}
    assert {tuple(job.split("/")[:2]) for job in ids} == expected
    # gru con dos búsquedas y dos finalistas en cada ámbito y ridge solo en el conjunto.
    assert len(ids) == 4 * 3 + 1 and summary["completed"]["training_jobs"] == 13
    finalists = [i for i, job in enumerate(ids) if "/finalist-" in job]
    assert max(i for i, job in enumerate(ids) if "/search-" in job) < min(finalists)
    again = Recorder()
    assert window("fold-006", again)["status"] == "completed" and not again.calls
    following = Recorder()
    window("fold-007", following)
    assert {tuple(call["id"].split("/")[:2]) for call in following.calls} == {
        ("US+CN", "fold-007"),
        ("US", "fold-007"),
        ("CN", "fold-001"),
    }


class Precision(Recorder):
    """Doble que registra la precisión vigente y puede contradecir la declarada."""

    def __init__(self, *, recorded=None, flip=False):
        super().__init__()
        self.seen, self.recorded, self.flip = [], recorded, flip

    def __call__(self, run):
        import torch

        self.seen.append(numerics.current())
        report = super().__call__(run)
        if self.recorded:
            report["runtime"] = dict(numerics=dict(self.recorded))
        if self.flip:
            torch.backends.cudnn.allow_tf32 = True
        return report


def _window(joint_campaign, output, recorder, window="fold-006"):
    return engine.run_campaign(
        joint_campaign.campaign,
        joint_campaign.views,
        output,
        executors=doubles(recorder),
        lease=nullcontext,
        stop=SimpleNamespace(requested=False),
        window=window,
    )


def test_the_launcher_fixes_strict_fp32_and_records_it_in_every_receipt(joint_campaign, tmp_path):
    import torch

    torch.backends.cudnn.allow_tf32 = True
    recorder = Precision()
    _window(joint_campaign, tmp_path / "out", recorder)
    assert recorder.seen and all(seen == numerics.STRICT_FP32 for seen in recorder.seen)
    receipts = list((tmp_path / "out/jobs").rglob("receipt.json"))
    assert len(receipts) == len(recorder.calls)
    assert all(
        json.loads(path.read_text())["numerics"] == numerics.STRICT_FP32 for path in receipts
    )
    marker = json.loads((tmp_path / "out/campaign.json").read_text())
    assert marker["numerics"] == numerics.STRICT_FP32
    # Un recibo que registra otra precisión se rechaza al reanudar.
    path = receipts[0]
    receipt = json.loads(path.read_text())
    receipt["numerics"] = dict(numerics.STRICT_FP32, cudnn_allow_tf32=True)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="registra otra precisión numérica"):
        _window(joint_campaign, tmp_path / "out", Precision())


@pytest.mark.parametrize(
    ("recorder", "message"),
    [
        (lambda: Precision(recorded={"cudnn_allow_tf32": True}), "registra otra precisión"),
        (lambda: Precision(recorded={"matmul_precision": "high"}), "registra otra precisión"),
        (lambda: Precision(flip=True), "terminó con otra precisión"),
    ],
    ids=["report_with_cudnn_tf32", "report_with_high_precision", "executor_changes_flags"],
)
def test_a_job_that_runs_or_reports_another_precision_is_not_confirmed(
    joint_campaign, tmp_path, recorder, message
):
    with pytest.raises(ValueError, match=message):
        _window(joint_campaign, tmp_path / "out", recorder())
    assert not list((tmp_path / "out/jobs").rglob("receipt.json"))


def test_recorded_precision_follows_every_alias_in_nested_reports():
    report = dict(
        identity=dict(numerics=dict(cudnn_allow_tf32=False, matmul_tf32=True)),
        cases=[dict(matmul_precision="highest"), dict(cudnn_tf32=True)],
    )
    assert numerics.recorded(report, numerics.STRICT_FP32) == [
        "informe.identity.numerics.matmul_tf32=True",
        "informe.cases[1].cudnn_tf32=True",
    ]
    assert numerics.recorded(dict(a=[dict(b=1)]), numerics.STRICT_FP32) == []


def test_comparison_excludes_ineligible_china_rows_and_pairs_joint_with_separate(joint_campaign):
    campaign_path, config = joint_campaign.campaign, joint_campaign.comparison
    paths = {
        scope: engine.write_sources(
            campaign_path, joint_campaign.views, joint_campaign.output, scope
        )
        for scope in ("US+CN", "US", "CN")
    }
    report, sessions = comparison.evaluate_walk_forward(config, paths["US+CN"], "US+CN")
    windows = report["arms"]["gru"]["seeds"]["42"]["windows"]
    excluded = {w: e.get("excluded_rows") for w, e in windows.items()}
    assert all(excluded[f"fold-{i:03d}"] is None for i in range(6, 19))
    assert excluded["fold-003"]["evaluation"]["CN"] > 0
    assert excluded["fold-003"]["calibration"]["CN"] > 0
    # Ninguna sesión china anterior a 2011 entra en las métricas.
    china = sessions.filter(pa_equal(sessions["market"], "CN"))
    years = np.asarray([moment.year for moment in china["prediction_at"].to_pylist()])
    assert years.min() >= 2011 and len(years) > 0
    assert report["joint_design"]["eligible_windows"]["CN"][0] == "fold-006"
    for scope, first_joint in (("US", "fold-000"), ("CN", "fold-006")):
        manifest = json.loads(paths[scope].read_text())
        assert set(manifest["arms"]) == {"gru", "gru_joint"}
        joint_view = manifest["windows"]["fold-000"]["joint_view"]
        assert (
            joint_view["sha256"]
            == joint_campaign.checked["US+CN"]["windows"][first_joint]["sha256"]
        )
        scoped, rows = comparison.evaluate_walk_forward(config, paths[scope], scope)
        assert set(scoped["arms"]) == {"zero", "gru", "gru_joint"}
        assert set(scoped["contrasts"][scope]) == {"joint_vs_separate"}
        mae = scoped["contrasts"][scope]["joint_vs_separate"]["mae"]
        assert [row["name"] for row in mae["contrasts"]] == ["gru_joint-gru"]
        assert set(rows["market"].to_pylist()) == {scope}
        # El brazo conjunto restringido descarta las filas del otro mercado de la ventana.
        dropped = scoped["arms"]["gru_joint"]["seeds"]["42"]["windows"]["fold-012"]
        other = "CN" if scope == "US" else "US"
        assert dropped["excluded_rows"]["evaluation"][other] > 0


def test_window_aggregates_score_each_scope_with_its_joint_design(joint_campaign, monkeypatch):
    from mars_titan.evaluation import window_aggregates
    from mars_titan.training import rolling_retention as rolling

    declared = comparison.load_config(joint_campaign.comparison)
    # Sin la cartera ni los estratos de liquidez, que necesitarían la edición de precios,
    # solo queda el walk-forward.
    plain = {
        key: value
        for key, value in declared.items()
        if key not in (comparison.LONG_SHORT_FIELD, comparison.LIQUIDITY_FIELD)
    }
    monkeypatch.setattr(rolling.Rolling, "comparison_config", lambda self: plain)
    retention = rolling.load_retention(CONFIGS / "baselines/historical-masked-retention-v2.json")
    # Solo se prueban los agregados, así que el recorrido no lleva la etapa de políticas.
    retention = dict(retention, consumers={})
    rows = order.campaign_windows(plan.load_campaign(joint_campaign.campaign))
    walker = rolling.Rolling(
        retention, joint_campaign.campaign, joint_campaign.views, joint_campaign.output, rows
    )
    walker.folder = joint_campaign.root / "rolling-aggregates"
    row = next(row for row in rows if row["id"] == "fold-012")
    assert set(walker.aggregates(row)) == {"US+CN", "US", "CN"}
    for scope, window in row["scopes"].items():
        path = engine.write_sources(
            joint_campaign.campaign, joint_campaign.views, joint_campaign.output, scope
        )
        sources = comparison.load_sources(path, declared, scope)
        scoped = comparison.scope_config(declared, scope)
        stored = window_aggregates.read(walker.folder / "aggregates", scoped, sources, window)
        assert window_aggregates.same(stored, comparison._score_window(sources, scoped, window))
        # En un mercado, los agregados incluyen el brazo conjunto prestado.
        if scope != "US+CN":
            assert {arm for arm, _ in stored[0]} == {"zero", "gru", "gru_joint"}


def pa_equal(column, value):
    import pyarrow.compute as pc

    return pc.equal(column, value)


def test_sources_reject_a_borrowed_arm_from_another_window(joint_campaign):
    path = engine.write_sources(
        joint_campaign.campaign, joint_campaign.views, joint_campaign.output, "CN"
    )
    manifest = json.loads(path.read_text())
    entry = manifest["arms"]["gru_joint"]["42"]
    entry["fold-001"], entry["fold-000"] = entry["fold-000"], entry["fold-001"]
    tampered = path.with_name("tampered.json")
    atomic_json(tampered, manifest)
    config = comparison.load_config(joint_campaign.comparison)
    with pytest.raises(ValueError, match="usa otra vista"):
        comparison.load_sources(tampered, config, "CN")
    # Sin la vista conjunta emparejada, las fuentes de un ámbito con brazos prestados no valen.
    manifest = json.loads(path.read_text())
    manifest["windows"]["fold-000"].pop("joint_view")
    atomic_json(tampered, manifest)
    with pytest.raises(ValueError, match="necesita su vista"):
        comparison.load_sources(tampered, config, "CN")


def test_budget_counts_from_targets_equal_the_prepared_views(joint_campaign):
    loaded = plan.load_campaign(joint_campaign.campaign)
    labels = json.loads(Path(joint_campaign.parent).read_text())["roots"]["labels"]
    counts = budget.campaign_window_counts(loaded, labels)
    for scope, record in joint_campaign.checked.items():
        assert counts[scope] == {w: v["counts"] for w, v in record["windows"].items()}, scope


def test_budget_projection_applies_the_documented_formula(joint_campaign):
    loaded = plan.load_campaign(joint_campaign.campaign)
    counts = {
        scope: {w: v["counts"] for w, v in record["windows"].items()}
        for scope, record in joint_campaign.checked.items()
    }
    rate, ratio, epochs = 1000.0, 4.0, 7
    rates = budget.uniform_rates(loaded, rate, inference_ratio=ratio)
    result = budget.project(loaded, counts, rates, epochs=epochs, fixed_hours=2.0, target_hours=5.0)
    # Referencia neuronal: épocas por (ajuste / caudal + validación / inferencia) más una
    # pasada final de los cuatro tramos a la velocidad de inferencia.
    seconds = 0.0
    for job in plan.plan_campaign(loaded):
        if job["arm"] != "gru":
            continue
        rows = counts[job["scope"]][job["window"]]
        seconds += epochs * (rows["train"] / rate + rows["validation"] / (rate * ratio))
        seconds += sum(rows.values()) / (rate * ratio)
    assert result["stages"]["base"] == pytest.approx(seconds / 3600, rel=1e-12)
    assert result["epochs"] == epochs and result["total_hours"] == pytest.approx(
        result["neural_hours"] + 2.0
    )
    assert result["throughput_factor_needed"] == pytest.approx(result["neural_hours"] / 3.0)
    doubled = budget.project(
        loaded, counts, budget.uniform_rates(loaded, 2 * rate, inference_ratio=ratio), epochs=epochs
    )
    assert doubled["stages"]["base"] == pytest.approx(result["stages"]["base"] / 2)
    with pytest.raises(ValueError, match="superar las horas fijas"):
        budget.project(loaded, counts, rates, fixed_hours=5.0, target_hours=5.0)
    with pytest.raises(ValueError, match="positivos"):
        budget.uniform_rates(loaded, 0)


def test_declared_counts_cover_the_v2_campaign_and_reproduce_the_views_receipt():
    loaded = campaign()
    counts, _ = budget.read_counts(
        Path("reports/data/campaign-a-v2-window-counts-20261009.json"), loaded
    )
    receipt = json.loads(Path("reports/data/campaign-a-views-20261009.json").read_text())
    scopes = receipt["preparation"]["scopes"]
    for scope in ("US", "CN"):
        assert list(counts[scope].values()) == scopes[scope]["counts_by_window"]
    joint = counts["US+CN"]
    assert sum(row["train"] for row in joint.values()) == 125_553_766
    # El lado US del conjunto son exactamente las filas de la vista US.
    us = scopes["US"]["counts_by_window"]
    china = [{k: joint[w][k] - us[i][k] for k in us[i]} for i, w in enumerate(joint)]
    assert china[6:] == scopes["CN"]["counts_by_window"]
    assert china[0] == dict(train=0, validation=0, calibration=0, evaluation=0)


def test_read_counts_rejects_counts_that_miss_a_window(tmp_path):
    document = json.loads(
        Path("reports/data/campaign-a-v2-window-counts-20261009.json").read_text()
    )
    document["counts"]["CN"].pop("fold-012")
    atomic_json(tmp_path / "counts.json", document)
    with pytest.raises(ValueError, match="ventanas de la campaña"):
        budget.read_counts(tmp_path / "counts.json", campaign())


def test_target_counts_purge_a_label_that_matures_exactly_at_the_boundary(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    def moments(*texts):
        return pa.array(np.array(texts, dtype="datetime64[us]"), type=pa.timestamp("us", tz="UTC"))

    folder = tmp_path / "labels/US/AAA"
    folder.mkdir(parents=True)
    pq.write_table(
        pa.table(
            dict(
                prediction_at=moments("2004-03-30T20:00", "2004-03-30T20:00", "2004-03-29"),
                # Madura justo en la frontera del ajuste de fold-000, antes y sin objetivo.
                target_available_at=moments("2004-04-01", "2004-03-31T21:00", "2004-03-30"),
                target=pa.array([0.1, 0.2, None], type=pa.float64()),
            )
        ),
        folder / "labels.parquet",
    )
    protocol = json.loads(US_V2.read_text())
    counts = budget.target_window_counts(tmp_path / "labels", protocol)
    assert counts["fold-000"] == dict(train=1, validation=0, calibration=0, evaluation=0)


# Política de datos: solo la edición real


def test_the_plan_checks_every_declared_document_of_the_campaign_and_its_stages():
    value = campaign()
    names = [path.name for path in data_policy.documents(value, plan.LATER_STAGES)]
    assert names == [
        "historical-masked-campaign-a-v2.json",
        "tabular-historical-masked-v2.json",
        "chronological-training-historical-masked.json",
        "chronological-training.json",
        "episodic-readout-historical-masked.json",
        "cm-v1-factorial.json",
        "historical-masked-adapter-stage-a-v2.json",
        "adapter-matrix-v3.json",
        "historical-masked-rl-stage-a-v2.json",
        "historical-masked-rl-policies.json",
        "historical-masked-ablation-stage-a-v2.json",
    ]
    assert len(plan.plan_campaign(value)) == 2341 + 133
    # Las fuentes de predictor de las políticas son estados ajustados con la edición real.
    from mars_titan.simulation import policy_plan

    assert set(policy_plan.PREDICTOR_SOURCES) <= data_policy.REAL_SOURCES
    assert plan.check_campaign(CAMPAIGN)["data_policy"] == "real_edition_only"


@pytest.mark.parametrize(
    ("stage", "matrix", "message"),
    [
        (dict(scenario="hmm_regimes"), {}, "clave .scenario"),
        (dict(environment=dict(kind="synthetic_market")), {}, "valor .environment.kind"),
        (dict(environment=dict(source="simulated")), {}, "fuente .environment.source"),
        ({}, dict(budget=dict(augmentation=True)), "clave .budget.augmentation"),
        ({}, dict(input_policy="strict_four_modalities_v1"), "no es la política"),
        ({}, dict(rows="resampled_with_replacement"), "valor .rows"),
    ],
    ids=[
        "stage_scenario",
        "synthetic_environment",
        "simulated_source",
        "augmented_adapters",
        "another_input_policy",
        "resampled_rows",
    ],
)
def test_the_plan_rejects_synthetic_resampled_or_foreign_data_in_a_registered_stage(
    tmp_path, monkeypatch, stage, matrix, message
):
    path = edited(tmp_path, lambda value: None)
    value = plan.load_campaign(path)
    folder = tmp_path / "stages"
    folder.mkdir()
    atomic_json(folder / "matrix.json", dict(kind="matrix", **matrix))
    atomic_json(
        folder / "stage.json",
        dict(kind="stage", campaign=str(path.resolve()), matrix="matrix.json", **stage),
    )
    clean = dict(stages={}, joint_stage=str(folder / "other.json"))
    atomic_json(folder / "other.json", dict(campaign=str(CAMPAIGN.resolve())))
    monkeypatch.setattr(
        plan,
        "LATER_STAGES",
        dict(clean=clean, dirty=dict(stages=dict(A=str(folder / "stage.json")))),
    )
    with pytest.raises(ValueError, match=message):
        plan.plan_campaign(value)


def test_the_plan_rejects_row_resampling_in_xgboost_and_views_of_another_edition(monkeypatch):
    from mars_titan.models.baselines import external_boosting

    value = campaign()
    monkeypatch.setattr(external_boosting, "ROW_SAMPLING", dict(subsample=0.8, colsample_bytree=1))
    with pytest.raises(ValueError, match="XGBoost remuestrea"):
        plan.plan_campaign(value)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="vistas de la edición histórica"):
        plan.plan_campaign(dict(value, input_policy="strict_four_modalities_v1"))


def test_findings_read_keys_and_values_but_not_the_prose_that_explains_a_decision():
    document = dict(
        pending=["decidida con fixtures técnicos"],
        cases=[dict(name="ok"), dict(name="bootstrap_rows")],
        tapes="reconstructed_tapes",
        source="edition_views",
    )
    assert data_policy.findings(document, "p", "doc") == [
        "doc: valor .cases[1].name='bootstrap_rows'"
    ]
    assert data_policy.findings(dict(dataset="toy"), "p", "doc") == ["doc: fuente .dataset='toy'"]
