"""Walk-forward por etapas de la campaña A: roles, cadena, control en línea y disjunción.

Las pruebas del plan no leen datos. Las del verificador preparan vistas sobre el corpus
técnico, las alteran en sitio y las restauran, y escriben recibos de la cadena a mano. Nada
se ajusta, no hay pasos de optimizador y la GPU no se usa.
"""

import math

import numpy as np
import pytest

from mars_titan.training import campaign_chain as chain
from mars_titan.training import campaign_online_controls as online
from mars_titan.training import campaign_plan as plan
from mars_titan.training import campaign_schedule as order
from mars_titan.training import masked_campaign as engine
from tests.training.test_campaign_a_joint import CAMPAIGN, CONFIGS, edited

JOINT = "US+CN"


def campaign():
    return plan.load_campaign(CAMPAIGN)


def micros(day):
    return int(np.datetime64(day, "us").astype(np.int64))


# Roles por ventana


def test_v2_declares_the_staged_roles_of_every_window():
    value = campaign()
    assert value["walk_forward_stages"] == chain.DESIGN
    first = chain.window_roles(value, JOINT, "fold-000")
    assert first["posttraining"] is None and first["rl"] is None
    assert first["chain"]["candidates"] == ["base"]
    assert first["test"] == ["2005-01-01", "2006-01-01"]
    roles = chain.window_roles(value, JOINT, "fold-005")
    assert roles["base"]["train"] == ["2000-01-01", "2009-04-01"]
    assert roles["posttraining"] == dict(
        parent_window="fold-004",
        parent="base_selected_state",
        fit=["2009-01-01", "2009-04-01"],
        validation=["2009-04-01", "2009-10-01"],
        calibration=["2009-10-01", "2010-01-01"],
        evaluation=["2010-01-01", "2011-01-01"],
    )
    assert roles["chain"]["candidates"] == ["frozen_parent", "adapter", "continuation"]
    assert roles["rl"] == dict(
        train=["fold-000", "fold-001", "fold-002", "fold-003"],
        validation="fold-004",
        evaluation="fold-005",
    )
    # China separada empieza en 2011 y su primera ventana tampoco tiene padre.
    assert chain.window_roles(value, "CN", "fold-000")["posttraining"] is None
    assert chain.window_roles(value, "CN", "fold-001")["posttraining"]["fit"] == [
        "2011-01-01",
        "2011-04-01",
    ]


def test_new_rows_start_after_everything_the_parent_used_and_end_with_the_train_split():
    value = campaign()
    for scope in value["scopes"]:
        windows = chain.scope_windows(value, scope)
        for (_, parent), (window, fold) in zip(windows, windows[1:], strict=False):
            start, end = chain.posttraining_rows(parent, fold)
            assert start == parent["calibration"][1] == parent["evaluation"][0]
            assert end == fold["train"][1] and start < end
            assert chain.parent_window(value, scope, window) == parent["id"]
    folds = dict(chain.scope_windows(value, JOINT))
    with pytest.raises(ValueError, match="no tiene filas nuevas"):
        chain.posttraining_rows(folds["fold-001"], folds["fold-001"])
    with pytest.raises(ValueError, match="no tiene filas nuevas"):
        chain.posttraining_rows(folds["fold-002"], folds["fold-001"])
    assert chain.parent_window(value, JOINT, "fold-000") is None
    with pytest.raises(ValueError, match="no es una ventana"):
        chain.parent_window(value, JOINT, "fold-019")


def test_rl_windows_expand_and_need_three_previous_evaluations():
    windows = [f"fold-{i:03d}" for i in range(8)]
    assert [chain.rl_windows(windows, w) for w in windows[:4]] == [None] * 4
    assert chain.rl_windows(windows, "fold-004") == dict(
        train=windows[:3], validation="fold-003", evaluation="fold-004"
    )
    assert chain.rl_windows(windows, "fold-007")["train"] == windows[:6]
    with pytest.raises(ValueError, match="no tiene evaluación"):
        chain.rl_windows(windows[1:], "fold-000")


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v["walk_forward_stages"]["chain"].update(min_improvement=0),
        lambda v: v["walk_forward_stages"]["chain"].update(min_improvement=0.001),
        lambda v: v["walk_forward_stages"]["posttraining"]["candidates"].append("base_retrain"),
        lambda v: v["walk_forward_stages"]["posttraining"].update(parent="chain_previous"),
        lambda v: v["walk_forward_stages"]["rl"].update(min_train_windows=2),
        lambda v: v["walk_forward_stages"]["rl"].update(train="rolling_previous_evaluations"),
        lambda v: v["walk_forward_stages"].update(test="validation"),
        lambda v: v["walk_forward_stages"].pop("test"),
    ],
    ids=[
        "integer_improvement",
        "loose_improvement",
        "retrain_as_candidate",
        "chain_parent",
        "two_rl_windows",
        "rolling_rl",
        "test_on_validation",
        "missing_test",
    ],
)
def test_the_staged_declaration_admits_no_other_design(tmp_path, change):
    with pytest.raises(ValueError, match="staged_chain_v1"):
        plan.load_campaign(edited(tmp_path, change))


def test_the_staged_design_runs_window_by_window(tmp_path):
    with pytest.raises(ValueError, match="ventana a ventana"):
        plan.load_campaign(edited(tmp_path, lambda v: v.update(execution={"order": "by_scope"})))


# Calendario y dependencias


def staged_jobs(value, base, *, arm="rnn", seed=42):
    """Adaptadores y políticas sintéticos que siguen el contrato de la cadena."""
    windows = [name for name, _ in chain.scope_windows(value, JOINT)]
    adapters = [
        dict(
            id=f"{JOINT}/{window}/{arm}__head/fit-s{seed}",
            scope=JOINT,
            window=window,
            anchor=window,
            base_arm=arm,
            seed=seed,
            depends=chain.parent_jobs(base, JOINT, parent, arm, seed),
        )
        for parent, window in zip(windows, windows[1:], strict=False)
    ]
    policies = []
    for window in windows:
        rows = chain.rl_windows(windows, window)
        if rows is None:
            continue
        read = [*rows["train"], rows["validation"], window]
        policies.append(
            dict(
                id=f"{JOINT}/US/{window}/{arm}__chain/ppo/fit-s7",
                scope=JOINT,
                window=window,
                anchor=window,
                train=rows["train"],
                validation=rows["validation"],
                predictor=chain.chain_arm(arm),
                predictor_seed=seed,
                seed=7,
                depends=[chain.chain_job_id(JOINT, name, arm, seed) for name in read],
            )
        )
    return adapters, policies


def searches(base, prefix):
    return sorted(job["id"] for job in base if job["id"].startswith(f"{prefix}/search-"))


def test_the_schedule_selects_the_chain_after_posttraining_and_before_the_policy():
    value = campaign()
    base = plan.plan_campaign(value)
    adapters, policies = staged_jobs(value, base)
    schedule = order.window_schedule(value, base, dict(adapters=adapters, rl=policies))
    phase = {name: index for index, name in enumerate(order.PHASES)}
    assert phase["online"] < phase["adapters"] < phase["chain"] < phase["rl"]
    chains = [row["phases"][phase["chain"]]["jobs"] for row in schedule]
    assert chains == [[chain.chain_job_id(JOINT, row["window"], "rnn", 42)] for row in schedule]
    assert [len(row["phases"][phase["rl"]]["jobs"]) for row in schedule] == [0] * 4 + [1] * 15
    # La cadena de la ventana 0 es la base y depende de las búsquedas que la eligen.
    jobs = {job["id"]: job for job in chain.chain_jobs(value, base, adapters)}
    first = jobs[chain.chain_job_id(JOINT, "fold-000", "rnn", 42)]
    assert first["depends"] == searches(base, f"{JOINT}/fold-000/rnn")
    assert len(first["depends"]) == 2
    assert jobs[chain.chain_job_id(JOINT, "fold-003", "rnn", 42)]["depends"] == [
        f"{JOINT}/fold-003/rnn__head/fit-s42"
    ]
    assert chain.parent_jobs(base, JOINT, "fold-002", "rnn", 43) == [
        f"{JOINT}/fold-002/rnn/finalist-s43"
    ]
    with pytest.raises(ValueError, match="no tiene posentrenamiento"):
        chain.chain_jobs(value, base, adapters[:-1])
    with pytest.raises(ValueError, match="no elige el estado"):
        chain.parent_jobs(base, JOINT, "fold-002", "rnn", 45)


def _broken(kind):
    value = campaign()
    base = plan.plan_campaign(value)
    adapters, policies = staged_jobs(value, base)
    if kind == "adapter_without_parent":
        adapters[3]["depends"] = adapters[3]["depends"][1:]
    elif kind == "adapter_with_older_parent":
        adapters[3]["depends"] = chain.parent_jobs(base, JOINT, "fold-001", "rnn", 42)
    elif kind == "adapter_in_first_window":
        adapters.append(dict(adapters[0], id="first", window="fold-000", depends=[]))
    elif kind == "policy_missing_a_window":
        policies[5]["depends"] = policies[5]["depends"][1:]
    elif kind == "policy_with_two_windows":
        policies[0] = dict(policies[0], train=policies[0]["train"][1:])
    elif kind == "policy_without_predictor_seed":
        policies[0].pop("predictor_seed")
    elif kind == "policy_reads_a_later_chain":
        policies[0]["depends"].append(chain.chain_job_id(JOINT, "fold-005", "rnn", 42))
    return value, base, dict(adapters=adapters, rl=policies)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("adapter_without_parent", "no depende del estado elegido de la base en fold-003"),
        ("adapter_with_older_parent", "no depende del estado elegido de la base en fold-003"),
        ("adapter_in_first_window", "la primera ventana no tiene posentrenamiento"),
        ("policy_missing_a_window", "no depende de la cadena de todas las ventanas"),
        ("policy_with_two_windows", "no depende de la cadena de todas las ventanas"),
        ("policy_without_predictor_seed", "semilla de su predictor"),
        ("policy_reads_a_later_chain", "fase posterior"),
    ],
)
def test_the_schedule_rejects_stages_that_break_the_staged_dependencies(kind, message):
    value, base, stages = _broken(kind)
    with pytest.raises(ValueError, match=message):
        order.window_schedule(value, base, stages)


def test_without_the_staged_declaration_the_schedule_keeps_the_old_contract(tmp_path):
    value = plan.load_campaign(edited(tmp_path, lambda v: v.pop("walk_forward_stages")))
    base = plan.plan_campaign(value)
    adapters, _ = staged_jobs(value, base)
    first = dict(adapters[0], id="first", window="fold-000", depends=[])
    schedule = order.window_schedule(value, base, dict(adapters=[first, *adapters]))
    assert all(not row["phases"][order.PHASES.index("chain")]["jobs"] for row in schedule)


# Regla de la cadena


def candidate(kind, job, score):
    return dict(kind=kind, arm=job, job=job, receipt_sha256="0" * 64, score=score)


def test_the_frozen_parent_is_replaced_only_by_a_strict_improvement():
    frozen = candidate("frozen_parent", "p", 1.0)
    tie = candidate("adapter", "a", 1.0)
    better = candidate("continuation", "c", 0.5)
    also = candidate("adapter", "b", 0.5)
    assert chain.choose([frozen, tie]) is frozen
    assert chain.choose([frozen]) is frozen
    assert chain.choose([tie, frozen, better]) is better
    # Entre candidatos empatados decide el identificador.
    assert chain.choose([frozen, better, also]) is also
    assert chain.choose([frozen, candidate("adapter", "a", 1.0 - 1e-12)])["job"] == "a"
    for broken in (
        [tie],
        [frozen, dict(frozen, job="q")],
        [frozen, candidate("adapter", "a", math.nan)],
        [frozen, candidate("base", "a", 0.1)],
        [frozen, dict(tie, job="p")],
    ):
        with pytest.raises(ValueError, match="padre congelado"):
            chain.choose(broken)


def test_row_fingerprints_ignore_order_and_combine_asset_by_asset():
    markets = ["US", "US", "CN", "US"]
    symbols = ["A", "B", "A", "A"]
    rows = [3, 1, 3, 7]
    whole = chain.row_fingerprint(markets, symbols, rows)
    assert whole == chain.row_fingerprint(markets[::-1], symbols[::-1], rows[::-1])
    parts = {
        ("US", "A"): chain.asset_digest([7, 3]),
        ("US", "B"): chain.asset_digest([1]),
        ("CN", "A"): chain.asset_digest([3]),
    }
    assert whole == chain.combine(parts) and whole[0] == 4
    assert chain.row_fingerprint(["US"], ["A"], [4]) != chain.row_fingerprint(["CN"], ["A"], [4])
    assert chain.row_fingerprint(markets, symbols, [3, 1, 3, 8]) != whole
    with pytest.raises(ValueError, match="sin repetir"):
        chain.row_fingerprint(["US", "US"], ["A", "A"], [1, 1])
    with pytest.raises(ValueError, match="no aporta filas"):
        chain.combine({("US", "A"): (0, "x")})


# Control en línea y variante B


def test_the_online_control_starts_from_the_selected_parent_with_the_bank_cap():
    value = campaign()
    base = plan.plan_campaign(value)
    jobs = [job for job in base if job["kind"] == online.ONLINE]
    assert len(jobs) == 153 == value["online_controls"]["limits"]["max_online_jobs"]
    assert plan.count_jobs(value)["online_jobs"] == 153
    by_id = {job["id"]: job for job in jobs}
    first = by_id["US/fold-004/transformer_compact_online/online-s42"]
    assert first["depends"] == [
        *searches(base, "US/fold-004/transformer_compact"),
        *searches(base, "US/fold-004/mars_titan_m1"),
    ]
    assert len(first["depends"]) == 4
    other = by_id["CN/fold-012/transformer_compact_online/online-s44"]
    assert other["depends"] == [
        "CN/fold-012/transformer_compact/finalist-s44",
        "CN/fold-012/mars_titan_m1/finalist-s44",
    ]
    assert all(
        (job["model"], job["stage"], job["regenerable"], job["family"])
        == ("neural", "online", False, plan.NEURAL)
        and job["case"]["rule"]["update_cap"] == online.CAP
        for job in jobs
    )
    # El calendario los pone tras elegir padre y tope en su ventana.
    schedule = order.window_schedule(value, plan.plan_campaign(value))
    assert sum(len(row["phases"][order.PHASES.index("online")]["jobs"]) for row in schedule) == 153


def test_the_online_control_respects_its_job_limit(tmp_path):
    def tight(value):
        value["online_controls"]["limits"]["max_online_jobs"] = 152

    with pytest.raises(ValueError, match="max_online_jobs=152"):
        plan.plan_campaign(plan.load_campaign(edited(tmp_path, tight)))


@pytest.mark.parametrize(
    "change",
    [
        lambda arm: arm.update(cap_arm="mars_titan_m2"),
        lambda arm: arm.update(parent_arm="gru"),
        lambda arm: arm.update(partitions=["evaluation"]),
        lambda arm: arm["rule"].update(optimizer="adam"),
        lambda arm: arm["rule"].update(update_cap="rows"),
        lambda arm: arm["rule"].update(learning_rate=0.0),
        lambda arm: arm["rule"].update(learning_rate=1),
        lambda arm: arm["rule"].update(block_rows=64.0),
        lambda arm: arm["rule"].update(update_every=0),
        lambda arm: arm["rule"].update(max_grad_norm=-1.0),
        lambda arm: arm["rule"].pop("max_grad_norm"),
    ],
    ids=[
        "other_cap",
        "other_parent",
        "evaluation_only",
        "adam",
        "row_cap",
        "zero_rate",
        "integer_rate",
        "float_block",
        "zero_frequency",
        "negative_clip",
        "missing_clip",
    ],
)
def test_the_online_control_admits_only_its_declared_rule(tmp_path, change):
    def broken(value):
        change(value["online_controls"]["arms"][online.ARM])

    with pytest.raises(ValueError, match="se limita con las escrituras del banco"):
        plan.load_campaign(edited(tmp_path, broken))


def test_the_online_control_needs_its_parent_and_cap_in_every_scope(tmp_path):
    def no_reader(declared):
        declared["joint_design"]["separate_controls"] = ["transformer_compact"]

    path = edited(tmp_path, lambda value: None, comparison_change=no_reader)
    with pytest.raises(ValueError, match="En US el control en línea necesita"):
        plan.load_campaign(path)


def test_the_online_rule_blocks_the_launch_until_every_value_is_declared(tmp_path):
    pending = [r for r in plan.launch_blockers(campaign()) if r.startswith(online.ARM)]
    assert pending == [
        f"{online.ARM}.rule.{name} sigue pendiente"
        for name in ("learning_rate", "block_rows", "update_every", "max_grad_norm")
    ]

    def declared(value):
        value["online_controls"]["arms"][online.ARM]["rule"].update(
            learning_rate=1e-4, block_rows=64, update_every=1, max_grad_norm=1.0
        )

    loaded = plan.load_campaign(edited(tmp_path, declared))
    assert not [r for r in plan.launch_blockers(loaded) if r.startswith(online.ARM)]


def test_variant_b_is_counted_but_never_launched(tmp_path, learning_doubles):
    path = CONFIGS / "baselines/historical-masked-campaign-b.json"
    report = plan.check_campaign(path)
    assert report["launch_blockers"] == [plan.RETIRED[str(path)]]
    assert report["counts"]["training_jobs"] > 0 and report["counts"]["prediction_jobs"] > 0
    assert plan.RETIRED[str(path)] not in plan.launch_blockers(campaign())
    with pytest.raises(ValueError, match="no se puede lanzar"):
        engine.run_campaign(path, {}, tmp_path / "out")
    assert not (tmp_path / "out").exists()
