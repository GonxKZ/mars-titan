"""Walk-forward por etapas de la campaña A: roles, cadena, control en línea y disjunción.

Las pruebas del plan no leen datos. Las del verificador preparan vistas sobre el corpus
técnico, las alteran en sitio y las restauran, y escriben recibos de la cadena a mano. Nada
se ajusta, no hay pasos de optimizador y la GPU no se usa.
"""

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.walk_forward_receipt import RECEIPT_KIND
from mars_titan.training import campaign_budget as budget
from mars_titan.training import campaign_chain as chain
from mars_titan.training import campaign_online_controls as online
from mars_titan.training import campaign_plan as plan
from mars_titan.training import campaign_schedule as order
from mars_titan.training import chain_disjunction as disjunction
from mars_titan.training import masked_campaign as engine
from tests.training.test_campaign_a_joint import CAMPAIGN, CONFIGS, edited, reduced
from tests.training.test_walk_forward_v2_views import fixture

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
        train=["fold-001", "fold-002", "fold-003"],
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
    edge = dict(
        folds["fold-001"],
        train=[folds["fold-001"]["train"][0], folds["fold-000"]["calibration"][1]],
    )
    with pytest.raises(ValueError, match="no tiene filas nuevas"):
        chain.posttraining_rows(folds["fold-000"], edge)
    assert chain.parent_window(value, JOINT, "fold-000") is None
    with pytest.raises(ValueError, match="no es una ventana"):
        chain.parent_window(value, JOINT, "fold-019")


def test_rl_windows_use_the_three_evaluations_before_validation_or_the_declared_expansion():
    windows = [f"fold-{i:03d}" for i in range(8)]
    for rule in (chain.RL_RULE, chain.RL_EXPANDING):
        assert [chain.rl_windows(windows, w, rule) for w in windows[:4]] == [None] * 4
        assert chain.rl_windows(windows, "fold-004", rule) == dict(
            train=windows[:3], validation="fold-003", evaluation="fold-004"
        )
    assert chain.rl_windows(windows, "fold-007") == dict(
        train=windows[3:6], validation="fold-006", evaluation="fold-007"
    )
    assert chain.rl_windows(windows, "fold-007", chain.RL_EXPANDING)["train"] == windows[:6]
    with pytest.raises(ValueError, match="no es una regla"):
        chain.rl_windows(windows, "fold-007", "rolling_previous_evaluations")
    with pytest.raises(ValueError, match="no tiene evaluación"):
        chain.rl_windows(windows[1:], "fold-000")


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v["walk_forward_stages"]["chain"].update(min_improvement=0),
        lambda v: v["walk_forward_stages"]["chain"].update(min_improvement=0.001),
        lambda v: v["walk_forward_stages"]["posttraining"]["candidates"].append("base_retrain"),
        lambda v: v["walk_forward_stages"]["posttraining"].update(parent="chain_previous"),
        lambda v: v["walk_forward_stages"]["rl"].update(train_windows=4),
        lambda v: v["walk_forward_stages"]["rl"].update(train="expanding_prior_evaluations_v1"),
        lambda v: v["walk_forward_stages"].update(test="validation"),
        lambda v: v["walk_forward_stages"].pop("test"),
    ],
    ids=[
        "integer_improvement",
        "loose_improvement",
        "retrain_as_candidate",
        "chain_parent",
        "four_rl_windows",
        "expanding_rl_as_main",
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
    """Construye a mano planes de adaptadores y políticas que siguen el contrato de la cadena.

    Los adaptadores llevan detrás sus selecciones de la cadena, como `plan_chain`. Solo son
    listas de trabajos para el calendario. No leen datos ni ajustan modelos.
    """
    windows = [name for name, _ in chain.scope_windows(value, JOINT)]
    fits = [
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
    chains = [
        dict(
            id=chain.chain_job_id(JOINT, window, arm, seed),
            scope=JOINT,
            window=window,
            anchor=window,
            base_arm=arm,
            seed=seed,
            stage="chain",
            depends=(
                [fits[index - 1]["id"]]
                if index
                else chain.parent_jobs(base, JOINT, window, arm, seed)
            ),
        )
        for index, window in enumerate(windows)
    ]
    # Como en la etapa de políticas de A v2, cada mercado tiene sus trabajos en el ámbito
    # conjunto, con las ventanas en las que es elegible, y lee la cadena de ese ámbito.
    eligible = value["comparison_config"]["resolved_scopes"][JOINT]["eligible"]
    policies = []
    for market in ("US", "CN"):
        names = [name for name in windows if name in eligible[market]]
        for window in names:
            rows = chain.rl_windows(names, window)
            if rows is None:
                continue
            read = [*rows["train"], rows["validation"], window]
            policies.append(
                dict(
                    id=f"{JOINT}/{market}/{window}/{arm}/ppo/fit-s7",
                    scope=JOINT,
                    market=market,
                    window=window,
                    anchor=window,
                    train=rows["train"],
                    validation=rows["validation"],
                    predictor=arm,
                    predictor_seed=seed,
                    seed=7,
                    depends=[chain.chain_job_id(JOINT, name, arm, seed) for name in read],
                )
            )
    return fits + chains, policies


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
    assert [len(row["phases"][phase["adapters"]]["jobs"]) for row in schedule] == [0] + [1] * 18
    # US empieza en fold-004 y CN, elegible desde fold-006, en fold-010.
    assert [len(row["phases"][phase["rl"]]["jobs"]) for row in schedule] == (
        [0] * 4 + [1] * 6 + [2] * 9
    )
    first_cn = next(job for job in policies if job["market"] == "CN")
    assert first_cn["window"] == "fold-010" and first_cn["train"] == [
        "fold-006",
        "fold-007",
        "fold-008",
    ]
    assert first_cn["depends"][0] == chain.chain_job_id(JOINT, "fold-006", "rnn", 42)
    # La cadena de la ventana 0 es la base y depende de las búsquedas que la eligen.
    assert chain.parent_jobs(base, JOINT, "fold-000", "rnn", 42) == searches(
        base, f"{JOINT}/fold-000/rnn"
    )
    assert chain.parent_jobs(base, JOINT, "fold-002", "rnn", 43) == [
        f"{JOINT}/fold-002/rnn/finalist-s43"
    ]
    with pytest.raises(ValueError, match="no elige el estado"):
        chain.parent_jobs(base, JOINT, "fold-002", "rnn", 45)


def test_policies_read_the_chain_of_their_scope_with_the_windows_of_the_design():
    from mars_titan.simulation import policy_plan

    stage = policy_plan.load_stage(plan.LATER_STAGES["rl_policy_comparison"]["joint_stage"])
    resolved = stage["campaign"]["comparison_config"]["resolved_scopes"][JOINT]
    # Las ventanas de cada mercado del plan de políticas son las de `rl_windows`.
    for market in resolved["markets"]:
        names = [name for name in resolved["windows"] if name in resolved["eligible"][market]]
        rows = policy_plan.scope_windows(stage, JOINT, market)
        assert [row["window"] for row in rows] == [
            name for name in names if chain.rl_windows(names, name)
        ]
        anchors = {row["window"]: row for row in rows}
        for row in rows:
            expected = chain.rl_windows(names, row["anchor"])
            anchor = anchors[row["anchor"]]
            assert (anchor["train"], anchor["validation"]) == (
                expected["train"],
                expected["validation"],
            )
    # Cada trabajo depende de la cadena de su ámbito en todo lo que lee `predictor_reads`.
    seed = stage["policies"]["predictor"]["seed"]
    for job in policy_plan.plan_stage(stage):
        reads = policy_plan.predictor_reads(stage, job)
        assert {scope for scope, _, _ in reads} == {job["scope"]}
        chains = {d for d in job["depends"] if chain.CHAIN_SUFFIX in d}
        assert chains == {chain.chain_job_id(*read, seed) for read in reads}


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
    elif kind == "policy_validates_before_its_training":
        policies[5] = dict(policies[5], train=[*policies[5]["train"][:2], policies[5]["window"]])
    elif kind == "policy_evaluates_before_validation":
        policies[5] = dict(policies[5], validation=policies[5]["window"])
    elif kind == "policy_without_predictor_seed":
        policies[0].pop("predictor_seed")
    elif kind == "policy_reads_a_later_chain":
        policies[0]["depends"].append(chain.chain_job_id(JOINT, "fold-005", "rnn", 42))
    elif kind == "policy_reads_a_chain_outside_the_plan":
        policies[0]["depends"].append(chain.chain_job_id(JOINT, "fold-000", "lstm", 42))
    elif kind == "policy_reads_the_chain_of_another_scope":
        cn = next(job for job in policies if job["market"] == "CN")
        cn["depends"] = [
            chain.chain_job_id("CN", name, "rnn", 42)
            for name in [*cn["train"], cn["validation"], cn["window"]]
        ]
    elif kind == "policy_reads_the_chain_of_another_window":
        cn = next(job for job in policies if job["market"] == "CN")
        cn["depends"][0] = chain.chain_job_id(JOINT, "fold-000", "rnn", 42)
    return value, base, dict(adapters=adapters, rl=policies)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("adapter_without_parent", "no depende del estado elegido de la base en fold-003"),
        ("adapter_with_older_parent", "no depende del estado elegido de la base en fold-003"),
        ("adapter_in_first_window", "la primera ventana no tiene posentrenamiento"),
        ("policy_missing_a_window", "no depende de la cadena de todas las ventanas"),
        ("policy_with_two_windows", "no ajusta con tres ventanas anteriores"),
        ("policy_validates_before_its_training", "no ajusta con tres ventanas anteriores"),
        ("policy_evaluates_before_validation", "no ajusta con tres ventanas anteriores"),
        ("policy_without_predictor_seed", "semilla de su predictor"),
        ("policy_reads_a_later_chain", "fase posterior"),
        ("policy_reads_a_chain_outside_the_plan", "ajena al plan"),
        ("policy_reads_the_chain_of_another_scope", "lee la cadena de otro ámbito"),
        ("policy_reads_the_chain_of_another_window", "no depende de la cadena de todas"),
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
    adapters = [job for job in adapters if job.get("stage") != "chain"]
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
    jobs = [job for job in base if job["kind"] == plan.ONLINE]
    # Solo el ámbito conjunto evalúa el control en línea: 19 ventanas por tres semillas.
    assert len(jobs) == 57 == value["online_controls"]["limits"]["max_online_jobs"]
    assert {job["scope"] for job in jobs} == {JOINT} == set(online.scopes(value))
    assert plan.count_jobs(value)["online_jobs"] == 57
    by_id = {job["id"]: job for job in jobs}
    first = by_id[f"{JOINT}/fold-004/transformer_compact_online/online-s42"]
    assert first["depends"] == [
        *searches(base, f"{JOINT}/fold-004/transformer_compact"),
        *searches(base, f"{JOINT}/fold-004/mars_titan_m1"),
    ]
    assert len(first["depends"]) == 4
    other = by_id[f"{JOINT}/fold-012/transformer_compact_online/online-s44"]
    assert other["depends"] == [
        f"{JOINT}/fold-012/transformer_compact/finalist-s44",
        f"{JOINT}/fold-012/mars_titan_m1/finalist-s44",
    ]
    assert all(
        (job["model"], job["stage"], job["regenerable"], job["family"])
        == ("neural", "online", False, plan.NEURAL)
        and job["case"]["rule"]["update_cap"] == online.CAP
        for job in jobs
    )
    # El calendario los pone tras elegir padre y tope en su ventana.
    schedule = order.window_schedule(value, plan.plan_campaign(value))
    assert sum(len(row["phases"][order.PHASES.index("online")]["jobs"]) for row in schedule) == 57
    # El control conectado por su sección ya no figura entre las familias pendientes.
    assert plan.ONLINE_CONTROL not in plan.pending_families(value)


def test_the_online_control_respects_its_job_limit(tmp_path):
    def tight(value):
        value["online_controls"]["limits"]["max_online_jobs"] = 56

    with pytest.raises(ValueError, match="max_online_jobs=56"):
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


def test_the_online_control_needs_its_parent_and_cap_where_it_is_evaluated(monkeypatch):
    value = campaign()
    section = value["online_controls"]
    arms = plan.scope_arms

    # Sin el lector en US ni en CN el control sigue siendo válido: solo se evalúa en US+CN.
    monkeypatch.setattr(
        plan,
        "scope_arms",
        lambda c, s: [a for a in arms(c, s) if s == JOINT or a != online.CAP_ARM],
    )
    assert online.declared(section, value) is section
    monkeypatch.setattr(
        plan, "scope_arms", lambda c, s: [a for a in arms(c, s) if a != online.CAP_ARM]
    )
    with pytest.raises(ValueError, match="En US\\+CN el control en línea necesita"):
        online.declared(section, value)
    monkeypatch.setattr(plan, "scope_arms", lambda c, s: [a for a in arms(c, s) if a != online.ARM])
    with pytest.raises(ValueError, match="no evalúa transformer_compact_online en ningún"):
        online.declared(section, value)
    assert online.plan_online(value, plan.plan_campaign(value)) == []


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


# Presupuesto


def test_new_rows_count_only_targets_inside_the_span_with_mature_labels(tmp_path):
    value = campaign()
    protocol = value["comparison_config"]["resolved_scopes"]["US"]["protocols"]["US"]
    folder = tmp_path / "labels" / "US" / "A"
    folder.mkdir(parents=True)
    moments = ["2004-12-31", "2005-01-03", "2005-03-30", "2005-03-31", "2005-02-01"]
    matured = ["2005-01-03", "2005-01-04", "2005-03-31", "2005-04-01", "2005-02-02"]
    targets = [0.1, 0.2, 0.3, 0.4, None]
    stamp = pa.timestamp("us", tz="UTC")
    table = pa.table(
        dict(
            prediction_at=pa.array(np.array(moments, dtype="datetime64[us]"), stamp),
            target_available_at=pa.array(np.array(matured, dtype="datetime64[us]"), stamp),
            target=pa.array(targets, pa.float64()),
        )
    )
    pq.write_table(table, folder / "labels.parquet")
    rows = budget.target_posttraining_rows(tmp_path / "labels", protocol)
    assert list(rows) == [f"fold-{i:03d}" for i in range(1, 19)]
    # En fold-001 entran las del 3 de enero y del 30 de marzo. La del 31 madura en abril
    # y la de febrero no tiene objetivo.
    assert rows["fold-001"] == 2 and sum(rows.values()) == 2


def test_declared_new_rows_cover_every_window_with_a_parent_and_add_up_by_market(tmp_path):
    value = campaign()
    report = Path("reports/data/campaign-a-v2-window-counts-20261009.json")
    fresh = budget.read_posttraining_rows(report, value)
    counts, _ = budget.read_counts(report, value)
    joint, us, cn = fresh[JOINT], fresh["US"], fresh["CN"]
    assert list(joint) == [w for w, _ in chain.scope_windows(value, JOINT)][1:]
    # El conjunto suma US y, desde fold-007, la ventana CN con los mismos tramos.
    assert joint["fold-001"] == us["fold-001"]
    assert [joint[w] - us[w] for w in list(joint)[6:]] == list(cn.values())
    assert all(fresh[s][w] < counts[s][w]["train"] for s in fresh for w in fresh[s])
    document = json.loads(report.read_text())
    document["posttraining_rows"]["CN"].pop("fold-012")
    atomic_json(tmp_path / "counts.json", document)
    with pytest.raises(ValueError, match="filas nuevas de cada ventana"):
        budget.read_posttraining_rows(tmp_path / "counts.json", value)


def test_staged_adapters_fit_only_new_rows_and_skip_the_first_window():
    from mars_titan.posttraining import campaign_stage as adapters
    from mars_titan.training.campaign_throughput import POSTTRAINING

    value = campaign()
    stage = adapters.load_stage(plan.LATER_STAGES["posttraining_adapter_matrix"]["joint_stage"])
    windows = [name for name, _ in chain.scope_windows(value, JOINT)]
    # El ajuste crece 10.000 filas por ventana, así que la cota de filas nuevas sin recuento
    # es 10.000 - 1.000 - 500 = 8.500 y el recuento declara 100.000.
    counts = {
        scope: {
            w: dict(train=1_000_000 + 10_000 * i, validation=1000, calibration=500, evaluation=2000)
            for i, (w, _) in enumerate(chain.scope_windows(value, scope))
        }
        for scope in value["scopes"]
    }
    fresh = {scope: {w: 100_000 for w in list(rows)[1:]} for scope, rows in counts.items()}
    rates = budget.uniform_rates(value, 1000.0, inference_ratio=2.0, stage=stage)
    counted = budget.project(value, counts, rates, stage=stage, fresh=fresh)
    bounded = budget.project(value, counts, rates, stage=stage)
    staged = counted["estimate"]["families"][POSTTRAINING]
    # La medida uniforme cubre las redes con caudal neuronal. Titans-MAC, la GRU candidata,
    # MARS-TITAN, CM-v1 y la cadena trivial de los tabulares quedan sin estimar.
    neural = set(rates[plan.NEURAL])
    jobs = [job for job in adapters.plan_stage(stage) if job["base_arm"] in neural]
    assert {job["window"] for job in jobs} == set(windows[1:])
    assert (
        set(staged["without_estimate"])
        == {j["base_arm"] for j in adapters.plan_stage(stage)} - neural
    )
    epochs = stage["matrix"]["budget"]["epochs"]
    fits = [job for job in jobs if job["kind"] == plan.FIT]
    frozen = [job for job in jobs if job["kind"] == adapters.FROZEN]
    parents = len({(j["scope"], j["window"], j["base_arm"], j["seed"]) for j in fits})
    assert len(frozen) == parents == staged["parent_caches"]

    def expected(train):
        per_fit = epochs * train / 1000 + ((epochs + 2) * 1000 + 2500) / 2000
        cache = (train + 1000) / 2000 + 3500 / 2000
        return len(fits) * per_fit + parents * cache

    assert (staged["training_jobs"], staged["prediction_jobs"]) == (len(fits), len(frozen))
    assert staged["fit_rows"] == "counted"
    assert math.isclose(staged["hours"] * 3600, expected(100_000))
    assert counted["adapters_design"] == "staged_chain_v1"
    assert math.isclose(counted["stages"]["adapters"], staged["hours"])
    assert bounded["adapters_fit_rows"] == "bounded_from_window_counts"
    assert math.isclose(bounded["stages"]["adapters"] * 3600, expected(8_500))
    # El control en línea predice calibración y evaluación y da como mucho un paso por fila.
    online_seconds = 57 * (2500 / 2000 + 2500 / 1000)
    assert math.isclose(counted["families"]["online_control"] * 3600, online_seconds)


# Verificador de disjunción sobre vistas preparadas


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    root = tmp_path_factory.mktemp("chain-views")
    data = fixture(root / "data", ("US", "CN"))
    campaign_path, _ = reduced(root / "config")
    checked = engine.prepare_views(campaign_path, data.parent, root / "views")
    value = plan.load_campaign(campaign_path)
    paths = {scope: root / "views" / scope for scope in checked}
    return value, paths, disjunction.verify(value, paths, workers=2)


def test_the_verifier_proves_every_disjunction_on_prepared_views(views):
    value, paths, report = views
    assert report["failures"] == [] and report["training_executed"] is False
    for scope in value["scopes"]:
        windows = report["scopes"][scope]
        assert "posttraining" not in windows["fold-000"]
        for window, summary in list(windows.items())[1:]:
            post = summary["posttraining"]
            start = micros(post["fit"][0])
            assert post["overlap"] == 0 and post["new_rows"]["rows"] > 0
            assert post["parent_max_maturity"] < start <= post["new_min_decision"]
            assert post["new_max_decision"] < micros(post["fit"][1])
            assert summary["misplaced"] == summary["after_cutoff"] == 0, window
    # Recuento independiente de las filas nuevas de US/fold-001 desde las etiquetas.
    manifest = json.loads((paths["US"] / "fold-001" / "manifest.json").read_text())
    start, end = (micros(day) for day in report["scopes"]["US"]["fold-001"]["posttraining"]["fit"])
    rows, keys = 0, []
    for path in sorted(Path(manifest["roots"]["labels"]).glob("US/*/labels.parquet")):
        table = pq.read_table(path, columns=["sample_row", "prediction_at", "partition"])
        moment = table["prediction_at"].cast(pa.int64()).to_numpy()
        train = np.asarray(table["partition"].to_pylist(), dtype=object) == "train"
        new = train & (moment >= start) & (moment < end)
        rows += int(new.sum())
        keys += [(path.parent.name, row) for row in table["sample_row"].to_numpy()[new]]
    expected = chain.row_fingerprint(["US"] * len(keys), [k[0] for k in keys], [k[1] for k in keys])
    assert report["scopes"]["US"]["fold-001"]["posttraining"]["new_rows"] == dict(
        rows=rows, sha256=expected[1]
    )
    # La evaluación de US es la misma en el ámbito conjunto y en el separado.
    joint = report["scopes"][JOINT]["fold-006"]["markets"]
    assert (
        joint["US"]["evaluation"]
        == report["scopes"]["US"]["fold-006"]["markets"]["US"]["evaluation"]
    )
    assert (
        joint["CN"]["evaluation"]
        == report["scopes"]["CN"]["fold-000"]["markets"]["CN"]["evaluation"]
    )


class Tampered:
    """Cambia en sitio el archivo de etiquetas de un activo y lo restaura al salir.

    Así cada prueba altera una sola frontera y las demás reutilizan las vistas preparadas.
    """

    def __init__(self, views, scope, window, change):
        manifest = json.loads((views[scope] / window / "manifest.json").read_text())
        asset = next(a for a in manifest["assets"] if a["market"] == "US")
        self.path = Path(manifest["roots"]["labels"]) / "US" / asset["symbol"] / "labels.parquet"
        self.change = change

    def __enter__(self):
        self.original = self.path.read_bytes()
        table = pq.read_table(self.path)
        pq.write_table(self.change(table), self.path)
        return self

    def __exit__(self, *_):
        self.path.write_bytes(self.original)


def _set(table, column, index, value):
    values = table[column].to_pylist()
    values[index] = value
    return table.set_column(
        table.schema.get_field_index(column), column, pa.array(values, table[column].type)
    )


def _first(table, partition, after=None):
    moments = table["prediction_at"].to_pylist()
    for index, name in enumerate(table["partition"].to_pylist()):
        if name == partition and (after is None or moments[index].isoformat() >= after):
            return index
    raise AssertionError(partition)


def test_the_verifier_detects_a_parent_row_inside_the_new_rows(views):
    value, paths, _ = views

    def into_parent(table):
        # La primera fila de evaluación de 2005 pasa a la calibración del padre.
        return _set(table, "partition", _first(table, "evaluation"), "calibration")

    with Tampered(paths, "US", "fold-000", into_parent):
        failures = disjunction.verify(value, paths, workers=2)["failures"]
    assert "US/fold-000: 1 filas fuera de su tramo" in failures
    assert "US/fold-001: 1 filas nuevas ya las usó el padre" in failures
    assert "US/fold-001: una etiqueta del padre madura con las filas nuevas" in failures


def test_the_verifier_detects_a_row_of_2024(views):
    value, paths, _ = views

    def late(table):
        index = _first(table, "evaluation")
        moment = np.datetime64("2024-01-03", "us").astype(object)
        return _set(table, "target_available_at", index, moment)

    with Tampered(paths, "US", "fold-018", late):
        failures = disjunction.verify(value, paths, workers=2)["failures"]
    assert "US/fold-018: 1 filas de 2024" in failures
    assert "US/fold-018: 1 filas fuera de su tramo" in failures


def test_the_verifier_detects_different_test_rows_between_scopes(views):
    value, paths, _ = views

    def dropped(table):
        return _set(table, "partition", _first(table, "evaluation"), None)

    with Tampered(paths, "US", "fold-006", dropped):
        failures = disjunction.verify(value, paths, workers=2)["failures"]
    assert any(f.startswith("US: evaluaciones distintas") and "US/fold-006" in f for f in failures)


def test_the_verifier_detects_base_receipts_with_other_test_rows(views, tmp_path):
    value, paths, _ = views
    for name, digest in (("a", "1" * 64), ("b", "2" * 64)):
        atomic_json(
            tmp_path / "jobs" / JOINT / "fold-001" / name / "receipt.json",
            dict(
                identity=dict(id=name, scope=JOINT, window="fold-001"),
                predictions=dict(evaluation=dict(rows_sha256=digest)),
            ),
        )
    report = disjunction.verify(value, paths, campaign_output=tmp_path, workers=2)
    assert report["failures"] == [f"{JOINT}/fold-001: los recibos base evalúan filas distintas"]
    assert report["base_receipts"][f"{JOINT}/fold-001"]["jobs"] == 2


def write_selection(root, value, report, *, change=None, receipt_change=None):
    """Escribe a mano la selección de la cadena de US/fold-001 para gru y su recibo de US.

    El candidato elegido es un adaptador. `change` y `receipt_change` alteran el documento
    o el recibo antes de escribirlos para probar cada regla del contrato.
    """
    scope, window, arm, seed = "US", "fold-001", "gru", 42
    summary = report["scopes"][scope][window]
    resolved = value["comparison_config"]["resolved_scopes"][scope]
    folder = chain.chain_folder(root, scope, window, arm, seed)
    until = summary["markets"]["US"]["max_maturity"]["calibration"]
    job = f"{scope}/{window}/gru__head/fit-s42"
    receipt = dict(
        kind=RECEIPT_KIND,
        schema_version=1,
        protocol=resolved["protocols"]["US"],
        fold=resolved["windows"][window],
        parent=dict(id=job, sha256="a" * 64),
        labels_used_until=until,
        predictions=dict(
            evaluation=dict(rows=summary["markets"]["US"]["evaluation"]["rows"], sha256="b" * 64)
        ),
    )
    if receipt_change:
        receipt_change(receipt)
    atomic_json(folder / "US.json", receipt)
    post = summary["posttraining"]
    document = dict(
        kind=chain.SELECTION_KIND,
        schema_version=1,
        campaign_sha256=value["sha256"],
        stage_sha256="c" * 64,
        scope=scope,
        window=window,
        base_arm=arm,
        seed=seed,
        rule=chain.RULE,
        parent_window="fold-000",
        parent=dict(
            job="US/fold-000/gru/search-0", receipt_sha256="d" * 64, checkpoint_sha256="e" * 64
        ),
        candidates=[
            candidate("frozen_parent", "US/fold-001/gru__frozen_parent/carry-s42", 1.0),
            dict(candidate("adapter", job, 0.5), receipt_sha256="a" * 64),
        ],
        selected=dict(kind="adapter", arm="gru__head", job=job, receipt_sha256="a" * 64),
        state=dict(path="jobs/x/state.pt", sha256="f" * 64),
        fit_rows=dict(
            first_decision=post["new_min_decision"],
            last_decision=post["new_max_decision"],
            **post["new_rows"],
        ),
        markets=dict(US=sha256(folder / "US.json")),
        labels_used_until=receipt["labels_used_until"],
        confirmed_at_utc="2026-10-10T00:00:00+00:00",
    )
    if change:
        change(document)
    atomic_json(folder / chain.SELECTION, document)
    return folder


def test_a_chain_selection_that_follows_the_contract_passes(views, tmp_path):
    value, paths, report = views
    write_selection(tmp_path, value, report)
    checked = disjunction.verify(value, paths, posttraining=tmp_path, workers=2)
    assert checked["failures"] == []
    tapes = checked["selections"]["US/fold-001/gru__chain/seed-42"]["tapes"]["US"]
    assert tapes["labels_used_until"] < tapes["first_decision"]
    document = chain.read_selection(tmp_path, "US", "fold-001", "gru", 42)
    assert document["selected"]["kind"] == "adapter" and set(document["receipts"]) == {"US"}
    assert chain.read_selection(tmp_path, "US", "fold-002", "gru", 42) is None


@pytest.mark.parametrize(
    ("change", "receipt_change", "message"),
    [
        (
            None,
            lambda r: r.update(labels_used_until=r["labels_used_until"] - 1),
            "anterior a las que usó",
        ),
        (
            lambda d: d["fit_rows"].update(rows=d["fit_rows"]["rows"] - 1),
            None,
            "las filas de ajuste no son las filas nuevas",
        ),
        (
            None,
            lambda r: r["predictions"]["evaluation"].update(rows=1),
            "no tiene las filas de la vista",
        ),
    ],
    ids=["label_before_calibration", "other_fit_rows", "other_test_rows"],
)
def test_the_verifier_rejects_chain_receipts_that_break_a_frontier(
    views, tmp_path, change, receipt_change, message
):
    value, paths, report = views
    write_selection(tmp_path, value, report, change=change, receipt_change=receipt_change)
    failures = disjunction.verify(value, paths, posttraining=tmp_path, workers=2)["failures"]
    assert len(failures) == 1 and message in failures[0]


@pytest.mark.parametrize(
    ("change", "receipt_change", "message"),
    [
        (
            lambda d: d.update(
                selected=dict(
                    kind="frozen_parent",
                    arm="gru",
                    job=d["candidates"][0]["job"],
                    receipt_sha256="0" * 64,
                )
            ),
            None,
            "no sigue la regla",
        ),
        (lambda d: d["selected"].update(kind="continuation"), None, "no sigue la regla"),
        (lambda d: d.update(fit_rows=None), None, "no sigue la regla"),
        (lambda d: d.update(parent_window=None), None, "elige el estado de la base"),
        (lambda d: d.update(parent_window="fold-005"), None, "no parte de la ventana anterior"),
        (lambda d: d.update(rule="lowest_test_error"), None, "no cumple su contrato"),
        (lambda d: d["markets"].update(US="0" * 64), None, "no corresponde a su selección"),
        (None, lambda r: r["parent"].update(id="other"), "no es el del predictor elegido"),
        (
            lambda d: d.update(labels_used_until=d["labels_used_until"] + 1),
            None,
            "no es el del predictor elegido",
        ),
    ],
    ids=[
        "frozen_without_better_score",
        "kind_of_another_candidate",
        "adapter_without_fit_rows",
        "first_window_with_candidates",
        "other_parent_window",
        "other_rule",
        "changed_receipt",
        "receipt_of_another_job",
        "different_last_label",
    ],
)
def test_reading_a_chain_selection_rejects_a_broken_contract(
    views, tmp_path, change, receipt_change, message
):
    value, paths, report = views
    write_selection(tmp_path, value, report, change=change, receipt_change=receipt_change)
    with pytest.raises(ValueError, match=message):
        disjunction.verify(value, paths, posttraining=tmp_path, workers=2)


def test_the_cli_writes_the_report_and_fails_with_any_finding(views, tmp_path):
    value, paths, _ = views
    argv = ["--campaign", value["path"], "--output", str(tmp_path / "report.json")]
    argv += [f"--views={scope}={path}" for scope, path in paths.items()] + ["--workers", "2"]
    assert disjunction.main(argv) == 0
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["kind"] == disjunction.KIND and written["failures"] == []
    folder = tmp_path / "jobs" / JOINT / "fold-002"
    for name, digest in (("a", "1" * 64), ("b", "2" * 64)):
        atomic_json(
            folder / name / "receipt.json",
            dict(
                identity=dict(id=name, scope=JOINT, window="fold-002"),
                predictions=dict(evaluation=dict(rows_sha256=digest)),
            ),
        )
    assert disjunction.main([*argv, "--campaign-output", str(tmp_path)]) == 1


def test_the_verifier_can_check_some_scopes_only(views):
    value, paths, _ = views
    report = disjunction.verify(value, {"US": paths["US"]}, scopes=["US"], workers=2)
    assert list(report["scopes"]) == ["US"] and report["failures"] == []
    with pytest.raises(ValueError, match="no es de la campaña"):
        disjunction.verify(value, paths, scopes=["EU"], workers=2)
