"""Pruebas de la parada temprana que declara la campaña, en modo individual, en grupo y en el plan.

Estas pruebas no leen vistas ni datos, no reservan la GPU y no ajustan ningún modelo.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from mars_titan.training import campaign_plan as plan
from mars_titan.training.selection import JOINT_PLATEAU, VALIDATION_PLATEAU
from tests.training.test_campaign_plan import CAMPAIGNS, write_variant

JOINT_CONFIG = Path("configs/baselines/historical-masked-campaign-a-joint-stop.json")
EXTENSIONS = Path("configs/baselines/historical-masked-campaign-extensions.json")
TITANS_GROUP = [
    "titans_transformer_direct",
    "titans_mac_disabled",
    "titans_mac_frozen",
    "titans_mac_online",
]
# La comparación contrasta el Transformer compacto con Titans-MAC `transformer_direct` y la
# GRU con la GRU episódica, que a su vez se contrasta con `mac_online`. Sin las ampliaciones,
# la GRU episódica no tiene entrenador y el grupo conectado tiene seis brazos.
ENCODER_GROUP = ["gru", "transformer_compact", *TITANS_GROUP]
EARLY = dict(
    stopping=JOINT_PLATEAU,
    patience=5,
    min_delta=1e-05,
    minimum_epochs=5,
    max_epochs=30,
    group_epoch=plan.GROUP_EPOCH,
    groups=dict(encoders_and_cores=ENCODER_GROUP),
)
RULE = dict(metric="session_mae", patience=5, min_delta=1e-05, minimum_epochs=5, max_epochs=30)


def extended(campaign):
    sections = json.loads(EXTENSIONS.read_text())["sections"]
    return plan.extend_campaign(
        campaign, sections, limits=dict(max_training_jobs=5265, max_prediction_jobs=0)
    )


def test_joint_campaign_a_keeps_the_design_and_declares_its_rule_before_results():
    joint, declared = (json.loads(path.read_text()) for path in (JOINT_CONFIG, CAMPAIGNS["A"]))
    # Solo cambian el nombre y la parada. Ventanas, brazos, semillas y límites son los mismos.
    assert {key for key in joint if joint.get(key) != declared.get(key)} == {"name", "early_stop"}
    assert joint["status"] == plan.DECLARED and joint["final_test_opened"] is False
    report = plan.check_campaign(JOINT_CONFIG)
    assert report["protocol_stopping_rule"]["stopping"] == "fixed_budget"
    assert report["stopping_rule"] == dict(RULE, stopping=JOINT_PLATEAU)
    assert report["early_stop"]["individual_rule"] == dict(RULE, stopping=VALIDATION_PLATEAU)
    assert report["early_stop"]["group_epoch"] == "maximum_of_first_plateaus"
    counts = report["counts"]
    # Son los mismos 2385 ajustes completos que en A. Los 1080 del grupo se dividen en meseta
    # y continuación, y la meseta no cuenta como ajuste adicional.
    assert (counts["training_jobs"], counts["prediction_jobs"]) == (2385, 0)
    # Son seis brazos por 45 ventanas reentrenadas (19 + 13 + 13) por 4 ajustes.
    assert counts["plateau_jobs"] == 6 * (19 + 13 + 13) * 4 == 1080


def test_extended_joint_campaign_groups_mars_titan_and_cm_v1():
    campaign = extended(plan.load_campaign(JOINT_CONFIG))
    counts = plan.count_jobs(campaign)
    assert (counts["training_jobs"], counts["prediction_jobs"]) == (5265, 0)
    # El grupo de codificadores y núcleos suma siete brazos con la GRU episódica, el de los
    # lectores episódicos once y el de los núcleos de CM-v1 dos. Son 20 brazos con 4 ajustes en
    # 45 ventanas. Los dos brazos B6 no tienen épocas y quedan fuera de los grupos.
    assert counts["plateau_jobs"] == 20 * 4 * 45 == 3600
    jobs = plan.plan_campaign(campaign)
    grouped = {job["arm"] for job in jobs if job.get("phase") == plan.PLATEAU}
    assert grouped == set(campaign["early_stop"]["membership"])
    assert campaign["early_stop"]["membership"]["gru_episodic"] == "encoders_and_cores"
    # El contraste de M1 con CM-v1 B obliga a agrupar los lectores de MARS-TITAN y CM-v1.
    assert campaign["early_stop"]["membership"]["cm_v1_b"] == "episodic_readers"
    for job in jobs:
        if job["arm"] == "gru_episodic" and job["stage"] == "search":
            assert job["case"]["stopping_rule"] == dict(RULE, stopping=JOINT_PLATEAU)
            assert job["phase"] in (plan.PLATEAU, plan.JOINT)
    corrections = [job for job in jobs if job["arm"] in ("mars_titan_b6", "mars_titan_b6_bias")]
    assert corrections and all(
        "phase" not in job and "stopping_rule" not in (job.get("case") or {}) for job in corrections
    )


def test_the_b6_correction_cannot_join_a_stopping_group():
    """B6 no tiene épocas, así que una parada conjunta no puede decidir nada sobre ella."""
    campaign = extended(plan.load_campaign(JOINT_CONFIG))
    early = campaign["early_stop"]
    early["groups"]["episodic_readers"].append("mars_titan_b6")
    early["membership"]["mars_titan_b6"] = "episodic_readers"
    with pytest.raises(ValueError, match="no puede pertenecer a un grupo"):
        plan.plan_campaign(campaign)


def test_each_grouped_fit_splits_into_plateau_and_continuation_in_dependency_order():
    campaign = plan.load_campaign(JOINT_CONFIG)
    jobs = plan.plan_campaign(campaign)
    position = {job["id"]: index for index, job in enumerate(jobs)}
    assert len(position) == len(jobs)
    for job in jobs:
        assert all(position[dependency] < position[job["id"]] for dependency in job["depends"])
    plateaus = {job["id"]: job for job in jobs if job.get("phase") == plan.PLATEAU}
    finals = [job for job in jobs if job.get("phase") == plan.JOINT]
    assert len(plateaus) == len(finals) == 1080
    for final in finals:
        plateau = plateaus[final["plateau"]]
        head, _, name = final["id"].rpartition("/")
        assert plateau["id"] == f"{head}/plateau-{name}"
        # La meseta repite el mismo ajuste, con su caso, semilla, ventana y dependencias.
        assert {k: v for k, v in plateau.items() if k not in ("id", "phase")} == {
            k: v
            for k, v in final.items()
            if k not in ("id", "phase", "plateau", "joint_group", "depends")
        } | dict(depends=plateau["depends"])
        assert final["depends"] == plateau["depends"] + final["joint_group"]
        members = [plateaus[key] for key in final["joint_group"]]
        assert sorted(member["arm"] for member in members) == sorted(ENCODER_GROUP)
        # Cada familia nombra sus casos a su manera y el grupo los empareja por posición.
        assert {(m["scope"], m["window"], m["seed"], m["stage"]) for m in members} == {
            (final["scope"], final["window"], final["seed"], final["stage"])
        }
        # Los finalistas heredan el caso, con su regla, del ganador de la búsqueda.
        if final["stage"] != "search":
            assert final["case"] is None
        elif final["family"] == plan.NEURAL:
            assert final["case"]["selection"]["stopping"] == JOINT_PLATEAU
        else:
            assert final["case"]["stopping_rule"] == dict(RULE, stopping=JOINT_PLATEAU)
    # Hay un grupo por ámbito, ventana reentrenada, semilla y caso, 45 x (2 casos + 2 finalistas).
    groups = Counter(tuple(final["joint_group"]) for final in finals)
    assert len(groups) == 45 * 4 and set(groups.values()) == {6}
    # Las referencias sin contraste emparejado y los tabulares conservan sus trabajos y la
    # parada individual.
    neural = [
        job for job in jobs if job["family"] == plan.NEURAL and job["arm"] not in ENCODER_GROUP
    ]
    assert neural and all("phase" not in job for job in neural)
    searches = [job["case"] for job in neural if job["stage"] == "search"]
    assert {case["selection"]["stopping"] for case in searches} == {VALIDATION_PLATEAU}
    assert {case["epochs"] for case in searches} == {30}


def test_finalists_wait_for_the_selected_cases_and_carry_their_own_continuation():
    jobs = {job["id"]: job for job in plan.plan_campaign(plan.load_campaign(JOINT_CONFIG))}
    final = jobs["US/fold-000/titans_mac_online/finalist-s43"]
    assert final["phase"] == plan.JOINT
    searches = [f"US/fold-000/titans_mac_online/search-{case}" for case in ("lr1e-4", "lr1e-3")]
    assert jobs[final["plateau"]]["depends"] == searches
    assert sorted(final["joint_group"]) == sorted(
        f"US/fold-000/{arm}/plateau-finalist-s43" for arm in ENCODER_GROUP
    )


def test_individual_mode_declares_no_group_and_no_plateau_job(tmp_path):
    early = {k: v for k, v in EARLY.items() if k not in ("groups", "group_epoch")}
    path = write_variant(tmp_path, "A", early_stop=dict(early, stopping=VALIDATION_PLATEAU))
    report = plan.check_campaign(path)
    assert report["stopping_rule"] == dict(RULE, stopping=VALIDATION_PLATEAU)
    assert report["early_stop"]["joint_rule"] is None and report["early_stop"]["groups"] == {}
    assert "plateau_jobs" not in report["counts"]
    assert report["counts"]["training_jobs"] == 2385
    jobs = plan.plan_campaign(plan.load_campaign(path))
    titans = [job for job in jobs if job["family"] == plan.TITANS and job["stage"] == "search"]
    assert titans and all(
        job["case"]["stopping_rule"] == dict(RULE, stopping=VALIDATION_PLATEAU) for job in titans
    )


def test_campaign_without_early_stop_keeps_the_protocol_jobs_and_identities():
    campaign = plan.load_campaign(CAMPAIGNS["A"])
    assert campaign["early_stop"] is None and campaign["rule"] == campaign["protocol_rule"]
    jobs = plan.plan_campaign(campaign)
    assert all("phase" not in job and "stopping_rule" not in (job["case"] or {}) for job in jobs)
    assert plan.check_campaign(CAMPAIGNS["A"])["early_stop"] is None


@pytest.mark.parametrize(
    ("early", "message"),
    [
        (dict(EARLY, stopping="fixed_budget"), "modo individual o conjunto"),
        ({k: v for k, v in EARLY.items() if k != "patience"}, "modo individual o conjunto"),
        (dict(EARLY, extra=1), "modo individual o conjunto"),
        (dict(EARLY, max_epochs=0), "métrica"),
        (dict(EARLY, minimum_epochs=30), "mínimo de épocas"),
        (dict(EARLY, group_epoch="minimum_of_plateaus"), "época común"),
        ({k: v for k, v in EARLY.items() if k != "groups"}, "época común"),
        (dict(EARLY, groups=dict(one=["titans_mac_online"])), "época común"),
        (dict(EARLY, groups=dict(a=TITANS_GROUP[:2], b=TITANS_GROUP[1:])), "un solo grupo"),
        (dict(EARLY, groups=dict(a=["titans_mac_online", "ridge"])), "un solo grupo"),
        (dict(EARLY, groups=dict(a=["titans_mac_online", "zero"])), "un solo grupo"),
        (dict(EARLY, groups=dict(a=["titans_mac_online", "unknown"])), "un solo grupo"),
        (dict(EARLY, groups=dict(a=TITANS_GROUP + TITANS_GROUP[:1])), "época común"),
        (dict(EARLY, stopping=VALIDATION_PLATEAU), "no declara grupos"),
    ],
)
def test_early_stop_section_rejects_an_undeclared_or_ambiguous_rule(tmp_path, early, message):
    with pytest.raises(ValueError, match=message):
        plan.load_campaign(write_variant(tmp_path, "A", early_stop=early))


def test_paired_contrasts_must_share_a_group(tmp_path):
    # Los cuatro controles de Titans-MAC forman contrastes emparejados entre sí.
    split = dict(EARLY, groups=dict(a=ENCODER_GROUP[:4], b=ENCODER_GROUP[4:]))
    with pytest.raises(ValueError, match="mismo grupo de parada conjunta"):
        plan.plan_campaign(plan.load_campaign(write_variant(tmp_path, "A", early_stop=split)))
    # Dos referencias neuronales también pueden formar un grupo con la regla conjunta.
    other = dict(EARLY, groups=dict(a=ENCODER_GROUP[1:], b=["gru", "lstm"]))
    campaign = plan.load_campaign(write_variant(tmp_path, "A", early_stop=other))
    for name in ("gru", "lstm"):
        assert campaign["neural"]["candidates"][name][0][1]["selection"]["stopping"] == (
            JOINT_PLATEAU
        )
    jobs = plan.plan_campaign(campaign)
    gru = [job for job in jobs if job["arm"] == "gru" and job.get("phase") == plan.JOINT]
    # Las referencias se emparejan por posición del caso, con nombres propios de cada familia.
    first = next(job for job in gru if job["stage"] == "search")
    assert sorted(first["joint_group"]) == sorted(
        f"US/fold-000/{arm}/plateau-search-{arm}-{first['candidate'].split('-')[1]}"
        for arm in ("gru", "lstm")
    )


def test_parent_and_child_cannot_share_a_group(tmp_path):
    campaign = extended(plan.load_campaign(JOINT_CONFIG))
    membership = dict(campaign["early_stop"]["membership"], mars_titan_m1="encoders_and_cores")
    groups = {}
    for arm, group in membership.items():
        groups.setdefault(group, []).append(arm)
    broken = dict(
        campaign, early_stop=dict(campaign["early_stop"], groups=groups, membership=membership)
    )
    with pytest.raises(ValueError, match="sin padres dentro"):
        plan.plan_campaign(broken)
