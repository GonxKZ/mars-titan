"""Declaración preparada de la GRU candidata, MARS-TITAN y CM-v1, sin activarla.

Se comprueba que los límites preparados son los recuentos exactos de la campaña, de la
etapa de adaptadores y de la etapa de políticas, que ampliar una campaña cargada equivale
a declarar las secciones en su archivo y que las configuraciones A y B no cambian. No se
leen vistas, no se usa la GPU y no se ajusta nada.
"""

import json
from pathlib import Path

import pytest

from mars_titan.data.storage import atomic_json
from mars_titan.posttraining import campaign_stage as adapters
from mars_titan.simulation import policy_plan
from mars_titan.training import campaign_extensions as extensions
from mars_titan.training import campaign_plan as plan
from tests.training.test_campaign_plan import CAMPAIGNS, write_variant

DECLARATION = Path("configs/baselines/historical-masked-campaign-extensions.json")
RL_STAGES = {
    v: Path(f"configs/simulation/historical-masked-rl-stage-{v.lower()}.json") for v in "AB"
}
ADDED = (plan.EPISODIC, plan.MARS, plan.CM)
# Ajustes y traslados por variante: campaña declarada, cada familia añadida y total.
COUNTS = dict(
    A=dict(
        declared=(2385, 0),
        families={plan.EPISODIC: (180, 0), plan.MARS: (1620, 0), plan.CM: (1080, 0)},
        extended=(5265, 0),
        adapters=(10332, 1302),
        rl=dict(
            declared=dict(training_jobs=1800, carried_jobs=0, reference_jobs=1221),
            extended=dict(training_jobs=2808, carried_jobs=0, reference_jobs=2775),
        ),
    ),
    B=dict(
        declared=(901, 868),
        families={plan.EPISODIC: (68, 84), plan.MARS: (612, 756), plan.CM: (408, 336)},
        extended=(1989, 2044),
        adapters=(1479, 2436),
        rl=dict(
            declared=dict(training_jobs=600, carried_jobs=1200, reference_jobs=1221),
            extended=dict(training_jobs=936, carried_jobs=1872, reference_jobs=2775),
        ),
    ),
)


def pair(counts):
    return counts["training_jobs"], counts["prediction_jobs"]


@pytest.mark.parametrize("variant", ["A", "B"])
def test_prepared_limits_are_the_exact_counts_of_the_campaign_and_both_stages(variant):
    expected = COUNTS[variant]
    report = extensions.check_variant(extensions.load_extensions(DECLARATION), variant)
    campaign = report["campaign"]
    assert pair(campaign["declared"]) == expected["declared"]
    assert pair(campaign["extended"]) == expected["extended"]
    limits = campaign["limits"]
    assert (limits["max_training_jobs"], limits["max_prediction_jobs"]) == expected["extended"]
    added = [expected["families"][family] for family in ADDED]
    assert tuple(map(sum, zip(expected["declared"], *added, strict=True))) == expected["extended"]
    # Con las tres secciones solo queda pendiente el control en línea, cuyos trabajos declara
    # la campaña A por etapas. M3 tampoco queda pendiente.
    assert set(campaign["pending_families"]) == {plan.ONLINE_CONTROL}
    assert pair(report["adapter_stage"]["counts"]) == expected["adapters"]
    rl = report["rl_stage"]
    assert rl["predictors"] == dict(declared=11, extended=25)
    for name in ("declared", "extended"):
        counts = expected["rl"][name]
        assert {key: rl[name][key] for key in counts} == counts
        assert rl[name]["evaluation_jobs"] == counts["carried_jobs"] + counts["reference_jobs"]
    assert rl["limits"] == dict(
        max_training_jobs=rl["extended"]["training_jobs"],
        max_evaluation_jobs=rl["extended"]["evaluation_jobs"],
    )
    assert rl["declared_limits"] == json.loads(RL_STAGES[variant].read_text())["limits"]


@pytest.mark.parametrize("variant", ["A", "B"])
def test_each_added_family_brings_its_own_jobs(variant):
    prepared = extensions.load_extensions(DECLARATION)
    campaign = extensions.extended_campaign(prepared, plan.load_campaign(CAMPAIGNS[variant]))
    jobs = plan.plan_campaign(campaign)
    for family in ADDED:
        selected = [job for job in jobs if job["family"] == family]
        counted = (
            sum(job["kind"] == plan.FIT for job in selected),
            sum(job["kind"] == plan.CARRY for job in selected),
        )
        assert counted == COUNTS[variant]["families"][family]
        assert campaign[family]["declared_in_campaign"] is False
    # La campaña cargada no cambia: misma huella y sin las secciones añadidas.
    declared = plan.load_campaign(CAMPAIGNS[variant])
    assert campaign["sha256"] == declared["sha256"] and campaign["path"] == declared["path"]
    assert not any(declared.get(family) for family in ADDED)


@pytest.mark.parametrize("variant", ["A", "B"])
def test_extending_a_campaign_equals_declaring_the_sections_in_its_file(tmp_path, variant):
    prepared = extensions.load_extensions(DECLARATION)
    sections = {}
    for family, section in prepared["sections"].items():
        keys = ["declaration" if family == plan.CM else "recipe"]
        keys += ["correction_recipe"] if "correction_recipe" in section else []
        sections[family] = dict(
            section, **{key: str((DECLARATION.parent / section[key]).resolve()) for key in keys}
        )
    limits = prepared["variants"][variant]["limits"]
    declared = plan.load_campaign(write_variant(tmp_path, variant, **sections, limits=limits))
    extended = extensions.extended_campaign(prepared, plan.load_campaign(CAMPAIGNS[variant]))
    assert plan.plan_campaign(extended) == plan.plan_campaign(declared)
    assert plan.count_jobs(extended) == plan.count_jobs(declared)
    for family in ADDED:
        resolved = dict(extended[family])
        assert resolved.pop("declared_in_campaign") is False
        # Solo cambia la ruta escrita: relativa a la carpeta de la campaña o absoluta.
        raw = ["declaration" if family == plan.CM else "recipe"]
        raw += ["correction_recipe"] if "correction_recipe" in resolved else []
        for key in raw:
            assert resolved.pop(key) != declared[family][key]
        assert resolved == {key: value for key, value in declared[family].items() if key not in raw}
    assert plan.check_campaign(write_variant(tmp_path, variant, **sections, limits=limits))[
        "pending_families"
    ] == plan.pending_families(extended)


def test_the_declared_configurations_stay_untouched():
    for variant in "AB":
        campaign = plan.load_campaign(CAMPAIGNS[variant])
        assert not any(campaign.get(family) for family in ADDED)
        assert set(plan.pending_families(campaign)) == {*ADDED, plan.ONLINE_CONTROL}
        counts = plan.count_jobs(campaign)
        assert (counts["training_jobs"], counts["prediction_jobs"]) == COUNTS[variant]["declared"]
        stage = policy_plan.load_stage(RL_STAGES[variant])
        assert len(stage["predictors"]) == 11
        assert (
            policy_plan.count_stage(stage)["training_jobs"] == stage["limits"]["max_training_jobs"]
        )
    report = extensions.check_extensions(DECLARATION)
    assert report["status"] == "checked" and report["sections"] == sorted(ADDED)
    assert set(report["variants"]) == {"A", "B"} and report["final_test_opened"] is False


@pytest.mark.parametrize(
    ("where", "value", "message"),
    [
        (("limits", "max_training_jobs"), 5264, "prevé 5265 trabajos.*max_training_jobs=5264"),
        (("limits", "max_prediction_jobs"), 1, "prevé 0 trabajos.*max_prediction_jobs=1"),
        (("rl_stage", "limits", "max_training_jobs"), 2809, "políticas ampliada prevé 2808"),
        (("rl_stage", "limits", "max_evaluation_jobs"), 2774, "prevé 2775 trabajos"),
    ],
)
def test_prepared_limits_must_match_the_counts_exactly(where, value, message):
    prepared = extensions.load_extensions(DECLARATION)
    target = prepared["variants"]["A"]
    for key in where[:-1]:
        target = target[key]
    target[where[-1]] = value
    with pytest.raises(ValueError, match=message):
        extensions.check_variant(prepared, "A")


def test_the_adapter_stage_must_not_change_with_the_added_families(monkeypatch):
    prepared = extensions.load_extensions(DECLARATION)
    count = adapters.count_stage

    def counted(stage, jobs=None):
        result = count(stage, jobs)
        if stage["campaign"].get(plan.MARS):
            result = dict(result, training_jobs=result["training_jobs"] + 1)
        return result

    monkeypatch.setattr(adapters, "count_stage", counted)
    with pytest.raises(ValueError, match="adaptadores no declara las familias ampliadas"):
        extensions.check_variant(prepared, "A")


def test_extended_policy_stage_resolves_every_producer_of_the_campaign():
    prepared = extensions.load_extensions(DECLARATION)
    campaign = plan.load_campaign(CAMPAIGNS["B"])
    extended = extensions.extended_campaign(prepared, campaign)
    stage = policy_plan.load_stage(RL_STAGES["B"])
    resolved = extensions.extended_policies(prepared, stage, extended)
    producers = [spec["arm"] for spec in plan._arm_specs(extended) if not spec["helper"]]
    assert resolved["predictors"] == producers and len(producers) == 25
    assert "mars_titan_m3" in producers
    assert not set(plan.CM_CORES) & set(resolved["predictors"])
    assert resolved["universe_predictor"] == stage["universe_predictor"]
    assert resolved["limits"] == prepared["variants"]["B"]["rl_stage"]["limits"]
    # Sin límites nuevos conserva los suyos, y con la campaña de otra variante se rechaza.
    assert extensions.extended_policy_stage(stage, extended)["limits"] == stage["limits"]
    other = extensions.extended_campaign(prepared, plan.load_campaign(CAMPAIGNS["A"]))
    with pytest.raises(ValueError, match="no parte de esta campaña"):
        extensions.extended_policy_stage(stage, other)
    with pytest.raises(ValueError, match="enteros declarados"):
        extensions.extended_policy_stage(stage, extended, dict(max_training_jobs=1))


def test_extending_rejects_sections_already_declared_or_foreign(tmp_path):
    prepared = extensions.load_extensions(DECLARATION)
    campaign = plan.load_campaign(CAMPAIGNS["A"])
    titans = json.loads(CAMPAIGNS["A"].read_text())[plan.TITANS]
    with pytest.raises(ValueError, match="ya declara titans_mac"):
        plan.extend_campaign(campaign, {plan.TITANS: titans})
    for sections in ({}, {"titans": {}}, {plan.EPISODIC: None}):
        with pytest.raises(ValueError, match="Solo se añaden secciones opcionales"):
            plan.extend_campaign(campaign, sections)
    with pytest.raises(ValueError, match="enteros declarados"):
        plan.extend_campaign(
            campaign, prepared["sections"], limits=dict(max_training_jobs=1, other=2)
        )
    # Las secciones se validan con las reglas de la campaña: un padre que no es mac_online.
    mars = dict(prepared["sections"][plan.MARS], parent_arm="titans_mac_frozen")
    with pytest.raises(ValueError, match="padre el brazo mac_online"):
        plan.extend_campaign(campaign, {plan.MARS: mars})
    # Una copia de la campaña fuera de su carpeta no es la que amplía la declaración.
    copied = plan.load_campaign(write_variant(tmp_path, "A"))
    with pytest.raises(ValueError, match="no amplía esta campaña"):
        extensions.extended_campaign(prepared, copied)


def test_the_core_recipe_of_cm_v1_admits_block_accumulation(tmp_path):
    """C acumula por bloques, así que el núcleo conserva la receta de Titans-MAC con 128.

    Lo que sigue sin admitirse es una receta de núcleo que no cumple la regla común.
    """
    core = json.loads(
        Path("configs/titans/chronological-training-historical-masked.json").read_text()
    )
    declaration = json.loads(Path("configs/titans/cm-v1-factorial.json").read_text())
    readout = Path("configs/titans/episodic-readout-historical-masked.json").resolve()
    declaration["base"].update(core_recipe=str(tmp_path / "core.json"), readout_recipe=str(readout))
    atomic_json(tmp_path / "cm.json", declaration)
    section = dict(declaration=str(tmp_path / "cm.json"), search_seed=42)
    limits = dict(max_training_jobs=100_000, max_prediction_jobs=100_000)
    path = write_variant(tmp_path, "A", cm_v1=section, limits=limits)
    for rows in (128, None):
        core["recipe"]["accumulation_rows"] = rows
        atomic_json(tmp_path / "core.json", core)
        loaded = plan.load_campaign(path)[plan.CM]
        assert loaded["recipes"]["core_recipe"] == str(tmp_path / "core.json")
    core["recipe"].update(accumulation_rows=128, epochs=core["recipe"]["epochs"] + 1)
    atomic_json(tmp_path / "core.json", core)
    with pytest.raises(ValueError, match="regla de parada del protocolo"):
        plan.load_campaign(path)


def test_the_prepared_file_keeps_its_schema_and_its_folder(tmp_path):
    document = json.loads(DECLARATION.read_text())
    # Fuera de la carpeta de las campañas, sus rutas relativas cambiarían de significado.
    absolute = {
        name: dict(entry, campaign=str(CAMPAIGNS[name].resolve()))
        for name, entry in document["variants"].items()
    }
    for variants in (document["variants"], absolute):
        atomic_json(tmp_path / "copy.json", dict(document, variants=variants))
        with pytest.raises(ValueError, match="carpeta de las campañas"):
            extensions.load_extensions(tmp_path / "copy.json")
    for change in (
        dict(status="declared_not_executed"),
        dict(final_test_opened=True),
        dict(sections={"titans": {}}),
        dict(extra=1),
        dict(variants={"C": document["variants"]["A"]}),
        dict(variants={"A": dict(document["variants"]["A"], rl_stage={"path": "x"})}),
    ):
        atomic_json(tmp_path / "changed.json", document | change)
        with pytest.raises(ValueError, match="esquema|variante preparada"):
            extensions.load_extensions(tmp_path / "changed.json")


def test_the_check_command_prints_both_variants(capsys):
    import runpy

    script = runpy.run_path("scripts/run_masked_campaign.py", run_name="script")
    assert script["main"](["extensions", "--extensions", str(DECLARATION)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert {
        name: pair(value["campaign"]["extended"]) for name, value in report["variants"].items()
    } == {name: value["extended"] for name, value in COUNTS.items()}
