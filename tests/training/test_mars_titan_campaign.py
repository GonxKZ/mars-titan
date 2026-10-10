"""MARS-TITAN dentro de la campaña con máscaras, en CPU y sin modificar pesos.

El plan se comprueba sobre la comparación declarada. La ejecución registra
`titans_mac_online`, `mars_titan_m1`, `mars_titan_m3` y la corrección `mars_titan_b6` en una
campaña B reducida sobre US con los ejecutores reales y el optimizador que solo registra
gradientes, y se detiene tras las tres primeras ventanas: una reentrenada y dos trasladadas.
Los demás brazos son dobles.
"""

import json
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.walk_forward_receipt import read_window_receipt
from mars_titan.training import campaign_plan as plan
from mars_titan.training import mars_titan_correction as mc
from mars_titan.training import mars_titan_run, search_cases
from mars_titan.training import mars_titan_walk_forward as mw
from mars_titan.training import masked_campaign as engine
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.carried_predictions import CARRIED_PARTITIONS, REGENERATED_PARTITIONS
from tests.suite_support import skip_without_episodic_native
from tests.training.test_campaign_plan import CAMPAIGNS, write_variant
from tests.training.test_masked_campaign import Recorder, doubles
from tests.training.test_titans_campaign import ARM as TITANS_ARM
from tests.training.test_titans_campaign import permitted as permitted
from tests.training.test_titans_campaign import titans_campaign
from tests.training.test_titans_walk_forward import Factory
from tests.training.test_walk_forward_v2_views import fixture

DECLARED = Path("configs/titans/episodic-readout-historical-masked.json")
CORRECTION = Path("configs/titans/mature-correction-historical-masked.json")
ARM = "mars_titan_m1"
M3_ARM = "mars_titan_m3"
B6_ARM = "mars_titan_b6"
ARMS = {
    "mars_titan_m0": {"episodic_bank": "m0_no_bank"},
    "mars_titan_m1": {"episodic_bank": "m1"},
    "mars_titan_m2": {"episodic_bank": "m2"},
    "mars_titan_m3": {"episodic_bank": "m3"},
    "mars_titan_m1_k2": {"episodic_bank": "m1", "refinements": 2},
    "mars_titan_m1_k4": {"episodic_bank": "m1", "refinements": 4},
    "mars_titan_m1_k4_first_read": {
        "episodic_bank": "m1",
        "refinements": 4,
        "refinement_episodes": "first_read",
    },
    "mars_titan_b6": {"associative_memory": {"rule": "proximal", "key": "codec"}},
    "mars_titan_b6_bias": {"associative_memory": {"rule": "proximal", "key": "constant"}},
}
CORRECTIONS = {"mars_titan_b6", "mars_titan_b6_bias"}
# Brazos de MARS-TITAN que recorre la campaña B reducida con los ejecutores reales.
RUN_ARMS = (ARM, M3_ARM, B6_ARM)


def cases(arm):
    return ("eta5e-2", "eta25e-2") if arm in CORRECTIONS else ("lr1e-4", "lr1e-3")


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def section(recipe=DECLARED, **changes):
    value = dict(
        recipe=str(Path(recipe).resolve()),
        arms=ARMS,
        pending_arms={},
        parent_arm=TITANS_ARM,
        search_seed=42,
        correction_recipe=str(CORRECTION.resolve()),
    )
    # Un cambio a None retira el campo de la sección.
    return {key: item for key, item in (value | changes).items() if item is not None}


def declared(tmp_path, variant="A", **changes):
    limits = dict(max_training_jobs=100_000, max_prediction_jobs=100_000)
    return write_variant(tmp_path, variant, mars_titan=section(**changes), limits=limits)


def test_constants_repeat_the_runner_names_without_importing_torch():
    assert plan.MARS_RECIPE == mars_titan_run.RECIPE
    assert plan.MARS_SEARCHED == search_cases.SEARCHED
    assert mars_titan_run.checked_search_cases is search_cases.checked_search_cases


@pytest.mark.parametrize("variant", ["A", "B"])
def test_mars_arms_search_two_cases_per_window_after_their_titans_parent(tmp_path, variant):
    campaign = plan.load_campaign(declared(tmp_path, variant))
    jobs = plan.plan_campaign(campaign)
    base = plan.plan_campaign(plan.load_campaign(CAMPAIGNS[variant]))
    order = {job["id"]: index for index, job in enumerate(jobs)}
    # Los trabajos de los demás brazos no cambian al declarar MARS-TITAN.
    assert [job for job in jobs if job["family"] != plan.MARS] == base
    mars = [job for job in jobs if job["family"] == plan.MARS]
    assert {job["arm"] for job in mars} == set(ARMS)
    titans = [job for job in base if job["arm"] == TITANS_ARM]
    for arm in ARMS:
        own = [job for job in mars if job["arm"] == arm]
        assert [(j["window"], j["stage"], j["seed"]) for j in own] == [
            (j["window"], j["stage"], j["seed"]) for j in titans
        ]
    for job in mars:
        assert job["model"] == plan.MARS and job["kind"] in (plan.FIT, plan.CARRY)
        origin = f"{job['scope']}/{job['window']}/{TITANS_ARM}/"
        if job["stage"] == "search":
            assert job["parent"] == TITANS_ARM
            assert job["case"]["components"] == ARMS[job["arm"]]
            assert job["case"]["search_case"] == job["candidate"] in cases(job["arm"])
            recipe = CORRECTION if job["arm"] in CORRECTIONS else DECLARED
            assert job["case"]["recipe"] == str(recipe.resolve())
            assert sorted(job["depends"]) == sorted(
                f"{origin}search-{name}" for name in ("lr1e-4", "lr1e-3")
            )
        elif job["stage"] == "finalist":
            assert f"{origin}finalist-s{job['seed']}" in job["depends"]
            assert sum(dep.startswith(origin) for dep in job["depends"]) == 1
        else:
            assert "parent" not in job and not any(TITANS_ARM in d for d in job["depends"])
        # Cada dependencia aparece antes en el plan, así la ejecución en orden la encuentra.
        assert all(order[dep] < order[job["id"]] for dep in job["depends"])
    counts = plan.count_jobs(campaign)
    titans_counts = counts["scopes"]["US"]["arms"][TITANS_ARM]
    for arm in ARMS:
        assert counts["scopes"]["US"]["arms"][arm] == titans_counts


def test_m3_has_a_producer_and_no_mars_arm_stays_pending(tmp_path):
    report = plan.check_campaign(declared(tmp_path, "B"))
    pending = report["pending_families"]
    assert "mars_titan" not in pending and "titans_mac" not in pending and "cm_v1" in pending
    jobs = plan.plan_campaign(plan.load_campaign(declared(tmp_path, "B")))
    m3 = [job for job in jobs if job["arm"] == M3_ARM]
    assert m3 and all(
        job["case"]["components"] == {"episodic_bank": "m3"} for job in m3 if job.get("case")
    )
    counts = report["counts"]["scopes"]["US"]["arms"]
    assert counts[M3_ARM] == counts["mars_titan_m2"] == counts[TITANS_ARM]
    plain = plan.check_campaign(CAMPAIGNS["B"])["pending_families"]
    assert len(plain["mars_titan"]["arms"]) == 9 and "motives" not in plain["mars_titan"]
    # Un brazo sin productor puede seguir declarándose pendiente con su motivo.
    motive = "Brazo retirado para una comprobación"
    partial_arms = {k: v for k, v in ARMS.items() if k != M3_ARM}
    report = plan.check_campaign(
        declared(tmp_path, "B", arms=partial_arms, pending_arms={M3_ARM: motive})
    )
    assert report["pending_families"]["mars_titan"]["motives"] == {M3_ARM: motive}


def test_with_every_section_declared_no_compared_arm_lacks_a_producer(tmp_path):
    """Los 27 brazos de la comparación tienen productor con las cuatro secciones, salvo el
    control en línea, cuyos trabajos declara la campaña A por etapas."""
    from tests.training.test_candidate_walk_forward import section as gru_section

    cm = dict(declaration=str(Path("configs/titans/cm-v1-factorial.json").resolve()))
    path = write_variant(
        tmp_path,
        "B",
        episodic_gru=gru_section(),
        mars_titan=section(),
        cm_v1=cm | dict(search_seed=42),
        limits=dict(max_training_jobs=100_000, max_prediction_jobs=100_000),
    )
    report = plan.check_campaign(path)
    # Solo queda el control en línea, cuyos trabajos declara la campaña A por etapas.
    assert set(report["pending_families"]) == {plan.ONLINE_CONTROL}
    campaign = plan.load_campaign(path)
    compared = campaign["comparison_config"]["arms"]
    planned = {job["arm"] for job in plan.plan_campaign(campaign)}
    assert len(compared) == 27
    controls = {name for name, arm in compared.items() if arm["family"] == "control"}
    online = {name for name, arm in compared.items() if arm["family"] == plan.ONLINE_CONTROL}
    assert online == {"transformer_compact_online"}
    # Los núcleos de CM-v1 se ajustan como trabajos propios y no son brazos comparados.
    assert planned - set(compared) == set(plan.CM_CORES)
    assert planned & set(compared) == set(compared) - controls - online and M3_ARM in planned


def recipe_with(tmp_path, change):
    document = json.loads(DECLARED.read_text())
    change(document)
    path = tmp_path / "readout.json"
    atomic_json(path, document)
    return path


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(parent_arm="titans_mac_frozen"), "padre el brazo mac_online"),
        (
            dict(arms={k: v for k, v in ARMS.items() if k != "mars_titan_m3"}, pending_arms={}),
            "motivo pendiente",
        ),
        (dict(pending_arms={"mars_titan_m3": ""}), "motivo pendiente"),
        (
            dict(arms=ARMS | {"mars_titan_m3": {"episodic_bank": "m4"}}, pending_arms={}),
            "banco episódico",
        ),
        (dict(pending_arms={"mars_titan_m3": "motivo"}), "motivo pendiente"),
        (dict(arms=ARMS | {"mars_titan_m0": {"refinements": 2}}), "banco episódico"),
        (dict(arms=ARMS | {"mars_titan_m2": ARMS["mars_titan_m1"]}), "distinta"),
        (dict(search_seed=43), "semilla de búsqueda"),
        (dict(extra=True), "no cumple"),
        (
            dict(arms=ARMS | {B6_ARM: {"associative_memory": {"rule": "proximal"}}}),
            "corrección B6",
        ),
        (
            dict(
                arms=ARMS
                | {
                    B6_ARM: {
                        "associative_memory": {"rule": "proximal", "key": "codec", "rate": 0.1}
                    }
                }
            ),
            "corrección B6",
        ),
        (
            dict(arms=ARMS | {B6_ARM: ARMS[B6_ARM] | {"episodic_bank": "m1"}}),
            "corrección B6",
        ),
        (dict(correction_recipe=None), "receta de la corrección"),
    ],
)
def test_section_rejects_arms_parents_and_seeds_outside_the_design(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        plan.load_campaign(declared(tmp_path, **changes))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["recipe"].update(epochs=29), "regla de parada"),
        (lambda d: d["recipe"].update(loss="mae"), "pinball"),
        (lambda d: d["walk_forward"]["search_cases"].pop("lr1e-3"), "tantos"),
        (
            lambda d: d["walk_forward"].update(
                search_cases={"a": {"weight_decay": 0.1}, "b": {"weight_decay": 0.0}}
            ),
            "tantos",
        ),
        (lambda d: d.update(recipe_name="other"), "pinball"),
    ],
)
def test_section_requires_the_fair_search_and_the_protocol_rule(tmp_path, change, message):
    path = recipe_with(tmp_path, change)
    with pytest.raises(ValueError, match=message):
        plan.load_campaign(declared(tmp_path, recipe=path))


def test_correction_recipe_is_declared_only_with_correction_arms(tmp_path):
    readers = {k: v for k, v in ARMS.items() if k not in CORRECTIONS}
    pending = {name: "Brazo B6 retirado para una comprobación" for name in CORRECTIONS}
    with pytest.raises(ValueError, match="receta de la corrección"):
        plan.load_campaign(declared(tmp_path, arms=readers, pending_arms=pending))
    changes = dict(arms=readers, pending_arms=pending, correction_recipe=None)
    campaign = plan.load_campaign(declared(tmp_path, **changes))
    assert set(campaign[plan.MARS]["candidates"]) == set(readers)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["walk_forward"]["search_cases"].pop("eta5e-2"), "casos de η"),
        (lambda d: d.update(recipe_name="other"), "casos de η"),
        (
            lambda d: d["walk_forward"].update(
                search_cases={"a": {"learning_rate": 0.1}, "b": {"learning_rate": 0.2}}
            ),
            "casos de η",
        ),
    ],
)
def test_correction_recipe_needs_as_many_cases_as_the_fair_search(tmp_path, change, message):
    document = json.loads(CORRECTION.read_text())
    change(document)
    path = tmp_path / "correction.json"
    atomic_json(path, document)
    with pytest.raises(ValueError, match=message):
        plan.load_campaign(declared(tmp_path, correction_recipe=str(path)))


def test_section_needs_the_titans_section_of_its_parent(tmp_path):
    path = write_variant(
        tmp_path,
        "A",
        titans_mac=None,
        mars_titan=section(),
        limits=dict(max_training_jobs=100_000, max_prediction_jobs=0),
    )
    with pytest.raises(ValueError, match="padre el brazo mac_online"):
        plan.load_campaign(path)


def mars_campaign(folder):
    """Preparar una campaña B reducida, con una semilla, con Titans-MAC y M1, M3 y B6."""
    path = titans_campaign(folder)
    comparison = json.loads((folder / "comparison.json").read_text())
    for arm in RUN_ARMS:
        comparison["arms"][arm] = dict(family="mars_titan", output="quantile_head_v1", seeds=[42])
        comparison["comparison"]["families"]["references_vs_zero"]["variants"].append(arm)
    atomic_json(folder / "comparison.json", comparison)
    document = json.loads(DECLARED.read_text())
    titans = json.loads((folder / "titans.json").read_text())
    document["recipe"].update(
        update_instants=titans["recipe"]["truncation"],
        block_rows=titans["recipe"]["block_rows"],
        bank_capacity=8,
    )
    atomic_json(folder / "readout.json", document)
    campaign = json.loads(path.read_text())
    atomic_json(folder / "correction.json", json.loads(CORRECTION.read_text()))
    campaign[plan.MARS] = dict(
        recipe="readout.json",
        arms={arm: ARMS[arm] for arm in RUN_ARMS},
        pending_arms={},
        parent_arm=TITANS_ARM,
        search_seed=42,
        correction_recipe="correction.json",
    )
    atomic_json(path, campaign)
    return path


class StopAfter:
    """Pedir la parada cuando existe el recibo de un trabajo, entre dos trabajos."""

    def __init__(self, receipt):
        self.receipt = receipt

    @property
    def requested(self):
        return self.receipt.is_file()


@pytest.fixture(scope="module")
def campaign_run(tmp_path_factory, permitted):
    skip_without_episodic_native()
    root = tmp_path_factory.mktemp("mars-titan-campaign")
    data = fixture(root / "data", ("US",))
    campaign = mars_campaign(root / "config")
    prepared = engine.prepare_views(campaign, data.parent, root / "views")
    views = {"US": root / "views" / "US"}
    jobs = plan.plan_campaign(plan.load_campaign(campaign))
    windows = list(prepared["US"]["windows"])[:3]
    last = [j for j in jobs if j["arm"] in RUN_ARMS and j["window"] == windows[2]][-1]
    factory = Factory()
    executors = doubles(Recorder())
    for model, fit, carry in (
        (plan.TITANS, wf.titans_fit, wf.titans_carry),
        (plan.MARS, mw.mars_titan_fit, mw.mars_titan_carry),
    ):
        executors[model, "fit"] = dict(
            executors[model, "fit"], run=partial(fit, device="cpu", optimizer_factory=factory)
        )
        executors[model, "carry"] = dict(
            executors[model, "carry"], run=partial(carry, device="cpu")
        )
    # Los ejecutores declaran fastpath=False durante cada trabajo y restauran el del proceso.
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(True)
    try:
        summary = engine.run_campaign(
            campaign,
            views,
            root / "out",
            executors=executors,
            lease=nullcontext,
            stop=StopAfter(root / "out" / "jobs" / last["id"] / "receipt.json"),
        )
        assert torch.backends.mha.get_fastpath_enabled()
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    return SimpleNamespace(
        root=root,
        output=root / "out",
        prepared=prepared,
        summary=summary,
        factory=factory,
        windows=windows,
        jobs=[j for j in jobs if j["arm"] in (*RUN_ARMS, TITANS_ARM) and j["window"] in windows],
    )


def receipt(run, job_id):
    return json.loads((run.output / "jobs" / job_id / "receipt.json").read_text())


@pytest.mark.parametrize("arm", RUN_ARMS)
def test_mars_jobs_start_from_the_selected_titans_window_and_share_its_rows(campaign_run, arm):
    assert campaign_run.summary["status"] == "paused"
    mars = [job for job in campaign_run.jobs if job["arm"] == arm]
    assert [job["kind"] for job in mars] == ["fit", "fit", "carry", "carry"]
    assert campaign_run.factory.instances and all(
        item.calls == len(item.records) > 0 for item in campaign_run.factory.instances
    )
    output = campaign_run.output
    for job in mars:
        own = receipt(campaign_run, job["id"])
        report = json.loads((output / own["report"]["path"]).read_text())
        if job["kind"] == "fit":
            key = min(
                (
                    (receipt(campaign_run, f"US/{job['window']}/{TITANS_ARM}/search-{n}"), n)
                    for n in ("lr1e-4", "lr1e-3")
                ),
                key=lambda item: (item[0]["score"], item[1]),
            )
            parent, name = key
            sources = own["identity"]["sources"]
            assert sources["parent"] == f"US/{job['window']}/{TITANS_ARM}/search-{name}"
            assert sources["parent_sha256"] == sha256(
                output / "jobs" / sources["parent"] / "receipt.json"
            )
            assert report["request"]["parent"]["checkpoint_sha256"] == parent["parent"]["sha256"]
            assert report["request"]["search_case"] == job["candidate"]
            assert own["parent"] == dict(id=job["id"], sha256=report["checkpoint"]["sha256"])
            other = receipt(campaign_run, f"US/{job['window']}/{TITANS_ARM}/search-{name}")
        else:
            anchor = report["anchor"]
            assert anchor["components"] == ARMS[arm]
            assert anchor["checkpoint_sha256"] == own["parent"]["sha256"]
            reset = report["memory_policy"]["associative" if arm == B6_ARM else "bank"]
            assert reset.startswith(("empty_at_each_pass", "zero_at_each_pass"))
            assert report["kind"] == (mc.CARRY_KIND if arm == B6_ARM else mw.CARRY_KIND)
            other = receipt(campaign_run, f"US/{job['window']}/{TITANS_ARM}/carry-s42")
        for partition in ("calibration", "evaluation"):
            mine, theirs = own["predictions"][partition], other["predictions"][partition]
            assert mine["rows"] == theirs["rows"]
            assert mine["rows_sha256"] == theirs["rows_sha256"]


def test_m3_fits_its_scalers_on_each_window_and_carries_those_of_its_anchor(campaign_run):
    output = campaign_run.output
    fitted = {}
    for job in campaign_run.jobs:
        if job["arm"] != M3_ARM:
            continue
        own = receipt(campaign_run, job["id"])
        report = json.loads((output / own["report"]["path"]).read_text())
        folder = (output / own["report"]["path"]).parent
        if job["kind"] == "fit":
            fit = json.loads((folder / "fit/run.json").read_text())
            scalers = fit["identity"]["retention"]["scalers"]
            # Las escalas proceden del índice de entrenamiento de la propia ventana.
            assert scalers["source_sha256"] == report["identity"]["indices"]["train"]
            fitted[job["window"]] = mw.WriteScalers.from_fields(scalers).fingerprint()
        else:
            anchor = receipt(campaign_run, own["identity"]["sources"]["source"])
            anchor_fit = output / anchor["attempt"] / "fit/run.json"
            scalers = json.loads(anchor_fit.read_text())["identity"]["retention"]["scalers"]
            expected = mw.WriteScalers.from_fields(scalers).fingerprint()
            assert report["memory_policy"]["write_scalers_sha256"] == expected
            assert expected in fitted.values()
    assert fitted


def test_window_receipts_publish_the_mars_arm_with_its_selected_state(campaign_run):
    output = campaign_run.output
    for job in campaign_run.jobs:
        if job["arm"] != ARM:
            continue
        path = output / "windows/US" / job["window"] / ARM / "seed-42" / "US.json"
        record = json.loads(path.read_text())
        assert read_window_receipt(record).fold == job["window"]
        if job["kind"] == "carry":
            assert record["parent"] == receipt(campaign_run, job["id"])["parent"]


def test_carry_repeats_bit_for_bit_and_needs_a_new_destination(campaign_run, tmp_path):
    job = next(j for j in campaign_run.jobs if j["arm"] == ARM and j["kind"] == "carry")
    own = receipt(campaign_run, job["id"])
    source = own["identity"]["sources"]["source"]
    anchor = campaign_run.output / receipt(campaign_run, source)["attempt"]
    windows = campaign_run.prepared["US"]["windows"]
    anchor_view, view = windows[job["anchor"]]["path"], windows[job["window"]]["path"]
    first = mw.carry_mars_titan(anchor, anchor_view, view, tmp_path / "first", device="cpu")
    report = json.loads((campaign_run.output / own["report"]["path"]).read_text())
    for name in mw.CARRIED:
        assert first["predictions"][name]["sha256"] == report["predictions"][name]["sha256"]
    with pytest.raises(ValueError, match="directorio nuevo"):
        mw.carry_mars_titan(anchor, anchor_view, view, tmp_path / "first", device="cpu")
    with pytest.raises(ValueError, match="no es la de su ajuste"):
        mw.carry_mars_titan(anchor, view, view, tmp_path / "other", device="cpu")


def test_correction_carry_repeats_bit_for_bit_and_never_fits(campaign_run, tmp_path):
    job = next(j for j in campaign_run.jobs if j["arm"] == B6_ARM and j["kind"] == "carry")
    own = receipt(campaign_run, job["id"])
    source = own["identity"]["sources"]["source"]
    anchor = campaign_run.output / receipt(campaign_run, source)["attempt"]
    windows = campaign_run.prepared["US"]["windows"]
    anchor_view, view = windows[job["anchor"]]["path"], windows[job["window"]]["path"]
    first = mc.carry_correction(anchor, anchor_view, view, tmp_path / "first", device="cpu")
    report = json.loads((campaign_run.output / own["report"]["path"]).read_text())
    assert "regenerated" not in first
    for name in CARRIED_PARTITIONS:
        assert first["predictions"][name]["sha256"] == report["predictions"][name]["sha256"]
    anchored = json.loads((anchor / "run.json").read_text())
    assert first["anchor"]["variant_sha256"] == anchored["identity"]["variant_sha256"]
    assert not (anchor / "fit").exists()
    with pytest.raises(ValueError, match="no es la de su ventana"):
        mc.carry_correction(anchor, view, view, tmp_path / "other", device="cpu")


def test_correction_frozen_parent_reproduces_the_carry_and_predicts_the_next_validation(
    campaign_run, tmp_path
):
    """Padre congelado de la cadena trivial de B6: el traslado de la variante B más val_k.

    Cada tramo empieza con A en cero, así que la calibración y la evaluación deben ser las
    del traslado de la campaña desde la misma ancla. La validación de la ventana posterior
    es la que puntúa la cadena y tiene las filas de su vista.
    """
    job = next(j for j in campaign_run.jobs if j["arm"] == B6_ARM and j["kind"] == "carry")
    own = receipt(campaign_run, job["id"])
    source = own["identity"]["sources"]["source"]
    anchor = campaign_run.output / receipt(campaign_run, source)["attempt"]
    windows = campaign_run.prepared["US"]["windows"]
    anchor_view, view = windows[job["anchor"]]["path"], windows[job["window"]]["path"]
    frozen = mc.carry_correction(
        anchor, anchor_view, view, tmp_path / "frozen", device="cpu", frozen_parent=True
    )
    assert frozen["frozen_parent"] is True and frozen["kind"] == mc.CARRY_KIND
    assert "regenerated" not in frozen and "modality_ablation" not in frozen
    assert set(frozen["predictions"]) == {"validation", *CARRIED_PARTITIONS}
    carried = json.loads((campaign_run.output / own["report"]["path"]).read_text())
    for name in CARRIED_PARTITIONS:
        assert frozen["predictions"][name]["sha256"] == carried["predictions"][name]["sha256"]
    counts = json.loads(Path(view).read_text())["counts"]
    assert frozen["predictions"]["validation"]["rows"] == counts["validation"] > 0
    assert frozen["fold"]["validation"][0] >= frozen["anchor"]["fold"]["validation"][1]
    # El padre congelado predice otra ventana: ni regenera el ancla ni ablaciona modalidades.
    for options in (dict(regenerate=True), dict(modality_ablation="mask_news")):
        with pytest.raises(ValueError, match="sin ablación ni regeneración"):
            mc.carry_correction(
                anchor,
                anchor_view,
                view,
                tmp_path / "other",
                device="cpu",
                frozen_parent=True,
                **options,
            )
    assert not (tmp_path / "other").exists()


def test_correction_fit_regenerates_its_three_partitions_bit_for_bit(campaign_run, tmp_path):
    """La retención v2 repite el ajuste B6 por inferencia antes de liberar sus filas.

    Cada tramo empieza con A en cero, así que repetir validación, calibración y evaluación
    sobre la propia vista debe dar las mismas tablas. Otra vista no se admite.
    """
    job = next(j for j in campaign_run.jobs if j["arm"] == B6_ARM and j["kind"] == "fit")
    own = receipt(campaign_run, job["id"])
    attempt = campaign_run.output / own["attempt"]
    windows = campaign_run.prepared["US"]["windows"]
    view = windows[job["window"]]["path"]
    again = mc.carry_correction(
        attempt, view, view, tmp_path / "again", device="cpu", regenerate=True
    )
    report = json.loads((attempt / "run.json").read_text())
    assert again["regenerated"] is True
    assert set(again["predictions"]) == set(REGENERATED_PARTITIONS)
    for name in REGENERATED_PARTITIONS:
        assert again["predictions"][name]["sha256"] == report["predictions"][name]["sha256"]
    later = next(w for w in windows if w > job["window"])
    with pytest.raises(ValueError, match="propia ventana del ancla"):
        mc.carry_correction(
            attempt, view, windows[later]["path"], tmp_path / "later", device="cpu", regenerate=True
        )


def test_finalists_pick_their_own_search_and_receive_their_parent_finalist(tmp_path):
    campaign = plan.load_campaign(declared(tmp_path, "A"))
    jobs = plan.plan_campaign(campaign)
    window = jobs[0]["window"]
    prefix = f"US/{window}"
    state = engine._Campaign(campaign, {}, tmp_path / "out", {}, {}, None)

    def confirmed(job_id, score, case=None):
        state.receipts[job_id] = dict(
            score=score,
            sha256=f"{len(state.receipts):064x}",
            attempt=f"jobs/{job_id}/attempt-0001",
            identity=dict(case=case, seed=42),
            parent=dict(id=job_id, sha256=f"{len(state.receipts) + 100:064x}"),
        )

    by_id = {job["id"]: job for job in jobs}
    for name in ("lr1e-4", "lr1e-3"):
        confirmed(f"{prefix}/{TITANS_ARM}/search-{name}", 0.5 if name == "lr1e-4" else 0.1)
        own = by_id[f"{prefix}/{ARM}/search-{name}"]
        confirmed(own["id"], 0.3 if name == "lr1e-4" else 0.4, own["case"])
    # El finalista del padre tiene la menor puntuación, pero no puede ganar la búsqueda propia.
    confirmed(f"{prefix}/{TITANS_ARM}/finalist-s43", 0.01)
    search = by_id[f"{prefix}/{ARM}/search-lr1e-4"]
    case, anchor, sources = state.resolve(search)
    assert anchor is None and case == search["case"]
    assert sources["parent"] == f"{prefix}/{TITANS_ARM}/search-lr1e-3"
    parent = state.parent_of(search)
    assert parent["checkpoint_sha256"] == state.receipts[sources["parent"]]["parent"]["sha256"]
    finalist = by_id[f"{prefix}/{ARM}/finalist-s43"]
    case, _, sources = state.resolve(finalist)
    assert sources["source"] == f"{prefix}/{ARM}/search-lr1e-4"
    assert case == by_id[sources["source"]]["case"] | dict(seed=43)
    assert sources["parent"] == f"{prefix}/{TITANS_ARM}/finalist-s43"
    assert state.parent_of(finalist)["job"] == sources["parent"]
    # Un traslado parte del estado elegido en su ancla, que ya fija su padre.
    assert state.parent_of(dict(search, stage="carry", kind=plan.CARRY)) is None
