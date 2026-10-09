"""Factorial CM-v1 dentro de la campaña con máscaras, en CPU y sin modificar pesos.

El plan se comprueba sobre la comparación declarada. La ejecución registra los dos núcleos y
los cuatro brazos en una campaña B reducida sobre US con los ejecutores reales y el
optimizador que solo registra gradientes, y se detiene tras las tres primeras ventanas: una
reentrenada y dos trasladadas. Los demás brazos son dobles.
"""

import json
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.walk_forward_receipt import read_window_receipt
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.training import campaign_plan as plan
from mars_titan.training import cm_v1_factorial as cm
from mars_titan.training import masked_campaign as engine
from tests.training.test_campaign_plan import CAMPAIGNS, write_variant
from tests.training.test_mars_titan_campaign import StopAfter
from tests.training.test_masked_campaign import Recorder, doubles, write_campaign
from tests.training.test_titans_campaign import ARM as TITANS_ARM
from tests.training.test_titans_campaign import RECIPE as TITANS_RECIPE
from tests.training.test_titans_campaign import permitted as permitted
from tests.training.test_titans_walk_forward import Factory
from tests.training.test_walk_forward_v2_views import fixture

DECLARED = Path("configs/titans/cm-v1-factorial.json")
READOUT = Path("configs/titans/episodic-readout-historical-masked.json")


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def declared(tmp_path, variant="A", declaration=DECLARED, **changes):
    section = dict(declaration=str(Path(declaration).resolve()), search_seed=42) | changes
    limits = dict(max_training_jobs=100_000, max_prediction_jobs=100_000)
    return write_variant(tmp_path, variant, cm_v1=section, limits=limits)


def test_constants_repeat_the_factorial_names_without_importing_torch():
    assert plan.CM_NAME == cm.NAME
    assert plan.CM_CORES == tuple(cm.CORES)
    assert all(cm.CORES[core] == cm.ARMS[arm][0] for arm, core in plan.CM_ARMS.items())
    assert set(plan.CM_ARMS) == set(cm.ARMS)
    assert engine.EXECUTORS[plan.CM_CORE_MODEL, plan.FIT]["resumable"]
    assert (plan.CM_CORE_MODEL, plan.CARRY) not in engine.EXECUTORS


@pytest.mark.parametrize("variant", ["A", "B"])
def test_cores_fit_once_per_case_and_each_arm_starts_from_its_core(tmp_path, variant):
    campaign = plan.load_campaign(declared(tmp_path, variant))
    jobs = plan.plan_campaign(campaign)
    base = plan.plan_campaign(plan.load_campaign(CAMPAIGNS[variant]))
    order = {job["id"]: index for index, job in enumerate(jobs)}
    assert [job for job in jobs if job["family"] != plan.CM] == base
    titans = [job for job in base if job["arm"] == TITANS_ARM]
    fits = [(j["window"], j["stage"], j["seed"]) for j in titans if j["kind"] == plan.FIT]
    for core in plan.CM_CORES:
        own = [job for job in jobs if job["arm"] == core]
        assert [(j["window"], j["stage"], j["seed"]) for j in own] == fits
        assert all(j["model"] == plan.CM_CORE_MODEL and "parent" not in j for j in own)
        assert {j["case"]["core"] for j in own if j["stage"] == "search"} == {core}
    for arm, core in plan.CM_ARMS.items():
        own = [job for job in jobs if job["arm"] == arm]
        assert [(j["window"], j["stage"], j["seed"]) for j in own] == [
            (j["window"], j["stage"], j["seed"]) for j in titans
        ]
        for job in own:
            origin = f"{job['scope']}/{job['window']}/{core}/"
            if job["stage"] == "search":
                assert job["parent"] == core and job["case"]["arm"] == arm
                assert set(job["case"]) == cm.ARM_FIELDS
                assert sorted(job["depends"]) == sorted(
                    f"{origin}search-{name}" for name in ("lr1e-4", "lr1e-3")
                )
            elif job["stage"] == "finalist":
                assert f"{origin}finalist-s{job['seed']}" in job["depends"]
            else:
                assert "parent" not in job and not any(core in d for d in job["depends"])
    for job in jobs:
        assert all(order[dep] < order[job["id"]] for dep in job["depends"])
    counts = plan.count_jobs(campaign)["scopes"]["US"]["arms"]
    for arm in plan.CM_ARMS:
        assert counts[arm] == counts[TITANS_ARM]
    for core in plan.CM_CORES:
        assert all(entry["carry"] == 0 for entry in counts[core].values())
    # Los núcleos emiten la cabeza de cuantiles de sus brazos y se validan con sus columnas.
    for name in (*plan.CM_CORES, *plan.CM_ARMS):
        assert plan.arm_output(campaign, name) == plan.QUANTILE_HEAD
    with pytest.raises(ValueError, match="trabajo auxiliar"):
        plan.arm_output(campaign, "cm_v1_core_x")
    pending = plan.check_campaign(declared(tmp_path, variant))["pending_families"]
    assert "cm_v1" not in pending
    assert "cm_v1" in plan.check_campaign(CAMPAIGNS[variant])["pending_families"]


def changed(tmp_path, name, change):
    document = json.loads((DECLARED.parent / name).read_text())
    change(document)
    path = tmp_path / name
    atomic_json(path, document)
    return path


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(search_seed=7), "semilla de búsqueda"),
        (dict(extra=True), "no cumple"),
    ],
)
def test_section_rejects_undeclared_fields_and_seeds(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        plan.load_campaign(declared(tmp_path, **change))


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        ("cm-v1-factorial.json", lambda d: d.update(name="other"), "cuatro brazos"),
        ("cm-v1-factorial.json", lambda d: d["arms"].pop("cm_v1_bm"), "cuatro brazos"),
        (
            "cm-v1-factorial.json",
            lambda d: d["arms"]["cm_v1_bm"].update(control=True),
            "cuatro brazos",
        ),
        (
            "chronological-training-historical-masked.json",
            lambda d: d["recipe"].update(epochs=29),
            "regla de parada",
        ),
        (
            "episodic-readout-historical-masked.json",
            lambda d: d["recipe"].update(loss="mae"),
            "pinball",
        ),
        (
            "episodic-readout-historical-masked.json",
            lambda d: d["walk_forward"]["search_cases"].pop("lr1e-3"),
            "tantos",
        ),
    ],
)
def test_section_validates_the_declaration_and_both_recipes(tmp_path, name, change, message):
    for other in ("cm-v1-factorial.json", "chronological-training-historical-masked.json"):
        if other != name:
            atomic_json(tmp_path / other, json.loads((DECLARED.parent / other).read_text()))
    if name != "episodic-readout-historical-masked.json":
        atomic_json(tmp_path / READOUT.name, json.loads(READOUT.read_text()))
    changed(tmp_path, name, change)
    with pytest.raises(ValueError, match=message):
        plan.load_campaign(declared(tmp_path, declaration=tmp_path / "cm-v1-factorial.json"))


def cm_campaign(folder):
    """Campaña B reducida con los cuatro brazos de CM-v1 y una semilla, sobre recetas técnicas."""
    path = write_campaign(folder, arms=("gru", "ridge"))
    comparison = json.loads((folder / "comparison.json").read_text())
    for arm in plan.CM_ARMS:
        comparison["arms"][arm] = dict(family="cm_v1", output="quantile_head_v1", seeds=[42])
        comparison["comparison"]["families"]["references_vs_zero"]["variants"].append(arm)
    atomic_json(folder / "comparison.json", comparison)
    core = json.loads(Path(TITANS_RECIPE).read_text())
    core["predictor"].update(hidden_size=32, dtype="float64")
    core["recipe"].update(truncation=3, block_rows=2)
    atomic_json(folder / "titans.json", core)
    readout = json.loads(READOUT.read_text())
    readout["recipe"].update(update_instants=3, block_rows=2, bank_capacity=8)
    atomic_json(folder / "readout.json", readout)
    document = json.loads(DECLARED.read_text())
    document["base"].update(core_recipe="titans.json", readout_recipe="readout.json")
    document["control"].update(rank=2, frequency=2, grid_size=16, threshold=0.0, max_flows=1)
    document["control"]["weight"] = 0.5
    document["consolidation"].update(frontier=2, new_candidates=2)
    atomic_json(folder / "cm.json", document)
    campaign = json.loads(path.read_text())
    campaign[plan.CM] = dict(declaration="cm.json", search_seed=42)
    atomic_json(path, campaign)
    return path


@pytest.fixture(scope="module")
def campaign_run(tmp_path_factory, permitted):
    root = tmp_path_factory.mktemp("cm-v1-campaign")
    data = fixture(root / "data", ("US",))
    campaign = cm_campaign(root / "config")
    prepared = engine.prepare_views(campaign, data.parent, root / "views")
    views = {"US": root / "views" / "US"}
    jobs = plan.plan_campaign(plan.load_campaign(campaign))
    windows = list(prepared["US"]["windows"])[:3]
    last = [j for j in jobs if j["family"] == plan.CM and j["window"] == windows[2]][-1]
    factory = Factory()
    executors = doubles(Recorder())
    for key, run in (
        ((plan.CM_CORE_MODEL, plan.FIT), partial(cm.cm_v1_core_fit, optimizer_factory=factory)),
        ((plan.CM, plan.FIT), partial(cm.cm_v1_fit, optimizer_factory=factory)),
        ((plan.CM, plan.CARRY), cm.cm_v1_carry),
    ):
        executors[key] = dict(executors[key], run=partial(run, device="cpu"))
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
        jobs=[j for j in jobs if j["family"] == plan.CM and j["window"] in windows],
    )


def receipt(run, job_id):
    return json.loads((run.output / "jobs" / job_id / "receipt.json").read_text())


def predictions(run, job, partition):
    own = receipt(run, job["id"])
    folder = (run.output / own["report"]["path"]).parent
    table = pq.read_table(folder / f"{partition}-predictions.parquet").sort_by("sample_id")
    return {
        name: table[name].to_pylist() for name in ("sample_id", "prediction", *QUANTILE_COLUMNS)
    }


def test_arms_start_from_the_selected_core_of_their_window(campaign_run):
    assert campaign_run.summary["status"] == "paused"
    output = campaign_run.output
    cores = [job for job in campaign_run.jobs if job["arm"] in plan.CM_CORES]
    assert [job["kind"] for job in cores] == ["fit"] * 4
    for arm, core in plan.CM_ARMS.items():
        own = [job for job in campaign_run.jobs if job["arm"] == arm]
        assert [job["kind"] for job in own] == ["fit", "fit", "carry", "carry"]
        for job in own:
            mine = receipt(campaign_run, job["id"])
            report = json.loads((output / mine["report"]["path"]).read_text())
            if job["kind"] == "carry":
                assert report["kind"] == cm.CARRY_KIND
                assert report["anchor"]["arm"] == arm
                continue
            best, name = min(
                (
                    (receipt(campaign_run, f"US/{job['window']}/{core}/search-{n}"), n)
                    for n in ("lr1e-4", "lr1e-3")
                ),
                key=lambda item: (item[0]["score"], item[1]),
            )
            sources = mine["identity"]["sources"]
            assert sources["parent"] == f"US/{job['window']}/{core}/search-{name}"
            assert sources["parent_sha256"] == sha256(
                output / "jobs" / sources["parent"] / "receipt.json"
            )
            assert report["request"]["parent"]["checkpoint_sha256"] == best["parent"]["sha256"]
            assert report["request"]["arm"] == arm
    assert campaign_run.factory.instances and all(
        item.calls == len(item.records) > 0 for item in campaign_run.factory.instances
    )


def test_only_the_compared_arms_publish_window_receipts(campaign_run):
    output = campaign_run.output
    for window in campaign_run.windows:
        for core in plan.CM_CORES:
            assert not (output / "windows/US" / window / core).exists()
        for arm in plan.CM_ARMS:
            record = json.loads(
                (output / "windows/US" / window / arm / "seed-42/US.json").read_text()
            )
            assert read_window_receipt(record).fold == window


def test_b_and_b_plus_c_emit_the_same_rows_while_c_has_not_changed_the_core(campaign_run):
    by_arm = {}
    for job in campaign_run.jobs:
        by_arm.setdefault(job["arm"], []).append(job)
    for first, second in (("cm_v1_b", "cm_v1_bc"), ("cm_v1_bm", "cm_v1_bcm")):
        for left, right in zip(by_arm[first], by_arm[second], strict=True):
            for partition in ("calibration", "evaluation"):
                assert predictions(campaign_run, left, partition) == predictions(
                    campaign_run, right, partition
                )


def test_carry_repeats_bit_for_bit_and_needs_a_new_destination(campaign_run, tmp_path):
    job = next(j for j in campaign_run.jobs if j["arm"] == "cm_v1_bcm" and j["kind"] == "carry")
    own = receipt(campaign_run, job["id"])
    source = own["identity"]["sources"]["source"]
    anchor = campaign_run.output / receipt(campaign_run, source)["attempt"]
    windows = campaign_run.prepared["US"]["windows"]
    anchor_view, view = windows[job["anchor"]]["path"], windows[job["window"]]["path"]
    first = cm.carry_cm_v1(anchor, anchor_view, view, tmp_path / "first", device="cpu")
    report = json.loads((campaign_run.output / own["report"]["path"]).read_text())
    for name in ("calibration", "evaluation"):
        assert first["predictions"][name]["sha256"] == report["predictions"][name]["sha256"]
    with pytest.raises(ValueError, match="directorio nuevo"):
        cm.carry_cm_v1(anchor, anchor_view, view, tmp_path / "first", device="cpu")
    with pytest.raises(ValueError, match="no es la de su ajuste"):
        cm.carry_cm_v1(anchor, view, view, tmp_path / "other", device="cpu")
