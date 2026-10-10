"""Titans-MAC en el walk-forward por etapas de la etapa de adaptadores, sin modificar pesos.

La campaña base es una variante A reducida a dos ventanas US (evaluación de 2022 y 2023)
con `titans_mac_online` ajustado en las dos con los ejecutores reales y un optimizador que
solo registra gradientes. La etapa parte del estado elegido en la primera ventana, predice
la segunda con el padre congelado y la ajusta con la continuación completa y el adaptador
de la cabeza solo sobre sus filas nuevas. AdamW se sustituye por un registrador que no
hereda de `torch.optim.Optimizer` y exige pesos sin cambios. Sin GPU.
"""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.storage import atomic_json
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.posttraining import campaign_stage, staged_chain
from mars_titan.posttraining import chronological_matrix as cm
from mars_titan.training import masked_campaign as engine
from mars_titan.training.campaign_plan import TITANS
from tests.posttraining.campaign_fixture import CpuLease
from tests.posttraining.real_only import real_data_only
from tests.training.test_masked_campaign import write_campaign
from tests.training.test_titans_campaign import (
    ARM,
    RECIPE,
    executors,
    permitted,  # noqa: F401
    selected_search,
    unfused,  # noqa: F401
)
from tests.training.test_titans_walk_forward import Factory, Recorder
from tests.training.test_walk_forward_v2_views import fixture

CONFIGS = Path("configs")
FIRST, NEXT = "fold-000", "fold-001"
FROZEN_JOB = f"US/{NEXT}/{ARM}__frozen_parent/frozen-s42"
FITS = {name: f"US/{NEXT}/{ARM}__{name}/fit-s42" for name in ("full_continuation", "head")}


class StageRecorder(Recorder):
    """Registrador de las ventanas de Titans-MAC con la firma de AdamW.

    No guarda copias de los gradientes. Solo cuenta los pasos y exige pesos sin cambios y
    gradientes finitos.
    """

    made = []

    def __init__(self, groups, lr, weight_decay):
        super().__init__(groups)
        self.initial = [p.detach().clone() for g in self.param_groups for p in g["params"]]
        StageRecorder.made.append(self)

    def step(self):
        current = [p for group in self.param_groups for p in group["params"]]
        assert all(torch.equal(a, b) for a, b in zip(current, self.initial, strict=True))
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in current)
        self.calls += 1


def staged_campaign(folder):
    """Campaña A con dos ventanas US, la GRU sustituta y un brazo Titans-MAC reducido."""
    path = write_campaign(folder, variant="A", arms=("gru",))
    protocol = json.loads(
        (CONFIGS / "evaluation/historical-masked-us-walk-forward-v2.json").read_text()
    )
    protocol["first_validation_start"] = "2021-04-01"
    atomic_json(folder / "us-protocol.json", protocol)
    declared = json.loads((folder / "comparison.json").read_text())
    declared["scopes"]["US"]["protocols"]["US"] = str((folder / "us-protocol.json").resolve())
    declared["arms"][ARM] = dict(family="titans_mac", output="quantile_head_v1", seeds=[42])
    declared["comparison"]["families"]["references_vs_zero"]["variants"].append(ARM)
    atomic_json(folder / "comparison.json", declared)
    recipe = json.loads(Path(RECIPE).read_text())
    recipe["predictor"].update(hidden_size=32, dtype="float64")
    recipe["recipe"].update(truncation=3, block_rows=2)
    atomic_json(folder / "titans.json", recipe)
    campaign = json.loads(path.read_text())
    campaign[TITANS] = dict(recipe="titans.json", arms={ARM: "mac_online"}, search_seed=42)
    atomic_json(path, campaign)
    return path


@pytest.fixture(scope="module")
def campaign_run(tmp_path_factory, unfused, permitted):  # noqa: F811
    root = tmp_path_factory.mktemp("staged-titans-campaign")
    data = fixture(root / "data", ("US",))
    campaign = staged_campaign(root / "config")
    prepared = engine.prepare_views(campaign, data.parent, root / "views")
    assert list(prepared["US"]["windows"]) == [FIRST, NEXT]
    views = {"US": root / "views" / "US"}
    summary = engine.run_campaign(
        campaign,
        views,
        root / "out",
        executors=executors(Factory()),
        lease=nullcontext,
        stop=SimpleNamespace(requested=False),
    )
    assert summary["status"] == "completed"
    return SimpleNamespace(
        root=root, campaign=campaign, views=views, output=root / "out", prepared=prepared
    )


def write_stage(campaign_run, folder):  # noqa: F811
    """Etapa A con el brazo Titans-MAC, la continuación y el adaptador de la cabeza."""
    folder.mkdir(parents=True, exist_ok=True)
    campaign = json.loads(Path(campaign_run.campaign).read_text())
    matrix = json.loads((CONFIGS / "posttraining/adapter-matrix-v3.json").read_text())
    matrix["budget"].update(seeds=[42], epochs=1, batch_size=campaign["neural"]["batch_size"])
    atomic_json(folder / "matrix.json", matrix)
    stage = json.loads(
        (CONFIGS / "posttraining/historical-masked-adapter-stage-a.json").read_text()
    )
    stage.update(
        campaign=str(Path(campaign_run.campaign).resolve()),
        matrix="matrix.json",
        scopes=["US"],
        arms=[ARM],
        limits=dict(max_training_jobs=100, max_prediction_jobs=100),
    )
    atomic_json(folder / "stage.json", stage)
    return folder / "stage.json"


CASES = ("full_continuation", "head")
ORIGINAL_CASES = cm.cases


def few_cases(*args, **options):
    """Solo la continuación y la cabeza, para acotar la prueba. El plan sigue siendo el real."""
    return [item for item in ORIGINAL_CASES(*args, **options) if item["id"].endswith(CASES)]


@pytest.fixture(scope="module", autouse=True)
def two_cases():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cm, "cases", few_cases)
        yield


def run(stage_path, campaign_run, output):  # noqa: F811
    with pytest.MonkeyPatch.context() as patch, real_data_only():
        patch.setattr(torch.optim, "AdamW", StageRecorder)
        return campaign_stage.run_stage(
            stage_path,
            campaign_run.views,
            campaign_run.output,
            output,
            lease=CpuLease,
            device="cpu",
            stop=SimpleNamespace(requested=False),
        )


@pytest.fixture(scope="module")
def stage_run(campaign_run, tmp_path_factory, two_cases):  # noqa: F811
    folder = tmp_path_factory.mktemp("chronological-stage")
    path = write_stage(campaign_run, folder / "config")
    StageRecorder.made.clear()
    summary = run(path, campaign_run, folder / "stage")
    return dict(path=path, output=folder / "stage", summary=summary, made=list(StageRecorder.made))


def receipts(output):
    return {
        str(path.parent.relative_to(output / "jobs")): json.loads(path.read_text())
        for path in (output / "jobs").rglob("receipt.json")
    }


def rows(path):
    table = pq.read_table(path).sort_by("sample_id")
    return {name: table[name].to_pylist() for name in ("sample_id", "target", *QUANTILE_COLUMNS)}


def test_stage_plans_the_frozen_parent_and_the_cases_of_the_second_window(stage_run):
    stage = campaign_stage.load_stage(stage_run["path"])
    active, awaiting = campaign_stage.stage_arms(stage)
    assert active == {
        ARM: dict(family=cm.TITANS, variant="mac_online", bank=True, design=cm.TITANS)
    }
    assert awaiting == {}
    jobs = campaign_stage.plan_stage(stage)
    assert [job["id"] for job in jobs] == [FROZEN_JOB, *FITS.values()]
    assert all(job["parent_window"] == FIRST for job in jobs)
    counts = campaign_stage.count_stage(stage, jobs)
    assert (counts["training_jobs"], counts["prediction_jobs"]) == (2, 1)
    chains = campaign_stage.plan_chain(stage, jobs)
    assert [job["id"] for job in chains] == [
        staged_chain.chain_job_id("US", window, ARM, 42) for window in (FIRST, NEXT)
    ]


def test_fits_reproduce_the_frozen_parent_rows_without_weight_changes(stage_run, campaign_run):  # noqa: F811
    summary, output = stage_run["summary"], stage_run["output"]
    assert summary["status"] == "completed"
    expected = dict(training_jobs=2, prediction_jobs=1, selection_jobs=2)
    assert summary["completed"] == summary["planned"] == expected
    assert stage_run["made"] and all(item.calls > 0 for item in stage_run["made"])
    found = receipts(output)
    assert set(found) == {FROZEN_JOB, *FITS.values()}
    frozen = found[FROZEN_JOB]
    assert (
        frozen["updates"] == 0 and frozen["reported_score"] is None and frozen["fit_rows"] is None
    )
    _, parent = selected_search(campaign_run.output, FIRST)
    assert frozen["parent"] == parent["parent"]
    for job_id in FITS.values():
        receipt = found[job_id]
        for partition in ("validation", "calibration", "evaluation"):
            mine = rows(output / receipt["predictions"][partition]["path"])
            theirs = rows(output / frozen["predictions"][partition]["path"])
            # mac_online con correcciones nulas y pesos sin cambios reproduce al padre.
            assert mine == theirs, (job_id, partition)
        assert receipt["score"] == frozen["score"]
    # Igualdad de actualizaciones entre la continuación y el adaptador.
    updates = {found[job_id]["updates"] for job_id in FITS.values()}
    assert len(updates) == 1 and min(updates) > 0


def test_fits_read_only_the_rows_after_the_parent_calibration(stage_run, campaign_run):  # noqa: F811
    output = stage_run["output"]
    proof = json.loads((output / "windows-data/US" / NEXT / "fit-rows.json").read_text())
    assert (proof["start"], proof["end"]) == ("2022-01-01", "2022-04-01")
    assert proof["rows"] > 0 and not any(proof["intersection"].values())
    since = int(np.datetime64("2022-01-01", "us").astype(np.int64))
    for job_id in FITS.values():
        receipt = receipts(output)[job_id]
        assert receipt["fit_rows"]["sha256"] == proof["sha256"]
        assert receipt["labels_used_until"] == proof["labels_used_until"]
        report = json.loads((output / receipt["run"]["path"]).read_text())
        placement = report["identity"]["posttraining"]["placement"]
        assert placement["design"] == "staged_previous_window_v1"
        assert placement["parent_window"] == FIRST
        assert (placement["fit_start"], placement["fit_end"]) == ("2022-01-01", "2022-04-01")
        train = report["identity"]["phases"]["train"]
        assert train["decision_start"] == since
        # El calentamiento lee entradas anteriores sin sus etiquetas.
        assert train["warmup_start"] < since
    # Los brazos cronológicos no abren la ventana de cohortes de los brazos neuronales.
    assert sorted(p.name for p in (output / "windows-data/US" / NEXT).iterdir()) == [
        "fit-rows.json"
    ]


def test_adapter_fits_train_only_the_head_corrections(stage_run):
    output = stage_run["output"]
    receipt = receipts(output)[FITS["head"]]
    report = json.loads((output / receipt["run"]["path"]).read_text())
    posttraining = report["identity"]["posttraining"]
    assert posttraining["adapter"]["trainable_parameters"] > 0
    assert posttraining["base_parameters_sha256"]
    roles = {tuple(group["role"] for group in item.param_groups) for item in stage_run["made"]}
    assert ("adapters",) in roles and any(len(value) > 1 for value in roles)


def test_the_chain_keeps_the_frozen_parent_when_no_case_improves_it(stage_run):
    output = stage_run["output"]
    first = staged_chain.read_selection(output, "US", FIRST, ARM, 42)
    assert first["selected"]["kind"] == "base" and first["parent_window"] is None
    chosen = staged_chain.read_selection(output, "US", NEXT, ARM, 42)
    assert chosen["parent_window"] == FIRST
    assert {item["kind"] for item in chosen["candidates"]} == {
        "frozen_parent",
        "continuation",
        "adapter",
    }
    # Todos empatan: el padre congelado solo cede ante una mejora estricta.
    assert chosen["selected"]["kind"] == "frozen_parent"
    assert chosen["selected"]["job"] == FROZEN_JOB and chosen["fit_rows"] is None
    assert stage_run["summary"]["chain"][staged_chain.chain_job_id("US", NEXT, ARM, 42)] == dict(
        kind="frozen_parent", job=FROZEN_JOB
    )
    market = chosen["receipts"]["US"]
    assert market.fold == NEXT and market.labels_used_until == chosen["labels_used_until"]


def test_confirmed_jobs_release_their_observation_indices(stage_run):
    output = stage_run["output"]
    assert not [path for path in (output / "jobs").rglob("indices") if path.is_dir()]
    assert stage_run["summary"]["released_index_bytes"]


def test_rerunning_confirms_without_new_optimizers(stage_run, campaign_run):  # noqa: F811
    StageRecorder.made.clear()
    again = run(stage_run["path"], campaign_run, stage_run["output"])
    assert again["status"] == "completed" and not StageRecorder.made
    assert again["chain"] == stage_run["summary"]["chain"]
