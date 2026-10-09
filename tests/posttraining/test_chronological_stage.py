"""Brazos cronológicos de Titans-MAC en la etapa de adaptadores, sin modificar pesos.

La campaña base es la de `test_titans_campaign`: variante B sobre US con `titans_mac_online`
ajustado en las ventanas reentrenadas y trasladado en las demás. La etapa añade la
continuación completa y el adaptador de la cabeza. AdamW se sustituye por un registrador
que no hereda de `torch.optim.Optimizer` y exige pesos sin cambios. Sin GPU.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.storage import atomic_json
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.posttraining import campaign_stage
from mars_titan.posttraining import chronological_matrix as cm
from tests.posttraining.campaign_fixture import CpuLease
from tests.posttraining.real_only import real_data_only
from tests.training.test_titans_campaign import (
    ARM,
    campaign_run,  # noqa: F401
    permitted,  # noqa: F401
    selected_search,
    unfused,  # noqa: F401
)
from tests.training.test_titans_walk_forward import Recorder

CONFIGS = Path("configs/posttraining")


class StageRecorder(Recorder):
    """Registrador de las ventanas de Titans-MAC con la firma de AdamW.

    No guarda copias de los gradientes, que en 14 ajustes ocupan varios GB. Solo cuenta
    los pasos y exige pesos sin cambios y gradientes finitos.
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


def write_stage(campaign_run, folder):  # noqa: F811
    """Etapa B con el brazo Titans-MAC, la continuación y el adaptador de la cabeza."""
    folder.mkdir(parents=True, exist_ok=True)
    campaign = json.loads(Path(campaign_run.campaign).read_text())
    matrix = json.loads((CONFIGS / "adapter-matrix-v3.json").read_text())
    matrix["budget"].update(seeds=[42], epochs=1, batch_size=campaign["neural"]["batch_size"])
    atomic_json(folder / "matrix.json", matrix)
    stage = json.loads((CONFIGS / "historical-masked-adapter-stage-b.json").read_text())
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


@pytest.fixture(scope="module")
def stage_run(campaign_run, tmp_path_factory, two_cases):  # noqa: F811
    folder = tmp_path_factory.mktemp("chronological-stage")
    path = write_stage(campaign_run, folder / "config")
    StageRecorder.made.clear()
    with pytest.MonkeyPatch.context() as patch, real_data_only():
        patch.setattr(torch.optim, "AdamW", StageRecorder)
        summary = campaign_stage.run_stage(
            path,
            campaign_run.views,
            campaign_run.output,
            folder / "stage",
            lease=CpuLease,
            device="cpu",
            stop=type("Stop", (), {"requested": False})(),
        )
    return dict(path=path, output=folder / "stage", summary=summary, made=list(StageRecorder.made))


def receipts(output):
    return {
        str(path.parent.relative_to(output / "jobs")): json.loads(path.read_text())
        for path in (output / "jobs").rglob("receipt.json")
    }


def rows(path):
    table = pq.read_table(path).sort_by("sample_id")
    return {name: table[name].to_pylist() for name in ("sample_id", "target", *QUANTILE_COLUMNS)}


def test_stage_plans_titans_arms_with_their_cases(stage_run):
    stage = campaign_stage.load_stage(stage_run["path"])
    active, awaiting = campaign_stage.stage_arms(stage)
    assert active == {
        ARM: dict(family=cm.TITANS, variant="mac_online", bank=True, design=cm.TITANS)
    }
    assert awaiting == {}
    jobs = campaign_stage.plan_stage(stage)
    arms = {job["arm"] for job in jobs}
    assert arms == {f"{ARM}__full_continuation", f"{ARM}__head"}
    counts = campaign_stage.count_stage(stage, jobs)
    assert (counts["training_jobs"], counts["prediction_jobs"]) == (14, 24)


def test_fits_and_carries_reproduce_the_base_rows_without_weight_changes(
    stage_run,
    campaign_run,  # noqa: F811
):
    summary, output = stage_run["summary"], stage_run["output"]
    assert summary["status"] == "completed"
    assert summary["completed"] == summary["planned"] == dict(training_jobs=14, prediction_jobs=24)
    assert stage_run["made"] and all(item.calls > 0 for item in stage_run["made"])
    found = receipts(output)
    by_parent = {}
    for job_id, receipt in found.items():
        identity = receipt["identity"]
        scope, window = identity["scope"], identity["window"]
        if identity["kind"] == "fit":
            by_parent.setdefault((window, identity["seed"]), set()).add(receipt["updates"])
            _, base = selected_search(campaign_run.output, window)
        else:
            base = json.loads(
                (
                    campaign_run.output / "jobs" / scope / window / ARM / "carry-s42/receipt.json"
                ).read_text()
            )
            assert receipt["updates"] == 0 and receipt["score"] is None
        for partition in ("calibration", "evaluation"):
            mine = rows(output / receipt["predictions"][partition]["path"])
            theirs = rows(campaign_run.output / base["predictions"][partition]["path"])
            # mac_online con correcciones nulas reproduce los bits del padre.
            assert mine == theirs, (job_id, partition)
    # Igualdad de actualizaciones entre la continuación y el adaptador de cada padre.
    assert all(len(values) == 1 and min(values) > 0 for values in by_parent.values())


def test_adapter_fits_train_only_the_head_corrections(stage_run):
    output = stage_run["output"]
    heads = [
        receipt
        for receipt in receipts(output).values()
        if receipt["identity"]["kind"] == "fit" and receipt["identity"]["point"] == "head"
    ]
    assert len(heads) == 7
    for receipt in heads:
        report = json.loads((output / receipt["run"]["path"]).read_text())
        posttraining = report["identity"]["posttraining"]
        assert posttraining["adapter"]["trainable_parameters"] > 0
        assert posttraining["base_parameters_sha256"]
    roles = {tuple(group["role"] for group in item.param_groups) for item in stage_run["made"]}
    assert ("adapters",) in roles and any(len(value) > 1 for value in roles)


def test_confirmed_jobs_release_their_observation_indices(stage_run):
    output = stage_run["output"]
    assert not [path for path in (output / "jobs").rglob("indices") if path.is_dir()]
    assert stage_run["summary"]["released_index_bytes"]
    # Los brazos cronológicos no abren la ventana de cohortes de los brazos neuronales.
    assert not (output / "windows-data").exists()


def test_rerunning_confirms_without_new_optimizers(stage_run, campaign_run):  # noqa: F811
    StageRecorder.made.clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.optim, "AdamW", StageRecorder)
        again = campaign_stage.run_stage(
            stage_run["path"],
            campaign_run.views,
            campaign_run.output,
            stage_run["output"],
            lease=CpuLease,
            device="cpu",
            stop=type("Stop", (), {"requested": False})(),
        )
    assert again["status"] == "completed" and not StageRecorder.made
