"""Modo matriz de la cola: brazos y controles por padre y semilla, hasta el paso.

El optimizador del fixture `recorder` no hereda de `torch.optim.Optimizer` y exige pesos
sin cambios. La GPU, la selección de padres y el contrato de codificadores se sustituyen
para recorrer la cola en CPU con el corpus técnico con máscaras.
"""

import importlib
import json
from types import SimpleNamespace

import pytest

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.posttraining import adapter_matrix
from mars_titan.posttraining.matrix_runs import MatrixParent
from mars_titan.training.learning_hold import LearningHoldError
from tests.posttraining.masked_fixture import masked_ordered, masked_parent

queue = importlib.import_module("mars_titan.posttraining.queue")


class CpuLease:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def check(self):
        pass


@pytest.fixture
def edition(tmp_path):
    view, ordered, report = masked_ordered(tmp_path / "data")
    return SimpleNamespace(view=view, ordered=ordered, report=report)


def matrix(folder, seeds=(42, 43)):
    value = json.loads(open("configs/posttraining/adapter-matrix-v2.json").read())
    value["budget"].update(seeds=list(seeds), epochs=1, batch_size=2)
    path = folder / "matrix.json"
    atomic_json(path, value)
    return path


def proof(edition, folder, heads):
    records = {}
    for family, head in heads.items():
        records[family] = {}
        for seed in (42, 43):
            path, _ = masked_parent(
                edition.ordered, folder / f"{family}-{seed}", family, seed=seed, head=head
            )
            records[family][str(seed)] = dict(
                report=str(path), sha256=sha256(path), parent_seed=seed, shared_deterministic=False
            )
    return dict(
        schema_version=2,
        parent_seed_policy="matching",
        manifest=str(edition.view),
        manifest_sha256=sha256(edition.view),
        counts=edition.report["counts"],
        parents={family: values["42"] for family, values in records.items()},
        parents_by_seed=records,
    )


def patch(monkeypatch, value):
    monkeypatch.setattr(queue, "GpuLease", CpuLease)
    monkeypatch.setattr(queue, "matching_parents", lambda *_args, **_kwargs: value)
    monkeypatch.setattr(
        queue,
        "encoder_contract",
        lambda *_args: dict(supervision_sha256=value["manifest_sha256"], encoders={}),
    )
    monkeypatch.setattr(
        queue,
        "MatrixParent",
        lambda *args, **kwargs: MatrixParent(*args, **dict(kwargs, device="cpu", lease=None)),
    )


def test_matrix_mode_runs_every_case_per_parent_and_seed_and_resumes(
    tmp_path, recorder, edition, monkeypatch
):
    value = proof(edition, tmp_path / "parents", {"gru": QUANTILE_HEAD, "dlinear": None})
    patch(monkeypatch, value)
    config = matrix(tmp_path)
    declared, digest = adapter_matrix.read_matrix(config)
    output = tmp_path / "queue"
    inputs = tmp_path / "inputs"
    args = (config, inputs / "reference.json", inputs / "tabular.json", inputs / "encoded", output)
    stop = SimpleNamespace(requested=False)

    def pause(optimizer):
        if sum(len(item.calls) for item in recorder.optimizers) == 5:
            stop.requested = True

    recorder.on_step = pause
    paused = queue.run_queue(*args, stop=stop)
    assert paused["status"] == "paused" and paused["kind"] == "adapter_matrix_queue"
    stop.requested, recorder.on_step = False, None
    summary = queue.run_queue(*args, stop=stop)
    assert summary["status"] == "completed"
    # GRU de cuantiles: cuatro brazos y la continuación. DLinear escalar: añade la lineal.
    expected = {
        f"{family}/{item['id']}"
        for family, head in (("gru", QUANTILE_HEAD), ("dlinear", "scalar"))
        for item in adapter_matrix.cases(declared, digest, family, head=head)
    }
    assert set(summary["runs"]) == expected
    assert summary["planned_runs"] == summary["completed_runs"] == len(expected) == 2 * (5 + 6)
    # Mismas actualizaciones en todos los brazos y controles, iguales al plan registrado.
    updates = {run["updates"] for run in summary["runs"].values()}
    assert len(updates) == 1
    for name, budget in summary["budgets"].items():
        assert {row["updates"] for row in budget["rows"][1:]} == updates
        assert budget["rows"][0]["control"] == "frozen_parent"
        if name.startswith("gru/"):
            assert budget["head"] == QUANTILE_HEAD and set(budget["excluded"]) == {
                "linear_residual"
            }
        else:
            assert budget["head"] == "scalar" and budget["excluded"] == {}
    (count,) = updates
    assert sum(len(item.calls) for item in recorder.optimizers) == len(expected) * count
    for key, run in summary["runs"].items():
        report = read_manifest(output / run["path"])[0]
        mode = report["identity"]["case"]["mode"]
        assert mode == ("neural_pinball" if key.startswith("gru/") else report["mode"])
        assert report["global_step"] == count
    # Repetir verifica los casos confirmados sin crear optimizadores.
    created = len(recorder.optimizers)
    assert queue.run_queue(*args, stop=stop)["status"] == "completed"
    assert len(recorder.optimizers) == created
    # El plan de cada padre se registró antes del primer ajuste y no se reescribe.
    plan = output / "parents/gru/seed-42/plan.json"
    value = read_manifest(plan)[0]
    assert value == summary["budgets"]["gru/seed-42"] and value["matrix_sha256"] == digest
    value["rows"][1]["updates"] += 1
    atomic_json(plan, value)
    with pytest.raises(ValueError, match="plan"):
        queue.run_queue(*args, stop=stop)


def test_matrix_mode_keeps_the_objective_mode_separate(tmp_path, monkeypatch):
    # La cola de #128 sigue leyendo su diseño. Una matriz no pasa por read_design.
    with pytest.raises(ValueError):
        queue.read_design(matrix(tmp_path))
    plan, cases, _ = queue.read_design("configs/baselines/real-continuations-v2.json")
    assert {item["case"]["mode"] for item in cases("gru")} == {
        *plan["modes"],
        *plan["neural_controls"],
    }
    assert "neural_pinball" not in {item["case"]["mode"] for item in cases("gru")}


def test_matrix_mode_stops_before_reading_with_the_hold(tmp_path, learning_hold, monkeypatch):
    learning_hold(False)
    monkeypatch.setattr(queue, "GpuLease", lambda: pytest.fail("No debe reservar la GPU"))
    output = tmp_path / "queue"
    with pytest.raises(LearningHoldError):
        queue.run_matrix_queue(
            matrix(tmp_path), tmp_path / "r.json", tmp_path / "t.json", tmp_path / "e", output
        )
    assert not output.exists()
