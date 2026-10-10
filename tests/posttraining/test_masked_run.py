"""Brazos de la matriz y controles con máscaras, hasta el paso del optimizador.

El optimizador inyectado no hereda de torch.optim.Optimizer y nunca cambia pesos.
Registra los parámetros recibidos y sus gradientes en cada llamada. Así se recorre
lectura, forward, pérdida, backward, recorte, checkpoint, selección y evaluación sin
aplicar ninguna actualización mientras siga el bloqueo de aprendizaje.
"""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.environments.actions import ActionGrid
from mars_titan.posttraining import adapter_matrix
from mars_titan.posttraining.heldout import _adjustment, evaluate_partition
from mars_titan.posttraining.inputs import PairedInputs, fit_normalization
from mars_titan.posttraining.parent_cache import ParentCache
from mars_titan.posttraining.parents import load_parent
from mars_titan.posttraining.run import run_case
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.kernel_policy import FP32_STRICT
from tests.posttraining.masked_fixture import masked_ordered, masked_parent, sources


@pytest.fixture
def edition(tmp_path):
    view, ordered, report = masked_ordered(tmp_path / "data")
    yield SimpleNamespace(view=view, ordered=ordered, report=report)


def setup(edition, folder, kind):
    path, _ = masked_parent(edition.ordered, folder / "parent", kind)
    parent = load_parent(edition.ordered, path, device="cpu", diagnostic=True)
    train, validation = sources(edition.ordered)
    cache = ParentCache(
        folder / "parent.sqlite", parent.identity["checkpoint_sha256"], "masked", parent.predict
    )
    data = PairedInputs(train, validation, cache)
    grid = ActionGrid.from_dict(edition.report["grid"])
    return parent, data, grid, fit_normalization(data, batch_size=2), (train, validation, cache)


def matrix_cases(family):
    declared, digest = adapter_matrix.read_matrix("configs/posttraining/adapter-matrix-v1.json")
    return {item["id"]: item for item in adapter_matrix.cases(declared, digest, family)}, declared


def predictions(folder):
    table = pq.read_table(folder / "validation-predictions.parquet").to_pydict()
    return np.asarray(table["center"]), np.asarray(table["parent"])


@pytest.mark.parametrize(
    ("kind", "arm"),
    [
        ("gru", "head+fusion"),
        ("gru", "fusion_full_rank"),
        ("dlinear", "head"),
        ("transformer", "readout"),
        ("transformer", "head+readout+fusion"),
    ],
)
def test_adapter_arm_trains_only_its_parameters_and_starts_at_the_parent(
    tmp_path, recorder, edition, kind, arm
):
    parent, data, grid, scale, handles = setup(edition, tmp_path, kind)
    cases, declared = matrix_cases(kind)
    case = cases[f"seed-42/{arm}"]["case"]
    output = tmp_path / "arm"
    before = {k: v.clone() for k, v in parent.model.state_dict().items() if torch.is_tensor(v)}
    report = run_case(
        data, output, case, grid, scale, parent=parent, batch_size=2, device="cpu", diagnostic=True
    )
    assert report["status"] == "completed" and report["mode"] == "neural_mae"
    updates = declared["budget"]["epochs"] * data.budget("real", 2)["updates"]
    assert report["global_step"] == report["total_steps"] == updates
    identity = report["identity"]
    assert identity["adapter"]["trainable_parameters"] == report["trainable_parameters"]
    assert identity["adapter"]["seed"] == adapter_matrix.adapter_seed(case)
    assert {"posttraining/adapter_matrix.py", "data/input_policy.py"} <= set(identity["code"])
    (optimizer,) = recorder.optimizers
    assert len(optimizer.calls) == updates
    assert sum(value.numel() for value in optimizer.parameters) == report["trainable_parameters"]
    assert all(all(grad is not None for grad in call) for call in optimizer.calls)
    # Pesos sin cambios: la selección conserva el padre y la salida coincide con él.
    assert report["selection"]["best_epoch"] == 0
    center, inherited = predictions(output)
    np.testing.assert_array_equal(center, inherited)
    after = parent.model.state_dict()
    assert all(torch.equal(value, after[key]) for key, value in before.items())
    assert all(not value.requires_grad for value in parent.model.parameters())
    restored, _, neural = _adjustment(output / "run.json", report, parent, "cpu")
    assert neural and not any(value.requires_grad for value in restored.parameters())
    assert (
        sum(
            value.numel()
            for name, value in restored.named_parameters()
            if "parametrizations" in name and not name.endswith("original")
        )
        == report["trainable_parameters"]
    )
    for handle in handles:
        handle.close()


@pytest.mark.parametrize("control", ["linear_residual", "full_continuation"])
def test_controls_share_budget_and_start_from_the_parent_output(
    tmp_path, recorder, edition, control
):
    parent, data, grid, scale, handles = setup(edition, tmp_path, "transformer")
    cases, declared = matrix_cases("transformer")
    report = run_case(
        data,
        tmp_path / control,
        cases[f"seed-43/{control}"]["case"],
        grid,
        scale,
        parent=parent,
        batch_size=2,
        device="cpu",
        diagnostic=True,
    )
    assert report["global_step"] == declared["budget"]["epochs"] * data.budget("real", 2)["updates"]
    assert "adapter" not in report["identity"]
    center, inherited = predictions(tmp_path / control)
    np.testing.assert_array_equal(center, inherited)
    baseline = report["baseline"]
    assert baseline["center"]["absolute_error"] == baseline["parent"]["absolute_error"]
    (optimizer,) = recorder.optimizers
    expected = (
        sum(value.numel() for value in parent.model.parameters())
        if control == "full_continuation"
        else data.features + 1
    )
    assert sum(value.numel() for value in optimizer.parameters) == expected
    for handle in handles:
        handle.close()


STRICT = (False, False, "highest")


def tf32_flags():
    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )


def allow_tf32():
    """Valores que PyTorch admite por defecto o que otro componente del proceso podría fijar."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


@pytest.fixture
def restored_flags():
    before = tf32_flags()
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, precision = before
    torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize(("kind", "arm"), [("gru", "head+fusion"), ("transformer", "readout")])
def test_parent_and_adapter_run_with_the_strict_precision_of_the_parent_case(
    tmp_path, recorder, edition, restored_flags, kind, arm
):
    path, _ = masked_parent(edition.ordered, tmp_path / "parent", kind, precision=FP32_STRICT)
    allow_tf32()
    parent = load_parent(edition.ordered, path, device="cpu", diagnostic=True)
    # Cargar el padre fija la política que registró su ajuste.
    assert tf32_flags() == STRICT
    assert parent.precision == FP32_STRICT
    assert parent.identity["kernel_policy"]["cudnn_allow_tf32"] is False
    seen = []
    # El brazo copia el padre con este gancho, así que registra también su forward.
    parent.model.register_forward_pre_hook(lambda module, args: seen.append(tf32_flags()))
    train, validation = sources(edition.ordered)
    cache = ParentCache(
        tmp_path / "parent.sqlite", parent.identity["checkpoint_sha256"], "masked", parent.predict
    )
    data = PairedInputs(train, validation, cache)
    scale = fit_normalization(data, batch_size=2)
    steps = []
    recorder.on_step = lambda optimizer: steps.append(tf32_flags())
    # Otro componente vuelve a permitir TF32 antes del ajuste, que debe fijar la política.
    allow_tf32()
    cases, _ = matrix_cases(kind)
    report = run_case(
        data,
        tmp_path / "arm",
        cases[f"seed-42/{arm}"]["case"],
        grid=ActionGrid.from_dict(edition.report["grid"]),
        normalization=scale,
        parent=parent,
        batch_size=2,
        device="cpu",
        diagnostic=True,
    )
    assert report["status"] == "completed"
    assert seen and steps and set(seen) | set(steps) == {STRICT}
    numerics = report["identity"]["numerics"]
    assert (
        numerics["cuda_matmul_allow_tf32"],
        numerics["cudnn_allow_tf32"],
        numerics["matmul_precision"],
    ) == STRICT
    assert report["identity"]["parent"]["kernel_policy"] == parent.kernel_policy
    # Una predicción del padre con TF32 permitido se rechaza en lugar de cambiar sus bits.
    allow_tf32()
    raw = train(0)
    with pytest.raises(ValueError, match="política de precisión"):
        parent.predict(raw["inputs"], raw["presence"])
    for handle in (train, validation, cache):
        handle.close()


def test_an_adapter_whose_process_allows_tf32_mid_run_fails_at_the_next_save(
    tmp_path, recorder, edition, restored_flags
):
    path, _ = masked_parent(edition.ordered, tmp_path / "parent", precision=FP32_STRICT)
    parent = load_parent(edition.ordered, path, device="cpu", diagnostic=True)
    train, validation = sources(edition.ordered)
    cache = ParentCache(
        tmp_path / "parent.sqlite", parent.identity["checkpoint_sha256"], "masked", parent.predict
    )
    data = PairedInputs(train, validation, cache)
    scale = fit_normalization(data, batch_size=2)
    recorder.on_step = lambda optimizer: allow_tf32()
    cases, _ = matrix_cases("gru")
    with pytest.raises(ValueError, match="política de precisión"):
        run_case(
            data,
            tmp_path / "arm",
            cases["seed-42/head+fusion"]["case"],
            ActionGrid.from_dict(edition.report["grid"]),
            scale,
            parent=parent,
            batch_size=2,
            device="cpu",
            diagnostic=True,
            checkpoint_seconds=1e-9,
        )
    # El guardado que sigue al primer paso lo detecta, sin esperar a la evaluación final.
    assert len(recorder.optimizers[0].calls) == 1
    for handle in (train, validation, cache):
        handle.close()


def test_a_final_evaluation_under_tf32_is_not_confirmed(
    tmp_path, recorder, edition, restored_flags, monkeypatch
):
    from mars_titan.posttraining import run as module

    path, _ = masked_parent(edition.ordered, tmp_path / "parent", precision=FP32_STRICT)
    parent = load_parent(edition.ordered, path, device="cpu", diagnostic=True)
    train, validation = sources(edition.ordered)
    cache = ParentCache(
        tmp_path / "parent.sqlite", parent.identity["checkpoint_sha256"], "masked", parent.predict
    )
    data = PairedInputs(train, validation, cache)
    scale = fit_normalization(data, batch_size=2)
    real = module.evaluate

    def evaluate(*args, destination=None, **options):
        # Solo la evaluación final escribe predicciones. Antes de ella se permite TF32.
        if destination is not None:
            allow_tf32()
        return real(*args, destination=destination, **options)

    monkeypatch.setattr(module, "evaluate", evaluate)
    cases, _ = matrix_cases("gru")
    with pytest.raises(ValueError, match="política de precisión"):
        run_case(
            data,
            tmp_path / "arm",
            cases["seed-42/head+fusion"]["case"],
            ActionGrid.from_dict(edition.report["grid"]),
            scale,
            parent=parent,
            batch_size=2,
            device="cpu",
            diagnostic=True,
        )
    report = json.loads((tmp_path / "arm/run.json").read_text())
    assert report["status"] == "failed"
    for handle in (train, validation, cache):
        handle.close()


def test_paused_arm_resumes_its_adapter_state_and_cursor(tmp_path, recorder, edition):
    parent, data, grid, scale, handles = setup(edition, tmp_path, "gru")
    cases, declared = matrix_cases("gru")
    case = cases["seed-44/head+fusion"]["case"]
    options = dict(parent=parent, batch_size=2, device="cpu", diagnostic=True)
    paused = run_case(data, tmp_path / "arm", case, grid, scale, max_updates=3, **options)
    assert paused["status"] == "paused" and paused["global_step"] == 3
    resumed = run_case(data, tmp_path / "arm", case, grid, scale, resume=True, **options)
    assert resumed["status"] == "completed"
    assert (
        resumed["global_step"] == declared["budget"]["epochs"] * data.budget("real", 2)["updates"]
    )
    assert sum(len(optimizer.calls) for optimizer in recorder.optimizers) == resumed["global_step"]
    other = cases["seed-44/fusion"]["case"]
    with pytest.raises(ValueError, match="identidad"):
        run_case(data, tmp_path / "arm", other, grid, scale, resume=True, **options)
    for handle in handles:
        handle.close()


def test_a_checkpoint_is_not_reused_by_another_arm(tmp_path, recorder, edition):
    parent, data, grid, scale, handles = setup(edition, tmp_path, "gru")
    cases, _ = matrix_cases("gru")
    options = dict(parent=parent, batch_size=2, device="cpu", diagnostic=True)
    reports = {
        arm: run_case(data, tmp_path / arm, cases[f"seed-42/{arm}"]["case"], grid, scale, **options)
        for arm in ("fusion", "fusion_full_rank")
    }
    with pytest.raises(ValueError):
        _adjustment(tmp_path / "fusion/run.json", reports["fusion_full_rank"], parent, "cpu")
    for handle in handles:
        handle.close()


def test_frozen_evaluation_reads_presence_and_rebuilds_the_arm(tmp_path, recorder, edition):
    parent, data, grid, scale, handles = setup(edition, tmp_path, "transformer")
    cases, _ = matrix_cases("transformer")
    report = run_case(
        data,
        tmp_path / "arm",
        cases["seed-42/readout+fusion"]["case"],
        grid,
        scale,
        parent=parent,
        batch_size=2,
        device="cpu",
        diagnostic=True,
    )
    model, action_grid, neural = _adjustment(tmp_path / "arm/run.json", report, parent, "cpu")
    dataset = CorpusDataset(edition.view, input_policy=HISTORICAL_MASKED)
    for partition in ("calibration", "evaluation"):
        path = tmp_path / f"{partition}.parquet"
        scores = evaluate_partition(
            dataset,
            parent,
            partition,
            path,
            stop=StopRequest(),
            device="cpu",
            model=model,
            grid=action_grid,
            neural=neural,
        )
        assert scores["center"]["samples"] == edition.report["counts"][partition]
        table = pq.read_table(path).to_pydict()
        np.testing.assert_array_equal(table["center"], table["parent"])
    for handle in handles:
        handle.close()


def test_masked_queue_declares_the_policy_and_runs_until_each_step(
    tmp_path, recorder, edition, monkeypatch
):
    import importlib

    from mars_titan.data.cohort_files import read_manifest
    from mars_titan.data.storage import atomic_json, sha256
    from mars_titan.posttraining.queue import read_design, run_queue

    module = importlib.import_module("mars_titan.posttraining.queue")
    path, _ = masked_parent(edition.ordered, tmp_path / "parent")
    proof = dict(
        manifest=str(edition.view),
        manifest_sha256=sha256(edition.view),
        counts=edition.report["counts"],
        parents={"gru": dict(report=str(path), sha256=sha256(path))},
    )

    class CpuLease:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def check(self):
            pass

    real_run = module.run_case

    def cpu_run(*args, **kwargs):
        return real_run(*args, **dict(kwargs, device="cpu", diagnostic=True, lease=None))

    monkeypatch.setattr(module, "GpuLease", CpuLease)
    monkeypatch.setattr(module, "selected_parents", lambda *_args: proof)
    monkeypatch.setattr(
        module,
        "encoder_contract",
        lambda *_args: dict(supervision_sha256=proof["manifest_sha256"], encoders={}),
    )
    monkeypatch.setattr(
        module,
        "load_parent",
        lambda ordered, report, **_: load_parent(ordered, report, device="cpu", diagnostic=True),
    )
    monkeypatch.setattr(module, "run_case", cpu_run)
    plan = read_design("configs/baselines/real-continuations-v2.json")[0]
    plan.update(seeds=[7], epochs=1, batch_size=2, auxiliary_samples=2)
    config = tmp_path / "config.json"
    for invalid in ("strict_inputs_v1", "historical_masked_v0"):
        atomic_json(config, plan | {"input_policy": invalid})
        with pytest.raises(ValueError, match="diseño"):
            read_design(config)
    atomic_json(config, plan | {"input_policy": HISTORICAL_MASKED})
    output = tmp_path / "queue"
    inputs = tmp_path / "inputs"
    args = (config, inputs / "reference.json", inputs / "tabular.json", inputs / "encoded", output)
    summary = run_queue(*args)
    assert summary["status"] == "completed"
    assert summary["completed_runs"] == summary["planned_runs"] == 8
    ordered = read_manifest(output / "ordered/manifest.json")[0]
    assert ordered["input_policy"] == HISTORICAL_MASKED
    normalization = read_manifest(output / "parents/gru/normalization.json")[0]
    assert (
        normalization["window"]["train_partition_sha256"]
        == (ordered["partitions"]["train"]["sha256"])
    )
    for mode in ("mae", "klpo_exact", "neural_mae"):
        result = read_manifest(output / f"parents/gru/runs/seed-7/real/{mode}/run.json")[0]
        assert result["identity"]["parent"]["input_policy"] == HISTORICAL_MASKED
        assert result["global_step"] == result["total_steps"]
    assert sum(len(optimizer.calls) for optimizer in recorder.optimizers) == 8 * 3
    # Repetir la cola verifica los casos confirmados sin volver a ajustarlos.
    calls = len(recorder.optimizers)
    assert run_queue(*args)["status"] == "completed"
    assert len(recorder.optimizers) == calls
