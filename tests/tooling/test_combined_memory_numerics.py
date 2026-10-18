"""Aislar el grafo de las proyecciones en inferencia, sin cambiar pesos compartidos."""

import copy
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from benchmarks import combined_memory_numerics as audit
from mars_titan.models.titans.config import MemoryConfig
from mars_titan.models.titans.neural_memory import NeuralMemory


def memory(device="cpu"):
    return NeuralMemory(MemoryConfig(dim=4, depth=2), dtype=torch.float64, device=device).eval()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_local_projection_graph_preserves_parameters_and_internal_adaptation(shared, device):
    model = memory(device)
    original = {name: p.clone().detach() for name, p in model.named_parameters()}
    frozen = copy.deepcopy(model).requires_grad_(False)
    state = model.initial_state(1 if shared else 2)
    token = torch.ones(2, 1, 4, dtype=torch.float64, device=device)
    flags = []
    with torch.no_grad(), audit.projection_local_graph(model):
        handle = model.key_projection.register_forward_hook(
            lambda module, inputs, output: flags.append(output.requires_grad)
        )
        after = model.update_shared(token[:, 0], state) if shared else model.update(token, state)
        handle.remove()
    assert flags == [False]
    assert after.steps.tolist() == [1] * (1 if shared else 2)
    with torch.no_grad():
        expected = (
            frozen.update_shared(token[:, 0], state) if shared else frozen.update(token, state)
        )
    for actual, reference in zip(
        (*after.weights, *after.momentum), (*expected.weights, *expected.momentum), strict=True
    ):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert any(not torch.equal(a, b) for a, b in zip(after.weights, state.weights, strict=True))
    assert all(
        p.requires_grad and p.grad is None and torch.equal(p, original[name])
        for name, p in model.named_parameters()
    )
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules())


def test_projection_failure_restores_gradient_mode_and_hooks(monkeypatch):
    model = memory()

    def fail(*args):
        raise ValueError("fallo deliberado")

    monkeypatch.setattr(model.key_projection, "forward", fail)
    assert torch.is_grad_enabled()
    with torch.no_grad():
        with pytest.raises(ValueError, match="deliberado"):
            with audit.projection_local_graph(model):
                model.update(torch.ones(2, 1, 4, dtype=torch.float64), model.initial_state(2))
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules())


def test_projection_diagnostic_requires_an_outer_inference_scope():
    with pytest.raises(ValueError, match="no_grad"):
        with audit.projection_local_graph(memory()):
            pass
    with torch.no_grad(), pytest.raises(ValueError, match="no_grad"):
        with audit.projection_local_graph(memory().train()):
            pass


def test_numerical_replay_keeps_the_learning_hold(tmp_path, monkeypatch):
    from mars_titan.training.learning_hold import LearningHoldError

    path = tmp_path / "hold.json"
    path.write_text('{"training_allowed":false}')
    monkeypatch.setenv("MARS_TITAN_TRAINING_HOLD", str(path))
    with pytest.raises(LearningHoldError):
        audit.replay(tmp_path, "fold-018/cell7-42", "fp32_tracked", tmp_path / "out")


@pytest.mark.parametrize("route,setting", [("fp32_flat_tracked", None), ("fp32_frozen", "1")])
def test_linear_dispatch_mode_is_explicit_before_opening_sources(
    tmp_path, monkeypatch, learning_doubles, route, setting
):
    if setting is None:
        monkeypatch.delenv("TORCH_LINEAR_FLATTEN_3D", raising=False)
    else:
        monkeypatch.setenv("TORCH_LINEAR_FLATTEN_3D", setting)
    with pytest.raises(ValueError, match="TORCH_LINEAR"):
        audit.replay(tmp_path, "fold-018/cell7-42", route, tmp_path / "out")


def test_numerical_replay_has_no_cpu_fallback(tmp_path, monkeypatch, learning_doubles):
    monkeypatch.delenv("TORCH_LINEAR_FLATTEN_3D", raising=False)
    monkeypatch.setattr(audit, "verify_prepared", lambda _: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda:0"):
        audit.replay(tmp_path, "fold-018/cell0-42", "fp32_frozen", tmp_path / "out")


@pytest.fixture
def replay_boundary(tmp_path, monkeypatch, learning_doubles):
    """Dobles en las fronteras de corpus y CUDA. El contrato de publicación es real."""

    @dataclass
    class Recipe:
        precision: str | None = "fp32_strict"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 1).eval()
            self.eval()

        def _parameter_digest(self):
            return tuple(tuple(p.detach().flatten().tolist()) for p in self.parameters())

    state = SimpleNamespace(model=Model(), fault=None, recipe=None)
    table = pa.table(dict(sample_id=["a", "b"], prediction=[0.1, 0.2]))
    parent = tmp_path / "runs/fold-018/cell7-42"
    parent.mkdir(parents=True)
    pq.write_table(table, parent / "original.parquet")
    report = dict(
        identity=dict(recipe={}, window={}),
        request={},
        checkpoint=dict(sha256="a" * 64),
        predictions=dict(evaluation=dict(path="original.parquet")),
    )
    monkeypatch.setattr(audit, "verify_prepared", lambda _: None)
    monkeypatch.setattr(audit.mars, "_parent", lambda *a: (report, "b" * 64))
    monkeypatch.setattr(audit.mars, "_frozen_parent", lambda *a: state.model)
    monkeypatch.setattr(audit, "CorpusDataset", lambda *a, **kw: object())
    monkeypatch.setattr(audit, "window_phases", lambda *a: {})
    monkeypatch.setattr(audit.window, "walk_forward_options", lambda _: dict(warmup_months=12))
    monkeypatch.setattr(
        audit.window,
        "_sources",
        lambda *a: dict(train=SimpleNamespace(specification=lambda: {}), evaluation=object()),
    )
    monkeypatch.setattr(audit.window, "case_recipe", lambda *a: Recipe())
    monkeypatch.setattr(audit, "_numerics", lambda: dict(kind="fixture_cuda_boundary"))
    monkeypatch.setattr(audit, "_implementation", lambda: dict(kind="fixture"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("set_device", "set_per_process_memory_fraction", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 4)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: 8)
    monkeypatch.setattr(torch, "set_num_threads", lambda *a: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *a: None)

    class Inference:
        def __init__(self, model, recipe):
            assert model is state.model
            state.recipe = recipe

        def predict(self, source, rows):
            assert not torch.is_grad_enabled()
            if state.fault == "weights":
                state.model.head.weight.add_(1)
            elif state.fault == "gradients":
                state.model.head.weight.grad = torch.ones_like(state.model.head.weight)
            elif state.fault == "optimizer":
                torch.optim.SGD(state.model.parameters(), lr=0.01).step()
            return dict(samples=2)

    monkeypatch.setattr(audit, "ChronologicalInference", Inference)
    monkeypatch.setattr(
        audit.window,
        "checked_tables",
        lambda *a: [table.slice(1) if state.fault == "rows" else table],
    )
    return state


@pytest.mark.parametrize("route", audit.ROUTES)
def test_replay_publishes_only_predictions_and_preserves_parameter_contracts(
    tmp_path, monkeypatch, replay_boundary, route
):
    monkeypatch.setenv("TORCH_LINEAR_FLATTEN_3D", "1" if "_flat_" in route else "0")
    out = tmp_path / "out"
    result = audit.replay(tmp_path, "fold-018/cell7-42", route, out)
    assert result == json.loads((out / "run.json").read_text())
    assert result["optimizer_steps"] == 0 and result["final_test_opened"] is False
    assert result["shared_parameters_unchanged"] and result["shared_gradients_none"]
    assert result["maximum_difference_to_original"] == result["mean_difference_to_original"] == 0
    assert result["target_metrics_included"] is False
    assert pq.read_table(out / "predictions.parquet").column_names == ["sample_id", "prediction"]
    assert result["predictions"]["sha256"] == audit.sha256(out / "predictions.parquet")
    assert all(
        p.requires_grad == (not route.endswith("_frozen"))
        for p in replay_boundary.model.parameters()
    )
    if route.startswith("fp64"):
        assert replay_boundary.model.head.weight.dtype == torch.float64
        assert replay_boundary.recipe.precision is None
    else:
        assert replay_boundary.model.head.weight.dtype == torch.float32
        assert replay_boundary.recipe.precision == "fp32_strict"
    with pytest.raises(FileExistsError):
        audit.replay(tmp_path, "fold-018/cell7-42", route, out)


@pytest.mark.parametrize("fault", ["weights", "gradients", "rows", "optimizer"])
def test_replay_rejects_learning_or_changed_rows(tmp_path, monkeypatch, replay_boundary, fault):
    monkeypatch.delenv("TORCH_LINEAR_FLATTEN_3D", raising=False)
    replay_boundary.fault = fault
    with pytest.raises((ValueError, RuntimeError)):
        audit.replay(tmp_path, "fold-018/cell7-42", "fp32_tracked", tmp_path / "out")
    assert not (tmp_path / "out/run.json").exists()
    # El gancho temporal se retira incluso si un optimizador intentó entrar.
    from torch.optim.optimizer import _global_optimizer_pre_hooks

    assert audit.forbid_optimizer_steps not in _global_optimizer_pre_hooks.values()


def test_replay_rejects_an_undeclared_parent(tmp_path, learning_doubles):
    with pytest.raises(ValueError, match="referencias declaradas"):
        audit.replay(tmp_path, "fold-015/cell7-42", "fp32_tracked", tmp_path / "out")


def test_numerical_cli_forwards_paths_and_route(tmp_path, monkeypatch, capsys):
    import sys

    calls = []
    monkeypatch.setattr(audit, "replay", lambda *a: calls.append(a) or dict(status="fixture"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--work",
            str(tmp_path),
            "--parent",
            "fold-018/cell7-42",
            "--route",
            "fp64_frozen",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    audit.main()
    assert calls == [(tmp_path, "fold-018/cell7-42", "fp64_frozen", tmp_path / "out")]
    assert json.loads(capsys.readouterr().out) == dict(status="fixture")
