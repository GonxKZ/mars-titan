"""Registrador de trazas: estadísticas, cadencia, presupuesto, recuperación y paridad.

Los tensores y el modelo pequeño se escriben en la prueba. Hay pasadas hacia delante y
hacia atrás, pero ningún optimizador: los pesos solo cambian en la prueba de la razón de
actualización, donde se modifican a mano para comprobar la fórmula.
"""

import json

import numpy as np
import pytest
import torch

from mars_titan.learning_traces.parameters import (
    OTHER,
    UpdateProbe,
    group_parameters,
    record_gradients,
)
from mars_titan.learning_traces.recorder import (
    STATS,
    TraceConfig,
    TraceRecorder,
    read_traces,
    tensor_stats,
)

IDENTITY = dict(arm="fixture", seed=42, window="fold-000")


def recorder(tmp_path, every=2, max_bytes=1 << 20, enabled=True):
    return TraceRecorder(tmp_path / "traces", TraceConfig(every, max_bytes, enabled), IDENTITY)


def test_stats_match_numpy_over_finite_values():
    values = torch.tensor([1.0, -3.0, float("nan"), 2.5, float("inf"), 0.5], dtype=torch.float32)
    stats = dict(zip(STATS, tensor_stats(values).tolist(), strict=True))
    finite = np.array([1.0, -3.0, 2.5, 0.5])
    assert stats["mean"] == pytest.approx(finite.mean(), rel=1e-15)
    assert stats["std"] == pytest.approx(finite.std(), rel=1e-15)
    assert (stats["min"], stats["max"], stats["abs_max"]) == (-3.0, 2.5, 3.0)
    assert stats["l2"] == pytest.approx(np.linalg.norm(finite), rel=1e-15)
    assert stats["finite_fraction"] == pytest.approx(4 / 6)


def test_stats_without_finite_values_are_undefined():
    nothing = tensor_stats(torch.tensor([float("nan"), float("inf")]))
    assert np.isnan(nothing[:-1].numpy()).all() and nothing[-1].item() == 0.0
    assert np.isnan(tensor_stats(torch.empty(0)).numpy()).all()


def test_flushes_write_numbered_parts_with_the_declared_schema(tmp_path):
    traces = recorder(tmp_path)
    assert [traces.due(step) for step in range(5)] == [True, False, True, False, True]
    traces.tensor("activation", torch.arange(4.0), group="memory")
    traces.scalar("loss", torch.tensor(0.25))
    traces.scalar("learning_rate", 1e-3)
    assert traces.flush(0, "train") == len(STATS) + 2
    traces.scalar("loss", 0.125)
    assert traces.flush(2, "train") == 1
    assert traces.flush(4, "train") == 0
    manifest, table = read_traces(tmp_path / "traces")
    assert manifest["identity"] == IDENTITY and manifest["exhausted_at_step"] is None
    rows = table.to_pylist()
    assert sorted({row["step"] for row in rows}) == [0, 2]
    loss = [row["value"] for row in rows if row["metric"] == "loss"]
    assert loss == [0.25, 0.125]
    mean = [r for r in rows if r["metric"] == "activation" and r["stat"] == "mean"]
    assert mean[0]["value"] == 1.5 and mean[0]["group"] == "memory"


def test_invalid_names_are_rejected(tmp_path):
    traces = recorder(tmp_path)
    with pytest.raises(ValueError, match="Nombre"):
        traces.scalar("Loss with spaces", 1.0)
    with pytest.raises(ValueError, match="Fase"):
        traces.flush(0, "Train!")


def test_the_budget_is_declared_when_exhausted(tmp_path):
    traces = recorder(tmp_path, every=1, max_bytes=4096)
    step = 0
    while traces.active:
        for index in range(20):
            traces.tensor(f"layer_{index}", torch.randn(64))
        traces.flush(step, "train")
        step += 1
        assert step < 100
    manifest, table = read_traces(tmp_path / "traces")
    assert manifest["exhausted_at_step"] == step - 1
    assert sum(p.stat().st_size for p in (tmp_path / "traces" / "parts").iterdir()) <= 4096
    traces.scalar("loss", 1.0)
    assert traces.flush(step, "train") == 0
    assert set(table.column("step").to_pylist()) == set(range(step - 1))


def test_resuming_removes_parts_written_after_the_checkpoint(tmp_path):
    traces = recorder(tmp_path, every=1)
    for step in range(2):
        traces.scalar("loss", float(step))
        traces.flush(step, "train")
    saved = json.loads(json.dumps(traces.state_dict()))
    traces.scalar("loss", 99.0)
    traces.flush(2, "train")
    resumed = recorder(tmp_path, every=1)
    resumed.load_state_dict(saved)
    resumed.scalar("loss", 2.0)
    resumed.flush(2, "train")
    _, table = read_traces(tmp_path / "traces")
    assert table.column("value").to_pylist() == [0.0, 1.0, 2.0]


def test_a_missing_confirmed_part_is_detected(tmp_path):
    traces = recorder(tmp_path, every=1)
    for step in range(2):
        traces.scalar("loss", float(step))
        traces.flush(step, "train")
    saved = traces.state_dict()
    (tmp_path / "traces" / "parts" / "part-00000001.parquet").unlink()
    with pytest.raises(ValueError, match="Faltan"):
        recorder(tmp_path, every=1).load_state_dict(saved)


def test_a_disabled_recorder_writes_nothing(tmp_path):
    traces = recorder(tmp_path, enabled=False)
    assert not traces.due(0)
    traces.tensor("activation", torch.ones(3))
    assert traces.flush(0, "train") == 0
    assert not (tmp_path / "traces").exists()


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = torch.nn.Linear(6, 8)
        self.head = torch.nn.Linear(8, 1)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, traces=None):
        hidden = torch.tanh(self.memory(x))
        if traces is not None:
            traces.tensor("hidden", hidden, group="memory")
        return self.head(self.dropout(hidden))


def gradients_and_rng(traces):
    torch.manual_seed(5)
    model = Tiny()
    x = torch.randn(32, 6)
    loss = model(x, traces).square().mean()
    loss.backward()
    if traces is not None:
        record_gradients(
            traces, group_parameters(model.named_parameters(), {"memory": ["memory."]})
        )
        traces.scalar("loss", loss)
        traces.flush(0, "train")
    grads = [p.grad.clone() for p in model.parameters()]
    return grads, torch.random.get_rng_state()


def test_traces_do_not_change_gradients_or_the_random_state(tmp_path):
    plain, plain_rng = gradients_and_rng(None)
    traced, traced_rng = gradients_and_rng(recorder(tmp_path, every=1))
    for a, b in zip(plain, traced, strict=True):
        assert torch.equal(a, b)
    assert torch.equal(plain_rng, traced_rng)
    _, table = read_traces(tmp_path / "traces")
    groups = {(r["metric"], r["group"]) for r in table.to_pylist()}
    assert {("grad_l2", "memory"), ("grad_l2", OTHER), ("weight_l2", "memory")} <= groups


def test_parameter_groups_are_disjoint_and_complete():
    model = Tiny()
    grouped = group_parameters(model.named_parameters(), {"memory": ["memory."]})
    assert {name for name, _ in grouped["memory"]} == {"memory.weight", "memory.bias"}
    assert {name for name, _ in grouped[OTHER]} == {"head.weight", "head.bias"}
    with pytest.raises(ValueError, match="varios grupos"):
        group_parameters(model.named_parameters(), {"a": ["memory."], "b": ["memory.w"]})


def test_the_update_ratio_follows_its_formula(tmp_path):
    model = Tiny()
    grouped = group_parameters(model.named_parameters(), {"head": ["head."]})
    old = [p.detach().clone() for _, p in grouped["head"]]
    probe, traces = UpdateProbe(), recorder(tmp_path, every=1)
    probe.before(grouped)
    with torch.no_grad():
        # Cambio escrito a mano para comprobar la fórmula. No es un paso de optimizador.
        model.head.weight.mul_(1.5)
    probe.after(traces, grouped)
    traces.flush(0, "train")
    _, table = read_traces(tmp_path / "traces")
    ratio = {r["group"]: r["value"] for r in table.to_pylist() if r["metric"] == "update_ratio"}
    delta = torch.cat(
        [(p.detach() - o).reshape(-1) for (_, p), o in zip(grouped["head"], old, strict=True)]
    )
    base = torch.cat([o.reshape(-1) for o in old])
    expected = (delta.double().norm() / base.double().norm()).item()
    assert ratio["head"] == pytest.approx(expected, rel=1e-12)
    assert ratio[OTHER] == 0.0
