"""Ganchos comunes de trazas, observador de la memoria de Titans y lectura validada (#448).

Los modelos son pequeños y se construyen en la prueba. Hay pasadas hacia delante y hacia
atrás, pero ningún optimizador cambia pesos: el doble `Steps` solo cuenta las llamadas.
"""

import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.learning_traces import hooks
from mars_titan.learning_traces.hooks import (
    EMPTY_CURSOR,
    SKIP,
    TrainingTraces,
    observing,
    open_recorder,
    open_traces,
    trace_identity,
    traced_update,
)
from mars_titan.learning_traces.memory import SATURATION, MemoryObserver
from mars_titan.learning_traces.parameters import parameter_groups
from mars_titan.learning_traces.recorder import (
    SCHEMA,
    TraceConfig,
    TraceRecorder,
    read_traces,
    trace_name,
)
from mars_titan.models.titans import GateBias, MemoryConfig, NeuralMemory

IDENTITY = dict(family="fixture", run_identity_sha256="0" * 64)


def recorder(tmp_path, every=1, max_bytes=1 << 20):
    return TraceRecorder(tmp_path / "traces", TraceConfig(every, max_bytes), IDENTITY)


def rows(tmp_path):
    return read_traces(tmp_path / "traces")[1].to_pylist()


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = torch.nn.Linear(6, 8)
        self.dropout = torch.nn.Dropout(0.25)
        self.head = torch.nn.Linear(8, 1)

    def forward(self, x):
        return self.head(self.dropout(torch.tanh(self.memory(x))))


class Steps:
    """Optimizador de la prueba: cuenta los pasos y nunca cambia pesos."""

    def __init__(self, parameters):
        self.param_groups = [dict(params=list(parameters), lr=0.25, role="core")]
        self.calls = 0

    def step(self):
        self.calls += 1


def test_one_flush_keeps_the_step_and_phase_of_each_statistic(tmp_path):
    traces = recorder(tmp_path)
    with traces.at(3, "train"):
        traces.scalar("loss", 1.0)
        with traces.at(4, "validation"):
            traces.tensor("score", torch.tensor([1.0, 3.0]))
        traces.scalar("lr", torch.tensor(0.5))
    assert traces.pending == 3
    assert traces.flush() == 2 + 7
    found = {(r["metric"], r["stat"]): (r["step"], r["phase"]) for r in rows(tmp_path)}
    assert found[("loss", "value")] == (3, "train")
    assert found[("score", "mean")] == (4, "validation")
    assert found[("lr", "value")] == (3, "train")
    # Fuera de un bloque `at` el volcado necesita su paso y su fase.
    traces.scalar("loss", 2.0)
    with pytest.raises(ValueError, match="sin paso"):
        traces.flush()
    with pytest.raises(ValueError, match="paso"), traces.at(-1, "train"):
        pass


def test_the_first_step_that_does_not_fit_is_declared(tmp_path):
    traces = recorder(tmp_path, max_bytes=4096)
    step = 0
    while traces.active:
        step += 1
        for index in range(20):
            with traces.at(step, "train"):
                traces.tensor(f"layer_{index}", torch.randn(16))
        traces.flush()
        assert step < 100
    manifest, table = read_traces(tmp_path / "traces")
    assert manifest["exhausted_at_step"] == step
    assert max(table.column("step").to_pylist()) == step - 1


def write_part(folder, table):
    (folder / "parts").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, folder / "parts" / "part-00000001.parquet")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("phase", "Train!", "nombre inválido"),
        ("metric", "1loss", "nombre inválido"),
        ("stat", "median", "otra estadística"),
        ("step", -2, "paso negativo"),
    ],
)
def test_reading_rejects_rows_outside_the_contract(tmp_path, column, value, message):
    traces = recorder(tmp_path)
    traces.scalar("loss", 1.0)
    traces.flush(0, "train")
    row = dict(step=0, phase="train", metric="loss", group="all", stat="value", value=1.0)
    row[column] = value
    write_part(tmp_path / "traces", pa.Table.from_pylist([row], schema=SCHEMA))
    with pytest.raises(ValueError, match=message):
        read_traces(tmp_path / "traces")


def test_reading_rejects_parts_beyond_the_declared_budget(tmp_path):
    traces = recorder(tmp_path)
    traces.scalar("loss", 1.0)
    traces.flush(0, "train")
    path = tmp_path / "traces" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["max_bytes"] = 1024
    path.write_text(json.dumps(manifest))
    assert traces.bytes > 1024
    with pytest.raises(ValueError, match="presupuesto"):
        read_traces(tmp_path / "traces")


def test_names_are_normalised_to_the_contract():
    assert trace_name("Readout.Bank Gate") == "readout.bank_gate"
    assert trace_name("0.weight") == "weight"
    assert trace_name("") == "unnamed" and len(trace_name("a" * 200)) == 96


def step_once(traces, model, optimizer, step, *, threshold=0.1):
    torch.manual_seed(7)
    x = torch.randn(16, 6)
    loss = model(x).square().mean()
    loss.backward()
    with traced_update(traces, step, optimizer, loss) as probe:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), threshold)
        probe.clip(norm, threshold)
        optimizer.step()
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    return loss.detach(), gradients, torch.random.get_rng_state()


def test_a_traced_step_matches_an_untraced_step_bit_for_bit(tmp_path):
    torch.manual_seed(3)
    plain = Tiny()
    traced = Tiny()
    traced.load_state_dict(plain.state_dict())
    traces = open_traces(
        tmp_path / "traces", dict(every=1, max_bytes=1 << 20), IDENTITY, dict(model=traced)
    )
    first = step_once(None, plain, Steps(plain.parameters()), 1)
    second = step_once(traces, traced, Steps(traced.parameters()), 1)
    assert torch.equal(first[0], second[0]) and torch.equal(first[2], second[2])
    assert all(torch.equal(a, b) for a, b in zip(first[1], second[1], strict=True))
    traces.checkpoint()
    found = {(r["metric"], r["group"]): r["value"] for r in rows(tmp_path)}
    assert found[("loss", "all")] == first[0].item()
    assert {("grad_l2", "model.memory"), ("grad_l2", "model.head")} <= set(found)
    assert found[("update_ratio", "model.memory")] == 0.0
    assert found[("clip_threshold", "all")] == 0.1 and found[("learning_rate", "core")] == 0.25
    # La norma total del recorte es la de los gradientes antes de recortar, por grupo.
    groups = found[("grad_l2", "model.memory")] ** 2 + found[("grad_l2", "model.head")] ** 2
    assert found[("grad_l2_total", "all")] == pytest.approx(groups**0.5, rel=1e-6)
    # El umbral es menor que la norma, así que el paso recorta en los dos recorridos.
    assert found[("grad_l2_total", "all")] > 0.1


def test_the_native_candidate_is_grouped_and_has_no_neural_memory():
    # La GRU candidata nativa devuelve un diccionario y no es un módulo de PyTorch.
    weight, frozen = torch.zeros(2, requires_grad=True), torch.zeros(2)
    native = SimpleNamespace(named_parameters=lambda: dict(update_weight=weight, head_bias=frozen))
    grouped = parameter_groups(dict(model=native))
    assert list(grouped) == ["model.update_weight"]
    assert grouped["model.update_weight"][0][1] is weight
    assert not MemoryObserver(dict(model=native))


def test_an_optimizer_without_a_group_rate_records_no_learning_rate(tmp_path):
    model = Tiny()
    traces = open_traces(
        tmp_path / "traces", dict(every=1, max_bytes=1 << 20), IDENTITY, dict(model=model)
    )
    optimizer = Steps(model.parameters())
    del optimizer.param_groups[0]["lr"]
    step_once(traces, model, optimizer, 1)
    traces.checkpoint()
    metrics = {r["metric"] for r in rows(tmp_path)}
    assert "loss" in metrics and "learning_rate" not in metrics and optimizer.calls == 1


def test_untraced_steps_and_disabled_traces_do_nothing(tmp_path):
    model = Tiny()
    traces = open_traces(
        tmp_path / "traces", dict(every=2, max_bytes=1 << 20), IDENTITY, dict(model=model)
    )
    with traced_update(traces, 1, Steps(model.parameters())) as probe:
        assert probe is SKIP
    with traced_update(None, 2, Steps(model.parameters())) as probe:
        assert probe is SKIP
    assert traces.recorder.pending == 0
    disabled = dict(every=1, max_bytes=1 << 20, enabled=False)
    assert open_traces(tmp_path / "off", disabled, IDENTITY, dict(model=model)) is None
    assert open_recorder(tmp_path / "off", None, IDENTITY) is None
    assert not (tmp_path / "off").exists()


def test_validation_records_the_event_and_flushes(tmp_path):
    model = Tiny()
    traces = open_traces(
        tmp_path / "traces", dict(every=5, max_bytes=1 << 20), IDENTITY, dict(model=model)
    )
    traces.validation(dict(kind="validation", epoch=2, global_step=7, score=0.125))
    assert traces.recorder.pending == 0 and traces.recorder.sequence == 1
    found = {(r["metric"], r["group"]): (r["step"], r["phase"], r["value"]) for r in rows(tmp_path)}
    assert found[("validation.score", "all")] == (7, "validation", 0.125)
    assert found[("validation.epoch", "all")] == (7, "validation", 2.0)
    assert ("weight_l2", "model.head") in found
    assert not any(metric == "validation.kind" for metric, _ in found)


def test_resuming_from_a_saved_cursor_discards_later_parts(tmp_path):
    model = Tiny()
    config = dict(every=1, max_bytes=1 << 20)
    traces = open_traces(tmp_path / "traces", config, IDENTITY, dict(model=model))
    traces.restore(None)
    traces.validation(dict(global_step=1, score=1.0))
    saved = json.loads(json.dumps(traces.checkpoint()))
    traces.validation(dict(global_step=2, score=2.0))
    again = open_traces(tmp_path / "traces", config, IDENTITY, dict(model=model))
    again.restore(saved)
    again.validation(dict(global_step=2, score=3.0))
    scores = [(r["step"], r["value"]) for r in rows(tmp_path) if r["metric"] == "validation.score"]
    assert scores == [(1, 1.0), (2, 3.0)]
    fresh = open_traces(tmp_path / "traces", config, IDENTITY, dict(model=model))
    fresh.restore(None)
    assert fresh.recorder.state_dict() == EMPTY_CURSOR
    assert not list((tmp_path / "traces" / "parts").glob("part-*"))


def test_pending_statistics_are_flushed_before_they_grow_without_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(hooks, "MAX_PENDING", 10)
    model = Tiny()
    traces = open_traces(
        tmp_path / "traces", dict(every=1, max_bytes=1 << 20), IDENTITY, dict(model=model)
    )
    step_once(traces, model, Steps(model.parameters()), 1)
    assert traces.recorder.pending == 0 and traces.recorder.sequence == 1


def test_identity_fingerprints_the_run_and_keeps_extra_fields():
    first = trace_identity("titans_mac", dict(seed=42, window="fold-000"))
    second = trace_identity("titans_mac", dict(window="fold-000", seed=42), rule="sgd")
    assert first["run_identity_sha256"] == second["run_identity_sha256"]
    assert second["family"] == "titans_mac" and second["rule"] == "sgd"
    with pytest.raises(ValueError):
        TrainingTraces(object(), dict(model=Tiny()))


def memory_module():
    config = MemoryConfig(
        dim=8,
        depth=2,
        max_batch=3,
        max_tokens=8,
        gate_bias=GateBias(alpha_half_life=4.0, eta=0.5, theta=0.05),
        residual_layer_norm=True,
    )
    return NeuralMemory(config, dtype=torch.float32)


def memory_pass(module, traces=None, step=1):
    generator = torch.Generator().manual_seed(11)
    tokens = 3.0 * torch.randn(3, 8, 8, generator=generator)
    query = torch.randn(3, 2, 8, generator=generator)
    with observing(traces, step):
        state = module.update(
            tokens, module.initial_state(3, differentiable=True), differentiable=True
        )
    read = module.read(query, state)
    loss = read.square().sum() + sum(w.square().sum() for w in state.weights)
    gradients = torch.autograd.grad(loss, list(module.parameters()))
    return read, state, gradients


class Capture:
    def __init__(self):
        self.calls = []

    def __call__(self, **values):
        self.calls.append({key: value for key, value in values.items()})


def test_memory_observer_reads_the_gates_without_changing_the_memory(tmp_path):
    plain = memory_pass(memory_module())
    module = memory_module()
    observer = MemoryObserver(dict(core=module))
    assert observer and observer.memories[0][0] == "core"
    traces = TrainingTraces(recorder(tmp_path), dict(core=module), observers=[observer])
    before = torch.random.get_rng_state()
    traced = memory_pass(module, traces)
    assert torch.equal(before, torch.random.get_rng_state())
    assert module.observer is None
    assert torch.equal(plain[0], traced[0])
    for a, b in zip((*plain[1].weights, *plain[2]), (*traced[1].weights, *traced[2]), strict=True):
        assert torch.equal(a, b)
    capture = Capture()
    module.observer = capture
    memory_pass(module)
    module.observer = None
    observer.record(traces.recorder)
    traces.recorder.flush(1, "train")
    found = {(r["metric"], r["group"]): r["value"] for r in rows(tmp_path)}
    alphas = torch.stack([call["alpha"].double().reshape(-1) for call in capture.calls])
    assert found[("memory.tokens", "core")] == 8.0
    assert found[("memory.alpha.mean", "core")] == pytest.approx(alphas.mean().item(), rel=1e-12)
    assert found[("memory.alpha.max", "core")] == alphas.max().item()
    low = (alphas < SATURATION).double().mean().item()
    assert found[("memory.alpha.low_fraction", "core")] == pytest.approx(low, abs=1e-15)
    loss = torch.stack([call["loss"].double() / 3 for call in capture.calls])
    assert found[("memory.associative_loss.mean", "core")] == pytest.approx(
        loss.mean().item(), rel=1e-12
    )
    norms = torch.cat(
        [torch.linalg.vector_norm(c["weights"][1].double(), dim=(1, 2)) for c in capture.calls]
    )
    assert found[("memory.fast_weight_l2.1.mean", "core")] == pytest.approx(
        norms.mean().item(), rel=1e-12
    )
    assert ("memory.theta.low_fraction", "core") not in found
    # Tras registrar, el observador empieza de cero.
    assert observer.layers == {}


def test_memory_observer_only_listens_on_traced_steps(tmp_path):
    module = memory_module()
    observer = MemoryObserver(dict(core=module))
    traces = TrainingTraces(recorder(tmp_path, every=2), dict(core=module), observers=[observer])
    memory_pass(module, traces, step=1)
    assert observer.layers == {}
    memory_pass(module, traces, step=2)
    assert observer.layers["core"].tokens == 8
    assert not MemoryObserver(dict(model=Tiny()))


def test_a_checkpoint_discards_what_the_observers_accumulated(tmp_path):
    # Lo acumulado no viaja con el estado guardado. Si se conservara, el proceso que sigue
    # registraría más tokens que uno reanudado desde ese mismo punto de control.
    module = memory_module()
    observer = MemoryObserver(dict(core=module))
    traces = TrainingTraces(recorder(tmp_path), dict(core=module), observers=[observer])
    memory_pass(module, traces, step=1)
    assert observer.layers["core"].tokens == 8
    traces.checkpoint()
    assert observer.layers == {}
    memory_pass(module, traces, step=2)
    assert observer.layers["core"].tokens == 8
