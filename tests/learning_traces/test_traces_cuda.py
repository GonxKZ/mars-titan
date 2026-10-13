"""Paridad bit a bit de las trazas en cuda:0 (#448). La marca `cuda` la omite sin GPU.

Repite en el dispositivo lo que las pruebas de cada familia comprueban en CPU con sus
entrenadores: con trazas en cada paso, la salida, la pérdida, los gradientes y el estado
de los generadores aleatorios son los mismos bits que sin trazas. Se recorren las cinco
referencias neuronales (también la base de los adaptadores del postentrenamiento) y la
memoria neuronal de Titans-MAC con su observador, que es la que se engancha al forward,
sola y dentro del bloque MAC completo (atención con tokens persistentes, lectura y escritura
de la memoria en varios segmentos). El doble `Steps` cuenta los pasos y nunca cambia pesos.
TF32 queda desactivado, como en la campaña.
"""

import pytest
import torch

from mars_titan.learning_traces.hooks import TrainingTraces, observing, open_traces, traced_update
from mars_titan.learning_traces.memory import MemoryObserver
from mars_titan.learning_traces.recorder import TraceConfig, TraceRecorder, read_traces
from mars_titan.models.quantile_head import pinball_loss
from mars_titan.models.titans import MACConfig, MemoryConfig, TitansMAC
from tests.learning_traces.test_hooks import IDENTITY, Steps, memory_module
from tests.models.test_adapter_forms import FAMILIES, batch, parent

pytestmark = pytest.mark.cuda
DEVICE = torch.device("cuda:0")


@pytest.fixture(autouse=True)
def strict_fp32(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def same_bits(left, right):
    return left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.int32), right.contiguous().view(torch.int32)
    )


def reference_steps(model, traces, steps=3):
    """Pasos de `reference_run` en el dispositivo: pinball, backward, norma y paso nulo."""
    inputs, presence = batch(size=6)
    inputs = {name: value.to(DEVICE) for name, value in inputs.items()}
    target = torch.linspace(-1, 1, 6, device=DEVICE)
    optimizer = Steps(model.parameters())
    outputs = []
    for step in range(1, steps + 1):
        model.zero_grad(set_to_none=True)
        emitted = model(inputs, presence.to(DEVICE))
        loss = pinball_loss(emitted, target)
        loss.backward()
        with traced_update(traces, step, optimizer, loss) as probe:
            probe.clip(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0), 1.0)
            optimizer.step()
        outputs.append(
            (emitted.detach(), loss.detach(), [p.grad.clone() for p in model.parameters()])
        )
    return outputs


@pytest.mark.parametrize("kind", FAMILIES)
def test_reference_models_keep_their_bits_with_traces_on_cuda(kind, tmp_path):
    plain = parent(kind).train().requires_grad_(True).to(DEVICE)
    traced = parent(kind).train().requires_grad_(True).to(DEVICE)
    traces = open_traces(
        tmp_path / "traces", dict(every=1, max_bytes=1 << 20), IDENTITY, dict(model=traced)
    )
    torch.cuda.manual_seed(5)
    expected = reference_steps(plain, None)
    rng = torch.cuda.get_rng_state(DEVICE)
    torch.cuda.manual_seed(5)
    found = reference_steps(traced, traces)
    assert torch.equal(rng, torch.cuda.get_rng_state(DEVICE))
    for (a_out, a_loss, a_grads), (b_out, b_loss, b_grads) in zip(expected, found, strict=True):
        assert same_bits(a_out, b_out) and same_bits(a_loss, b_loss)
        assert all(same_bits(a, b) for a, b in zip(a_grads, b_grads, strict=True))
    traces.checkpoint()
    _, table = read_traces(tmp_path / "traces")
    assert set(table.column("step").to_pylist()) == {1, 2, 3}
    assert {"loss", "grad_l2", "weight_l2", "update_ratio"} <= set(
        table.column("metric").to_pylist()
    )


def memory_pass(module, traces):
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    tokens = 3.0 * torch.randn(3, 8, 8, generator=generator, device=DEVICE)
    query = torch.randn(3, 2, 8, generator=generator, device=DEVICE)
    with observing(traces, 1):
        state = module.update(
            tokens, module.initial_state(3, differentiable=True), differentiable=True
        )
    read = module.read(query, state)
    loss = read.square().sum() + sum(w.square().sum() for w in state.weights)
    return read, state.weights, torch.autograd.grad(loss, list(module.parameters()))


def test_the_titans_memory_observer_keeps_its_bits_on_cuda(tmp_path):
    plain = memory_pass(memory_module().to(DEVICE), None)
    module = memory_module().to(DEVICE)
    observer = MemoryObserver(dict(core=module))
    recorder = TraceRecorder(tmp_path / "traces", TraceConfig(1, 1 << 20), IDENTITY)
    traces = TrainingTraces(recorder, dict(core=module), observers=[observer])
    rng = torch.cuda.get_rng_state(DEVICE)
    traced = memory_pass(module, traces)
    assert torch.equal(rng, torch.cuda.get_rng_state(DEVICE)) and module.observer is None
    assert same_bits(plain[0], traced[0])
    for a, b in zip((*plain[1], *plain[2]), (*traced[1], *traced[2]), strict=True):
        assert same_bits(a, b)
    observer.record(recorder)
    recorder.flush(1, "train")
    _, table = read_traces(tmp_path / "traces")
    metrics = set(table.column("metric").to_pylist())
    assert {"memory.alpha.mean", "memory.associative_loss.mean", "memory.tokens"} <= metrics


def mac_steps(core, traces, steps=3):
    """Pasos de un núcleo MAC en línea: dos segmentos por paso, pérdida y paso nulo."""
    generator = torch.Generator(device=DEVICE).manual_seed(13)
    optimizer = Steps(core.parameters())
    outputs = []
    for step in range(1, steps + 1):
        core.zero_grad(set_to_none=True)
        # Cada paso empieza con la memoria inicial, como un tramo truncado del entrenador.
        state = core.initial_state(3, differentiable=True)
        loss = 0.0
        with observing(traces, step):
            for _ in range(2):
                segment = torch.randn(3, 4, 8, generator=generator, device=DEVICE)
                prediction, state = core(segment, state, differentiable=True)
                loss = loss + prediction.square().mean()
        loss.backward()
        with traced_update(traces, step, optimizer, loss) as probe:
            probe.clip(torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0), 1.0)
            optimizer.step()
        outputs.append(
            (prediction.detach(), loss.detach(), [p.grad.clone() for p in core.parameters()])
        )
    return outputs


def mac_core():
    torch.manual_seed(17)
    memory = MemoryConfig(dim=8, depth=2, max_batch=3, max_tokens=4)
    config = MACConfig(memory=memory, heads=2, persistent_tokens=2, max_segment=4)
    return TitansMAC(config, device=DEVICE)


def test_the_mac_block_keeps_its_bits_with_traces_on_cuda(tmp_path):
    expected = mac_steps(mac_core(), None)
    core = mac_core()
    traces = open_traces(
        tmp_path / "traces",
        dict(every=1, max_bytes=1 << 20),
        IDENTITY,
        dict(core=core),
        observers=[MemoryObserver(dict(core=core))],
    )
    rng = torch.cuda.get_rng_state(DEVICE)
    found = mac_steps(core, traces)
    assert torch.equal(rng, torch.cuda.get_rng_state(DEVICE))
    for (a_out, a_loss, a_grads), (b_out, b_loss, b_grads) in zip(expected, found, strict=True):
        assert same_bits(a_out, b_out) and same_bits(a_loss, b_loss)
        assert all(same_bits(a, b) for a, b in zip(a_grads, b_grads, strict=True))
    traces.checkpoint()
    _, table = read_traces(tmp_path / "traces")
    assert set(table.column("step").to_pylist()) == {1, 2, 3}
    assert {"memory.alpha.mean", "grad_l2", "update_ratio"} <= set(
        table.column("metric").to_pylist()
    )
