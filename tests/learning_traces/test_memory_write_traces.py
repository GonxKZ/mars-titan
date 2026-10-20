"""Escrituras compartidas y plasticidad observadas sin alterar pesos ni gradientes."""

import pytest
import torch

from mars_titan.learning_traces.hooks import TrainingTraces, observing
from mars_titan.learning_traces.memory import MemoryObserver
from mars_titan.learning_traces.recorder import TraceConfig, TraceRecorder, read_traces
from mars_titan.models.titans.config import MemoryConfig, MemoryMultiscale
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.models.titans.plasticity_config import MemoryPlasticity
from mars_titan.models.titans.session_context import MARKET_MEMORY, SessionContext

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


def memory(device="cpu", *, plasticity=None, depth=2, multiscale=None, dtype=torch.float64):
    config = MemoryConfig(
        4,
        depth=depth,
        max_batch=3,
        max_tokens=3,
        parameter_seed=42,
        plasticity=plasticity,
        multiscale=multiscale,
    )
    return NeuralMemory(config, device=device, dtype=dtype)


def traces_for(module, folder, *, every=1, max_bytes=1 << 20):
    recorder = TraceRecorder(folder, TraceConfig(every, max_bytes), {"family": "fixture"})
    observer = MemoryObserver({"core": module})
    return TrainingTraces(recorder, {"core": module}, observers=[observer]), observer


def metric_rows(folder):
    return {row["metric"]: row["value"] for row in read_traces(folder)[1].to_pylist()}


def shared_pass(module, traces=None, *, differentiable=True, step=1):
    tokens = torch.tensor(
        [[1.0, 2.0, -1.0, 0.5], [-0.5, 1.0, 3.0, -2.0], [2.0, -0.2, 0.7, 1.0]],
        dtype=module._reference().dtype,
        device=module._reference().device,
    )
    state = module.initial_state(1, differentiable=differentiable)
    with observing(traces, step):
        state = module.update_shared(tokens, state, differentiable=differentiable)
    output = module.read_shared(tokens, state)
    gradients = (
        torch.autograd.grad(output.square().sum(), tuple(module.parameters()))
        if differentiable
        else ()
    )
    return output, state, gradients


def flow_pass(module, traces=None, *, differentiable=True, step=1):
    tokens = (
        torch.arange(
            36, dtype=module._reference().dtype, device=module._reference().device
        ).reshape(3, 3, 4)
        / 10
    )
    state = module.initial_state(3, differentiable=differentiable)
    with observing(traces, step):
        state = module.update(tokens, state, differentiable=differentiable)
    output = module.read(tokens[:, :1], state)
    gradients = (
        torch.autograd.grad(output.square().sum(), tuple(module.parameters()))
        if differentiable
        else ()
    )
    return output, state, gradients


def flush(traces, observer, step=1):
    observer.record(traces.recorder)
    traces.recorder.flush(step, "technical")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("depth", [1, 2])
def test_shared_write_is_counted_once_and_its_loss_is_not_divided_twice(tmp_path, device, depth):
    module = memory(device, depth=depth)
    traces, observer = traces_for(module, tmp_path)
    events = []
    # La pérdida de referencia procede de la misma entrada, antes de toda escritura.
    module.observer = lambda **event: events.append(event)
    shared_pass(module)
    module.observer = None
    assert len(events) == 1
    assert events[0]["shared_rows"] == 3
    shared_pass(module, traces)
    flush(traces, observer)
    rows = metric_rows(tmp_path)
    assert rows["memory.shared.writes"] == 1
    assert rows["memory.shared.rows"] == 3
    assert rows["memory.shared.rows_per_write.mean"] == 3
    assert "memory.tokens" not in rows
    assert "memory.associative_loss.mean" not in rows
    assert rows["memory.shared.associative_loss.mean"] == events[0]["loss"].item()
    norm = torch.linalg.vector_norm(events[0]["weights"][0], dim=(1, 2)).item()
    assert rows["memory.shared.fast_weight_l2.0.mean"] == pytest.approx(norm, rel=1e-14)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("capture", ["none", "eligibility", "constant"])
def test_plasticity_reports_actual_row_gains_and_only_applied_capture(
    tmp_path, device, shared, capture
):
    plasticity = MemoryPlasticity(
        write="coherence",
        capture=capture,
        capture_constant=0.3,
        gradient_contract="scalar_controls_v2",
    )
    module = memory(device, plasticity=plasticity, multiscale=MemoryMultiscale())
    traces, observer = traces_for(module, tmp_path)
    if shared:
        _, state, _ = shared_pass(module, traces)
    else:
        tokens = torch.arange(24, dtype=torch.float64, device=device).reshape(2, 3, 4) / 10
        # Una observación permite contrastar la traza con el estado persistido de ese paso.
        with observing(traces, 1):
            state = module.update(tokens[:, :1], module.initial_state(2))
    flush(traces, observer)
    rows = metric_rows(tmp_path)
    prefix = "memory.shared" if shared else "memory"
    for layer in range(2):
        mean, energy, tag = state.modulation[3 * layer : 3 * layer + 3]
        coherence = (
            mean.square().sum(-1, keepdim=True)
            / ((1 - plasticity.beta) * energy + plasticity.epsilon)
        ).clamp(0, 1)
        expected = plasticity.write_floor + (1 - plasticity.write_floor) * coherence
        key = f"{prefix}.plasticity.write_gain.{layer}.mean"
        assert rows[key] == pytest.approx(expected.mean().item(), rel=1e-12)
        assert rows[f"{prefix}.plasticity.coherence.{layer}.mean"] == pytest.approx(
            coherence.mean().item(), rel=1e-12
        )
        assert rows[f"{prefix}.plasticity.eligibility.{layer}.mean"] == pytest.approx(
            tag.mean().item(), rel=1e-12
        )
        key = f"{prefix}.plasticity.capture_gain.{layer}.mean"
        # La cascada aplica una ganancia unitaria cuando capture='none'.
        if capture == "none":
            expected_capture = torch.ones_like(tag)
        elif capture == "constant":
            expected_capture = torch.full_like(tag, 0.3)
        else:
            expected_capture = 0.1 + 0.9 * tag * coherence.mean(dim=1, keepdim=True)
        assert rows[key] == pytest.approx(expected_capture.mean().item(), rel=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_no_cascade_does_not_report_an_unused_capture_gain(tmp_path, device):
    module = memory(device, plasticity=MemoryPlasticity(write="none"))
    traces, observer = traces_for(module, tmp_path)
    shared_pass(module, traces, differentiable=False)
    flush(traces, observer)
    rows = metric_rows(tmp_path)
    assert rows["memory.shared.plasticity.write_gain.0.mean"] == 1
    assert not any("capture_gain" in key for key in rows)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("differentiable", [False, True])
@pytest.mark.parametrize("plasticity", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_shared_tracing_preserves_outputs_full_state_gradients_and_rng(
    tmp_path, device, dtype, differentiable, plasticity, shared
):
    options = dict(
        dtype=dtype,
        plasticity=MemoryPlasticity(capture="eligibility") if plasticity else None,
        multiscale=MemoryMultiscale() if plasticity else None,
    )
    run = shared_pass if shared else flow_pass
    plain = run(memory(device, **options), differentiable=differentiable)
    module = memory(device, **options)
    traces, observer = traces_for(module, tmp_path)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device != "cpu" else None
    traced = run(module, traces, differentiable=differentiable)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    assert module.observer is None
    assert torch.equal(plain[0], traced[0])
    for before, after in zip(
        (
            *plain[1].weights,
            *plain[1].momentum,
            *plain[1].levels,
            *plain[1].modulation,
            plain[1].steps,
            *plain[2],
        ),
        (
            *traced[1].weights,
            *traced[1].momentum,
            *traced[1].levels,
            *traced[1].modulation,
            traced[1].steps,
            *traced[2],
        ),
        strict=True,
    ):
        assert torch.equal(before, after)
    # El observador no conserva tensores de entrada ni grafos de autograd.
    layer = observer.layers["core"]
    for running in layer.values.values():
        assert running.total.grad_fn is None and not running.total.requires_grad
    flush(traces, observer)
    assert observer.layers == {}


def test_shared_cadence_recovery_and_rejection_leave_no_duplicate_events(tmp_path):
    module = memory()
    traces, observer = traces_for(module, tmp_path, every=2)
    shared_pass(module, traces, step=1)
    assert observer.layers == {}
    shared_pass(module, traces, step=2)
    flush(traces, observer, 2)
    cursor = traces.checkpoint()
    shared_pass(module, traces, step=4)
    flush(traces, observer, 4)
    traces.restore(cursor)
    rows = read_traces(tmp_path)[1].to_pylist()
    assert {row["step"] for row in rows} == {2}
    assert sum(row["value"] for row in rows if row["metric"] == "memory.shared.writes") == 1
    with pytest.raises(ValueError), observing(traces, 6):
        module.update_shared(torch.empty(0, 4, dtype=torch.float64), module.initial_state(1))
    assert observer.layers["core"].shared_writes == 0
    assert observer.layers["core"].tokens == 0
    assert module.observer is None


def test_shared_and_per_flow_statistics_keep_distinct_denominators(tmp_path):
    module = memory(depth=1)
    traces, observer = traces_for(module, tmp_path)
    shared_pass(module, traces, differentiable=False)
    tokens = torch.ones(2, 3, 4, dtype=torch.float64)
    with observing(traces, 1):
        module.update(tokens, module.initial_state(2))
    flush(traces, observer)
    rows = metric_rows(tmp_path)
    assert rows["memory.tokens"] == 3
    assert rows["memory.shared.writes"] == 1
    assert rows["memory.shared.rows"] == 3
    assert rows["memory.associative_loss.mean"] != rows["memory.shared.associative_loss.mean"]


@pytest.mark.parametrize("device", DEVICES)
def test_session_context_records_one_shared_write_after_its_blocks_are_committed(tmp_path, device):
    context = SessionContext(
        MARKET_MEMORY,
        4,
        seed=42,
        memory_options=dict(plasticity=MemoryPlasticity()),
    ).to(device=device, dtype=torch.float64)
    traces, observer = traces_for(context, tmp_path)
    tokens = torch.arange(12, dtype=torch.float64, device=device).reshape(3, 4) / 10
    flows = ("US/A", "US/B", "US/C")
    state = context.initial_state("fixture")
    with observing(traces, 1):
        _, state = context.read(tokens[:2], flows[:2], (10, 10), state)
        _, state = context.read(tokens[2:], flows[2:], (10,), state)
        assert observer.layers["core.memory"].shared_writes == 0
        state = context.commit(state)
        # Confirmar de nuevo, sin filas pendientes, no crea otra escritura.
        context.commit(state)
    flush(traces, observer)
    rows = read_traces(tmp_path)[1].to_pylist()
    recorded = {row["metric"]: row["value"] for row in rows}
    assert recorded["memory.shared.writes"] == 1
    assert recorded["memory.shared.rows"] == 3
    assert recorded["memory.shared.plasticity.write_gain.0.mean"] >= 0.1
    assert {row["group"] for row in rows} == {"core.memory"}


def test_shared_trace_obeys_the_existing_byte_limit(tmp_path):
    module = memory(plasticity=MemoryPlasticity())
    traces, observer = traces_for(module, tmp_path, max_bytes=1024)
    shared_pass(module, traces)
    flush(traces, observer)
    assert not traces.recorder.active
    assert traces.recorder.exhausted_at == 1
    assert traces.recorder.sequence == 0
    assert read_traces(tmp_path)[1].num_rows == 0
    assert not tuple((tmp_path / "parts").iterdir())
