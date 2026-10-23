"""Ecuaciones y contrato del candidato temporal experimental, sin optimizador."""

import importlib
import importlib.util
import math
from copy import deepcopy

import pytest


def api():
    assert importlib.util.find_spec("mars_titan.models.kronlstm") is not None, (
        "Falta el componente experimental KronLSTM"
    )
    return importlib.import_module("mars_titan.models.kronlstm")


def tensors(device, dtype):
    import torch

    dtype = getattr(torch, dtype)
    q = torch.linspace(-0.7, 0.8, 24, dtype=dtype, device=device).reshape(2, 3, 2, 2)
    k = torch.nn.functional.normalize(q.flip(-1) + 1.1, dim=-1)
    v = torch.linspace(0.3, 1.2, 36, dtype=dtype, device=device).reshape(2, 3, 2, 3)
    g = -torch.linspace(0.01, 0.4, 24, dtype=dtype, device=device).reshape(2, 3, 2, 2)
    beta = torch.linspace(0.1, 0.8, 12, dtype=dtype, device=device).reshape(2, 3, 2)
    target = torch.linspace(0.2, 0.7, 12, dtype=dtype, device=device).reshape(2, 3, 2)
    state = torch.linspace(-0.2, 0.3, 24, dtype=dtype, device=device).reshape(2, 2, 2, 3)
    return q, k, v, g, beta, target, state


def kronecker_reference(q, k, v, g, beta, target, initial, *, decoupled, scale):
    """Construir la transición densa diminuta, independiente del scan factorizado."""
    import torch

    batches, steps, heads, keys = q.shape
    values = v.shape[-1]
    rows, final = [], []
    identity = torch.eye(keys * values, dtype=q.dtype, device=q.device)
    for batch in range(batches):
        by_head, last = [], []
        for head in range(heads):
            state = initial[batch, head].flatten()
            outputs = []
            for step in range(steps):
                key, value = k[batch, step, head], v[batch, step, head]
                if g is not None:
                    decay = g[batch, step, head].exp()
                    if g.ndim == 3:
                        decay = decay.expand(keys)
                    state = torch.diag(decay.repeat_interleave(values)) @ state
                pair = torch.kron(key, value)
                norm = value.square().sum()
                denominator = norm.clamp_min(1e-18)
                erase = beta[batch, step, head] / denominator
                write = (1 if decoupled else beta[batch, step, head]) / denominator
                goal = 1 if target is None else target[batch, step, head]
                state = (identity - erase * torch.outer(pair, pair)) @ state
                state = state + write * norm * goal * pair
                outputs.append((q[batch, step, head] * scale) @ state.reshape(keys, values))
            by_head.append(torch.stack(outputs))
            last.append(state.reshape(keys, values))
        rows.append(torch.stack(by_head, dim=1))
        final.append(torch.stack(last))
    return torch.stack(rows), torch.stack(final)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("decoupled,expected", [(True, 0.66), (False, 0.26)])
def test_scalar_rule_decays_then_writes_then_reads(device, dtype, decoupled, expected):
    import torch

    module = api()
    options = dict(dtype=getattr(torch, dtype), device=device)
    unit = torch.ones(1, 1, 1, 1, **options)
    gate = torch.full((1, 1, 1), 0.2, **options)
    output, state = module.kronlstm_scan(
        unit,
        unit,
        unit,
        torch.full_like(unit, math.log(0.8)),
        gate,
        target_gate=torch.full_like(gate, 0.5),
        initial_state=torch.full_like(unit, 0.25),
        decouple_erase=decoupled,
    )
    torch.testing.assert_close(output, torch.full_like(unit, expected))
    torch.testing.assert_close(state, torch.full_like(unit, expected))
    assert state.dtype == output.dtype == unit.dtype


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("decoupled", [True, False])
@pytest.mark.parametrize("decay", ["vector", "scalar", "none"])
def test_scan_matches_explicit_kronecker_transition(device, dtype, decoupled, decay):
    import torch

    module = api()
    q, k, v, g, beta, target, initial = tensors(device, dtype)
    if decay == "scalar":
        g = g[..., 0]
    elif decay == "none":
        g = None
    expected = kronecker_reference(
        q, k, v, g, beta, target, initial, decoupled=decoupled, scale=2**-0.5
    )
    actual = module.kronlstm_scan(
        q, k, v, g, beta, target_gate=target, initial_state=initial, decouple_erase=decoupled
    )
    tolerance = 1e-12 if dtype == "float64" else 2e-6
    for observed, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, wanted, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("value,expected", [(0.0, 0.0), (1e-12, 5e-19), (2.0, 1.0)])
def test_norm_floor_only_clamps_the_division_and_write_ignores_zero_beta(value, expected):
    import torch

    unit = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    gate = torch.zeros(1, 1, 1, dtype=torch.float64)
    output, state = api().kronlstm_scan(
        unit, unit, unit * value, None, gate, target_gate=gate + 0.5
    )
    torch.testing.assert_close(output, torch.full_like(unit, expected), rtol=1e-12, atol=0)
    torch.testing.assert_close(state, torch.full_like(unit, expected), rtol=1e-12, atol=0)


def test_missing_target_is_only_allowed_for_coupled_unit_target():
    import torch

    unit = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    beta = torch.full((1, 1, 1), 0.25, dtype=torch.float64)
    with pytest.raises(ValueError, match="target_gate|objetivo"):
        api().kronlstm_scan(unit, unit, unit, None, beta, target_gate=None)
    output, state = api().kronlstm_scan(
        unit, unit, unit, None, beta, target_gate=None, decouple_erase=False
    )
    torch.testing.assert_close(output, unit * 0.25, rtol=0, atol=0)
    torch.testing.assert_close(state, unit * 0.25, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_scan_gradients_match_the_dense_transition_for_every_argument(device, dtype):
    import torch

    values = tuple(value.detach().requires_grad_() for value in tensors(device, dtype))
    q, k, v, g, beta, target, initial = values
    actual = api().kronlstm_scan(q, k, v, g, beta, target_gate=target, initial_state=initial)
    expected = kronecker_reference(q, k, v, g, beta, target, initial, decoupled=True, scale=2**-0.5)
    observed_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in actual), values, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in expected), values
    )
    tolerance = 1e-10 if dtype == "float64" else 2e-5
    for observed, wanted in zip(observed_gradients, expected_gradients, strict=True):
        assert torch.isfinite(observed).all()
        assert torch.count_nonzero(observed) > 0
        torch.testing.assert_close(observed, wanted, rtol=tolerance, atol=tolerance)


def test_vector_control_erases_the_vector_residual_with_the_same_write():
    import torch

    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float64)
    state = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float64)
    beta = torch.full((1, 1, 1), 0.5, dtype=torch.float64)
    scalar, _ = api().kronlstm_scan(
        q, q, v, None, beta, target_gate=beta / 2, initial_state=state, scale=1.0
    )
    vector, _ = api().kronlstm_scan(
        q, q, v, None, beta, target_gate=beta / 2, initial_state=state, scale=1.0, rule="vector"
    )
    torch.testing.assert_close(scalar, torch.tensor([[[[1.0, 1.25]]]]).double())
    torch.testing.assert_close(vector, torch.tensor([[[[0.5, 1.25]]]]).double())
    torch.testing.assert_close(
        state, torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]).double(), rtol=0, atol=0
    )


def test_orthogonal_pair_preserves_identity_only_for_the_scalar_rule():
    import torch

    key = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    value = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float64)
    initial = torch.eye(2, dtype=torch.float64)[None, None]
    beta = torch.ones(1, 1, 1, dtype=torch.float64)
    _, scalar = api().kronlstm_scan(
        key, key, value, None, beta, target_gate=beta * 0, initial_state=initial, rule="scalar"
    )
    _, vector = api().kronlstm_scan(
        key, key, value, None, beta, target_gate=beta * 0, initial_state=initial, rule="vector"
    )
    torch.testing.assert_close(scalar, initial, rtol=0, atol=0)
    torch.testing.assert_close(
        vector, torch.diag(torch.tensor([0.0, 1.0])).double()[None, None], rtol=0, atol=0
    )


@pytest.mark.parametrize("field", ["q", "k", "v", "g", "beta", "target", "state"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_scan_rejects_nonfinite_values_without_mutating_its_inputs(field, invalid):
    import torch

    names = ("q", "k", "v", "g", "beta", "target", "state")
    values = dict(zip(names, tensors("cpu", "float64"), strict=True))
    values[field].view(-1)[0] = invalid
    before = {name: value.clone() for name, value in values.items()}
    with pytest.raises(ValueError, match="finito|NaN"):
        api().kronlstm_scan(
            values["q"],
            values["k"],
            values["v"],
            values["g"],
            values["beta"],
            target_gate=values["target"],
            initial_state=values["state"],
        )
    for name in names:
        torch.testing.assert_close(values[name], before[name], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("fault", ["g_shape", "gate_shape", "state_shape", "dtype", "rule"])
def test_scan_rejects_invalid_contracts_before_broadcasting(fault):
    q, k, v, g, beta, target, state = tensors("cpu", "float64")
    if fault == "g_shape":
        g = g[..., :1]
    elif fault == "gate_shape":
        beta = beta.unsqueeze(-1)
    elif fault == "state_shape":
        state = state[:, :, :1]
    elif fault == "dtype":
        k = k.float()
    with pytest.raises(ValueError):
        api().kronlstm_scan(
            q,
            k,
            v,
            g,
            beta,
            target_gate=target,
            initial_state=state,
            rule="unknown" if fault == "rule" else "scalar",
        )


def encoder(*, device="cpu", dtype="float64", rule="scalar", init_seed=42, **changes):
    import torch

    settings = dict(input_size=3, context=8, hidden_size=32, layers=2, heads=4, max_batch=4)
    settings.update(changes)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(init_seed)
        return (
            api()
            .CompactPriceKronLSTM(**settings, rule=rule)
            .to(device=device, dtype=getattr(torch, dtype))
        )


def prices(device, dtype, *, batch=2):
    import torch

    return torch.linspace(
        -0.7, 1.1, batch * 8 * 3, device=device, dtype=getattr(torch, dtype)
    ).reshape(batch, 8, 3)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("rule", ["scalar", "vector"])
def test_complete_encoder_is_causal_and_reset_per_window(device, dtype, rule):
    import torch

    model = encoder(device=device, dtype=dtype, rule=rule).eval()
    inputs = prices(device, dtype)
    altered = inputs.clone()
    altered[:, 4:] += 7
    before = deepcopy(model.state_dict())
    cpu_rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(0).clone() if device == "cuda:0" else None
    sequence = model.encode_sequence(inputs)
    changed = model.encode_sequence(altered)
    representation, checks = model.encode_last(inputs)
    assert sequence.shape == (2, 8, 32)
    assert isinstance(checks, list) and checks
    assert all(torch.isfinite(value).all() for value, _ in checks)
    torch.testing.assert_close(sequence[:, :4], changed[:, :4], rtol=0, atol=0)
    assert not torch.allclose(sequence[:, -1], changed[:, -1])
    torch.testing.assert_close(representation, sequence[:, -1], rtol=0, atol=0)
    torch.testing.assert_close(model(inputs), representation, rtol=0, atol=0)
    model(altered)
    torch.testing.assert_close(model(inputs), representation, rtol=0, atol=0)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if gpu_rng is not None:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state(0))
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        else:
            assert value == before[name]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_complete_encoder_propagates_gradients_through_all_mechanisms(device, dtype):
    import torch

    model = encoder(device=device, dtype=dtype)
    inputs = prices(device, dtype).requires_grad_()
    output = model.encode_sequence(inputs)
    gradients = torch.autograd.grad(output[:, -1].square().sum(), (inputs, *model.parameters()))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert all(torch.count_nonzero(value) > 0 for value in gradients)
    prefix_gradient = torch.autograd.grad(model.encode_sequence(inputs)[:, :3, 0].sum(), inputs)[0]
    assert torch.count_nonzero(prefix_gradient[:, 3:]) == 0
    assert torch.count_nonzero(prefix_gradient[:, :3]) > 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cache_chunks_steps_and_reload_preserve_cursor_and_results(device, dtype):
    import torch

    model = encoder(device=device, dtype=dtype).eval()
    inputs = prices(device, dtype)
    with torch.no_grad():
        expected = model.encode_sequence(inputs)
        first, state = model.encode_steps(inputs[:, :3])
        assert state["cursor"] == 3
        snapshot = deepcopy(state)
        second, finished = model.encode_steps(inputs[:, 3:], state)
        assert state["cursor"] == 3 and finished["cursor"] == 8
        torch.testing.assert_close(torch.cat((first, second), 1), expected, rtol=1e-5, atol=1e-6)
        payload = model.export_state(state)
        restored = encoder(device=device, dtype=dtype).eval()
        restored.load_state_dict(deepcopy(model.state_dict()))
        rebound = restored.restore_state(payload)
        assert rebound["cursor"] == 3
        next_token, next_state = restored.encode_step(inputs[:, 3], rebound)
        torch.testing.assert_close(next_token, expected[:, 3], rtol=1e-5, atol=1e-6)
        assert next_state["cursor"] == 4 and rebound["cursor"] == 3
        payload["layers"][0]["recurrent_state"].zero_()
        for original, previous in zip(state["layers"], snapshot["layers"], strict=True):
            for name in ("recurrent_state", "conv_state"):
                torch.testing.assert_close(original[name], previous[name], rtol=0, atol=0)


@pytest.mark.parametrize("seed", tuple(range(8)))
def test_seeded_fp32_cpu_chunking_preserves_whole_window(seed):
    import torch

    before = torch.random.get_rng_state().clone()
    model = encoder(device="cpu", dtype="float32", init_seed=seed).eval()
    inputs = prices("cpu", "float32")
    with torch.no_grad():
        expected = model.encode_sequence(inputs)
        first, state = model.encode_steps(inputs[:, :3])
        second, finished = model.encode_steps(inputs[:, 3:], state)
    assert state["cursor"] == 3 and finished["cursor"] == 8
    torch.testing.assert_close(torch.cat((first, second), dim=1), expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.cuda
@pytest.mark.parametrize("seed", tuple(range(8)))
def test_seeded_fp32_cuda_chunking_preserves_whole_window(seed):
    import torch

    model = encoder(device="cuda:0", dtype="float32", init_seed=seed).eval()
    inputs = prices("cuda:0", "float32")
    with torch.no_grad():
        expected = model.encode_sequence(inputs)
        first, state = model.encode_steps(inputs[:, :3])
        second, finished = model.encode_steps(inputs[:, 3:], state)
    assert state["cursor"] == 3 and finished["cursor"] == 8
    torch.testing.assert_close(torch.cat((first, second), dim=1), expected, rtol=1e-5, atol=1e-6)


def test_cpu_qkv_execution_identity_declares_the_accumulation_precision():
    model = encoder(dtype="float32")
    assert model.configuration["backend"] == "pytorch_reference_cpu_qkv64_v1"
    assert model.configuration.get("cpu_float32_qkv_accumulation") == "float64"
    assert all(
        block.configuration.get("cpu_float32_qkv_accumulation") == "float64"
        for block in model.blocks
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_qkv_module_preserves_hooks_output_dtype_and_manual_reference_gradients(dtype):
    import torch
    from torch.nn import functional as F

    model = encoder(dtype=dtype)
    projection = model.blocks[0].qkv_proj
    assert isinstance(projection, torch.nn.Linear) and projection.bias is None
    inputs = (
        torch.linspace(-0.8, 1.1, 192, dtype=getattr(torch, dtype))
        .reshape(2, 3, 32)
        .requires_grad_()
    )
    captured = []
    hook = projection.register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        actual = projection(inputs)
    finally:
        hook.remove()
    assert len(captured) == 1 and captured[0] is actual
    assert actual.dtype == inputs.dtype == projection.weight.dtype
    expected = F.linear(inputs.double(), projection.weight.double()).to(inputs.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    coefficient = torch.linspace(0.2, 0.9, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    variables = (inputs, projection.weight)
    observed = torch.autograd.grad((actual * coefficient).sum(), variables)
    wanted = torch.autograd.grad((expected * coefficient).sum(), variables)
    for gradient, reference in zip(observed, wanted, strict=True):
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_encoder_fixture_reuses_declared_seed_without_changing_parent_rng(device):
    import torch

    before_cpu = torch.random.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state("cuda:0").clone() if device == "cuda:0" else None
    first = encoder(device=device, init_seed=42)
    repeated = encoder(device=device, init_seed=42)
    different = encoder(device=device, init_seed=43)
    torch.testing.assert_close(first.projection.weight, repeated.projection.weight, rtol=0, atol=0)
    assert not torch.equal(first.projection.weight, different.projection.weight)
    assert torch.equal(torch.random.get_rng_state(), before_cpu)
    if before_cuda is not None:
        assert torch.equal(torch.cuda.get_rng_state("cuda:0"), before_cuda)


@pytest.mark.parametrize("fault", ["configuration", "shape", "dtype", "nan", "cursor", "weights"])
def test_cache_is_validated_before_a_step_without_mutating_previous_state(fault, monkeypatch):
    import torch

    model = encoder().eval()
    inputs = prices("cpu", "float64")
    with torch.no_grad():
        _, state = model.encode_steps(inputs[:, :2])
    if fault == "configuration":
        state["configuration"]["rule"] = "vector"
    elif fault == "shape":
        state["layers"][0]["recurrent_state"] = state["layers"][0]["recurrent_state"][:1]
    elif fault == "dtype":
        state["layers"][0]["conv_state"] = state["layers"][0]["conv_state"].float()
    elif fault == "nan":
        state["layers"][0]["recurrent_state"].view(-1)[0] = float("nan")
    elif fault == "cursor":
        state["cursor"] = True
    elif fault == "weights":
        with torch.no_grad():
            model.projection.weight.add_(0.1)
    previous = deepcopy(state)

    def forbidden(*args, **kwargs):
        pytest.fail("Un estado incompatible no puede entrar en la proyección")

    monkeypatch.setattr(model.projection, "forward", forbidden)
    with pytest.raises(ValueError):
        model.encode_step(inputs[:, 2], state)
    assert state["cursor"] == previous["cursor"]
    for actual, wanted in zip(state["layers"], previous["layers"], strict=True):
        for name in ("recurrent_state", "conv_state"):
            torch.testing.assert_close(actual[name], wanted[name], rtol=0, atol=0, equal_nan=True)


@pytest.mark.cuda
def test_cache_device_mismatch_fails_before_projection(monkeypatch):
    import torch

    model = encoder(device="cuda:0").eval()
    inputs = prices("cuda:0", "float64")
    with torch.no_grad():
        _, state = model.encode_steps(inputs[:, :2])
    state["layers"][0]["conv_state"] = state["layers"][0]["conv_state"].cpu()

    def forbidden(*args, **kwargs):
        pytest.fail("La caché de otro dispositivo no puede entrar en la proyección")

    monkeypatch.setattr(model.projection, "forward", forbidden)
    with pytest.raises(ValueError, match="dispositivo"):
        model.encode_step(inputs[:, 2], state)
    assert state["cursor"] == 2


def test_checkpoint_contract_fails_before_copy_and_cache_rejects_other_weights():
    import torch

    source, restored = encoder().eval(), encoder(init_seed=43).eval()
    inputs = prices("cpu", "float64")
    with torch.no_grad():
        _, state = source.encode_steps(inputs[:, :2])
    payload = source.export_state(state)
    with pytest.raises(ValueError, match="pesos|parámetros"):
        restored.restore_state(payload)
    restored.load_state_dict(deepcopy(source.state_dict()))
    saved = deepcopy(source.state_dict())
    saved["projection.weight"].zero_()
    saved["_extra_state"]["rule"] = "vector"
    before = restored.projection.weight.detach().clone()
    with pytest.raises(ValueError, match="contrato|configuración"):
        restored.load_state_dict(saved, strict=False)
    torch.testing.assert_close(restored.projection.weight, before, rtol=0, atol=0)
    accepted = restored.restore_state(payload)
    assert accepted["cursor"] == 2


@pytest.mark.parametrize(
    "options",
    [dict(context=True), dict(hidden_size=0), dict(heads=3), dict(layers=0), dict(max_batch=0)],
)
def test_invalid_architecture_is_rejected_without_consuming_rng(options):
    import torch

    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        encoder(**options)
    assert torch.equal(before, torch.random.get_rng_state())


def test_configuration_copies_are_independent_and_identify_the_full_block():
    model = encoder()
    settings = model.configuration
    assert settings["state_policy"] == "reset_per_window"
    assert settings["rule"] == "scalar"
    assert settings["conv_size"] == 4
    assert settings["normalize_v"] is True
    assert settings["decouple_erase"] is True
    assert settings["activation"] == "swiglu"
    assert settings["intermediate_size"] == 256
    settings["rule"] = "vector"
    assert model.configuration["rule"] == "scalar"


def test_zero_windows_are_finite_and_batches_do_not_share_recurrent_state():
    import torch

    model = encoder().eval()
    with torch.no_grad():
        zero = model(torch.zeros(2, 8, 3, dtype=torch.float64))
        torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)
        inputs = prices("cpu", "float64")
        expected = model(inputs)
        reversed_batch = model(inputs.flip(0)).flip(0)
        separate = torch.cat([model(row[None]) for row in inputs])
    torch.testing.assert_close(reversed_batch, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(separate, expected, rtol=1e-10, atol=1e-10)


def test_oversized_batch_and_nonfinite_prices_fail_before_computation(monkeypatch):
    import torch

    model = encoder()

    def forbidden(*args, **kwargs):
        pytest.fail("No debe calcularse una ventana inválida")

    monkeypatch.setattr(model.projection, "forward", forbidden)
    with pytest.raises(ValueError):
        model(torch.zeros(5, 8, 3, dtype=torch.float64))
    invalid = prices("cpu", "float64")
    invalid[0, -1, -1] = float("inf")
    with pytest.raises(ValueError, match="finito|NaN"):
        model(invalid)


def test_cache_rejects_a_changed_installed_rule():
    import torch

    model = encoder().eval()
    inputs = prices("cpu", "float64")
    with torch.no_grad():
        _, state = model.encode_steps(inputs[:, :2])
    model.blocks[0].rule = "vector"
    with pytest.raises(ValueError, match="peso|configuración|semántica|contrato"):
        model.encode_step(inputs[:, 2], state)
    with pytest.raises(ValueError):
        model.export_state(state)
    with pytest.raises(ValueError):
        model.restore_state(state)
    assert state["cursor"] == 2


def test_cache_rejects_a_changed_installed_norm_epsilon():
    import torch

    model = encoder().eval()
    inputs = prices("cpu", "float64")
    with torch.no_grad():
        _, state = model.encode_steps(inputs[:, :2])
    model.blocks[0].attn_norm.eps = 0.02
    with pytest.raises(ValueError, match="peso|configuración|semántica|contrato"):
        model.encode_step(inputs[:, 2], state)
    with pytest.raises(ValueError):
        model.export_state(state)
    with pytest.raises(ValueError):
        model.restore_state(state)
    assert state["cursor"] == 2


@pytest.mark.parametrize("fault", ["nan", "shape"])
def test_checkpoint_tensor_faults_fail_before_any_parameter_is_copied(fault):
    import torch

    source, destination = encoder(), encoder()
    incoming = deepcopy(source.state_dict())
    incoming["projection.weight"].fill_(1.0)
    name = "blocks.0.qkv_proj.weight"
    if fault == "nan":
        incoming[name].view(-1)[0] = float("nan")
    else:
        incoming[name] = incoming[name][:1]
    previous = deepcopy(destination.state_dict())
    with pytest.raises((ValueError, RuntimeError)):
        destination.load_state_dict(incoming, strict=False)
    for key, value in destination.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, previous[key], rtol=0, atol=0)
        else:
            assert value == previous[key]


def test_restored_cache_capacity_is_checked_before_reading_or_cloning_tensors(monkeypatch):
    import torch

    model = encoder(hidden_size=128, heads=1, layers=2, max_batch=8192)
    scalar = torch.zeros(1, dtype=torch.float64)
    payload = dict(
        configuration=model.configuration,
        weights_sha256="0" * 64,
        cursor=1,
        layers=tuple(
            dict(
                recurrent_state=scalar.expand(8192, 1, 128, 128),
                conv_state=scalar.expand(8192, 384, 4),
            )
            for _ in range(2)
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("El presupuesto debe rechazarse antes de leer o clonar la caché")

    monkeypatch.setattr(model, "_cache", forbidden)
    with pytest.raises(ValueError, match="presupuesto|límite|capacidad"):
        model.restore_state(payload)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("field", ["A_log", "dt_bias"])
def test_raw_decay_parameters_cannot_hide_infinity_after_transformations(device, field):
    import torch

    model = encoder(device=device).eval()
    with torch.no_grad():
        getattr(model.blocks[0], field).fill_(float("-inf"))
    with pytest.raises(ValueError, match="NaN|infinito|finito"):
        model.encode_last(prices(device, "float64"))
