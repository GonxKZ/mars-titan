"""Memoria M(x) = x + LN(MLP(x)) de las actas, sin optimizador externo ni datos."""

import hashlib
import importlib
import json
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from mars_titan.models.titans.config import LAYER_NORM_EPS, canonical

# Capturadas en develop f9789a47, antes de introducir residual_layer_norm.
V1 = dict(
    memory="fb306941d3c59afc43c818f12f4766595293e98c94fdef60dd15b8017be802fb",
    mac="52e8ee7ca208bbb2f0a5f35af7166369ae5435ea78d303e6947a6589087eedb4",
    outputs="b0a3cc5d73b9f23bf09b587e497148c5b720ae0af08201fcb423ed7d36de29cb",
    state="8f4535c1fd99d7daf131068833253f341c70c7b26465b0ebc44b10c07b5277f5",
)
GATE_BIAS = dict(
    memory="57ad06d370cf1d6ec6a4057614bb857459a373c3bd5aa16ad140ebf3e0a55e5d",
    mac="1a785041a7af28e12cd25ffd1b4139147ea672a8ba4e350793268616285b3637",
    outputs="403e7aec32290e3cba928b9488fb57b53575658306641eb226bd663cec90c518",
    state="c65d9854acc36234a3d1efdee472aa2c240ab06249145823a28e69c5c68cd6ef",
)
# Huella de identidades y parámetros de FinancialConfig en las cuatro variantes, las dos
# cabezas y con y sin gate_bias, también capturada en develop f9789a47.
FINANCIAL_V1 = "2cc3ada6f5ca25094e8f007320da7b209f5adbc1a9c3f28ecd304b976f0b91ef"


def api():
    return importlib.import_module("mars_titan.models.titans")


def tensors_digest(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def mac_trace(**options):
    """Cinco segmentos online en FP64, como la captura de develop."""
    package = api()
    memory = package.MemoryConfig(dim=4, depth=2, max_batch=2, max_tokens=3, **options)
    config = package.MACConfig(memory=memory, heads=2, persistent_tokens=2, max_segment=3)
    module = package.TitansMAC(config, dtype=torch.float64)
    state = module.initial_state(2)
    generator = torch.Generator().manual_seed(7)
    outputs = []
    for _ in range(5):
        segment = torch.randn(2, 3, 4, generator=generator, dtype=torch.float64)
        output, state = module(segment, state)
        outputs.append(output)
    return dict(
        memory=memory.fingerprint(),
        mac=config.fingerprint(),
        outputs=tensors_digest(outputs),
        state=tensors_digest([*state.memory.weights, *state.memory.momentum]),
    )


def financial_digest():
    from test_financial_adapter import specification

    module = importlib.import_module("mars_titan.models.titans.financial")
    records = {}
    for bias in (None, dict(alpha_half_life=256.0, eta=0.5, theta=0.05)):
        for head in ("scalar", "quantile_head_v1"):
            for variant in module.VARIANTS:
                options = {} if head == "scalar" else dict(head=head)
                config = module.FinancialConfig(
                    specification(), variant=variant, hidden_size=32, gate_bias=bias, **options
                )
                model = module.FinancialPredictor(config, dtype=torch.float64)
                key = f"{'v1' if bias is None else 'gate_bias'}/{head}/{variant}"
                records[key] = [
                    hashlib.sha256(canonical(config.identity()).encode()).hexdigest(),
                    model._parameter_id,
                ]
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def memory(dim=3, depth=2, **options):
    package = api()
    config = package.MemoryConfig(
        dim=dim, depth=depth, max_batch=2, max_tokens=2, residual_layer_norm=True, **options
    )
    return package.NeuralMemory(config, dtype=torch.float64)


def test_disabled_option_keeps_v1_and_gate_bias_identities_outputs_and_states():
    assert mac_trace() == V1
    assert mac_trace(gate_bias=api().GateBias()) == GATE_BIAS
    config = api().MemoryConfig(dim=4)
    assert "memory_function" not in config.identity()
    assert config.identity()["residual"] is False and config.identity()["layer_norm"] is False


def test_disabled_option_keeps_the_financial_identities_and_parameters():
    assert financial_digest() == FINANCIAL_V1


def test_residual_memory_is_a_new_identity_with_the_same_parameter_draws():
    package = api()
    rng = torch.get_rng_state().clone()
    v1 = package.MemoryConfig(dim=4, depth=2)
    residual = package.MemoryConfig(dim=4, depth=2, residual_layer_norm=True)
    identity = residual.identity()
    assert residual.fingerprint() != v1.fingerprint()
    assert identity["residual"] is True and identity["layer_norm"] is True
    assert identity["memory_function"] == {
        "form": "input_plus_layer_norm_of_mlp",
        "source": "neurips_2025_section_3_3",
        "layer_norm_eps": LAYER_NORM_EPS,
        "layer_norm_affine": False,
        "expansion": 1,
    }
    left = package.NeuralMemory(v1, dtype=torch.float64).state_dict()
    right = package.NeuralMemory(residual, dtype=torch.float64).state_dict()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert set(left) == set(right)
    for name, value in left.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(right[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("depth", [1, 2])
def test_read_is_the_input_plus_layer_norm_of_the_mlp(depth):
    model = memory(depth=depth)
    generator = torch.Generator().manual_seed(3)
    state = model.initial_state(2)
    state = replace(
        state,
        weights=tuple(
            torch.randn(2, 3, 3, generator=generator, dtype=torch.float64) for _ in range(depth)
        ),
    )
    query = torch.randn(2, 2, 3, generator=generator, dtype=torch.float64)
    hidden = query @ state.weights[0].transpose(1, 2)
    if depth == 2:
        hidden = F.gelu(hidden) @ state.weights[1].transpose(1, 2)
    centred = hidden - hidden.mean(-1, keepdim=True)
    variance = centred.square().mean(-1, keepdim=True)
    expected = query + centred / torch.sqrt(variance + LAYER_NORM_EPS)
    torch.testing.assert_close(model.read(query, state), expected, rtol=1e-12, atol=1e-12)


def test_empty_memory_reads_the_query_instead_of_zero():
    model = memory()
    state = model.initial_state(1)
    empty = replace(state, weights=tuple(torch.zeros_like(w) for w in state.weights))
    query = torch.tensor([[[0.3, -0.2, 0.5]]], dtype=torch.float64)
    torch.testing.assert_close(model.read(query, empty), query, rtol=0, atol=0)
    v1 = api().NeuralMemory(api().MemoryConfig(dim=3, depth=2), dtype=torch.float64)
    torch.testing.assert_close(
        v1.read(query, replace(v1.initial_state(1), weights=empty.weights)),
        torch.zeros_like(query),
        rtol=0,
        atol=0,
    )


def test_update_follows_equation_three_with_the_residual_associative_loss():
    model = memory(gate_bias=api().GateBias())
    generator = torch.Generator().manual_seed(11)
    state = model.initial_state(1)
    state = replace(
        state,
        momentum=tuple(
            0.01 * torch.randn(1, 3, 3, generator=generator, dtype=torch.float64) for _ in range(2)
        ),
    )
    token = torch.randn(1, 1, 3, generator=generator, dtype=torch.float64)
    with torch.no_grad():
        x = token[:, 0]
        keys = F.normalize(model.key_projection(x), dim=-1, eps=1e-12)
        values = model.value_projection(x)
        alpha = model.alpha_projection(x).sigmoid().unsqueeze(-1)
        eta = model.eta_projection(x).sigmoid().unsqueeze(-1)
        theta = model.config.theta_max * model.theta_projection(x).sigmoid().unsqueeze(-1)

    def loss(first, second):
        hidden = F.gelu(keys @ first[0].T) @ second[0].T
        centred = hidden - hidden.mean(-1, keepdim=True)
        normed = centred / torch.sqrt(centred.square().mean(-1, keepdim=True) + LAYER_NORM_EPS)
        return (keys + normed - values).square().sum()

    weights = tuple(w.clone().requires_grad_(True) for w in state.weights)
    gradients = torch.autograd.grad(loss(*weights), weights)
    momentum = tuple(eta * m - theta * g for m, g in zip(state.momentum, gradients, strict=True))
    expected = tuple((1 - alpha) * w + s for w, s in zip(state.weights, momentum, strict=True))
    result = model.update(token, state)
    actual_values = (*result.weights, *result.momentum)
    for actual, wanted in zip(actual_values, (*expected, *momentum), strict=True):
        torch.testing.assert_close(actual, wanted, rtol=1e-12, atol=1e-14)


def test_residual_updates_pass_gradcheck_for_inputs_weights_and_momentum():
    model = memory(dim=3, depth=2)
    state = model.initial_state(1, differentiable=True)
    generator = torch.Generator().manual_seed(5)
    values = (0.5 * torch.randn(1, 2, 3, generator=generator, dtype=torch.float64)).requires_grad_()
    parameters = (*state.weights, *(m.detach().requires_grad_() for m in state.momentum))

    def function(x, w1, w2, m1, m2):
        initial = replace(state, weights=(w1, w2), momentum=(m1, m2))
        result = model.update(x, initial, differentiable=True)
        return (*result.weights, *result.momentum, model.read(x, result))

    assert torch.autograd.gradcheck(function, (values, *parameters), eps=1e-6, atol=2e-5, rtol=2e-4)


def test_v1_and_residual_contracts_reject_each_other_states_and_parameters():
    package = api()
    v1 = package.NeuralMemory(package.MemoryConfig(dim=3, depth=2), dtype=torch.float64)
    residual = memory()
    with pytest.raises(ValueError, match="contrato"):
        residual.validate_state(v1.initial_state(1))
    with pytest.raises(ValueError, match="contrato"):
        v1.validate_state(residual.initial_state(1))
    with pytest.raises(ValueError):
        residual.load_state_dict(v1.state_dict())
    with pytest.raises(ValueError):
        residual.restore_state(v1.export_state(v1.initial_state(1)))


@pytest.mark.parametrize("options", [dict(dim=1), dict(dim=4, residual_layer_norm=1)])
def test_invalid_residual_declarations_are_rejected(options):
    declared = dict(residual_layer_norm=True) | options
    with pytest.raises(ValueError):
        api().MemoryConfig(**declared)


def test_mac_with_residual_memory_keeps_segment_causality_and_flow_isolation():
    package = api()
    memory_config = package.MemoryConfig(
        dim=4, depth=2, max_batch=2, max_tokens=2, residual_layer_norm=True
    )
    config = package.MACConfig(
        memory=memory_config,
        heads=2,
        persistent_tokens=2,
        max_segment=2,
    )
    core = package.TitansMAC(config, dtype=torch.float64)
    state = core.initial_state(2)
    x = torch.arange(16, dtype=torch.float64).reshape(2, 2, 4) / 13
    output, after = core(x, state)
    changed = x.clone()
    changed[1] *= -3
    other, _ = core(changed, state)
    torch.testing.assert_close(other[0], output[0], rtol=0, atol=0)
    assert not torch.equal(other[1], output[1])
    later, _ = core(x * 5, after)
    again, _ = core(x, state)
    torch.testing.assert_close(again, output, rtol=0, atol=0)
    assert torch.isfinite(later).all()


def test_financial_controls_declare_and_pair_the_residual_memory():
    from test_financial_adapter import specification

    module = importlib.import_module("mars_titan.models.titans.financial")
    models = {}
    for variant in module.VARIANTS:
        config = module.FinancialConfig(
            specification(), variant=variant, hidden_size=32, memory_residual_layer_norm=True
        )
        assert config.identity()["memory_residual_layer_norm"] is True
        models[variant] = module.FinancialPredictor(config, dtype=torch.float64)
        if models[variant].mac is not None:
            assert models[variant].mac.memory.config.residual_layer_norm is True
    for variant in ("transformer_direct", "mac_disabled", "mac_frozen"):
        receipt = module.copy_paired_parameters(models["mac_online"], models[variant])
        assert receipt["runtime_state_transferred"] is False
    plain = module.FinancialPredictor(
        module.FinancialConfig(specification(), variant="mac_frozen", hidden_size=32),
        dtype=torch.float64,
    )
    with pytest.raises(ValueError):
        module.copy_paired_parameters(models["mac_online"], plain)
    with pytest.raises(ValueError, match="booleano"):
        module.FinancialConfig(
            specification(), variant="mac_online", hidden_size=32, memory_residual_layer_norm=1
        )


def test_residual_memory_keeps_the_mac_output_away_from_the_v1_scale():
    """Escala técnica en 64 observaciones con tokens de norma cercana a 1."""
    package = api()
    norms = {}
    for residual in (False, True):
        memory_config = package.MemoryConfig(
            dim=16,
            depth=2,
            max_batch=2,
            max_tokens=1,
            gate_bias=package.GateBias(),
            residual_layer_norm=residual,
        )
        core = package.TitansMAC(
            package.MACConfig(memory=memory_config, heads=4, persistent_tokens=4, max_segment=1),
            dtype=torch.float64,
        )
        generator = torch.Generator().manual_seed(2026)
        state, outputs = core.initial_state(2), []
        for _ in range(64):
            token = torch.randn(2, 1, 16, generator=generator, dtype=torch.float64) / 4
            output, state = core(token, state)
            outputs.append(output.norm(dim=-1).mean().item())
        norms[residual] = outputs
    # Con la memoria v1 la salida queda por debajo de 0,05. Con residual y LN, por encima
    # de 0,1 en todas las observaciones y sin decaer entre la primera y la última ventana.
    assert max(norms[False]) < 0.05
    assert min(norms[True]) > 0.1
    assert sum(norms[True][-8:]) > 0.5 * sum(norms[True][:8])
