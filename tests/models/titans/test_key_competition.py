"""Ecuaciones, límites y controles de las representaciones competitivas."""

import importlib
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from mars_titan.models.titans.config import MACConfig, MemoryConfig
from mars_titan.models.titans.mac import TitansMAC

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


def api():
    return importlib.import_module("mars_titan.models.titans.key_competition")


def options(**changes):
    return importlib.import_module(
        "mars_titan.models.titans.key_competition_config"
    ).KeyCompetition(**changes)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_selection_preserves_signed_values_and_has_a_fixed_coordinate_control(device, dtype):
    value = torch.tensor([[1.0, 2.0, 4.0, 0.0]], device=device, dtype=dtype)
    actual = api().competitive_normalize(value, options(retained=2))
    expected = torch.tensor([[0.0, 2.0, 4.0, 0.0]], device=device, dtype=dtype) / (20**0.5)
    torch.testing.assert_close(actual, expected)
    fixed = api().competitive_normalize(value, options(retained=2, selection="fixed"))
    torch.testing.assert_close(
        fixed, torch.tensor([[1.0, 0.0, 4.0, 0.0]], device=device, dtype=dtype) / (17**0.5)
    )
    combined = api().competitive_normalize(value, options(center=True, retained=2))
    torch.testing.assert_close(
        combined, torch.tensor([[0.0, 0.0, 2.25, -1.75]], device=device, dtype=dtype) / (8.125**0.5)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_ties_select_lower_coordinates_and_zero_vectors_stay_finite(device):
    value = torch.tensor([[-2.0, 2.0, -2.0, 1.0]], device=device, dtype=torch.float64)
    result = api().competitive_normalize(value, options(retained=2))
    torch.testing.assert_close(
        result, torch.tensor([[-2.0, 2.0, 0.0, 0.0]], device=device, dtype=torch.float64) / (8**0.5)
    )
    zero = torch.zeros(2, 4, device=device, dtype=torch.float64, requires_grad=True)
    transformed = api().competitive_normalize(zero, options(center=True, retained=2))
    assert torch.count_nonzero(transformed) == 0
    transformed.sum().backward()
    assert torch.isfinite(zero.grad).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_keeping_all_features_preserves_normalization_and_its_complete_backward(device, dtype):
    generator = torch.Generator(device=device).manual_seed(71)
    value = torch.randn(
        4, 3, 8, device=device, dtype=dtype, generator=generator, requires_grad=True
    )
    upstream = torch.randn(4, 3, 8, device=device, dtype=dtype, generator=generator)
    actual = api().competitive_normalize(value, options(retained=8))
    expected = F.normalize(value, dim=-1, eps=1e-12)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (a,) = torch.autograd.grad((actual * upstream).sum(), value, retain_graph=True)
    (b,) = torch.autograd.grad((expected * upstream).sum(), value)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_centering_and_topk_can_increase_similarity_instead_of_separating_patterns():
    vectors = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
    normalized = F.normalize(vectors, dim=-1)
    centered = api().competitive_normalize(vectors, options(center=True))
    assert (normalized[0] @ normalized[1]).item() == 0
    torch.testing.assert_close(centered[0] @ centered[1], torch.tensor(1.0, dtype=torch.float64))
    vectors = torch.tensor([[1.0, 0.9, 0.0], [1.0, 0.0, 0.9]], dtype=torch.float64)
    selected = api().competitive_normalize(vectors, options(retained=1))
    assert torch.equal(selected[0], selected[1])
    assert not torch.equal(vectors[0], vectors[1])


def test_gradients_match_finite_differences_away_from_selection_boundaries():
    value = torch.tensor([[1.0, 2.2, -4.0, 0.3]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: api().competitive_normalize(x, options(center=True, retained=2)), (value,)
    )


@pytest.mark.parametrize("shared", [False, True])
def test_actual_associative_write_uses_the_selected_key_in_both_write_routes(shared):
    from mars_titan.models.titans.neural_memory import NeuralMemory

    model = NeuralMemory(
        MemoryConfig(4, depth=1, key_competition=options(retained=2)), dtype=torch.float64
    )
    with torch.no_grad():
        model.initial_weights[0].copy_(torch.eye(4, dtype=torch.float64))
        model.key_projection.weight.copy_(torch.eye(4, dtype=torch.float64))
        for layer in (
            model.value_projection,
            model.alpha_projection,
            model.eta_projection,
            model.theta_projection,
        ):
            layer.weight.zero_()
    token = torch.tensor([[1.0, 2.0, 4.0, 0.0]], dtype=torch.float64)
    state = model.initial_state(1)
    result = (
        model.update_shared(token.repeat(3, 1), state)
        if shared
        else model.update(token.unsqueeze(1), state)
    )
    expected = torch.tensor(
        [
            [
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.48, -0.04, 0.0],
                [0.0, -0.04, 0.42, 0.0],
                [0.0, 0.0, 0.0, 0.5],
            ]
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result.weights[0], expected, rtol=1e-14, atol=1e-14)


def test_mac_read_receives_the_competitive_query(monkeypatch):
    model = TitansMAC(
        MACConfig(
            MemoryConfig(4, depth=1, key_competition=options(retained=2)),
            heads=1,
            persistent_tokens=0,
        ),
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.query_projection.weight.copy_(torch.eye(4, dtype=torch.float64))
    seen = []
    read = model.memory.read

    def checked(query, state):
        seen.append(query.detach().clone())
        return read(query, state)

    monkeypatch.setattr(model.memory, "read", checked)
    output, state = model(
        torch.tensor([[[1.0, 2.0, 4.0, 0.0]]], dtype=torch.float64), model.initial_state(1)
    )
    torch.testing.assert_close(
        seen[0], torch.tensor([[[0.0, 2.0, 4.0, 0.0]]], dtype=torch.float64) / (20**0.5)
    )
    assert torch.isfinite(output).all() and state.memory.steps.tolist() == [1]


@pytest.mark.parametrize(
    "changes",
    [
        dict(center=1),
        dict(retained=0),
        dict(retained=True),
        dict(selection="unknown"),
        dict(selection="fixed"),
    ],
)
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        options(**changes)


@pytest.mark.parametrize("fault", ["nan", "inf", "overflow", "width", "half", "empty"])
def test_invalid_or_overflowing_input_is_rejected(fault):
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    if fault == "nan":
        value[0, 0] = float("nan")
    if fault == "inf":
        value[0, 0] = float("inf")
    if fault == "overflow":
        value *= 1e307
    if fault == "width":
        value = value[:, :1]
    if fault == "half":
        value = value.half()
    if fault == "empty":
        value = value[:0]
    with pytest.raises(ValueError):
        api().competitive_normalize(value, options(retained=2))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complete_mac_control_preserves_parameters_gradients_rng_and_recovery(device, dtype):
    base = MemoryConfig(8, depth=2, qkv_silu=True, qkv_convolution=4)
    assert (
        MemoryConfig(8).fingerprint()
        == "31a72e10c72ee4d8736d052963bc37d3afa76fa46b76381e991bee63367082f6"
    )
    controlled = replace(base, key_competition=options(retained=8))
    models = [
        TitansMAC(
            MACConfig(c, heads=2, persistent_tokens=2, attention_layout="sequence_contiguous_v1"),
            device=device,
            dtype=dtype,
        )
        for c in (base, controlled)
    ]
    for left, right in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    values = torch.arange(64, device=device, dtype=dtype).reshape(2, 4, 8) / 50
    outputs, states, gradients = [], [], []
    rng = torch.random.get_rng_state().clone()
    for model in models:
        state = model.initial_state(2)
        output, state = model(values, state, differentiable=True)
        outputs.append(output)
        states.append(state)
        gradients.append(
            torch.autograd.grad(output.square().sum(), tuple(model.parameters()), allow_unused=True)
        )
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for left, right in zip(*gradients, strict=True):
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    restored = models[1].restore_state(models[1].export_state(states[1]))
    next_value = values[:, :1].contiguous()
    a, _ = models[1](next_value, states[1])
    b, _ = models[1](next_value, restored)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("config", ["unknown", options(retained=9), options()])
def test_memory_rejects_incompatible_feature_contracts(config):
    with pytest.raises(ValueError):
        MemoryConfig(8, key_competition=config, normalize_qk=config != options())


def test_market_read_receives_the_same_transform_as_its_write(monkeypatch):
    from mars_titan.models.titans.session_context import MARKET_MEMORY, SessionContext

    model = SessionContext(
        MARKET_MEMORY, 4, seed=42, memory_options=dict(key_competition=options(retained=2))
    )
    model.double()
    with torch.no_grad():
        model.query.copy_(torch.eye(4, dtype=torch.float64))
    state = model.initial_state("a" * 64)
    original = model.memory.read_shared
    seen = []

    def checked(query, memory):
        seen.append(query.detach().clone())
        return original(query, memory)

    monkeypatch.setattr(model.memory, "read_shared", checked)
    value = model._read(
        torch.tensor([[1.0, 2.0, 4.0, 0.0]], dtype=torch.float64), ("US/AAA",), 0, state
    )
    torch.testing.assert_close(
        seen[0], torch.tensor([[0.0, 2.0, 4.0, 0.0]], dtype=torch.float64) / (20**0.5)
    )
    assert torch.isfinite(value).all()


def test_small_vectors_use_the_historical_epsilon_and_strides_preserve_values():
    value = torch.tensor([[1e-13, -1e-14]], dtype=torch.float64)
    actual = api().competitive_normalize(value, options(retained=1))
    torch.testing.assert_close(
        actual, torch.tensor([[0.1, 0.0]], dtype=torch.float64), rtol=1e-14, atol=0
    )
    value = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4).transpose(0, 1)
    config = options(center=True, retained=2)
    actual = api().competitive_normalize(value, config)
    expected = api().competitive_normalize(value.contiguous(), config)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.flip(0), api().competitive_normalize(value.flip(0), config), rtol=0, atol=0
    )


@pytest.mark.parametrize("module_name", ["key_competition", "key_competition_config"])
def test_the_execution_identity_changes_when_a_feature_implementation_changes(
    tmp_path, monkeypatch, module_name
):
    from mars_titan.training.titans_walk_forward import _code

    module = importlib.import_module("mars_titan.models.titans." + module_name)
    source = tmp_path / "implementation.py"
    source.write_text("implementation revision one\n")
    monkeypatch.setattr(module, "__file__", str(source))
    first = _code()
    source.write_text("implementation revision two\n")
    assert _code() != first
