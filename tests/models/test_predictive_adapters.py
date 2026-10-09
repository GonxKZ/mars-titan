"""Adaptadores nulos sobre un padre congelado, sin pasos de optimizador."""

import pytest
import torch
from torch.nn.utils import parametrize

from mars_titan.models.baselines.multimodal import (
    PRESENCE_FUSION,
    STRICT_FUSION,
    MultimodalReference,
)
from mars_titan.models.predictive_adaptation import (
    AdapterTarget,
    LowRankDelta,
    ResidualDelta,
    adapted_copy,
    trainable_parameters,
)

DIMENSIONS = dict(prices=5, news=7, charts=6, fundamentals=3, macro=6)
FAMILIES = ("rnn", "lstm", "gru", "dlinear", "transformer")


def parent(kind, fusion=PRESENCE_FUSION, hidden=32, layers=2):
    torch.manual_seed(7)
    model = MultimodalReference(
        kind,
        DIMENSIONS,
        context=8,
        hidden_size=hidden,
        layers=layers,
        dropout=0.0,
        transformer=dict(heads=2, feedforward_multiplier=2) if kind == "transformer" else None,
        mask_fusion=fusion,
    )
    return model.eval().requires_grad_(False)


def batch(size=5, seed=3):
    generator = torch.Generator().manual_seed(seed)
    inputs = {
        name: torch.randn(
            (size, 8, width) if name == "prices" else (size, width), generator=generator
        )
        for name, width in DIMENSIONS.items()
    }
    presence = torch.tensor([[1, 1, 1, 0, 1], [1, 0, 1, 1, 0]] * size, dtype=torch.bool)[:size]
    for index, name in enumerate(("news", "fundamentals", "macro"), start=1):
        column = (1, 3, 4)[index - 1]
        inputs[name] = inputs[name] * presence[:, column : column + 1]
    return inputs, presence


def targets(kind, rank=2, layers=2, hidden=32):
    result = [
        AdapterTarget("head", "weight", "residual"),
        AdapterTarget("head", "bias", "residual"),
        AdapterTarget("fusion.0", "weight", "low_rank", rank=rank, alpha=float(rank)),
    ]
    if kind == "transformer":
        for layer in range(layers):
            prefix = f"price_encoder.blocks.{layer}.self_attn"
            result.extend(
                (
                    AdapterTarget(prefix, "in_proj_weight", "low_rank", rank, 2.0, (0, hidden)),
                    AdapterTarget(f"{prefix}.out_proj", "weight", "low_rank", rank, 2.0),
                )
            )
    return result


def forward(model, inputs, presence, fusion):
    return model(inputs, presence if fusion == PRESENCE_FUSION else None)


@pytest.mark.parametrize("fusion", [STRICT_FUSION, PRESENCE_FUSION])
@pytest.mark.parametrize("kind", FAMILIES)
def test_zero_adapters_reproduce_the_parent_exactly_in_every_mode(kind, fusion):
    original = parent(kind, fusion)
    child = adapted_copy(original, targets(kind), seed=11)
    inputs, presence = batch()
    with torch.inference_mode():
        expected = forward(original, inputs, presence, fusion)
        assert torch.equal(forward(child, inputs, presence, fusion), expected)
    # La comparación exacta se hace con los mismos núcleos. El modo de autograd
    # cambia la ruta de LSTM y de la atención también para el propio padre.
    original.train()
    reference = forward(original, inputs, presence, fusion)
    child.train()
    output = forward(child, inputs, presence, fusion)
    assert torch.equal(output.detach(), reference)
    torch.testing.assert_close(reference, expected, rtol=0, atol=1e-6)
    output.square().sum().backward()
    adapters = {name for name, value in child.named_parameters() if value.requires_grad}
    assert adapters and all(".parametrizations." in name for name in adapters)
    for name, value in child.named_parameters():
        if name in adapters:
            assert value.grad is not None and torch.isfinite(value.grad).all()
        else:
            assert value.grad is None and not value.requires_grad
    gradients = dict(child.named_parameters())
    # Con U nula, V no recibe gradiente en el primer paso y U sí.
    assert torch.count_nonzero(gradients["fusion.0.parametrizations.weight.0.down"].grad) == 0
    assert torch.count_nonzero(gradients["fusion.0.parametrizations.weight.0.up"].grad) > 0
    assert torch.count_nonzero(gradients["head.parametrizations.weight.0.delta"].grad) > 0


@pytest.mark.parametrize("kind", FAMILIES)
def test_the_parent_keeps_its_weights_and_receives_no_parametrization(kind):
    original = parent(kind)
    before = {
        name: value.clone()
        for name, value in original.state_dict().items()
        if torch.is_tensor(value)
    }
    child = adapted_copy(original, targets(kind), seed=11)
    with torch.no_grad():
        for value in child.parameters():
            if value.requires_grad:
                value.add_(1)
    assert not any(parametrize.is_parametrized(module) for module in original.modules())
    assert all(not value.requires_grad for value in original.parameters())
    after = original.state_dict()
    assert all(torch.equal(value, after[name]) for name, value in before.items())


@pytest.mark.parametrize("kind", FAMILIES)
def test_trainable_parameters_match_the_declared_targets(kind):
    original = parent(kind)
    declared = targets(kind)
    child = adapted_copy(original, declared, seed=11)
    expected = sum(
        target.trainable_parameters(
            tuple(getattr(original.get_submodule(target.module), target.tensor).shape)
        )
        for target in declared
    )
    assert trainable_parameters(child) == expected
    head, fusion = 32 + 1, 2 * (32 + 5 * 32 + 5)
    readout = 2 * (2 * (32 + 32) + 2 * (32 + 32)) if kind == "transformer" else 0
    assert expected == head + fusion + readout


def test_query_adapter_changes_only_query_rows_of_the_packed_projection():
    original = parent("transformer")
    child = adapted_copy(original, targets("transformer"), seed=11)
    attention = child.price_encoder.blocks[0].self_attn
    adapter = attention.parametrizations.in_proj_weight[0]
    with torch.no_grad():
        adapter.up.fill_(0.5)
    base = original.price_encoder.blocks[0].self_attn.in_proj_weight
    effective = attention.in_proj_weight
    assert not torch.equal(effective[:32], base[:32])
    assert torch.equal(effective[32:], base[32:])
    torch.testing.assert_close(
        effective[:32] - base[:32], adapter.scaling * adapter.up @ adapter.down, rtol=0, atol=1e-6
    )


def test_adapter_initialization_uses_its_own_seed_and_not_the_global_generator():
    original = parent("transformer")
    state = torch.random.get_rng_state()
    first = adapted_copy(original, targets("transformer"), seed=11)
    assert torch.equal(torch.random.get_rng_state(), state)
    second = adapted_copy(original, targets("transformer"), seed=11)
    other = adapted_copy(original, targets("transformer"), seed=12)
    name = "fusion.0.parametrizations.weight.0.down"
    values = [dict(model.named_parameters())[name] for model in (first, second, other)]
    assert torch.equal(values[0], values[1]) and not torch.equal(values[0], values[2])


def test_states_reload_only_into_the_same_adapter_layout():
    original = parent("transformer")
    child = adapted_copy(original, targets("transformer"), seed=11)
    with torch.no_grad():
        for value in child.parameters():
            if value.requires_grad:
                value.normal_(generator=torch.Generator().manual_seed(5))
    inputs, presence = batch()
    restored = adapted_copy(original, targets("transformer"), seed=99)
    restored.load_state_dict(child.state_dict(), strict=True)
    with torch.inference_mode():
        assert torch.equal(restored(inputs, presence), child(inputs, presence))
    # Dimensiones iguales no bastan: otra colocación tiene otras claves y se rechaza.
    other = adapted_copy(original, targets("transformer")[:2], seed=11)
    with pytest.raises(RuntimeError):
        other.load_state_dict(child.state_dict(), strict=True)


@pytest.mark.parametrize(
    "declared",
    [
        [],
        [AdapterTarget("missing", "weight", "residual")],
        [AdapterTarget("price_encoder", "positions", "residual")],
        [AdapterTarget("head", "weight", "residual")] * 2,
        [AdapterTarget("fusion.0", "weight", "low_rank", rank=65, alpha=1.0)],
        [AdapterTarget("fusion.0", "weight", "low_rank", rank=2, alpha=1.0, rows=(0, 33))],
        [AdapterTarget("head", "bias", "low_rank", rank=1, alpha=1.0)],
    ],
)
def test_invalid_targets_are_rejected_before_the_parent_is_copied(declared):
    with pytest.raises((ValueError, AttributeError)):
        adapted_copy(parent("transformer"), declared, seed=1)


@pytest.mark.parametrize(
    "arguments",
    [
        dict(form="other"),
        dict(form="low_rank"),
        dict(form="low_rank", rank=2),
        dict(form="residual", rank=2, alpha=1.0),
        dict(form="residual", rows=[0, 1]),
    ],
)
def test_target_declaration_is_complete(arguments):
    with pytest.raises(ValueError):
        AdapterTarget("head", "weight", **arguments)


def test_already_adapted_tensors_and_invalid_seeds_are_rejected():
    child = adapted_copy(parent("gru"), targets("gru"), seed=1)
    with pytest.raises(ValueError, match="original"):
        adapted_copy(child, [AdapterTarget("head", "weight", "residual")], seed=1)
    for seed in (-1, 1.0, 2**63):
        with pytest.raises(ValueError):
            adapted_copy(parent("gru"), targets("gru"), seed=seed)


def test_delta_modules_validate_shapes_and_generators():
    generator = torch.Generator().manual_seed(1)
    with pytest.raises(ValueError):
        ResidualDelta((2, 3, 4))
    with pytest.raises(ValueError):
        LowRankDelta((4,), 1, alpha=1.0, generator=generator)
    with pytest.raises(ValueError):
        LowRankDelta((4, 3), 1, alpha=float("nan"), generator=generator)
    with pytest.raises(ValueError):
        LowRankDelta((4, 3), 1, alpha=1.0, generator=None)
    delta = LowRankDelta((4, 3), 2, alpha=4.0, rows=(1, 3), generator=generator)
    assert delta.scaling == 2.0 and delta.up.shape == (2, 2) and delta.down.shape == (2, 3)
    weight = torch.randn(4, 3)
    assert torch.equal(delta(weight), weight)
    assert torch.equal(ResidualDelta((4,))(weight[:, 0]), weight[:, 0])
