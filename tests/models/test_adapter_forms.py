"""Formas de adaptador de #444 sobre padres congelados, sin pasos de optimizador.

Cada forma empieza en la identidad: con sus correcciones nulas el brazo debe emitir los
mismos bits que el padre, en inferencia y en entrenamiento. Además se contrasta cada forma
con su ecuación escrita a mano y sus gradientes con diferencias finitas en float64.
"""

import copy

import pytest
import torch
from torch.nn import functional

from mars_titan.models.baselines.multimodal import (
    PRESENCE_FUSION,
    STRICT_FUSION,
    MultimodalReference,
)
from mars_titan.models.predictive_adaptation import (
    MODULE_ROOT,
    AdapterTarget,
    BottleneckAdapter,
    ColumnGain,
    RowGain,
    SharedRowGain,
    WeightDecomposedDelta,
    adapted_copy,
    adapter_names,
    base_digest,
    is_adapter_name,
    target_shape,
    trainable_parameters,
)
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.posttraining import adapter_variety

DIMENSIONS = dict(prices=5, news=7, charts=6, fundamentals=3, macro=6)
FAMILIES = ("rnn", "lstm", "gru", "dlinear", "transformer")
HIDDEN = 32


def parent(kind, fusion=PRESENCE_FUSION, layers=2, dtype=torch.float32):
    torch.manual_seed(7)
    model = MultimodalReference(
        kind,
        DIMENSIONS,
        context=8,
        hidden_size=HIDDEN,
        layers=layers,
        dropout=0.0,
        transformer=dict(heads=2, feedforward_multiplier=2) if kind == "transformer" else None,
        mask_fusion=fusion,
        head=QUANTILE_HEAD,
    )
    return model.to(dtype).eval().requires_grad_(False)


def batch(size=5, seed=3, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    inputs = {
        name: torch.randn(
            (size, 8, width) if name == "prices" else (size, width),
            generator=generator,
            dtype=dtype,
        )
        for name, width in DIMENSIONS.items()
    }
    presence = torch.tensor([[1, 1, 1, 0, 1], [1, 0, 1, 1, 0]] * size, dtype=torch.bool)[:size]
    for column, name in ((1, "news"), (3, "fundamentals"), (4, "macro")):
        inputs[name] = inputs[name] * presence[:, column : column + 1]
    return inputs, presence


def forward(model, fusion, inputs, presence):
    return model(inputs, presence if fusion == PRESENCE_FUSION else None)


def same_bits(left, right):
    """Igualdad bit a bit, más estricta que `torch.equal`, que da por iguales 0,0 y -0,0."""
    kind = {torch.float32: torch.int32, torch.float64: torch.int64}[left.dtype]
    return left.dtype == right.dtype and torch.equal(left.view(kind), right.view(kind))


SPECS = {
    "fusion_dora": ("fusion", dict(form="dora", rank=2, alpha=2.0)),
    "readout_dora": ("readout", dict(form="dora", rank=2, alpha=2.0)),
    "fusion_ia3": ("fusion", dict(form="ia3")),
    "readout_ia3": ("readout", dict(form="ia3")),
    "fusion_parallel_adapter": ("fusion", dict(form="parallel_adapter", rank=4, alpha=4.0)),
    "fusion_serial_adapter": ("fusion", dict(form="serial_adapter", rank=4, alpha=4.0)),
    "bias": ("bias", dict(form="residual")),
    "norm": ("norm", dict(form="residual")),
}


def arm_targets(model, arm):
    point, spec = SPECS[arm]
    spec = dict(spec, invalidates=["cached_parent_predictions"])
    return adapter_variety.neural_targets(model, point, spec)


def applicable(kind, arm):
    point = SPECS[arm][0]
    return (point != "readout" or kind == "transformer") and (
        point != "norm" or kind in adapter_variety.NORM_FAMILIES
    )


CASES = [(kind, arm) for kind in FAMILIES for arm in SPECS if applicable(kind, arm)]


@pytest.mark.parametrize("fusion", [STRICT_FUSION, PRESENCE_FUSION])
@pytest.mark.parametrize(("kind", "arm"), CASES)
def test_null_forms_reproduce_the_parent_bit_for_bit(kind, arm, fusion):
    original = parent(kind, fusion)
    child = adapted_copy(original, arm_targets(original, arm), seed=11)
    inputs, presence = batch()
    with torch.inference_mode():
        expected = forward(original, fusion, inputs, presence)
        assert same_bits(forward(child, fusion, inputs, presence), expected)
    # En entrenamiento se compara con el padre en el mismo modo, porque el modo de
    # autograd cambia la ruta de algunos núcleos también para el propio padre.
    original.train()
    child.train()
    reference = forward(original, fusion, inputs, presence)
    output = forward(child, fusion, inputs, presence)
    assert same_bits(output.detach(), reference)
    output.square().sum().backward()
    names = set(adapter_names(child))
    assert names and all(is_adapter_name(name) for name in names)
    for name, value in child.named_parameters():
        if name in names:
            assert value.requires_grad and value.grad is not None
            assert torch.isfinite(value.grad).all()
        else:
            assert not value.requires_grad and value.grad is None
    assert any(torch.count_nonzero(child.get_parameter(name).grad) for name in names)


@pytest.mark.parametrize(("kind", "arm"), CASES)
def test_forms_leave_the_parent_and_its_base_untouched(kind, arm):
    original = parent(kind)
    before = {k: v.clone() for k, v in original.state_dict().items() if torch.is_tensor(v)}
    digest = base_digest(original)
    child = adapted_copy(original, arm_targets(original, arm), seed=11)
    assert base_digest(child) == digest
    with torch.no_grad():
        for name in adapter_names(child):
            child.get_parameter(name).add_(0.25)
    # Mover las correcciones cambia la salida, pero no la huella de la base ni el padre.
    inputs, presence = batch()
    with torch.inference_mode():
        assert not torch.equal(child(inputs, presence), original(inputs, presence))
    assert base_digest(child) == digest
    assert not any(module._forward_hooks for module in original.modules())
    assert not hasattr(original, MODULE_ROOT)
    after = original.state_dict()
    assert all(torch.equal(value, after[name]) for name, value in before.items())


@pytest.mark.parametrize(("kind", "arm"), CASES)
def test_declared_counts_match_the_trainable_parameters(kind, arm):
    original = parent(kind)
    targets = arm_targets(original, arm)
    child = adapted_copy(original, targets, seed=11)
    declared = sum(
        target.trainable_parameters(target_shape(original, target)) for target in targets
    )
    assert trainable_parameters(child) == declared > 0


def test_the_transformer_keeps_its_fast_path_with_every_form():
    """Ninguna forma registra ganchos dentro de las capas del codificador Transformer.

    Un gancho dentro de `TransformerEncoderLayer` desactiva su ruta rápida en inferencia y
    cambiaría los bits del brazo frente al padre. Las formas de la lectura son
    reparametrizaciones y los adaptadores por módulo solo se enganchan en la fusión.
    """
    original = parent("transformer")
    for arm in SPECS:
        child = adapted_copy(original, arm_targets(original, arm), seed=3)
        for block in child.price_encoder.blocks:
            assert not any(
                module._forward_hooks or module._forward_pre_hooks for module in block.modules()
            )


def test_selective_points_enumerate_every_bias_and_norm():
    model = parent("transformer", layers=1)
    biases = {(t.module, t.tensor) for t in adapter_variety.bias_targets(model)}
    expected = {
        (name.rpartition(".")[0], name.rpartition(".")[2])
        for name, value in model.named_parameters()
        if "bias" in name.rpartition(".")[2]
    }
    assert biases == expected
    assert ("price_encoder.blocks.0.self_attn", "in_proj_bias") in biases
    assert ("price_encoder.blocks.0.norm1", "bias") in biases
    norms = {(t.module, t.tensor) for t in adapter_variety.norm_targets(model)}
    assert norms == {
        (module, tensor)
        for module in (
            "price_encoder.blocks.0.norm1",
            "price_encoder.blocks.0.norm2",
            "price_encoder.norm",
        )
        for tensor in ("weight", "bias")
    }
    gru = parent("gru", layers=2)
    names = {t.tensor for t in adapter_variety.bias_targets(gru) if t.module == "price_encoder"}
    assert names == {"bias_ih_l0", "bias_hh_l0", "bias_ih_l1", "bias_hh_l1"}
    with pytest.raises(ValueError, match="normalizaciones"):
        adapter_variety.norm_targets(gru)


def test_weight_decomposition_follows_its_equation_away_from_the_start():
    torch.manual_seed(1)
    weight = torch.randn(6, 5, dtype=torch.float64)
    generator = torch.Generator().manual_seed(4)
    adapter = WeightDecomposedDelta(
        weight.shape, 2, alpha=3.0, rows=(1, 5), generator=generator, dtype=torch.float64
    )
    with torch.no_grad():
        adapter.up.normal_()
        adapter.magnitude.normal_()
    block = weight[1:5]
    direction = block + 1.5 * adapter.up @ adapter.down
    magnitude = block.norm(dim=1) + adapter.magnitude
    expected = weight.clone()
    expected[1:5] = magnitude[:, None] * direction / direction.norm(dim=1, keepdim=True)
    torch.testing.assert_close(adapter(weight), expected, rtol=1e-14, atol=0)
    # Las filas fuera del bloque se copian sin operar.
    assert torch.equal(adapter(weight)[[0, 5]], weight[[0, 5]])
    # Con U nula y magnitud nula, la matriz es exactamente la del padre.
    with torch.no_grad():
        adapter.up.zero_()
        adapter.magnitude.zero_()
    assert torch.equal(adapter(weight), weight)


def test_gains_follow_their_equations():
    torch.manual_seed(2)
    weight, bias = torch.randn(6, 4, dtype=torch.float64), torch.randn(6, dtype=torch.float64)
    rows = RowGain(weight.shape, rows=(0, 3), dtype=torch.float64)
    shared = SharedRowGain(rows)
    columns = ColumnGain(weight.shape, dtype=torch.float64)
    with torch.no_grad():
        rows.gain.copy_(torch.tensor([0.5, -0.25, 2.0]))
        columns.gain.copy_(torch.tensor([0.1, 0.2, -0.3, 0.4]))
    factor = torch.tensor([1.5, 0.75, 3.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert torch.equal(rows(weight), factor[:, None] * weight)
    assert torch.equal(shared(bias), factor * bias)
    assert list(shared.parameters()) == []
    expected = weight @ torch.diag(1 + columns.gain.detach())
    torch.testing.assert_close(columns(weight), expected, rtol=1e-15, atol=0)


def test_ia3_readout_equals_rescaled_keys_and_values():
    """La ganancia de la consulta y de las columnas de out_proj es (IA)³ sobre claves y valores.

    Se calcula la atención a mano con l_k ⊙ K y l_v ⊙ V (Liu et al., 2022, sección 3.3) y se
    compara con la `MultiheadAttention` adaptada, con sesgos de proyección no nulos.
    """
    torch.manual_seed(5)
    attention = torch.nn.MultiheadAttention(8, 2, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        attention.in_proj_bias.normal_()
        attention.out_proj.bias.normal_()
    holder = torch.nn.Module()
    holder.attention = attention
    targets = [
        AdapterTarget("attention", "in_proj_weight", "gain_rows", rows=(0, 8)),
        AdapterTarget(
            "attention", "in_proj_bias", "gain_rows", rows=(0, 8), shares="in_proj_weight"
        ),
        AdapterTarget("attention.out_proj", "weight", "gain_columns"),
    ]
    adapted = adapted_copy(holder, targets, seed=1)
    keys = torch.linspace(-0.5, 0.7, 8, dtype=torch.float64)
    values = torch.linspace(0.3, -0.4, 8, dtype=torch.float64)
    with torch.no_grad():
        adapted.attention.parametrizations.in_proj_weight[0].gain.copy_(keys)
        adapted.attention.out_proj.parametrizations.weight[0].gain.copy_(values)
    x = torch.randn(3, 5, 8, dtype=torch.float64)
    mask = torch.ones(5, 5, dtype=torch.bool).triu(1)
    output, _ = adapted.attention(x, x, x, attn_mask=mask, need_weights=False)
    w, b = attention.in_proj_weight, attention.in_proj_bias
    q, k, v = (x @ w[i * 8 : (i + 1) * 8].T + b[i * 8 : (i + 1) * 8] for i in range(3))
    k, v = k * (1 + keys), v * (1 + values)
    heads = []
    for head in range(2):
        part = slice(4 * head, 4 * (head + 1))
        scores = q[..., part] @ k[..., part].transpose(1, 2) / 2.0
        heads.append(torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1) @ v[..., part])
    expected = attention.out_proj(torch.cat(heads, dim=-1))
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("form", ["parallel_adapter", "serial_adapter"])
def test_bottleneck_adapters_follow_their_equation(form):
    torch.manual_seed(6)
    block = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.SiLU()).double()
    holder = torch.nn.Module()
    holder.fusion = block
    adapted = adapted_copy(holder, [AdapterTarget("fusion", "output", form, 2, 4.0)], seed=8)
    adapter = getattr(adapted, MODULE_ROOT)["fusion"]
    with torch.no_grad():
        adapter.up.normal_()
        adapter.up_bias.normal_()
        adapter.down_bias.normal_()
    x = torch.randn(4, 5, dtype=torch.float64)
    base = block(x)
    source = x if form == "parallel_adapter" else base
    hidden = torch.relu(source @ adapter.down.T + adapter.down_bias)
    expected = base + 2.0 * (hidden @ adapter.up.T + adapter.up_bias)
    torch.testing.assert_close(adapted.fusion(x), expected, rtol=1e-14, atol=1e-15)
    assert target_shape(holder, AdapterTarget("fusion", "output", form, 2, 4.0)) == (
        (3, 5) if form == "parallel_adapter" else (3, 3)
    )


def _gradcheck(module, *arguments):
    """Gradientes analíticos frente a diferencias finitas centradas, en float64."""
    names = [name for name, value in module.named_parameters() if value.requires_grad]
    values = tuple(module.get_parameter(n).detach().clone().requires_grad_(True) for n in names)

    def function(*current):
        return torch.func.functional_call(module, dict(zip(names, current, strict=True)), arguments)

    assert torch.autograd.gradcheck(function, values, eps=1e-6, atol=1e-7, rtol=1e-5)


def test_weight_decomposition_gradients_match_finite_differences():
    torch.manual_seed(3)
    weight = torch.randn(5, 4, dtype=torch.float64)
    adapter = WeightDecomposedDelta(
        weight.shape,
        2,
        alpha=2.0,
        rows=(1, 4),
        generator=torch.Generator().manual_seed(2),
        dtype=torch.float64,
    )
    with torch.no_grad():
        adapter.up.normal_()
        adapter.magnitude.uniform_(-0.1, 0.1)
    _gradcheck(adapter, weight)


class _SharedGains(torch.nn.Module):
    """Peso y sesgo con la misma ganancia por filas, como la consulta de (IA)³."""

    def __init__(self, shape):
        super().__init__()
        self.source = RowGain(shape, rows=(1, 4), dtype=torch.float64)
        self.shared = SharedRowGain(self.source)

    def forward(self, weight, bias):
        return self.source(weight).sum(dim=1) + self.shared(bias)


def test_gain_gradients_match_finite_differences():
    torch.manual_seed(4)
    weight, bias = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, dtype=torch.float64)
    rows = RowGain(weight.shape, rows=(1, 4), dtype=torch.float64)
    columns = ColumnGain(weight.shape, dtype=torch.float64)
    shared = _SharedGains(weight.shape)
    with torch.no_grad():
        for adapter in (rows, columns, shared.source):
            adapter.gain.normal_()
    _gradcheck(rows, weight)
    _gradcheck(columns, weight)
    # El parámetro compartido recibe la suma de los dos caminos, el del peso y el del sesgo.
    assert [name for name, _ in shared.named_parameters()] == ["source.gain"]
    _gradcheck(shared, weight, bias)


class _Branch(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, value):
        return self.inner.branch(value)


@pytest.mark.parametrize("form", ["parallel_adapter", "serial_adapter"])
def test_bottleneck_gradients_match_finite_differences(form):
    torch.manual_seed(5)
    adapter = BottleneckAdapter(
        (3, 4),
        2,
        alpha=2.0,
        form=form,
        generator=torch.Generator().manual_seed(1),
        dtype=torch.float64,
    )
    with torch.no_grad():
        adapter.up.normal_()
        adapter.up_bias.normal_()
        adapter.down_bias.fill_(0.3)
    x = torch.randn(6, 4, dtype=torch.float64)
    # Ninguna preactivación cerca del corte de la ReLU, donde la derivada no existe.
    assert (x @ adapter.down.T + adapter.down_bias).abs().min() > 1e-3
    _gradcheck(_Branch(adapter), x)


def test_null_dora_gradient_reaches_up_and_magnitude_only():
    original = parent("gru")
    child = adapted_copy(original, arm_targets(original, "fusion_dora"), seed=2)
    child.train()
    inputs, presence = batch()
    child(inputs, presence).square().sum().backward()
    chain = child.fusion[0].parametrizations.weight[0]
    # Con U nula, la bajada D no recibe gradiente en el primer paso, como en LoRA.
    assert torch.count_nonzero(chain.down.grad) == 0
    assert torch.count_nonzero(chain.up.grad) > 0
    assert torch.count_nonzero(chain.magnitude.grad) > 0


@pytest.mark.parametrize("arm", ["fusion_parallel_adapter", "readout_ia3", "fusion_dora", "bias"])
def test_adapted_state_reloads_into_the_same_arm_only(arm):
    original = parent("transformer")
    child = adapted_copy(original, arm_targets(original, arm), seed=9)
    with torch.no_grad():
        for name in adapter_names(child):
            child.get_parameter(name).uniform_(-0.2, 0.2)
    state = copy.deepcopy(child.state_dict())
    restored = adapted_copy(original, arm_targets(original, arm), seed=9)
    restored.load_state_dict(state)
    inputs, presence = batch()
    with torch.inference_mode():
        assert same_bits(restored(inputs, presence), child(inputs, presence))
    other = "fusion_serial_adapter" if arm != "fusion_serial_adapter" else "fusion_dora"
    with pytest.raises(RuntimeError):
        adapted_copy(original, arm_targets(original, other), seed=9).load_state_dict(state)


def test_a_deep_copy_rebinds_the_module_adapter_to_the_copy():
    original = parent("gru")
    child = adapted_copy(original, arm_targets(original, "fusion_parallel_adapter"), seed=4)
    twin = copy.deepcopy(child)
    with torch.no_grad():
        getattr(twin, MODULE_ROOT)["fusion"].up.fill_(0.5)
    inputs, presence = batch()
    with torch.inference_mode():
        assert same_bits(child(inputs, presence), original(inputs, presence))
        assert not torch.equal(twin(inputs, presence), original(inputs, presence))


@pytest.mark.parametrize(
    "targets",
    [
        [AdapterTarget("fusion.0", "bias", "gain_rows", shares="weight")],
        [
            AdapterTarget("fusion.0", "weight", "gain_rows", rows=(0, 4)),
            AdapterTarget("fusion.0", "bias", "gain_rows", rows=(0, 5), shares="weight"),
        ],
        [AdapterTarget("fusion.0", "weight", "gain_columns", rows=(0, 4))],
        [AdapterTarget("head", "output", "parallel_adapter", 2, 2.0)],
        [AdapterTarget("fusion", "output", "serial_adapter", 2, 2.0)] * 2,
        [AdapterTarget("fusion.0", "weight", "dora", 65, 1.0)],
    ],
)
def test_invalid_targets_fail_before_copying(targets):
    with pytest.raises(ValueError):
        adapted_copy(parent("gru"), targets, seed=1)


def test_dora_rejects_a_null_row_of_the_parent():
    original = parent("gru")
    with torch.no_grad():
        original.fusion[0].weight[3].zero_()
    with pytest.raises(ValueError, match="filas nulas"):
        adapted_copy(original, [AdapterTarget("fusion.0", "weight", "dora", 2, 2.0)], seed=1)


@pytest.mark.parametrize(
    "arguments",
    [
        dict(tensor="output", form="dora", rank=2, alpha=1.0),
        dict(tensor="weight", form="parallel_adapter", rank=2, alpha=1.0),
        dict(tensor="weight", form="gain_rows", rank=2, alpha=1.0),
        dict(tensor="weight", form="dora"),
        dict(tensor="bias", form="residual", shares="weight"),
        dict(tensor="bias", form="gain_rows", shares="bias"),
    ],
)
def test_targets_reject_inconsistent_declarations(arguments):
    with pytest.raises(ValueError):
        AdapterTarget("fusion.0", **arguments)


def test_identity_only_adds_the_shared_gain_when_declared():
    plain = AdapterTarget("fusion.0", "weight", "gain_rows").identity((4, 3))
    shared = AdapterTarget("fusion.0", "bias", "gain_rows", shares="weight").identity((4,))
    assert "shares" not in plain and plain["trainable_parameters"] == 4
    assert shared["shares"] == "weight" and shared["trainable_parameters"] == 0


def test_bottleneck_branch_uses_relu_and_the_declared_scale():
    adapter = BottleneckAdapter(
        (2, 3), 1, alpha=3.0, form="parallel_adapter", generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        adapter.down.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
        adapter.up.fill_(1.0)
    value = torch.tensor([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    expected = 3.0 * functional.relu(value[:, :1]).expand(2, 2)
    assert torch.equal(adapter.branch(value), expected)
