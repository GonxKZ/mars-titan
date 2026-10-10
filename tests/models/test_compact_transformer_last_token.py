"""Camino de último token del codificador de precios, lote ampliado y finitud agrupada.

`CompactPriceTransformer.forward` calcula completa toda capa salvo la última, y en la
última solo las claves y valores de la ventana y el resto para el último token. Las
pruebas fijan la paridad con `encode_sequence(...)[:, -1]`: estricta en FP64 y, en FP32,
con un error frente a FP64 del mismo orden que el del camino completo. Las capas propias
se contrastan con `nn.TransformerEncoderLayer` en su ruta estándar. Sin optimizador.
"""

import warnings
from copy import deepcopy

import pytest
import torch

from mars_titan.models.baselines import transformer as module
from mars_titan.models.baselines.multimodal import MultimodalReference, transformer_batch_options
from mars_titan.models.baselines.transformer import (
    MAX_BATCH_LIMIT,
    CompactPriceTransformer,
    attention_budget,
    first_nonfinite,
    validate_attention_budget,
)

DIMENSIONS = dict(prices=5, news=8, charts=7, fundamentals=6, macro=9)
# Tolerancias declaradas. FP64: el mismo cálculo con otro orden de operaciones.
FP64_OUTPUT = dict(rtol=1e-12, atol=1e-13)
FP64_GRADIENT = dict(rtol=1e-10, atol=1e-12)
# FP32: el error del último token frente a FP64 no supera el del camino completo por más de
# este factor (más un suelo absoluto para errores casi nulos de los dos caminos).
FP32_FACTOR, FP32_FLOOR = 4.0, 2e-7


def encoder(*, layers=1, heads=4, hidden=32, context=16, multiplier=2, seed=7, **options):
    torch.manual_seed(seed)
    return CompactPriceTransformer(
        5,
        context=context,
        hidden_size=hidden,
        layers=layers,
        heads=heads,
        feedforward_multiplier=multiplier,
        **options,
    )


def gradients(model, output, prices):
    return torch.autograd.grad(output.square().sum(), [prices, *model.parameters()])


ARCHITECTURES = [
    dict(layers=1, heads=4),
    dict(layers=2, heads=4),
    dict(layers=1, heads=1, hidden=64),
    dict(layers=2, heads=8, hidden=64, multiplier=4),
]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("training", [False, True])
def test_last_token_matches_the_full_sequence_in_float64(architecture, training):
    model = encoder(**architecture).double().train(training)
    prices = torch.randn(6, 16, 5, dtype=torch.float64, requires_grad=True)
    full = model.encode_sequence(prices)[:, -1]
    last = model(prices)
    torch.testing.assert_close(last, full, **FP64_OUTPUT)
    for left, right in zip(
        gradients(model, last, prices), gradients(model, full, prices), strict=True
    ):
        torch.testing.assert_close(left, right, **FP64_GRADIENT)


def pytorch_layers(model, prices):
    """Referencia independiente: las capas de PyTorch en entrenamiento, sin ruta nativa."""
    mask = model.causal_mask
    encoded = model.projection(prices) + model.positions.to(prices.dtype)
    training = model.training
    model.train()
    try:
        for block in model.blocks:
            encoded = block(encoded, src_mask=mask, is_causal=True)
    finally:
        model.train(training)
    return model.norm(encoded)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("training", [False, True])
def test_explicit_layers_match_the_pytorch_encoder_layer_in_float64(architecture, training):
    model = encoder(**architecture).double().train(training)
    prices = torch.randn(5, 16, 5, dtype=torch.float64, requires_grad=True)
    expected = pytorch_layers(model, prices)
    observed = model.encode_sequence(prices)
    torch.testing.assert_close(observed, expected, **FP64_OUTPUT)
    for left, right in zip(
        gradients(model, observed, prices), gradients(model, expected, prices), strict=True
    ):
        torch.testing.assert_close(left, right, **FP64_GRADIENT)


@pytest.mark.parametrize("layers", [1, 2])
def test_inference_without_gradient_never_takes_the_native_encoder_route(layers, monkeypatch):
    calls = []
    native = torch._transformer_encoder_layer_fwd

    def record(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(torch, "_transformer_encoder_layer_fwd", record)
    model = encoder(layers=layers).eval()
    prices = torch.randn(4, 16, 5)
    with torch.no_grad():
        model(prices), model.encode_sequence(prices)
    assert calls == []


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_float32_error_against_float64_is_not_worse_than_the_full_sequence(architecture):
    model = encoder(**architecture).train()
    reference = deepcopy(model).double()
    errors = dict(full=[], last=[])
    for seed in range(4):
        generator = torch.Generator().manual_seed(seed)
        prices = torch.randn(8, 16, 5, generator=generator)
        exact_input = prices.double().requires_grad_()
        exact = reference.encode_sequence(exact_input)[:, -1]
        exact_gradients = gradients(reference, exact, exact_input)
        for name, function in (
            ("full", lambda x: model.encode_sequence(x)[:, -1]),
            ("last", model),
        ):
            values = prices.clone().requires_grad_()
            output = function(values)
            observed = gradients(model, output, values)
            output_error = (output.double() - exact).abs().max().item()
            gradient_error = max(
                ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()
                for a, b in zip(observed, exact_gradients, strict=True)
                if b.norm() > 1e-9
            )
            errors[name].append((output_error, gradient_error))
    for index in range(2):
        full = max(error[index] for error in errors["full"])
        last = max(error[index] for error in errors["last"])
        assert last <= FP32_FACTOR * full + FP32_FLOOR, (index, last, full)


def test_last_token_ignores_dropout_rng_and_keeps_the_causal_prefix():
    model = encoder(layers=2).train()
    prices = torch.randn(4, 16, 5)
    changed = prices.clone()
    changed[:, :-1] = torch.randn_like(changed[:, :-1])
    state = torch.get_rng_state().clone()
    first, second = model(prices), model(prices)
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    # El último token depende de toda la ventana: cambiar el pasado lo cambia.
    assert not torch.allclose(model(changed), first)


@pytest.mark.parametrize("position", [0, 7, 15])
def test_every_position_of_the_window_reaches_the_last_token(position):
    model = encoder(layers=1).double()
    prices = torch.randn(3, 16, 5, dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(model(prices).sum(), prices)[0]
    assert derivative[:, position].abs().sum() > 0


def test_default_contract_is_unchanged_and_explicit_batch_extends_the_budget():
    default = encoder(context=64, hidden=64)
    configuration = default.configuration
    assert configuration["max_batch"] == 256
    assert configuration["max_attention_elements"] == 2**24
    assert default.max_batch == CompactPriceTransformer.max_batch == 256
    assert encoder(context=64, hidden=64, max_batch=256).configuration == configuration
    wide = encoder(context=64, hidden=64, max_batch=1024)
    assert wide.configuration == {
        **configuration,
        "max_batch": 1024,
        "max_attention_elements": 2**26,
    }
    assert attention_budget(256) == attention_budget(1) == 2**24
    assert attention_budget(MAX_BATCH_LIMIT) == 2**24 // 256 * MAX_BATCH_LIMIT
    validate_attention_budget(1024, context=64, heads=4, layers=2, max_batch=1024)
    with pytest.raises(ValueError, match="lote"):
        validate_attention_budget(1025, context=64, heads=4, layers=1, max_batch=1024)
    with pytest.raises(ValueError, match="lote"):
        validate_attention_budget(257, context=64, heads=4, layers=1)
    with pytest.raises(ValueError, match="presupuesto conjunto"):
        validate_attention_budget(1024, context=512, heads=8, layers=2, max_batch=1024)


def test_wider_batch_runs_and_the_default_rejects_it():
    wide, default = encoder(max_batch=512), encoder()
    default.load_state_dict({**wide.state_dict(), "_extra_state": default.configuration})
    prices = torch.randn(512, 16, 5)
    assert wide(prices).shape == (512, 32)
    with pytest.raises(ValueError, match="lote"):
        default(prices)
    with pytest.raises(ValueError, match="contrato"):
        default.load_state_dict(wide.state_dict())


@pytest.mark.parametrize("value", [0, MAX_BATCH_LIMIT + 1, 256.0, True, None])
def test_invalid_batch_limits_fail_before_using_rng(value):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="lote máximo"):
        CompactPriceTransformer(5, context=16, hidden_size=32, max_batch=value)
    assert torch.equal(before, torch.get_rng_state())


def test_multimodal_reference_passes_and_restores_its_batch_limit():
    model = MultimodalReference("transformer", DIMENSIONS, context=8, max_batch=512)
    assert model.price_encoder.max_batch == 512
    configuration = model.configuration
    assert configuration["max_batch"] == 512
    arguments = dict(configuration)
    arguments.pop("price_encoder_contract")
    restored = MultimodalReference(**arguments)
    restored.load_state_dict(model.state_dict())
    plain = MultimodalReference("transformer", DIMENSIONS, context=8)
    assert "max_batch" not in plain.configuration
    with pytest.raises(ValueError, match="Transformer"):
        MultimodalReference("gru", DIMENSIONS, context=8, max_batch=512)


@pytest.mark.parametrize(
    ("kind", "batch", "expected"),
    [
        ("transformer", 256, {}),
        ("transformer", 1, {}),
        ("transformer", 257, {"max_batch": 257}),
        ("transformer", 512, {"max_batch": 512}),
        ("gru", 512, {}),
        ("dlinear", 4096, {}),
    ],
)
def test_batch_option_only_changes_the_contract_above_the_default(kind, batch, expected):
    assert transformer_batch_options(kind, batch) == expected
    model = MultimodalReference(
        kind, DIMENSIONS, context=8, **transformer_batch_options(kind, batch)
    )
    if kind == "transformer":
        assert model.price_encoder.max_batch == max(batch, 256)


def test_reference_run_accepts_wide_transformer_batches(monkeypatch):
    from mars_titan.training import reference_run

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    case = dict(
        kind="transformer",
        loss="mae",
        learning_rate=1e-3,
        seed=42,
        epochs=1,
        huber_delta=0.01,
        architecture=dict(
            hidden_size=64,
            layers=2,
            dropout=0.0,
            transformer=dict(heads=4, feedforward_multiplier=2),
        ),
    )
    reference_run._options(case, 512, 60, 0)
    with pytest.raises(ValueError):
        reference_run._options(case, 4097, 60, 0)


def test_grouped_checks_report_the_first_failure_in_the_original_order():
    values = [torch.ones(3), torch.tensor([1.0, float("nan")]), torch.tensor([float("inf")])]
    checks = list(zip(values, ("primero", "segundo", "tercero"), strict=True))
    assert first_nonfinite(checks) == "segundo"
    assert first_nonfinite(checks[:1]) is None
    assert first_nonfinite([]) is None
    assert first_nonfinite(checks[2:]) == "tercero"


@pytest.mark.parametrize(
    ("modalities", "expected"),
    [
        (("news", "prices"), "modalidades"),
        (("prices",), "precios"),
        (("macro",), "modalidades"),
    ],
)
def test_multimodal_transformer_keeps_the_message_of_the_first_nonfinite_check(
    modalities, expected
):
    model = MultimodalReference("transformer", DIMENSIONS, context=8, layers=2)
    values = {
        name: torch.randn((2, 8, size) if name == "prices" else (2, size))
        for name, size in DIMENSIONS.items()
    }
    for name in modalities:
        values[name].view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match=expected):
        model(values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sin dispositivo CUDA")
def test_transformer_reference_forward_waits_for_the_gpu_once():
    model = MultimodalReference("transformer", DIMENSIONS, context=8, layers=2).cuda()
    values = {
        name: torch.randn((4, 8, size) if name == "prices" else (4, size), device="cuda")
        for name, size in DIMENSIONS.items()
    }
    model(values)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode(1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model(values)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert sum("synchroniz" in str(item.message) for item in caught) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sin dispositivo CUDA")
@pytest.mark.parametrize("layers", [1, 2])
def test_cuda_last_token_matches_float64_like_the_full_sequence(layers):
    model = encoder(layers=layers, hidden=64, context=64).cuda().train()
    reference = deepcopy(model).double()
    prices = torch.randn(256, 64, 5, device="cuda")
    exact = reference.encode_sequence(prices.double())[:, -1]
    torch.testing.assert_close(reference(prices.double()), exact, **FP64_OUTPUT)
    full = (model.encode_sequence(prices)[:, -1].double() - exact).abs().max().item()
    last = (model(prices).double() - exact).abs().max().item()
    assert last <= FP32_FACTOR * full + FP32_FLOOR


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sin dispositivo CUDA")
@pytest.mark.parametrize("layers", [1, 2])
def test_cuda_inference_without_gradient_matches_training_mode(layers):
    # La ruta nativa de PyTorch usaría GELU con tanh en evaluación: difiere en torno a 1e-4.
    model = encoder(layers=layers, hidden=64, context=64).cuda()
    prices = torch.randn(256, 64, 5, device="cuda")
    trained = model.train()(prices)
    with torch.no_grad():
        evaluated = model.eval()(prices)
    torch.testing.assert_close(evaluated, trained.detach(), rtol=0, atol=0)


def test_module_level_constants_keep_the_previous_defaults():
    assert module._MAX_BATCH == 256
    assert module._MAX_ATTENTION_ELEMENTS == 2**24
