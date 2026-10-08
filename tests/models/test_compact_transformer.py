"""Contrato técnico del codificador causal, sin pasos de optimizador."""

from copy import deepcopy

import pytest
import torch

from mars_titan.models.baselines.initialization import initialize_weights
from mars_titan.models.baselines.multimodal import MultimodalReference
from mars_titan.models.baselines.transformer import CompactPriceTransformer

DIMENSIONS = dict(prices=5, news=8, charts=7, fundamentals=6, macro=9)


def reference(**options):
    settings = dict(context=8, hidden_size=32, layers=2, dropout=0.0)
    settings.update(options)
    return MultimodalReference("transformer", DIMENSIONS, **settings)


def inputs(*, batch=3, context=8, dtype=torch.float32, gradients=False):
    return {
        name: torch.randn(
            (batch, context, width) if name == "prices" else (batch, width),
            dtype=dtype,
            requires_grad=gradients,
        )
        for name, width in DIMENSIONS.items()
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transformer_fusion_and_head_receive_gradients_from_every_modality(dtype):
    torch.manual_seed(123)
    model = reference().to(dtype=dtype)
    values = inputs(dtype=dtype, gradients=True)
    predicted = model(values)
    assert predicted.shape == (3,)
    assert model.encode(values).shape == (3, 32)
    predicted.square().sum().backward()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
        for value in values.values()
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert torch.equal(predicted, model.head(model.encode(values)).squeeze(-1))


@pytest.mark.parametrize("training", [False, True])
def test_future_suffix_cannot_change_prefix_or_receive_its_gradient(training):
    torch.manual_seed(22)
    encoder = reference().price_encoder.train(training)
    prices = inputs(gradients=True)["prices"]
    changed = prices.detach().clone()
    changed[:, 4:] = 100 * torch.randn_like(changed[:, 4:])
    observed = encoder.encode_sequence(prices)
    altered = encoder.encode_sequence(changed)
    torch.testing.assert_close(observed[:, :4], altered[:, :4], rtol=0, atol=0)
    assert not torch.allclose(observed[:, -1], altered[:, -1])
    derivative = torch.autograd.grad(observed[:, :4, 0].sum(), prices)[0]
    assert torch.count_nonzero(derivative[:, 4:]) == 0
    assert derivative[:, :4].abs().sum() > 0
    torch.testing.assert_close(encoder(prices), observed[:, -1], rtol=0, atol=0)


def test_positions_distinguish_identical_observations_at_different_times():
    encoder = reference().price_encoder.eval()
    with torch.no_grad():
        sequence = encoder.encode_sequence(torch.zeros(2, 8, 5))
    assert not torch.allclose(sequence[:, 0], sequence[:, -1], rtol=0, atol=1e-6)


def test_windows_batches_and_rng_have_no_hidden_carry():
    torch.manual_seed(81)
    model = reference(dropout=0.2).eval()
    values = inputs()
    other = inputs()
    rng = torch.get_rng_state().clone()
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        expected = model(values)
        model(other)
        repeated = model(values)
        reversed_batch = model({key: value.flip(0) for key, value in values.items()})
        separate = torch.cat(
            [model({key: value[i : i + 1] for key, value in values.items()}) for i in range(3)]
        )
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, weights[name]) for name, value in model.state_dict().items())
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
    torch.testing.assert_close(reversed_batch.flip(0), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(separate, expected, rtol=1e-5, atol=1e-6)


def test_configuration_and_weights_reconstruct_the_same_predictor(tmp_path):
    model = reference(transformer=dict(heads=2, feedforward_multiplier=4)).double().eval()
    values = inputs(dtype=torch.float64)
    expected = model(values)
    path = tmp_path / "reference.pt"
    torch.save(dict(config=model.configuration, model=model.state_dict()), path)
    stored = torch.load(path, weights_only=True)
    arguments = dict(stored["config"])
    contract = arguments.pop("price_encoder_contract")
    restored = MultimodalReference(**arguments).double().eval()
    restored.load_state_dict(stored["model"], strict=True)
    assert restored.configuration["price_encoder_contract"] == contract
    assert contract["causal"] is True
    assert contract["position_encoding"] == "sinusoidal_v1"
    torch.testing.assert_close(restored(values), expected, rtol=0, atol=0)
    copied = restored.configuration
    copied["transformer"]["heads"] = 8
    assert restored.configuration["transformer"]["heads"] == 2


def test_restored_positions_do_not_depend_on_the_constructor_default_dtype():
    source = reference().double().eval()
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        restored = reference().eval()
    finally:
        torch.set_default_dtype(previous)
    restored.load_state_dict(source.state_dict())
    values = inputs(dtype=torch.float64)
    torch.testing.assert_close(restored(values), source(values), rtol=0, atol=0)


def test_causal_prefix_is_unchanged_in_inference_mode():
    encoder = reference().price_encoder.eval()
    prices = inputs()["prices"]
    changed = prices.clone()
    changed[:, 4:] *= 17
    with torch.inference_mode():
        original = encoder.encode_sequence(prices)
        observed = encoder.encode_sequence(changed)
    torch.testing.assert_close(original[:, :4], observed[:, :4], rtol=0, atol=0)


def test_autocast_is_not_silently_enabled():
    encoder = reference().price_encoder
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            encoder(torch.zeros(2, 8, 5))


def test_initialization_restores_weights_only_after_matching_resolved_contract(tmp_path):
    source, restored = reference().eval(), reference().eval()
    path = tmp_path / "source.pt"
    hashes = {"fixture": "a" * 64}
    torch.save(
        dict(
            model=source.state_dict(),
            config=source.configuration,
            input_hashes=hashes,
            next_epoch=1,
        ),
        path,
    )
    before = path.read_bytes()
    rng = torch.get_rng_state().clone()
    initialize_weights(restored, path, config=restored.configuration, hashes=hashes)
    assert torch.equal(rng, torch.get_rng_state())
    assert path.read_bytes() == before
    values = inputs()
    torch.testing.assert_close(restored(values), source(values), rtol=0, atol=0)


@pytest.mark.parametrize(
    "fault",
    [
        "heads",
        "causal",
        "boolean_type",
        "positions",
        "normalization",
        "version",
        "missing_contract",
        "float_heads",
        "false_model_contract",
        "false_kind",
        "nan_contract",
    ],
)
def test_checkpoint_contract_rejects_same_shapes_with_different_semantics(tmp_path, fault):
    source = reference()
    restored = reference(
        transformer=dict(heads=2 if fault == "heads" else 4, feedforward_multiplier=2)
    )
    source_config, config = source.configuration, restored.configuration
    if fault == "causal":
        source_config["price_encoder_contract"]["causal"] = False
    elif fault == "boolean_type":
        source_config["price_encoder_contract"]["causal"] = 1
    elif fault == "positions":
        source_config["price_encoder_contract"]["position_encoding"] = "none"
    elif fault == "normalization":
        source_config["price_encoder_contract"]["layer_norm_eps"] = 1e-4
    elif fault == "version":
        source_config["price_encoder_contract"]["schema_version"] = 2
    elif fault == "missing_contract":
        del source_config["price_encoder_contract"]
        del config["price_encoder_contract"]
    elif fault == "float_heads":
        source_config["transformer"]["heads"] = 4.0
    elif fault == "false_model_contract":
        source_config["transformer"]["heads"] = config["transformer"]["heads"] = 2
        source_config["price_encoder_contract"]["heads"] = config["price_encoder_contract"][
            "heads"
        ] = 2
    elif fault == "false_kind":
        source_config["kind"] = config["kind"] = "gru"
    elif fault == "nan_contract":
        source_config["price_encoder_contract"]["layer_norm_eps"] = float("nan")
    path = tmp_path / "source.pt"
    hashes = {"fixture": "b" * 64}
    torch.save(
        dict(model=source.state_dict(), config=source_config, input_hashes=hashes, next_epoch=1),
        path,
    )
    original = deepcopy(restored.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="contrato|arquitectura"):
        initialize_weights(restored, path, config=config, hashes=hashes)
    assert all(torch.equal(value, original[name]) for name, value in restored.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "options",
    [
        dict(context=True),
        dict(context=1),
        dict(context=513),
        dict(hidden_size=16),
        dict(layers=3),
        dict(transformer={}),
        dict(transformer=dict(heads=True, feedforward_multiplier=2)),
        dict(transformer=dict(heads=3, feedforward_multiplier=2)),
        dict(transformer=dict(heads=4, feedforward_multiplier=1)),
        dict(transformer=dict(heads=4, feedforward_multiplier=2, causal=False)),
    ],
)
def test_invalid_configuration_is_rejected_without_consuming_rng(options):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        reference(**options)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("kind", ["gru", "rnn", "lstm", "dlinear"])
def test_previous_encoders_reject_transformer_options_before_initialization(kind):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Transformer|transformer"):
        MultimodalReference(kind, DIMENSIONS, transformer=dict(heads=4, feedforward_multiplier=2))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.int64, torch.bool, torch.complex64]
)
def test_price_encoder_rejects_unsupported_dtypes(dtype):
    encoder = reference().price_encoder
    with pytest.raises(ValueError, match="tipo|dtype|precisión"):
        encoder(torch.ones(2, 8, 5, dtype=dtype))


@pytest.mark.parametrize("modality", list(DIMENSIONS))
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_inputs_cannot_become_predictions(modality, invalid):
    model = reference()
    values = inputs()
    values[modality].view(-1)[0] = invalid
    with pytest.raises(ValueError, match="finit|valor"):
        model(values)


@pytest.mark.parametrize("modality", list(DIMENSIONS))
def test_mixed_input_dtypes_fail_explicitly(modality):
    model = reference()
    values = inputs()
    values[modality] = values[modality].double()
    with pytest.raises(ValueError, match="tipo|dtype|precisión"):
        model(values)


@pytest.mark.parametrize("shape", [(0, 8, 5), (3, 7, 5), (3, 8, 4), (8, 5)])
def test_malformed_price_windows_are_rejected(shape):
    encoder = reference().price_encoder
    with pytest.raises(ValueError, match="forma|ventana|lote"):
        encoder(torch.zeros(shape))


def test_batch_limit_does_not_enter_attention(monkeypatch):
    encoder = reference().price_encoder

    def forbidden(*args, **kwargs):
        pytest.fail("No debe calcular atención de un lote fuera del presupuesto")

    monkeypatch.setattr(encoder.blocks[0], "forward", forbidden)
    with pytest.raises(ValueError, match="lote|presupuesto"):
        encoder(torch.zeros(257, 8, 5))


def test_joint_attention_limit_rejects_large_context_before_computation(monkeypatch):
    encoder = reference(
        context=512, transformer=dict(heads=8, feedforward_multiplier=2)
    ).price_encoder

    def forbidden(*args, **kwargs):
        pytest.fail("No debe calcular atención fuera del presupuesto conjunto")

    monkeypatch.setattr(encoder.blocks[0], "forward", forbidden)
    with pytest.raises(ValueError, match="presupuesto"):
        encoder(torch.zeros(8, 512, 5))


def test_non_tensor_and_device_mismatch_fail_at_the_contract():
    model = reference()
    values = inputs()
    with pytest.raises(ValueError, match="tensor|tipo"):
        model({**values, "news": [1, 2]})
    with pytest.raises(ValueError, match="dispositivo"):
        model({**values, "prices": torch.empty(3, 8, 5, device="meta")})


@pytest.mark.parametrize(
    "options",
    [
        dict(input_size=True),
        dict(input_size=0),
        dict(input_size=2049),
        dict(context=1),
        dict(context=513),
        dict(hidden_size=16),
        dict(layers=0),
    ],
)
def test_standalone_encoder_rejects_invalid_dimensions_before_using_rng(options):
    arguments = dict(input_size=5, context=8, hidden_size=32, layers=1)
    arguments.update(options)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="forma|arquitectura"):
        CompactPriceTransformer(**arguments)
    assert torch.equal(before, torch.get_rng_state())


def test_transformer_is_not_implicitly_admitted_to_the_campaign_design():
    from mars_titan.training.reference_design import design_cases

    with pytest.raises(ValueError, match="familias"):
        design_cases(["transformer"])
