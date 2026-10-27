"""Contratos del ajuste Dust experimental, sin pasos de optimizador."""

import importlib
import json

import pytest
import torch
from test_financial_adapter import raw_batch, specification

from mars_titan.models.titans.auxiliary_horizons import AuxiliaryHorizons
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.models.titans.laplace_scale import LaplaceScaleHead

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
DTYPES = [torch.float32, torch.float64]


def dust_api():
    try:
        return importlib.import_module("mars_titan.training.dust")
    except ModuleNotFoundError:
        pytest.fail("Falta el adaptador financiero completo de Dust")


def financial_case(*, device="cpu", dtype=torch.float64, layers=1, **changes):
    config = FinancialConfig(
        specification(),
        hidden_size=32,
        layers=layers,
        seed=42,
        max_batch=8,
        **changes,
    )
    model = FinancialPredictor(config, device=device, dtype=dtype).eval()
    batch = DecisionBatch.from_corpus(raw_batch(), config.inputs, device=device, dtype=dtype)
    return model, batch


def snapshots(model):
    return {
        name: (id(value), value._version, value.requires_grad, value.detach().clone())
        for name, value in (*model.named_parameters(), *model.named_buffers())
    }


def assert_unchanged(model, before):
    current = dict((*model.named_parameters(), *model.named_buffers()))
    assert current.keys() == before.keys()
    for name, (identity, version, requires_grad, expected) in before.items():
        value = current[name]
        assert (id(value), value._version, value.requires_grad) == (
            identity,
            version,
            requires_grad,
        )
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    model.verify_parameter_identity()


def estimator_for(model, *, loss="mse", **changes):
    module = dust_api()
    settings = dict(
        population=4,
        chunk_size=2,
        sigma=0.01,
        seed=73,
        max_forward_calls=256,
        max_workspace_bytes=64 * 1024**2,
    )
    settings.update(changes)
    adapter = module.FinancialDustAdapter(model)
    return module.DustEstimator(adapter, module.DustConfig(**settings), loss=loss)


def test_direct_adapter_keeps_every_parameter_and_the_default_forward():
    module = dust_api()
    model, batch = financial_case()
    before = snapshots(model)
    adapter = module.FinancialDustAdapter(model)
    expected = model.prepare(batch, model.initial_state(batch.flow_ids), differentiable=True)
    actual = adapter.forward(batch.inputs, batch.presence)
    torch.testing.assert_close(actual, expected.point_predictions, rtol=0, atol=0)
    owned = dict(model.named_parameters())
    exposed = dict(adapter.named_parameters())
    assert exposed.keys() == owned.keys()
    assert all(exposed[name] is value for name, value in owned.items())
    assert len(exposed) == 28
    assert sum(value.numel() for value in exposed.values()) == 14_593
    identity = adapter.identity()
    assert identity["experimental"] is True
    assert identity["kind"] == "dust_financial_transformer_direct_scalar_v1"
    assert identity["trainable_scope"] == "full_trainable_predictor_only"
    assert identity["parameter_names"] == list(owned)
    json.dumps(identity, allow_nan=False)
    assert_unchanged(model, before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layers", [1, 2])
def test_instrumented_clean_core_matches_prepare_values_and_all_bp_gradients(device, dtype, layers):
    module = dust_api()
    model, batch = financial_case(device=device, dtype=dtype, layers=layers)
    adapter = module.FinancialDustAdapter(model)
    prepared = model.prepare(batch, model.initial_state(batch.flow_ids), differentiable=True)
    clean = adapter.forward(batch.inputs, batch.presence)
    instrumented = adapter.instrumented_forward(batch.inputs, batch.presence)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(clean, prepared.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(instrumented, clean, rtol=tolerance, atol=tolerance)
    parameters = tuple(model.parameters())
    target = torch.tensor([0.2, -0.1], dtype=dtype, device=device)
    reference = torch.autograd.grad((clean - target).square().mean(), parameters)
    actual = torch.autograd.grad((instrumented - target).square().mean(), parameters)
    for expected, value in zip(reference, actual, strict=True):
        torch.testing.assert_close(value, expected, rtol=5 * tolerance, atol=5 * tolerance)


@pytest.mark.parametrize(
    "changes",
    [
        dict(variant="mac_disabled"),
        dict(variant="mac_frozen"),
        dict(variant="mac_online"),
        dict(head="quantile_head_v1"),
        dict(presence_modulation="film_v1"),
        dict(presence_modulation="wide_fusion_v1"),
        dict(laplace_scale_head=LaplaceScaleHead()),
        dict(auxiliary_horizons=AuxiliaryHorizons()),
    ],
)
def test_unsupported_models_are_rejected_without_changing_the_predictor(changes):
    module = dust_api()
    model, _ = financial_case(**changes)
    before = snapshots(model)
    with pytest.raises(ValueError):
        module.FinancialDustAdapter(model)
    assert_unchanged(model, before)


def test_an_uncovered_parameter_cannot_silently_receive_a_zero_estimate():
    module = dust_api()
    model, _ = financial_case()
    model.register_parameter("not_a_dust_site", torch.nn.Parameter(torch.ones(2)))
    model._seal_parameters()
    before = snapshots(model)
    with pytest.raises(ValueError):
        module.FinancialDustAdapter(model)
    assert_unchanged(model, before)


@pytest.mark.parametrize("training", [True, False])
def test_nonzero_dropout_is_rejected_at_admission_without_consuming_parent_rng(training):
    module = dust_api()
    model, _ = financial_case()
    model.fusion[2].p = 0.5
    model.train(training)
    before = snapshots(model)
    parent_rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="Dropout|dropout|arquitectura|contrato"):
        module.FinancialDustAdapter(model)
    assert torch.equal(torch.random.get_rng_state(), parent_rng)
    assert_unchanged(model, before)


@pytest.mark.parametrize("scope", ["head_bias", "all_parameters"])
def test_parameter_freeze_is_rejected_before_admission_without_changing_the_model(scope):
    module = dust_api()
    model, _ = financial_case()
    if scope == "head_bias":
        model.head.bias.requires_grad_(False)
    else:
        model.requires_grad_(False)
    before = snapshots(model)
    parent_rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="entrenable|congelad|ajuste"):
        module.FinancialDustAdapter(model)
    assert torch.equal(torch.random.get_rng_state(), parent_rng)
    assert_unchanged(model, before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["linear", "layer_norm"])
def test_local_pullbacks_include_bias_and_normalization_scale(device, kind):
    module = importlib.import_module("mars_titan.training.dust.financial")
    generator = torch.Generator(device=device).manual_seed(291)
    values = torch.randn(2, 3, 4, generator=generator, device=device, dtype=torch.float64)
    if kind == "linear":
        weight = torch.randn(5, 4, generator=generator, device=device, dtype=torch.float64)
        bias = torch.randn(5, generator=generator, device=device, dtype=torch.float64)
        weight.requires_grad_()
        bias.requires_grad_()
        output = torch.nn.functional.linear(values, weight, bias)
        pullback = module.linear_pullback
        local_input = values
    else:
        weight = torch.randn(4, generator=generator, device=device, dtype=torch.float64)
        bias = torch.randn(4, generator=generator, device=device, dtype=torch.float64)
        weight.requires_grad_()
        bias.requires_grad_()
        output = torch.nn.functional.layer_norm(values, (4,), weight, bias, eps=1e-5)
        local_input = torch.nn.functional.layer_norm(values, (4,), eps=1e-5)
        pullback = module.layer_norm_pullback
    error = torch.randn(output.shape, generator=generator, device=device, dtype=torch.float64)
    expected = torch.autograd.grad((output * error).sum(), (weight, bias))
    actual = pullback(local_input, error)
    assert len(actual) == 2
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_estimate_covers_qkv_norms_and_all_biases_without_autograd_or_effects(device, monkeypatch):
    model, batch = financial_case(device=device)
    estimator = estimator_for(model)
    before = snapshots(model)
    parent_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device != "cpu" else None
    target = torch.tensor([0.2, -0.1], dtype=torch.float64, device=device)

    def forbidden(*args, **kwargs):
        raise AssertionError("La estimación Dust no puede usar autograd")

    with monkeypatch.context() as patch:
        patch.setattr(torch.autograd, "grad", forbidden)
        patch.setattr(torch.Tensor, "backward", forbidden)
        result = estimator.estimate(batch.inputs, batch.presence, target)
    expected = dict(model.named_parameters())
    assert result.gradients.keys() == expected.keys()
    for name, parameter in expected.items():
        value = result.gradients[name]
        assert value.shape == parameter.shape
        assert value.dtype == parameter.dtype
        assert value.device == parameter.device
        assert not value.requires_grad
        assert torch.isfinite(value).all()
        assert parameter.grad is None
    for name in (
        "price_encoder.blocks.0.self_attn.in_proj_weight",
        "price_encoder.blocks.0.self_attn.in_proj_bias",
        "price_encoder.blocks.0.norm1.weight",
        "price_encoder.blocks.0.norm1.bias",
        "encoders.news.0.bias",
        "head.weight",
        "head.bias",
    ):
        assert torch.count_nonzero(result.gradients[name]) > 0
    reference = (estimator.adapter.forward(batch.inputs, batch.presence) - target).square().mean()
    torch.testing.assert_close(result.loss, reference.detach(), rtol=0, atol=0)
    assert result.counters["full_forward_calls"] == 28
    assert result.counters["local_attention_calls"] == 7
    assert result.counters["head_loss_calls"] == 2
    assert result.counters["full_forward_rows"] == 53 * len(target) + len(target)
    assert_unchanged(model, before)
    assert torch.equal(torch.random.get_rng_state(), parent_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)


def test_population_centering_uses_all_draws_and_the_unbiased_scale():
    module = importlib.import_module("mars_titan.training.dust.estimator")
    rewards = torch.tensor([[1.0, 9.0], [2.0, 3.0], [4.0, 2.0], [7.0, -1.0]])
    expected = (rewards - rewards.mean(dim=0, keepdim=True)) * (4 / 3)
    actual = module.center_rewards(rewards)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.sum(dim=0), torch.zeros(2), rtol=0, atol=2e-6)


@pytest.mark.parametrize("chunk", [1, 2])
def test_closed_scalar_head_credit_preserves_sign_population_and_centering_scale(
    chunk, monkeypatch
):
    model, _ = financial_case()
    estimator = estimator_for(model, sigma=0.1, chunk_size=chunk)
    directions = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float64).reshape(4, 1, 1)
    monkeypatch.setattr(estimator, "_noise", lambda _: directions.clone())
    counters = dict(head_loss_calls=0)
    actual = estimator._head_error(
        torch.zeros(1, 1, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        counters,
    )
    # Pérdidas 1,21/0,81 y recompensas centradas ±0,2 con corrección 4/3.
    torch.testing.assert_close(
        actual, torch.tensor([[-8 / 3]], dtype=torch.float64), rtol=1e-12, atol=1e-12
    )
    assert counters["head_loss_calls"] == 4 // chunk


def test_private_rng_checkpoint_reproduces_the_next_estimate_exactly():
    model, batch = financial_case()
    first = estimator_for(model)
    target = torch.tensor([0.2, -0.1], dtype=torch.float64)
    first.estimate(batch.inputs, batch.presence, target)
    saved = first.state_dict()
    expected = first.estimate(batch.inputs, batch.presence, target)
    restored = estimator_for(model)
    restored.load_state_dict(saved)
    actual = restored.estimate(batch.inputs, batch.presence, target)
    for name, value in expected.gradients.items():
        torch.testing.assert_close(actual.gradients[name], value, rtol=0, atol=0)
    assert actual.counters == expected.counters
    assert first.totals["full_forward_rows"] == restored.totals["full_forward_rows"] == 216
    assert first.totals["logical_rows"] == restored.totals["logical_rows"] == 4
    assert first.identity()["centering"]["correction"] == "population_over_population_minus_one"
    changed = estimator_for(model, sigma=0.02)
    with pytest.raises(ValueError):
        changed.load_state_dict(saved)


def test_probe_failure_restores_private_rng_counters_weights_and_pending_gradients(monkeypatch):
    model, batch = financial_case()
    model.head.bias.grad = torch.full_like(model.head.bias, 7.0)
    estimator = estimator_for(model)
    before = estimator.state_dict()
    weights = snapshots(model)
    target = torch.tensor([0.2, -0.1], dtype=torch.float64)

    def fail_after_noise(*args, **kwargs):
        assert not torch.equal(estimator.generator.get_state(), before["rng"])
        raise RuntimeError("Fallo de un probe después de generar ruido")

    with monkeypatch.context() as patch:
        patch.setattr(estimator.adapter, "instrumented_forward", fail_after_noise)
        with pytest.raises(RuntimeError, match="después de generar ruido"):
            estimator.estimate(batch.inputs, batch.presence, target)
    assert torch.equal(estimator.generator.get_state(), before["rng"])
    assert estimator.steps == before["steps"]
    assert estimator.totals == before["totals"]
    assert_unchanged(model, weights)
    for name, parameter in model.named_parameters():
        if name == "head.bias":
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7.0))
        else:
            assert parameter.grad is None
    fresh = estimator_for(model)
    actual = estimator.estimate(batch.inputs, batch.presence, target)
    expected = fresh.estimate(batch.inputs, batch.presence, target)
    for name, value in expected.gradients.items():
        torch.testing.assert_close(actual.gradients[name], value, rtol=0, atol=0)
    assert actual.counters == expected.counters


@pytest.mark.parametrize("schema", [True, 1.0])
def test_checkpoint_schema_requires_an_integer_before_replacing_private_rng(schema):
    model, _ = financial_case()
    estimator = estimator_for(model)
    different = estimator_for(model)
    torch.randn(3, generator=different.generator)
    supplied = different.state_dict()
    supplied["schema_version"] = schema
    before = estimator.state_dict()
    assert not torch.equal(supplied["rng"], before["rng"])
    with pytest.raises(ValueError):
        estimator.load_state_dict(supplied)
    assert torch.equal(estimator.state_dict()["rng"], before["rng"])
    assert estimator.steps == before["steps"]
    assert estimator.totals == before["totals"]


@pytest.mark.parametrize("budget", ["max_forward_calls", "max_workspace_bytes"])
def test_budget_is_rejected_before_any_probe_or_rng_consumption(budget, monkeypatch):
    model, batch = financial_case()
    estimator = estimator_for(model, **{budget: 1})
    before = snapshots(model)
    state = estimator.state_dict()

    def forbidden(*args, **kwargs):
        raise AssertionError("El presupuesto debe comprobarse antes del forward")

    monkeypatch.setattr(estimator.adapter, "forward", forbidden)
    monkeypatch.setattr(estimator.adapter, "instrumented_forward", forbidden)
    with pytest.raises(ValueError, match="presupuesto"):
        estimator.estimate(batch.inputs, batch.presence, torch.zeros(2, dtype=torch.float64))
    assert torch.equal(estimator.state_dict()["rng"], state["rng"])
    assert_unchanged(model, before)


@pytest.mark.parametrize("fault", ["input", "target"])
def test_nonfinite_data_is_rejected_before_estimation_without_effects(fault):
    model, batch = financial_case()
    estimator = estimator_for(model)
    before = snapshots(model)
    state = estimator.state_dict()
    inputs = {name: value.clone() for name, value in batch.inputs.items()}
    target = torch.tensor([0.2, -0.1], dtype=torch.float64)
    if fault == "input":
        inputs["news"][0, 0] = float("nan")
    else:
        target[0] = float("inf")
    with pytest.raises(ValueError):
        estimator.estimate(inputs, batch.presence, target)
    assert torch.equal(estimator.state_dict()["rng"], state["rng"])
    assert_unchanged(model, before)


@pytest.mark.parametrize(
    "changes",
    [
        dict(population=1),
        dict(population=True),
        dict(chunk_size=0),
        dict(chunk_size=True),
        dict(sigma=0.0),
        dict(sigma=float("nan")),
        dict(sigma=float("inf")),
        dict(max_forward_calls=True),
        dict(max_workspace_bytes=0),
    ],
)
def test_configuration_rejects_invalid_population_noise_and_limits(changes):
    module = dust_api()
    with pytest.raises(ValueError):
        module.DustConfig(**changes)
