"""Escala de Laplace con β-NLL y β = 1 (F1 de #749), sin pasos de optimizador."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats
from test_financial_adapter import api, raw_batch, setup
from torch.nn import functional

from mars_titan.models.quantile_head import LEVELS, QUANTILE_HEAD
from mars_titan.models.titans.laplace_scale import (
    LAPLACE_SCALE_HEAD,
    LaplaceScale,
    LaplaceScaleHead,
    laplace_beta_nll,
    laplace_quantiles,
)
from mars_titan.models.titans.session_ic_objective import SessionICObjective
from mars_titan.training.search_cases import checked_predictor_changes

HEAD = LaplaceScaleHead(floor=1e-6, initial_scale=0.02)


def rows(dtype=torch.float64, count=7, seed=3):
    generator = torch.Generator().manual_seed(seed)
    point = torch.randn(count, generator=generator, dtype=dtype) * 0.02
    scale = torch.rand(count, generator=generator, dtype=dtype) * 0.03 + 0.005
    target = torch.randn(count, generator=generator, dtype=dtype) * 0.02
    return point, scale, target


def gradients(point, scale, target):
    point, scale = point.clone().requires_grad_(), scale.clone().requires_grad_()
    laplace_beta_nll(torch.stack((point, scale), dim=-1), target).backward()
    return point.grad, scale.grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_both_gradients_match_their_closed_form(dtype):
    point, scale, target = rows(dtype)
    # Una fila con error nulo comprueba también el subgradiente cero del valor absoluto.
    point[2] = target[2]
    d_point, d_scale = gradients(point, scale, target)
    count = len(target)
    # ∂L/∂μ = sign(μ − y), exactamente el del error absoluto, sin redondeo de b/b.
    assert torch.equal(d_point, torch.sign(point - target) / count)
    control = point.clone().requires_grad_()
    functional.l1_loss(control, target).backward()
    assert torch.equal(d_point, control.grad)
    # ∂L/∂b = sg(b) · (1/b) · (1 − |y − μ|/b).
    error = (target - point).abs()
    expected = scale * (1 / scale) * (1 - error / scale) / count
    tolerance = dict(rtol=1e-12, atol=0) if dtype == torch.float64 else dict(rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(d_scale, expected, **tolerance)


def test_scale_gradient_is_that_of_the_weighted_nll_with_the_weight_held_fixed():
    point, scale, target = rows()
    weight = scale.clone()

    def weighted(mu, b):
        return (weight * ((target - mu).abs() / b + torch.log(b))).mean()

    inputs = (point.clone().requires_grad_(), scale.clone().requires_grad_())
    # La pérdida con el peso como constante sí es una función ordinaria y gradcheck la acepta.
    assert torch.autograd.gradcheck(weighted, inputs, eps=1e-8, atol=1e-7)
    reference = torch.autograd.grad(weighted(*inputs), inputs)
    for value, expected in zip(gradients(point, scale, target), reference, strict=True):
        torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-15)


def test_value_is_the_detached_scale_times_the_laplace_nll_without_log_two():
    point, scale, target = rows()
    value = laplace_beta_nll(torch.stack((point, scale), dim=-1), target)
    nll = -stats.laplace.logpdf(target.numpy(), loc=point.numpy(), scale=scale.numpy())
    expected = np.mean(scale.numpy() * (nll - np.log(2)))
    assert value.item() == pytest.approx(expected, rel=1e-12)


def test_the_scale_gradient_vanishes_at_the_mean_absolute_deviation():
    point, _, target = rows(count=64)
    optimum = (target - point).abs().mean()
    scale = torch.full_like(target, optimum.item())
    # Con una escala común, la suma de ∂L/∂b se anula en b = media de |y − μ|.
    assert gradients(point, scale, target)[1].sum().abs() < 1e-14
    assert gradients(point, scale * 1.1, target)[1].sum() > 0
    assert gradients(point, scale * 0.9, target)[1].sum() < 0


def test_loss_rejects_malformed_outputs():
    point, scale, target = rows()
    with pytest.raises(ValueError, match="mediana y escala"):
        laplace_beta_nll(torch.stack((point, scale, scale), dim=-1), target)
    with pytest.raises(ValueError, match="mediana y escala"):
        laplace_beta_nll(torch.stack((point, scale), dim=-1).float(), target)
    with pytest.raises(ValueError, match="mediana y escala"):
        laplace_beta_nll(torch.zeros(0, 2, dtype=torch.float64), target[:0])


def test_quantiles_are_those_of_the_laplace_and_keep_the_median():
    point, scale, _ = rows(torch.float32)
    levels = laplace_quantiles(point, scale)
    assert levels.shape == (len(point), len(LEVELS)) and levels.dtype == torch.float32
    assert torch.equal(levels[:, 2], point)
    assert (levels.diff(dim=-1) >= 0).all()
    expected = stats.laplace.ppf(
        np.asarray(LEVELS),
        loc=point.double().numpy()[:, None],
        scale=scale.double().numpy()[:, None],
    )
    np.testing.assert_allclose(levels.double().numpy(), expected, rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError, match="mediana y una escala"):
        laplace_quantiles(point, scale[:-1])


@pytest.mark.parametrize(
    "options",
    [
        dict(floor=0.0),
        dict(floor=1e-6, initial_scale=1.5e-6),
        dict(initial_scale=2.0),
        dict(floor=True),
        dict(initial_scale=float("nan")),
        dict(floor=0.05),
    ],
)
def test_invalid_configurations_are_refused(options):
    with pytest.raises(ValueError):
        LaplaceScaleHead(**options)


def test_scale_starts_at_the_declared_value_without_global_rng():
    before = torch.random.get_rng_state().clone()
    module = LaplaceScale(8, HEAD, dtype=torch.float64)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.count_nonzero(module.weight) == 0
    assert not module.session_level and module.row_decomposable
    point = torch.randn(5, dtype=torch.float64)
    outputs, emitted = module(point, torch.randn(5, 8, dtype=torch.float64), ["US"] * 5)
    assert emitted is point and torch.equal(outputs[:, 0], point)
    torch.testing.assert_close(
        outputs[:, 1], torch.full((5,), 0.02, dtype=torch.float64), rtol=1e-14, atol=0
    )
    assert torch.equal(module.levels(outputs), laplace_quantiles(point, outputs[:, 1]))
    with pytest.raises(ValueError, match="configuración de Laplace"):
        LaplaceScale(8, asdict(HEAD))


def test_declaration_travels_as_a_predictor_change_with_its_own_identity():
    base, _ = setup()
    options = dict(laplace_scale_head=asdict(HEAD))
    assert checked_predictor_changes(options) == options
    changed = replace(base.config, **options)
    assert changed.laplace_scale_head == HEAD
    assert "laplace_scale_head" not in base.config.identity()
    identity = changed.identity()
    # F1 es un objetivo de salida, como el de IC por sesión, y se identifica igual.
    assert identity["output"] == LAPLACE_SCALE_HEAD
    assert identity["output_objective"] == HEAD.identity()
    reference = base.config.identity()
    assert reference["output"] != LAPLACE_SCALE_HEAD and "output_objective" not in reference
    assert {k: v for k, v in identity.items() if k not in ("output", "output_objective")} == {
        k: v for k, v in reference.items() if k != "output"
    }
    with pytest.raises(ValueError, match="toda su configuración"):
        replace(base.config, laplace_scale_head=dict(floor=1e-6))
    with pytest.raises(ValueError, match="cabeza escalar"):
        replace(base.config, head=QUANTILE_HEAD, laplace_scale_head=HEAD)
    with pytest.raises(ValueError, match="sus valores o None"):
        replace(base.config, laplace_scale_head="laplace_scale_head_v1")
    with pytest.raises(ValueError, match="excluyentes"):
        replace(base.config, laplace_scale_head=HEAD, session_ic_objective=SessionICObjective())
    with pytest.raises(ValueError):
        checked_predictor_changes(dict(laplace_scale_head={}))


@pytest.mark.parametrize("variant", ["transformer_direct", "mac_online"])
def test_disabled_head_is_the_reference_and_enabled_head_keeps_its_median(variant):
    control, batch = setup(variant)
    reference, _ = setup(variant)
    assert reference.config == control.config and reference.output_objective is None
    assert reference._parameter_id == control._parameter_id
    before = torch.random.get_rng_state().clone()
    model = api().FinancialPredictor(
        replace(control.config, laplace_scale_head=HEAD), dtype=torch.float64
    )
    assert torch.equal(before, torch.random.get_rng_state())
    shared = dict(model.named_parameters())
    names = set(shared) - {"output_objective.weight", "output_objective.bias"}
    assert names == set(dict(control.named_parameters()))
    for name, value in control.named_parameters():
        assert torch.equal(value, shared[name]), name
    expected = control.prepare(batch, control.initial_state(batch.flow_ids))
    result = model.prepare(batch, model.initial_state(batch.flow_ids))
    assert expected.outputs is None
    # La mediana no cambia al añadir la escala, que empieza en su valor declarado.
    assert torch.equal(result.point_predictions, expected.point_predictions)
    assert torch.equal(result.outputs[:, 0], result.point_predictions)
    torch.testing.assert_close(result.outputs[:, 1], torch.full((2,), 0.02, dtype=torch.float64))
    assert result.quantiles is None


def laplace_and_control_gradients(scale_weight=None):
    control, batch = setup()
    model = api().FinancialPredictor(
        replace(control.config, laplace_scale_head=HEAD), dtype=torch.float64
    )
    if scale_weight is not None:
        # Estado posterior simulado sin optimizador: la escala ya depende del estado codificado.
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            model.output_objective.weight.normal_(0.0, scale_weight, generator=generator)
        model._seal_parameters()
    target = torch.tensor([0.01, -0.03], dtype=torch.float64)
    result = model.prepare(batch, model.initial_state(batch.flow_ids), differentiable=True)
    model.output_objective.loss(result.outputs, target, None).backward()
    expected = control.prepare(batch, control.initial_state(batch.flow_ids), differentiable=True)
    functional.l1_loss(expected.point_predictions, target).backward()
    grads = {name: value.grad for name, value in model.named_parameters()}
    references = {name: value.grad for name, value in control.named_parameters()}
    return grads, references


@pytest.mark.parametrize("scale_weight", [None, 0.05])
def test_gradients_reach_the_scale_and_the_median_keeps_the_l1_gradient(scale_weight):
    grads, references = laplace_and_control_gradients(scale_weight)
    # La cabeza de la mediana solo ve ∂L/∂μ, que es el del error absoluto.
    assert torch.equal(grads["head.weight"], references["head.weight"])
    assert torch.equal(grads["head.bias"], references["head.bias"])
    assert torch.count_nonzero(grads["output_objective.weight"])
    assert grads["output_objective.bias"].abs() > 0
    trunk = ("fusion.0.weight", "mac.persistent")
    if scale_weight is None:
        # Con los pesos de la escala en cero, el tronco recibe el gradiente del control.
        assert all(torch.equal(grads[name], references[name]) for name in trunk)
    else:
        # Después recibe además el gradiente de la escala.
        assert not any(torch.equal(grads[name], references[name]) for name in trunk)


def test_state_dict_round_trip_restores_the_scale():
    control, batch = setup()
    config = replace(control.config, laplace_scale_head=HEAD)
    model = api().FinancialPredictor(config, dtype=torch.float64)
    with torch.no_grad():
        model.output_objective.weight.normal_(generator=torch.Generator().manual_seed(5))
    model._seal_parameters()
    restored = api().FinancialPredictor(config, dtype=torch.float64)
    assert restored._parameter_id != model._parameter_id
    restored.load_state_dict(model.state_dict())
    assert restored._parameter_id == model._parameter_id
    left = model.prepare(batch, model.initial_state(batch.flow_ids))
    right = restored.prepare(batch, restored.initial_state(batch.flow_ids))
    assert torch.equal(left.outputs, right.outputs)
    assert not torch.allclose(left.outputs[:, 1], torch.full_like(left.outputs[:, 1], 0.02))
    with pytest.raises(ValueError, match="contrato completo"):
        control.load_state_dict(model.state_dict())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_declared_arms_build_with_the_walk_forward_factory(device):
    from mars_titan.training.financial_run import parameter_roles
    from mars_titan.training.titans_walk_forward import _predictor

    declaration = json.loads(Path("configs/experiments/laplace-scale-head-v1.json").read_text())
    recipe = json.loads(Path(declaration["base_recipe"]).read_text())
    recipe["predictor"]["head"] = declaration["base_head"]
    recipe["recipe"]["loss"] = declaration["base_loss"]
    base, _ = setup()
    scales = {}
    for name, arm in declaration["arms"].items():
        model, pairing = _predictor(
            recipe,
            base.config.inputs,
            "mac_online",
            42,
            device,
            changes=arm["predictor_changes"] or None,
        )
        assert pairing is None and model.config.head == "scalar"
        batch = api().DecisionBatch.from_corpus(
            raw_batch(), model.config.inputs, device=device, dtype=torch.float32
        )
        result = model.prepare(batch, model.initial_state(batch.flow_ids))
        scales[name] = None if result.outputs is None else result.outputs[:, 1]
        if result.outputs is not None:
            assert "output_objective.weight" in parameter_roles(model)["shared"]
    assert scales["absolute_error_control"] is None
    assert torch.isfinite(scales["laplace_scale_head"]).all()
