"""t de Student asimétrica con β-NLL (L2 de #787), sin pasos de optimizador."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import integrate, stats
from test_financial_adapter import api, raw_batch, setup

from mars_titan.models.quantile_head import LEVELS, QUANTILE_HEAD
from mars_titan.models.titans.laplace_scale import LaplaceScaleHead
from mars_titan.models.titans.output_objectives import reads_scalar_point
from mars_titan.models.titans.skewed_t_objective import (
    SKEWED_T_OBJECTIVE,
    SkewedT,
    SkewedTObjective,
    skewed_t_beta_nll,
    skewed_t_nll,
    skewed_t_quantiles,
)
from mars_titan.training.search_cases import checked_predictor_changes

OBJECTIVE = SkewedTObjective(initial_scale=0.02, initial_dof=6.0, max_skew=2.5)


@pytest.fixture
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def rows(dtype=torch.float64, count=9, seed=3):
    """Salidas [μ, σ, ξ, ν] y objetivos con asimetrías a los dos lados de 1."""
    generator = torch.Generator().manual_seed(seed)
    point = torch.randn(count, generator=generator, dtype=dtype) * 0.02
    scale = torch.rand(count, generator=generator, dtype=dtype) * 0.03 + 0.005
    skew = torch.exp(torch.randn(count, generator=generator, dtype=dtype) * 0.5)
    dof = torch.rand(count, generator=generator, dtype=dtype) * 20 + 2.5
    target = torch.randn(count, generator=generator, dtype=dtype) * 0.03
    return torch.stack((point, scale, skew, dof), dim=-1), target


def density(y, point, scale, skew, dof):
    """Densidad de la ecuación (7) escrita con la t de SciPy."""
    z = (y - point) / scale
    x = z * skew if z < 0 else z / skew
    return 2 / (scale * (skew + 1 / skew)) * stats.t.pdf(x, dof)


def cdf(y, point, scale, skew, dof):
    """Función de distribución integrada a mano a partir de la de la t de SciPy."""
    z = (y - point) / scale
    mass = 1 + skew**2
    if z < 0:
        return 2 / mass * stats.t.cdf(skew * z, dof)
    return 1 / mass + 2 * skew**2 / mass * (stats.t.cdf(z / skew, dof) - 0.5)


@pytest.mark.parametrize(
    "options",
    [
        dict(floor=0.0),
        dict(floor=1e-6, initial_scale=1.5e-6),
        dict(initial_scale=2.0),
        dict(dof_floor=1.5),
        dict(dof_floor=5.0, initial_dof=5.0),
        dict(initial_dof=150.0),
        dict(max_skew=1.0),
        dict(max_skew=12.0),
        dict(floor=True),
        # Un booleano en un campo donde 1,0 sería válido.
        dict(initial_scale=True),
        dict(floor=0.05, initial_scale=0.2),
        dict(initial_dof=float("nan")),
        dict(max_skew="3"),
    ],
)
def test_invalid_configurations_are_refused(options):
    with pytest.raises(ValueError):
        SkewedTObjective(**options)


def test_the_identity_records_every_declared_value_and_choice():
    assert OBJECTIVE.name == SKEWED_T_OBJECTIVE
    assert OBJECTIVE.identity() == dict(
        schema_version=1,
        name="skewed_t_objective_v1",
        floor=1e-6,
        initial_scale=0.02,
        dof_floor=2.0,
        initial_dof=6.0,
        max_skew=2.5,
        distribution="fernandez_steel_skewed_student_t",
        scale="softplus_of_linear_on_encoded_plus_floor",
        skew="max_skew_to_the_power_tanh_of_linear_on_encoded",
        dof="dof_floor_plus_softplus_of_linear_on_encoded",
        initial="zero_weights_constant_biases",
        loss="beta_nll_mean_of_detached_scale_times_full_nll",
        beta=1.0,
        point="median_of_the_fitted_distribution_with_scipy",
        emitted_quantiles="exact_quantiles_at_quantile_head_levels_with_scipy",
    )
    built = OBJECTIVE.build(8, seed=0, dtype=torch.float64)
    assert type(built) is SkewedT and built.config is OBJECTIVE
    assert built.weight.dtype == torch.float64 and built.weight.shape == (3, 8)


def test_parameters_start_at_their_declared_values_without_global_rng():
    before = torch.random.get_rng_state().clone()
    module = SkewedT(8, OBJECTIVE, dtype=torch.float64)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.count_nonzero(module.weight) == 0
    assert not module.session_level and module.row_decomposable and not module.scalar_point
    point = torch.randn(5, dtype=torch.float64)
    outputs, emitted = module(point, torch.randn(5, 8, dtype=torch.float64), ["US"] * 5)
    assert torch.equal(outputs[:, 0], point)
    expected = torch.tensor([0.02, 1.0, 6.0], dtype=torch.float64).expand(5, 3)
    torch.testing.assert_close(outputs[:, 1:], expected, rtol=1e-14, atol=0)
    # Con ξ = 1 la mediana emitida es la posición, sin redondeo.
    assert torch.equal(emitted, point)
    assert torch.equal(module.levels(outputs), skewed_t_quantiles(outputs.detach(), LEVELS))
    with pytest.raises(ValueError, match="su configuración"):
        SkewedT(8, asdict(OBJECTIVE))


def test_the_emitted_point_is_the_median_of_the_fitted_distribution():
    module = SkewedT(4, OBJECTIVE, dtype=torch.float64)
    with torch.no_grad():
        module.weight.normal_(generator=torch.Generator().manual_seed(11))
    point = torch.randn(6, dtype=torch.float64, generator=torch.Generator().manual_seed(12))
    encoded = torch.randn(6, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(13))
    outputs, emitted = module(point.requires_grad_(), encoded, ["US"] * 6)
    assert outputs.requires_grad and not emitted.requires_grad
    assert torch.equal(emitted, module.levels(outputs)[:, LEVELS.index(0.5)])
    values = outputs.detach()
    assert (values[:, 2] != 1).all() and not torch.equal(emitted, values[:, 0])
    for row, median in zip(values.tolist(), emitted.tolist(), strict=True):
        assert cdf(median, *row) == pytest.approx(0.5, abs=1e-12)
        # La mediana queda del lado de la cola larga: por encima de μ si ξ > 1.
        assert (median > row[0]) is (row[2] > 1)


def test_float32_outputs_keep_their_dtype():
    module = SkewedT(4, OBJECTIVE)
    point, encoded = torch.zeros(3), torch.ones(3, 4)
    outputs, emitted = module(point, encoded, ["US"] * 3)
    assert outputs.dtype == emitted.dtype == module.levels(outputs).dtype == torch.float32
    assert emitted.device == point.device


def test_the_parameters_stay_inside_their_declared_ranges():
    module = SkewedT(4, OBJECTIVE, dtype=torch.float64)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[50.0] * 4, [-50.0] * 4, [-50.0] * 4]))
    scale, skew, dof = module.parameters_of(torch.ones(3, 4, dtype=torch.float64) * 10)
    assert (scale > OBJECTIVE.floor).all()
    torch.testing.assert_close(skew, torch.full((3,), 1 / 2.5, dtype=torch.float64))
    torch.testing.assert_close(dof, torch.full((3,), 2.0, dtype=torch.float64))


def test_the_nll_is_the_negative_log_density_of_equation_seven():
    outputs, target = rows()
    nll = skewed_t_nll(outputs, target).tolist()
    for row, value, loss in zip(outputs.tolist(), target.tolist(), nll, strict=True):
        assert loss == pytest.approx(-math.log(density(value, *row)), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("skew", [0.5, 1.0, 2.0])
def test_the_density_integrates_to_one_with_the_declared_mass_below_the_location(skew):
    point, scale, dof = 0.01, 0.02, 4.5
    left = integrate.quad(lambda y: density(y, point, scale, skew, dof), -np.inf, point)[0]
    right = integrate.quad(lambda y: density(y, point, scale, skew, dof), point, np.inf)[0]
    assert left + right == pytest.approx(1.0, abs=1e-9)
    # ξ > 1 deja menos masa a la izquierda de μ y alarga la cola derecha.
    assert left == pytest.approx(1 / (1 + skew**2), abs=1e-9)


def test_the_full_nll_gradients_match_finite_differences():
    outputs, target = rows(count=6)
    assert torch.autograd.gradcheck(
        lambda values: skewed_t_nll(values, target).sum(),
        (outputs.clone().requires_grad_(),),
        eps=1e-7,
        atol=1e-6,
        rtol=1e-6,
    )


def test_the_beta_nll_holds_the_scale_weight_fixed():
    outputs, target = rows()
    weighted = outputs.clone().requires_grad_()
    skewed_t_beta_nll(weighted, target).backward()
    plain = outputs.clone().requires_grad_()
    (outputs[:, 1] * skewed_t_nll(plain, target)).mean().backward()
    torch.testing.assert_close(weighted.grad, plain.grad, rtol=1e-14, atol=0)
    value = (outputs[:, 1] * skewed_t_nll(outputs, target)).mean()
    assert float(skewed_t_beta_nll(outputs, target)) == pytest.approx(float(value), rel=1e-15)


def test_the_location_gradient_is_the_bounded_score_of_the_t():
    outputs, target = rows()
    values = outputs.clone().requires_grad_()
    skewed_t_beta_nll(values, target).backward()
    point, scale, skew, dof = outputs.unbind(-1)
    z = (target - point) / scale
    side = torch.where(z < 0, skew, 1 / skew)
    x = z * side
    expected = -side * (dof + 1) * x / (dof + x.square()) / len(target)
    torch.testing.assert_close(values.grad[:, 0], expected, rtol=1e-12, atol=1e-15)
    # La puntuación no depende de la escala de la fila y está acotada por (ν + 1)/(2√ν).
    bound = side * (dof + 1) / (2 * dof.sqrt()) / len(target)
    assert (values.grad[:, 0].abs() <= bound * (1 + 1e-12)).all()


def test_loss_rejects_malformed_outputs():
    outputs, target = rows()
    for bad in (outputs[:, :3], outputs.float(), outputs[:-1], outputs.unsqueeze(0)):
        with pytest.raises(ValueError, match="grados por etiqueta"):
            skewed_t_beta_nll(bad, target)
    with pytest.raises(ValueError, match="grados por etiqueta"):
        skewed_t_beta_nll(outputs[:0], target[:0])


def test_quantiles_invert_the_distribution_function():
    outputs, _ = rows()
    quantiles = skewed_t_quantiles(outputs, LEVELS)
    assert quantiles.shape == (len(outputs), len(LEVELS)) and quantiles.dtype == outputs.dtype
    assert (quantiles.diff(dim=1) > 0).all()
    for row, values in zip(outputs.tolist(), quantiles.tolist(), strict=True):
        for level, value in zip(LEVELS, values, strict=True):
            assert cdf(value, *row) == pytest.approx(level, abs=1e-10)


def test_symmetric_quantiles_are_those_of_the_student_t():
    outputs, _ = rows()
    outputs[:, 2] = 1.0
    quantiles = skewed_t_quantiles(outputs, LEVELS)
    point, scale, _, dof = (column.numpy()[:, None] for column in outputs.unbind(-1))
    expected = point + scale * stats.t.ppf(np.asarray(LEVELS)[None, :], dof)
    np.testing.assert_allclose(quantiles.numpy(), expected, rtol=1e-12, atol=1e-14)
    # La mediana coincide con la posición cuando ξ = 1.
    np.testing.assert_allclose(quantiles[:, LEVELS.index(0.5)].numpy(), point[:, 0], atol=1e-15)


def test_quantiles_need_detached_finite_outputs():
    outputs, _ = rows()
    with pytest.raises(ValueError, match="sin gradiente"):
        skewed_t_quantiles(outputs.clone().requires_grad_(), LEVELS)
    with pytest.raises(ValueError, match="por fila"):
        skewed_t_quantiles(outputs[:, :3], LEVELS)
    broken = outputs.clone()
    broken[0, 1] = math.nan
    with pytest.raises(ValueError, match="no son finitos"):
        skewed_t_quantiles(broken, LEVELS)


def test_declaration_travels_as_a_predictor_change_with_its_own_identity():
    base, _ = setup()
    options = dict(skewed_t_objective=asdict(OBJECTIVE))
    assert checked_predictor_changes(options) == options
    changed = replace(base.config, **options)
    assert changed.skewed_t_objective == OBJECTIVE
    # El punto emitido es la mediana, así que el lector episódico lo rechaza como padre.
    assert not reads_scalar_point(OBJECTIVE.build(8, seed=0, dtype=torch.float64))
    assert "skewed_t_objective" not in base.config.identity()
    identity = changed.identity()
    assert identity["output"] == SKEWED_T_OBJECTIVE
    assert identity["output_objective"] == OBJECTIVE.identity()
    reference = base.config.identity()
    assert {k: v for k, v in identity.items() if k not in ("output", "output_objective")} == {
        k: v for k, v in reference.items() if k != "output"
    }
    with pytest.raises(ValueError, match="toda su configuración"):
        replace(base.config, skewed_t_objective=dict(floor=1e-6))
    with pytest.raises(ValueError, match="cabeza escalar"):
        replace(base.config, head=QUANTILE_HEAD, skewed_t_objective=OBJECTIVE)
    with pytest.raises(ValueError, match="sus valores o None"):
        replace(base.config, skewed_t_objective=SKEWED_T_OBJECTIVE)
    # F1 es su control y no puede declararse a la vez.
    with pytest.raises(ValueError, match="excluyentes"):
        replace(base.config, skewed_t_objective=OBJECTIVE, laplace_scale_head=LaplaceScaleHead())
    with pytest.raises(ValueError):
        checked_predictor_changes(dict(skewed_t_objective={}))


@pytest.mark.parametrize("variant", ["transformer_direct", "mac_online"])
def test_disabled_objective_is_the_reference_and_enabled_keeps_the_location(variant):
    control, batch = setup(variant)
    before = torch.random.get_rng_state().clone()
    model = api().FinancialPredictor(
        replace(control.config, skewed_t_objective=OBJECTIVE), dtype=torch.float64
    )
    assert torch.equal(before, torch.random.get_rng_state())
    shared = dict(model.named_parameters())
    names = set(shared) - {"output_objective.weight", "output_objective.bias"}
    assert names == set(dict(control.named_parameters()))
    for name, value in control.named_parameters():
        assert torch.equal(value, shared[name]), name
    expected = control.prepare(batch, control.initial_state(batch.flow_ids))
    result = model.prepare(batch, model.initial_state(batch.flow_ids))
    # La posición es la salida escalar del control, y los demás parámetros sus valores iniciales.
    assert torch.equal(result.point_predictions, expected.point_predictions)
    assert torch.equal(result.outputs[:, 0], result.point_predictions)
    initial = torch.tensor([0.02, 1.0, 6.0], dtype=torch.float64).expand(2, 3)
    torch.testing.assert_close(result.outputs[:, 1:], initial)
    assert result.quantiles is None


@pytest.mark.parametrize("moved", [False, True])
def test_gradients_reach_the_three_projections_and_the_location_gets_the_t_score(moved):
    control, batch = setup()
    model = api().FinancialPredictor(
        replace(control.config, skewed_t_objective=OBJECTIVE), dtype=torch.float64
    )
    if moved:
        # Estado posterior simulado sin optimizador: σ, ξ y ν ya dependen del estado codificado.
        with torch.no_grad():
            model.output_objective.weight.normal_(
                0.0, 0.05, generator=torch.Generator().manual_seed(7)
            )
        model._seal_parameters()
    target = torch.tensor([0.01, -0.03], dtype=torch.float64)
    result = model.prepare(batch, model.initial_state(batch.flow_ids), differentiable=True)
    assert not result.point_predictions.requires_grad
    loss = model.output_objective.loss(result.outputs, target, None)
    (d_outputs,) = torch.autograd.grad(loss, result.outputs, retain_graph=True)
    loss.backward()
    grads = {name: value.grad for name, value in model.named_parameters()}
    assert torch.count_nonzero(grads["output_objective.weight"].abs().sum(1)) == 3
    assert torch.count_nonzero(grads["output_objective.bias"]) == 3
    # La cabeza escalar recibe la puntuación acotada de la t en lugar del signo del error.
    point, scale, skew, dof = result.outputs.detach().unbind(-1)
    side = torch.where(target < point, skew, 1 / skew)
    x = (target - point) / scale * side
    score = -side * (dof + 1) * x / (dof + x.square()) / len(target)
    torch.testing.assert_close(d_outputs[:, 0], score, rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(grads["head.bias"], score.sum().reshape(1), rtol=1e-12, atol=1e-15)
    # Con W = 0 el tronco solo recibe el gradiente de μ. Después recibe también el de σ, ξ y ν.
    location = api().FinancialPredictor(model.config, dtype=torch.float64)
    location.load_state_dict(model.state_dict())
    again = location.prepare(batch, location.initial_state(batch.flow_ids), differentiable=True)
    (again.outputs[:, 0] * d_outputs[:, 0]).sum().backward()
    only = {name: value.grad for name, value in location.named_parameters()}
    same = [torch.equal(grads[name], only[name]) for name in ("fusion.0.weight", "mac.persistent")]
    assert not any(same) if moved else all(same)


def test_state_dict_round_trip_restores_the_projections():
    control, batch = setup()
    config = replace(control.config, skewed_t_objective=OBJECTIVE)
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
    assert torch.equal(left.point_predictions, right.point_predictions)
    # Con ξ ≠ 1 el punto emitido es la mediana y deja de coincidir con μ.
    median = model.output_objective.levels(left.outputs)[:, LEVELS.index(0.5)]
    assert torch.equal(left.point_predictions, median)
    assert not torch.equal(left.point_predictions, left.outputs[:, 0])
    with pytest.raises(ValueError, match="contrato completo"):
        control.load_state_dict(model.state_dict())


def test_the_frozen_consumer_emits_the_median_and_refuses_the_reader(explicit_fastpath):
    from test_episodic_readout import CONTEXT, reader

    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    control, batch = setup()
    model = api().FinancialPredictor(
        replace(control.config, skewed_t_objective=OBJECTIVE), dtype=torch.float64
    )
    with torch.no_grad():
        model.output_objective.weight.normal_(generator=torch.Generator().manual_seed(5))
    model._seal_parameters()
    model.eval().requires_grad_(False)
    engine = FrozenFinancialConsumer(model)
    engine.verify()
    state = model.initial_state(batch.flow_ids)
    result = engine.prepare(batch, state, context_id=CONTEXT)
    expected = model.prepare(batch, state)
    assert torch.equal(result.point_predictions, expected.point_predictions)
    assert not torch.equal(result.point_predictions, expected.outputs[:, 0])
    # El lector refinaría μ con la cabeza escalar, que ya no es el punto emitido.
    with pytest.raises(ValueError, match="objetivo de salida"):
        FrozenFinancialConsumer(model, readout=reader(refinements=1).eval().requires_grad_(False))
    # Los mismos pesos con otra cota de asimetría ya no son el cálculo declarado.
    model.output_objective.config = replace(OBJECTIVE, max_skew=2.0)
    with pytest.raises(ValueError, match="cálculo"):
        engine.verify()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_declared_arms_build_with_the_walk_forward_factory(device):
    from mars_titan.training.financial_run import parameter_roles
    from mars_titan.training.titans_walk_forward import _predictor

    declaration = json.loads(Path("configs/experiments/skewed-t-objective-v1.json").read_text())
    recipe = json.loads(Path(declaration["base_recipe"]).read_text())
    recipe["predictor"]["head"] = declaration["base_head"]
    recipe["recipe"]["loss"] = declaration["base_loss"]
    base, _ = setup()
    outputs = {}
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
        outputs[name] = result.outputs
        if result.outputs is not None:
            assert "output_objective.weight" in parameter_roles(model)["shared"]
    assert outputs["absolute_error_control"] is None
    assert outputs["laplace_scale_head"].shape[1] == 2
    assert outputs["skewed_t_objective"].shape[1] == 4
    assert torch.isfinite(outputs["skewed_t_objective"]).all()
