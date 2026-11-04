"""Cabezas oponentes con integrador neutral (A4 de #749), sin pasos de optimizador."""

import pytest
import torch
from scipy import stats
from torch.nn import functional

from mars_titan.models.titans.opponent_heads import (
    OPPONENT_HEADS,
    OpponentHeads,
    OpponentSpread,
    opponent_heads,
    opponent_loss,
)


def rows(dtype=torch.float64, count=9, seed=5):
    generator = torch.Generator().manual_seed(seed)
    center = torch.randn(count, generator=generator, dtype=dtype) * 0.02
    spread = torch.rand(count, generator=generator, dtype=dtype) * 0.01
    target = torch.randn(count, generator=generator, dtype=dtype) * 0.02
    return center, spread, target


def gradients(center, spread, target, config):
    center, spread = center.clone().requires_grad_(), spread.clone().requires_grad_()
    opponent_loss(torch.stack((center, spread), dim=-1), target, config).backward()
    return center.grad, spread.grad


@pytest.mark.parametrize("family", ["expectile", "quantile"])
def test_value_matches_the_asymmetric_loss_of_each_head(family):
    center, spread, target = rows()
    config = OpponentHeads(family=family, level=0.25)
    value = opponent_loss(torch.stack((center, spread), dim=-1), target, config)
    expected = 0.0
    for head, level in ((center - spread, 0.25), (center + spread, 0.75)):
        for y, h in zip(target.tolist(), head.tolist(), strict=True):
            u = y - h
            weight = level if u >= 0 else 1 - level
            expected += weight * (u * u if family == "expectile" else abs(u))
    assert value.item() == pytest.approx(expected / len(target), rel=1e-12)


def test_with_zero_spread_the_quantile_family_is_the_absolute_error_of_the_reference():
    center, _, target = rows()
    spread = torch.zeros_like(center)
    for level in (0.25, 0.5):
        config = OpponentHeads(family="quantile", level=level)
        d_center, d_spread = gradients(center, spread, target, config)
        control = center.clone().requires_grad_()
        functional.l1_loss(control, target).backward()
        assert torch.equal(d_center, control.grad)
        if level == 0.5:
            assert torch.equal(d_spread, torch.zeros_like(d_spread))
        else:
            # −(1 − 2τ)/N por fila con error no nulo.
            assert torch.allclose(d_spread, torch.full_like(d_spread, -0.5 / len(target)))


def test_with_zero_spread_the_expectile_family_is_the_squared_error_and_opens_the_heads():
    center, _, target = rows()
    spread = torch.zeros_like(center)
    variant = OpponentHeads(family="expectile", level=0.25)
    d_center, d_spread = gradients(center, spread, target, variant)
    control = center.clone().requires_grad_()
    functional.mse_loss(control, target).backward()
    torch.testing.assert_close(d_center, control.grad, rtol=1e-12, atol=0)
    error = (target - center).abs()
    torch.testing.assert_close(d_spread, -2 * error * 0.5 / len(target), rtol=1e-12, atol=0)
    _, symmetric = gradients(center, spread, target, OpponentHeads(level=0.5))
    assert torch.equal(symmetric, torch.zeros_like(symmetric))


@pytest.mark.parametrize("family", ["expectile", "quantile"])
def test_gradients_follow_autograd_of_the_explicit_formula(family):
    center, spread, target = rows()
    config = OpponentHeads(family=family, level=0.2)

    def explicit(c, d):
        total = 0
        for head, level in ((c - d, 0.2), (c + d, 0.8)):
            u = target - head
            w = torch.where(u < 0, 1 - level, level)
            total = total + w * (u * u if family == "expectile" else u.abs())
        return total.mean()

    inputs = (center.clone().requires_grad_(), spread.clone().requires_grad_())
    if family == "expectile":
        assert torch.autograd.gradcheck(explicit, inputs, eps=1e-8, atol=1e-7)
    reference = torch.autograd.grad(explicit(*inputs), inputs)
    for value, expected in zip(gradients(center, spread, target, config), reference, strict=True):
        torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("family", ["expectile", "quantile"])
def test_the_integrator_estimates_the_midpoint_of_both_levels_not_the_median(family):
    # Con una muestra asimétrica, el mínimo sobre constantes pone cada cabeza en su nivel.
    # El centro es la media de los dos niveles, que no coincide con la mediana ni con la media.
    generator = torch.Generator().manual_seed(3)
    target = -torch.log(torch.rand(101, generator=generator, dtype=torch.float64))
    config = OpponentHeads(family=family, level=0.25)
    if family == "quantile":
        ordered = target.sort().values
        # Con 101 valores, los cuantiles 0,25 y 0,75 son únicos: los estadísticos 26 y 76.
        low, high = ordered[25], ordered[75]
    else:
        low, high = (
            torch.tensor(stats.expectile(target.numpy(), alpha=a), dtype=torch.float64)
            for a in (0.25, 0.75)
        )
    center, spread = (low + high) / 2, (high - low) / 2

    def loss(c, d):
        prediction = torch.stack((c, d)).expand(len(target), 2)
        return opponent_loss(prediction, target, config)

    best = loss(center, spread)
    for dc, dd in ((1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)):
        assert best < loss(center + dc, spread + dd)
    if family == "expectile":
        inputs = (center.clone().requires_grad_(), spread.clone().requires_grad_())
        for value in torch.autograd.grad(loss(*inputs), inputs):
            assert abs(value.item()) < 1e-12
    assert abs(center - target.median()) > 0.05 and abs(center - target.mean()) > 0.05


def test_the_spread_starts_at_zero_without_touching_the_global_generator():
    torch.manual_seed(11)
    before = torch.random.get_rng_state()
    head = OpponentSpread(6, OpponentHeads(), dtype=torch.float32)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert not head.session_level and head.row_decomposable and head.scalar_point
    encoded, center = torch.randn(4, 6), torch.randn(4)
    outputs, point = head(center, encoded, ["US"] * 4)
    assert point is center and torch.equal(outputs[:, 1], torch.zeros(4))
    pair = opponent_heads(center, outputs[:, 1])
    assert torch.equal(pair.mean(dim=-1), center) and pair.shape == (4, 2)
    assert torch.equal(head.loss(outputs, center, None), torch.zeros(()))


def test_identity_separates_the_variant_from_its_control_and_the_families():
    variant = OpponentHeads().identity()
    control = OpponentHeads(level=0.5).identity()
    quantile = OpponentHeads(family="quantile").identity()
    assert variant["name"] == control["name"] == OPPONENT_HEADS
    assert variant["role"] == "opponent_variant" and control["role"] == "symmetric_control"
    assert len({str(variant), str(control), str(quantile)}) == 3
    assert variant["levels"] == [0.25, 0.75]


@pytest.mark.parametrize(
    "options",
    [
        dict(family="median"),
        dict(level=0.0),
        dict(level=0.6),
        dict(level=True),
        dict(level=float("nan")),
    ],
)
def test_invalid_configurations_are_rejected(options):
    with pytest.raises(ValueError):
        OpponentHeads(**options)


@pytest.mark.parametrize(
    "prediction,target",
    [
        (torch.zeros(3), torch.zeros(3)),
        (torch.zeros(3, 3), torch.zeros(3)),
        (torch.zeros(0, 2), torch.zeros(0)),
        (torch.zeros(3, 2, dtype=torch.float64), torch.zeros(3)),
    ],
)
def test_malformed_predictions_are_rejected(prediction, target):
    with pytest.raises(ValueError):
        opponent_loss(prediction, target, OpponentHeads())
    with pytest.raises(ValueError):
        opponent_loss(torch.zeros(3, 2), torch.zeros(3), {"family": "expectile"})


def test_the_spread_and_the_pair_reject_what_they_cannot_interpret():
    with pytest.raises(ValueError, match="configuración de las cabezas"):
        OpponentSpread(4, {"family": "expectile", "level": 0.25})
    with pytest.raises(ValueError, match="una semiamplitud por fila"):
        opponent_heads(torch.zeros(3), torch.zeros(2))
    with pytest.raises(ValueError, match="una semiamplitud por fila"):
        opponent_heads(torch.zeros(3, 1), torch.zeros(3, 1))


def test_declaration_travels_as_an_exclusive_output_objective():
    from dataclasses import asdict, replace

    from test_financial_adapter import setup

    from mars_titan.models.quantile_head import QUANTILE_HEAD
    from mars_titan.models.titans.laplace_scale import LaplaceScaleHead
    from mars_titan.training.search_cases import checked_predictor_changes

    base, _ = setup()
    options = dict(opponent_heads=asdict(OpponentHeads()))
    assert checked_predictor_changes(options) == options
    changed = replace(base.config, **options)
    assert changed.opponent_heads == OpponentHeads()
    identity = changed.identity()
    assert identity["output"] == OPPONENT_HEADS
    assert identity["output_objective"] == OpponentHeads().identity()
    with pytest.raises(ValueError, match="cabeza escalar"):
        replace(base.config, head=QUANTILE_HEAD, opponent_heads=OpponentHeads())
    with pytest.raises(ValueError, match="excluyentes"):
        replace(base.config, opponent_heads=OpponentHeads(), laplace_scale_head=LaplaceScaleHead())
    with pytest.raises(ValueError, match="toda su configuración"):
        replace(base.config, opponent_heads=dict(level=0.25))
    with pytest.raises(ValueError, match="sus valores o None"):
        replace(base.config, opponent_heads="expectile")


@pytest.mark.parametrize("variant", ["transformer_direct", "mac_online"])
def test_enabled_heads_keep_the_reference_weights_and_emit_the_same_point(variant):
    from dataclasses import replace

    from test_financial_adapter import api, setup

    control, batch = setup(variant)
    before = torch.random.get_rng_state().clone()
    model = api().FinancialPredictor(
        replace(control.config, opponent_heads=OpponentHeads()), dtype=torch.float64
    )
    assert torch.equal(before, torch.random.get_rng_state())
    shared = dict(model.named_parameters())
    assert set(shared) - {"output_objective.weight", "output_objective.bias"} == set(
        dict(control.named_parameters())
    )
    for name, value in control.named_parameters():
        assert torch.equal(value, shared[name]), name
    expected = control.prepare(batch, control.initial_state(batch.flow_ids))
    result = model.prepare(batch, model.initial_state(batch.flow_ids))
    assert expected.outputs is None
    assert torch.equal(result.point_predictions, expected.point_predictions)
    assert torch.equal(result.outputs[:, 0], result.point_predictions)
    assert torch.equal(result.outputs[:, 1], torch.zeros(len(batch.flow_ids), dtype=torch.float64))


@pytest.mark.parametrize(
    "config,reference",
    [
        (OpponentHeads(family="quantile", level=0.25), functional.l1_loss),
        (OpponentHeads(family="quantile", level=0.5), functional.l1_loss),
        (OpponentHeads(family="expectile", level=0.25), functional.mse_loss),
        (OpponentHeads(family="expectile", level=0.5), functional.mse_loss),
    ],
)
def test_at_zero_spread_the_trunk_sees_the_symmetric_loss_of_its_family(config, reference):
    from dataclasses import replace

    from test_financial_adapter import api, setup

    control, batch = setup()
    model = api().FinancialPredictor(
        replace(control.config, opponent_heads=config), dtype=torch.float64
    )
    target = torch.tensor([0.01, -0.03], dtype=torch.float64)
    result = model.prepare(batch, model.initial_state(batch.flow_ids), differentiable=True)
    model.output_objective.loss(result.outputs, target, None).backward()
    expected = control.prepare(batch, control.initial_state(batch.flow_ids), differentiable=True)
    reference(expected.point_predictions, target).backward()
    grads = {name: value.grad for name, value in model.named_parameters()}
    for name, value in control.named_parameters():
        torch.testing.assert_close(grads[name], value.grad, rtol=1e-12, atol=1e-18)
    spread = grads["output_objective.bias"]
    # Con el control simétrico d no recibe gradiente y sigue en cero sin ningún paso.
    assert (spread.abs() > 0).item() == (config.level < 0.5)
    if config.level < 0.5:
        assert spread.item() < 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_declared_arms_build_with_the_walk_forward_factory_and_pair_their_controls(device):
    import json
    from pathlib import Path

    from test_financial_adapter import api, raw_batch, setup

    from mars_titan.training.titans_walk_forward import _predictor

    declaration = json.loads(Path("configs/experiments/opponent-heads-v1.json").read_text())
    assert declaration["default_enabled"] is False and declaration["final_test_access"] is False
    recipe = json.loads(Path(declaration["base_recipe"]).read_text())
    recipe["predictor"]["head"] = declaration["base_head"]
    recipe["recipe"]["loss"] = declaration["base_loss"]
    base, _ = setup()
    arms = declaration["arms"]
    for name, arm in arms.items():
        if arm["role"] == "variant":
            control = arms[arm["control"]]["predictor_changes"]["opponent_heads"]
            variant = arm["predictor_changes"]["opponent_heads"]
            assert control["family"] == variant["family"] and control["level"] == 0.5
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
        assert (result.outputs is None) == (arm["role"] == "reference"), name
