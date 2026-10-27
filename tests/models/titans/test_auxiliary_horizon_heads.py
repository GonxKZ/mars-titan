"""Cabezas auxiliares de horizontes de F5 (#749) en el predictor, sin pasos de optimizador."""

import math
from dataclasses import asdict, replace

import pytest
import torch
from test_financial_adapter import api, setup

from mars_titan.models.titans.auxiliary_horizons import (
    AUXILIARY_HORIZONS,
    AuxiliaryHeads,
    AuxiliaryHorizons,
    auxiliary_errors,
)
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.models.titans.regional_variants import (
    RegionAxisMixingConfig,
    SharedWorkspaceConfig,
)
from mars_titan.training.search_cases import checked_predictor_changes

CONFIG = AuxiliaryHorizons(horizons=(5, 20), weight=0.5, maturity_slack=2)


@pytest.mark.parametrize(
    "options",
    [
        dict(horizons=(20, 5)),
        dict(horizons=(1, 5)),
        dict(horizons=(5, 5)),
        dict(horizons=()),
        dict(horizons=(5, 20.0)),
        dict(weight=-0.1),
        dict(weight=float("inf")),
        dict(weight=True),
        dict(maturity_slack=-1),
        dict(maturity_slack=1.0),
    ],
)
def test_invalid_configurations_are_refused(options):
    with pytest.raises(ValueError):
        AuxiliaryHorizons(**options)


def test_declaration_travels_as_a_predictor_change_with_its_own_identity():
    base, _ = setup()
    options = dict(auxiliary_horizons=dict(horizons=[5, 20], weight=0.5, maturity_slack=2))
    assert checked_predictor_changes(options) == options
    changed = replace(base.config, **options)
    assert changed.auxiliary_horizons == CONFIG
    identity = changed.identity()
    assert identity["auxiliary_horizons"]["name"] == AUXILIARY_HORIZONS
    assert identity["auxiliary_horizons"]["horizons"] == [5, 20]
    assert {k: v for k, v in identity.items() if k != "auxiliary_horizons"} == (
        base.config.identity()
    )
    control = replace(base.config, auxiliary_horizons=replace(CONFIG, weight=0.0))
    assert control.identity()["auxiliary_horizons"]["weight"] == 0.0
    with pytest.raises(ValueError, match="maturity_slack"):
        replace(base.config, auxiliary_horizons=dict(horizons=[5, 20], weight=0.5))
    with pytest.raises(ValueError, match="AuxiliaryHorizons"):
        replace(base.config, auxiliary_horizons="auxiliary_horizons_v1")


OPERATOR = RegionalFusionConfig(topology="ring")


@pytest.mark.parametrize(
    "regional",
    [
        dict(regional_fusion=OPERATOR),
        dict(regional_variant=RegionAxisMixingConfig(OPERATOR)),
        dict(regional_variant=SharedWorkspaceConfig(OPERATOR)),
    ],
)
def test_operators_that_mix_rows_are_refused_because_tokens_are_recomputed_alone(regional):
    base, batch = setup()
    with pytest.raises(ValueError, match="fusión regional"):
        replace(base.config, auxiliary_horizons=CONFIG, **regional)
    model = api().FinancialPredictor(replace(base.config, **regional), dtype=torch.float64)
    with pytest.raises(ValueError, match="fila a fila"):
        model.fused_tokens(batch.inputs, batch.presence)


def test_heads_use_their_own_generator_and_leave_the_reference_untouched():
    control, batch = setup()
    before = torch.random.get_rng_state().clone()
    model = api().FinancialPredictor(
        replace(control.config, auxiliary_horizons=CONFIG), dtype=torch.float64
    )
    assert torch.equal(before, torch.random.get_rng_state())
    names = set(dict(model.named_parameters())) - {"auxiliary_heads.weight", "auxiliary_heads.bias"}
    assert names == set(dict(control.named_parameters()))
    for name, value in control.named_parameters():
        assert torch.equal(value, dict(model.named_parameters())[name]), name
    heads = model.auxiliary_heads
    assert heads.weight.shape == (2, 32) and heads.bias.shape == (2,)
    bound = 1 / math.sqrt(32)
    assert heads.weight.abs().max() <= bound and torch.count_nonzero(heads.weight)
    again = AuxiliaryHeads(32, CONFIG, seed=42, dtype=torch.float64)
    other = AuxiliaryHeads(32, CONFIG, seed=43, dtype=torch.float64)
    assert torch.equal(again.weight, heads.weight) and not torch.equal(other.weight, heads.weight)
    left = control.prepare(batch, control.initial_state(batch.flow_ids))
    right = model.prepare(batch, model.initial_state(batch.flow_ids))
    # La predicción y el token no cambian: las cabezas solo intervienen en el ajuste.
    assert torch.equal(left.point_predictions, right.point_predictions)
    assert torch.equal(left.detached_tokens, right.detached_tokens)
    with pytest.raises(ValueError, match="configuración"):
        AuxiliaryHeads(32, asdict(CONFIG), seed=42)


@pytest.mark.parametrize("modulation", [None, "film_v1", "wide_fusion_v1"])
def test_row_tokens_are_the_tokens_of_the_emission(modulation):
    control, batch = setup()
    config = replace(control.config, presence_modulation=modulation, auxiliary_horizons=CONFIG)
    model = api().FinancialPredictor(config, dtype=torch.float64)
    emitted = model.prepare(batch, model.initial_state(batch.flow_ids)).detached_tokens
    assert torch.equal(model.fused_tokens(batch.inputs, batch.presence), emitted)
    # Cada fila se calcula sola: el orden y la compañía del bloque no cambian su token.
    alone = model.fused_tokens(
        {name: value[1:] for name, value in batch.inputs.items()}, batch.presence[1:]
    )
    torch.testing.assert_close(alone, emitted[1:], rtol=1e-12, atol=1e-14)


def test_errors_follow_the_declared_formula_and_gradients():
    generator = torch.Generator().manual_seed(11)
    outputs = torch.randn(6, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    targets = torch.randn(6, generator=generator, dtype=torch.float64)
    heads = torch.tensor([0, 1, 1, 0, 1, 0])
    errors = auxiliary_errors(outputs, targets, heads, (5, 20))
    expected = torch.stack(
        [
            (outputs[i, h] - targets[i]).abs() / math.sqrt((5, 20)[h])
            for i, h in enumerate(heads.tolist())
        ]
    )
    torch.testing.assert_close(errors, expected, rtol=0, atol=0)
    assert torch.autograd.gradcheck(
        lambda value: auxiliary_errors(value, targets, heads, (5, 20)).mean(), (outputs,)
    )
    errors.sum().backward()
    # Cada fila solo deriva la salida de su horizonte.
    assert torch.count_nonzero(outputs.grad) == 6
    with pytest.raises(ValueError, match="horizonte"):
        auxiliary_errors(outputs.detach(), targets, heads, (5,))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_declared_arms_build_with_the_walk_forward_factory(device):
    import json
    from pathlib import Path

    from test_financial_adapter import raw_batch

    from mars_titan.models.titans.financial_inputs import DecisionBatch
    from mars_titan.training.financial_run import parameter_roles
    from mars_titan.training.titans_walk_forward import _predictor

    declaration = json.loads(Path("configs/experiments/auxiliary-horizons-v1.json").read_text())
    recipe = json.loads(Path(declaration["base_recipe"]).read_text())
    base, _ = setup()
    weights = {}
    for name, arm in declaration["arms"].items():
        model, pairing = _predictor(
            recipe,
            base.config.inputs,
            "mac_online",
            42,
            device,
            changes=arm["predictor_changes"],
        )
        assert pairing is None
        assert "auxiliary_heads.weight" in parameter_roles(model)["shared"]
        batch = DecisionBatch.from_corpus(
            raw_batch(), model.config.inputs, device=device, dtype=torch.float32
        )
        result = model.prepare(batch, model.initial_state(batch.flow_ids))
        assert torch.equal(model.fused_tokens(batch.inputs, batch.presence), result.detached_tokens)
        weights[name] = model.config.auxiliary_horizons.weight
    assert weights == dict(weight_zero_control=0.0, auxiliary_horizons=0.5)
