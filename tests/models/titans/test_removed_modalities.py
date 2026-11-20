"""Modalidades retiradas del predictor: la misma ruta que una ausencia real, sin optimizadores."""

import pytest
import torch

from mars_titan.models.baselines.multimodal import MODALITIES
from mars_titan.models.titans.financial import REMOVED_MODALITIES
from tests.models.titans.test_financial_adapter import api, specification
from tests.models.titans.test_presence_modulation import decisions, fusion_input, prepare

CHARTS = MODALITIES.index("charts")
HIDDEN = 32


def build(removed=None, *, historical=True, **options):
    config = api().FinancialConfig(
        specification(historical=historical),
        variant="mac_online",
        hidden_size=HIDDEN,
        layers=1,
        seed=42,
        removed_modalities=None if removed is None else dict(names=list(removed)),
        **options,
    )
    return api().FinancialPredictor(config, dtype=torch.float64)


def without(batch, names):
    keep = torch.ones_like(batch.presence)
    for name in names:
        keep[:, MODALITIES.index(name)] = False
    return batch.with_absences(keep, REMOVED_MODALITIES)


@pytest.mark.parametrize("names", [("charts",), ("news", "charts"), ("charts", "macro")])
@pytest.mark.parametrize("training", [True, False])
def test_a_removed_modality_is_a_real_absence_in_the_fit_and_in_the_emission(names, training):
    control, removed = build(), build(names)
    # No añade parámetros: los dos predictores tienen los mismos pesos con la misma semilla.
    assert set(control.state_dict()) == set(removed.state_dict())
    for name, value in control.state_dict().items():
        if torch.is_tensor(value):
            assert torch.equal(removed.state_dict()[name], value), name
    control.train(training)
    removed.train(training)
    batch = decisions(complete=True)
    expected = prepare(control, without(batch, names))
    actual = prepare(removed, batch)
    assert torch.equal(actual.detached_tokens, expected.detached_tokens)
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    observed, _ = fusion_input(removed, batch)
    for name in names:
        index = MODALITIES.index(name)
        assert torch.count_nonzero(observed[:, index * HIDDEN : (index + 1) * HIDDEN]) == 0
        assert not observed[:, 5 * HIDDEN + index].any()
    # El modelo de control sí ve la modalidad: sus salidas son otras.
    assert not torch.equal(prepare(control, batch).detached_tokens, actual.detached_tokens)


def test_the_predictions_do_not_depend_on_the_removed_inputs_and_give_them_no_gradient():
    model = build(("charts",))
    batch = decisions(complete=True)
    generator = torch.Generator().manual_seed(9)
    changed = dict(batch.inputs)
    changed["charts"] = torch.randn(
        batch.inputs["charts"].shape, generator=generator, dtype=torch.float64
    )
    other = batch.derived(changed, "prueba_graficos_distintos")
    first, second = prepare(model, batch), prepare(model, other)
    assert torch.equal(first.detached_tokens, second.detached_tokens)
    assert torch.equal(first.point_predictions, second.point_predictions)
    # Con otra modalidad distinta sí cambian.
    changed = dict(batch.inputs)
    changed["macro"] = torch.randn(
        batch.inputs["macro"].shape, generator=generator, dtype=torch.float64
    )
    third = prepare(model, batch.derived(changed, "prueba_macro_distinto"))
    assert not torch.equal(first.detached_tokens, third.detached_tokens)
    prepare(model, batch, differentiable=True).point_predictions.sum().backward()
    weight = model.encoders["charts"][0].weight
    assert weight.grad is None or torch.count_nonzero(weight.grad) == 0
    assert model.encoders["macro"][0].weight.grad.abs().sum() > 0


def test_the_identity_names_the_removed_modalities_only_when_declared():
    assert "removed_modalities" not in build().config.identity()
    model = build(("macro", "charts"))
    # Se guardan en el orden de la fusión, se declaren como se declaren.
    assert model.config.removed_modalities == ("charts", "macro")
    assert model.config.identity()["removed_modalities"] == dict(
        name=REMOVED_MODALITIES,
        names=["charts", "macro"],
        route="presence_bit_false_and_zero_inputs",
        phase="fit_and_emission",
    )
    assert build(("charts",)).config.identity() != build(("macro",)).config.identity()


@pytest.mark.parametrize(
    "names",
    [
        (),
        ("prices",),
        ("charts", "charts"),
        ("volume",),
        (1,),
        ("news", "charts", "fundamentals", "macro"),
    ],
)
def test_invalid_removals_are_rejected(names):
    with pytest.raises(ValueError, match="removed_modalities"):
        build(names)


def test_removals_need_the_masked_policy_and_a_plain_declaration():
    with pytest.raises(ValueError, match="removed_modalities"):
        build(("charts",), historical=False)
    for declared in (dict(names=["charts"], extra=1), "charts", dict(other=["charts"])):
        with pytest.raises(ValueError, match="removed_modalities"):
            api().FinancialConfig(
                specification(historical=True),
                variant="mac_online",
                hidden_size=HIDDEN,
                layers=1,
                seed=42,
                removed_modalities=declared,
            )


def test_the_row_wise_tokens_of_the_auxiliary_horizons_refuse_a_removed_modality():
    model = build(("charts",))
    batch = decisions(complete=True)
    with pytest.raises(ValueError, match="modalidades retiradas"):
        model.fused_tokens(batch.inputs, batch.presence)


def test_a_removal_is_not_combined_with_the_auxiliary_horizons():
    horizons = dict(horizons=[5, 20], maturity_slack=2, weight=0.5)
    with pytest.raises(ValueError, match="removed_modalities con horizontes auxiliares"):
        build(("charts",), auxiliary_horizons=horizons)
