"""Predictor con la ganancia aprendida de la mediana (`learned_median_gain_v1`)."""

import math

import pytest
import torch
from test_financial_adapter import VARIANTS
from test_financial_learning_scale import build, emit

from mars_titan.models.quantile_head import MEDIAN_GAIN
from mars_titan.models.titans.episodic_readout import apply_episodic_readout, head_predictions
from mars_titan.models.titans.frozen_financial import _module_settings


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def test_without_the_declaration_nothing_changes():
    base, batch = build()
    explicit, _ = build(median_gain=None)
    assert explicit.config.identity() == base.config.identity()
    assert "median_gain" not in base.config.identity() and base.head.median_gain is None
    assert "head.median_gain" not in base.state_dict()
    assert "median_gain" not in _module_settings(base.head)
    torch.testing.assert_close(
        emit(explicit, batch).point_predictions, emit(base, batch).point_predictions, rtol=0, atol=0
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_predictor_with_the_gain_starts_as_the_reference(variant):
    base, batch = build(variant)
    predictor, _ = build(variant, median_gain=MEDIAN_GAIN)
    reference, names = dict(base.named_parameters()), dict(predictor.named_parameters())
    assert set(names) - set(reference) == {"head.median_gain"}
    for name, value in reference.items():
        assert torch.equal(names[name], value), name
    expected, actual = emit(base, batch), emit(predictor, batch)
    torch.testing.assert_close(actual.quantiles, expected.quantiles, rtol=0, atol=0)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    identity = predictor.config.identity()
    assert identity["median_gain"]["name"] == MEDIAN_GAIN
    assert {k: v for k, v in identity.items() if k != "median_gain"} == base.config.identity()
    assert _module_settings(predictor.head)["median_gain"] == MEDIAN_GAIN
    assert predictor._config_id() != base._config_id()


def test_the_gain_scales_the_emitted_median_and_reaches_the_readers():
    base, batch = build()
    predictor, _ = build(median_gain=MEDIAN_GAIN)
    with torch.no_grad():
        # Una mediana libre no nula, para que la ganancia tenga algo que escalar.
        row = torch.linspace(-0.5, 0.5, predictor.head.weight.shape[1], dtype=torch.float64)
        predictor.head.weight[2].copy_(row)
        base.head.weight[2].copy_(row)
        predictor.head.median_gain.fill_(math.log(0.2))
    predictor._seal_parameters()
    base._seal_parameters()
    plain, prepared = emit(base, batch), emit(predictor, batch)
    assert plain.point_predictions.abs().sum() > 0
    torch.testing.assert_close(
        prepared.point_predictions, 0.2 * plain.point_predictions, rtol=1e-12, atol=1e-15
    )
    # La ganancia no cambia el estado de trabajo: está en la cabeza.
    torch.testing.assert_close(prepared.working_state, plain.working_state, rtol=0, atol=0)
    # Un lector aplica la cabeza del núcleo al estado de trabajo y recibe la mediana a escala.
    points, quantiles = head_predictions(predictor.head, prepared.working_state)
    torch.testing.assert_close(points, prepared.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(quantiles, prepared.quantiles, rtol=0, atol=0)
    assert torch.equal(
        apply_episodic_readout(prepared, predictor.head).quantiles, prepared.quantiles
    )
    # Y recibe gradiente de lo emitido.
    differentiable = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids, differentiable=True), differentiable=True
    )
    (gradient,) = torch.autograd.grad(
        differentiable.point_predictions.sum(), predictor.head.median_gain
    )
    assert gradient.abs() > 0


def test_the_state_dict_round_trips_and_the_declaration_is_checked():
    source, batch = build(median_gain=MEDIAN_GAIN)
    with torch.no_grad():
        source.head.median_gain.fill_(-1.3)
    source._seal_parameters()
    target, _ = build(median_gain=MEDIAN_GAIN)
    target.load_state_dict(source.state_dict())
    assert target.head.median_gain.item() == -1.3
    torch.testing.assert_close(
        emit(target, batch).quantiles, emit(source, batch).quantiles, rtol=0, atol=0
    )
    plain, _ = build()
    with pytest.raises((ValueError, RuntimeError)):
        plain.load_state_dict(source.state_dict())
    for value in ("learned", True, 1.0):
        with pytest.raises(ValueError, match="learned_median_gain_v1"):
            build(median_gain=value)
    with pytest.raises(ValueError, match="quantile_head_v1"):
        build(head="scalar", median_gain=MEDIAN_GAIN)
