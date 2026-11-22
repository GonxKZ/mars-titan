"""Desplazamiento conocido de la etiqueta en la cabeza (`known_label_offset_v1`).

La etiqueta resta una cantidad que se conoce al predecir (el alfa del modelo de mercado, una
columna del bloque del gráfico). Con `known_offset` la cabeza predice lo operable y el predictor
emite la predicción de la etiqueta restando ese desplazamiento, exacto, a sus cinco cuantiles.
Las pruebas calculan salidas sobre lotes ficticios, sin optimizador ni pasos.
"""

import pytest
import torch
from test_financial_adapter import DIMENSIONS
from test_financial_learning_scale import build, emit

from mars_titan.models.quantile_head import median
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

OFFSET = dict(column=-2, scale=0.5)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def opened(model, seed=4):
    """La misma cabeza con pesos no nulos en todas sus filas, como tras algunos pasos."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        noise = torch.randn(model.head.weight.shape, dtype=torch.float64, generator=generator)
        model.head.weight.copy_(0.1 * noise)
    model._seal_parameters()
    return model


def test_the_offset_is_subtracted_from_the_five_quantiles_and_nothing_else_changes():
    plain, batch = build()
    shifted, _ = build(known_offset=OFFSET)
    for (name, left), (_, right) in zip(
        plain.named_parameters(), shifted.named_parameters(), strict=True
    ):
        assert torch.equal(left, right), name
    opened(plain), opened(shifted)
    expected, actual = emit(plain, batch), emit(shifted, batch)
    # El desplazamiento es la columna tal como se recibe, no la que el predictor escala.
    raw = batch.inputs["charts"][:, OFFSET["column"]]
    offset = OFFSET["scale"] * raw
    assert offset.abs().sum() > 0 and len(set(offset.tolist())) > 1
    scaled = shifted.learning_scaler(batch.inputs, batch.presence)["charts"][:, OFFSET["column"]]
    assert not torch.equal(scaled, raw)
    torch.testing.assert_close(
        actual.quantiles, expected.quantiles - offset.unsqueeze(1), rtol=0, atol=0
    )
    torch.testing.assert_close(actual.point_predictions, median(actual.quantiles), rtol=0, atol=0)
    torch.testing.assert_close(
        actual.point_predictions, expected.point_predictions - offset, rtol=0, atol=0
    )
    # Ni la representación ni la memoria dependen del desplazamiento.
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.next_state.mac.memory.weights[0],
        expected.next_state.mac.memory.weights[0],
        rtol=0,
        atol=0,
    )
    # Los cuantiles conservan su orden: todos se desplazan lo mismo.
    assert bool((actual.quantiles.diff(dim=1) >= 0).all()) == bool(
        (expected.quantiles.diff(dim=1) >= 0).all()
    )


def test_a_new_predictor_starts_predicting_minus_the_offset():
    shifted, batch = build(known_offset=OFFSET)
    plain, _ = build()
    # La mediana nace en cero: la predicción de la etiqueta nace en menos el desplazamiento.
    assert torch.count_nonzero(emit(plain, batch).point_predictions) == 0
    offset = OFFSET["scale"] * batch.inputs["charts"][:, OFFSET["column"]]
    torch.testing.assert_close(emit(shifted, batch).point_predictions, -offset, rtol=0, atol=0)


def test_the_offset_carries_no_gradient_and_the_head_receives_the_same_one():
    plain, batch = build()
    shifted, _ = build(known_offset=OFFSET)
    opened(plain), opened(shifted)
    gradients = []
    for model in (plain, shifted):
        state = model.initial_state(batch.flow_ids, differentiable=True)
        emitted = model.prepare(batch, state, differentiable=True)
        # Un funcional lineal de los cuantiles: su gradiente no depende de un desplazamiento.
        (gradient,) = torch.autograd.grad(emitted.quantiles.sum(), model.head.weight)
        gradients.append(gradient)
    torch.testing.assert_close(gradients[1], gradients[0], rtol=0, atol=0)


def test_the_declaration_enters_the_identity_only_when_declared():
    plain, _ = build()
    shifted, _ = build(known_offset=OFFSET)
    before, identity = plain.config.identity(), shifted.config.identity()
    assert "known_offset" not in before
    assert {key: value for key, value in identity.items() if key != "known_offset"} == before
    declared = identity["known_offset"]
    assert declared["name"] == "known_label_offset_v1" and declared["schema_version"] == 1
    assert declared["column"] == -2 and declared["scale"] == 0.5
    assert plain._config_id() != shifted._config_id()
    other, _ = build(known_offset=dict(column=-2, scale=0.25))
    assert other._config_id() != shifted._config_id()


@pytest.mark.parametrize(
    "value",
    [
        dict(column=0, scale=0.5),
        dict(column=2, scale=0.5),
        dict(column=True, scale=0.5),
        dict(column=-2.0, scale=0.5),
        dict(column=-2),
        dict(scale=0.5),
        dict(column=-2, scale=0.0),
        dict(column=-2, scale=-1.0),
        dict(column=-2, scale=float("inf")),
        dict(column=-2, scale="0.5"),
        dict(column=-2, scale=True),
        dict(column=-2, scale=0.5, other=1),
        dict(column=-(DIMENSIONS["charts"] + 1), scale=0.5),
        "alpha_percent",
        (-2, 0.5),
    ],
)
def test_invalid_declarations_are_rejected(value):
    with pytest.raises(ValueError, match="known_offset"):
        build(known_offset=value)


def test_it_needs_the_quantile_head_and_the_chart_block():
    with pytest.raises(ValueError, match="cuantiles"):
        build(head="scalar", known_offset=OFFSET)
    with pytest.raises(ValueError, match="gráfico"):
        build(known_offset=OFFSET, removed_modalities=dict(names=["charts"]))
    # La primera columna del bloque todavía le pertenece.
    build(known_offset=dict(column=-DIMENSIONS["charts"], scale=0.5))


def test_the_readers_refuse_a_parent_with_the_offset():
    shifted, _ = build(known_offset=OFFSET)
    plain, _ = build()
    with pytest.raises(ValueError, match="desplazamiento conocido"):
        FrozenFinancialConsumer(shifted.eval().requires_grad_(False))
    FrozenFinancialConsumer(plain.eval().requires_grad_(False))
