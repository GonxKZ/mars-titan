"""Rama densa de los rasgos técnicos del bloque del gráfico (`technical_branch_v2`).

Las pruebas calculan salidas y gradientes sobre lotes ficticios, sin optimizador ni pasos.
"""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
from test_financial_adapter import DIMENSIONS, VARIANTS
from test_financial_learning_scale import build, emit

from mars_titan.models.titans.episodic_readout import apply_episodic_readout, head_predictions
from mars_titan.models.titans.frozen_financial import declared_sources
from mars_titan.models.titans.technical_branch import (
    TECHNICAL_DENSE,
    TechnicalBranch,
    TechnicalBranchConfig,
    TechnicalDense,
    TechnicalDenseConfig,
    build_technical_branch,
    parse_technical_branch,
)
from mars_titan.training.search_cases import checked_predictor_changes


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


COLUMNS = min(6, DIMENSIONS["charts"])
DENSE = TechnicalDenseConfig(columns=COLUMNS, hidden=16, depth=2)


def opened(layer, seed=9):
    """La misma rama con una capa de salida no nula, como tras algunos pasos de ajuste."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        layer.outer.weight.copy_(
            0.1 * torch.randn(layer.outer.weight.shape, dtype=torch.float64, generator=generator)
        )
        layer.outer.bias.copy_(
            0.1 * torch.randn(layer.outer.bias.shape, dtype=torch.float64, generator=generator)
        )
    return layer


def dense(config=DENSE, *, hidden=12, seed=3):
    return opened(TechnicalDense(hidden, config, clip=5.0, seed=seed, dtype=torch.float64))


def reference(layer, charts):
    """La misma ecuación fila a fila y unidad a unidad, con bucles de Python."""
    features = charts[:, charts.shape[1] - layer.config.columns :]
    rows = []
    for row in features:
        hidden = row
        for linear in layer.layers:
            units = []
            for weight, bias in zip(linear.weight, linear.bias, strict=True):
                value = sum(w * x for w, x in zip(weight, hidden, strict=True)) + bias
                units.append(value * torch.sigmoid(value))
            hidden = torch.stack(units)
        rows.append(
            torch.stack(
                [
                    sum(w * x for w, x in zip(weight, hidden, strict=True)) + bias
                    for weight, bias in zip(layer.outer.weight, layer.outer.bias, strict=True)
                ]
            )
        )
    return torch.stack(rows)


def test_the_dense_branch_matches_a_row_by_row_reference():
    layer = dense()
    charts = 2 * torch.randn(
        7, COLUMNS + 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1)
    )
    result = layer(charts)
    assert result.shape == (7, 12) and torch.count_nonzero(result) > 0
    torch.testing.assert_close(result, reference(layer, charts), rtol=1e-12, atol=1e-13)
    # Solo lee las últimas columnas: el resto del bloque no cambia el residuo.
    other = charts.clone()
    other[:, :3] += 1.0
    torch.testing.assert_close(layer(other), result, rtol=0, atol=0)
    # Cada fila se calcula por separado.
    torch.testing.assert_close(layer(charts[2:5]), result[2:5], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_the_depth_is_the_number_of_hidden_layers(depth):
    layer = TechnicalDense(12, replace(DENSE, depth=depth), clip=5.0, seed=3, dtype=torch.float64)
    assert [(m.in_features, m.out_features) for m in layer.layers] == [
        (COLUMNS if index == 0 else 16, 16) for index in range(depth)
    ]
    assert (layer.outer.in_features, layer.outer.out_features) == (16, 12)


def test_the_output_starts_at_zero_and_still_receives_gradient():
    layer = TechnicalDense(12, DENSE, clip=5.0, seed=3, dtype=torch.float64)
    charts = torch.randn(
        5, COLUMNS, dtype=torch.float64, generator=torch.Generator().manual_seed(2)
    )
    result = layer(charts)
    assert torch.count_nonzero(result) == 0
    assert torch.count_nonzero(layer.outer.weight) == 0 == torch.count_nonzero(layer.outer.bias)
    weight, bias = torch.autograd.grad(result.sum(), (layer.outer.weight, layer.outer.bias))
    assert weight.abs().sum() > 0 and bias.abs().sum() > 0
    # La misma semilla da los mismos pesos, y otra semilla, otros.
    same = TechnicalDense(12, DENSE, clip=5.0, seed=3, dtype=torch.float64)
    other = TechnicalDense(12, DENSE, clip=5.0, seed=4, dtype=torch.float64)
    for (name, left), (_, right) in zip(
        layer.state_dict().items(), same.state_dict().items(), strict=True
    ):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), name
    assert not torch.equal(layer.layers[0].weight, other.layers[0].weight)
    # No consume el generador global.
    torch.manual_seed(11)
    expected = torch.rand(3)
    torch.manual_seed(11)
    TechnicalDense(12, DENSE, clip=5.0, seed=3)
    assert torch.equal(torch.rand(3), expected)


def test_the_declaration_chooses_the_branch():
    assert parse_technical_branch(asdict(DENSE)) == DENSE
    assert parse_technical_branch(DENSE) is DENSE
    first = TechnicalBranchConfig(columns=COLUMNS, bins=8, embedding=4, hidden=16)
    assert parse_technical_branch(asdict(first)) == first
    assert type(build_technical_branch(12, DENSE, clip=5.0, seed=1)) is TechnicalDense
    assert type(build_technical_branch(12, first, clip=5.0, seed=1)) is TechnicalBranch
    assert DENSE.identity()["kind"] == TECHNICAL_DENSE == "technical_branch_v2"
    assert DENSE.identity() != first.identity()
    assert replace(DENSE, depth=3).identity() != DENSE.identity()


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_predictor_with_the_dense_branch_starts_as_the_reference(variant):
    base, batch = build(variant)
    predictor, _ = build(variant, technical_branch=asdict(DENSE))
    reference_parameters, names = dict(base.named_parameters()), dict(predictor.named_parameters())
    assert set(names) - set(reference_parameters) == {
        "technical_branch.layers.0.weight",
        "technical_branch.layers.0.bias",
        "technical_branch.layers.1.weight",
        "technical_branch.layers.1.bias",
        "technical_branch.outer.weight",
        "technical_branch.outer.bias",
    }
    for name, value in reference_parameters.items():
        assert torch.equal(names[name], value), name
    expected, actual = emit(base, batch), emit(predictor, batch)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)
    identity = predictor.config.identity()["technical_branch"]
    assert identity["kind"] == TECHNICAL_DENSE and identity["columns"] == COLUMNS
    assert identity["output"] == "residual_with_zero_initialized_output_layer"
    assert declared_sources(predictor.config) == ("mars_titan.models.titans.technical_branch",)


def test_the_predictor_adds_the_residual_of_its_normalized_features():
    base, batch = build()
    predictor, _ = build(technical_branch=asdict(DENSE))
    opened(predictor.technical_branch)
    predictor._seal_parameters()
    plain, prepared = emit(base, batch), emit(predictor, batch)
    scaled = predictor.learning_scaler(batch.inputs, batch.presence)["charts"]
    expected = plain.working_state + predictor.technical_branch(scaled)
    torch.testing.assert_close(prepared.working_state, expected, rtol=0, atol=0)
    assert not torch.equal(prepared.quantiles, plain.quantiles)
    # La rama no toca la memoria: el estado siguiente es el de la referencia.
    torch.testing.assert_close(
        prepared.next_state.mac.memory.weights[0],
        plain.next_state.mac.memory.weights[0],
        rtol=0,
        atol=0,
    )
    # Y todas sus capas reciben gradiente de la predicción.
    differentiable = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids, differentiable=True), differentiable=True
    )
    gradients = torch.autograd.grad(
        differentiable.quantiles.sum(), list(predictor.technical_branch.parameters())
    )
    assert all(gradient.abs().sum() > 0 for gradient in gradients)


def test_the_state_dict_round_trips_and_another_size_is_another_identity():
    source, batch = build(technical_branch=asdict(DENSE))
    opened(source.technical_branch)
    source._seal_parameters()
    target, _ = build(technical_branch=asdict(DENSE))
    target.load_state_dict(source.state_dict())
    torch.testing.assert_close(
        emit(target, batch).point_predictions, emit(source, batch).point_predictions, rtol=0, atol=0
    )
    for changed in (replace(DENSE, depth=1), replace(DENSE, hidden=8)):
        other, _ = build(technical_branch=asdict(changed))
        with pytest.raises((ValueError, RuntimeError)):
            other.load_state_dict(source.state_dict())


def test_a_reader_receives_the_dense_branch_through_the_working_state():
    predictor, batch = build(technical_branch=asdict(DENSE))
    opened(predictor.technical_branch)
    predictor._seal_parameters()
    prepared = emit(predictor, batch)
    points, quantiles = head_predictions(predictor.head, prepared.working_state)
    torch.testing.assert_close(quantiles, prepared.quantiles, rtol=0, atol=0)
    torch.testing.assert_close(points, prepared.point_predictions, rtol=0, atol=0)
    result = apply_episodic_readout(prepared, predictor.head)
    assert torch.equal(result.quantiles, prepared.quantiles)


def test_the_source_and_the_registry_admit_it():
    assert declared_sources(SimpleNamespace(technical_branch=DENSE)) == (
        "mars_titan.models.titans.technical_branch",
    )
    changes = dict(technical_branch=asdict(DENSE))
    assert checked_predictor_changes(changes) == changes
    base, _ = build()
    assert replace(base.config, **changes).technical_branch == DENSE


@pytest.mark.parametrize(
    "value",
    [
        dict(columns=COLUMNS, hidden=16),
        dict(columns=0, hidden=16, depth=2),
        dict(columns=COLUMNS, hidden=0, depth=2),
        dict(columns=COLUMNS, hidden=16, depth=0),
        dict(columns=COLUMNS, hidden=16, depth=9),
        dict(columns=COLUMNS, hidden=16, depth=2, otra=1),
        dict(columns=COLUMNS, hidden=16, depth=True),
        dict(columns=COLUMNS, hidden=16, depth=2, bins=8),
    ],
)
def test_invalid_declarations_are_rejected(value):
    with pytest.raises(ValueError):
        build(technical_branch=value)


def test_the_dense_branch_needs_normalized_chart_columns():
    with pytest.raises(ValueError, match="learning_scale"):
        build(technical_branch=asdict(DENSE), learning_scale=None)
    wide = asdict(replace(DENSE, columns=DIMENSIONS["charts"] + 1))
    with pytest.raises(ValueError, match="más columnas"):
        build(technical_branch=wide)
    with pytest.raises(ValueError, match="retirarlo"):
        build(technical_branch=asdict(DENSE), removed_modalities=dict(names=["charts"]))
    with pytest.raises(ValueError, match="recorte"):
        TechnicalDense(8, DENSE, clip=True, seed=1)
    with pytest.raises(ValueError, match="FP32 o FP64"):
        TechnicalDense(8, DENSE, clip=5.0, seed=1, dtype=torch.float16)
    with pytest.raises(ValueError, match="TechnicalDenseConfig"):
        TechnicalDense(8, TechnicalBranchConfig(columns=COLUMNS), clip=5.0, seed=1)
    layer = dense()
    with pytest.raises(ValueError, match="bloque del gráfico"):
        layer(torch.zeros(2, COLUMNS - 1, dtype=torch.float64))
