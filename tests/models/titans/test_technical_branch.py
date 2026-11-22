"""Rama de los rasgos técnicos del bloque del gráfico, con codificación lineal a trozos.

Las pruebas calculan salidas y gradientes sobre lotes ficticios, sin optimizador ni pasos.
"""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
from test_financial_adapter import DIMENSIONS, VARIANTS
from test_financial_learning_scale import DECLARED, build, emit

from mars_titan.models.titans.episodic_readout import apply_episodic_readout, head_predictions
from mars_titan.models.titans.frozen_financial import declared_sources
from mars_titan.models.titans.technical_branch import (
    TECHNICAL_BRANCH,
    TechnicalBranch,
    TechnicalBranchConfig,
    normal_edges,
)
from mars_titan.training.search_cases import checked_predictor_changes


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


COLUMNS = min(6, DIMENSIONS["charts"])
BRANCH = TechnicalBranchConfig(columns=COLUMNS, bins=8, embedding=4, hidden=16)


def branch(config=BRANCH, *, hidden=12, seed=3, gate=0.5):
    layer = TechnicalBranch(hidden, config, clip=5.0, seed=seed, dtype=torch.float64)
    with torch.no_grad():
        layer.gate.fill_(gate)
    return layer


def reference(layer, charts):
    """La misma ecuación fila a fila, rasgo a rasgo y tramo a tramo, con bucles de Python."""
    config, edges = layer.config, layer.edges
    features = charts[:, charts.shape[1] - config.columns :]
    rows = []
    for row in features:
        parts = []
        for index, value in enumerate(row):
            code = []
            for left, right in zip(edges[:-1], edges[1:], strict=True):
                code.append(min(max((value - left) / (right - left), 0.0), 1.0))
            code = torch.stack([torch.as_tensor(item, dtype=torch.float64) for item in code])
            parts.append(code @ layer.embedding_weight[index] + layer.embedding_bias[index])
        hidden = torch.nn.functional.silu(layer.norm(layer.inner(torch.cat(parts))))
        rows.append(layer.gate * layer.outer(hidden))
    return torch.stack(rows)


def test_the_edges_are_the_normal_quantiles_between_the_clip():
    edges = normal_edges(8, 5.0, torch.float64)
    assert edges.shape == (9,) and edges[0] == -5 and edges[-1] == 5 and edges[4] == 0
    assert bool((edges[1:] > edges[:-1]).all())
    torch.testing.assert_close(edges, -edges.flip(0))
    # El cuartil de una normal estándar.
    assert edges[2].item() == pytest.approx(-0.6744897501960817, abs=1e-12)
    with pytest.raises(ValueError, match="recorte"):
        normal_edges(64, 1.0)


def test_the_branch_matches_a_row_by_row_reference():
    layer = branch()
    generator = torch.Generator().manual_seed(1)
    charts = torch.randn(7, COLUMNS + 9, generator=generator, dtype=torch.float64) * 2.5
    charts[0, -1], charts[1, -2] = 9.0, -9.0
    torch.testing.assert_close(layer(charts), reference(layer, charts), rtol=1e-12, atol=1e-14)
    # Solo lee sus columnas, las últimas del bloque, y cada fila por separado.
    other = charts.clone()
    other[:, : charts.shape[1] - COLUMNS] += 3
    assert torch.equal(layer(other), layer(charts))
    moved = charts.clone()
    moved[3] += 1
    assert torch.equal(layer(moved)[[0, 1, 2, 4, 5, 6]], layer(charts)[[0, 1, 2, 4, 5, 6]])


def test_the_encoding_is_piecewise_linear_monotone_and_saturates():
    layer = branch()
    values = torch.linspace(-7, 7, 281, dtype=torch.float64).unsqueeze(1).repeat(1, COLUMNS)
    code = layer.encode(values)
    assert code.shape == (281, COLUMNS, 8) and code.min() == 0 and code.max() == 1
    assert bool((code[1:] >= code[:-1]).all())
    # La suma de los tramos, pesada con sus anchuras, devuelve el valor acotado.
    widths = layer.edges[1:] - layer.edges[:-1]
    rebuilt = layer.edges[0] + (code * widths).sum(-1)
    torch.testing.assert_close(rebuilt, values.clamp(-5, 5), rtol=0, atol=1e-12)
    assert torch.equal(code[0], torch.zeros_like(code[0]))
    assert torch.equal(code[-1], torch.ones_like(code[-1]))


def test_the_gate_starts_closed_and_still_receives_gradient():
    layer = TechnicalBranch(12, BRANCH, clip=5.0, seed=3, dtype=torch.float64)
    charts = torch.randn(
        5, COLUMNS, dtype=torch.float64, generator=torch.Generator().manual_seed(2)
    )
    result = layer(charts)
    assert torch.count_nonzero(result) == 0 and layer.gate.item() == 0
    (gradient,) = torch.autograd.grad(result.sum(), layer.gate)
    assert gradient.abs() > 0
    # La misma semilla da los mismos pesos, y otra semilla, otros.
    same = TechnicalBranch(12, BRANCH, clip=5.0, seed=3, dtype=torch.float64)
    other = TechnicalBranch(12, BRANCH, clip=5.0, seed=4, dtype=torch.float64)
    for (name, left), (_, right) in zip(
        layer.state_dict().items(), same.state_dict().items(), strict=True
    ):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), name
    assert not torch.equal(layer.inner.weight, other.inner.weight)
    # No consume el generador global.
    torch.manual_seed(11)
    expected = torch.rand(3)
    torch.manual_seed(11)
    TechnicalBranch(12, BRANCH, clip=5.0, seed=3)
    assert torch.equal(torch.rand(3), expected)


def test_disabled_field_keeps_identity_parameters_and_predictions():
    base, batch = build()
    explicit, _ = build(technical_branch=None)
    assert explicit.config.identity() == base.config.identity()
    assert "technical_branch" not in base.config.identity() and base.technical_branch is None
    # La base de estas pruebas declara learning_scale, cuyo módulo entra en la huella de código.
    assert declared_sources(base.config) == ("mars_titan.models.learning_scale",)
    expected, actual = emit(base, batch), emit(explicit, batch)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_predictor_with_the_branch_starts_as_the_reference(variant):
    base, batch = build(variant)
    predictor, _ = build(variant, technical_branch=asdict(BRANCH))
    reference_parameters, names = dict(base.named_parameters()), dict(predictor.named_parameters())
    assert set(names) - set(reference_parameters) == {
        "technical_branch.embedding_weight",
        "technical_branch.embedding_bias",
        "technical_branch.inner.weight",
        "technical_branch.inner.bias",
        "technical_branch.norm.weight",
        "technical_branch.norm.bias",
        "technical_branch.outer.weight",
        "technical_branch.outer.bias",
        "technical_branch.gate",
    }
    for name, value in reference_parameters.items():
        assert torch.equal(names[name], value), name
    expected, actual = emit(base, batch), emit(predictor, batch)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)
    identity = predictor.config.identity()["technical_branch"]
    assert identity["kind"] == TECHNICAL_BRANCH and identity["columns"] == COLUMNS
    assert identity["position"] == "after_mac_and_session_mixers_before_head"
    assert declared_sources(predictor.config) == (
        "mars_titan.models.titans.technical_branch",
        "mars_titan.models.learning_scale",
    )


def test_the_predictor_adds_the_residual_of_its_normalized_features():
    base, batch = build()
    predictor, _ = build(technical_branch=asdict(BRANCH))
    with torch.no_grad():
        predictor.technical_branch.gate.fill_(0.3)
    predictor._seal_parameters()
    plain, prepared = emit(base, batch), emit(predictor, batch)
    scaled = predictor.learning_scaler(batch.inputs, batch.presence)["charts"]
    expected = plain.working_state + predictor.technical_branch(scaled)
    torch.testing.assert_close(prepared.working_state, expected, rtol=0, atol=0)
    # La mediana empieza en cero con zero_median: el cambio se ve en los demás cuantiles.
    assert not torch.equal(prepared.quantiles, plain.quantiles)
    # La rama no toca la memoria: el estado siguiente es el de la referencia.
    torch.testing.assert_close(
        prepared.next_state.mac.memory.weights[0],
        plain.next_state.mac.memory.weights[0],
        rtol=0,
        atol=0,
    )
    # Y la puerta recibe gradiente de la predicción.
    differentiable = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids, differentiable=True), differentiable=True
    )
    (gradient,) = torch.autograd.grad(
        differentiable.quantiles.sum(), predictor.technical_branch.gate
    )
    assert gradient.abs() > 0


def test_the_state_dict_round_trips_and_another_size_is_another_identity():
    source, batch = build(technical_branch=asdict(BRANCH))
    with torch.no_grad():
        source.technical_branch.gate.fill_(0.2)
    source._seal_parameters()
    target, _ = build(technical_branch=asdict(BRANCH))
    target.load_state_dict(source.state_dict())
    torch.testing.assert_close(
        emit(target, batch).point_predictions, emit(source, batch).point_predictions, rtol=0, atol=0
    )
    other, _ = build(technical_branch=asdict(replace(BRANCH, bins=4)))
    with pytest.raises((ValueError, RuntimeError)):
        other.load_state_dict(source.state_dict())


def test_a_reader_receives_the_branch_through_the_working_state():
    predictor, batch = build(technical_branch=asdict(BRANCH))
    with torch.no_grad():
        predictor.technical_branch.gate.fill_(0.3)
    predictor._seal_parameters()
    prepared = emit(predictor, batch)
    # Un lector aplica la cabeza del núcleo a la representación de trabajo, refinada o no:
    # como el residuo de la rama ya está en ella, obtiene los mismos cuantiles.
    points, quantiles = head_predictions(predictor.head, prepared.working_state)
    torch.testing.assert_close(quantiles, prepared.quantiles, rtol=0, atol=0)
    torch.testing.assert_close(points, prepared.point_predictions, rtol=0, atol=0)
    result = apply_episodic_readout(prepared, predictor.head)
    assert torch.equal(result.quantiles, prepared.quantiles)


def test_the_source_enters_the_trainer_identity_only_when_declared():
    assert declared_sources(SimpleNamespace()) == ()
    assert declared_sources(SimpleNamespace(technical_branch=None)) == ()
    assert declared_sources(SimpleNamespace(technical_branch=BRANCH)) == (
        "mars_titan.models.titans.technical_branch",
    )


def test_the_predictor_change_registry_admits_it():
    changes = dict(technical_branch=asdict(BRANCH))
    assert checked_predictor_changes(changes) == changes
    base, _ = build()
    assert replace(base.config, **changes).technical_branch == BRANCH


@pytest.mark.parametrize(
    "value",
    [
        dict(columns=COLUMNS),
        dict(columns=0, bins=8, embedding=4, hidden=16),
        dict(columns=COLUMNS, bins=1, embedding=4, hidden=16),
        dict(columns=COLUMNS, bins=8, embedding=4, hidden=16, otra=1),
        dict(columns=True, bins=8, embedding=4, hidden=16),
        "technical_branch_v1",
    ],
)
def test_invalid_declarations_are_rejected(value):
    with pytest.raises(ValueError):
        build(technical_branch=value)


def test_the_branch_needs_normalized_chart_columns():
    with pytest.raises(ValueError, match="learning_scale"):
        build(technical_branch=asdict(BRANCH), learning_scale=None)
    wide = asdict(replace(BRANCH, columns=DIMENSIONS["charts"] + 1))
    with pytest.raises(ValueError, match="más columnas"):
        build(technical_branch=wide)
    with pytest.raises(ValueError, match="retirarlo"):
        build(technical_branch=asdict(BRANCH), removed_modalities=dict(names=["charts"]))
    with pytest.raises(ValueError, match="recorte"):
        TechnicalBranch(8, BRANCH, clip=True, seed=1)
    with pytest.raises(ValueError, match="FP32 o FP64"):
        TechnicalBranch(8, BRANCH, clip=5.0, seed=1, dtype=torch.float16)
    layer = branch()
    with pytest.raises(ValueError, match="bloque del gráfico"):
        layer(torch.zeros(2, COLUMNS - 1, dtype=torch.float64))
    assert DECLARED["clip"] == 5.0
