"""Rama densa de los rasgos técnicos del bloque del gráfico (`technical_branch_v2`).

Las pruebas calculan salidas y gradientes sobre lotes ficticios, sin optimizador ni pasos.
"""

from dataclasses import asdict, replace
from statistics import NormalDist
from types import SimpleNamespace

import pytest
import torch
from test_financial_adapter import DIMENSIONS, VARIANTS
from test_financial_learning_scale import build, emit

from mars_titan.models.titans.episodic_readout import apply_episodic_readout, head_predictions
from mars_titan.models.titans.frozen_financial import declared_sources
from mars_titan.models.titans.technical_branch import (
    ENSEMBLE_MEMBERS,
    TECHNICAL_DENSE,
    BatchEnsembleLinear,
    TechnicalBranch,
    TechnicalBranchConfig,
    TechnicalDense,
    TechnicalDenseConfig,
    build_technical_branch,
    parse_technical_branch,
)
from mars_titan.training.financial_run import clip_parameter_groups
from mars_titan.training.search_cases import checked_predictor_changes


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


COLUMNS = min(6, DIMENSIONS["charts"])
DENSE = TechnicalDenseConfig(columns=COLUMNS, hidden=16, depth=2)
RANKED = replace(DENSE, session_ranks=True)


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
    assert declared_sources(predictor.config) == (
        "mars_titan.models.titans.technical_branch",
        "mars_titan.models.learning_scale",
    )


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


def normal_scores(values):
    """Puntuaciones normales de los rangos de una columna, con los empates promediados."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks, start = [0.0] * len(values), 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        for position in range(start, stop + 1):
            ranks[order[position]] = (start + stop) / 2 + 1
        start = stop + 1
    return [NormalDist().inv_cdf((rank - 0.5) / len(values)) for rank in ranks]


def ranked_reference(layer, charts, markets):
    """Rasgos y, a su lado, sus rangos normales entre las filas de cada mercado."""
    features = charts[:, charts.shape[1] - layer.config.columns :]
    scores = torch.zeros_like(features)
    for market in set(markets):
        rows = [index for index, value in enumerate(markets) if value == market]
        for column in range(features.shape[1]):
            column_scores = normal_scores([features[row, column].item() for row in rows])
            for row, score in zip(rows, column_scores, strict=True):
                scores[row, column] = score
    return torch.cat((features, scores), dim=1)


def sample(rows=9, seed=4):
    charts = 2 * torch.randn(
        rows, COLUMNS + 3, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)
    )
    # Empates en una columna, como los que deja el recorte de learning_scale.
    charts[:4, -1] = 5.0
    markets = tuple("CN" if index % 3 == 0 else "US" for index in range(rows))
    return charts, markets


def test_session_ranks_add_the_normal_scores_of_each_market_to_the_features():
    layer = opened(TechnicalDense(12, RANKED, clip=5.0, seed=3, dtype=torch.float64))
    assert layer.session_level and not dense().session_level
    assert layer.layers[0].in_features == 2 * COLUMNS and layer.layers[1].in_features == 16
    charts, markets = sample()
    result = layer(charts, markets)
    expected = reference(replace_inputs(layer), ranked_reference(layer, charts, markets))
    torch.testing.assert_close(result, expected, rtol=1e-11, atol=1e-12)
    features = charts[:, -COLUMNS:]
    torch.testing.assert_close(
        layer.ranks(features, markets),
        ranked_reference(layer, charts, markets)[:, COLUMNS:],
        rtol=1e-12,
        atol=1e-12,
    )
    # Los empates comparten rango y, sin empates, las puntuaciones de cada mercado suman cero.
    scores = layer.ranks(features, markets)
    assert torch.unique(scores[[1, 2], -1]).numel() == 1
    for market in ("US", "CN"):
        rows = [index for index, value in enumerate(markets) if value == market]
        assert scores[rows][:, :-1].sum(dim=0).abs().max() < 1e-12


def replace_inputs(layer):
    """Vista de la rama cuya referencia lee todas las columnas que recibe."""
    return SimpleNamespace(
        config=SimpleNamespace(columns=2 * layer.config.columns),
        layers=layer.layers,
        outer=layer.outer,
    )


def test_session_ranks_follow_the_rows_and_keep_the_markets_apart():
    layer = opened(TechnicalDense(12, RANKED, clip=5.0, seed=3, dtype=torch.float64))
    charts, markets = sample()
    result = layer(charts, markets)
    order = torch.tensor([4, 0, 8, 2, 6, 1, 7, 3, 5])
    moved = layer(charts[order], tuple(markets[index] for index in order.tolist()))
    torch.testing.assert_close(moved, result[order], rtol=1e-12, atol=1e-13)
    # Cambiar las filas de un mercado no cambia el residuo de las del otro.
    other = charts.clone()
    china = [index for index, value in enumerate(markets) if value == "CN"]
    states = [index for index, value in enumerate(markets) if value == "US"]
    other[china] = other[china] * -1.5 + 0.3
    changed = layer(other, markets)
    torch.testing.assert_close(changed[states], result[states], rtol=0, atol=0)
    assert not torch.equal(changed[china], result[china])
    # Y sí depende de las demás filas de su mercado: el rango de una fila cambia con ellas.
    shifted = charts.clone()
    shifted[states[0], -COLUMNS:] += 10.0
    assert not torch.equal(layer(shifted, markets)[states[1]], result[states[1]])
    with pytest.raises(ValueError, match="mercado de cada fila"):
        layer(charts)
    with pytest.raises(ValueError, match="mercado de cada fila"):
        layer(charts, markets[:-1])
    # Las entradas no reciben gradiente a través de los rangos.
    leaf = charts.clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(layer(leaf, markets).sum(), leaf)
    assert torch.isfinite(gradient).all() and gradient[:, :3].abs().sum() == 0


def test_the_identity_only_names_the_session_ranks_when_declared():
    plain, ranked = DENSE.identity(), RANKED.identity()
    assert "session" not in plain and plain["encoding"] == "none_normalized_features"
    assert TechnicalDenseConfig(columns=COLUMNS, hidden=16, depth=2, session_ranks=False) == DENSE
    assert ranked["session"] == "rows_of_one_market_and_decision_instant_in_one_complete_block"
    assert ranked != plain and ranked["kind"] == plain["kind"]
    assert parse_technical_branch(asdict(RANKED)) == RANKED
    declared = dict(columns=COLUMNS, hidden=16, depth=2)
    assert parse_technical_branch(declared) == DENSE
    for value in ("yes", 1, None):
        with pytest.raises(ValueError):
            parse_technical_branch(dict(declared, session_ranks=value))


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_predictor_with_session_ranks_needs_the_complete_session(variant):
    base, batch = build(variant)
    predictor, _ = build(variant, technical_branch=asdict(RANKED))
    assert predictor.session_level and not base.session_level
    with pytest.raises(ValueError, match="sesión completa"):
        emit(predictor, batch)
    # Empieza emitiendo lo mismo que sin la rama.
    expected, actual = emit(base, batch), emit(predictor, batch, session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    opened(predictor.technical_branch)
    predictor._seal_parameters()
    prepared = emit(predictor, batch, session_complete=True)
    scaled = predictor.learning_scaler(batch.inputs, batch.presence)["charts"]
    markets = tuple(flow.split("/", 1)[0] for flow in batch.flow_ids)
    residual = predictor.technical_branch(scaled, markets)
    torch.testing.assert_close(
        prepared.working_state, expected.working_state + residual, rtol=0, atol=0
    )
    assert torch.count_nonzero(residual) > 0
    assert predictor.config.identity()["technical_branch"] == RANKED.identity()
    # El orden de las filas del bloque no cambia lo que recibe cada flujo.
    assert declared_sources(predictor.config) == (
        "mars_titan.models.titans.technical_branch",
        "mars_titan.models.learning_scale",
    )


NORMED = replace(DENSE, layer_norm=True)
SCALED = replace(DENSE, feature_scaling=True)


def test_the_identity_only_names_the_options_that_are_declared():
    base = DENSE.identity()
    assert "layer_norm" not in base and "feature_scaling" not in base
    assert replace(DENSE, layer_norm=False, feature_scaling=False).identity() == base
    normed, scaled = NORMED.identity(), SCALED.identity()
    assert normed["normalization"] == "layer_norm_before_silu_on_each_hidden_layer"
    assert scaled["feature_scaling"] == "learned_scale_per_input_initialized_at_one_v1"
    assert {k: v for k, v in scaled.items() if k != "feature_scaling"} == base
    declared = dict(columns=COLUMNS, hidden=16, depth=2)
    assert parse_technical_branch(declared) == DENSE
    assert parse_technical_branch({**declared, "layer_norm": True}) == NORMED
    assert parse_technical_branch({**declared, "feature_scaling": True}) == SCALED
    for value in ({**declared, "feature_scaling": 1}, {**declared, "layer_norm": "si"}):
        with pytest.raises(ValueError, match="booleano"):
            parse_technical_branch(value)
    with pytest.raises(ValueError, match="technical_branch debe declarar"):
        parse_technical_branch({**declared, "otra": True})


@pytest.mark.parametrize("config", [SCALED, replace(RANKED, feature_scaling=True)])
def test_feature_scaling_starts_as_the_plain_branch_and_scales_one_input(config):
    plain = dense(replace(config, feature_scaling=False))
    scaled = dense(config)
    inputs = COLUMNS * (2 if config.session_ranks else 1)
    extra = set(dict(scaled.named_parameters())) - set(dict(plain.named_parameters()))
    assert extra == {"input_scale"} and scaled.input_scale.shape == (inputs,)
    assert torch.equal(scaled.input_scale, torch.ones(inputs, dtype=torch.float64))
    for name, value in plain.named_parameters():
        assert torch.equal(dict(scaled.named_parameters())[name], value), name
    charts, markets = sample()
    arguments = (charts, markets) if config.session_ranks else (charts,)
    # Con la escala en uno, la misma salida bit a bit.
    assert torch.equal(scaled(*arguments), plain(*arguments))
    # Duplicar la escala de una entrada equivale a duplicar esa entrada.
    with torch.no_grad():
        scaled.input_scale[1] = 2.0
    doubled = charts.clone()
    doubled[:, charts.shape[1] - COLUMNS + 1] *= 2.0
    if not config.session_ranks:
        torch.testing.assert_close(scaled(charts), plain(doubled), rtol=1e-12, atol=1e-12)
    assert not torch.equal(scaled(*arguments), plain(*arguments))
    # La escala recibe gradiente y la rama no consume el generador global al construirse.
    (gradient,) = torch.autograd.grad(scaled(*arguments).sum(), scaled.input_scale)
    assert gradient.abs().sum() > 0
    torch.manual_seed(11)
    expected = torch.rand(3)
    torch.manual_seed(11)
    TechnicalDense(12, config, clip=5.0, seed=3)
    assert torch.equal(torch.rand(3), expected)


def test_layer_norm_normalizes_each_hidden_layer_before_its_activation():
    plain, normed = dense(DENSE), dense(NORMED)
    extra = set(dict(normed.named_parameters())) - set(dict(plain.named_parameters()))
    assert extra == {f"norms.{i}.{name}" for i in range(2) for name in ("weight", "bias")}
    for name, value in plain.named_parameters():
        assert torch.equal(dict(normed.named_parameters())[name], value), name
    charts, _ = sample()
    hidden = charts[:, charts.shape[1] - COLUMNS :]
    for layer, norm in zip(normed.layers, normed.norms, strict=True):
        before = layer(hidden)
        centred = before - before.mean(dim=1, keepdim=True)
        scale = torch.sqrt(centred.square().mean(dim=1, keepdim=True) + norm.eps)
        hidden = torch.nn.functional.silu(centred / scale * norm.weight + norm.bias)
    torch.testing.assert_close(normed(charts), normed.outer(hidden), rtol=1e-12, atol=1e-12)
    # Cada fila se normaliza sola: cambiar otra fila no la altera.
    other = charts.clone()
    other[0] += 3.0
    assert torch.equal(normed(other)[1:], normed(charts)[1:])
    assert not torch.equal(normed(charts), plain(charts))


def test_a_state_of_another_option_is_another_identity():
    for config in (NORMED, SCALED):
        source, target = dense(config), dense(DENSE)
        with pytest.raises((ValueError, RuntimeError)):
            target.load_state_dict(source.state_dict())
        same = dense(config, seed=5)
        same.load_state_dict(source.state_dict())
        charts, _ = sample()
        assert torch.equal(same(charts), source(charts))
    with pytest.raises(ValueError, match="hidden"):
        TechnicalDenseConfig(columns=COLUMNS, hidden=1, depth=1, layer_norm=True)


ENSEMBLE = replace(DENSE, ensemble_members=4)


def opened_ensemble(layer, seed=9):
    """Rama en conjunto con salida y adaptadores no triviales, como tras varios pasos de ajuste."""
    generator = torch.Generator().manual_seed(seed)
    dtype = layer.outer.weight.dtype
    with torch.no_grad():
        for linear in (*layer.layers, layer.outer):
            linear.r.copy_(
                1.0 + 0.25 * torch.randn(linear.r.shape, dtype=dtype, generator=generator)
            )
            linear.s.copy_(
                1.0 + 0.25 * torch.randn(linear.s.shape, dtype=dtype, generator=generator)
            )
            linear.bias.copy_(
                0.15 * torch.randn(linear.bias.shape, dtype=dtype, generator=generator)
            )
        layer.outer.weight.copy_(
            0.1 * torch.randn(layer.outer.weight.shape, dtype=dtype, generator=generator)
        )
        if layer.input_scale is not None:
            layer.input_scale.copy_(
                1.0 + 0.2 * torch.randn(layer.input_scale.shape, dtype=dtype, generator=generator)
            )
        if layer.norms is not None:
            for norm in layer.norms:
                norm.weight.copy_(
                    1.0 + 0.2 * torch.randn(norm.weight.shape, dtype=dtype, generator=generator)
                )
                norm.bias.copy_(
                    0.1 * torch.randn(norm.bias.shape, dtype=dtype, generator=generator)
                )
    return layer


def manual_ensemble_reference(layer, charts, markets=None):
    """Media de K redes construidas a mano con los pesos efectivos de cada miembro."""
    config = layer.config
    features = charts[:, charts.shape[1] - config.columns :]
    if config.session_ranks:
        features = torch.cat((features, layer.ranks(features.detach(), markets)), dim=1)
    if layer.input_scale is not None:
        features = features * layer.input_scale
    member_outputs = []
    for member in range(config.ensemble_members):
        hidden = features
        for index, block in enumerate(layer.layers):
            effective = block.weight * block.s[member].unsqueeze(1) * block.r[member].unsqueeze(0)
            hidden = hidden @ effective.T + block.bias[member]
            if layer.norms is not None:
                norm = layer.norms[index]
                centred = hidden - hidden.mean(dim=1, keepdim=True)
                scale = torch.sqrt(centred.square().mean(dim=1, keepdim=True) + norm.eps)
                hidden = centred / scale * norm.weight + norm.bias
            hidden = torch.nn.functional.silu(hidden)
        outer = layer.outer
        effective_outer = outer.weight * outer.s[member].unsqueeze(1) * outer.r[member].unsqueeze(0)
        member_outputs.append(hidden @ effective_outer.T + outer.bias[member])
    return torch.stack(member_outputs, dim=1).mean(dim=1)


def expected_parameter_count(config, hidden_size):
    """Cuenta cerrada de parámetros: compartidos más K vectores (r, s, bias) por capa lineal."""
    first = config.columns * (2 if config.session_ranks else 1)
    widths = [first, *([config.hidden] * config.depth), hidden_size]
    members = config.ensemble_members
    total = 0
    for before, after in zip(widths[:-1], widths[1:], strict=True):
        if members == 1:
            total += after * before + after
        else:
            total += after * before + members * (before + 2 * after)
    if config.layer_norm:
        total += 2 * config.depth * config.hidden
    if config.feature_scaling:
        total += first
    return total


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ensemble_default_keeps_identity_parameters_outputs_and_gradients_identical(dtype):
    omitted = TechnicalDenseConfig(columns=COLUMNS, hidden=16, depth=2)
    explicit = TechnicalDenseConfig(columns=COLUMNS, hidden=16, depth=2, ensemble_members=1)
    assert explicit == omitted
    assert explicit.identity() == omitted.identity()
    assert "ensemble_members" not in omitted.identity()
    assert "ensemble" not in omitted.identity()
    declared = dict(columns=COLUMNS, hidden=16, depth=2)
    assert parse_technical_branch(declared) == omitted
    assert parse_technical_branch({**declared, "ensemble_members": 1}) == omitted

    left = TechnicalDense(12, omitted, clip=5.0, seed=3, dtype=dtype)
    right = TechnicalDense(12, explicit, clip=5.0, seed=3, dtype=dtype)
    assert list(dict(left.named_parameters())) == list(dict(right.named_parameters()))
    for (name, first), (_, second) in zip(
        left.state_dict().items(), right.state_dict().items(), strict=True
    ):
        if isinstance(first, torch.Tensor):
            assert torch.equal(first, second), name

    # También con la capa de salida abierta: salida y gradientes idénticos bit a bit.
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        weight = 0.1 * torch.randn(left.outer.weight.shape, dtype=dtype, generator=generator)
        bias = 0.1 * torch.randn(left.outer.bias.shape, dtype=dtype, generator=generator)
        for branch in (left, right):
            branch.outer.weight.copy_(weight)
            branch.outer.bias.copy_(bias)
    charts = torch.randn(7, COLUMNS + 3, dtype=dtype, generator=torch.Generator().manual_seed(11))
    out_left, out_right = left(charts), right(charts)
    assert torch.equal(out_left, out_right)
    grads_left = torch.autograd.grad(out_left.sum(), list(left.parameters()))
    grads_right = torch.autograd.grad(out_right.sum(), list(right.parameters()))
    for first, second in zip(grads_left, grads_right, strict=True):
        assert torch.equal(first, second)

    # En el predictor completo, la huella y la emisión son idénticas con opción ausente o 1.
    plain, batch = build(technical_branch=declared)
    one, _ = build(technical_branch={**declared, "ensemble_members": 1})
    four, _ = build(technical_branch={**declared, "ensemble_members": 4})
    assert one.config.identity() == plain.config.identity()
    assert one._config_id() == plain._config_id()
    assert four._config_id() != plain._config_id()
    assert torch.equal(emit(one, batch).point_predictions, emit(plain, batch).point_predictions)


def test_ensemble_configuration_validates_members_and_updates_identity():
    assert ENSEMBLE_MEMBERS == (1, 4, 8, 16)
    base = DENSE.identity()
    declared = dict(columns=COLUMNS, hidden=16, depth=2)
    identities = {1: base}
    for members in (4, 8, 16):
        config = parse_technical_branch({**declared, "ensemble_members": members})
        assert config == replace(DENSE, ensemble_members=members)
        identity = config.identity()
        assert identity["ensemble_members"] == members
        assert identity["ensemble"] == "batch_ensemble_tabm_v1"
        assert identity["ensemble_aggregation"] == "mean_over_members"
        assert {
            k: v
            for k, v in identity.items()
            if k
            not in {
                "ensemble_members",
                "ensemble",
                "ensemble_initialization",
                "ensemble_aggregation",
            }
        } == base
        identities[members] = identity
    assert len({tuple(sorted(item.items())) for item in identities.values()}) == 4
    changes = dict(technical_branch=asdict(ENSEMBLE))
    assert checked_predictor_changes(changes) == changes

    for invalid in (0, -4, 2, 3, 5, 7, 12, 32, True, False, 4.0, "4", None):
        with pytest.raises(ValueError, match="ensemble_members"):
            parse_technical_branch({**declared, "ensemble_members": invalid})
        with pytest.raises(ValueError, match="ensemble_members"):
            TechnicalDenseConfig(columns=COLUMNS, hidden=16, depth=2, ensemble_members=invalid)
    # La rama de codificación a trozos (v1) no declara ensemble_members.
    with pytest.raises(ValueError):
        parse_technical_branch(
            dict(columns=COLUMNS, bins=8, embedding=4, hidden=16, ensemble_members=4)
        )


@pytest.mark.parametrize("members", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize(
    "options",
    [
        dict(depth=2),
        dict(depth=1),
        dict(depth=3),
        dict(depth=2, session_ranks=True),
        dict(depth=2, layer_norm=True),
        dict(depth=2, feature_scaling=True),
        dict(depth=3, session_ranks=True, layer_norm=True, feature_scaling=True),
    ],
)
def test_batchensemble_matches_independent_manual_member_networks(members, dtype, options):
    config = TechnicalDenseConfig(columns=COLUMNS, hidden=16, ensemble_members=members, **options)
    layer = opened_ensemble(TechnicalDense(12, config, clip=5.0, seed=5, dtype=dtype), seed=17)
    charts, markets = sample()
    charts = charts.to(dtype)
    arguments = (charts, markets) if config.session_ranks else (charts,)
    actual = layer(*arguments)
    expected = manual_ensemble_reference(layer, charts, markets if config.session_ranks else None)
    assert actual.shape == (len(charts), 12) and actual.dtype == dtype
    rtol, atol = (1e-12, 1e-13) if dtype == torch.float64 else (1e-5, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    # Si todos los r y s valen 1 y los sesgos coinciden con K=1, coincide con la rama simple.
    plain = TechnicalDense(12, replace(config, ensemble_members=1), clip=5.0, seed=5, dtype=dtype)
    aligned = TechnicalDense(12, config, clip=5.0, seed=5, dtype=dtype)
    with torch.no_grad():
        aligned.layers[0].r.fill_(1.0)
        weight = 0.1 * torch.randn(
            plain.outer.weight.shape, dtype=dtype, generator=torch.Generator().manual_seed(21)
        )
        bias = 0.1 * torch.randn(
            plain.outer.bias.shape, dtype=dtype, generator=torch.Generator().manual_seed(22)
        )
        plain.outer.weight.copy_(weight)
        plain.outer.bias.copy_(bias)
        aligned.outer.weight.copy_(weight)
        aligned.outer.bias.copy_(bias.unsqueeze(0).expand(members, -1))
    torch.testing.assert_close(aligned(*arguments), plain(*arguments), rtol=rtol, atol=atol)


@pytest.mark.parametrize("members", [4, 8, 16])
def test_batchensemble_initialization_follows_tabm_and_starts_members_distinct(members):
    # Con 16 columnas, los K vectores de entrada en ±1 empiezan todos distintos por pares.
    config = TechnicalDenseConfig(columns=16, hidden=12, depth=3, ensemble_members=members)
    plain = TechnicalDense(
        8, replace(config, ensemble_members=1), clip=5.0, seed=3, dtype=torch.float64
    )
    ensemble = TechnicalDense(8, config, clip=5.0, seed=3, dtype=torch.float64)
    ensemble_fp32 = TechnicalDense(8, config, clip=5.0, seed=3, dtype=torch.float32)

    # Los pesos compartidos y los sesgos iniciales coinciden bit a bit con K=1 de la misma semilla.
    for left, right in zip(plain.layers, ensemble.layers, strict=True):
        assert torch.equal(right.weight, left.weight)
        for member in range(members):
            assert torch.equal(right.bias[member], left.bias)

    # Primera capa: r aleatorio en ±1 (idéntico en FP32 y FP64) y s en 1.
    first_r = ensemble.layers[0].r
    assert first_r.shape == (members, 16)
    assert torch.equal(first_r.abs(), torch.ones_like(first_r))
    assert (first_r == 1.0).any() and (first_r == -1.0).any()
    assert torch.unique(first_r, dim=0).shape[0] == members
    assert torch.equal(first_r.to(torch.float32), ensemble_fp32.layers[0].r)
    assert torch.equal(ensemble.layers[0].s, torch.ones((members, 12), dtype=torch.float64))

    # Resto de capas ocultas y salida: todos los r y s empiezan en 1; la salida empieza en 0.
    for block in ensemble.layers[1:]:
        assert torch.equal(block.r, torch.ones_like(block.r))
        assert torch.equal(block.s, torch.ones_like(block.s))
    assert torch.equal(ensemble.outer.r, torch.ones((members, 12), dtype=torch.float64))
    assert torch.equal(ensemble.outer.s, torch.ones((members, 8), dtype=torch.float64))
    assert torch.count_nonzero(ensemble.outer.weight) == 0
    assert torch.count_nonzero(ensemble.outer.bias) == 0

    # Con la salida abierta de forma compartida, los K miembros producen salidas distintas.
    charts = torch.randn(5, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(13))
    with torch.no_grad():
        ensemble.outer.weight.copy_(
            0.1
            * torch.randn(
                ensemble.outer.weight.shape,
                dtype=torch.float64,
                generator=torch.Generator().manual_seed(14),
            )
        )
    hidden = charts.unsqueeze(1).expand(-1, members, -1)
    for block in ensemble.layers:
        hidden = torch.nn.functional.silu(block(hidden))
    per_member = ensemble.outer(hidden)
    assert per_member.shape == (5, members, 8)
    assert per_member.std(dim=1).min() > 0
    for left_member in range(members):
        for right_member in range(left_member + 1, members):
            assert not torch.allclose(per_member[:, left_member], per_member[:, right_member])


@pytest.mark.parametrize("members", [4, 8, 16])
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("session_ranks", [False, True])
@pytest.mark.parametrize("layer_norm", [False, True])
@pytest.mark.parametrize("feature_scaling", [False, True])
def test_batchensemble_parameter_count_matches_shared_weights_plus_member_vectors(
    members, depth, session_ranks, layer_norm, feature_scaling
):
    config = TechnicalDenseConfig(
        columns=COLUMNS,
        hidden=16,
        depth=depth,
        session_ranks=session_ranks,
        layer_norm=layer_norm,
        feature_scaling=feature_scaling,
        ensemble_members=members,
    )
    layer = TechnicalDense(12, config, clip=5.0, seed=3, dtype=torch.float64)
    actual = sum(parameter.numel() for parameter in layer.parameters())
    assert actual == expected_parameter_count(config, 12)
    assert all(isinstance(block, BatchEnsembleLinear) for block in layer.layers)
    assert isinstance(layer.outer, BatchEnsembleLinear)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batchensemble_is_deterministic_with_seed_and_does_not_touch_global_rng(dtype):
    config = TechnicalDenseConfig(
        columns=COLUMNS,
        hidden=16,
        depth=2,
        session_ranks=True,
        layer_norm=True,
        feature_scaling=True,
        ensemble_members=8,
    )
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        # No consume el generador global al construirse.
        torch.manual_seed(19)
        expected_draw = torch.rand(4)
        torch.manual_seed(19)
        first = TechnicalDense(12, config, clip=5.0, seed=7, dtype=dtype)
        assert torch.equal(torch.rand(4), expected_draw)

        same = TechnicalDense(12, config, clip=5.0, seed=7, dtype=dtype)
        other = TechnicalDense(12, config, clip=5.0, seed=8, dtype=dtype)
        for (name, left), (_, right) in zip(
            first.state_dict().items(), same.state_dict().items(), strict=True
        ):
            if isinstance(left, torch.Tensor):
                assert torch.equal(left, right), name
        assert not torch.equal(first.layers[0].weight, other.layers[0].weight)
        assert not torch.equal(first.layers[0].r, other.layers[0].r)

        opened_ensemble(first, seed=31)
        opened_ensemble(same, seed=31)
        charts, markets = sample()
        charts = charts.to(dtype)
        out_first = first(charts, markets)
        out_repeat = first(charts, markets)
        out_same = same(charts, markets)
        assert torch.equal(out_first, out_repeat)
        assert torch.equal(out_first, out_same)

        grads_first = torch.autograd.grad(out_first.sum(), list(first.parameters()))
        grads_same = torch.autograd.grad(out_same.sum(), list(same.parameters()))
        for left, right in zip(grads_first, grads_same, strict=True):
            assert torch.equal(left, right)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layer_norm", [False, True])
@pytest.mark.parametrize("session_ranks", [False, True])
def test_batchensemble_gradients_are_finite_at_zero_init_and_with_constant_inputs(
    dtype, layer_norm, session_ranks
):
    config = TechnicalDenseConfig(
        columns=COLUMNS,
        hidden=16,
        depth=2,
        session_ranks=session_ranks,
        layer_norm=layer_norm,
        feature_scaling=True,
        ensemble_members=4,
    )
    layer = TechnicalDense(12, config, clip=5.0, seed=3, dtype=dtype)
    markets = ("US", "US", "US", "CN", "CN")

    for constant_value in (2.0, 0.0):
        charts = torch.full((5, COLUMNS + 2), constant_value, dtype=dtype)
        arguments = (charts, markets) if session_ranks else (charts,)
        # Con salida inicializada a cero (predicción constante nula): salida 0 y gradientes finitos.
        result = layer(*arguments)
        assert torch.count_nonzero(result) == 0
        grads = torch.autograd.grad(result.sum(), list(layer.parameters()))
        assert all(torch.isfinite(grad).all() for grad in grads)
        by_name = dict(zip((name for name, _ in layer.named_parameters()), grads, strict=True))
        assert by_name["outer.bias"].abs().sum() > 0
        if constant_value != 0.0:
            assert by_name["outer.weight"].abs().sum() > 0

    # Con la salida abierta y entradas constantes: todos los gradientes son finitos y no nulos.
    opened = opened_ensemble(TechnicalDense(12, config, clip=5.0, seed=3, dtype=dtype), seed=41)
    for constant_value in (2.0, 0.0):
        charts = torch.full((5, COLUMNS + 2), constant_value, dtype=dtype)
        arguments = (charts, markets) if session_ranks else (charts,)
        result = opened(*arguments)
        assert torch.isfinite(result).all()
        grads = torch.autograd.grad(result.sum(), list(opened.parameters()))
        assert all(torch.isfinite(grad).all() for grad in grads)
        by_name = dict(zip((name for name, _ in opened.named_parameters()), grads, strict=True))
        if constant_value != 0.0:
            for key in (
                "layers.0.r",
                "layers.0.s",
                "layers.0.weight",
                "layers.0.bias",
                "outer.r",
                "outer.s",
                "outer.weight",
                "outer.bias",
            ):
                assert by_name[key].abs().sum() > 0, key

    # Lote vacío: conserva forma y finitud sin fallar.
    empty = torch.zeros((0, COLUMNS + 2), dtype=dtype)
    empty_out = opened(empty, ()) if session_ranks else opened(empty)
    assert empty_out.shape == (0, 12) and empty_out.dtype == dtype


def test_batchensemble_preserves_row_and_column_isolation_without_leakage():
    plain_config = TechnicalDenseConfig(
        columns=COLUMNS, hidden=16, depth=2, layer_norm=True, ensemble_members=4
    )
    plain = opened_ensemble(TechnicalDense(12, plain_config, clip=5.0, seed=3, dtype=torch.float64))
    charts, markets = sample()
    result = plain(charts)
    # Solo lee las últimas columnas del bloque del gráfico.
    other_prefix = charts.clone()
    other_prefix[:, :3] += 7.0
    assert torch.equal(plain(other_prefix), result)
    # Sin rangos por sesión, modificar una fila no altera las demás.
    moved = charts.clone()
    moved[0] += 4.0
    assert torch.equal(plain(moved)[1:], result[1:])
    torch.testing.assert_close(plain(charts[2:5]), result[2:5], rtol=1e-12, atol=1e-13)

    # Con rangos por sesión: equivariante ante permutaciones y mercados aislados.
    ranked_config = replace(plain_config, session_ranks=True)
    ranked = opened_ensemble(
        TechnicalDense(12, ranked_config, clip=5.0, seed=3, dtype=torch.float64)
    )
    ranked_out = ranked(charts, markets)
    order = torch.tensor([4, 0, 8, 2, 6, 1, 7, 3, 5])
    permuted = ranked(charts[order], tuple(markets[index] for index in order.tolist()))
    torch.testing.assert_close(permuted, ranked_out[order], rtol=1e-12, atol=1e-13)
    china = [index for index, value in enumerate(markets) if value == "CN"]
    states = [index for index, value in enumerate(markets) if value == "US"]
    other_market = charts.clone()
    other_market[china] = other_market[china] * -1.5 + 0.3
    changed = ranked(other_market, markets)
    assert torch.equal(changed[states], ranked_out[states])
    assert not torch.equal(changed[china], ranked_out[china])
    leaf = charts.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(ranked(leaf, markets).sum(), leaf)
    assert torch.isfinite(grad).all() and grad[:, :3].abs().sum() == 0

    # Validación de entradas mal formadas en BatchEnsembleLinear y TechnicalDense.
    be = plain.layers[0]
    with pytest.raises(ValueError, match="BatchEnsembleLinear"):
        be(torch.zeros((3, COLUMNS), dtype=torch.float64))
    with pytest.raises(ValueError, match="BatchEnsembleLinear"):
        be(torch.zeros((3, 8, COLUMNS), dtype=torch.float64))
    with pytest.raises(ValueError, match="BatchEnsembleLinear"):
        be(torch.zeros((3, 4, COLUMNS + 1), dtype=torch.float64))
    with pytest.raises(ValueError, match="BatchEnsembleLinear"):
        be(torch.zeros((3, 4, COLUMNS), dtype=torch.float32))
    with pytest.raises(ValueError, match="miembros"):
        BatchEnsembleLinear(4, 4, 1)
    with pytest.raises(ValueError, match="FP32 o FP64"):
        BatchEnsembleLinear(4, 4, 4, dtype=torch.float16)
    with pytest.raises(ValueError, match="entradas"):
        BatchEnsembleLinear(0, 4, 4)
    with pytest.raises(ValueError, match="salidas"):
        BatchEnsembleLinear(4, 0, 4)
    # Desajuste entre la configuración y las capas construidas en TechnicalDense.
    mismatched = TechnicalDense(12, plain_config, clip=5.0, seed=3, dtype=torch.float64)
    mismatched.config = replace(plain_config, ensemble_members=8)
    with pytest.raises(ValueError, match="conjunto de la rama técnica"):
        mismatched(charts)


@pytest.mark.parametrize("variant", VARIANTS)
def test_predictor_with_batchensemble_starts_as_reference(variant):
    config = TechnicalDenseConfig(
        columns=COLUMNS,
        hidden=16,
        depth=2,
        session_ranks=True,
        layer_norm=True,
        feature_scaling=True,
        ensemble_members=4,
    )
    base, batch = build(variant)
    predictor, _ = build(variant, technical_branch=asdict(config))
    base_params = dict(base.named_parameters())
    pred_params = dict(predictor.named_parameters())
    for name, value in base_params.items():
        assert torch.equal(pred_params[name], value), name
    expected = emit(base, batch)
    actual = emit(predictor, batch, session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)
    # Al arrancar con mediana constante nula y rama en cero, los gradientes son finitos.
    differentiable = predictor.prepare(
        batch,
        predictor.initial_state(batch.flow_ids, differentiable=True),
        differentiable=True,
        session_complete=True,
    )
    grads = torch.autograd.grad(
        differentiable.quantiles.sum(),
        list(predictor.technical_branch.parameters()),
    )
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_predictor_with_batchensemble_adds_residual_clips_and_round_trips():
    config = TechnicalDenseConfig(
        columns=COLUMNS,
        hidden=16,
        depth=2,
        session_ranks=True,
        layer_norm=True,
        feature_scaling=True,
        ensemble_members=4,
    )
    base, batch = build()
    predictor, _ = build(technical_branch=asdict(config))
    opened_ensemble(predictor.technical_branch, seed=55)
    predictor._seal_parameters()
    plain = emit(base, batch)
    prepared = emit(predictor, batch, session_complete=True)
    scaled = predictor.learning_scaler(batch.inputs, batch.presence)["charts"]
    markets = tuple(flow.split("/", 1)[0] for flow in batch.flow_ids)
    residual = predictor.technical_branch(scaled, markets)
    torch.testing.assert_close(
        prepared.working_state, plain.working_state + residual, rtol=0, atol=0
    )
    assert not torch.equal(prepared.quantiles, plain.quantiles)
    torch.testing.assert_close(
        prepared.next_state.mac.memory.weights[0],
        plain.next_state.mac.memory.weights[0],
        rtol=0,
        atol=0,
    )
    differentiable = predictor.prepare(
        batch,
        predictor.initial_state(batch.flow_ids, differentiable=True),
        differentiable=True,
        session_complete=True,
    )
    grads = torch.autograd.grad(
        differentiable.quantiles.sum(), list(predictor.technical_branch.parameters())
    )
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)

    # Todos los parámetros nuevos de la rama entran en el grupo funcional technical_head.
    groups = clip_parameter_groups(predictor)
    branch_names = {
        name for name, _ in predictor.named_parameters() if name.startswith("technical_branch.")
    }
    assert branch_names <= set(groups["technical_head"])
    assert any(name.endswith(".r") for name in branch_names)
    assert any(name.endswith(".s") for name in branch_names)
    assert declared_sources(predictor.config) == (
        "mars_titan.models.titans.technical_branch",
        "mars_titan.models.learning_scale",
    )

    # Round-trip de state_dict y rechazo ante distinto número de miembros.
    target, _ = build(technical_branch=asdict(config))
    target.load_state_dict(predictor.state_dict())
    torch.testing.assert_close(
        emit(target, batch, session_complete=True).point_predictions,
        prepared.point_predictions,
        rtol=0,
        atol=0,
    )
    for other_members in (1, 8):
        other, _ = build(technical_branch=asdict(replace(config, ensemble_members=other_members)))
        with pytest.raises((ValueError, RuntimeError)):
            other.load_state_dict(predictor.state_dict())


# --- absent_rows: rangos sin contar las filas ausentes (`ranked` y `masked_v1`) ---------------


def _synthetic_prices_with_untraded(seed=0):
    """Precios sintéticos con factor de mercado y sesiones sin volumen para activar untraded_63."""
    from tests.data.test_technical_features import market_prices

    _, sessions, table, market = market_prices(seed=seed)
    prices = table.copy()
    # Una sesión sin volumen cada 30 días garantiza untraded_63 > 0 en cualquier ventana de 63.
    prices[10::30, 4] = 0.0
    return sessions, prices, market


def _masked_reference_scores(values, present):
    """Puntuaciones normales de cada columna solo sobre sus filas presentes, con bucles Python."""
    rows, cols = values.shape
    scores = torch.zeros_like(values)
    for col in range(cols):
        active = [row for row in range(rows) if bool(present[row, col])]
        if not active:
            continue
        col_scores = normal_scores([values[row, col].item() for row in active])
        for row, score in zip(active, col_scores, strict=True):
            scores[row, col] = score
    return scores


def _realistic_session_features(version, *, dtype=torch.float64):
    """Bloque de rasgos reales de varias edades y con/sin factor de mercado para US y CN."""
    from mars_titan.data import technical_features as technical

    rows, markets = [], []
    # 7 activos US con edades variadas (sin 63, sin 252, completos) y uno sin factor de mercado.
    us_specs = [
        (0, 35, True),
        (1, 55, True),
        (2, 160, True),
        (3, 200, True),
        (4, 300, True),
        (5, 340, True),
        (6, 380, False),
    ]
    # 5 activos CN sin factor de mercado en ninguna fila y con edades variadas.
    cn_specs = [
        (10, 40, False),
        (11, 150, False),
        (12, 210, False),
        (13, 310, False),
        (14, 360, False),
    ]
    for market_name, specs in (("US", us_specs), ("CN", cn_specs)):
        for seed, row_index, has_factor in specs:
            sessions, prices, factor = _synthetic_prices_with_untraded(seed=seed)
            table = (
                technical.technical_features(prices, sessions, sessions, factor, version=version)
                if has_factor
                else technical.technical_features(prices, sessions, version=version)
            )
            rows.append(table[row_index])
            markets.append(market_name)
    import numpy as np

    features = torch.tensor(np.stack(rows), dtype=dtype)
    return features, tuple(markets)


def _realistic_predictor(
    version, *, absent_rows="ranked", variant="mac_online", dtype=torch.float64
):
    """Predictor con `learning_scale` ajustado sobre rasgos reales donde la mediana de los
    indicadores vale 1 (de modo que el escalador desplaza 1 -> 0 y 0 -> -clip)."""
    import numpy as np
    from test_financial_adapter import specification

    from mars_titan.data import technical_features as technical
    from mars_titan.models.learning_scale import LearningScale, fit_statistics
    from mars_titan.models.quantile_head import QUANTILE_HEAD
    from mars_titan.models.titans.financial import (
        DecisionBatch,
        FinancialConfig,
        FinancialPredictor,
    )

    width = len(technical.VERSIONS[version])
    prefix = 4
    dims = dict(DIMENSIONS, charts=prefix + width)
    spec = specification(historical=True)
    spec = type(spec)(
        source_sha256=spec.source_sha256,
        view_sha256=spec.view_sha256,
        representation=spec.representation,
        dimensions=dims,
        input_policy=spec.input_policy,
    )
    sessions, prices, factor = _synthetic_prices_with_untraded(seed=21)
    fit_table = technical.technical_features(prices, sessions, sessions, factor, version=version)
    # 200 filas: ~85% con los tres indicadores a 1 (filas 260..430) y ~15% tempranas (30..60).
    chosen = np.concatenate([np.arange(30, 60), np.arange(260, 430)])
    fit_count = len(chosen)
    rng = np.random.default_rng(7)
    fit_inputs = {
        name: rng.normal(
            0.0, 1.0, size=(fit_count, 64, w) if name == "prices" else (fit_count, w)
        ).astype(np.float32)
        for name, w in dims.items()
    }
    fit_inputs["prices"][..., -1] = 1.0
    fit_inputs["charts"][:, prefix:] = fit_table[chosen]
    for name in ("fundamentals", "macro"):
        fit_inputs[name][:, 1] = 1.0
        fit_inputs[name][:, 2] = np.abs(fit_inputs[name][:, 2])
    presence = np.ones((fit_count, 5), dtype=np.bool_)
    scale_cfg = dict(clip=5.0, price_returns=True, zero_median=True)
    stats = fit_statistics(
        LearningScale(**scale_cfg), fit_inputs, presence, rng.normal(scale=0.02, size=fit_count)
    )
    branch_cfg = TechnicalDenseConfig(
        columns=width, hidden=16, depth=2, session_ranks=True, absent_rows=absent_rows
    )
    config = FinancialConfig(
        spec,
        variant=variant,
        hidden_size=32,
        layers=1,
        seed=42,
        head=QUANTILE_HEAD,
        learning_scale=scale_cfg,
        technical_branch=asdict(branch_cfg),
    )
    model = FinancialPredictor(config, dtype=dtype)
    model.set_learning_statistics(stats)
    # Construir un lote con activos US y CN de distintas edades e indicadores mezclados.
    features, markets = _realistic_session_features(version, dtype=torch.float32)
    n_rows = len(markets)
    batch_rng = np.random.default_rng(13)
    batch_inputs = {
        name: batch_rng.normal(
            0.0, 0.5, size=(n_rows, 64, w) if name == "prices" else (n_rows, w)
        ).astype(np.float32)
        for name, w in dims.items()
    }
    batch_inputs["prices"][..., -1] = 1.0
    batch_inputs["charts"][:, prefix:] = features.numpy()
    for name in ("fundamentals", "macro"):
        batch_inputs[name][:, 1] = 1.0
        batch_inputs[name][:, 2] = np.abs(batch_inputs[name][:, 2])
    at = 1_609_459_200_000_000
    raw = dict(
        inputs=batch_inputs,
        presence=np.ones((n_rows, 5), dtype=np.bool_),
        sample_ids=[f"{m}/A{i:03d}/{at}" for i, m in enumerate(markets)],
        market=list(markets),
        prediction_at=np.full(n_rows, at, dtype="datetime64[us]"),
        input_available_at=np.full(n_rows, at - 1, dtype="datetime64[us]"),
    )
    return model, DecisionBatch.from_corpus(raw, spec, dtype=dtype), markets


@pytest.mark.parametrize("version", ["long_horizon_v2", "long_horizon_v3"])
def test_absent_dependency_table_matches_technical_features_on_synthetic_prices(version):
    from mars_titan.data import technical_features as technical
    from mars_titan.models.titans.technical_branch import (
        ABSENT_DEPENDENCIES,
        ABSENT_VERSION_BY_COLUMNS,
        INDICATORS,
    )

    names = technical.VERSIONS[version]
    assert ABSENT_VERSION_BY_COLUMNS[len(names)] == version
    deps = ABSENT_DEPENDENCIES[version]
    assert tuple(deps) == INDICATORS
    # Ninguna columna se repite entre indicadores y ninguna columna indicadora depende de nada.
    listed = [name for group in deps.values() for name in group]
    assert len(listed) == len(set(listed))
    assert set(listed) < set(names)
    assert not (set(listed) & set(INDICATORS))

    # Comprobar sobre precios sintéticos (dos semillas) que donde cada indicador vale 0,
    # exactamente sus columnas dependientes (y el propio indicador) valen 0.
    hm, h63, h252 = (names.index(ind) for ind in INDICATORS)
    sessions, prices, factor = _synthetic_prices_with_untraded(seed=2)
    with_m = technical.technical_features(prices, sessions, sessions, factor, version=version)
    without_m = technical.technical_features(prices, sessions, version=version)

    def zero_columns(slice_rows):
        return {names[col] for col in range(len(names)) if not slice_rows[:, col].any()}

    # 1) Sin factor de mercado y con >= 252 sesiones: has_market_model=0, has_63=1, has_252=1.
    no_market = without_m[260:400]
    assert (no_market[:, hm] == 0).all()
    assert (no_market[:, h63] == 1).all() and (no_market[:, h252] == 1).all()
    assert zero_columns(no_market) == set(deps["has_market_model"]) | {"has_market_model"}

    # 2) Con factor de mercado, entre 126 y 250 sesiones: has_market_model=1, has_63=1, has_252=0.
    no_252 = with_m[130:251]
    assert (no_252[:, hm] == 1).all() and (no_252[:, h63] == 1).all()
    assert (no_252[:, h252] == 0).all()
    assert zero_columns(no_252) == set(deps["has_252_sessions"]) | {"has_252_sessions"}

    # 3) Con 21..61 sesiones: los tres indicadores valen 0 y los rasgos de <=21 sesiones no.
    no_63 = with_m[22:62]
    assert (no_63[:, hm] == 0).all() and (no_63[:, h63] == 0).all() and (no_63[:, h252] == 0).all()
    assert zero_columns(no_63) == set(listed) | set(INDICATORS)

    # 4) Con historia completa y factor de mercado: ninguna columna es idénticamente cero.
    full = with_m[260:400]
    assert (full[:, hm] == 1).all() and (full[:, h63] == 1).all() and (full[:, h252] == 1).all()
    assert zero_columns(full) == set()

    # 5) Frontera de una sesión (documentada en INFORME.md): las ventanas de nivel/OHLC sin
    # retorno previo arrancan en las filas 62 y 251 (una sesión antes que has_63 y has_252).
    assert with_m[62, h63] == 0 and with_m[63, h63] == 1
    assert with_m[251, h252] == 0 and with_m[252, h252] == 1
    assert with_m[124, hm] == 0 and with_m[125, hm] == 1
    assert with_m[125, names.index("realized_label_1")] == 0
    assert with_m[126, names.index("realized_label_1")] != 0


@pytest.mark.parametrize("version", ["long_horizon_v2", "long_horizon_v3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_absent_rows_ranked_default_is_bit_identical_to_explicit_and_baseline(version, dtype):
    from mars_titan.data import technical_features as technical
    from mars_titan.models.titans.session_ranks import gaussian_ranks

    width = len(technical.VERSIONS[version])
    default_cfg = TechnicalDenseConfig(columns=width, hidden=16, depth=2, session_ranks=True)
    explicit_cfg = TechnicalDenseConfig(
        columns=width, hidden=16, depth=2, session_ranks=True, absent_rows="ranked"
    )
    assert default_cfg == explicit_cfg and default_cfg.absent_rows == "ranked"
    assert default_cfg.identity() == explicit_cfg.identity()
    assert "absent_rows" not in default_cfg.identity()
    assert "absent_feature_version" not in default_cfg.identity()
    assert (
        default_cfg.identity()["normal_scores"]
        == "inverse_normal_of_average_rank_minus_half_over_the_rows"
    )

    layer = TechnicalDense(12, default_cfg, clip=5.0, seed=3, dtype=dtype)
    assert dict(layer.named_buffers()) == {}
    features, markets = _realistic_session_features(version, dtype=dtype)
    scores = layer.ranks(features, markets)
    for market in ("US", "CN"):
        rows = [i for i, m in enumerate(markets) if m == market]
        expected = gaussian_ranks(features[rows], torch.ones_like(features[rows], dtype=torch.bool))
        assert torch.equal(scores[rows], expected)


def test_absent_rows_identity_validation_and_state_dict():
    from mars_titan.models.titans.technical_branch import ABSENT_MASKED, ABSENT_RANKED

    base_v2 = TechnicalDenseConfig(columns=46, hidden=16, depth=2, session_ranks=True)
    masked_v2 = replace(base_v2, absent_rows=ABSENT_MASKED)
    masked_v3 = TechnicalDenseConfig(
        columns=62, hidden=16, depth=2, session_ranks=True, absent_rows=ABSENT_MASKED
    )
    assert base_v2.absent_rows == ABSENT_RANKED
    assert masked_v2.identity() != base_v2.identity()
    assert masked_v2.identity()["absent_rows"] == "masked_v1"
    assert masked_v2.identity()["absent_feature_version"] == "long_horizon_v2"
    assert masked_v3.identity()["absent_feature_version"] == "long_horizon_v3"
    assert masked_v2.identity()["absent_value"] == 0.0
    assert (
        masked_v2.identity()["normal_scores"]
        == "inverse_normal_of_average_rank_minus_half_over_present_rows"
    )
    assert parse_technical_branch(asdict(masked_v2)) == masked_v2
    assert parse_technical_branch(asdict(masked_v3)) == masked_v3
    assert checked_predictor_changes(dict(technical_branch=asdict(masked_v2))) == dict(
        technical_branch=asdict(masked_v2)
    )

    # Declaraciones inválidas rechazadas.
    for bad in ("masked_v2", "", True, 1, None):
        with pytest.raises(ValueError, match="absent_rows"):
            TechnicalDenseConfig(
                columns=46, hidden=16, depth=2, session_ranks=True, absent_rows=bad
            )
    with pytest.raises(ValueError, match="session_ranks"):
        TechnicalDenseConfig(
            columns=46, hidden=16, depth=2, session_ranks=False, absent_rows=ABSENT_MASKED
        )
    for bad_cols in (6, 30, 45, 47, 61, 63):
        with pytest.raises(ValueError, match="long_horizon_v2"):
            TechnicalDenseConfig(
                columns=bad_cols, hidden=16, depth=2, session_ranks=True, absent_rows=ABSENT_MASKED
            )

    # Round-trip de state_dict y rechazo al cruzar ranked <-> masked_v1.
    source = opened(TechnicalDense(12, masked_v2, clip=5.0, seed=3, dtype=torch.float64))
    same = TechnicalDense(12, masked_v2, clip=5.0, seed=7, dtype=torch.float64)
    same.load_state_dict(source.state_dict())
    features, markets = _realistic_session_features("long_horizon_v2", dtype=torch.float64)
    assert torch.equal(same(features, markets), source(features, markets))

    ranked = TechnicalDense(12, base_v2, clip=5.0, seed=3, dtype=torch.float64)
    with pytest.raises((ValueError, RuntimeError)):
        ranked.load_state_dict(source.state_dict())
    with pytest.raises((ValueError, RuntimeError)):
        source.load_state_dict(ranked.state_dict())


@pytest.mark.parametrize("version", ["long_horizon_v2", "long_horizon_v3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_ranks_exclude_absent_rows_zero_them_and_keep_independent_columns(version, dtype):
    from mars_titan.data import technical_features as technical
    from mars_titan.models.titans.technical_branch import ABSENT_DEPENDENCIES, ABSENT_MASKED

    names = technical.VERSIONS[version]
    width = len(names)
    ranked_cfg = TechnicalDenseConfig(columns=width, hidden=16, depth=2, session_ranks=True)
    masked_cfg = replace(ranked_cfg, absent_rows=ABSENT_MASKED)
    ranked_layer = TechnicalDense(12, ranked_cfg, clip=5.0, seed=3, dtype=dtype)
    masked_layer = TechnicalDense(12, masked_cfg, clip=5.0, seed=3, dtype=dtype)
    # Pesos idénticos sin consumir generador global.
    for name, value in ranked_layer.named_parameters():
        assert torch.equal(dict(masked_layer.named_parameters())[name], value), name

    features, markets = _realistic_session_features(version, dtype=dtype)
    us_rows = [i for i, m in enumerate(markets) if m == "US"]
    us_features = features[us_rows]
    us_markets = tuple("US" for _ in us_rows)

    actual = masked_layer.ranks(us_features, us_markets)
    baseline = ranked_layer.ranks(us_features, us_markets)
    assert actual.shape == us_features.shape and actual.dtype == dtype

    # Máscara esperada construida a mano a partir de las columnas indicadoras.
    deps = ABSENT_DEPENDENCIES[version]
    expected_present = torch.ones_like(us_features, dtype=torch.bool)
    for indicator, cols in deps.items():
        ind_col = names.index(indicator)
        flag = us_features[:, ind_col] > 0
        for col_name in cols:
            expected_present[:, names.index(col_name)] = flag
    expected = _masked_reference_scores(us_features.double(), expected_present).to(dtype)

    atol = 1e-6 if dtype == torch.float32 else 1e-12
    rtol = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    # Las posiciones ausentes valen exactamente cero.
    assert torch.equal(actual[~expected_present], torch.zeros_like(actual[~expected_present]))
    # Las columnas indicadoras y las independientes no cambian respecto a ranked.
    dependent_indices = {names.index(c) for group in deps.values() for c in group}
    independent_indices = [i for i in range(width) if i not in dependent_indices]
    assert torch.equal(actual[:, independent_indices], baseline[:, independent_indices])
    # Las columnas dependientes con mezcla de ausentes sí cambian respecto a ranked.
    dep = sorted(dependent_indices)
    assert not torch.equal(actual[:, dep], baseline[:, dep])

    # Perturbar valores de celdas ausentes no altera ningún rango (ni presentes ni ausentes).
    perturbed = us_features.clone()
    perturbed[~expected_present] = 7.5
    assert torch.equal(masked_layer.ranks(perturbed, us_markets), actual)

    # Cambiar el orden de filas permuta exactamente la salida de la rama completa.
    opened_masked = opened(TechnicalDense(12, masked_cfg, clip=5.0, seed=3, dtype=torch.float64))
    charts = torch.cat(
        (torch.randn(len(us_rows), 3, dtype=torch.float64), us_features.double()), dim=1
    )
    order = torch.tensor([5, 0, 6, 2, 1, 4, 3])
    out = opened_masked(charts, us_markets)
    moved = opened_masked(charts[order], us_markets)
    torch.testing.assert_close(moved, out[order], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("version", ["long_horizon_v2", "long_horizon_v3"])
def test_masked_ranks_work_per_market_and_handle_all_absent_and_single_present(version):
    from mars_titan.data import technical_features as technical
    from mars_titan.models.titans.technical_branch import ABSENT_DEPENDENCIES, ABSENT_MASKED

    names = technical.VERSIONS[version]
    width = len(names)
    cfg = TechnicalDenseConfig(
        columns=width, hidden=16, depth=2, session_ranks=True, absent_rows=ABSENT_MASKED
    )
    layer = opened(TechnicalDense(12, cfg, clip=5.0, seed=3, dtype=torch.float64))

    features, markets = _realistic_session_features(version, dtype=torch.float64)
    # 1) Bloque contiguo (US seguido de CN) y bloque intercalado (US/CN mezclados).
    interleaved_order = torch.tensor([0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 6])
    interleaved_features = features[interleaved_order]
    interleaved_markets = tuple(markets[i] for i in interleaved_order.tolist())

    contiguous_ranks = layer.ranks(features, markets)
    interleaved_ranks = layer.ranks(interleaved_features, interleaved_markets)
    torch.testing.assert_close(
        interleaved_ranks, contiguous_ranks[interleaved_order], rtol=0, atol=0
    )

    # Cada mercado coincide exactamente con calcularlo por separado; perturbar CN no toca US.
    us_idx = [i for i, m in enumerate(markets) if m == "US"]
    cn_idx = [i for i, m in enumerate(markets) if m == "CN"]
    torch.testing.assert_close(
        contiguous_ranks[us_idx],
        layer.ranks(features[us_idx], tuple("US" for _ in us_idx)),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        contiguous_ranks[cn_idx],
        layer.ranks(features[cn_idx], tuple("CN" for _ in cn_idx)),
        rtol=0,
        atol=0,
    )
    changed = features.clone()
    changed[cn_idx] = changed[cn_idx] * -2.0 + 0.5
    assert torch.equal(layer(changed, markets)[us_idx], layer(features, markets)[us_idx])

    # 2) En CN has_market_model=0 en todas las filas -> columnas de modelo de mercado todo ceros.
    mm_cols = [names.index(c) for c in ABSENT_DEPENDENCIES[version]["has_market_model"]]
    assert torch.count_nonzero(contiguous_ranks[cn_idx][:, mm_cols]) == 0

    # 3) Una sola fila presente en una columna dependiente -> su rango (y el resto) vale 0.
    single = features[:4].clone()
    single[:, names.index("has_252_sessions")] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=torch.float64
    )
    h252_cols = [names.index(c) for c in ABSENT_DEPENDENCIES[version]["has_252_sessions"]]
    single_ranks = layer.ranks(single, ("US",) * 4)
    assert torch.count_nonzero(single_ranks[:, h252_cols]) == 0

    # 4) Lote vacío devuelve tensor vacío de la misma forma (en ranks y en presence).
    empty = features[:0]
    assert layer.ranks(empty, ()).shape == (0, width)
    assert layer.presence(empty).shape == (0, width)
    assert not TechnicalBranch(
        12, TechnicalBranchConfig(columns=width), clip=5.0, seed=3
    ).uses_raw_charts
    assert layer.uses_raw_charts
    assert not TechnicalDense(
        12, replace(cfg, absent_rows="ranked"), clip=5.0, seed=3
    ).uses_raw_charts


@pytest.mark.parametrize("version", ["long_horizon_v2", "long_horizon_v3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_predictor_reads_unscaled_indicators_under_learning_scale(version, dtype):
    from mars_titan.data import technical_features as technical
    from mars_titan.models.titans.financial import copy_paired_parameters
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
    from mars_titan.models.titans.technical_branch import ABSENT_DEPENDENCIES, ABSENT_MASKED

    names = technical.VERSIONS[version]
    width = len(names)
    ranked_model, batch, markets = _realistic_predictor(version, absent_rows="ranked", dtype=dtype)
    masked_model, _, _ = _realistic_predictor(version, absent_rows=ABSENT_MASKED, dtype=dtype)

    # Confirmar que learning_scaler desplaza los indicadores 1 -> 0 y 0 -> negativo (< 0):
    # leer el indicador después de escalar marcaría todas las filas como ausentes.
    scaled_charts = masked_model.learning_scaler(batch.inputs, batch.presence)["charts"]
    raw_tech = batch.inputs["charts"][:, -width:]
    scaled_tech = scaled_charts[:, -width:]
    for indicator in ("has_market_model", "has_63_sessions", "has_252_sessions"):
        col = names.index(indicator)
        present_rows = raw_tech[:, col] > 0
        absent_rows = raw_tech[:, col] == 0
        assert present_rows.any() and absent_rows.any()
        assert (scaled_tech[present_rows, col] == 0).all()
        assert (scaled_tech[absent_rows, col] < 0).all()

    # Al construirse (capa outer en 0), ranked y masked emiten idéntico al control sin rama.
    start_ranked = emit(ranked_model, batch, session_complete=True)
    start_masked = emit(masked_model, batch, session_complete=True)
    torch.testing.assert_close(
        start_masked.point_predictions, start_ranked.point_predictions, rtol=0, atol=0
    )
    torch.testing.assert_close(
        start_masked.working_state, start_ranked.working_state, rtol=0, atol=0
    )

    # Al abrir outer, el predictor pasa raw_charts (sin escalar) a la rama técnica y
    # el residuo coincide bit a bit con technical_branch(scaled, markets, raw_charts=raw).
    with torch.no_grad():
        gen = torch.Generator().manual_seed(17)
        w_shape = masked_model.technical_branch.outer.weight.shape
        b_shape = masked_model.technical_branch.outer.bias.shape
        w = 0.1 * torch.randn(w_shape, dtype=dtype, generator=gen)
        b = 0.1 * torch.randn(b_shape, dtype=dtype, generator=gen)
        for model in (ranked_model, masked_model):
            model.technical_branch.outer.weight.copy_(w)
            model.technical_branch.outer.bias.copy_(b)
            model._seal_parameters()

    prep_ranked = emit(ranked_model, batch, session_complete=True)
    prep_masked = emit(masked_model, batch, session_complete=True)
    expected_residual = masked_model.technical_branch(
        scaled_charts, markets, raw_charts=batch.inputs["charts"]
    )
    torch.testing.assert_close(
        prep_masked.working_state,
        start_masked.working_state + expected_residual,
        rtol=0,
        atol=0,
    )
    # Los rangos calculados sobre scaled_tech con indicadores sin escalar tienen no-ceros en
    # presentes y ceros exactos en ausentes, y difieren del modo ranked.
    masked_ranks = masked_model.technical_branch.ranks(scaled_tech, markets, indicators=raw_tech)
    h252_col = names.index("return_252")
    assert (masked_ranks[raw_tech[:, names.index("has_252_sessions")] == 0, h252_col] == 0).all()
    assert (masked_ranks[raw_tech[:, names.index("has_252_sessions")] > 0, h252_col] != 0).any()
    assert not torch.equal(prep_masked.working_state, prep_ranked.working_state)

    # Compatible con controles emparejados (copy_paired_parameters) y FrozenFinancialConsumer.
    paired_frozen, _, _ = _realistic_predictor(
        version, absent_rows=ABSENT_MASKED, variant="mac_frozen", dtype=dtype
    )
    copy_paired_parameters(masked_model, paired_frozen)
    with pytest.raises(ValueError):
        copy_paired_parameters(masked_model, ranked_model)

    for param in masked_model.parameters():
        param.requires_grad_(False)
    masked_model.eval()
    consumer = FrozenFinancialConsumer(masked_model)
    with torch.no_grad():
        frozen = consumer.prepare(
            batch,
            masked_model.initial_state(batch.flow_ids),
            context_id="ctx",
            session_complete=True,
        )
    assert frozen.quantiles is not None
    torch.testing.assert_close(frozen.quantiles, prep_masked.quantiles, rtol=0, atol=0)
    torch.testing.assert_close(
        frozen.point_predictions, prep_masked.point_predictions, rtol=0, atol=0
    )
    branch_id = consumer.identity()["predictor"]["configuration"]["technical_branch"]
    assert branch_id["absent_rows"] == ABSENT_MASKED
    assert "has_252_sessions" in ABSENT_DEPENDENCIES[version]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_branch_is_causal_and_has_finite_gradients_with_constant_zero_prediction(dtype):
    import numpy as np

    from mars_titan.data import technical_features as technical
    from mars_titan.models.quantile_head import pinball_loss
    from mars_titan.models.titans.session_ic_objective import SessionICTerm
    from mars_titan.models.titans.technical_branch import ABSENT_MASKED

    # 1) Sin fugas: perturbar precios y factor de mercado futuros tras el corte no altera los
    # rasgos, la máscara de ausencia ni la salida de TechnicalDense en el corte.
    version = technical.NAME_V2
    width = len(technical.VERSIONS[version])
    cfg = TechnicalDenseConfig(
        columns=width, hidden=16, depth=2, session_ranks=True, absent_rows=ABSENT_MASKED
    )
    layer = opened(TechnicalDense(12, cfg, clip=5.0, seed=3, dtype=torch.float64)).to(dtype)

    cut = 280
    rows_before, rows_after = [], []
    specs = ((0, 40, True), (1, 180, True), (2, 270, True), (3, 275, False))
    for seed, row_idx, has_factor in specs:
        sessions, prices, factor = _synthetic_prices_with_untraded(seed=seed)
        changed_p = prices.copy()
        changed_p[cut:] *= 1.8
        changed_f = factor.copy()
        changed_f[cut:] = -changed_f[cut:] + 0.05
        if has_factor:
            a = technical.technical_features(prices, sessions, sessions, factor, version=version)
            b = technical.technical_features(
                changed_p, sessions, sessions, changed_f, version=version
            )
        else:
            a = technical.technical_features(prices, sessions, version=version)
            b = technical.technical_features(changed_p, sessions, version=version)
        rows_before.append(a[row_idx])
        rows_after.append(b[row_idx])
    f_before = torch.tensor(np.stack(rows_before), dtype=dtype)
    f_after = torch.tensor(np.stack(rows_after), dtype=dtype)
    markets = ("US",) * 4
    assert torch.equal(layer.presence(f_before), layer.presence(f_after))
    assert torch.equal(layer(f_before, markets), layer(f_after, markets))

    # 2) Gradientes finitos con predicción constante en cero (capa outer inicializada a cero
    # y zero_median=True en FinancialPredictor), tanto con pinball como con session_ic_term,
    # y también cuando una columna entera está ausente en todas las filas.
    zero_layer = TechnicalDense(12, cfg, clip=5.0, seed=3, dtype=dtype)
    all_absent = f_before.clone()
    all_absent[:, technical.VERSIONS[version].index("has_market_model")] = 0.0
    leaf = all_absent.requires_grad_(True)
    out = zero_layer(leaf, markets)
    assert torch.count_nonzero(out) == 0
    grads = torch.autograd.grad(out.sum(), (zero_layer.outer.weight, zero_layer.outer.bias, leaf))
    assert all(torch.isfinite(g).all() for g in grads)
    assert grads[0].abs().sum() > 0 and grads[1].abs().sum() > 0

    model, batch, batch_markets = _realistic_predictor(
        version, absent_rows=ABSENT_MASKED, dtype=dtype
    )
    prepared = model.prepare(
        batch,
        model.initial_state(batch.flow_ids, differentiable=True),
        differentiable=True,
        session_complete=True,
    )
    # La predicción puntual arranca exactamente constante en 0.
    assert torch.count_nonzero(prepared.point_predictions) == 0
    targets = torch.linspace(-0.02, 0.02, len(batch.flow_ids), dtype=dtype)
    ic_term = SessionICTerm(weight=3.0, minimum_rows=3)
    sessions_key = tuple((market, 0) for market in batch_markets)
    loss = pinball_loss(prepared.quantiles, targets) + ic_term.term(
        prepared.point_predictions, targets, sessions_key
    )
    params = [
        model.technical_branch.outer.weight,
        model.technical_branch.outer.bias,
        model.head.weight,
    ]
    model_grads = torch.autograd.grad(loss, params)
    assert all(torch.isfinite(g).all() for g in model_grads)
    assert all(g.abs().sum() > 0 for g in model_grads)

    # Con outer abierta, todas las capas de la rama técnica reciben gradiente finito y no nulo.
    with torch.no_grad():
        opened(model.technical_branch)
        model.technical_branch.to(dtype)
        model._seal_parameters()
    prepared_open = model.prepare(
        batch,
        model.initial_state(batch.flow_ids, differentiable=True),
        differentiable=True,
        session_complete=True,
    )
    loss_open = pinball_loss(prepared_open.quantiles, targets)
    branch_grads = torch.autograd.grad(loss_open, list(model.technical_branch.parameters()))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in branch_grads)

    # 3) Rechazar raw_charts / indicators inválidos, NaN en columnas indicadoras o estado corrupto.
    with pytest.raises(ValueError):
        layer(f_before, markets, raw_charts=f_before[:-1])
    with pytest.raises(ValueError):
        layer(
            f_before,
            markets,
            raw_charts=f_before.to(torch.float32 if dtype == torch.float64 else torch.float64),
        )
    with pytest.raises(ValueError):
        layer.ranks(f_before, markets, indicators=f_before[:, :-1])
    with pytest.raises(ValueError, match="presencia"):
        layer.presence(f_before[:, :-1])
    bad_ind = f_before.clone()
    bad_ind[0, technical.VERSIONS[version].index("has_252_sessions")] = float("nan")
    with pytest.raises(ValueError, match="finitos"):
        layer.ranks(f_before, markets, indicators=bad_ind)
    # presence() en modo ranked devuelve todo True; en modo masked_v1 detecta buffers ausentes.
    ranked_layer = TechnicalDense(
        12, replace(cfg, absent_rows="ranked"), clip=5.0, seed=3, dtype=dtype
    )
    assert torch.equal(ranked_layer.presence(f_before), torch.ones_like(f_before, dtype=torch.bool))
    broken = TechnicalDense(12, cfg, clip=5.0, seed=3, dtype=dtype)
    broken._buffers.pop("indicator_index")
    broken.indicator_index = None
    with pytest.raises(ValueError, match="máscara de ausencias"):
        broken(f_before, markets)
    with pytest.raises(ValueError, match="máscara de ausencias"):
        broken.presence(f_before)
