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
