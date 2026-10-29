"""Atención entre activos de la misma sesión, C5 de #751.

Las pruebas calculan salidas y gradientes sobre lotes ficticios, sin optimizador ni pasos.
"""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_financial_adapter import setup
from test_session_rank_objective import model, session_batch

from mars_titan.models.titans.cross_asset_attention import (
    CrossAssetAttention,
    CrossAssetAttentionConfig,
)
from mars_titan.training.search_cases import checked_predictor_changes

VARIANT = CrossAssetAttentionConfig()
CONTROL = CrossAssetAttentionConfig(mode="self_only")
MARKETS = ("US", "CN", "US", "US", "CN", "US")
AT = (7,) * len(MARKETS)


def module(config=VARIANT, *, hidden=16, seed=3):
    """Módulo con la proyección de salida distinta de cero, para que la mezcla se vea."""
    result = CrossAssetAttention(hidden, config, seed=seed, dtype=torch.float64)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(seed + 1)
        result.out.weight.copy_(torch.randn(hidden, hidden, generator=generator) / 4)
    return result


def encoded(rows=6, hidden=16, seed=9):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=generator, dtype=torch.float64)


def reference(layer, values, markets, instants):
    """La misma ecuación fila a fila y cabeza a cabeza, con bucles de Python."""
    heads, width = layer.config.heads, layer.head_dim
    q, k, v = layer.qkv(layer.norm(values)).chunk(3, dim=-1)
    rows = []
    for i in range(len(values)):
        if layer.config.mode == "same_session":
            group = [
                j
                for j in range(len(values))
                if (markets[j], instants[j]) == (markets[i], instants[i])
            ]
        else:
            group = [i]
        parts = []
        for h in range(heads):
            span = slice(h * width, (h + 1) * width)
            scores = torch.stack([q[i, span] @ k[j, span] / math.sqrt(width) for j in group])
            weights = torch.softmax(scores, dim=0)
            parts.append(sum(w * v[j, span] for w, j in zip(weights, group, strict=True)))
        rows.append(torch.cat(parts))
    return values + layer.out(torch.stack(rows))


@pytest.mark.parametrize("config", [VARIANT, CONTROL, CrossAssetAttentionConfig(heads=1)])
def test_the_mixing_matches_a_row_by_row_reference(config):
    layer, values = module(config), encoded()
    instants = (7, 7, 7, 8, 7, 8)
    torch.testing.assert_close(
        layer(values, MARKETS, instants), reference(layer, values, MARKETS, instants)
    )


def test_each_row_only_reads_rows_of_its_market_and_instant():
    layer, values = module(), encoded()
    base = layer(values, MARKETS, AT)
    other_market = values.clone()
    other_market[[1, 4]] += 5
    changed = layer(other_market, MARKETS, AT)
    us = [0, 2, 3, 5]
    assert torch.equal(changed[us], base[us])
    same_market = values.clone()
    same_market[0] += 5
    assert not torch.equal(layer(same_market, MARKETS, AT)[2], base[2])
    # La misma fila en otro instante del bloque deja de influir.
    instants = (7, 7, 7, 7, 7, 8)
    split = layer(values, MARKETS, instants)
    moved = values.clone()
    moved[5] += 5
    assert torch.equal(layer(moved, MARKETS, instants)[[0, 2, 3]], split[[0, 2, 3]])


def test_the_control_reads_only_its_own_row_with_the_same_parameters():
    variant, control, values = module(VARIANT), module(CONTROL), encoded()
    for (name, a), (_, b) in zip(
        variant.state_dict().items(), control.state_dict().items(), strict=True
    ):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), name
    result = control(values, MARKETS, AT)
    # Cambiar las demás filas de su sesión no mueve la fila 0. Las formas son las mismas,
    # así que BLAS calcula la fila igual y la comparación es exacta.
    others = values.clone()
    others[[2, 3, 5]] += 5
    assert torch.equal(control(others, MARKETS, AT)[0], result[0])
    # Con una sola fila por grupo, el peso es uno y la mezcla es W_o W_v n_i.
    value = control.qkv(control.norm(values)).chunk(3, dim=-1)[2]
    torch.testing.assert_close(result, values + control.out(value), rtol=0, atol=0)


def test_permuting_the_rows_permutes_the_outputs():
    layer, values = module(), encoded()
    order = [3, 0, 5, 1, 4, 2]
    expected = layer(values, MARKETS, AT)[order]
    actual = layer(values[order], tuple(MARKETS[i] for i in order), AT)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


def test_gradients_reach_other_rows_of_the_session_only_in_the_variant():
    values = encoded().requires_grad_()
    for config, crosses in ((VARIANT, True), (CONTROL, False)):
        layer = module(config)
        (gradient,) = torch.autograd.grad(layer(values, MARKETS, AT)[0].sum(), values)
        # La fila 0 es de US. Las filas de CN nunca reciben gradiente de ella.
        assert torch.count_nonzero(gradient[[1, 4]]) == 0
        assert bool(torch.count_nonzero(gradient[[2, 3, 5]])) is crosses
    layer = module(CONTROL)
    layer(values, MARKETS, AT).sum().backward()
    query, key, value = layer.qkv.weight.grad.chunk(3)
    # En el control el softmax de un solo elemento no depende de consultas ni claves.
    assert torch.count_nonzero(query) == torch.count_nonzero(key) == 0
    assert torch.count_nonzero(value) > 0 and torch.count_nonzero(layer.out.weight.grad) > 0


def test_disabled_field_keeps_identity_parameters_and_predictions():
    base, batch = setup()
    explicit = model(cross_asset_attention=None)
    assert explicit.config.identity() == base.config.identity()
    assert "cross_asset_attention" not in base.config.identity()
    assert base.cross_asset_attention is None and not base.session_level
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)


@pytest.mark.parametrize("config", [VARIANT, CONTROL])
def test_the_variant_and_its_control_start_as_the_reference(config):
    base, _ = setup()
    batch = session_batch(6, markets=list(MARKETS))
    predictor = model(cross_asset_attention=config)
    reference_parameters = dict(base.named_parameters())
    names = dict(predictor.named_parameters())
    assert set(names) - set(reference_parameters) == {
        "cross_asset_attention.norm.weight",
        "cross_asset_attention.norm.bias",
        "cross_asset_attention.qkv.weight",
        "cross_asset_attention.out.weight",
    }
    for name, value in reference_parameters.items():
        assert torch.equal(names[name], value), name
    assert torch.count_nonzero(predictor.cross_asset_attention.out.weight) == 0
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True
    )
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)
    identity = predictor.config.identity()["cross_asset_attention"]
    assert identity["mode"] == config.mode and identity["position"] == "after_mac_before_head"


def test_c5_needs_the_complete_session_in_one_block():
    batch = session_batch()
    predictor = model(cross_asset_attention=VARIANT)
    assert predictor.session_level
    with pytest.raises(ValueError, match="sesión completa"):
        predictor.prepare(batch, predictor.initial_state(batch.flow_ids))
    predictor.prepare(batch, predictor.initial_state(batch.flow_ids), session_complete=True)


def test_the_predictor_mixes_the_representations_of_its_session():
    batch = session_batch(6, markets=list(MARKETS))
    predictor = model(cross_asset_attention=VARIANT)
    with torch.no_grad():
        predictor.cross_asset_attention.out.weight.fill_(0.05)
    predictor._seal_parameters()
    state = predictor.initial_state(batch.flow_ids)
    prepared = predictor.prepare(batch, state, session_complete=True)
    base, _ = setup()
    plain = base.prepare(batch, base.initial_state(batch.flow_ids))
    layer = predictor.cross_asset_attention
    markets = tuple(flow.split("/", 1)[0] for flow in batch.flow_ids)
    expected = layer(plain.working_state, markets, batch.prediction_at)
    torch.testing.assert_close(prepared.working_state, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        prepared.point_predictions, predictor.head(expected).squeeze(-1), rtol=0, atol=0
    )
    # C5 no toca la memoria: el estado siguiente es el de la referencia.
    torch.testing.assert_close(
        prepared.next_state.mac.memory.weights[0],
        plain.next_state.mac.memory.weights[0],
        rtol=0,
        atol=0,
    )


def test_the_state_dict_round_trips_and_the_control_is_another_identity():
    batch = session_batch()
    source = model(cross_asset_attention=VARIANT)
    with torch.no_grad():
        source.cross_asset_attention.out.weight.fill_(0.1)
    source._seal_parameters()
    target = model(cross_asset_attention=VARIANT)
    target.load_state_dict(source.state_dict())
    expected = source.prepare(batch, source.initial_state(batch.flow_ids), session_complete=True)
    actual = target.prepare(batch, target.initial_state(batch.flow_ids), session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    with pytest.raises(ValueError):
        model(cross_asset_attention=CONTROL).load_state_dict(source.state_dict())


def test_the_predictor_change_registry_admits_it():
    changes = dict(cross_asset_attention=asdict(CONTROL))
    assert checked_predictor_changes(changes) == changes
    base, _ = setup()
    assert replace(base.config, **changes).cross_asset_attention == CONTROL


@pytest.mark.parametrize(
    "value",
    [
        dict(heads=3, mode="same_session"),
        dict(heads=4, mode="all_markets"),
        dict(heads=4),
        dict(heads=True, mode="same_session"),
        "cross_asset_attention_v1",
    ],
)
def test_invalid_declarations_are_rejected(value):
    base, _ = setup()
    with pytest.raises(ValueError):
        replace(base.config, cross_asset_attention=value)


def test_heads_must_divide_the_hidden_size():
    with pytest.raises(ValueError, match="múltiplo"):
        CrossAssetAttention(30, CrossAssetAttentionConfig(heads=4), seed=1)


def test_construction_refuses_other_configurations_and_precisions():
    with pytest.raises(ValueError, match="CrossAssetAttentionConfig"):
        CrossAssetAttention(16, asdict(VARIANT), seed=1)
    with pytest.raises(ValueError, match="FP32 o FP64"):
        CrossAssetAttention(16, VARIANT, seed=1, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "change",
    [
        dict(values=encoded()[:, :8]),
        dict(values=encoded().float()),
        dict(values=encoded()[:0], markets=(), instants=()),
        dict(markets=MARKETS[:-1]),
        dict(instants=AT[:-1]),
    ],
)
def test_each_row_needs_its_representation_market_and_instant(change):
    arguments = dict(values=encoded(), markets=MARKETS, instants=AT) | change
    with pytest.raises(ValueError, match="un mercado y un instante por fila"):
        module()(arguments["values"], arguments["markets"], arguments["instants"])


def test_autocast_is_refused():
    layer = CrossAssetAttention(16, VARIANT, seed=1)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        layer(encoded().float(), MARKETS, AT)


def test_the_source_enters_the_trainer_identity_only_when_active():
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    name = "mars_titan.models.titans.cross_asset_attention"
    trainer = object.__new__(ChronologicalTrainer)
    # La receta decide si F2 y F3 añaden su código a la identidad.
    trainer.recipe = ChronologicalRecipe()
    for config, present in ((None, False), (VARIANT, True)):
        trainer.predictor = SimpleNamespace(
            config=SimpleNamespace(
                temporal_encoder=None,
                regional_variant=None,
                cross_asset_attention=config,
                asset_groups=None,
                holographic_network=None,
                mature_reliability=None,
            )
        )
        assert (name in trainer._code()) is present


def test_declared_configurations_build_with_their_controls():
    root = Path(__file__).resolve().parents[3]
    declaration = json.loads(
        (root / "configs/experiments/cross-asset-attention-v1.json").read_text()
    )
    assert declaration["default_enabled"] is False and declaration["final_test_access"] is False
    base, _ = setup()
    reference_parameters = dict(base.named_parameters())
    identities = set()
    for name, arm in declaration["arms"].items():
        changes = arm["predictor_changes"]
        assert checked_predictor_changes(changes or None) == (changes or None)
        predictor = model(**changes)
        identities.add(json.dumps(predictor.config.identity(), sort_keys=True))
        for key, value in reference_parameters.items():
            assert torch.equal(dict(predictor.named_parameters())[key], value), (name, key)
    assert len(identities) == len(declaration["arms"])
    for contrast in declaration["contrasts"]:
        assert {contrast["variant"], contrast["control"]} <= set(declaration["arms"])


def test_the_frozen_consumer_rejects_c5_until_it_registers_it():
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    with pytest.raises(ValueError, match="atención entre activos"):
        FrozenFinancialConsumer(model(cross_asset_attention=VARIANT))
