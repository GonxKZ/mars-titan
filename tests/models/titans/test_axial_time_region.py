"""Atención axial en el tiempo y entre regiones, C2 de #751.

Las pruebas calculan salidas, gradientes y estados sobre lotes ficticios, sin optimizador ni
pasos de ajuste.
"""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_financial_adapter import DIMENSIONS, api, setup, specification
from test_session_rank_objective import AT, model

from mars_titan.models.titans.axial_time_region import (
    AXIAL_TIME_REGION,
    REGIONS,
    TIME_ONLY,
    AxialTimeRegion,
    AxialTimeRegionConfig,
    RegionHistory,
    join_history_rows,
    map_history_rows,
    parse_axial_time_region,
)
from mars_titan.training.search_cases import checked_predictor_changes

VARIANT = AxialTimeRegionConfig()
CONTROL = AxialTimeRegionConfig(kind=TIME_ONLY)
DAY = 86_400_000_000
HIDDEN = 16


def layer(config=VARIANT, *, hidden=HIDDEN, seed=3):
    """Módulo con las dos proyecciones de salida distintas de cero, para que se vean."""
    result = AxialTimeRegion(hidden, config, seed=seed, dtype=torch.float64)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(seed + 1)
        for weight in (result.time_out.weight, result.second_out.weight):
            weight.copy_(torch.randn(hidden, hidden, generator=generator, dtype=torch.float64) / 4)
    return result


def inputs(rows=3, *, hidden=HIDDEN, window=8, seed=9):
    """Bloques actuales, presencia e historial con huecos y regiones ausentes."""
    generator = torch.Generator().manual_seed(seed)
    blocks = torch.randn(rows, REGIONS, hidden, generator=generator, dtype=torch.float64)
    presence = torch.ones(rows, REGIONS, dtype=torch.bool)
    presence[0, [1, 3]] = False
    presence[2, 4] = False
    blocks = blocks * presence.unsqueeze(-1)
    past = torch.randn(rows, window - 1, REGIONS, hidden, generator=generator, dtype=torch.float64)
    present = torch.rand(rows, window - 1, REGIONS, generator=generator) > 0.3
    # El flujo 1 solo lleva dos emisiones: lo anterior queda ausente.
    present[1, : window - 3] = False
    return blocks, presence, RegionHistory(past * present.unsqueeze(-1), present)


def reference(module, blocks, presence, history):
    """Las ecuaciones de C2 flujo a flujo, región a región y cabeza a cabeza."""
    config, size = module.config, module.hidden_size
    window, heads, width = config.window, config.heads, module.head_dim
    if presence is None:
        presence = torch.ones(blocks.shape[:2], dtype=torch.bool)
    rho, lag = module.region_embedding, module.lag_embedding
    q1, k1, v1 = module.time_qkv.weight.split(size)
    q2, k2, v2 = module.second_qkv.weight.split(size)

    def attend(query, keys, values):
        parts = []
        for h in range(heads):
            span = slice(h * width, (h + 1) * width)
            scores = torch.stack([query[span] @ key[span] for key in keys]) / math.sqrt(width)
            weights = torch.softmax(scores, dim=0)
            parts.append(sum(w * value[span] for w, value in zip(weights, values, strict=True)))
        return torch.cat(parts)

    output = blocks.clone()
    for f in range(len(blocks)):
        sequence = [history.blocks[f, p] for p in range(window - 1)] + [blocks[f]]
        mask = [history.present[f, p] for p in range(window - 1)] + [presence[f]]
        tokens = [sequence[w] + rho + lag[window - 1 - w] for w in range(window)]
        updated = []
        for r in range(REGIONS):
            allowed = [w for w in range(window) if mask[w][r] or w == window - 1]
            normed = [module.time_norm(tokens[w][r]) for w in allowed]
            read = attend(q1 @ normed[-1], [k1 @ n for n in normed], [v1 @ n for n in normed])
            updated.append(blocks[f, r] + module.time_out(read))
        for r in range(REGIONS):
            if not presence[f, r]:
                continue
            if config.kind == AXIAL_TIME_REGION:
                mixed = [module.second_norm(updated[s] + rho[s]) for s in range(REGIONS)]
                allowed = [s for s in range(REGIONS) if presence[f, s] or s == r]
                second = attend(
                    q2 @ mixed[r],
                    [k2 @ mixed[s] for s in allowed],
                    [v2 @ mixed[s] for s in allowed],
                )
            else:
                allowed = [w for w in range(window - 1) if mask[w][r]]
                again = [module.second_norm(tokens[w][r]) for w in allowed]
                again.append(module.second_norm(updated[r] + rho[r] + lag[0]))
                second = attend(q2 @ again[-1], [k2 @ n for n in again], [v2 @ n for n in again])
            output[f, r] = updated[r] + module.second_out(second)
    following = RegionHistory(
        torch.cat((history.blocks, blocks.unsqueeze(1)), dim=1)[:, 1:],
        torch.cat((history.present, presence.unsqueeze(1)), dim=1)[:, 1:],
    )
    return output, following


@pytest.mark.parametrize(
    "config",
    [VARIANT, CONTROL, AxialTimeRegionConfig(heads=1), AxialTimeRegionConfig(TIME_ONLY, 3, 2)],
)
def test_the_layer_matches_a_reference_written_position_by_position(config):
    module = layer(config)
    blocks, presence, history = inputs(window=config.window)
    result, following = module(blocks, presence, history)
    expected, shifted = reference(module, blocks, presence, history)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(following.blocks, shifted.blocks)
    assert torch.equal(following.present, shifted.present)
    assert following.blocks.is_contiguous() and following.present.is_contiguous()


def test_without_presence_every_region_counts_as_present():
    module = layer()
    blocks, _, history = inputs()
    result, following = module(blocks, None, history)
    expected, _ = reference(module, blocks, None, history)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    assert following.present[:, -1].all()


@pytest.mark.parametrize("config", [VARIANT, CONTROL])
def test_absent_positions_of_the_past_are_never_read(config):
    module = layer(config)
    blocks, presence, history = inputs()
    base, _ = module(blocks, presence, history)
    noisy = history.blocks + 7 * (~history.present).unsqueeze(-1)
    result, _ = module(blocks, presence, RegionHistory(noisy, history.present))
    assert torch.equal(result, base)
    # Un valor presente del pasado sí cambia la emisión de su región.
    changed = history.blocks.clone()
    flow, position, region = (history.present.nonzero()[0]).tolist()
    changed[flow, position, region] += 1
    moved, _ = module(blocks, presence, RegionHistory(changed, history.present))
    assert not torch.equal(moved[flow, region], base[flow, region])


@pytest.mark.parametrize("config", [VARIANT, CONTROL])
def test_an_absent_region_keeps_its_block_and_sends_nothing(config):
    module = layer(config)
    blocks, presence, history = inputs()
    base, _ = module(blocks, presence, history)
    # El flujo 0 no tiene noticias en esta emisión: su bloque sale intacto.
    assert torch.equal(base[0, 1], blocks[0, 1])
    # Su historial de noticias no llega a las demás regiones de la emisión.
    changed = history.blocks.clone()
    changed[0, :, 1] += 3
    result, _ = module(blocks, presence, RegionHistory(changed, history.present))
    assert torch.equal(result[0, [0, 2, 4]], base[0, [0, 2, 4]])


@pytest.mark.parametrize("config", [VARIANT, CONTROL])
def test_a_region_absent_now_and_in_all_its_past_stays_finite_and_intact(config):
    """Sin ninguna posición presente, la consulta propia evita un softmax vacío."""
    module = layer(config)
    blocks, presence, _ = inputs()
    empty = module.initial_history(3, device="cpu")
    result, _ = module(blocks, presence, empty)
    assert torch.isfinite(result).all()
    assert torch.equal(result[0, 1], blocks[0, 1]) and torch.equal(result[2, 4], blocks[2, 4])
    expected, _ = reference(module, blocks, presence, empty)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


def test_only_the_axial_layer_mixes_regions():
    blocks, presence, history = inputs()
    changed = history.blocks.clone()
    changed[2, :, 0] += 2
    current = blocks.clone()
    current[2, 0] += 2
    for config, crosses in ((VARIANT, True), (CONTROL, False)):
        module = layer(config)
        base, _ = module(blocks, presence, history)
        for past, now in ((changed, blocks), (history.blocks, current)):
            result, _ = module(now, presence, RegionHistory(past, history.present))
            assert bool((result[2, 1:4] != base[2, 1:4]).any()) is crosses


def test_rows_are_computed_independently():
    module = layer()
    blocks, presence, history = inputs()
    base, following = module(blocks, presence, history)
    order = [2, 0, 1]
    permuted = map_history_rows(history, lambda value: value[order])
    result, moved = module(blocks[order], presence[order], permuted)
    torch.testing.assert_close(result, base[order], rtol=1e-14, atol=1e-14)
    assert torch.equal(moved.blocks, following.blocks[order])
    single, _ = module(blocks[:1], presence[:1], map_history_rows(history, lambda v: v[:1]))
    torch.testing.assert_close(single, base[:1], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("config", [VARIANT, CONTROL])
def test_zero_projections_return_the_blocks_exactly(config):
    module = AxialTimeRegion(HIDDEN, config, seed=3, dtype=torch.float64)
    assert torch.count_nonzero(module.time_out.weight) == 0
    assert torch.count_nonzero(module.second_out.weight) == 0
    blocks, presence, history = inputs()
    result, _ = module(blocks, presence, history)
    assert torch.equal(result, blocks)


def test_the_variant_and_its_control_share_parameters_and_initialization():
    variant, control = (AxialTimeRegion(HIDDEN, c, seed=3) for c in (VARIANT, CONTROL))
    left, right = variant.state_dict(), control.state_dict()
    assert [name for name in left if name.endswith("_extra_state")] == ["_extra_state"]
    for name, value in left.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, right[name]), name
    assert sum(p.numel() for p in variant.parameters()) == sum(
        p.numel() for p in control.parameters()
    )
    # Otra semilla da otros pesos y la construcción no consume el generador global.
    state = torch.random.get_rng_state()
    other = AxialTimeRegion(HIDDEN, VARIANT, seed=4)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert not torch.equal(other.time_qkv.weight, variant.time_qkv.weight)


@pytest.mark.parametrize("config", [VARIANT, CONTROL])
def test_gradients_reach_every_parameter_and_the_stored_history(config):
    module = layer(config)
    blocks, presence, history = inputs()
    past = history.blocks.clone().requires_grad_()
    result, following = module(
        blocks.requires_grad_(), presence, RegionHistory(past, history.present)
    )
    result.square().sum().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0, name
    # El gradiente llega al pasado presente y nunca a las posiciones ausentes.
    assert torch.count_nonzero(past.grad * (~history.present).unsqueeze(-1)) == 0
    assert torch.count_nonzero(past.grad) > 0
    # El historial siguiente conserva el grafo de los bloques actuales.
    assert following.blocks.requires_grad


def test_the_gradient_keeps_the_zero_projections_learning():
    module = AxialTimeRegion(HIDDEN, VARIANT, seed=3, dtype=torch.float64)
    blocks, presence, history = inputs()
    result, _ = module(blocks, presence, history)
    result.square().sum().backward()
    assert torch.count_nonzero(module.time_out.weight.grad) > 0
    assert torch.count_nonzero(module.second_out.weight.grad) > 0


def test_counts_reject_history_older_than_the_first_emission():
    module = layer()
    _, _, history = inputs()
    steps = torch.tensor([10, 2, 7])
    module.validate_counts(history, steps)
    wrong = history.present.clone()
    wrong[1, 0, 0] = True
    with pytest.raises(ValueError, match="primera emisión"):
        module.validate_counts(RegionHistory(history.blocks, wrong), steps)
    empty = module.initial_history(3)
    module.validate_counts(empty, torch.zeros(3, dtype=torch.int64))
    assert not empty.present.any() and torch.count_nonzero(empty.blocks) == 0


@pytest.mark.parametrize(
    "change",
    [
        lambda h: None,
        lambda h: (h.blocks, h.present),
        lambda h: RegionHistory(h.blocks[:, 1:], h.present[:, 1:]),
        lambda h: RegionHistory(h.blocks[:2], h.present[:2]),
        lambda h: RegionHistory(h.blocks.float(), h.present),
        lambda h: RegionHistory(h.blocks, h.present.long()),
        lambda h: RegionHistory(h.blocks.to_sparse(), h.present),
        lambda h: RegionHistory(h.blocks[..., :8], h.present),
    ],
)
def test_a_malformed_history_is_refused(change):
    module = layer()
    blocks, presence, history = inputs()
    with pytest.raises(ValueError, match="historial de C2"):
        module(blocks, presence, change(history))


@pytest.mark.parametrize(
    "change",
    [
        dict(blocks=inputs()[0][:, :4]),
        dict(blocks=inputs()[0].float()),
        dict(blocks=inputs()[0][:0]),
        dict(presence=inputs()[1].long()),
        dict(presence=inputs()[1][:2]),
    ],
)
def test_blocks_and_presence_need_their_contract(change):
    module = layer()
    blocks, presence, history = inputs()
    arguments = dict(blocks=blocks, presence=presence) | change
    with pytest.raises(ValueError, match="C2"):
        module(arguments["blocks"], arguments["presence"], history)


def test_autocast_is_refused():
    module = AxialTimeRegion(HIDDEN, VARIANT, seed=1)
    blocks, presence, history = inputs()
    history = map_history_rows(
        history, lambda value: value.float() if value.is_floating_point() else value
    )
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        module(blocks.float(), presence, history)


@pytest.mark.parametrize(
    "value",
    [
        dict(kind="axial_v2", window=8, heads=4),
        dict(kind=AXIAL_TIME_REGION, window=1, heads=4),
        dict(kind=AXIAL_TIME_REGION, window=33, heads=4),
        dict(kind=AXIAL_TIME_REGION, window=8, heads=3),
        dict(kind=AXIAL_TIME_REGION, window=8.0, heads=4),
        dict(kind=AXIAL_TIME_REGION, window=8),
        AXIAL_TIME_REGION,
    ],
)
def test_invalid_declarations_are_rejected(value):
    with pytest.raises(ValueError):
        parse_axial_time_region(value)
    base, _ = setup()
    with pytest.raises(ValueError):
        replace(base.config, axial_time_region=value)


def test_construction_refuses_other_configurations_and_precisions():
    with pytest.raises(ValueError, match="AxialTimeRegionConfig"):
        AxialTimeRegion(HIDDEN, asdict(VARIANT), seed=1)
    with pytest.raises(ValueError, match="múltiplo"):
        AxialTimeRegion(30, VARIANT, seed=1)
    with pytest.raises(ValueError, match="FP32 o FP64"):
        AxialTimeRegion(HIDDEN, VARIANT, seed=1, dtype=torch.bfloat16)


def test_loading_weights_checks_the_declared_configuration():
    variant, control = (AxialTimeRegion(HIDDEN, c, seed=1) for c in (VARIANT, CONTROL))
    copied = AxialTimeRegion(HIDDEN, VARIANT, seed=2)
    copied.load_state_dict(variant.state_dict())
    assert torch.equal(copied.time_qkv.weight, variant.time_qkv.weight)
    # Los pesos tienen la misma forma, pero la configuración del control es otra.
    with pytest.raises(ValueError, match="contrato del módulo"):
        control.load_state_dict(variant.state_dict())


def test_sealing_refuses_a_c2_module_other_than_the_declared_one():
    predictor = model(axial_time_region=VARIANT)
    original, size = predictor.axial_time_region, predictor.config.hidden_size
    others = (
        None,
        AxialTimeRegion(size, CONTROL, seed=1, dtype=torch.float64),
        AxialTimeRegion(size, VARIANT, seed=1, dtype=torch.float32),
    )
    for module in others:
        predictor.axial_time_region = module
        with pytest.raises(ValueError, match="C2 no corresponde a la configuración declarada"):
            predictor._seal_parameters()
    predictor.axial_time_region = original
    predictor._seal_parameters()
    # Un módulo sin declaración tampoco se admite.
    plain = model()
    plain.axial_time_region = original
    with pytest.raises(ValueError, match="C2 no corresponde a la configuración declarada"):
        plain._seal_parameters()


def test_the_identity_and_the_operation_counts_follow_the_kind():
    variant, control = (AxialTimeRegion(HIDDEN, c, seed=1) for c in (VARIANT, CONTROL))
    assert variant.config.identity()["second_layer"].startswith("attention_across")
    assert control.config.identity()["second_layer"].startswith("second_causal")
    assert variant.operation_counts(10) == dict(axial_time_pairs=400, axial_second_pairs=250)
    assert control.operation_counts(10) == dict(axial_time_pairs=400, axial_second_pairs=400)
    assert variant.bytes_per_flow() == 7 * 5 * HIDDEN * 4 + 35
    assert parse_axial_time_region(asdict(CONTROL)) == CONTROL
    assert parse_axial_time_region(None) is None


def test_joining_and_mapping_rows_keep_the_order():
    _, _, history = inputs()
    parts = [map_history_rows(history, lambda value, i=i: value[i : i + 1]) for i in range(3)]
    joined = join_history_rows(parts)
    assert torch.equal(joined.blocks, history.blocks)
    assert torch.equal(joined.present, history.present)


def batch(rows=4, *, step=0, seed=5, absent=()):
    """Lote de un instante con los mismos flujos en cada paso y ausencias elegidas."""
    generator = np.random.default_rng(seed + step)
    at = AT + step * DAY
    values = {
        name: generator.normal(size=(rows, 64, width) if name == "prices" else (rows, width))
        .astype(np.float32)
        .round(3)
        for name, width in DIMENSIONS.items()
    }
    values["fundamentals"][:, 1:] = [1, 0.5]
    values["macro"][:, 1:] = [1, 0]
    # Precios y gráficos son obligatorios. Las ausencias solo afectan al resto.
    presence = np.ones((rows, 5), dtype=np.bool_)
    for row, column in absent:
        presence[row, column] = False
    for name, column in (("news", 1), ("fundamentals", 3), ("macro", 4)):
        values[name][~presence[:, column]] = 0
    names = [f"US/A{index:03}" for index in range(rows)]
    raw = dict(
        inputs=values,
        presence=presence,
        sample_ids=[f"{name}/{at}" for name in names],
        market=["US"] * rows,
        prediction_at=np.full(rows, at, dtype="datetime64[us]"),
        input_available_at=np.full(rows, at - 1, dtype="datetime64[us]"),
    )
    return api().DecisionBatch.from_corpus(raw, specification(), dtype=torch.float64)


ABSENT = [(), ((0, 1), (2, 3)), ((1, 4),), ((0, 1), (3, 4))]


def run(predictor, steps=4):
    """Recorrido de varios pasos con el estado de cada uno, sin gradientes."""
    state = predictor.initial_state(batch().flow_ids)
    results = []
    for step in range(steps):
        prepared = predictor.prepare(batch(step=step, absent=ABSENT[step % 4]), state)
        results.append(prepared)
        state = prepared.next_state
    return results, state


def moved(config, value=0.05):
    """Predictor con C2 y sus proyecciones de salida distintas de cero."""
    predictor = model(axial_time_region=config)
    with torch.no_grad():
        predictor.axial_time_region.time_out.weight.fill_(value)
        predictor.axial_time_region.second_out.weight.fill_(-value)
    predictor._seal_parameters()
    return predictor


def test_disabled_field_keeps_identity_parameters_and_predictions():
    base, _ = setup()
    explicit = model(axial_time_region=None)
    assert explicit.config.identity() == base.config.identity()
    assert "axial_time_region" not in base.config.identity()
    assert explicit.axial_time_region is None
    state = explicit.initial_state(batch().flow_ids)
    assert state.regions is None and "regions" not in explicit.export_state(state)
    expected, _ = run(base)
    actual, _ = run(explicit)
    for left, right in zip(actual, expected, strict=True):
        assert torch.equal(left.point_predictions, right.point_predictions)


@pytest.mark.parametrize("config", [VARIANT, CONTROL])
def test_each_kind_starts_as_the_reference_at_every_step(config):
    base, _ = setup()
    predictor = model(axial_time_region=config)
    reference_parameters = dict(base.named_parameters())
    names = dict(predictor.named_parameters())
    assert {name.split(".")[0] for name in set(names) - set(reference_parameters)} == {
        "axial_time_region"
    }
    for name, value in reference_parameters.items():
        assert torch.equal(names[name], value), name
    expected, expected_state = run(base)
    actual, state = run(predictor)
    for left, right in zip(actual, expected, strict=True):
        assert torch.equal(left.point_predictions, right.point_predictions)
        assert torch.equal(left.working_state, right.working_state)
    assert torch.equal(state.mac.memory.weights[0], expected_state.mac.memory.weights[0])
    assert predictor.config.identity()["axial_time_region"]["kind"] == config.kind


def test_the_history_keeps_the_last_window_minus_one_emissions():
    config = AxialTimeRegionConfig(window=3)
    predictor = moved(config)
    state = predictor.initial_state(batch().flow_ids)
    blocks = []
    for step in range(4):
        current = batch(step=step, absent=ABSENT[step])
        state = predictor.prepare(current, state).next_state
        price, _ = predictor.price_encoder.encode_last(current.inputs["prices"])
        captured = {}

        def keep(values, presence, captured=captured):
            captured["blocks"] = values
            return values

        from mars_titan.models.baselines.multimodal import fuse_modalities

        fuse_modalities(
            price,
            current.inputs,
            predictor.encoders,
            predictor.fusion,
            current.presence,
            modulation=keep,
        )
        blocks.append((captured["blocks"], current.presence))
    history = state.regions
    assert history.blocks.shape == (4, 2, REGIONS, 32)
    for position, (values, presence) in enumerate(blocks[-2:]):
        torch.testing.assert_close(history.blocks[:, position], values, rtol=0, atol=0)
        assert torch.equal(history.present[:, position], presence)
    assert state.observed_steps.tolist() == [4] * 4
    predictor.axial_time_region.validate_counts(history, state.observed_steps)


def test_the_history_changes_the_prediction_once_the_projections_move():
    predictor = moved(VARIANT)
    results, _ = run(predictor, steps=3)
    fresh = predictor.initial_state(batch().flow_ids)
    alone = predictor.prepare(batch(step=2, absent=ABSENT[2]), fresh)
    # La misma emisión sin pasado da otra predicción: el historial interviene.
    assert not torch.equal(alone.point_predictions, results[2].point_predictions)


def test_the_state_round_trips_through_export_restore_and_gather():
    predictor = moved(VARIANT)
    _, state = run(predictor, steps=3)
    following = batch(step=3, absent=ABSENT[3])
    expected = predictor.prepare(following, state)
    restored = predictor.restore_state(predictor.export_state(state))
    assert torch.equal(restored.regions.blocks, state.regions.blocks)
    actual = predictor.prepare(following, restored)
    assert torch.equal(actual.point_predictions, expected.point_predictions)
    payload = predictor.export_state_cpu(state)
    assert set(payload["regions"]) == {"schema_version", "blocks", "present"}
    order = list(reversed(state.flow_ids))
    gathered = predictor.gather_state(
        {flow: (payload, state.flow_ids.index(flow)) for flow in order}, order
    )
    selected = predictor.select_state(state, order)
    assert torch.equal(gathered.regions.blocks, selected.regions.blocks)
    assert torch.equal(gathered.regions.present, selected.regions.present)
    usage = predictor.state_usage(state)
    base, _ = setup()
    plain = base.state_usage(run(base, steps=3)[1])
    history = state.regions.blocks.numel() * 8 + state.regions.present.numel()
    assert usage["tensor_bytes"] - plain["tensor_bytes"] == history


def test_a_payload_with_an_impossible_history_is_refused():
    predictor = moved(VARIANT)
    state = predictor.initial_state(batch().flow_ids)
    payload = predictor.export_state(state)
    payload["regions"]["present"][0, 0, 0] = True
    with pytest.raises(ValueError, match="primera emisión"):
        predictor.restore_state(payload)
    payload = predictor.export_state(state)
    payload.pop("regions")
    with pytest.raises(ValueError, match="campos incompatibles"):
        predictor.restore_state(payload)


def test_a_state_without_its_history_is_refused():
    predictor = moved(VARIANT)
    state = predictor.initial_state(batch().flow_ids)
    missing = replace(state, regions=None)
    with pytest.raises(ValueError, match="historial de C2"):
        predictor.prepare(batch(), missing)
    # La exportación y el recuento de bytes también lo exigen, sin pasar por el módulo.
    for refuse in (predictor.export_state, predictor.state_usage):
        with pytest.raises(ValueError, match="historial de C2"):
            refuse(missing)
    base, _ = setup()
    plain = base.initial_state(batch().flow_ids)
    with pytest.raises(ValueError, match="Sin C2"):
        base.prepare(batch(), replace(plain, regions=state.regions))


def test_row_by_row_tokens_refuse_c2():
    predictor = moved(VARIANT)
    current = batch()
    with pytest.raises(ValueError, match="fila a fila"):
        predictor.fused_tokens(current.inputs, current.presence)


def test_the_run_engine_detaches_and_gathers_the_history():
    from mars_titan.training.financial_run import FlowStates, _detached

    predictor = moved(VARIANT)
    state = predictor.initial_state(batch().flow_ids, differentiable=True)
    prepared = predictor.prepare(batch(), state, differentiable=True)
    following = prepared.next_state
    assert following.regions.blocks.requires_grad
    assert not _detached(following).regions.blocks.requires_grad
    states = FlowStates()
    states.put(following)
    order = list(reversed(following.flow_ids))
    gathered = states.gather(order)
    assert torch.equal(gathered.regions.blocks, following.regions.blocks[[3, 2, 1, 0]])
    assert torch.equal(gathered.regions.present, following.regions.present[[3, 2, 1, 0]])
    # Con todas las filas en su orden se reutiliza el mismo estado.
    assert states.gather(list(following.flow_ids)) is following
    states.detach()
    assert not states.gather(order).regions.blocks.requires_grad


def test_the_predictor_change_registry_admits_it():
    changes = dict(axial_time_region=asdict(CONTROL))
    assert checked_predictor_changes(changes) == changes
    base, _ = setup()
    assert replace(base.config, **changes).axial_time_region == CONTROL


@pytest.mark.parametrize(
    "changes",
    [
        dict(variant="mac_frozen"),
        dict(presence_modulation="film_v1"),
        dict(
            regional_variant=dict(
                operator=dict(
                    topology="ring",
                    routing="attention",
                    cycles=1,
                    local_steps=1,
                    summary=True,
                    local_gain=0.25,
                    summary_gain=0.25,
                ),
                mixing="none",
            )
        ),
        dict(auxiliary_horizons=dict(horizons=[5], weight=0.5, maturity_slack=0)),
    ],
)
def test_it_is_exclusive_with_other_modulations_and_row_wise_tokens(changes):
    base, _ = setup()
    with pytest.raises(ValueError):
        replace(base.config, axial_time_region=VARIANT, **changes)


def test_it_does_not_combine_with_the_holographic_network():
    # H1 necesita la edición con el catálogo de sus familias, que su prueba ya construye.
    from test_holographic_network import H1, api, specification

    options = dict(variant="mac_online", hidden_size=32, holographic_network=H1)
    api().FinancialConfig(specification(), **options)
    with pytest.raises(ValueError, match="ni la red holográfica"):
        api().FinancialConfig(specification(), axial_time_region=VARIANT, **options)


def test_the_source_enters_the_trainer_identity_only_when_active():
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    name = "mars_titan.models.titans.axial_time_region"
    trainer = object.__new__(ChronologicalTrainer)
    trainer.recipe = ChronologicalRecipe()
    for config, present in ((None, False), (VARIANT, True)):
        trainer.predictor = SimpleNamespace(
            config=SimpleNamespace(
                temporal_encoder=None,
                regional_variant=None,
                cross_asset_attention=None,
                holographic_network=None,
                axial_time_region=config,
            )
        )
        assert (name in trainer._code()) is present


def test_the_frozen_consumer_rejects_c2_until_it_registers_it():
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    with pytest.raises(ValueError, match="historial de C2"):
        FrozenFinancialConsumer(model(axial_time_region=VARIANT))


def test_the_frozen_parent_loader_refuses_it_before_building(tmp_path, monkeypatch):
    from mars_titan.training import mars_titan_walk_forward as walk

    document = dict(predictor=dict(hidden_size=32, dtype="float64"))
    report = dict(request=dict(predictor_changes=dict(axial_time_region=asdict(VARIANT))))

    def forbidden(*_args, **_options):
        pytest.fail("Se construyó o leyó un padre con C2 sin registro congelado")

    monkeypatch.setattr(walk, "FinancialPredictor", forbidden)
    monkeypatch.setattr(walk, "read_manifest", forbidden)
    with pytest.raises(ValueError, match="historial de regiones de C2"):
        walk._frozen_parent(document, specification(), 42, "cpu", tmp_path, report)


@pytest.mark.parametrize("in_recipe", (False, True))
def test_the_parent_declaration_is_refused_before_checkpoint_verification(
    tmp_path, monkeypatch, in_recipe
):
    import hashlib

    from mars_titan.training import mars_titan_walk_forward as walk

    view = tmp_path / "view.json"
    view.write_text("{}\n")
    predictor = dict(hidden_size=32, dtype="float64")
    request = dict(
        variant="mac_online", seed=42, view_sha256=hashlib.sha256(view.read_bytes()).hexdigest()
    )
    if in_recipe:
        predictor["axial_time_region"] = asdict(VARIANT)
    else:
        request["predictor_changes"] = dict(axial_time_region=asdict(VARIANT))
    report = dict(
        kind=walk.TITANS_KIND,
        status="completed",
        final_test_opened=False,
        checkpoint={},
        request=request,
        identity=dict(recipe=dict(predictor=predictor)),
    )
    (tmp_path / "run.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        walk, "_verify", lambda *_a: pytest.fail("Se verificó el checkpoint de un padre con C2")
    )
    with pytest.raises(ValueError, match="historial de regiones de C2"):
        walk._parent(tmp_path, view, 42)


def test_declared_configurations_build_with_their_controls():
    root = Path(__file__).resolve().parents[3]
    declaration = json.loads(
        (root / "configs/experiments/axial-time-region-v1.json").read_text(encoding="utf-8")
    )
    assert declaration["default_enabled"] is False and declaration["final_test_access"] is False
    base, _ = setup()
    reference_parameters = dict(base.named_parameters())
    identities, counts = set(), {}
    for name, arm in declaration["arms"].items():
        changes = arm["predictor_changes"]
        assert checked_predictor_changes(changes or None) == (changes or None)
        predictor = model(**changes)
        identities.add(json.dumps(predictor.config.identity(), sort_keys=True))
        counts[name] = sum(p.numel() for p in predictor.parameters())
        for key, value in reference_parameters.items():
            assert torch.equal(dict(predictor.named_parameters())[key], value), (name, key)
    assert len(identities) == len(declaration["arms"])
    assert counts["axial_time_region"] == counts["time_only_control"] > counts["reference"]
    for contrast in declaration["contrasts"]:
        assert {contrast["variant"], contrast["control"]} <= set(declaration["arms"])
