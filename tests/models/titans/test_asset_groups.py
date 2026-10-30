"""Grupos de activos de la sesión con la ventana pasada, H4 de #783.

Las pruebas calculan salidas y gradientes sobre lotes ficticios, sin optimizador ni pasos.
"""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.data.asset_groups import session_memberships
from mars_titan.data.sector_map import SectorMap
from mars_titan.models.titans.asset_groups import (
    ASSET_GROUP_MEMBERSHIP,
    AssetGroupMembership,
    AssetGroupsConfig,
)
from mars_titan.models.titans.cross_asset_attention import CrossAssetAttentionConfig
from mars_titan.models.titans.fractal_shared_operator import FractalSharedOperatorConfig
from mars_titan.training.search_cases import checked_predictor_changes
from tests.data.test_asset_group_measures import sectors
from tests.models.test_price_presence_families import titans_specification
from tests.models.titans.test_financial_adapter import DIMENSIONS, api, specification

VARIANT = AssetGroupsConfig()
SMALL = AssetGroupsConfig(buckets=3, clusters=2, iterations=8)
SHUFFLED = replace(SMALL, mode="shuffled_rows")
SECTOR = AssetGroupsConfig(
    mode="current_sector", buckets=None, clusters=None, min_observations=None, iterations=None
)
MARKETS = ("US", "CN", "US", "US", "CN", "US", "US", "CN", "US", "US", "CN", "US")
FLOWS = tuple(f"{market}/A{index:03}" for index, market in enumerate(MARKETS))
ROWS = len(MARKETS)
AT = (7,) * ROWS
START = 1_609_459_200_000_000


def sector_map():
    groups = ("CN/Banks", "CN/unknown", "US/Energy", "US/Tech", "US/unknown")
    assignments = tuple(
        (flow, (0 if index % 2 else 1) if flow.startswith("CN") else 2 + index % 3)
        for index, flow in enumerate(FLOWS)
    )
    return SectorMap(groups, tuple(sorted(assignments)), (("fixture.csv", "0" * 64),))


def module(config=SMALL, *, hidden=16, seed=3):
    """Módulo con la proyección de salida distinta de cero, para que los grupos se vean."""
    sectors_map = sector_map() if config.mode == "current_sector" else None
    result = AssetGroupMembership(
        hidden, config, seed=seed, sector_map=sectors_map, dtype=torch.float64
    )
    with torch.no_grad():
        generator = torch.Generator().manual_seed(seed + 1)
        values = torch.randn(hidden, hidden, generator=generator, dtype=torch.float64)
        result.out.weight.copy_(values / 4)
    return result


def encoded(rows=ROWS, hidden=16, seed=9):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=generator, dtype=torch.float64)


def windows(rows=ROWS):
    return sectors(rows=rows, groups=3)[0]


def reference(layer, values, prices, flows, markets, instants):
    """La misma ecuación fila a fila y grupo a grupo, con bucles de Python."""
    normalized = layer.norm(values)
    mixed = torch.zeros_like(values)
    for key in dict.fromkeys(zip(markets, instants, strict=True)):
        members = [i for i in range(len(values)) if (markets[i], instants[i]) == key]
        membership, families = layer.memberships(prices[members], [flows[i] for i in members], key)
        for position, row in enumerate(members):
            total = torch.zeros(values.shape[1], dtype=values.dtype)
            for group in range(membership.shape[1]):
                weight = membership[:, group].sum()
                if membership[position, group] == 0:
                    continue
                pooled = sum(
                    membership[other, group] * normalized[j] for other, j in enumerate(members)
                )
                state = layer.outer(F.silu(layer.inner(pooled / weight)))
                total = total + membership[position, group] * state
            mixed[row] = total / max(families[position].item(), 1.0)
    return values + layer.out(mixed)


@pytest.mark.parametrize("config", [SMALL, SHUFFLED, SECTOR], ids=["past", "shuffled", "sector"])
def test_the_groups_match_a_row_by_row_reference(config):
    layer, values, prices = module(config), encoded(), windows()
    instants = (7, 7, 7, 8, 7, 8, 8, 7, 7, 7, 7, 8)
    torch.testing.assert_close(
        layer(values, prices, FLOWS, MARKETS, instants),
        reference(layer, values, prices, FLOWS, MARKETS, instants),
    )


def test_each_row_only_reads_rows_of_its_market_and_instant():
    layer, values, prices = module(), encoded(), windows()
    base = layer(values, prices, FLOWS, MARKETS, AT)
    china = [i for i, market in enumerate(MARKETS) if market == "CN"]
    usa = [i for i, market in enumerate(MARKETS) if market == "US"]
    other = values.clone()
    other[china] += 5
    assert torch.equal(layer(other, prices, FLOWS, MARKETS, AT)[usa], base[usa])
    moved = prices.clone()
    moved[china, :, :5] *= 3
    assert torch.equal(layer(values, moved, FLOWS, MARKETS, AT)[usa], base[usa])
    same = values.clone()
    same[usa[0]] += 5
    assert not torch.equal(layer(same, prices, FLOWS, MARKETS, AT)[usa[1]], base[usa[1]])


def test_the_memberships_are_the_ones_of_the_past_window():
    layer, prices = module(), windows()
    usa = [i for i, market in enumerate(MARKETS) if market == "US"]
    membership, families = layer.memberships(prices[usa], [FLOWS[i] for i in usa], ("US", 7))
    expected = session_memberships(
        prices[usa], buckets=3, clusters=2, min_observations=36, iterations=8
    )
    assert torch.equal(membership, expected)
    assert families.tolist() == [5.0] * len(usa)


def test_the_controls_have_the_same_parameters_and_the_shuffle_permutes_whole_rows():
    variant, shuffled, sector = module(SMALL), module(SHUFFLED), module(SECTOR)
    for other in (shuffled, sector):
        for (name, a), (_, b) in zip(
            variant.state_dict().items(), other.state_dict().items(), strict=True
        ):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b), name
    prices, flows = windows(8), FLOWS[:8]
    plain, _ = variant.memberships(prices, flows, ("US", 7))
    mixed, families = shuffled.memberships(prices, flows, ("US", 7))
    order = shuffled._permutation(("US", 7), 8)
    assert sorted(order.tolist()) == list(range(8)) and order.tolist() != list(range(8))
    # La permutación sale de la semilla del predictor, del mercado y del instante.
    digest = hashlib.sha256(f"{ASSET_GROUP_MEMBERSHIP}/3/US/7".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:7], "little"))
    assert torch.equal(order, torch.randperm(8, generator=generator))
    assert torch.equal(mixed, plain[order])
    assert families.tolist() == [5.0] * 8
    assert torch.equal(shuffled.memberships(prices, flows, ("US", 7))[0], mixed)
    # Otra sesión, otra permutación.
    assert not torch.equal(shuffled._permutation(("US", 8), 8), order)


def test_the_sector_control_groups_by_the_current_sector():
    layer = module(SECTOR)
    prices = windows()
    membership, families = layer.memberships(prices, FLOWS, ("US", 7))
    table = sector_map()
    for row, flow in enumerate(FLOWS):
        expected = torch.zeros(len(table.groups), dtype=torch.float64)
        expected[table.group(flow)] = 1
        assert torch.equal(membership[row], expected)
    assert families.tolist() == [1.0] * len(FLOWS)
    # El control no lee la ventana: cambiarla no cambia su salida.
    values = encoded()
    base = layer(values, prices, FLOWS, MARKETS, AT)
    assert torch.equal(layer(values, prices * 3, FLOWS, MARKETS, AT), base)


def test_a_row_with_fewer_families_reads_the_mean_of_its_own():
    layer, prices = module(), windows()
    # La fila 2 no tiene rango en ninguna sesión, así que le falta el diferencial.
    prices[2, :, 1] = prices[2, :, 2] = prices[2, :, 3]
    usa = [i for i, market in enumerate(MARKETS) if market == "US"]
    membership, families = layer.memberships(prices[usa], [FLOWS[i] for i in usa], ("US", 7))
    assert families.tolist() == [4.0 if row == 2 else 5.0 for row in usa]
    values = encoded()
    torch.testing.assert_close(
        layer(values, prices, FLOWS, MARKETS, AT),
        reference(layer, values, prices, FLOWS, MARKETS, AT),
    )


def test_the_weights_come_from_a_derived_seed_without_the_global_generator():
    torch.random.default_generator.manual_seed(11)
    before = torch.random.get_rng_state()
    layer = AssetGroupMembership(16, SMALL, seed=3, dtype=torch.float64)
    assert torch.equal(torch.random.get_rng_state(), before)
    digest = hashlib.sha256(f"{ASSET_GROUP_MEMBERSHIP}/3".encode()).digest()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int.from_bytes(digest[:7], "little"))
        inner = nn.Linear(16, 16, dtype=torch.float64)
        outer = nn.Linear(16, 16, dtype=torch.float64)
    assert torch.equal(layer.inner.weight, inner.weight)
    assert torch.equal(layer.outer.weight, outer.weight)
    other = AssetGroupMembership(16, SMALL, seed=4, dtype=torch.float64)
    assert not torch.equal(other.inner.weight, layer.inner.weight)


def test_a_row_without_history_keeps_its_representation_exactly():
    layer, values, prices = module(), encoded(), windows()
    prices[0, :40] = 0
    result = layer(values, prices, FLOWS, MARKETS, AT)
    assert torch.equal(result[0], values[0])
    # Sus vecinos sí reciben los grupos, sin ella.
    assert not torch.equal(result[2], values[2])


def test_gradients_reach_the_rows_of_the_session_and_never_the_window():
    values = encoded().requires_grad_()
    prices = windows().requires_grad_()
    layer = module()
    output = layer(values, prices, FLOWS, MARKETS, AT)
    (gradient,) = torch.autograd.grad(output[0].sum(), values, retain_graph=True)
    china = [i for i, market in enumerate(MARKETS) if market == "CN"]
    assert torch.count_nonzero(gradient[china]) == 0
    assert torch.count_nonzero(gradient[[2, 3, 5]]) > 0
    # Las pertenencias son constantes de la ventana: no hay camino del gradiente hasta ella.
    output.sum().backward()
    assert prices.grad is None


def test_at_construction_only_the_output_projection_receives_gradient():
    layer = AssetGroupMembership(16, SMALL, seed=3, dtype=torch.float64)
    assert torch.count_nonzero(layer.out.weight) == 0
    values, prices = encoded(), windows()
    result = layer(values, prices, FLOWS, MARKETS, AT)
    assert torch.equal(result, values)
    result.square().sum().backward()
    assert torch.count_nonzero(layer.out.weight.grad) > 0
    for name in ("inner", "outer"):
        assert torch.count_nonzero(getattr(layer, name).weight.grad) == 0, name


def test_permuting_the_rows_permutes_the_outputs():
    layer, values, prices = module(), encoded(), windows()
    order = [3, 0, 5, 1, 4, 2, 11, 10, 9, 8, 7, 6]
    expected = layer(values, prices, FLOWS, MARKETS, AT)[order]
    actual = layer(
        values[order], prices[order], [FLOWS[i] for i in order], [MARKETS[i] for i in order], AT
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def priced_batch(rows=8, *, markets=None, at=START, seed=5):
    """Lote de un instante con ventanas de sectores sintéticos y el bit de presencia."""
    generator = np.random.default_rng(seed)
    markets = markets or ["US"] * rows
    values = {
        name: generator.normal(size=(rows, width)).astype(np.float32).round(3)
        for name, width in DIMENSIONS.items()
        if name != "prices"
    }
    values["prices"] = windows(rows).numpy().astype(np.float32)
    values["fundamentals"][:, 1:] = [1, 0.5]
    values["macro"][:] = [0.2, 1, 0]
    names = [f"{market}/A{index:03}" for index, market in enumerate(markets)]
    raw = dict(
        inputs=values,
        presence=np.ones((rows, 5), dtype=np.bool_),
        sample_ids=[f"{name}/{at}" for name in names],
        market=markets,
        prediction_at=np.full(rows, at, dtype="datetime64[us]"),
        input_available_at=np.full(rows, at - 1, dtype="datetime64[us]"),
    )
    return api().DecisionBatch.from_corpus(raw, titans_specification(), dtype=torch.float64)


def model(**changes):
    config = api().FinancialConfig(
        titans_specification(), variant="mac_online", hidden_size=32, layers=1, seed=42
    )
    return api().FinancialPredictor(replace(config, **changes), dtype=torch.float64)


def changes(config):
    result = dict(asset_groups=config)
    if config.mode == "current_sector":
        result.update(sector_map=sector_map())
    return result


def test_disabled_field_keeps_identity_parameters_and_predictions():
    base, explicit = model(), model(asset_groups=None)
    batch = priced_batch()
    assert explicit.config.identity() == base.config.identity()
    assert "asset_groups" not in base.config.identity()
    assert base.asset_groups is None and not base.session_level
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)


@pytest.mark.parametrize("config", [VARIANT, SHUFFLED, SECTOR], ids=["h4", "shuffled", "sector"])
def test_h4_and_its_controls_start_as_the_reference(config):
    base, predictor = model(), model(**changes(config))
    batch = priced_batch()
    reference_parameters = dict(base.named_parameters())
    names = dict(predictor.named_parameters())
    assert set(names) - set(reference_parameters) == {
        f"asset_groups.{name}"
        for name in (
            "norm.weight",
            "norm.bias",
            "inner.weight",
            "inner.bias",
            "outer.weight",
            "outer.bias",
            "out.weight",
        )
    }
    for name, value in reference_parameters.items():
        assert torch.equal(names[name], value), name
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True
    )
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)
    identity = predictor.config.identity()["asset_groups"]
    assert identity["kind"] == ASSET_GROUP_MEMBERSHIP and identity["mode"] == config.mode
    assert identity["point_in_time"] is (config.mode != "current_sector")
    if config.mode == "current_sector":
        assert identity["sectors"] == sector_map().identity()


def test_h4_needs_the_complete_session_in_one_block():
    batch = priced_batch()
    predictor = model(asset_groups=SMALL)
    assert predictor.session_level
    with pytest.raises(ValueError, match="sesión completa"):
        predictor.prepare(batch, predictor.initial_state(batch.flow_ids))
    predictor.prepare(batch, predictor.initial_state(batch.flow_ids), session_complete=True)


def test_the_predictor_adds_the_groups_of_the_received_window():
    batch = priced_batch(markets=["US", "CN"] * 4)
    predictor = model(asset_groups=SMALL)
    with torch.no_grad():
        predictor.asset_groups.out.weight.fill_(0.05)
    predictor._seal_parameters()
    prepared = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True
    )
    base = model()
    plain = base.prepare(batch, base.initial_state(batch.flow_ids))
    markets = tuple(flow.split("/", 1)[0] for flow in batch.flow_ids)
    expected = predictor.asset_groups(
        plain.working_state, batch.inputs["prices"], batch.flow_ids, markets, batch.prediction_at
    )
    torch.testing.assert_close(prepared.working_state, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        prepared.point_predictions, predictor.head(expected).squeeze(-1), rtol=0, atol=0
    )
    # H4 no toca la memoria: el estado siguiente es el de la referencia.
    torch.testing.assert_close(
        prepared.next_state.mac.memory.weights[0],
        plain.next_state.mac.memory.weights[0],
        rtol=0,
        atol=0,
    )


def test_the_groups_read_the_received_window_and_not_the_session_ranks():
    from mars_titan.models.titans.session_ranks import SESSION_RANK_INPUTS

    batch = priced_batch()
    predictor = model(asset_groups=SMALL, session_rank_inputs=SESSION_RANK_INPUTS)
    seen = []
    original = predictor.asset_groups.forward

    def spy(encoded, prices, *arguments):
        seen.append(prices)
        return original(encoded, prices, *arguments)

    predictor.asset_groups.forward = spy
    predictor.prepare(batch, predictor.initial_state(batch.flow_ids), session_complete=True)
    assert len(seen) == 1 and torch.equal(seen[0], batch.inputs["prices"])


def test_the_state_dict_round_trips_and_each_mode_is_another_identity():
    batch = priced_batch()
    source = model(asset_groups=SMALL)
    with torch.no_grad():
        source.asset_groups.out.weight.fill_(0.1)
    source._seal_parameters()
    target = model(asset_groups=SMALL)
    target.load_state_dict(source.state_dict())
    expected = source.prepare(batch, source.initial_state(batch.flow_ids), session_complete=True)
    actual = target.prepare(batch, target.initial_state(batch.flow_ids), session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    for config in (SHUFFLED, SECTOR, replace(SMALL, buckets=4)):
        with pytest.raises(ValueError):
            model(**changes(config)).load_state_dict(source.state_dict())
    # El módulo comprueba su propia identidad, también fuera del predictor.
    layer = module(SMALL)
    module(SMALL).load_state_dict(layer.state_dict())
    for config in (SHUFFLED, SECTOR, replace(SMALL, buckets=4)):
        with pytest.raises(ValueError, match="contrato del módulo"):
            module(config).load_state_dict(layer.state_dict())


def test_the_predictor_change_registry_admits_it():
    declared = dict(asset_groups=asdict(SHUFFLED))
    assert checked_predictor_changes(declared) == declared
    assert model(**declared).config.asset_groups == SHUFFLED


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (dict(asdict(VARIANT), mode="random"), "modo de H4"),
        (dict(asdict(VARIANT), buckets=1), "cubos"),
        (dict(asdict(VARIANT), buckets=True), "cubos"),
        (dict(asdict(VARIANT), clusters=17), "grupos de correlación"),
        (dict(asdict(VARIANT), min_observations=64), "observaciones"),
        (dict(asdict(VARIANT), iterations=0), "iteraciones"),
        (dict(asdict(VARIANT), buckets=None), "cubos"),
        (dict(asdict(SECTOR), clusters=8), "control sectorial"),
        ({"mode": "past_window"}, "toda su configuración"),
        ("asset_group_membership_v1", "toda su configuración"),
    ],
)
def test_invalid_declarations_are_rejected(value, message):
    with pytest.raises(ValueError, match=message):
        model(asset_groups=value)


def test_the_sector_map_comes_only_with_the_sector_control():
    with pytest.raises(ValueError, match="sector_map"):
        model(asset_groups=SECTOR)
    with pytest.raises(ValueError, match="sector_map"):
        model(asset_groups=SMALL, sector_map=sector_map())
    assert model(**changes(SECTOR)).asset_groups.sector_map == sector_map()


def test_h4_needs_the_presence_window_and_does_not_combine_with_c3_c5_or_f5():
    from mars_titan.models.titans.auxiliary_horizons import AuxiliaryHorizons

    with pytest.raises(ValueError, match="bit de presencia"):
        api().FinancialConfig(specification(), variant="mac_online", asset_groups=SMALL)
    with pytest.raises(ValueError, match="H4 y C5"):
        model(asset_groups=SMALL, cross_asset_attention=CrossAssetAttentionConfig())
    with pytest.raises(ValueError, match="H4 y C3"):
        model(asset_groups=SMALL, fractal_shared_operator=FractalSharedOperatorConfig())
    with pytest.raises(ValueError, match="operaciones por sesión"):
        model(asset_groups=SMALL, auxiliary_horizons=AuxiliaryHorizons())


def test_construction_refuses_other_configurations_precisions_and_sector_maps():
    with pytest.raises(ValueError, match="AssetGroupsConfig"):
        AssetGroupMembership(16, asdict(SMALL), seed=1)
    with pytest.raises(ValueError, match="FP32 o FP64"):
        AssetGroupMembership(16, SMALL, seed=1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="reparto de sectores"):
        AssetGroupMembership(16, SECTOR, seed=1)
    with pytest.raises(ValueError, match="reparto de sectores"):
        AssetGroupMembership(16, SMALL, seed=1, sector_map=sector_map())
    with pytest.raises(ValueError, match="reparto de sectores"):
        AssetGroupMembership(16, SECTOR, seed=1, sector_map={"source": "fichas"})


@pytest.mark.parametrize(
    "change",
    [
        dict(values=encoded()[:, :8]),
        dict(values=encoded().float()),
        dict(values=encoded()[:0], prices=windows()[:0], flows=(), markets=(), instants=()),
        dict(prices=windows()[:, :, :5]),
        dict(prices=windows().float()),
        dict(prices=windows()[:-1]),
        dict(flows=FLOWS[:-1]),
        dict(markets=MARKETS[:-1]),
        dict(instants=AT[:-1]),
    ],
)
def test_each_row_needs_its_representation_window_flow_market_and_instant(change):
    arguments = (
        dict(values=encoded(), prices=windows(), flows=FLOWS, markets=MARKETS, instants=AT) | change
    )
    with pytest.raises(ValueError, match="un mercado y un instante por fila"):
        module()(*arguments.values())


def test_autocast_is_refused():
    layer = AssetGroupMembership(16, SMALL, seed=1)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        layer(encoded().float(), windows().float(), FLOWS, MARKETS, AT)


def test_the_sources_enter_the_trainer_identity_only_when_active():
    from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer

    names = ("mars_titan.models.titans.asset_groups", "mars_titan.data.asset_groups")
    trainer = object.__new__(ChronologicalTrainer)
    # La receta decide si F2 y F3 añaden su código a la identidad.
    trainer.recipe = ChronologicalRecipe()
    for config, present in ((None, False), (SMALL, True)):
        trainer.predictor = SimpleNamespace(
            config=SimpleNamespace(
                temporal_encoder=None,
                regional_variant=None,
                cross_asset_attention=None,
                asset_groups=config,
                holographic_network=None,
                axial_time_region=None,
                fractal_shared_operator=None,
                mature_reliability=None,
                three_factor_bridge=None,
            )
        )
        code = trainer._code()
        assert all((name in code) is present for name in names)


def test_the_frozen_consumer_rejects_h4_until_it_registers_it():
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    with pytest.raises(ValueError, match="grupos de activos"):
        FrozenFinancialConsumer(model(asset_groups=SMALL))


def test_dust_refuses_h4():
    from mars_titan.training.dust.financial import FinancialDustAdapter

    FinancialDustAdapter(model(variant="transformer_direct"))
    with pytest.raises(ValueError, match="sin ampliaciones"):
        FinancialDustAdapter(model(variant="transformer_direct", asset_groups=SMALL))


def test_the_mars_titan_parents_refuse_h4(tmp_path):
    from mars_titan.data.storage import sha256
    from mars_titan.training import mars_titan_walk_forward as walk
    from mars_titan.training.titans_walk_forward import KIND

    request = dict(predictor_changes=dict(asset_groups=asdict(SMALL)))
    with pytest.raises(ValueError, match="grupos de activos de H4"):
        walk._frozen_parent(
            dict(predictor=dict(dtype="float64")), None, 7, "cpu", tmp_path, dict(request=request)
        )
    view = tmp_path / "view.json"
    view.write_text("{}", encoding="utf-8")
    report = dict(
        kind=KIND,
        status="completed",
        final_test_opened=False,
        checkpoint={},
        request=dict(request, variant="mac_online", seed=7, view_sha256=sha256(view)),
    )
    (tmp_path / "run.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="grupos de activos de H4"):
        walk._parent(tmp_path, view, 7)


def test_declared_models_build_with_their_controls():
    root = Path(__file__).resolve().parents[3]
    declaration = json.loads(
        (root / "configs/experiments/asset-group-membership-v1.json").read_text()
    )
    assert declaration["default_enabled"] is False and declaration["final_test_access"] is False
    base = model()
    reference_parameters = dict(base.named_parameters())
    identities = set()
    for name, arm in declaration["arms"].items():
        declared = arm["predictor_changes"]
        assert checked_predictor_changes(declared or None) == (declared or None)
        options = dict(declared)
        if (options.get("asset_groups") or {}).get("mode") == "current_sector":
            options.update(sector_map=sector_map())
        predictor = model(**options)
        identities.add(json.dumps(predictor.config.identity(), sort_keys=True))
        for key, value in reference_parameters.items():
            assert torch.equal(dict(predictor.named_parameters())[key], value), (name, key)
    assert len(identities) == len(declaration["arms"])
    for contrast in declaration["contrasts"]:
        assert {contrast["variant"], contrast["control"]} <= set(declaration["arms"])
