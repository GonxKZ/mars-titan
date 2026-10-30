"""Fuentes de rol por fila de B3: mercado, modalidades, sector estadístico y su control.

Cada ruta esperada se deduce de la definición declarada. Los sectores se construyen con
factores de rendimiento conocidos, así que los grupos y su numeración se saben antes de
ejecutar. Ninguna prueba ajusta parámetros ni recorre datos reales.
"""

import hashlib

import numpy as np
import pytest
import torch

from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.regimes import REGIME_RULE, SLOTS, RegimeRule
from mars_titan.memory.role_binding import SOURCES, RoleBindingConfig
from mars_titan.memory.role_sources import (
    LABELS,
    MARKET_RULE,
    MODALITY_RULE,
    PERMUTED_SECTOR_RULE,
    ROW_RULES,
    SECTOR_RULE,
    MarketOccupancy,
    RowRoleRule,
)

AT = 1_615_000_000_000_000
MARKET, MODALITY, SECTOR, PERMUTED = (RowRoleRule(name) for name in ROW_RULES)
# Cada activo sigue el factor de su posición. La numeración esperada no coincide con el factor.
GROUPS = (2, 0, 2, 3, 1, 0, 3, 1, 2, 0, 1, 3)


def flows(market, count):
    return [f"{market}/A{index:03d}" for index in range(count)]


def factor_windows(groups, *, seed=0, absent=(), fill=0.0, constant=(), channels=6):
    """Ventanas [activos, 64, canales] cuyos rendimientos siguen el factor de su grupo.

    El ruido propio es mil veces menor que el factor, así que la correlación dentro de un grupo
    es casi uno y entre grupos la de dos series normales independientes. Las sesiones de
    `absent` faltan en todo el mercado y llevan `fill` en sus cinco canales.
    """
    generator = np.random.default_rng(seed)
    factors = generator.normal(0.0, 0.02, (max(groups) + 1, 63))
    steps = factors[list(groups)] + generator.normal(0.0, 2e-5, (len(groups), 63))
    steps[list(constant)] = 0.0
    closes = np.concatenate([np.zeros((len(groups), 1)), np.cumsum(steps, axis=1)], axis=1)
    values = np.zeros((len(groups), 64, 6))
    values[:, :, 0] = closes - 0.001
    values[:, :, 1] = closes + 0.004
    values[:, :, 2] = closes - 0.004
    values[:, :, 3] = closes
    values[:, :, 4] = 0.1
    values[:, :, 5] = 1.0
    for step in absent:
        values[:, step, :5] = fill
        values[:, step, 5] = 0.0
    return values[:, :, :channels].astype(np.float32)


def numbered(groups, unclassified=()):
    """Ruta de cada activo: los grupos se numeran desde 1 por el menor activo que contienen."""
    first = {}
    for asset, group in enumerate(groups):
        if asset not in unclassified:
            first.setdefault(group, len(first) + 1)
    return [0 if asset in unclassified else first[g] for asset, g in enumerate(groups)]


def permuted(routes, names, seed, market):
    """Rutas del control derivadas de nuevo con el generador declarado en su identidad."""
    text = f"{PERMUTED_SECTOR_RULE}/{seed}/{market}"
    entropy = int.from_bytes(hashlib.sha256(text.encode()).digest()[:16], "little")
    table = dict(zip(names, routes, strict=True))
    eligible = sorted(name for name in names if table[name] != 0)
    order = np.random.default_rng(np.random.SeedSequence(entropy)).permutation(len(eligible))
    shuffled = {eligible[k]: table[eligible[order[k]]] for k in range(len(eligible))}
    return [shuffled.get(name, table[name]) for name in names]


def presence(news, fundamentals):
    values = np.ones((len(news), 5), dtype=np.bool_)
    values[:, 1], values[:, 3] = news, fundamentals
    return values


def test_market_routes_follow_the_flow_prefix_without_reading_prices_or_presence():
    names = ["US/A000", "CN/600000.SS", "US/A001", "CN/000001.SZ", "CN/600519.SS"]
    routes, states = MARKET.routes(None, names, AT)
    assert routes.dtype == np.int64 and routes.tolist() == [1, 2, 1, 2, 2]
    assert states == [
        MarketOccupancy("CN", (0, 0, 3, 0, 0)),
        MarketOccupancy("US", (0, 2, 0, 0, 0)),
    ]
    assert [LABELS[MARKET_RULE][route] for route in (1, 2)] == ["US", "CN"]


def test_modality_routes_count_news_and_fundamentals_only():
    news = np.array([False, True, False, True, True])
    fundamentals = np.array([False, False, True, True, False])
    flags = presence(news, fundamentals)
    routes, states = MODALITY.routes(None, flows("US", 5), AT, flags)
    assert routes.dtype == np.int64 and routes.tolist() == [1, 2, 3, 4, 2]
    assert states == [MarketOccupancy("US", (0, 1, 2, 1, 1))]
    # El macro no distingue filas, aunque cambie entre ellas.
    changed = flags.copy()
    changed[:, 4] = [False, True, False, True, False]
    assert MODALITY.routes(None, flows("US", 5), AT, changed)[0].tolist() == routes.tolist()
    assert LABELS[MODALITY_RULE][4] == "news_and_fundamentals"


@pytest.mark.parametrize(
    "flags",
    [
        None,
        np.ones((5, 5), dtype=np.int64),
        np.ones((5, 4), dtype=np.bool_),
        np.ones((6, 5), dtype=np.bool_),
        np.ones(5, dtype=np.bool_),
    ],
)
def test_modality_needs_five_presence_bits_per_row(flags):
    with pytest.raises(ValueError, match="cinco bits"):
        MODALITY.routes(None, flows("US", 5), AT, flags)


def test_sector_routes_are_the_correlation_groups_numbered_by_their_smallest_asset():
    names = flows("US", len(GROUPS) + 1)
    groups = (*GROUPS, 0)
    prices = factor_windows(groups, constant=(len(GROUPS),))
    routes, states = SECTOR.routes(prices, names, AT, None)
    expected = numbered(groups, unclassified=(len(GROUPS),))
    assert routes.dtype == np.int64 and routes.tolist() == expected
    assert expected[:5] == [1, 2, 1, 3, 4]
    assert states == [MarketOccupancy("US", (1, 3, 3, 3, 3))]


def test_sector_routes_do_not_depend_on_row_order_or_on_other_markets():
    us, cn = factor_windows(GROUPS, seed=1), factor_windows((1, 0, 1, 0, 2, 2, 3, 3), seed=2)
    prices = np.concatenate([us, cn])
    names = flows("US", len(GROUPS)) + flows("CN", 8)
    routes, states = SECTOR.routes(prices, names, AT)
    assert routes[: len(GROUPS)].tolist() == SECTOR.routes(us, names[: len(GROUPS)], AT)[0].tolist()
    assert routes[len(GROUPS) :].tolist() == [1, 2, 1, 2, 3, 3, 4, 4]
    assert [state.market for state in states] == ["CN", "US"]
    order = np.random.default_rng(3).permutation(len(names))
    shuffled, _ = SECTOR.routes(prices[order], [names[i] for i in order], AT)
    assert shuffled.tolist() == routes[order].tolist()


def test_sector_never_reads_absent_sessions_and_accepts_windows_without_presence():
    names = flows("US", len(GROUPS))
    reference = SECTOR.routes(factor_windows(GROUPS), names, AT)[0]
    five = SECTOR.routes(factor_windows(GROUPS, channels=5), names, AT)[0]
    assert five.tolist() == reference.tolist()
    gap = SECTOR.routes(factor_windows(GROUPS, absent=(30,)), names, AT)[0]
    garbage = SECTOR.routes(factor_windows(GROUPS, absent=(30,), fill=5.0), names, AT)[0]
    assert gap.tolist() == garbage.tolist() == reference.tolist()


def test_sector_needs_the_declared_returns_between_present_sessions():
    names = flows("US", len(GROUPS))
    # Un hueco interior anula los dos rendimientos que lo tocan y deja 61.
    prices = factor_windows(GROUPS, absent=(30,))
    enough = RowRoleRule(SECTOR_RULE, min_returns=61).routes(prices, names, AT)[0]
    assert enough.tolist() == numbered(GROUPS)
    short = RowRoleRule(SECTOR_RULE, min_returns=62).routes(prices, names, AT)[0]
    assert short.tolist() == [0] * len(GROUPS)


def test_permuted_sector_keeps_the_routes_and_sizes_with_a_fixed_partner():
    names = flows("US", len(GROUPS) + 1) + flows("CN", 8)
    for seed in (0, 1):
        sector, permuted_routes = [], []
        for data in (0, 7):
            prices = np.concatenate(
                [
                    factor_windows((*GROUPS, 0), seed=data, constant=(len(GROUPS),)),
                    factor_windows((1, 0, 1, 0, 2, 2, 3, 3), seed=data + 1),
                ]
            )
            truth = SECTOR.routes(prices, names, AT)[0].tolist()
            control = RowRoleRule(PERMUTED_SECTOR_RULE, seed=seed).routes(prices, names, AT)[0]
            us, cn = slice(0, len(GROUPS) + 1), slice(len(GROUPS) + 1, None)
            expected = permuted(truth[us], names[us], seed, "US")
            expected += permuted(truth[cn], names[cn], seed, "CN")
            assert control.tolist() == expected
            assert sorted(control[us].tolist()) == sorted(truth[us])
            assert control[len(GROUPS)] == 0
            sector.append(truth)
            permuted_routes.append(control.tolist())
        assert sector[0] != permuted_routes[0]
    assert RowRoleRule(PERMUTED_SECTOR_RULE, seed=1).routes(prices, names, AT)[0].tolist() != (
        RowRoleRule(PERMUTED_SECTOR_RULE, seed=0).routes(prices, names, AT)[0].tolist()
    )


def test_sources_do_not_consume_numpy_or_torch_global_rng():
    numpy_before, torch_before = np.random.get_state(), torch.get_rng_state().clone()
    names = flows("US", len(GROUPS))
    prices, flags = factor_windows(GROUPS), presence([True] * 12, [False] * 12)
    for rule in (MARKET, MODALITY, SECTOR, PERMUTED):
        rule.routes(prices, names, AT, flags)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and numpy_before[2:] == numpy_after[2:]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert torch.equal(torch_before, torch.get_rng_state())


@pytest.mark.parametrize(
    ("names", "at", "message"),
    [
        ([], AT, "corte entero"),
        (flows("US", 12), np.int64(AT), "corte entero"),
        (flows("US", 12), float(AT), "corte entero"),
        (flows("US", 11) + ["EU/X"], AT, "US y CN"),
    ],
)
@pytest.mark.parametrize("rule", [MARKET, MODALITY, SECTOR, PERMUTED], ids=ROW_RULES)
def test_an_event_needs_its_flows_markets_and_integer_cutoff(rule, names, at, message):
    flags = presence([True] * 12, [True] * 12)
    with pytest.raises(ValueError, match=message):
        rule.routes(factor_windows(GROUPS), names, at, flags)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: None, "64 sesiones"),
        (lambda p: p[:, :63], "64 sesiones"),
        (lambda p: p[:, :, :4], "64 sesiones"),
        (lambda p: p[:11], "64 sesiones"),
        (lambda p: p[0], "64 sesiones"),
        (
            lambda p: np.where((np.arange(64) == 20)[:, None] & (np.arange(6) == 3), np.nan, p),
            "no finitos",
        ),
    ],
)
@pytest.mark.parametrize("rule", [SECTOR, PERMUTED], ids=["sector", "permuted"])
def test_malformed_sector_windows_are_rejected(rule, change, message):
    with pytest.raises(ValueError, match=message):
        rule.routes(change(factor_windows(GROUPS)), flows("US", 12), AT)


def test_repeated_flows_are_rejected_by_the_sector():
    names = flows("US", 11) + ["US/A000"]
    with pytest.raises(ValueError, match="ordenados"):
        SECTOR.routes(factor_windows(GROUPS), names, AT)


@pytest.mark.parametrize(
    "options",
    [
        dict(name=REGIME_RULE),
        dict(name="current_sector_v1"),
        dict(name=SECTOR_RULE, min_returns=1),
        dict(name=SECTOR_RULE, min_returns=64),
        dict(name=SECTOR_RULE, min_returns=True),
        dict(name=SECTOR_RULE, min_returns=42.0),
        dict(name=PERMUTED_SECTOR_RULE, seed=-1),
        dict(name=PERMUTED_SECTOR_RULE, seed=2**32),
        dict(name=PERMUTED_SECTOR_RULE, seed=True),
    ],
)
def test_rules_outside_the_declaration_are_rejected(options):
    with pytest.raises(ValueError):
        RowRoleRule(**options)


def test_limits_of_the_declaration_are_accepted():
    assert RowRoleRule(SECTOR_RULE, min_returns=2).min_returns == 2
    assert RowRoleRule(SECTOR_RULE, min_returns=63).min_returns == 63
    assert RowRoleRule(PERMUTED_SECTOR_RULE, seed=2**32 - 1).seed == 2**32 - 1


def test_each_identity_records_its_rule_and_only_its_parameters():
    common = dict(
        schema_version=1,
        kind="row_role_source",
        slots=SLOTS,
        grouping="own_route_for_each_row_of_the_event",
        fit="none_declared_before_execution",
    )
    sector = dict(
        common,
        window="calendar_price_window_v1_of_64_sessions_ending_at_the_decision",
        series="daily_log_close_returns_between_consecutive_present_sessions",
        regions="correlation_regions_average_linkage_on_one_minus_rho_cut_in_4_per_market",
        min_returns=42,
        unclassified="route_0_with_fewer_returns_than_declared_or_a_constant_series",
        numbering="from_1_by_the_smallest_asset_of_each_group",
    )
    assert MARKET.identity() == dict(
        common,
        rule=MARKET_RULE,
        labels=["unclassified", "US", "CN", "unused_3", "unused_4"],
        assignment="1_for_us_flows_2_for_cn_flows",
    )
    assert MODALITY.identity() == dict(
        common,
        rule=MODALITY_RULE,
        labels=list(LABELS[MODALITY_RULE]),
        assignment="1_plus_news_present_plus_2_times_fundamentals_present",
        ignored="prices_and_charts_required_macro_present_in_every_row",
    )
    labels = ["unclassified", "sector_1", "sector_2", "sector_3", "sector_4"]
    assert SECTOR.identity() == dict(sector, rule=SECTOR_RULE, labels=labels)
    assert RowRoleRule(PERMUTED_SECTOR_RULE, min_returns=30, seed=5).identity() == dict(
        sector,
        rule=PERMUTED_SECTOR_RULE,
        labels=labels,
        min_returns=30,
        control="same_routes_and_sizes_permuted_among_assets_with_history",
        generator="numpy_seed_sequence_of_sha256_of_rule_seed_and_market",
        seed=5,
    )
    assert all(len(LABELS[name]) == SLOTS for name in ROW_RULES)


def test_the_occupancy_record_names_each_route():
    state = MarketOccupancy("US", (1, 3, 0, 2, 0))
    assert state.record(AT, LABELS[SECTOR_RULE]) == dict(
        at=AT,
        market="US",
        assets=6,
        occupancy=dict(unclassified=1, sector_1=3, sector_2=0, sector_3=2, sector_4=0),
    )


@pytest.mark.parametrize("source", list(SOURCES))
def test_each_binding_source_builds_its_declared_rule(source):
    correction = MatureCorrection(
        AssociativeMemoryConfig("proximal"), role_binding=RoleBindingConfig(source=source)
    )
    kind = RowRoleRule if SOURCES[source] in ROW_RULES else RegimeRule
    assert type(correction.routing) is kind and correction.routing.name == SOURCES[source]
    assert correction.identity()["routing"] == correction.routing.identity()


@pytest.mark.parametrize(
    ("source", "routing"),
    [
        ("sector", RegimeRule(REGIME_RULE)),
        ("sector", RowRoleRule(PERMUTED_SECTOR_RULE)),
        ("regime", RowRoleRule(MARKET_RULE)),
    ],
)
def test_a_routing_rule_of_another_source_is_rejected(source, routing):
    with pytest.raises(ValueError, match="Solo una clave enrutada"):
        MatureCorrection(
            AssociativeMemoryConfig("proximal"),
            role_binding=RoleBindingConfig(source=source),
            routing=routing,
        )


def test_the_correction_forwards_the_presence_to_its_source():
    correction = MatureCorrection(
        AssociativeMemoryConfig("proximal"), role_binding=RoleBindingConfig(source="modality")
    )
    flags = presence([True, False, True], [True, True, False])
    routes, states = correction.routes(None, flows("US", 3), AT, flags)
    assert routes.tolist() == [4, 3, 2] and states == [MarketOccupancy("US", (0, 0, 1, 1, 1))]
    plain = MatureCorrection(AssociativeMemoryConfig("proximal"))
    assert plain.routes(None, flows("US", 3), AT, flags) == (None, [])
