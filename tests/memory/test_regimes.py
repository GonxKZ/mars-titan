"""Regla de régimen observable, control de calendario y claves enrutadas de B6.

Las ventanas se construyen a mano con rendimientos de mercado conocidos, así que cada ruta
esperada se deduce de la definición declarada y no de una ejecución anterior. Las claves
enrutadas se comparan con cinco memorias proximales independientes, una por compartimento.
Ninguna prueba ajusta parámetros ni recorre datos reales.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.memory.associative_memory import (
    CORRECTION_KEYS,
    KEY_SIZES,
    ROUTING,
    AssociativeMemory,
    AssociativeMemoryConfig,
    MatureCorrection,
)
from mars_titan.memory.regimes import (
    CALENDAR_RULE,
    REGIME_RULE,
    SLOTS,
    UNCLASSIFIED,
    MarketState,
    RegimeRule,
)

ASSETS = 5
REGIME = RegimeRule(REGIME_RULE, min_assets=ASSETS)
CALENDAR = RegimeRule(CALENDAR_RULE, min_assets=ASSETS)
AT = int(datetime(2021, 3, 15, 20, tzinfo=UTC).timestamp() * 1_000_000)
CALM, WILD, DRIFT = 0.002, 0.03, 0.004
# Ruta esperada de cada escenario: (tendencia, volatilidad creciente).
SCENARIOS = {
    "calm_up": (1, False, 1),
    "calm_down": (-1, False, 2),
    "turbulent_up": (1, True, 3),
    "turbulent_down": (-1, True, 4),
}


def micros(*moment):
    return int(datetime(*moment, tzinfo=UTC).timestamp() * 1_000_000)


def market_returns(trend, turbulent):
    """63 rendimientos alternos con escala distinta antes y después de los 21 últimos."""
    earlier, recent = (CALM, WILD) if turbulent else (WILD, CALM)
    signs = (-1.0) ** np.arange(63)
    scale = np.where(np.arange(63) < 42, earlier, recent)
    return scale * signs + trend * DRIFT


def windows(returns, assets=ASSETS, *, absent=(), spread=1e-3, fill=0.0, outlier=None):
    """Ventanas [activos, 64, 6] cuyo rendimiento de cierre mediano es `returns`.

    Cada activo suma una desviación simétrica, así que la mediana es la del mercado. Las
    sesiones de `absent` faltan en todo el mercado y llevan `fill` en sus cinco canales,
    aunque el contrato real las deja en +0.0, para comprobar que la regla no las lee.
    """
    steps = (
        np.asarray(returns, dtype=np.float64)[None, :]
        + np.linspace(-spread, spread, assets)[:, None]
    )
    if outlier is not None:
        steps[0] = outlier
    closes = np.concatenate([np.zeros((assets, 1)), np.cumsum(steps, axis=1)], axis=1)
    values = np.zeros((assets, 64, 6))
    values[:, :, 0] = closes - 0.001
    values[:, :, 1] = closes + 0.004
    values[:, :, 2] = closes - 0.004
    values[:, :, 3] = closes
    values[:, :, 4] = 0.1
    values[:, :, 5] = 1.0
    for step in absent:
        values[:, step, :5] = fill
        values[:, step, 5] = 0.0
    return values.astype(np.float32)


def scenario(name, **options):
    trend, turbulent, _ = SCENARIOS[name]
    return windows(market_returns(trend, turbulent), **options)


def flows(market, count=ASSETS):
    return [f"{market}/A{index:03d}" for index in range(count)]


@pytest.mark.parametrize("name", SCENARIOS)
def test_volatility_and_trend_give_the_declared_route(name):
    state = REGIME.state("US", scenario(name), AT)
    trend, turbulent, route = SCENARIOS[name]
    assert state.route == route and state.assets == ASSETS and state.returns == 63
    assert (state.recent_rms > state.earlier_rms) is turbulent
    assert (state.trend > 0) is (trend > 0)
    assert REGIME.labels[route] == name


def test_the_state_follows_the_declared_formulas():
    prices = scenario("turbulent_up")
    closes = prices[:, :, 3].astype(np.float64)
    market = np.median(np.diff(closes, axis=1), axis=0)
    state = REGIME.state("US", prices, AT)
    assert state.trend == float(np.median(closes[:, -1] - closes[:, 0]))
    assert state.recent_rms == float(np.sqrt(np.mean(market[-21:] ** 2)))
    assert state.earlier_rms == float(np.sqrt(np.mean(market[:-21] ** 2)))


def test_the_market_proxy_is_the_median_and_one_asset_cannot_flip_it():
    # Un activo con +0,5 por sesión cambiaría el signo de la media, no el de la mediana.
    prices = scenario("calm_down", outlier=np.full(63, 0.5))
    closes = prices[:, :, 3].astype(np.float64)
    assert np.mean(np.diff(closes, axis=1), axis=0).sum() > 0
    assert REGIME.state("US", prices, AT).route == 2


def test_the_trend_is_the_median_window_return_and_not_the_sum_of_daily_medians():
    # Cada día dos de cinco activos suben 0,01 y tres bajan 0,001, rotando. La mediana diaria
    # es siempre -0,001 y su suma -0,063, pero cada activo acumula unos +0,21 en la ventana.
    steps = np.full((ASSETS, 63), -0.001)
    for day in range(63):
        steps[[day % ASSETS, (day + 1) % ASSETS], day] = 0.01
    closes = np.concatenate([np.zeros((ASSETS, 1)), np.cumsum(steps, axis=1)], axis=1)
    prices = np.zeros((ASSETS, 64, 6))
    prices[:, :, 3], prices[:, :, 5] = closes, 1.0
    daily = np.median(np.diff(closes, axis=1), axis=0)
    assert np.all(daily < 0) and daily.sum() < 0
    state = REGIME.state("US", prices, AT)
    assert state.trend == float(np.median(closes[:, -1])) > 0.2
    assert REGIME.labels[state.route].endswith("_up")


def test_ties_count_as_calm_and_down():
    # Sin rendimientos la tendencia vale cero y las dos volatilidades también.
    flat = REGIME.state("US", windows(np.zeros(63), spread=0.0), AT)
    assert (flat.route, flat.trend, flat.recent_rms, flat.earlier_rms) == (2, 0.0, 0.0, 0.0)
    # Con ±0,5 las dos partes tienen exactamente la misma volatilidad y la suma es +0,5.
    equal = REGIME.state("US", windows(0.5 * (-1.0) ** np.arange(63), spread=0.0), AT)
    assert equal.recent_rms == equal.earlier_rms == 0.5 and equal.trend == 0.5
    assert equal.route == 1


def test_market_wide_absences_are_chained_and_their_fill_is_never_read():
    absent = (10, 11, 40)
    base = REGIME.state("US", scenario("turbulent_down", absent=absent), AT)
    for fill in (3.75, -1e3):
        assert REGIME.state("US", scenario("turbulent_down", absent=absent, fill=fill), AT) == base
    prices = scenario("turbulent_down", absent=absent)
    present = np.setdiff1d(np.arange(64), absent)
    closes = prices[:, present, 3].astype(np.float64)
    market = np.median(np.diff(closes, axis=1), axis=0)
    assert base.returns == len(market) == 60
    assert base.trend == float(np.median(closes[:, -1] - closes[:, 0]))
    assert base.recent_rms == float(np.sqrt(np.mean(market[-21:] ** 2)))


@pytest.mark.parametrize("assets", [5, 6])
def test_the_route_does_not_depend_on_the_order_of_the_rows(assets):
    prices = scenario("calm_up", assets=assets)
    state = REGIME.state("US", prices, AT)
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(assets)
        assert REGIME.state("US", prices[order], AT) == state


def test_cohorts_below_the_declared_minimums_stay_unclassified():
    assert REGIME.state("US", scenario("calm_up", assets=ASSETS - 1), AT).route == UNCLASSIFIED
    # 43 sesiones presentes dan 42 rendimientos, el mínimo. Con una ausencia más falta uno.
    enough = REGIME.state("US", scenario("calm_up", absent=tuple(range(21))), AT)
    short = REGIME.state("US", scenario("calm_up", absent=tuple(range(22))), AT)
    assert enough.returns == 42 and enough.route != UNCLASSIFIED
    assert short == MarketState("US", UNCLASSIFIED, ASSETS, 41, None, None, None)


def test_the_calendar_control_keeps_the_unclassified_rows_and_ignores_the_market():
    unclassified = [
        scenario("calm_up", assets=ASSETS - 1),
        scenario("calm_up", absent=tuple(range(22))),
    ]
    for prices in unclassified:
        assert CALENDAR.state("US", prices, AT).route == UNCLASSIFIED
    # Marzo de 2021: (2021·12 + 2) mod 4 = 2, ruta 3, con cualquier estado del mercado.
    assert {CALENDAR.state("US", scenario(name), AT).route for name in SCENARIOS} == {3}
    months = [micros(2021, month, 15) for month in range(1, 6)]
    assert [CALENDAR.state("CN", scenario("calm_up"), at).route for at in months] == [
        1,
        2,
        3,
        4,
        1,
    ]


def test_the_calendar_month_is_the_utc_month():
    before, after = micros(2021, 1, 31, 23, 30), micros(2021, 2, 1, 0, 30)
    assert CALENDAR.state("CN", scenario("calm_up"), before).route == 1
    assert CALENDAR.state("CN", scenario("calm_up"), after).route == 2


def test_an_event_is_grouped_by_market_once():
    us, cn = scenario("calm_up"), scenario("turbulent_down")
    prices = np.empty((10, 64, 6), dtype=np.float32)
    prices[0::2], prices[1::2] = us, cn
    names = [flow for pair in zip(flows("US"), flows("CN"), strict=True) for flow in pair]
    routes, states = REGIME.routes(prices, names, AT)
    assert routes.dtype == np.int64 and routes.tolist() == [1, 4] * 5
    assert [(s.market, s.route, s.assets) for s in states] == [("CN", 4, 5), ("US", 1, 5)]
    # Cada mercado se resume con todas sus filas: con medio evento quedaría sin clasificar.
    assert REGIME.routes(prices[:4], names[:4], AT)[0].tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p[:, :, :5], "canal de presencia"),
        (lambda p: p[:, :63], "canal de presencia"),
        (lambda p: np.where(np.arange(6) == 5, 0.5, p), "cero o uno"),
        (lambda p: np.concatenate([p[:, :, :5], np.zeros((5, 64, 1))], axis=2), "decisión"),
        (
            lambda p: np.where((np.arange(64) == 63)[:, None] & (np.arange(6) == 5), 0, p),
            "decisión",
        ),
    ],
)
def test_malformed_windows_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        REGIME.state("US", change(scenario("calm_up").copy()), AT)


def test_windows_with_their_own_absences_or_non_finite_closes_are_rejected():
    prices = scenario("calm_up")
    prices[2, 5, :5], prices[2, 5, 5] = 0.0, 0.0
    with pytest.raises(ValueError, match="no comparten"):
        REGIME.state("US", prices, AT)
    prices = scenario("calm_up")
    prices[1, 30, 3] = np.nan
    with pytest.raises(ValueError, match="no finitos"):
        REGIME.state("US", prices, AT)


@pytest.mark.parametrize(
    ("names", "at", "message"),
    [
        (flows("EU"), AT, "US y CN"),
        (flows("US", ASSETS - 1), AT, "corte del evento"),
        (flows("US"), np.int64(AT), "corte del evento"),
        (flows("US"), float(AT), "corte del evento"),
    ],
)
def test_an_event_needs_its_flows_markets_and_integer_cutoff(names, at, message):
    with pytest.raises(ValueError, match=message):
        REGIME.routes(scenario("calm_up"), names, at)


@pytest.mark.parametrize(
    "options",
    [
        dict(name="hmm"),
        dict(name=REGIME_RULE, min_assets=0),
        dict(name=REGIME_RULE, min_assets=True),
        dict(name=REGIME_RULE, recent_returns=0),
        dict(name=REGIME_RULE, recent_returns=42, min_returns=42),
        dict(name=REGIME_RULE, min_returns=64),
    ],
)
def test_rules_outside_the_declaration_are_rejected(options):
    with pytest.raises(ValueError):
        RegimeRule(**options)


def test_the_identity_declares_the_rule_and_its_minimums():
    regime, calendar = RegimeRule(REGIME_RULE).identity(), RegimeRule(CALENDAR_RULE).identity()
    assert regime["rule"] == REGIME_RULE and calendar["rule"] == CALENDAR_RULE
    assert regime["fit"] == calendar["fit"] == "none_declared_before_execution"
    assert (regime["min_assets"], regime["min_returns"], regime["recent_returns"]) == (20, 42, 21)
    assert "recent_returns" not in calendar and "assignment" not in regime
    assert regime["unclassified"] == calendar["unclassified"]
    assert RegimeRule(REGIME_RULE, min_assets=5).identity() != regime


# Claves enrutadas de B6.


def codec_inputs(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((rows, 64), generator=generator, dtype=torch.float32)


def correction(key, *, rate=0.25, forgetting=0.01):
    config = AssociativeMemoryConfig(
        "proximal", key_size=KEY_SIZES[key], rate=rate, forgetting=forgetting
    )
    return MatureCorrection(config, key=key)


def test_routed_keys_place_the_row_in_its_slot_with_unit_norm():
    inputs, routes = codec_inputs(6, 1), np.array([0, 1, 2, 3, 4, 2], dtype=np.int64)
    plain = correction("codec").keys(inputs)
    blocks = correction("codec_by_regime").keys(inputs, routes).reshape(6, SLOTS, 64)
    for row, route in enumerate(routes):
        assert torch.equal(blocks[row, route], plain[row])
        assert torch.count_nonzero(blocks[row]) == torch.count_nonzero(plain[row])
    norms = torch.linalg.vector_norm(blocks.reshape(6, -1), dim=1)
    torch.testing.assert_close(norms, torch.ones(6, dtype=torch.float64), rtol=0, atol=1e-15)
    hot = correction("regime").keys(inputs, routes)
    assert torch.equal(hot, torch.eye(SLOTS, dtype=torch.float64)[routes])
    assert torch.equal(correction("calendar").keys(inputs, routes), hot)


@pytest.mark.parametrize(
    ("key", "routes"),
    [
        ("codec", np.zeros(3, dtype=np.int64)),
        ("constant", np.zeros(3, dtype=np.int64)),
        ("regime", None),
        ("codec_by_calendar", None),
        ("regime", np.zeros(3, dtype=np.int32)),
        ("regime", np.array([0, 1, 5])),
        ("regime", np.array([0, -1, 2])),
        ("codec_by_regime", np.zeros(2, dtype=np.int64)),
        ("codec_by_regime", np.zeros((3, 1), dtype=np.int64)),
    ],
)
def test_routes_are_required_exactly_for_routed_keys_and_validated(key, routes):
    with pytest.raises(ValueError, match="ruta"):
        correction(key).keys(codec_inputs(3, 2), routes)


def test_each_routed_key_carries_its_own_rule():
    assert {key: correction(key).routing for key in CORRECTION_KEYS} == {
        key: None if key not in ROUTING else RegimeRule(ROUTING[key]) for key in CORRECTION_KEYS
    }
    config = AssociativeMemoryConfig("proximal", key_size=SLOTS)
    for key, routing in [
        ("regime", RegimeRule(CALENDAR_RULE)),
        ("calendar", RegimeRule(REGIME_RULE)),
        ("regime", REGIME_RULE),
    ]:
        with pytest.raises(ValueError, match="enrutada"):
            MatureCorrection(config, key=key, routing=routing)
    with pytest.raises(ValueError, match="enrutada"):
        MatureCorrection(AssociativeMemoryConfig("proximal"), routing=RegimeRule(REGIME_RULE))
    with pytest.raises(ValueError, match="320 coordenadas"):
        MatureCorrection(AssociativeMemoryConfig("proximal"), key="codec_by_regime")
    small = MatureCorrection(config, key="regime", routing=REGIME)
    assert small.identity()["routing"]["min_assets"] == ASSETS


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def test_unrouted_identities_keep_the_fingerprints_of_the_b6_arms():
    # Huellas calculadas con el código anterior al enrutamiento (a2bfafd4).
    codec = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.25, forgetting=0.01))
    constant = MatureCorrection(
        AssociativeMemoryConfig("proximal", key_size=1, rate=0.05, forgetting=0.01), key="constant"
    )
    assert fingerprint(codec.identity()).hexdigest() == (
        "754326957c3fceb05b6ed43533acef55840706e36927d3f570d5d830d327c8df"
    )
    assert fingerprint(constant.identity()).hexdigest() == (
        "279497c3a6662c8185d83749c503ede6638eff3db5373eee1546d35bd276c659"
    )
    routed = {key: correction(key).identity() for key in ROUTING}
    assert all(value["route_time"].startswith("decision_event") for value in routed.values())
    assert len({fingerprint(value).hexdigest() for value in routed.values()}) == len(ROUTING)


def write(memory, corrections, inputs, routes, values, start):
    rows = len(values)
    return memory.write(
        corrections.feedback(
            ids=list(range(start, start + rows)),
            decision_at=[10 * start] * rows,
            available_at=[10 * start + 5] * rows,
            keys=corrections.keys(inputs, routes),
            values=values,
        ),
        cutoff=10 * start + 5,
    )


def slot_update(previous, keys, values, rows, rate, retention):
    """Escritura proximal de un compartimento con los pesos 1/n de toda la cohorte."""
    system = torch.eye(keys.shape[1], dtype=torch.float64) + rate * keys.T @ keys / rows
    right = retention * previous + rate * keys.T @ values / rows
    return torch.linalg.solve(system, right)


def test_a_routed_write_is_one_proximal_update_per_slot():
    corrections, rate, retention = correction("codec_by_regime"), 0.25, 0.99
    plain = correction("codec")
    memory = AssociativeMemory(corrections.memory)
    slots = torch.zeros((SLOTS, 64, 1), dtype=torch.float64)
    cohorts = [
        (np.array([1, 1, 3, 0, 3, 1]), [0.5, -0.25, 1.0, 0.125, -0.75, 0.3]),
        (np.array([2, 2, 4]), [0.2, -0.4, 0.6]),
        (np.array([1, 2, 1, 4]), [-0.1, 0.9, 0.05, -0.3]),
    ]
    start = 1
    for seed, (routes, values) in enumerate(cohorts):
        inputs = codec_inputs(len(routes), 10 + seed)
        memory = write(memory, corrections, inputs, routes, values, start)
        start += len(routes)
        keys, targets = plain.keys(inputs), torch.tensor(values, dtype=torch.float64)[:, None]
        for slot in range(SLOTS):
            rows = torch.from_numpy(np.flatnonzero(routes == slot))
            # Un compartimento sin filas solo olvida, como una memoria con una cohorte vacía.
            slots[slot] = slot_update(
                slots[slot],
                keys.index_select(0, rows),
                targets.index_select(0, rows),
                len(routes),
                rate,
                retention,
            )
        torch.testing.assert_close(memory.matrix.reshape(SLOTS, 64, 1), slots, rtol=0, atol=1e-12)
    assert memory.writes == 13 and torch.count_nonzero(slots[0]) > 0


def test_a_routed_bias_write_has_its_closed_form():
    corrections, rate, retention = correction("regime", rate=2.0, forgetting=0.1), 2.0, 0.9
    memory = AssociativeMemory(corrections.memory, torch.ones((SLOTS, 1), dtype=torch.float64))
    routes, values = np.array([1, 1, 3, 3, 3]), [0.5, -0.25, 1.0, 0.125, -0.75]
    written = write(memory, corrections, codec_inputs(5, 4), routes, values, 1)
    expected = torch.full((SLOTS, 1), retention, dtype=torch.float64)
    for slot in (1, 3):
        chosen = [v for v, r in zip(values, routes, strict=True) if r == slot]
        share = len(chosen) / len(values)
        expected[slot] = (retention + rate * sum(chosen) / len(values)) / (1 + rate * share)
    torch.testing.assert_close(written.matrix, expected, rtol=0, atol=1e-14)


def test_the_comparison_declares_both_hypotheses_with_their_discard_controls():
    root = Path(__file__).resolve().parents[2]
    declaration = json.loads(
        (root / "configs/evaluation/regime-routing-comparison.json").read_text()
    )
    assert declaration["status"] == "declared_not_executed"
    assert declaration["executions"] == 0 and declaration["final_test_opened"] is False
    for name, rule in declaration["rules"].items():
        # La declaración repite la regla por defecto, la que reciben los brazos de la campaña.
        default = RegimeRule(rule["name"])
        assert rule == dict(
            name=default.name,
            min_assets=default.min_assets,
            recent_returns=default.recent_returns,
            min_returns=default.min_returns,
        )
        assert ROUTING[name] == rule["name"]
    campaign = json.loads(
        (root / "configs/baselines/historical-masked-campaign-extensions.json").read_text()
    )["sections"]["mars_titan"]["arms"]
    comparison = json.loads(
        (root / "configs/evaluation/historical-masked-2000-comparison.json").read_text()
    )["comparison"]["families"]
    for arm, entry in declaration["arms"].items():
        assert campaign[arm] == {"associative_memory": {"rule": "proximal", "key": entry["key"]}}
        assert entry["key_size"] == KEY_SIZES[entry["key"]]
    for hypothesis in declaration["hypotheses"].values():
        variant, control = (declaration["arms"][hypothesis[k]] for k in ("variant", "control"))
        assert (variant["role"], control["role"]) == ("innovation", "discard_control")
        assert variant["key_size"] == control["key_size"]
        assert ROUTING[variant["key"]] == REGIME_RULE and ROUTING[control["key"]] == CALENDAR_RULE
        family = comparison[hypothesis["family"]]
        assert family["base"] == hypothesis["control"]
        assert family["variants"] == [hypothesis["variant"]]
    extensions = json.loads((root / "configs/titans/mars-titan-extensions.json").read_text())
    ablations = {a["id"]: a for a in extensions["ablations"]}
    assert ablations["A13"]["change"] == {"associative_memory": {"key": ["regime", "calendar"]}}
    assert ablations["A14"]["change"] == {
        "associative_memory": {"key": ["codec_by_regime", "codec_by_calendar"]}
    }


def test_the_variant_builds_each_routed_arm_with_its_default_rule():
    from mars_titan.memory.mars_titan_variant import check_components, load_declaration

    declaration, seen = load_declaration(), set()
    for key in CORRECTION_KEYS:
        value = dict(rule="proximal", key=key, rate=0.25, forgetting=0.01)
        components, built = check_components(declaration, dict(associative_memory=value))
        assert components == dict(associative_memory=value)
        assert built.memory.key_size == KEY_SIZES[key]
        assert built.routing == (RegimeRule(ROUTING[key]) if key in ROUTING else None)
        seen.add(json.dumps(built.identity(), sort_keys=True))
    assert len(seen) == len(CORRECTION_KEYS)
    with pytest.raises(ValueError, match="clave de B6"):
        check_components(
            declaration,
            dict(associative_memory=dict(rule="proximal", key="hmm", rate=0.25, forgetting=0.01)),
        )
