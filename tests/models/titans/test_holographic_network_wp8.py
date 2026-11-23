"""WP8 (c): inicialización single_zero_v1 frente a zero_chain en la red holográfica."""

import copy
import csv
from dataclasses import asdict

import numpy as np
import pytest
import torch
from test_financial_adapter import api
from torch import nn

from mars_titan.data.accounting_catalog import JOINT_CONCEPTS
from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.models.titans import holographic_network as hn
from mars_titan.models.titans.holographic_network import (
    CATALOG_MEMBERSHIP_ONLY,
    HIDDEN_GROUPS_ONLY,
    HOLOGRAPHIC_NETWORK,
    MACRO_FAMILIES,
    HolographicNetwork,
    HolographicNetworkConfig,
    parse_holographic_network,
)

AT = 1_609_459_200_000_000
FUNDAMENTALS = list(JOINT_CONCEPTS)
SYNTHETIC_INDICATORS = [f"ind_{family}_{idx}" for family in MACRO_FAMILIES for idx in range(2)]
REPRESENTATION = dict(fundamental_concepts=FUNDAMENTALS, macro_indicators=SYNTHETIC_INDICATORS)
DIMENSIONS = dict(
    prices=5,
    news=2,
    charts=2,
    fundamentals=3 * len(FUNDAMENTALS),
    macro=3 * len(SYNTHETIC_INDICATORS),
)


class Name(str):
    """Subclase de str que no debe colarse como nombre de inicialización."""


@pytest.fixture
def synthetic_catalog(tmp_path, monkeypatch):
    """Catálogo macro sintético con las 12 familias para probar 19 y 38 áreas sin datos."""
    path = tmp_path / "macro-indicators.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "category"])
        writer.writeheader()
        for family in MACRO_FAMILIES:
            for idx in range(2):
                writer.writerow({"id": f"ind_{family}_{idx}", "category": family})
    hn.macro_categories.cache_clear()
    old_catalog = hn.CATALOG
    monkeypatch.setattr(hn, "CATALOG", path)
    # También actualizar el valor por defecto del parámetro catalog de area_layout/macro_categories.
    original_categories = hn.macro_categories
    monkeypatch.setattr(
        hn,
        "macro_categories",
        lambda catalog=path: original_categories(
            path if catalog in (old_catalog, hn.CATALOG) else catalog
        ),
    )
    yield path
    original_categories.cache_clear()


def masked_vector(generator, rows, concepts, present):
    observed = generator.random((rows, concepts)) < 0.6
    observed[present & ~observed.any(1), 0] = True
    observed[~present] = False
    data = np.where(observed, generator.normal(size=(rows, concepts)), 0)
    ages = np.where(observed, generator.uniform(0, 5, size=(rows, concepts)), 0)
    return np.concatenate([data, observed, ages], axis=1).astype(np.float32).round(3)


def module_inputs(rows=5, hidden=16, seed=9, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    presence = torch.ones(rows, 5, dtype=torch.bool)
    presence[1, 1] = presence[2, 4] = presence[3, 3] = False
    blocks = torch.randn(rows, 5, hidden, generator=generator, dtype=dtype)
    blocks = blocks * presence.unsqueeze(-1)
    raw = np.random.default_rng(seed)
    inputs = {
        "fundamentals": torch.tensor(
            masked_vector(raw, rows, len(FUNDAMENTALS), presence[:, 3].numpy()), dtype=dtype
        ),
        "macro": torch.tensor(
            masked_vector(raw, rows, len(SYNTHETIC_INDICATORS), presence[:, 4].numpy()), dtype=dtype
        ),
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoders = nn.ModuleDict(
            {
                key: nn.Sequential(nn.Linear(DIMENSIONS[key], hidden), nn.SiLU()).to(dtype=dtype)
                for key in ("fundamentals", "macro")
            }
        )
    return blocks, presence, inputs, encoders


def predictor_setup(dtype=torch.float64, *, zero_head=False, **changes):
    representation = dict(
        **policy_identity(HISTORICAL_MASKED),
        **REPRESENTATION,
        encoders={"fixture": "frozen-v1"},
        representation_code={"fixture.py": "c" * 64},
        text_aggregation="mean",
        context_sessions=64,
        news_lookback_sessions=5,
    )
    spec = api().FinancialInputSpec(
        source_sha256="a" * 64,
        view_sha256="b" * 64,
        representation=representation,
        dimensions=DIMENSIONS,
        input_policy=HISTORICAL_MASKED,
    )
    config = api().FinancialConfig(
        spec, variant="mac_online", hidden_size=32, layers=1, seed=42, **changes
    )
    model = api().FinancialPredictor(config, dtype=dtype)
    if zero_head:
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        model._seal_parameters()
    gen = np.random.default_rng(11)
    rows = 4
    presence = np.ones((rows, 5), dtype=np.bool_)
    presence[1, [1, 3]] = False
    values = {
        key: gen.normal(size=(rows, 64, width) if key == "prices" else (rows, width))
        .astype(np.float32)
        .round(3)
        for key, width in DIMENSIONS.items()
        if key in ("prices", "news", "charts")
    }
    values["news"][~presence[:, 1]] = 0
    values["fundamentals"] = masked_vector(gen, rows, len(FUNDAMENTALS), presence[:, 3])
    values["macro"] = masked_vector(gen, rows, len(SYNTHETIC_INDICATORS), presence[:, 4])
    raw = dict(
        inputs=values,
        presence=presence,
        sample_ids=[f"US/A{index:03}/{AT}" for index in range(rows)],
        market=["US"] * rows,
        prediction_at=np.full(rows, AT, dtype="datetime64[us]"),
        input_available_at=np.full(rows, AT - 1, dtype="datetime64[us]"),
    )
    batch = api().DecisionBatch.from_corpus(raw, spec, dtype=dtype)
    return model, batch


def test_default_initialization_is_zero_chain_and_preserves_legacy_identity_and_parser():
    default = HolographicNetworkConfig(areas=5)
    assert default.initialization == "zero_chain"
    identity = default.identity()
    assert "initialization" not in identity
    assert identity["area_weights"] == "shared_block_projection_plus_low_rank_adapter_up_from_zero"
    assert identity["membership"].endswith("_plus_learned_delta_from_zero")

    # Retrocompatibilidad de parse_holographic_network con 12 campos sin initialization.
    legacy_dict = {k: v for k, v in asdict(default).items() if k != "initialization"}
    assert len(legacy_dict) == 12
    parsed = parse_holographic_network(legacy_dict)
    assert parsed == default

    # Con initialization declarada también se analiza y cambia la identidad.
    single = parse_holographic_network(dict(legacy_dict, initialization="single_zero_v1"))
    assert single.initialization == "single_zero_v1"
    single_id = single.identity()
    assert single_id["initialization"] == "single_zero_v1"
    assert single_id != identity

    # Valores inválidos rechazados.
    for bad in ("random", True, None, 1, Name("single_zero_v1")):
        with pytest.raises(ValueError, match="inicialización"):
            HolographicNetworkConfig(areas=5, initialization=bad)
    with pytest.raises(ValueError, match="toda su configuración"):
        parse_holographic_network(dict(legacy_dict, unknown_field=1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", [HOLOGRAPHIC_NETWORK, CATALOG_MEMBERSHIP_ONLY])
@pytest.mark.parametrize("areas", [5, 19, 38])
def test_single_zero_v1_starts_bitwise_identical_to_parent_while_breaking_zero_chain(
    synthetic_catalog, dtype, kind, areas
):
    cfg_zero = HolographicNetworkConfig(kind=kind, areas=areas, initialization="zero_chain")
    cfg_single = HolographicNetworkConfig(kind=kind, areas=areas, initialization="single_zero_v1")

    net_zero = HolographicNetwork(16, cfg_zero, REPRESENTATION, seed=3).to(dtype=dtype)
    net_single = HolographicNetwork(16, cfg_single, REPRESENTATION, seed=3).to(dtype=dtype)

    # Solo readout.weight arranca a cero; area_up y membership_delta son pequeños no nulos.
    assert torch.count_nonzero(net_single.readout.weight) == 0
    assert torch.count_nonzero(net_zero.area_up) == 0
    assert torch.count_nonzero(net_single.area_up) > 0
    assert net_single.area_up.abs().max().item() < 0.1
    if kind == HOLOGRAPHIC_NETWORK:
        assert torch.count_nonzero(net_zero.membership_delta) == 0
        assert torch.count_nonzero(net_single.membership_delta) > 0
        assert net_single.membership_delta.abs().max().item() < 0.1

    # Todos los pesos no nulos compartidos coinciden bit a bit con zero_chain para la misma semilla.
    zero_params = dict(net_zero.named_parameters())
    for name, value in net_single.named_parameters():
        if name not in {"area_up", "membership_delta"}:
            assert torch.equal(value, zero_params[name]), name

    # La salida inicial coincide bit a bit con zero_chain y con los bloques de entrada (padre).
    arguments = module_inputs(dtype=dtype)
    out_zero = net_zero(*arguments)
    out_single = net_single(*arguments)
    assert torch.equal(out_single, arguments[0])
    assert torch.equal(out_single, out_zero)

    # En FinancialPredictor completo, la salida inicial coincide bit a bit con el padre sin red.
    base, batch = predictor_setup(dtype=dtype)
    pred_single, _ = predictor_setup(dtype=dtype, holographic_network=cfg_single)
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = pred_single.prepare(batch, pred_single.initial_state(batch.flow_ids))
    assert torch.equal(actual.point_predictions, expected.point_predictions)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("areas", [5, 19, 38])
def test_zero_chain_blocks_area_down_after_readout_step_whereas_single_zero_v1_delivers_gradient(
    synthetic_catalog, dtype, areas
):
    cfg_zero = HolographicNetworkConfig(areas=areas, initialization="zero_chain")
    cfg_single = HolographicNetworkConfig(areas=areas, initialization="single_zero_v1")
    net_zero = HolographicNetwork(16, cfg_zero, REPRESENTATION, seed=5).to(dtype=dtype)
    net_single = HolographicNetwork(16, cfg_single, REPRESENTATION, seed=5).to(dtype=dtype)
    arguments = module_inputs(dtype=dtype)

    # Paso 0 (con readout.weight == 0): por regla de la cadena solo readout.weight recibe gradiente.
    for net in (net_zero, net_single):
        net.zero_grad()
        net(*arguments).square().sum().backward()
        assert torch.count_nonzero(net.readout.weight.grad) > 0
        assert torch.isfinite(net.readout.weight.grad).all()
        assert torch.count_nonzero(net.area_down.grad) == 0

    # El gradiente de readout.weight en el paso 0 refleja la adaptación no nula de single_zero_v1.
    assert not torch.equal(net_single.readout.weight.grad, net_zero.readout.weight.grad)

    # Tras el primer paso que mueve readout.weight (sin tocar nada más):
    # - En zero_chain, area_up sigue a cero y area_down sigue recibiendo gradiente CERO.
    # - En single_zero_v1, TODOS los parámetros (incluido area_down) reciben gradiente no nulo.
    lr = 0.01
    for net in (net_zero, net_single):
        with torch.no_grad():
            for param in net.parameters():
                if param.grad is not None:
                    param.sub_(lr * param.grad)
        net.zero_grad()
        net(*arguments).square().sum().backward()

    assert torch.count_nonzero(net_zero.area_down.grad) == 0
    for name, param in net_single.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert torch.count_nonzero(param.grad) > 0, name


def test_single_zero_v1_gradients_are_finite_when_predictor_head_starts_at_zero():
    # Regla 4: gradientes finitos también cuando la última capa del predictor arranca en cero.
    cfg = HolographicNetworkConfig(areas=5, initialization="single_zero_v1")
    model, batch = predictor_setup(dtype=torch.float64, zero_head=True, holographic_network=cfg)
    prepared = model.prepare(batch, model.initial_state(batch.flow_ids), differentiable=True)
    assert torch.equal(prepared.point_predictions, torch.zeros_like(prepared.point_predictions))
    loss = (
        (
            prepared.point_predictions
            - torch.linspace(-0.01, 0.02, len(batch.flow_ids), dtype=torch.float64)
        )
        .square()
        .sum()
    )
    loss.backward()
    for name, param in model.holographic_network.named_parameters():
        assert param.grad is None or torch.isfinite(param.grad).all(), name


def test_single_zero_v1_state_dict_roundtrip_and_global_rng_isolation():
    before = torch.get_rng_state().clone()
    cfg_single = HolographicNetworkConfig(areas=5, initialization="single_zero_v1")
    cfg_zero = HolographicNetworkConfig(areas=5, initialization="zero_chain")
    source = HolographicNetwork(16, cfg_single, seed=7).double()
    assert torch.equal(torch.get_rng_state(), before)

    # Determinismo entre instancias con la misma semilla.
    replica = HolographicNetwork(16, cfg_single, seed=7).double()
    for (n1, p1), (n2, p2) in zip(
        source.named_parameters(), replica.named_parameters(), strict=True
    ):
        assert n1 == n2 and torch.equal(p1, p2)

    # Roundtrip de state_dict y rechazo al cargar sobre zero_chain.
    with torch.no_grad():
        source.readout.weight.fill_(0.05)
    payload = copy.deepcopy(source.state_dict())
    replica.load_state_dict(payload)
    blocks = torch.randn(3, 5, 16, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    presence = torch.ones(3, 5, dtype=torch.bool)
    assert torch.equal(replica(blocks, presence), source(blocks, presence))

    other = HolographicNetwork(16, cfg_zero, seed=7).double()
    with pytest.raises((ValueError, RuntimeError)):
        other.load_state_dict(payload)

    # H3 (sin áreas) también admite la opción de forma limpia y conserva la salida inicial.
    h3 = HolographicNetworkConfig(
        kind=HIDDEN_GROUPS_ONLY,
        areas=5,
        hidden_groups=4,
        rank=None,
        leakage=None,
        topology="complete",
        initialization="single_zero_v1",
    )
    net_h3 = HolographicNetwork(16, h3, seed=7).double()
    assert torch.equal(net_h3(blocks, presence), blocks)
    assert h3.identity()["initialization"] == "single_zero_v1"
