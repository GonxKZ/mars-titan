"""La medida de coste de H4 recorre sus tres modos con bloques pequeños y no ajusta nada."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks import asset_groups_cost as cost
from mars_titan.data.price_windows import PRICE_WINDOW_CHANNELS
from mars_titan.models.titans.asset_groups import AssetGroupsConfig

DECLARATION = Path(__file__).parents[2] / "configs/experiments/asset-group-membership-v1.json"


def test_the_protocol_measures_the_declared_modes_with_the_variant_configuration():
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    declared = [
        arm["predictor_changes"]["asset_groups"]
        for arm in declaration["arms"].values()
        if arm["predictor_changes"]
    ]
    protocol = cost.protocol("cpu")
    assert sorted(changes["mode"] for changes in declared) == sorted(protocol["modes"])
    assert protocol["orders"][1] == list(reversed(protocol["modes"]))
    variant = declaration["arms"]["asset_group_membership"]["predictor_changes"]["asset_groups"]
    defaults = AssetGroupsConfig()
    assert protocol["configuration"] == {
        name: getattr(defaults, name) for name in protocol["configuration"]
    }
    assert protocol["configuration"] == {
        name: value for name, value in variant.items() if name != "mode"
    }


def test_the_windows_follow_the_presence_layout():
    prices = cost.windows(32, torch.Generator().manual_seed(1))
    assert prices.shape == (32, 64, len(PRICE_WINDOW_CHANNELS)) and prices.dtype == torch.float32
    opened, high, low, close, volume, present = prices.unbind(-1)
    assert torch.equal(present, torch.ones_like(present))
    # El ancla es el cierre de la primera sesión y la apertura repite el cierre anterior.
    assert torch.equal(close[:, 0], torch.zeros(32))
    assert torch.equal(opened[:, 1:], close[:, :-1])
    assert (high > close).all() and (close > low).all()
    assert torch.equal(volume, torch.zeros_like(volume))


def test_the_sector_map_covers_every_measured_flow_within_its_market():
    mapping = cost.sector_map()
    for shape in cost.SHAPES:
        flows, markets, instants = cost.session_keys(*shape)
        assert len(set(flows)) == shape[0]
        assert len(set(zip(markets, instants, strict=True))) == shape[1]
        for flow, market in zip(flows, markets, strict=True):
            group = mapping.groups[mapping.group(flow)]
            assert group.startswith(f"{market}/S")


def test_a_small_run_covers_every_mode_and_keeps_the_optimizer_closed(monkeypatch):
    monkeypatch.setattr(cost, "SHAPES", ((8, 1), (16, 2)))
    monkeypatch.setattr(cost, "WARMUP", 0)
    monkeypatch.setattr(cost, "REPETITIONS", 1)
    result = cost.run("cpu")
    expected = {
        (mode, rows, kind) for mode in cost.MODES for rows in (8, 16) for kind in cost.PASSES
    }
    assert {(r["mode"], r["rows"], r["pass_"]) for r in result["summary"]} == expected
    assert all(r["p95_ms"] >= r["p50_ms"] > 0 for r in result["summary"])
    # Cada modo se mide en los dos órdenes.
    assert len(result["records"]) == 2 * len(expected)
    # Los tres modos tienen los mismos parámetros.
    assert len(set(result["parameters"].values())) == 1
    assert {len(values) for values in result["load_average"].values()} == {3}
    with pytest.raises(RuntimeError, match="optimizador"):
        cost.forbid_optimizer_steps(None, (), {})


def test_the_optimizer_guard_covers_every_measure(monkeypatch):
    events = []

    class Handle:
        def remove(self):
            events.append("removed")

    def register(hook):
        events.append(hook)
        return Handle()

    # torch.optim no expone el submódulo como atributo, así que se toma de los importados.
    optimizer = importlib.import_module("torch.optim.optimizer")
    monkeypatch.setattr(optimizer, "register_optimizer_step_pre_hook", register)
    monkeypatch.setattr(cost, "SHAPES", ((8, 1),))
    monkeypatch.setattr(cost, "measure", lambda *arguments: events.append("measured") or [1.0])
    cost.run("cpu")
    assert events[0] is cost.forbid_optimizer_steps and events[-1] == "removed"
    assert set(events[1:-1]) == {"measured"}


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("memberships", [("memberships", ["US/A", "US/B"])]),
        ("forward", [("call", False)]),
        ("backward", [("zero_grad", True), ("call", True), ("backward", True)]),
    ],
)
def test_the_warmup_calls_run_before_the_measured_ones(monkeypatch, kind, expected):
    monkeypatch.setattr(cost, "WARMUP", 2)
    monkeypatch.setattr(cost, "REPETITIONS", 3)
    calls, synchronized = [], []

    class Module:
        # Con el sector actual la medida no reparte grupos y solo llama a `memberships`.
        config = SimpleNamespace(mode="current_sector")

        def sessions(self, markets, instants):
            return [(("US", 1), [0, 1])]

        def memberships(self, prices, flows, key):
            calls.append(("memberships", flows))

        def zero_grad(self, set_to_none):
            calls.append(("zero_grad", set_to_none))

        def __call__(self, *arguments):
            calls.append(("call", torch.is_grad_enabled()))
            value = torch.ones(2, requires_grad=True)
            value.register_hook(lambda grad: calls.append(("backward", True)))
            return value

    inputs = (None, torch.zeros(2, 4), ["US/A", "US/B"], ["US", "US"], [1, 1])
    samples = cost.measure(Module(), inputs, kind, lambda: synchronized.append(kind))
    # Dos llamadas de calentamiento y tres medidas, cada una seguida de su sincronización.
    assert len(samples) == 3
    assert calls == expected * 5 and len(synchronized) == 5
