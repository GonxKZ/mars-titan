"""La medida de coste de P1 recorre la red y la regla con formas pequeñas y no ajusta nada."""

import json

import pytest
import torch
from torch.optim import optimizer

from benchmarks import three_factor_bridge_cost as cost
from mars_titan.models.titans.holographic_network import HolographicNetwork


def test_the_protocol_names_the_declared_network_and_every_bridge():
    declaration = json.loads(cost.DECLARATION.read_text(encoding="utf-8"))
    declared = [
        n for n, a in declaration["arms"].items() if "three_factor_bridge" in a["predictor_changes"]
    ]
    protocol = cost.protocol("cpu")
    assert protocol["rule"]["configurations"] == declared == list(cost.bridges())
    assert len(declared) == 4 and cost.REFERENCE in declaration["arms"]
    assert protocol["orders"][1] == list(reversed(cost.MODES))


def test_a_small_run_covers_both_modes_and_every_rule_and_keeps_the_optimizer_closed(
    monkeypatch,
):
    monkeypatch.setattr(cost, "ROWS", (8,))
    monkeypatch.setattr(cost, "EVENT_ROWS", (4,))
    monkeypatch.setattr(cost, "WARMUP", 1)
    monkeypatch.setattr(cost, "REPETITIONS", 1)
    hooks, register = [], optimizer.register_optimizer_step_pre_hook

    def registered(hook):
        hooks.append(hook)
        return register(hook)

    monkeypatch.setattr(optimizer, "register_optimizer_step_pre_hook", registered)
    calls, forward, backward = [], HolographicNetwork.forward, torch.Tensor.backward

    def called(self, *arguments, edge_bias=None, traces=None):
        calls.append((edge_bias, torch.is_grad_enabled()))
        return forward(self, *arguments, edge_bias=edge_bias, traces=traces)

    def propagated(self, *arguments, **options):
        calls.append(("backward",))
        return backward(self, *arguments, **options)

    monkeypatch.setattr(HolographicNetwork, "forward", called)
    monkeypatch.setattr(torch.Tensor, "backward", propagated)
    result = cost.run("cpu")
    # Dos órdenes por dos modos y dos pasadas, con una llamada de calentamiento y una medida.
    measured = [call for call in calls if len(call) == 2]
    assert len(measured) == 16 and calls.count(("backward",)) == 8
    # La referencia llega a la red sin pesos y P1 con los mismos pesos [filas, aristas] fijos
    # y dentro de la cota. `forward` va sin gradiente y `backward` lo propaga.
    assert sum(bias is None for bias, _ in measured) == 8
    weights = [bias for bias, _ in measured if bias is not None]
    assert all(bias is weights[0] for bias in weights) and weights[0].shape == (8, 38)
    assert 0 < weights[0].abs().max() <= cost.WEIGHT_SCALE
    assert sum(grad for _, grad in measured) == 8
    # La medida instala el gancho que rechaza cualquier paso de optimizador.
    assert hooks == [cost.forbid_optimizer_steps]
    expected = {(mode, 8, kind) for mode in cost.MODES for kind in cost.PASSES}
    assert {(r["mode"], r["rows"], r["pass_"]) for r in result["summary"]} == expected
    assert all(r["p95_ms"] >= r["p50_ms"] > 0 for r in result["summary"])
    # Cada modo se mide en los dos órdenes.
    assert len(result["records"]) == 2 * len(expected)
    # La referencia H1 de la declaración tiene 19 áreas en anillo, con dos aristas por área.
    assert result["edges"] == 38 and result["state_bytes"]["market_weights"] == 8 * 38
    assert result["memory"] == "not_measured_on_cpu"
    assert set(result["rule"]) == {f"{name}/4" for name in cost.bridges()}
    for value in result["rule"].values():
        assert len(value["event_microseconds"]["samples"]) == 1
        assert value["read_microseconds"]["p50"] > 0
    with pytest.raises(RuntimeError, match="optimizador"):
        cost.forbid_optimizer_steps(None, (), {})


def test_the_command_line_prints_the_protocol_and_writes_only_a_new_receipt(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr("sys.argv", ["three_factor_bridge_cost.py"])
    cost.main()
    assert json.loads(capsys.readouterr().out) == cost.protocol("cpu")
    output, calls = tmp_path / "nueva" / "cost.json", []

    def run(device):
        calls.append(device)
        return dict(device=device)

    monkeypatch.setattr(cost, "run", run)
    monkeypatch.setattr(
        "sys.argv", ["three_factor_bridge_cost.py", "--run", "--output", str(output)]
    )
    cost.main()
    assert calls == ["cpu"] and json.loads(output.read_text()) == dict(device="cpu")
    for arguments in (["--run"], ["--run", "--output", str(output)]):
        monkeypatch.setattr("sys.argv", ["three_factor_bridge_cost.py", *arguments])
        with pytest.raises(SystemExit, match="destino nuevo"):
            cost.main()
    assert calls == ["cpu"]


def test_only_the_first_cuda_device_is_measured(monkeypatch):
    for available, device in ((True, "cuda:1"), (False, "cuda:0")):
        monkeypatch.setattr(torch.cuda, "is_available", lambda value=available: value)
        with pytest.raises(RuntimeError, match="cuda:0 disponible"):
            cost.run(device)
