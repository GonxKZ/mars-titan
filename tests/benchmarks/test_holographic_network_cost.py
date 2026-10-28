"""La medida de coste de H1 recorre cada modelo declarado con bloques pequeños y no ajusta nada."""

import json

import pytest
import torch

from benchmarks import holographic_network_cost as cost


def test_the_protocol_measures_every_declared_arm():
    declaration = json.loads(cost.DECLARATION.read_text(encoding="utf-8"))
    changed = [name for name, arm in declaration["arms"].items() if arm["predictor_changes"]]
    assert cost.protocol("cpu")["modes"] == changed == list(cost.arms())
    assert cost.protocol("cpu")["orders"][1] == list(reversed(changed))


def test_the_inputs_follow_the_masked_layout():
    blocks, presence, family = cost.inputs(32, "cpu")
    assert presence[:, [0, 2]].all()
    assert torch.equal(blocks * ~presence.unsqueeze(-1), torch.zeros_like(blocks))
    for index, name, concepts in ((3, "fundamentals", 26), (4, "macro", 140)):
        values, observed, ages = family[name].split(concepts, dim=1)
        assert torch.equal(observed.bool().any(1), presence[:, index])
        assert torch.equal(values * (1 - observed), torch.zeros_like(values))
        assert torch.equal(ages * (1 - observed), torch.zeros_like(ages)) and (ages >= 0).all()


def test_a_small_run_covers_every_arm_and_keeps_the_optimizer_closed(monkeypatch):
    monkeypatch.setattr(cost, "ROWS", (8,))
    monkeypatch.setattr(cost, "WARMUP", 0)
    monkeypatch.setattr(cost, "REPETITIONS", 1)
    result = cost.run("cpu")
    modes = list(cost.arms())
    expected = {(mode, 8, kind) for mode in modes for kind in cost.PASSES}
    assert {(r["mode"], r["rows"], r["pass_"]) for r in result["summary"]} == expected
    assert all(r["p95_ms"] >= r["p50_ms"] > 0 for r in result["summary"])
    # Cada modelo se mide en los dos órdenes.
    assert len(result["records"]) == 2 * len(expected)
    assert set(result["parameters"]) == set(modes) and result["memory"] == "not_measured_on_cpu"
    assert result["parameters"]["h1_19_areas"] > result["parameters"]["h2_19_areas"]
    with pytest.raises(RuntimeError, match="optimizador"):
        cost.forbid_optimizer_steps(None, (), {})
