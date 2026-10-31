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
    # Las predicciones por área añaden su normalización, los votos, sus sesgos y la lectura.
    votes = 2 * cost.HIDDEN + 19 * cost.HIDDEN + 2 * 19
    for name in ("h1_19_areas_area_predictions", "h1_19_areas_area_readout_only"):
        assert result["parameters"][name] == result["parameters"]["h1_19_areas"] + votes
    with pytest.raises(RuntimeError, match="optimizador"):
        cost.forbid_optimizer_steps(None, (), {})


def test_the_voting_arms_propagate_through_the_same_nonzero_votes(monkeypatch):
    monkeypatch.setattr(cost, "ROWS", (8,))
    monkeypatch.setattr(cost, "WARMUP", 0)
    monkeypatch.setattr(cost, "REPETITIONS", 1)
    measure, seen = cost.measure, {}

    def recording(module, encoders, arguments, kind, synchronize):
        samples = measure(module, encoders, arguments, kind, synchronize)
        if kind == "backward" and module.config.area_predictions is not None:
            seen[module.config.area_predictions] = {
                name: (value.detach().clone(), value.grad)
                for name, value in module.named_parameters()
                if name.startswith("area_vote") or name == "readout.weight"
            }
        return samples

    monkeypatch.setattr(cost, "measure", recording)
    cost.run("cpu")
    declared = {raw.get("area_predictions") for raw in cost.arms().values()} - {None}
    assert set(seen) == declared == {0.0, 0.5}
    first, second = seen.values()
    for name in ("area_vote_weight", "area_vote_logits"):
        assert first[name][0].abs().sum() > 0
        torch.testing.assert_close(first[name][0], second[name][0], rtol=0, atol=0)
        # El desplazamiento entra en la pérdida, así que los votos y su lectura reciben gradiente.
        assert all(values[name][1].abs().sum() > 0 for values in seen.values())
    # La salida de la red también sigue en la pérdida de los modelos que votan.
    assert all(values["readout.weight"][1].abs().sum() > 0 for values in seen.values())
