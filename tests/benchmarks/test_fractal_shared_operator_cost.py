"""La medida de coste de C3 recorre los cinco modelos declarados y no ajusta nada."""

import json
import sys

import pytest
import torch

from benchmarks import fractal_shared_operator_cost as cost


def test_the_protocol_measures_every_declared_arm():
    declaration = json.loads(cost.DECLARATION.read_text(encoding="utf-8"))
    changed = [name for name, arm in declaration["arms"].items() if arm["predictor_changes"]]
    assert cost.protocol("cpu")["modes"] == changed == list(cost.arms())
    assert cost.protocol("cpu")["orders"][1] == list(reversed(changed))


@pytest.mark.parametrize(("rows", "sessions"), [(64, 1), (1024, 4)])
def test_the_inputs_follow_the_masked_layout_and_the_declared_sessions(rows, sessions):
    blocks, presence, flows, instants, prices = cost.inputs(rows, "cpu")
    assert presence[:, [0, 2]].all()
    assert torch.equal(blocks * ~presence.unsqueeze(-1), torch.zeros_like(blocks))
    assert len(set(flows)) == rows and prices.shape == (rows, 64, 6)
    assert (prices[..., 5] == 1).all()
    keys = {(flow.split("/")[0], at) for flow, at in zip(flows, instants, strict=True)}
    assert len(keys) == sessions


def test_a_small_run_covers_every_arm_and_keeps_the_optimizer_closed(monkeypatch):
    monkeypatch.setattr(cost, "ROWS", (64,))
    monkeypatch.setattr(cost, "WARMUP", 1)
    monkeypatch.setattr(cost, "REPETITIONS", 1)
    result = cost.run("cpu")
    modes = list(cost.arms())
    expected = {(mode, 64, kind) for mode in modes for kind in cost.PASSES}
    assert {(r["mode"], r["rows"], r["pass_"]) for r in result["summary"]} == expected
    assert all(r["p95_ms"] >= r["p50_ms"] > 0 for r in result["summary"])
    # Cada modelo se mide en los dos órdenes.
    assert len(result["records"]) == 2 * len(expected)
    assert set(result["parameters"]) == set(modes) and result["memory"] == "not_measured_on_cpu"
    assert result["parameters"]["per_level"] >= result["parameters"]["fractal_shared_operator"]
    with pytest.raises(RuntimeError, match="optimizador"):
        cost.forbid_optimizer_steps(None, (), {})


def test_the_command_line_prints_the_protocol_and_writes_only_a_new_receipt(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(sys, "argv", ["fractal_shared_operator_cost.py"])
    cost.main()
    assert json.loads(capsys.readouterr().out) == cost.protocol("cpu")
    output = tmp_path / "nueva" / "cost.json"
    calls = []

    def run(device):
        # La carpeta del recibo existe antes de medir.
        assert output.parent.is_dir()
        calls.append(device)
        return dict(device=device)

    monkeypatch.setattr(cost, "run", run)
    monkeypatch.setattr(
        sys, "argv", ["fractal_shared_operator_cost.py", "--run", "--output", str(output)]
    )
    cost.main()
    assert calls == ["cpu"] and json.loads(output.read_text()) == dict(device="cpu")
    for arguments in (["--run"], ["--run", "--output", str(output)]):
        monkeypatch.setattr(sys, "argv", ["fractal_shared_operator_cost.py", *arguments])
        with pytest.raises(SystemExit, match="destino nuevo"):
            cost.main()
    assert calls == ["cpu"]


def test_only_the_first_cuda_device_is_measured(monkeypatch):
    # Con CUDA disponible solo se mide en cuda:0, y sin CUDA ni siquiera en cuda:0.
    for available, device in ((True, "cuda:1"), (False, "cuda:0")):
        monkeypatch.setattr(torch.cuda, "is_available", lambda value=available: value)
        with pytest.raises(RuntimeError, match="cuda:0 disponible"):
            cost.run(device)
