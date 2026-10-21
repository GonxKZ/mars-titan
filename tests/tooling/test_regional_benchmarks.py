"""Rechazo del coste no acotado antes de tocar CUDA o crear resultados."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("script", ["benchmark.py", "session_benchmark.py"])
@pytest.mark.parametrize("fault", ["warmup", "repetitions", "duplicate_rows", "parent"])
def test_invalid_resource_requests_fail_before_selecting_a_device(
    tmp_path, monkeypatch, script, fault
):
    path = ROOT / "reports/engineering/regional-fusion-20261021" / script
    spec = importlib.util.spec_from_file_location("regional_cost_arguments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / ("missing/result.json" if fault == "parent" else "result.json")
    arguments = [str(path), "--output", str(output), "--rows", "64"]
    if script == "session_benchmark.py":
        arguments += ["--storage-root", str(tmp_path)]
    if fault == "warmup":
        arguments += ["--warmup", "17"]
    elif fault == "repetitions":
        arguments += ["--repetitions", "129"]
    elif fault == "duplicate_rows":
        arguments += ["--rows", "64", "64"]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def unexpected_device(*args):
        raise AssertionError("No debe seleccionarse CUDA para una petición inválida")

    monkeypatch.setattr(torch.cuda, "set_device", unexpected_device)
    with pytest.raises(ValueError, match="medida"):
        module.main()
    assert not output.exists()
