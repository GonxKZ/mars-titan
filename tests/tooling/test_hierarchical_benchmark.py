"""Rechazar destinos inseguros antes de cargar el recorrido CUDA."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def benchmark():
    path = ROOT / "benchmarks/hierarchical_bridge_cuda.py"
    spec = importlib.util.spec_from_file_location("hierarchical_cost", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_cost_receipt_is_never_overwritten(tmp_path):
    output = tmp_path / "receipt.json"
    output.write_text("original")
    with pytest.raises(ValueError, match="nueva"):
        benchmark().run(ROOT, output)
    assert output.read_text() == "original"


def test_run_requires_an_explicit_private_output(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hierarchical_bridge_cuda.py", "--run"])
    with pytest.raises(ValueError, match="salida"):
        benchmark().main()


def test_repository_copy_cannot_execute_the_cost_workload(tmp_path):
    output = tmp_path / "new.json"
    with pytest.raises(ValueError, match="copia privada"):
        benchmark().run(ROOT, output)
    assert not output.exists()
