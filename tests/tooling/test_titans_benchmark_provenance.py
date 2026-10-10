"""Los recibos de los medidores de Titans registran el dispositivo y el entorno utilizados."""

import importlib
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_receipt_records_the_requested_device_and_the_real_environment(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    benchmark = importlib.import_module("titans_gate_retention")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    options = dict(dim=64, flows=4, length=4096, window=8, threads=2, device="cuda:0")
    result = benchmark.provenance(
        "titans_mac_output_scale.py", Namespace(output=tmp_path / "recibo.json", **options)
    )
    assert result == {
        "command": "uv run --no-sync python benchmarks/titans_mac_output_scale.py --dim 64 "
        "--flows 4 --length 4096 --window 8 --threads 2 --device cuda:0 --output <recibo>",
        "environment": "CUDA_VISIBLE_DEVICES=0, OMP_NUM_THREADS=1, MKL_NUM_THREADS=<sin definir>",
    }
