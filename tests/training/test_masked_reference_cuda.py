"""Rutas CUDA de las referencias con máscaras, sin aplicar pasos de optimizador.

Se omiten sin GPU. El optimizador inyectado comprueba que los pesos no cambian.
"""

import importlib
import math
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.models.baselines.multimodal import PRESENCE_FUSION
from tests.models.test_masked_reference_fusion import KINDS, build, presence, values, zero_absent
from tests.training.test_masked_reference_run import (
    masked_case,
    masked_view,
    recording_optimizer,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requiere cuda:0")


@pytest.mark.parametrize("kind", KINDS)
def test_cuda_presence_fusion_matches_cpu_and_isolates_absent_blocks(kind, monkeypatch):
    # TF32 cambiaría la referencia numérica de float32 frente a CPU.
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    cpu = build(kind).float()
    device = torch.device("cuda:0")
    gpu = build(kind).float().to(device)
    mask = presence()
    inputs = zero_absent(values(), mask)
    expected = cpu({k: v.float() for k, v in inputs.items()}, mask)
    leaves = {k: v.float().to(device).requires_grad_() for k, v in inputs.items()}
    actual = gpu(leaves, mask.to(device))
    torch.testing.assert_close(actual.cpu(), expected.detach(), rtol=1e-4, atol=1e-5)
    actual.square().sum().backward()
    for index, name in enumerate(("news", "charts", "fundamentals", "macro"), 1):
        assert torch.count_nonzero(leaves[name].grad[~mask[:, index].to(device)]) == 0


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_cuda_masked_run_records_updates_without_changing_weights(tmp_path, monkeypatch, kind):
    record = SimpleNamespace(calls=[], optimizers=0)
    RecordingOptimizer = recording_optimizer(record)
    monkeypatch.setattr(torch.optim, "AdamW", RecordingOptimizer)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    engine = importlib.import_module("mars_titan.training.reference_run")
    manifest = masked_view(tmp_path)
    report = engine.run_reference_case(
        manifest,
        tmp_path / "run",
        masked_case(kind),
        batch_size=2,
        input_policy=HISTORICAL_MASKED,
        prediction_retention=engine.HELDOUT_FULL_TRAIN_SESSIONS,
    )
    assert report["status"] == "completed" and report["device"] == "cuda:0"
    assert report["identity"]["mask_fusion"] == PRESENCE_FUSION
    assert (
        len(record.calls) == report["global_step"] == 2 * math.ceil(report["samples"]["train"] / 2)
    )
    for partition in ("validation", "calibration", "evaluation"):
        table = pq.read_table(tmp_path / "run" / f"{partition}-predictions.parquet")
        assert len(table) == report["samples"][partition]
    assert report["attempts"][-1]["peak_vram_allocated_bytes"] > 0
