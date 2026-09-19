"""Comprobación opcional sobre los codificadores reales, sin datos financieros."""

import hashlib
import json
import os
import resource
import time
from datetime import UTC, datetime

import numpy as np
import pytest


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.skipif(
    os.environ.get("MARS_ENCODER_INTEGRATION") != "1",
    reason="La extracción real requiere activación explícita y CUDA, no usa alternativa CPU",
)
def test_real_encoders_preserve_state_and_repeat_representations():
    import torch

    from mars_titan.data.charts import chart_png
    from mars_titan.data.embeddings import FrozenEncoders

    started = time.perf_counter()
    encoders = FrozenEncoders()
    initialization = time.perf_counter() - started
    assert encoders.device == torch.device("cuda:0")
    assert len(encoders.spec["text_artifacts_sha256"]) == 4
    assert len(encoders.spec["tokenizer_backend_sha256"]) == 64
    assert encoders.spec["tokenizers_version"]
    assert encoders.spec["tokenizer_class"] == type(encoders.tokenizer).__name__
    models = [encoders.text_model, encoders.image_model]
    assert all(not model.training for model in models)
    assert all(not parameter.requires_grad for model in models for parameter in model.parameters())
    before = [model_digest(model) for model in models]
    texts = ["La empresa presenta sus resultados.", "A company publishes its results. " * 150]
    levels = np.linspace(10.0, 12.0, 64)
    ohlc = np.stack([levels, levels + 1, levels - 1, levels + 0.5], axis=1)
    png = chart_png(ohlc, end_index=63)
    text_first = np.stack([encoders.text(text) for text in texts])
    image_first = encoders.images([png])
    torch.cuda.synchronize()
    repeat_start = time.perf_counter()
    text_again = np.stack([encoders.text(text) for text in texts])
    image_again = encoders.images([png])
    torch.cuda.synchronize()
    repeat_seconds = time.perf_counter() - repeat_start
    assert text_first.shape == (2, 384)
    assert image_first.shape == (1, 512)
    np.testing.assert_allclose(text_first, text_again, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(image_first, image_again, rtol=1e-6, atol=1e-6)
    after = [model_digest(model) for model in models]
    assert before == after
    print(
        "ENCODER_AUDIT="
        + json.dumps(
            {
                "scope": "synthetic_encoder_parity_only",
                "checked_at_utc": datetime.now(UTC).isoformat(),
                "encoders": encoders.spec,
                "state_sha256_before": before,
                "state_sha256_after": after,
                "text_max_abs_difference": float(np.max(np.abs(text_first - text_again))),
                "image_max_abs_difference": float(np.max(np.abs(image_first - image_again))),
                "initialization_seconds": initialization,
                "repeat_seconds": repeat_seconds,
                "total_seconds": time.perf_counter() - started,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "rtol": 1e-6,
                "atol": 1e-6,
            },
            sort_keys=True,
        )
    )
