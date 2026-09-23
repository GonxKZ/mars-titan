"""La sonda reproduce el paso supervisado sin convertirlo en una época completa."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.data.embeddings import require_cuda
from mars_titan.models.baselines.multimodal import MultimodalReference


def module():
    path = Path("reports/analysis/benchmark_full_corpus.py")
    spec = importlib.util.spec_from_file_location("corpus_benchmark", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_benchmark_import_does_not_start_a_workload():
    assert callable(module().step)


def test_step_counts_rows_and_uses_all_modalities_on_cuda():
    engine = module()
    device = require_cuda()
    dimensions = dict(prices=5, news=3, charts=4, fundamentals=2, macro=5)
    model = MultimodalReference("gru", dimensions, context=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = {
        name: np.ones((4, 8, size) if name == "prices" else (4, size), dtype=np.float32)
        for name, size in dimensions.items()
    }
    stats = engine._statistics()
    before = model.head.weight.detach().clone()
    engine.step(model, optimizer, {"inputs": inputs, "target": np.zeros(4)}, device, stats)
    assert stats["samples"] == 4
    assert not torch.equal(before, model.head.weight)
    assert all(model.encoders[name][0].weight.grad is not None for name in model.encoders)
    with pytest.raises(ValueError, match="finita"):
        engine.step(
            model, optimizer, {"inputs": inputs, "target": np.full(4, np.nan)}, device, stats
        )


def test_incomplete_source_cannot_produce_a_full_corpus_profile(tmp_path, monkeypatch):
    from types import SimpleNamespace

    engine = module()
    monkeypatch.setattr(
        engine,
        "CorpusDataset",
        lambda *_: SimpleNamespace(
            manifest={"scope": "development_snapshot", "cohort_complete": False}
        ),
    )
    output = tmp_path / "profile"
    with pytest.raises(ValueError, match="completo"):
        engine.benchmark(tmp_path / "manifest.json", output)
    assert not output.exists()


def test_completed_profile_also_closes_its_progress_receipt(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    engine = module()
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "manifest.json"
    manifest.write_text("{}")

    def batches(**kwargs):
        assert kwargs["partition"] == "train"
        for i in range(25):
            yield dict(
                inputs={"prices": np.zeros((1, 8, 5), dtype=np.float32)},
                target=np.zeros(1),
                sample_ids=[f"US/A/{i}"],
            )

    dataset = SimpleNamespace(
        manifest={"scope": "full_corpus", "cohort_complete": True, "counts": {"train": 25}},
        roots={"prepared": source},
        assets=["US/A"],
        context=8,
        batches=batches,
    )
    model = SimpleNamespace(parameters=lambda: [])
    model.to = lambda *_: model
    monkeypatch.setattr(engine, "CorpusDataset", lambda *_: dataset)
    monkeypatch.setattr(engine, "MultimodalReference", lambda *_, **__: model)
    monkeypatch.setattr(engine, "require_cuda", lambda: "cuda:0")
    monkeypatch.setattr(engine, "seed_run", lambda *_: None)
    monkeypatch.setattr(engine, "scientific_identity", lambda: {})
    monkeypatch.setattr(engine, "step", lambda *_: None)
    monkeypatch.setattr(engine.torch.optim, "AdamW", lambda *_, **__: None)
    monkeypatch.setattr(engine.subprocess, "check_output", lambda *_, **__: "doble de prueba")
    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(engine.torch.cuda, name, lambda *_: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(engine.torch.cuda, name, lambda *_: 0)
    output = tmp_path / "profile"
    engine.benchmark(manifest, output)
    progress = json.loads((output / "progress.json").read_text())
    final = json.loads((output / "profile.json").read_text())
    assert progress == final and progress["status"] == "completed"
    assert len(progress["cases"]) == 24
