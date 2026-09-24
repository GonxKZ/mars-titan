"""Medición del recorrido completo y del contenido que entrega el lector."""

import importlib.util
import json
from pathlib import Path

import pytest


def benchmark():
    path = Path(__file__).resolve().parents[2] / "benchmarks/corpus_batches.py"
    if not path.is_file():
        pytest.fail("Falta el benchmark reproducible del lector por lotes")
    specification = importlib.util.spec_from_file_location("corpus_batches_benchmark", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_technical_fixture_has_real_shapes_and_explicit_exclusions(tmp_path):
    engine = benchmark()
    manifest = engine.prepare_corpus(tmp_path / "corpus", assets=2, rows=9, group_rows=4, stride=3)
    from mars_titan.training.corpus_inputs import CorpusDataset

    batches = list(
        CorpusDataset(manifest).batches(partition="train", batch_size=4, epoch=0, seed=42)
    )
    assert sum(len(batch["target"]) for batch in batches) == 6
    assert json.loads(manifest.read_text())["domain"] == "technical"
    assert {name: array.shape[1:] for name, array in batches[0]["inputs"].items()} == {
        "prices": (64, 5),
        "news": (384,),
        "charts": (512,),
        "fundamentals": (45,),
        "macro": (420,),
    }
    with pytest.raises(FileExistsError):
        engine.prepare_corpus(tmp_path / "corpus", assets=2, rows=9, group_rows=4, stride=3)


def test_measurement_traverses_all_rows_and_checks_the_full_output(tmp_path):
    engine = benchmark()
    manifest = engine.prepare_corpus(tmp_path / "corpus", assets=2, rows=9, group_rows=4, stride=3)
    first = engine.measure(manifest, batch_size=4, revision="current")
    second = engine.measure(manifest, batch_size=4, revision="current")
    assert first["rows"] == second["rows"] == 6
    assert first["batches"] == second["batches"] == 2
    assert first["input_bytes"] == 6 * 1681 * 4
    assert first["stream_sha256"] == second["stream_sha256"]
    assert first["reader_seconds"] > 0
    assert first["initialization_seconds"] > 0
    assert first["peak_rss_bytes"] > 0


def test_invalid_workload_is_rejected_before_creating_files(tmp_path):
    engine = benchmark()
    with pytest.raises(ValueError):
        engine.prepare_corpus(tmp_path / "corpus", assets=2, rows=9, group_rows=0, stride=3)
    assert not (tmp_path / "corpus").exists()
