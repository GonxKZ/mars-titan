"""Ajuste externo recuperable sobre la misma población y etiquetas del corpus."""

import importlib
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mars_titan.training.checkpoints import StopRequest
from tests.training.test_reference_run import training_corpus

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("cupy", "xgboost")),
    reason="Requiere el extra boosting y CUDA",
)


def module():
    return importlib.import_module("mars_titan.training.external_corpus")


def test_racing_creation_does_not_replace_another_run(tmp_path, monkeypatch):
    manifest = training_corpus(tmp_path / "data")
    output = tmp_path / "race"
    original = Path.mkdir
    intercepted = False

    def competing_mkdir(path, *args, **kwargs):
        nonlocal intercepted
        if path == output and not intercepted:
            intercepted = True
            original(output)
            (output / "run.json").write_text('{"origin":"existing_run"}')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", competing_mkdir)
    with pytest.raises((ValueError, FileExistsError)):
        module().run_external_reference(manifest, output, rounds=1)
    assert (output / "run.json").read_text() == '{"origin":"existing_run"}'


def test_full_tabular_population_has_persistent_model_and_session_errors(tmp_path):
    manifest = training_corpus(tmp_path / "data")
    output = tmp_path / "run"
    report = module().run_external_reference(manifest, output, rounds=3, batch_size=5)
    assert report["status"] == "completed"
    assert report["model"] == "xgboost_external_cuda" and report["device"] == "cuda:0"
    assert report["samples"] == {"train": 12, "validation": 6}
    assert report["fitted_rows"] == 12 and report["final_test_opened"] is False
    for partition, count in (("train", 12), ("validation", 6)):
        prediction = report["predictions"][partition]
        rows = pq.read_table(output / prediction["path"]).to_pylist()
        assert len({row["sample_id"] for row in rows}) == count
        assert prediction["metrics"]["session_count"] == count // 2
        assert prediction["metrics"]["session_mae"] >= 0
    assert (
        module().run_external_reference(manifest, output, rounds=3, batch_size=5, resume=True)
        == report
    )
    with pytest.raises(ValueError, match="nuev|existe"):
        module().run_external_reference(manifest, output, rounds=3, batch_size=5)


def test_paused_tree_round_resumes_without_changing_predictions(tmp_path, monkeypatch):
    engine = module()
    manifest = training_corpus(tmp_path / "data")
    continuous = engine.run_external_reference(manifest, tmp_path / "full", rounds=3)
    stop = StopRequest()
    original_save = engine.ExternalBoostingModel.save

    def pause(model, path):
        digest = original_save(model, path)
        if model.booster.num_boosted_rounds() == 1:
            stop.request_stop()
        return digest

    monkeypatch.setattr(engine.ExternalBoostingModel, "save", pause)
    report = engine.run_external_reference(
        manifest,
        tmp_path / "resumed",
        rounds=3,
        checkpoint_interval=1,
        stop=stop,
    )
    assert report["status"] == "paused" and report["completed_rounds"] == 1
    monkeypatch.setattr(engine.ExternalBoostingModel, "save", original_save)
    result = engine.run_external_reference(
        manifest,
        tmp_path / "resumed",
        rounds=3,
        checkpoint_interval=1,
        resume=True,
    )
    assert result["status"] == "completed" and result["completed_rounds"] == 3
    for partition in ("train", "validation"):
        table = pq.read_table(tmp_path / "full" / continuous["predictions"][partition]["path"])
        assert table.equals(
            pq.read_table(tmp_path / "resumed" / result["predictions"][partition]["path"])
        )
    with pytest.raises(ValueError, match="identidad|configuración"):
        engine.run_external_reference(manifest, tmp_path / "resumed", rounds=4, resume=True)
    checkpoint = tmp_path / "resumed" / result["checkpoint"]["path"]
    checkpoint.write_bytes("Modelo dañado".encode())
    with pytest.raises(ValueError, match="huella|integridad"):
        engine.run_external_reference(
            manifest,
            tmp_path / "resumed",
            rounds=3,
            checkpoint_interval=1,
            resume=True,
        )
