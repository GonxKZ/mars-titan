"""Entrenamiento técnico CN y recuperación con CUDA, fuera de las campañas científicas."""

import json
import os
from pathlib import Path

import pytest
import torch

from mars_titan.data.storage import sha256
from mars_titan.posttraining.completion import _inputs as completion_inputs
from mars_titan.training import temporal_search
from mars_titan.training.checkpoints import StopRequest
from tests.training.temporal_fixture import temporal_fixture


@pytest.mark.skipif(
    os.environ.get("MARS_TITAN_CUDA_INTEGRATION") != "1",
    reason="Requiere una ventana CUDA exclusiva y activación explícita",
)
def test_cn_cuda_search_recovers_confirmed_runs_and_admits_the_following_stages(
    tmp_path, monkeypatch
):
    assert torch.cuda.is_available(), "CUDA no disponible, sin sustitución por CPU"
    torch.cuda.set_device("cuda:0")
    fixture = temporal_fixture(tmp_path, "CN")
    stop = StopRequest()
    real_search = temporal_search.run_search

    def pause_after_first_fold(*args, **kwargs):
        result = real_search(*args, **kwargs)
        stop.request_stop()
        return result

    monkeypatch.setattr(temporal_search, "StopRequest", lambda: stop)
    monkeypatch.setattr(temporal_search, "run_search", pause_after_first_fold)
    paused = temporal_search.run_temporal_search(
        fixture.neural_config, fixture.views, fixture.reference
    )
    assert paused["status"] == "paused"
    assert paused["completed_runs"] == 3
    first = fixture.reference / "fold-000"
    summary = json.loads((first / "summary.json").read_text())
    checkpoints = []
    for case in summary["runs"]:
        folder = first / case["path"]
        report = json.loads((folder / "run.json").read_text())
        path = folder / report["checkpoint"]["path"]
        checkpoints.append((path, sha256(path), path.stat().st_mtime_ns))
        assert case["arm"] == "CN" and case["weighting"] == "natural"
        assert report["final_test_opened"] is False
    monkeypatch.setattr(temporal_search, "StopRequest", StopRequest)
    monkeypatch.setattr(temporal_search, "run_search", real_search)
    completed = temporal_search.run_temporal_search(
        fixture.neural_config, fixture.views, fixture.reference, resume=True
    )
    assert completed["status"] == "completed"
    assert completed["planned_runs"] == completed["completed_runs"] == 30
    assert completed["resources"]["device"] == "cuda:0"
    assert all(
        (sha256(path), path.stat().st_mtime_ns) == (digest, stamp)
        for path, digest, stamp in checkpoints
    )
    identity, stages = completion_inputs(fixture)
    assert identity["market"] == "CN"
    assert len(stages) == 30
    assert all(proof["arm"] == "CN" for proof in identity["references"].values())
    for proof in identity["references"].values():
        source = json.loads(Path(proof["manifest"]).read_text())
        assert source["technical_fixture"] is True
        assert source["temporal_view"]["protocol"]["market"] == "CN"
        assert source["final_test_opened"] is False
