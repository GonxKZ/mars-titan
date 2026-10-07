"""Recorrido CUDA técnico conjunto. Requiere una ventana exclusiva autorizada."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.corpus_source import prepare_causal_corpus
from mars_titan.posttraining.completion import _inputs as completion_inputs
from mars_titan.posttraining.heldout import evaluate_partition
from mars_titan.posttraining.parents import load_parent
from mars_titan.training import temporal_search
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.experiment_resources import GpuLease
from mars_titan.training.joint_temporal_corpus import prepare_joint_temporal_corpus
from mars_titan.training.temporal_search import run_temporal_search
from tests.training.test_joint_temporal import joint_fixture, market_sources, rows


@pytest.mark.skipif(
    os.environ.get("MARS_TITAN_CUDA_INTEGRATION") != "1",
    reason="Requiere una ventana CUDA exclusiva y activación explícita",
)
def test_joint_cuda_search_confirms_and_resumes_thirty_cases_without_losing_markets(
    tmp_path, monkeypatch
):
    assert torch.cuda.is_available(), "CUDA no disponible, sin sustitución por CPU"
    torch.cuda.set_device("cuda:0")
    fixture = joint_fixture(tmp_path, fold=None)
    views = tmp_path / "views"
    prepare_joint_temporal_corpus(fixture.manifest, market_sources(fixture), views)
    plan = json.loads(fixture.local["US"].neural_config.read_text())
    plan["arms"] = ["US+CN"]
    config = tmp_path / "neural.json"
    atomic_json(config, plan)
    output = tmp_path / "reference"
    stop = StopRequest()
    search = temporal_search.run_search

    def pause_after_first_fold(*args, **kwargs):
        result = search(*args, **kwargs)
        stop.request_stop()
        return result

    monkeypatch.setattr(temporal_search, "StopRequest", lambda: stop)
    monkeypatch.setattr(temporal_search, "run_search", pause_after_first_fold)
    paused = run_temporal_search(config, views, output)
    assert paused["status"] == "paused" and paused["completed_runs"] == 3
    first_summary = json.loads((output / "fold-000/summary.json").read_text())
    confirmed = {}
    for run in first_summary["runs"]:
        folder = output / "fold-000" / run["path"]
        report = json.loads((folder / "run.json").read_text())
        path = folder / report["checkpoint"]["path"]
        confirmed[path] = (sha256(path), path.stat().st_mtime_ns)
    monkeypatch.setattr(temporal_search, "StopRequest", StopRequest)
    monkeypatch.setattr(temporal_search, "run_search", search)
    result = run_temporal_search(config, views, output, resume=True)
    assert result["status"] == "completed"
    assert result["completed_runs"] == result["planned_runs"] == 30
    assert all(
        (sha256(path), path.stat().st_mtime_ns) == state for path, state in confirmed.items()
    )
    artifacts = {}
    for fold in result["folds"]:
        dataset = CorpusDataset(views / fold["id"] / "manifest.json")
        expected = rows(dataset, "validation")
        summary = json.loads((output / fold["id"] / "summary.json").read_text())
        for run in summary["runs"]:
            path = output / fold["id"] / run["path"] / "run.json"
            report = json.loads(path.read_text())
            predictions = path.parent / report["predictions"]["validation"]["path"]
            table = pq.read_table(predictions)
            assert set(table["sample_id"].to_pylist()) == expected.keys()
            assert set(table["market"].to_pylist()) == {"US", "CN"}
            artifacts[path] = sha256(path)
            artifacts[predictions] = sha256(predictions)
    repeated = run_temporal_search(config, views, output, resume=True)
    assert repeated["status"] == "completed" and repeated["completed_runs"] == 30
    assert all(sha256(path) == signature for path, signature in artifacts.items())
    args = SimpleNamespace(
        reference=output,
        encoded=fixture.local["US"].encoded,
        tabular_config=fixture.local["US"].tabular_config,
        post_config=fixture.local["US"].post_config,
    )
    identity, stages = completion_inputs(args)
    assert identity["market"] == "US+CN" and len(stages) == 30
    assert all(proof["arm"] == "US+CN" for proof in identity["references"].values())
    proof = identity["references"]["fold-000"]
    ordered = tmp_path / "ordered"
    prepare_causal_corpus(Path(proof["manifest"]), ordered)
    winner = first_summary["selected"]["US+CN-natural/gru"]
    run = next(row for row in first_summary["runs"] if row["id"] == winner)
    report_path = output / "fold-000" / run["path"] / "run.json"
    dataset = CorpusDataset(Path(proof["manifest"]))
    with GpuLease() as lease:
        parent = load_parent(ordered / "manifest.json", report_path, lease=lease)
        for partition in ("calibration", "evaluation"):
            path = tmp_path / f"heldout-{partition}.parquet"
            scores = evaluate_partition(
                dataset, parent, partition, path, stop=StopRequest(), device="cuda:0"
            )
            table = pq.read_table(path)
            assert set(table["sample_id"].to_pylist()) == rows(dataset, partition).keys()
            assert set(scores["prediction"]["by_market_session"]) == {"US", "CN"}
            assert scores["prediction"]["samples"] == dataset.manifest["counts"][partition]
        lease.check()
    assert all(
        (sha256(path), path.stat().st_mtime_ns) == state for path, state in confirmed.items()
    )
