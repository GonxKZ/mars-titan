"""Preflight temporal completo con corpus técnicos y guardas reales, sin entrenar."""

import json
import shutil

import numpy as np
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.posttraining.completion import _inputs
from mars_titan.posttraining.preparation import encoder_contract
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.temporal_search import _inputs as neural_inputs
from tests.training.temporal_fixture import confirmed_references, temporal_fixture


@pytest.mark.parametrize("market", ["US", "CN"])
def test_completion_admits_a_whole_technical_market_with_real_contracts(tmp_path, market):
    fixture = temporal_fixture(tmp_path, market)
    records, _, count = neural_inputs(fixture.neural_config, fixture.views)
    assert len(records) == 10 and count == 3
    assert not fixture.reference.exists()
    dataset = CorpusDataset(records[0]["manifest"])
    assert dataset.manifest["counts"] == dict(train=6, validation=2, calibration=1, evaluation=1)
    for partition, expected in dataset.manifest["counts"].items():
        batches = list(dataset.batches(partition=partition, batch_size=8, epoch=0, seed=42))
        assert sum(len(batch["target"]) for batch in batches) == expected
        for batch in batches:
            assert set(batch["market"]) == {market}
            assert batch["inputs"]["prices"].shape[1:] == (64, 5)
            assert batch["inputs"]["news"].shape[1] == 384
            assert batch["inputs"]["charts"].shape[1] == 512
            assert batch["inputs"]["fundamentals"].shape[1] == 9
            assert np.all(batch["inputs"]["macro"][:, 140:280] == 1)
    assert encoder_contract(records[0]["manifest"], fixture.encoded)["context"] == 64
    confirmed_references(fixture)
    identity, stages = _inputs(fixture)
    assert identity["market"] == market
    assert len(identity["references"]) == 10
    assert {p["arm"] for p in identity["references"].values()} == {market}
    assert len(stages) == 30
    assert [stage["stage"] for stage in stages[-10:]] == ["evaluation"] * 10
    assert [stage["planned_runs"] for stage in stages[-10:]] == [62] * 10


def test_completion_rejects_individually_valid_folds_from_different_markets(tmp_path):
    chinese = temporal_fixture(tmp_path / "cn", "CN")
    american = temporal_fixture(tmp_path / "us", "US")
    confirmed_references(chinese)
    confirmed_references(american)
    destination = chinese.reference / "fold-001"
    shutil.rmtree(destination)
    shutil.copytree(american.reference / "fold-001", destination)
    with pytest.raises(ValueError, match="mercado"):
        _inputs(chinese)


def test_completion_rejects_a_reference_market_that_disagrees_with_its_protocol(tmp_path):
    fixture = temporal_fixture(tmp_path, "CN")
    confirmed_references(fixture)
    root = fixture.reference / "fold-000"
    previous = root / "views/CN.json"
    renamed = root / "views/US.json"
    previous.rename(renamed)
    summary = json.loads((root / "summary.json").read_text())
    summary["identity"]["configuration"]["arms"] = ["US"]
    for run in summary["runs"]:
        run["arm"] = "US"
        assert json.loads((root / run["path"] / "run.json").read_text())["identity"][
            "manifest_sha256"
        ] == sha256(renamed)
    atomic_json(root / "summary.json", summary)
    with pytest.raises(ValueError, match="mercado"):
        _inputs(fixture)


@pytest.mark.parametrize("change", [{"arm": "US"}, {"weighting": "balanced_markets"}])
def test_completion_rejects_a_run_outside_the_declared_single_market(tmp_path, change):
    fixture = temporal_fixture(tmp_path, "CN")
    confirmed_references(fixture)
    _inputs(fixture)
    path = fixture.reference / "fold-000/summary.json"
    summary = json.loads(path.read_text())
    summary["runs"][0].update(change)
    atomic_json(path, summary)
    with pytest.raises(ValueError, match="mercado"):
        _inputs(fixture)
