"""Recorrido completo, cortes operativos y recuperación por activo."""

import importlib
import json

import pytest

from mars_titan.data.corpus_catalog import corpus_candidates, index_news
from mars_titan.data.inventory import inventory as update_inventory
from mars_titan.data.news_registry import initialize_verification
from mars_titan.data.storage import sha256
from tests.data.test_corpus_catalog import source_fixture


def setup(tmp_path):
    source, inventory, _ = source_fixture(
        tmp_path, symbols=("AAA", "BBB", "CCC"), missing=(("CCC", "fundamentals"),)
    )
    index = tmp_path / "index.sqlite"
    index_news(source, corpus_candidates(source, inventory, "US"), index, cutoff="2023-12-31")
    reviews = tmp_path / "reviews.sqlite"
    initialize_verification(index, reviews)
    return source, inventory, reviews


def prepare(*args, **kwargs):
    try:
        module = importlib.import_module("mars_titan.data.corpus_preparation")
    except ModuleNotFoundError:
        pytest.fail("Falta el recorrido recuperable de cohortes")
    return module.prepare_cohort(*args, **kwargs)


def test_operational_pause_resumes_without_losing_incomplete_assets(tmp_path):
    source, inventory, reviews = setup(tmp_path)
    output = tmp_path / "edition"
    args = (source, inventory, reviews, output)
    first = prepare(*args, cohort="original_audited", markets=("US",), stop_after_assets=1)
    assert first["status"] == "paused"
    assert first["candidate_count"] == 3 and len(first["assets"]) == 1
    path = output / "prepared/US/AAA/manifest.json"
    digest, modified = sha256(path), path.stat().st_mtime_ns
    final = prepare(*args, cohort="original_audited", markets=("US",))
    assert final["status"] == "completed" and final["failed_assets"] == 0
    assert len(final["assets"]) == 3
    assert final["assets"][-1]["state"] == "missing_modalities"
    assert final["training_ready"] is False
    assert sha256(path) == digest and path.stat().st_mtime_ns == modified
    assert final["snapshot_sha256"] == sha256(output / "reviews.sqlite")


def test_failed_asset_remains_visible_and_does_not_stop_other_assets(tmp_path):
    source, inventory, reviews = setup(tmp_path)
    path = source / "text/sp500_news/AAA.jsonl"
    path.write_text("{}\n")
    result = prepare(
        source, inventory, reviews, tmp_path / "output", cohort="original_audited", markets=("US",)
    )
    assert result["status"] == "completed_with_errors"
    assert result["failed_assets"] == 1
    assert result["assets"][0]["state"] == "failed"
    assert result["assets"][1]["state"] == "prepared"


def test_recovery_cannot_switch_cohort_or_adopt_unrelated_files(tmp_path):
    source, inventory, reviews = setup(tmp_path)
    output = tmp_path / "output"
    prepare(source, inventory, reviews, output, cohort="original_audited", markets=("US",))
    with pytest.raises(ValueError, match="configuración|edición"):
        prepare(source, inventory, reviews, output, cohort="externally_verified", markets=("US",))
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "note.json").write_text(json.dumps({"retain": True}))
    with pytest.raises(ValueError, match="edición|directorio"):
        prepare(source, inventory, reviews, unrelated, cohort="original_audited", markets=("US",))
    assert json.loads((unrelated / "note.json").read_text()) == {"retain": True}


def test_malformed_fundamentals_are_reported_without_hiding_other_assets(tmp_path):
    source, inventory, reviews = setup(tmp_path)
    (source / "table/financial_reports/aaa/facts.json").write_text("not valid json")
    update_inventory(source, inventory)
    result = prepare(
        source, inventory, reviews, tmp_path / "output", cohort="original_audited", markets=("US",)
    )
    assert result["failed_assets"] == 1
    assert result["assets"][0]["state"] == "failed"
    assert result["assets"][1]["state"] == "prepared"


def test_missing_inventoried_file_does_not_prevent_preparing_other_assets(tmp_path):
    source, inventory, reviews = setup(tmp_path)
    (source / "time_series/S&P500_time_series/aaa.csv").unlink()
    result = prepare(
        source, inventory, reviews, tmp_path / "output", cohort="original_audited", markets=("US",)
    )
    assert result["status"] == "completed_with_errors" and result["failed_assets"] == 1
    assert result["assets"][0]["state"] == "failed"
    assert result["assets"][1]["state"] == "prepared"


def test_unexpected_fact_shape_does_not_prevent_preparing_other_assets(tmp_path):
    source, inventory, reviews = setup(tmp_path)
    fact = dict(start=[], end="2022-03-31", filed="2022-05-01", val=100, accn="x")
    payload = dict(filings=[dict(facts={"us-gaap": {"Assets": {"units": {"USD": [fact]}}}})])
    (source / "table/financial_reports/aaa/facts.json").write_text(json.dumps(payload))
    update_inventory(source, inventory)
    result = prepare(
        source, inventory, reviews, tmp_path / "output", cohort="original_audited", markets=("US",)
    )
    assert result["status"] == "completed" and result["failed_assets"] == 0
    assert result["assets"][0]["state"] == "prepared"
    assert result["assets"][0]["counts"]["fundamentals"] == 0
    receipt = json.loads((tmp_path / "output/prepared/US/AAA/manifest.json").read_text())
    assert receipt["fundamentals_audit"]["invalid"] == 1
    assert result["assets"][1]["state"] == "prepared"
