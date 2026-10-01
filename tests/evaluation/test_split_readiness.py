"""La preparación temporal no presenta una campaña sin macros como ejecutable."""

import importlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from tests.evaluation.test_splits import configuration


def module():
    try:
        return importlib.import_module("mars_titan.evaluation.split_readiness")
    except ModuleNotFoundError:
        pytest.fail("Falta la comprobación de admisión de ventanas temporales")


def inputs(tmp_path):
    config = tmp_path / "inputs" / "protocol.json"
    admission = config.with_name("admission.json")
    atomic_json(config, configuration())
    artifact = config.with_name("complete-decisions.parquet")
    pq.write_table(
        pa.table(
            {
                "prediction_at": pa.array([], type=pa.timestamp("us", tz="UTC")),
                "macro_available_at": pa.array([], type=pa.timestamp("us", tz="UTC")),
                "indicator_count": pa.array([], type=pa.int32()),
            }
        ),
        artifact,
    )
    atomic_json(
        admission,
        dict(
            schema_version=1,
            policy="all_catalog_indicators_valid",
            market="US",
            start="2019-01-01",
            end="2023-12-31",
            source_sha256="a" * 64,
            catalog_sha256="b" * 64,
            required_indicator_ids=["macro_a", "macro_b"],
            required_indicator_count=2,
            total_decisions=1260,
            complete_decisions=0,
            excluded_decisions=1260,
            population_ready=False,
            missing_by_indicator={"macro_a": 100, "macro_b": 1260},
            complete_decisions_sha256=sha256(artifact),
        ),
    )
    return config, admission


def test_zero_complete_dates_blocks_all_windows_and_keeps_final_test_sealed(tmp_path):
    config, admission = inputs(tmp_path)
    output = tmp_path / "result.json"
    result = module().prepare_readiness(config, admission, output)
    assert result["status"] == "blocked"
    assert result["reason"] == "no_complete_macro_decisions"
    assert result["runnable_folds"] == []
    assert len(result["folds"]) == 5
    assert result["final_test"] == dict(
        start="2024-01-01", end="2025-01-01", opened=False, materialized=False
    )
    assert result["protocol_sha256"] == sha256(config)
    assert result["admission_sha256"] == sha256(admission)
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize(
    "change",
    [
        dict(population_ready=True),
        dict(complete_decisions=1),
        dict(market="CN"),
        dict(policy="masked_values_allowed"),
        dict(start="2020-01-01"),
        dict(end="2022-12-31"),
        dict(source_sha256="bad"),
        dict(required_indicator_count=140),
    ],
)
def test_inconsistent_admission_cannot_enable_a_protocol(tmp_path, change):
    config, admission = inputs(tmp_path)
    atomic_json(admission, json.loads(admission.read_text()) | change)
    with pytest.raises(ValueError):
        module().prepare_readiness(config, admission, tmp_path / "result.json")


def test_output_cannot_overwrite_input(tmp_path):
    config, admission = inputs(tmp_path)
    original = admission.read_bytes()
    with pytest.raises(ValueError):
        module().prepare_readiness(config, admission, admission)
    assert admission.read_bytes() == original


def test_cli_returns_blocked_status_without_starting_training(tmp_path, capsys):
    config, admission = inputs(tmp_path)
    code = module().main(
        [
            "--protocol",
            str(config),
            "--admission",
            str(admission),
            "--output",
            str(tmp_path / "result.json"),
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_macro_ready_windows_still_require_real_multimodal_rows_and_labels(tmp_path):
    config, admission = inputs(tmp_path)
    clock = MarketClock("US", "2019-01-01", "2023-12-31")
    artifact = admission.with_name("complete-decisions.parquet")
    table = pa.table(
        {
            "prediction_at": pa.array(clock.decisions, type=pa.timestamp("us", tz="UTC")),
            "macro_available_at": pa.array(clock.decisions, type=pa.timestamp("us", tz="UTC")),
            "indicator_count": pa.array([2] * len(clock.decisions), type=pa.int32()),
        }
    )
    pq.write_table(table, artifact)
    report = json.loads(admission.read_text())
    report.update(
        total_decisions=len(table),
        complete_decisions=len(table),
        excluded_decisions=0,
        population_ready=True,
        complete_decisions_sha256=sha256(artifact),
    )
    atomic_json(admission, report)
    result = module().prepare_readiness(config, admission, tmp_path / "result.json")
    assert result["status"] == "macro_windows_ready"
    assert len(result["macro_ready_folds"]) == 5
    assert result["runnable_folds"] == []
    assert result["scientific_training_started"] is False
    assert all(fold["macro_sessions"]["train"] > 500 for fold in result["folds"])


def test_corrupted_complete_index_is_rejected(tmp_path):
    config, admission = inputs(tmp_path)
    admission.with_name("complete-decisions.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="huella"):
        module().prepare_readiness(config, admission, tmp_path / "result.json")


def test_recent_macro_coverage_cannot_claim_three_years_of_training(tmp_path):
    config, admission = inputs(tmp_path)
    clock = MarketClock("US", "2021-01-01", "2023-12-31")
    artifact = admission.with_name("complete-decisions.parquet")
    table = pa.table(
        {
            "prediction_at": pa.array(clock.decisions, type=pa.timestamp("us", tz="UTC")),
            "macro_available_at": pa.array(clock.decisions, type=pa.timestamp("us", tz="UTC")),
            "indicator_count": pa.array([2] * len(clock.decisions), type=pa.int32()),
        }
    )
    pq.write_table(table, artifact)
    report = json.loads(admission.read_text())
    report.update(
        total_decisions=len(table),
        complete_decisions=len(table),
        excluded_decisions=0,
        population_ready=True,
        complete_decisions_sha256=sha256(artifact),
    )
    atomic_json(admission, report)
    result = module().prepare_readiness(config, admission, tmp_path / "result.json")
    assert result["status"] == "blocked"
    assert result["macro_ready_folds"] == []
