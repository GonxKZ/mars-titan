"""Planes y recibos experimentales sin recorrer datos ni estados privados."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from mars_titan.observatory.collector import Collector, planned_runs

PARENTS = {
    "rnn": "neural",
    "lstm": "neural",
    "gru": "neural",
    "dlinear": "neural",
    "ridge": "tabular",
    "xgboost": "tabular",
}
MIB = 1024**2


def configuration(path):
    return json.loads((Path(__file__).parents[2] / "configs" / path).read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source(kind="archive", domain="technical"):
    return {"id": "experiment", "path": "campaign", "kind": kind, "domain": domain}


def native_receipt():
    return {
        "schema_version": 1,
        "activity": "simulation",
        "model": "cash",
        "domain": "technical",
        "status": "completed",
        "global_step": 4,
        "total_steps": 4,
        "partition": "validation",
        "currency": "USD",
        "cost_bps": 10,
        "parent_frozen": True,
        "final_test_opened": False,
        "started_at_utc": "2026-01-01T00:00:00Z",
        "updated_at_utc": "2026-01-01T00:01:00Z",
        "total_seconds": 1.5,
        "identity": {
            "manifest_sha256": "a" * 64,
            "config": {"capital": 1000, "participation": 0.1, "cost_bps": 10},
        },
        "financial_validation": {
            "net_return": 0,
            "max_drawdown": 0,
            "costs": 0,
            "turnover": 0,
            "steps": 4,
            "completed": True,
            "invalid_reason": None,
        },
        "executable_peak_rss_bytes": 64 * MIB,
        "process_lifetime_peak_rss_bytes": 512 * MIB,
        "checkpoint": {
            "step": 4,
            "saved_at": "2026-01-01T00:01:00Z",
            "resumable": True,
            "path": "private/checkpoints/state.json",
        },
    }


def test_paired_plan_counts_modes_seeds_conditions_and_declared_families():
    config = configuration("baselines/paired-posttraining.json")
    assert planned_runs("paired_posttraining", config, parents=PARENTS) == 396
    assert planned_runs("paired_posttraining", config, parents={"ridge": "tabular"}) == 54
    assert planned_runs("paired_posttraining", config, parents={"gru": "neural"}) == 72
    config["seeds"] = [42]
    assert planned_runs("paired_posttraining", config, parents=PARENTS) == 132
    config["conditions"] = ["real"]
    assert planned_runs("paired_posttraining", config, parents=PARENTS) == 44


@pytest.mark.parametrize("parents", [6, {}, {"gru": "unknown"}, {"gru": ["neural"]}])
def test_paired_plan_rejects_missing_or_unknown_families(parents):
    with pytest.raises(ValueError):
        planned_runs(
            "paired_posttraining",
            configuration("baselines/paired-posttraining.json"),
            parents=parents,
        )


def test_financial_plan_counts_training_without_multiplying_reference_evaluations():
    config = configuration("simulation/comparators.json")
    assert planned_runs("financial", config) == 6
    config["seeds"] = [42]
    assert planned_runs("financial", config) == 2


@pytest.mark.parametrize("field,value", [("seeds", [True]), ("seeds", [42, 42]), ("modes", [""])])
def test_paired_plan_rejects_ambiguous_lists(field, value):
    config = configuration("baselines/paired-posttraining.json")
    config[field] = value
    with pytest.raises(ValueError):
        planned_runs("paired_posttraining", config, parents=PARENTS)


def test_configuration_pin_is_checked_even_when_the_file_is_cached(tmp_path):
    path = tmp_path / "frozen/paired.json"
    dump(path, configuration("baselines/paired-posttraining.json"))
    entry = {
        **source("paired_posttraining", "real"),
        "configuration": "frozen/paired.json",
        "configuration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parents": PARENTS,
    }
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        assert collector.collect([entry])["campaigns"][0]["planned_runs"] == 396
        with pytest.raises(ValueError, match="huella|configuración"):
            collector.collect([{**entry, "configuration_sha256": "0" * 64}])
        path.write_text(path.read_text() + "\n")
        with pytest.raises(ValueError, match="huella|configuración"):
            collector.collect([entry])


def test_old_checkouts_use_a_pinned_snapshot_only_when_the_config_is_missing(tmp_path):
    frozen = tmp_path / "frozen/paired.json"
    dump(frozen, configuration("baselines/paired-posttraining.json"))
    entry = {
        **source("paired_posttraining", "real"),
        "configuration": "configs/paired.json",
        "configuration_snapshot": "frozen/paired.json",
        "configuration_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "parents": PARENTS,
    }
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        assert collector.collect([entry])["campaigns"][0]["planned_runs"] == 396
        canonical = tmp_path / entry["configuration"]
        canonical.parent.mkdir()
        canonical.write_bytes(frozen.read_bytes())
        assert collector.collect([entry])["campaigns"][0]["planned_runs"] == 396
        canonical.write_text(canonical.read_text() + "\n")
        with pytest.raises(ValueError, match="huella"):
            collector.collect([entry])


@pytest.mark.parametrize("mode", ["klpo_full", "neural_mae", "neural_mse"])
def test_paired_dictionary_receipts_keep_primary_metrics_and_parent(mode, tmp_path):
    relative = f"parents/gru/runs/seed-42/real_synthetic/{mode}/run.json"
    metrics = {"primary": "median", "mae": 0.2, "mse": 0.1, "center": {"mae": 0.3}}
    report = {
        "schema_version": 1,
        "activity": "supervised_continuation"
        if mode.startswith("neural_")
        else "predictive_adaptation",
        "model": "gru",
        "domain": "real",
        "status": "completed",
        "global_step": 4,
        "total_steps": 4,
        "final_test_opened": False,
        "identity": {
            "case": {"mode": mode, "condition": "real_synthetic", "seed": 42, "epochs": 1},
            "dataset": {"train_sha256": "a" * 64, "validation_sha256": "b" * 64},
            "parent": {"model": "gru", "checkpoint_sha256": "c" * 64},
        },
        "predictions": {"validation": {"metrics": metrics}},
        "epochs": [{"epoch": 1, "validation": metrics}],
        "attempts": [
            {"total_seconds": 1.5, "process_lifetime_peak_rss_bytes": 128 * MIB},
            {"total_seconds": 2.5, "process_lifetime_peak_rss_bytes": 256 * MIB},
        ],
    }
    dump(tmp_path / "campaign" / relative, report)
    dump(
        tmp_path / "campaign/summary.json",
        {
            "status": "completed",
            "runs": {
                "gru/seed-42/real_synthetic/" + mode: {"path": relative, "status": "completed"}
            },
        },
    )
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        runs = collector.collect([source("paired_posttraining", "real")])["runs"]
    assert len(runs) == 1
    run = runs[0]
    assert run["activity"] == report["activity"]
    assert run["model_id"] == ("gru" if mode.startswith("neural_") else "adaptation")
    assert run["metrics"]["mae"] == run["history"][0]["mae"] == 0.2
    assert run["metrics"]["elapsed_seconds"] == 4
    assert run["metrics"]["ram_peak_mib"] == 256
    assert run["metadata"]["method"] == mode
    assert run["metadata"]["condition"] == "real_synthetic"
    assert run["metadata"]["parent_model"] == "gru"
    assert run["comparison_group"] is not None


def test_native_receipt_exports_memory_scopes_and_ignores_private_files(tmp_path, monkeypatch):
    report = native_receipt()
    dump(tmp_path / "campaign/cash/run.json", report)
    dump(tmp_path / "campaign/cash/private/checkpoints/run.json", {"sensitive": True})
    dump(tmp_path / "campaign/cash/private/checkpoints/latest.json", {"sensitive": True})
    (tmp_path / "campaign/cash/market.parquet").write_bytes(b"No se debe abrir")
    dump(
        tmp_path / "campaign/comparison.json",
        {
            "status": "completed",
            "runs": [{"name": "cash", "path": "cash/run.json", "status": "completed"}],
        },
    )
    opened = []
    read = Collector.read

    def checked_read(self, path, **kwargs):
        opened.append(path.relative_to(tmp_path))
        assert "private" not in path.relative_to(tmp_path).parts
        assert path.suffix != ".parquet"
        return read(self, path, **kwargs)

    monkeypatch.setattr(Collector, "read", checked_read)
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        snapshot = collector.collect([{**source(), "summary": "comparison.json"}])
    assert len(snapshot["runs"]) == 1
    assert snapshot["campaigns"][0]["status"] == "completed"
    run = snapshot["runs"][0]
    assert run["metrics"]["ram_peak_mib"] == 64
    assert run["metadata"]["ram_peak_scope"] == "executable"
    assert run["metadata"]["executable_peak_rss_mib"] == 64
    assert run["metadata"]["process_lifetime_peak_rss_mib"] == 512
    assert run["checkpoint"] == {"step": 4, "saved_at": "2026-01-01T00:01:00Z", "resumable": True}
    assert run["financial_validation"]["net_return"] == 0
    assert opened
    assert "private" not in json.dumps(snapshot)


def test_native_comparison_group_respects_configuration_capital_and_participation(tmp_path):
    groups = []
    for index, values in enumerate(({"capital": 1000}, {"capital": 2000}, {"participation": 0.5})):
        report = copy.deepcopy(native_receipt())
        report["identity"]["config"].update(values)
        dump(tmp_path / f"campaign/run-{index}/run.json", report)
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        groups = [run["comparison_group"] for run in collector.collect([source()])["runs"]]
    assert None not in groups and len(set(groups)) == 3


def test_native_zero_memory_is_observed_and_reserved_phases_hide_both_scopes(tmp_path):
    path = tmp_path / "campaign/cash/run.json"
    report = native_receipt()
    report["executable_peak_rss_bytes"] = 0
    dump(path, report)
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        run = collector.collect([source()])["runs"][0]
        assert run["metrics"]["ram_peak_mib"] == 0
        assert run["metadata"]["ram_peak_scope"] == "executable"
        report["phase"] = "test"
        dump(path, report)
        sealed = collector.collect([source()])["runs"][0]
        assert sealed["metrics"]["ram_peak_mib"] is None
        assert sealed["metadata"]["executable_peak_rss_mib"] is None
        assert sealed["metadata"]["process_lifetime_peak_rss_mib"] is None


def test_private_paths_cannot_be_selected_explicitly(tmp_path):
    dump(tmp_path / "campaign/private/checkpoints/summary.json", {"secret": True})
    with Collector(tmp_path, tmp_path / "cache.sqlite") as collector:
        with pytest.raises(ValueError, match="privad"):
            collector.collect([{**source(), "summary": "private/checkpoints/summary.json"}])


def test_receipt_discovery_does_not_count_summary_duplicates_toward_the_limit(tmp_path):
    dump(tmp_path / "campaign/case-0/run.json", native_receipt())
    dump(
        tmp_path / "campaign/summary.json",
        {
            "planned_runs": 4,
            "runs": [
                {"id": f"case-{index}", "path": f"case-{index}", "status": "queued"}
                for index in range(4)
            ],
        },
    )
    with Collector(tmp_path, tmp_path / "cache.sqlite", max_files=4) as collector:
        assert len(collector.collect([source()])["runs"]) == 4
        dump(tmp_path / "campaign/extra/run.json", native_receipt())
        with pytest.raises(ValueError, match="Demasiados"):
            collector.collect([source()])
