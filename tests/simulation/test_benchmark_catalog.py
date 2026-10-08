"""Identidad y preparación técnica de benchmarks sin ajustar modelos."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mars_titan.data.storage import sha256
from mars_titan.simulation import adaptation_scenarios
from mars_titan.simulation.benchmark_catalog import (
    benchmark_catalog,
    prepare_benchmark_profile,
)
from mars_titan.simulation.storage import read_tape

CONFIG = Path("configs/benchmarks/scenarios.json")


def settings():
    return json.loads(CONFIG.read_text())


def small_settings():
    config = settings()
    config["benchmarks"] = config["benchmarks"][:2]
    config["profiles"] = {"basic": config["profiles"]["basic"]}
    config["profiles"]["basic"]["generator"] = dict(
        assets=2, sessions=24, warmup_sessions=6, regime_sessions=6
    )
    config["profiles"]["basic"]["worlds"] = dict(train=1, validation=1, audit=1)
    return config


def test_catalog_fixes_names_counts_distinct_seeds_and_measured_scope():
    before = settings()
    result = benchmark_catalog(before)
    assert result == benchmark_catalog(copy.deepcopy(before))
    assert before == settings()
    assert result["suite_id"] == "mars_benchmarks_v1"
    assert result["scientific_evaluation_executed"] is False
    assert result["difficulty_empirically_measured"] is False
    assert len(result["cases"]) == 240
    assert len({case["seed"] for case in result["cases"]}) == 240
    assert {case["name"] for case in result["cases"]} == {
        "Silencio",
        "Faro",
        "Eco",
        "Giro",
        "Retorno",
        "Contraste",
        "Tormenta",
        "Umbral",
    }
    assert result["total_price_rows"] == 307200
    assert all(case["evaluator_only"] == (case["split"] == "audit") for case in result["cases"])
    assert all("truth" not in case for case in result["cases"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("final_test_opened", True),
        ("schema_version", True),
        ("fit_hmm", True),
        ("suite_id", "../escape"),
    ],
)
def test_unknown_learning_switches_and_invalid_identity_fail_before_writing(tmp_path, key, value):
    config = small_settings()
    config[key] = value
    with pytest.raises(ValueError):
        prepare_benchmark_profile(config, "basic", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "field,value", [("id", "../bad"), ("family", "unknown"), ("name", ""), ("name", "Eco\nfalse")]
)
def test_invalid_benchmark_fields_are_rejected(field, value):
    config = small_settings()
    config["benchmarks"][0][field] = value
    with pytest.raises(ValueError):
        benchmark_catalog(config)


def test_duplicate_names_families_and_profile_seeds_are_rejected():
    for field in ("id", "name", "family"):
        config = settings()
        config["benchmarks"][1][field] = config["benchmarks"][0][field]
        with pytest.raises(ValueError):
            benchmark_catalog(config)
    config = settings()
    config["profiles"]["extended"]["seed_base"] = config["profiles"]["basic"]["seed_base"]
    with pytest.raises(ValueError, match="semillas"):
        benchmark_catalog(config)


def test_identity_tracks_recipe_changes_and_rejects_excess_volume():
    config = settings()
    initial = benchmark_catalog(config)["identity_sha256"]
    config["profiles"]["basic"]["seed_base"] += 5
    assert benchmark_catalog(config)["identity_sha256"] != initial
    config["profiles"]["basic"].update(
        generator=dict(assets=128, sessions=364, warmup_sessions=64, regime_sessions=64),
        worlds=dict(train=64, validation=64, audit=64),
    )
    with pytest.raises(ValueError, match="volumen"):
        benchmark_catalog(config)


def test_preparation_reuses_exact_generator_without_fitting_or_policy_evaluation(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("La preparación no puede ajustar un HMM")

    monkeypatch.setattr(adaptation_scenarios, "fit_hmm", forbidden)
    config = small_settings()
    output = tmp_path / "prepared"
    report = prepare_benchmark_profile(config, "basic", output)
    assert report["status"] == "prepared"
    assert report["scientific_evaluation_executed"] is False
    assert report["source_index_sha256"] == sha256(output / "index.json")
    assert len(report["records"]) == 6
    source = json.loads((output / "index.json").read_text())
    assert "hmm" not in source and not (output / "hmm.json").exists()
    for record, original in zip(report["records"], source["records"], strict=True):
        assert record["path"] == original["path"]
        assert record["manifest_sha256"] == original["manifest_sha256"]
        tape = read_tape(output / record["path"])
        reference = adaptation_scenarios.generate_scenario(tape.identity["source"]["generator"])
        np.testing.assert_array_equal(tape.prices, reference["tape"].prices)
        assert record["benchmark_id"] in {"mars_silencio_v1", "mars_faro_v1"}
        expected = next(item for item in config["benchmarks"] if item["family"] == record["family"])
        assert (record["benchmark_id"], record["benchmark_name"]) == (
            expected["id"],
            expected["name"],
        )
    saved = (output / "benchmarks.json").read_bytes()
    with pytest.raises(ValueError):
        prepare_benchmark_profile(config, "basic", output)
    assert (output / "benchmarks.json").read_bytes() == saved


def test_unknown_profile_and_unrepresentative_schedule_fail(tmp_path):
    with pytest.raises(ValueError, match="perfil"):
        prepare_benchmark_profile(small_settings(), "missing", tmp_path / "bad")
    config = settings()
    config["profiles"]["basic"]["generator"]["sessions"] = 40
    with pytest.raises(ValueError, match="tres"):
        benchmark_catalog(config)


def test_cli_describes_without_creating_outputs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_named_benchmarks.py",
            "--config",
            str(CONFIG),
            "--describe",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    description = json.loads(result.stdout)
    assert description["suite_id"] == "mars_benchmarks_v1"
    assert len(description["cases"]) == 240
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("invalid", [None, [], True, {"unexpected": 1}])
@pytest.mark.parametrize("field", ["benchmarks", "profiles"])
def test_malformed_containers_fail_closed(field, invalid):
    config = settings()
    config[field] = invalid
    with pytest.raises(ValueError):
        benchmark_catalog(config)


@pytest.mark.parametrize("field", ["generator", "worlds"])
def test_nested_fields_require_exact_schema(field):
    config = settings()
    config["profiles"]["basic"][field]["extra"] = 1
    with pytest.raises(ValueError):
        benchmark_catalog(config)


def test_identity_and_plan_do_not_depend_on_profile_dictionary_order():
    config = settings()
    original = benchmark_catalog(config)
    config["profiles"] = dict(reversed(list(config["profiles"].items())))
    assert benchmark_catalog(config) == original


def test_runtime_and_generator_source_are_part_of_identity(monkeypatch):
    config = settings()
    result = benchmark_catalog(config)
    assert set(result["identity"]["runtime"]) == {"python", "numpy", "pyarrow"}
    monkeypatch.setattr("mars_titan.simulation.benchmark_catalog.version", lambda _: "changed")
    assert benchmark_catalog(config)["identity_sha256"] != result["identity_sha256"]
    monkeypatch.setattr("mars_titan.simulation.benchmark_catalog.sha256", lambda _: "0" * 64)
    assert benchmark_catalog(config)["identity"]["sources"] != result["identity"]["sources"]


def test_output_symlink_and_generation_failure_cannot_publish_prepared_catalog(
    tmp_path, monkeypatch
):
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        prepare_benchmark_profile(small_settings(), "basic", symlink / "new")
    assert not list(target.iterdir())

    def broken(*args, **kwargs):
        raise OSError("Fallo de escritura controlado")

    monkeypatch.setattr(adaptation_scenarios, "_write_context", broken)
    output = tmp_path / "failed"
    with pytest.raises(OSError):
        prepare_benchmark_profile(small_settings(), "basic", output)
    assert not (output / "benchmarks.json").exists()
    assert json.loads((output / "index.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "args",
    [[], ["--describe", "--profile", "basic"], ["--output", "unused"], ["--describe", "--fit-hmm"]],
)
def test_cli_rejects_ambiguous_modes_and_training_flags(args):
    result = subprocess.run(
        [sys.executable, "scripts/prepare_named_benchmarks.py", *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_cli_prepares_a_profile_using_the_same_public_contract(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(small_settings()))
    output = tmp_path / "prepared"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_named_benchmarks.py",
            "--config",
            str(config),
            "--profile",
            "basic",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    saved = json.loads((output / "benchmarks.json").read_text())
    assert report["worlds"] == 6
    assert report["identity_sha256"] == saved["identity_sha256"]
    assert report["scientific_evaluation_executed"] is False
