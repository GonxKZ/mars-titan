"""Declaración estadística E2 previa a una ventana, sin inferencia ni aprendizaje."""

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mars_titan.memory.spaced_study import load_study
from tests.memory.spaced_analysis_fixture import ANALYSIS, analysis_document

ROOT = Path(__file__).resolve().parents[2]


def profile(contract):
    return ROOT / (
        "configs/experiments/spaced-regional-utility-study-v1.json"
        if contract == "regional"
        else "configs/experiments/spaced-utility-study-v1.json"
    )


def store(tmp_path, document):
    path = tmp_path / "study.json"
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize("contract", ["standard", "regional"])
def test_analysis_is_part_of_the_study_identity_before_execution(tmp_path, contract):
    document = analysis_document(json.loads(profile(contract).read_text()), contract)
    study, digest = load_study(store(tmp_path, document))
    assert len(digest) == 64
    assert study.analysis.identity() == ANALYSIS
    assert study.identity()["analysis"] == ANALYSIS
    assert study.identity()["reader_contract"] == contract
    assert study.retention_for("reservoir").policy == "reservoir"
    assert study.analysis.bootstrap_options() == {
        key: value
        for key, value in ANALYSIS.items()
        if key not in ("rank_ic_min_assets", "primary_contrast", "min_relevant_mae")
    }


@pytest.mark.parametrize("contract", ["standard", "regional"])
def test_saved_comparison_profiles_leave_the_original_configs_unchanged(contract):
    path = ROOT / (
        "configs/experiments/spaced-regional-utility-comparison-v1.json"
        if contract == "regional"
        else "configs/experiments/spaced-utility-comparison-v1.json"
    )
    assert path.is_file(), "Falta la declaración independiente de comparación E2"
    study, _ = load_study(path)
    original, _ = load_study(profile(contract))
    assert study.analysis.identity() == ANALYSIS
    assert original.analysis is None
    assert "analysis" not in original.identity()
    assert ("reader_contract" in original.identity()) is (contract == "regional")
    value = dict(study.identity())
    value.pop("analysis")
    if contract == "standard":
        value.pop("reader_contract")
    assert value == original.identity()


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_length", 0),
        ("block_length", True),
        ("replicates", 9),
        ("replicates", 100001),
        ("seed", -1),
        ("seed", True),
        ("confidence", 1.0),
        ("confidence", 0),
        ("market_weighting", "rows"),
        ("sensitivity_block_lengths", [16]),
        ("sensitivity_block_lengths", [8, 8]),
        ("sensitivity_block_lengths", [True]),
        ("sensitivity_block_lengths", "8,32"),
        ("rank_ic_min_assets", 2),
        ("rank_ic_min_assets", 8193),
        ("primary_contrast", "best_observed_minus_baseline"),
        ("min_relevant_mae", -0.1),
        ("min_relevant_mae", float("inf")),
        ("min_relevant_mae", True),
    ],
)
def test_invalid_analysis_cannot_be_loaded(tmp_path, field, value):
    document = analysis_document(json.loads(profile("standard").read_text()), "standard")
    document["analysis"][field] = value
    with pytest.raises(ValueError):
        load_study(store(tmp_path, document))


@pytest.mark.parametrize("problem", ["unknown", "missing", "none", "list"])
def test_analysis_requires_exactly_the_declared_options(tmp_path, problem):
    document = analysis_document(json.loads(profile("standard").read_text()), "standard")
    if problem == "unknown":
        document["analysis"]["select_best"] = True
    elif problem == "missing":
        document["analysis"].pop("seed")
    else:
        document["analysis"] = None if problem == "none" else []
    with pytest.raises(ValueError):
        load_study(store(tmp_path, document))


@pytest.mark.parametrize("problem", ["schema", "contract", "controls", "final", "enabled"])
def test_comparison_cannot_impersonate_a_legacy_or_incomplete_study(tmp_path, problem):
    document = analysis_document(json.loads(profile("standard").read_text()), "standard")
    if problem == "schema":
        document["schema_version"] = 1
    elif problem == "contract":
        document["reader_contract"] = "regional"
    elif problem == "controls":
        document["arms"].remove("recent")
    elif problem == "final":
        document["final_test_access"] = True
    else:
        document["default_enabled"] = True
    with pytest.raises(ValueError):
        load_study(store(tmp_path, document))


def test_exported_analysis_cannot_mutate_the_loaded_declaration(tmp_path):
    document = analysis_document(json.loads(profile("standard").read_text()), "standard")
    study, _ = load_study(store(tmp_path, document))
    first = study.analysis.identity()
    first["sensitivity_block_lengths"].append(99)
    assert study.analysis.identity() == ANALYSIS
    with pytest.raises(FrozenInstanceError):
        study.analysis.seed = 1


def test_changing_only_the_margin_changes_the_identity(tmp_path):
    document = analysis_document(json.loads(profile("standard").read_text()), "standard")
    first, first_sha = load_study(store(tmp_path, document))
    altered = copy.deepcopy(document)
    altered["analysis"]["min_relevant_mae"] = 0.00002
    second, second_sha = load_study(store(tmp_path, altered))
    assert first_sha != second_sha
    assert first.identity() != second.identity()
