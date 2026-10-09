"""Campaña y búsqueda con la política declarada en su configuración, sin actualizar pesos."""

import json
from pathlib import Path

import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.training import reference_campaign, reference_search
from mars_titan.training.reference_design import TRANSFORMER_OPTIONS
from tests.training.historical_temporal_fixture import historical_temporal_fixture
from tests.training.test_masked_reference_run import cpu_runner, masked_view  # noqa: F401
from tests.training.test_reference_run import training_corpus

CONFIG = Path("configs/baselines/historical-masked-reference-search-us.json")


def search_plan(**changes):
    plan = json.loads(CONFIG.read_text())
    plan.update(
        scope="full_corpus",
        models=["gru"],
        case_indices=[0],
        finalist_seeds=[42],
        max_epochs=2,
        patience=1,
        batch_size=2,
        posttraining_epochs=1,
        continuation_selection=dict(metric="session_mae", patience=1, min_delta=0.0),
    )
    plan.update(changes)
    return plan


def write(tmp_path, plan, name="search.json"):
    path = tmp_path / name
    path.write_text(json.dumps(plan))
    return path


def test_repository_configuration_declares_policy_budget_and_transformer_cases():
    plan, cases, digest = reference_search._configuration(CONFIG)
    assert plan["input_policy"] == HISTORICAL_MASKED and len(digest) == 64
    assert plan["batch_size"] <= 256
    assert [case["id"] for case in cases if case["case"]["kind"] == "transformer"] == [
        "transformer-00-s42",
        "transformer-10-s42",
    ]
    for item in cases:
        case = item["case"]
        assert case["selection"]["stopping"] == "fixed_budget"
        expected = {"hidden_size", "layers", "dropout"}
        if case["kind"] == "transformer":
            assert case["architecture"]["transformer"] == TRANSFORMER_OPTIONS
            expected.add("transformer")
        assert set(case["architecture"]) == expected


@pytest.mark.parametrize(
    "changes,message",
    [
        (dict(input_policy="historical"), "versión 4"),
        (dict(stopping="never"), "versión 4"),
        (dict(prediction_retention="all"), "versión 4"),
        (dict(context_sessions=32), "versión 4"),
        (dict(models=["transformer"], batch_size=512), "Transformer"),
        (
            dict(
                continuation_selection=dict(
                    metric="session_mae", patience=1, min_delta=0.0, stopping="fixed_budget"
                )
            ),
            "presupuesto fijo",
        ),
    ],
)
def test_search_version_four_rejects_undeclared_or_incompatible_choices(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        reference_search._configuration(write(tmp_path, search_plan(**changes)))


def test_previous_search_versions_keep_their_closed_family_list(tmp_path):
    plan = json.loads(Path("configs/baselines/strict-temporal-search-us.json").read_text())
    _, cases, _ = reference_search._configuration(
        write(tmp_path, plan | dict(models=["gru"], batch_size=256))
    )
    assert all("stopping" not in case["case"]["selection"] for case in cases)
    with pytest.raises(ValueError, match="Transformer"):
        reference_search._configuration(
            write(tmp_path, plan | dict(models=["transformer"], batch_size=256))
        )


def test_masked_search_propagates_policy_to_views_cases_and_receipts(tmp_path, cpu_runner):  # noqa: F811
    manifest = masked_view(tmp_path)
    plan = search_plan(models=["gru", "transformer"])
    summary = reference_search.run_search(write(tmp_path, plan), manifest, tmp_path / "search")
    assert summary["status"] == "completed"
    assert summary["completed_runs"] == summary["planned_runs"] == 6
    view = json.loads((tmp_path / "search/views/US.json").read_text())
    assert view["input_policy"] == HISTORICAL_MASKED
    assert "data/input_policy.py" in summary["identity"]["scientific"]["code"]
    stages = set()
    for item in summary["runs"]:
        report = json.loads((tmp_path / "search" / item["path"] / "run.json").read_text())
        identity = report["identity"]
        stages.add(item["stage"])
        assert identity["input_policy"] == HISTORICAL_MASKED
        assert identity["prediction_retention"] == "heldout_full_train_sessions_v1"
        assert identity["case"]["selection"]["stopping"] == "fixed_budget"
        assert set(report["predictions"]) == {"validation", "calibration", "evaluation"}
        assert item["session_mae"] == report["predictions"]["validation"]["metrics"]["session_mae"]
        if item["stage"] == "posttraining":
            assert report["initialization"]["parent_checkpoint_sha256"]
    assert stages == {"search", "posttraining"}
    resumed = reference_search.run_search(
        write(tmp_path, plan), manifest, tmp_path / "search", resume=True
    )
    assert resumed["status"] == "completed" and resumed["runs"] == summary["runs"]


def campaign_config(tmp_path, *, schema=3, **changes):
    recipe = json.loads(Path("configs/baselines/expanded-reference-variants.json").read_text())
    recipe.update(models=["gru"], seeds=[42], losses=["mse"], learning_rates=[0.001], epochs=2)
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    config = dict(
        schema_version=schema,
        scope="full_corpus",
        arms=["US"],
        recipe="recipe.json",
        batch_size=2,
        checkpoint_seconds=900,
        pooled_weightings=["natural"],
        final_test_opened=False,
    )
    if schema == 3:
        config.update(
            input_policy=HISTORICAL_MASKED,
            architecture=dict(hidden_size=32, layers=1, dropout=0.0),
            prediction_retention="heldout_full_train_sessions_v1",
        )
    config.update(changes)
    return write(tmp_path, config, "campaign.json")


def test_masked_campaign_declares_policy_architecture_and_retention(tmp_path, cpu_runner):  # noqa: F811
    fixture = historical_temporal_fixture(tmp_path / "source")
    summary = reference_campaign.run_reference_campaign(
        campaign_config(tmp_path), fixture.parent, tmp_path / "campaign"
    )
    assert summary["status"] == "completed" and summary["completed_runs"] == 3
    assert {k: summary["identity"][k] for k in ("input_policy", "mask_contract")} == (
        policy_identity(HISTORICAL_MASKED)
    )
    for item in summary["runs"]:
        assert item["case"]["architecture"] == dict(hidden_size=32, layers=1, dropout=0.0)
        report = json.loads((tmp_path / "campaign" / item["path"] / "run.json").read_text())
        assert report["identity"]["mask_fusion"] == "zero_after_projection_then_concat_presence"
        assert set(report["predictions"]) == {"validation"}
        assert report["train_summary"]["metrics"]["samples"] == report["samples"]["train"]
    resumed = reference_campaign.run_reference_campaign(
        campaign_config(tmp_path), fixture.parent, tmp_path / "campaign", resume=True
    )
    assert resumed["status"] == "completed"


def test_strict_campaign_keeps_its_identity_fields(tmp_path, cpu_runner):  # noqa: F811
    manifest = training_corpus(tmp_path / "data")
    config = campaign_config(tmp_path, schema=2, scope="development_snapshot", batch_size=5)
    summary = reference_campaign.run_reference_campaign(config, manifest, tmp_path / "campaign")
    assert summary["status"] == "completed"
    assert set(summary["identity"]) == {
        "config_sha256",
        "recipe_sha256",
        "manifest_sha256",
        "scientific",
    }
    for item in summary["runs"]:
        assert "architecture" not in item["case"]
        report = json.loads((tmp_path / "campaign" / item["path"] / "run.json").read_text())
        assert report["identity"]["model_family"] == "legacy_cost_probe"
        assert set(report["predictions"]) == {"train", "validation"}


@pytest.mark.parametrize(
    "changes",
    [
        dict(input_policy="historical"),
        dict(prediction_retention="all"),
        dict(architecture=dict(hidden_size=48, layers=1, dropout=0.0)),
        dict(architecture=None),
    ],
)
def test_campaign_three_rejects_incomplete_declarations(tmp_path, changes):
    with pytest.raises(ValueError):
        reference_campaign._configuration(campaign_config(tmp_path, **changes))


def test_masked_policy_needs_the_declared_campaign_version(tmp_path, cpu_runner):  # noqa: F811
    fixture = historical_temporal_fixture(tmp_path / "source")
    config = campaign_config(tmp_path, schema=2)
    with pytest.raises(ValueError):
        reference_campaign.run_reference_campaign(config, fixture.parent, tmp_path / "campaign")
    assert not (tmp_path / "campaign").exists()
