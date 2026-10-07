"""Agregación de semillas y ventanas sin fabricar réplicas de mercado."""

import pyarrow as pa
import pytest

from mars_titan.evaluation.campaign_comparison import aggregate_results


def inputs():
    cases = []
    sessions = []
    for fold, identifier, score, selected in (
        ("f0", "seed42", 0.0, True),
        ("f0", "seed43", 4.0, True),
        ("f1", "seed42", 10.0, True),
        ("f0", "unselected", 0.0, False),
    ):
        case = dict(
            fold=fold,
            id=identifier,
            partition="evaluation",
            family="rnn",
            method="reference",
            included=selected,
            seed=43 if identifier == "seed43" else 42,
            primary="continuous",
            parent_id=None,
            best_epoch=None,
            selected_initial=None,
            stopped_early=None,
        )
        case.update(
            session_mae=score, session_mse=score**2, sessions=2, mae_parent=score, mae_zero=8.0
        )
        cases.append(case)
        for day in ("2023-01-03", "2023-01-04"):
            sessions.append(
                dict(
                    fold=fold,
                    id=identifier,
                    partition="evaluation",
                    family="rnn",
                    method="reference",
                    included=selected,
                    prediction_at=day,
                    market="US",
                    samples=3,
                    mae_prediction=score,
                    mae_parent=score,
                    mae_zero=8.0,
                )
            )
    return cases, pa.Table.from_pylist(sessions)


def test_folds_have_equal_weight_and_unselected_search_does_not_enter_the_mean():
    cases, sessions = inputs()
    result = aggregate_results(cases, sessions, block_lengths=(1,), repetitions=100, seed=42)
    overall = result["overall"][0]
    assert overall["models"] == 3
    assert overall["folds"] == 2
    assert overall["mean_fold_session_mae"] == 6
    assert overall["mean_fold_delta_zero"] == -2
    assert result["by_fold"][0]["session_mae"] == 2
    assert result["by_fold"][1]["session_mae"] == 10


def test_seeds_share_session_resampling_instead_of_increasing_sample_size():
    cases, sessions = inputs()
    result = aggregate_results(cases, sessions, block_lengths=(1,), repetitions=100, seed=42)
    first = next(row for row in result["intervals"] if row["fold"] == "f0")
    assert first["reference"] == "zero"
    assert first["sessions"] == 2
    assert first["models"] == 2
    assert first["estimate"] == -6
    assert first["lower"] == -6 and first["upper"] == -6


def test_missing_session_for_one_seed_is_rejected_instead_of_implicitly_reweighted():
    cases, sessions = inputs()
    rows = sessions.to_pylist()
    rows.pop(3)
    with pytest.raises(ValueError):
        aggregate_results(cases, pa.Table.from_pylist(rows), block_lengths=(1,), repetitions=100)


def test_duplicate_case_or_session_is_rejected():
    cases, sessions = inputs()
    with pytest.raises(ValueError):
        aggregate_results([*cases, cases[0]], sessions)
    with pytest.raises(ValueError):
        aggregate_results(cases, pa.concat_tables([sessions, sessions.slice(0, 1)]))


def test_duplicate_seed_is_rejected_even_if_case_names_differ():
    cases, sessions = inputs()
    cases[1]["seed"] = 42
    with pytest.raises(ValueError, match="semillas"):
        aggregate_results(cases, sessions)


def test_joint_bootstrap_keeps_each_market_separate_without_extra_sessions():
    cases, sessions = inputs()
    rows = sessions.to_pylist()
    chinese = [dict(row, market="CN", mae_prediction=row["mae_prediction"] + 20) for row in rows]
    for row in [*rows, *chinese]:
        row["mse_prediction"] = row["mae_prediction"] ** 2
    for case in cases:
        case["sessions"] = 4
    result = aggregate_results(
        cases, pa.Table.from_pylist([*rows, *chinese]), block_lengths=(1,), repetitions=100
    )
    assert {row["market"] for row in result["overall"]} == {"US", "CN"}
    assert {row["market"]: row["mean_fold_session_mae"] for row in result["overall"]} == {
        "US": 6,
        "CN": 26,
    }
    first = {row["market"]: row for row in result["intervals"] if row["fold"] == "f0"}
    assert first["US"]["estimate"] == -6
    assert first["CN"]["estimate"] == 14
    assert all(row["sessions"] == 2 and row["models"] == 2 for row in first.values())
    assert first["US"]["lower"] == first["US"]["upper"] == -6
    assert first["CN"]["lower"] == first["CN"]["upper"] == 14


def test_complete_pipeline_checks_archived_files_and_preserves_an_existing_output(tmp_path):
    import csv
    import json

    from mars_titan.data.storage import sha256
    from mars_titan.evaluation.campaign_comparison import compare_campaigns
    from tests.evaluation.test_comparison_sources import Campaign

    campaign = Campaign(tmp_path / "sources", materialize=True)
    output = tmp_path / "review"
    result = compare_campaigns(campaign.reference, campaign.completion, output, repetitions=100)
    assert result["status"] == "completed"
    assert result["counts"]["models"] == 7
    assert result["final_test_opened"] is False
    assert result["candidate_trained"] is False
    cases = list(csv.DictReader((output / "cases.csv").open()))
    assert len(cases) == 14
    assert all(len(row["checkpoint_sha256"]) == 64 for row in cases)
    assert all(len(row["source_report_sha256"]) == 64 for row in cases)
    assert all(float(row["session_mae"]) == pytest.approx(0.1) for row in cases)
    receipt = json.loads((output / "comparison.json").read_text())
    for name, digest in receipt["artifacts"].items():
        assert sha256(output / name) == digest
    digest = sha256(output / "comparison.json")
    with pytest.raises(ValueError, match="salida"):
        compare_campaigns(campaign.reference, campaign.completion, output, repetitions=100)
    assert sha256(output / "comparison.json") == digest


def test_joint_frozen_predictions_keep_population_counts_and_separate_intervals(tmp_path):
    import csv

    from mars_titan.evaluation.campaign_comparison import compare_campaigns
    from tests.evaluation.test_comparison_sources import Campaign

    campaign = Campaign(tmp_path / "sources", materialize=True, markets=("US", "CN"))
    output = tmp_path / "review"
    result = compare_campaigns(campaign.reference, campaign.completion, output, repetitions=100)
    assert result["counts"]["models"] == 7
    assert result["counts"]["prediction_files"] == 14
    assert result["method"]["market_stratification"] == "separate_markets"
    assert result["provenance"]["folds"][0]["market_counts"] == {
        market: dict(train=4, validation=2, calibration=2, evaluation=2) for market in ("US", "CN")
    }
    cases = list(csv.DictReader((output / "cases.csv").open()))
    assert len(cases) == 14
    assert all(row["sessions_US"] == row["sessions_CN"] == "1" for row in cases)
    assert all(row["samples_US"] == row["samples_CN"] == "2" for row in cases)
    assert {row["market"] for row in result["intervals"]} == {"US", "CN"}
    assert all(row["sessions"] == 1 and row["lower"] is None for row in result["intervals"])


def test_output_ancestor_of_science_is_rejected_before_reading_sources(tmp_path):
    from mars_titan.evaluation.campaign_comparison import compare_campaigns

    with pytest.raises(ValueError, match="salida"):
        compare_campaigns(tmp_path / "reference", tmp_path / "completion", tmp_path)
