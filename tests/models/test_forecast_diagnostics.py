"""Errores de pronóstico y comparación pareada con resultados calculados a mano."""

import importlib
import json
import math

import numpy as np
import pytest


def module():
    try:
        return importlib.import_module("mars_titan.models.baselines.diagnostics")
    except ModuleNotFoundError:
        pytest.fail("Falta el diagnóstico de errores de pronóstico")


def test_point_errors_use_prediction_minus_target_and_linear_quantiles():
    result = module().point_diagnostics([1, 2, 3, 4], [1, 4, 2, 8])

    assert result["n"] == 4
    assert result["mae"] == pytest.approx(1.75)
    assert result["mse"] == pytest.approx(5.25)
    assert result["rmse"] == pytest.approx(math.sqrt(5.25))
    assert result["bias"] == pytest.approx(1.25)
    assert result["median_absolute_error"] == pytest.approx(1.5)
    assert result["q90_absolute_error"] == pytest.approx(3.4)
    assert result["q95_absolute_error"] == pytest.approx(3.7)
    assert result["q99_absolute_error"] == pytest.approx(3.94)
    assert result["max_absolute_error"] == pytest.approx(4)
    assert result["quantile_method"] == "linear"
    assert result["r2"] == pytest.approx(-3.2)
    assert result["r2_reason"] is None
    json.dumps(result, allow_nan=False)


def test_correlations_use_average_ranks_for_ties():
    result = module().point_diagnostics([1, 1, 3, 4], [4, 2, 2, 1])

    assert result["pearson"] == pytest.approx(-4.25 / math.sqrt(32.0625))
    assert result["spearman"] == pytest.approx(-5 / 6)
    assert result["pearson_reason"] is None
    assert result["spearman_reason"] is None


@pytest.mark.parametrize(
    ("target", "prediction", "reason"),
    [([1], [2], "observaciones"), ([1, 1], [0, 2], "objetivo"), ([1, 2], [0, 0], "predicción")],
)
def test_undefined_correlations_include_a_reason(target, prediction, reason):
    result = module().point_diagnostics(target, prediction)

    for metric in ("pearson", "spearman"):
        assert result[metric] is None
        assert reason in result[f"{metric}_reason"]
    json.dumps(result, allow_nan=False)


def test_constant_target_has_no_r2_even_for_a_perfect_prediction():
    result = module().point_diagnostics([0, 0], [0, 0])

    assert result["r2"] is None
    assert "constante" in result["r2_reason"]
    assert result["mae"] == result["mse"] == result["rmse"] == 0


def test_direction_reports_abstentions_and_its_denominators():
    result = module().point_diagnostics([-1, 1, 2, -2, 0, 0], [-1, -1, 0, 1, 0, 3])

    assert result["direction"] == {
        "accuracy": pytest.approx(1 / 3),
        "accuracy_reason": None,
        "evaluated_n": 3,
        "correct_n": 1,
        "nonzero_target_n": 4,
        "target_zero_n": 2,
        "prediction_zero_n": 2,
        "abstained_nonzero_target_n": 1,
        "coverage": pytest.approx(0.75),
        "coverage_reason": None,
    }


@pytest.mark.parametrize("target", [[0, 0], [-1, 1]])
def test_direction_does_not_turn_a_zero_forecast_into_a_positive_sign(target):
    direction = module().point_diagnostics(target, [0, 0])["direction"]

    assert direction["accuracy"] is None
    assert direction["accuracy_reason"]
    assert direction["evaluated_n"] == direction["correct_n"] == 0
    if target == [0, 0]:
        assert direction["coverage"] is None
        assert direction["coverage_reason"]
    else:
        assert direction["coverage"] == 0
        assert direction["coverage_reason"] is None


def test_zero_reference_skill_and_paired_differences_have_the_expected_sign():
    reference = module().point_diagnostics([-2, 1, 3], [-1, 0, 2])["reference"]

    assert reference == {
        "kind": "zero",
        "mae": pytest.approx(2),
        "mse": pytest.approx(14 / 3),
        "mae_skill": pytest.approx(0.5),
        "mae_skill_reason": None,
        "mse_skill": pytest.approx(11 / 14),
        "mse_skill_reason": None,
        "paired_mae_difference": pytest.approx(-1),
        "paired_mse_difference": pytest.approx(-11 / 3),
    }


def test_explicit_reference_changes_the_comparison_without_changing_model_errors():
    result = module().point_diagnostics([1, 3], [2, 2], reference=[1, 2])

    assert result["mae"] == result["mse"] == 1
    reference = result["reference"]
    assert reference["kind"] == "provided"
    assert reference["mae"] == reference["mse"] == 0.5
    assert reference["mae_skill"] == reference["mse_skill"] == -1
    assert reference["paired_mae_difference"] == reference["paired_mse_difference"] == 0.5


def test_perfect_reference_has_undefined_relative_skill():
    reference = module().point_diagnostics([0, 0], [1, -1])["reference"]

    assert reference["mae_skill"] is reference["mse_skill"] is None
    assert reference["mae_skill_reason"]
    assert reference["mse_skill_reason"]
    assert reference["paired_mae_difference"] == reference["paired_mse_difference"] == 1


@pytest.mark.parametrize(
    ("target", "prediction", "reference"),
    [
        ([], [], None),
        ([1], [1, 2], None),
        ([1, 2], [1, 2], [0]),
        ([[1]], [1], None),
        ([1], [[1]], None),
        ([1], [1], [[1]]),
        (1, [1], None),
        ([np.nan], [1], None),
        ([1], [np.inf], None),
        ([1], [1], [-np.inf]),
        ([1j], [1], None),
        ([1], ["error"], None),
    ],
)
def test_invalid_vectors_fail_explicitly(target, prediction, reference):
    with pytest.raises(ValueError, match="vector|longitud|finito|reales"):
        module().point_diagnostics(target, prediction, reference)


def test_unrepresentable_squared_errors_fail_without_emitting_infinite_metrics():
    with pytest.raises(ValueError, match="rango numérico"):
        module().point_diagnostics([1e200, -1e200], [0, 0])


def test_unrepresentable_relative_skill_fails_without_emitting_infinity():
    with pytest.raises(ValueError, match="rango numérico"):
        module().point_diagnostics([1e-320, 1e-320], [1, 1])


def test_date_bootstrap_preserves_whole_dates_and_row_weighting():
    # La semilla 0 selecciona bloques [1, 1] y [1, 0]. Sus medias son 8/3 y 3/2.
    result = module().paired_date_bootstrap(
        ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-01"],
        [0, 0, 0, 0],
        [4, 0, 2, 0],
        block_length=2,
        repetitions=2,
        seed=0,
    )

    assert result["estimate"] == pytest.approx(1.5)
    assert result["interval"] == {
        "lower": pytest.approx(367 / 240),
        "upper": pytest.approx(211 / 80),
    }
    assert result["n"] == 4
    assert result["n_dates"] == 3
    assert result["block_length"] == 2
    assert result["repetitions"] == result["completed_repetitions"] == 2
    assert result["seed"] == 0
    assert result["confidence_level"] == 0.95
    assert result["metric"] == "paired_mae_difference"
    assert result["method"] == "moving_date_blocks_percentile"
    assert result["quantile_method"] == "linear"
    assert result["reference_kind"] == "zero"
    assert result["interval_reason"] is None
    assert "exploratorio" in result["warning"]
    json.dumps(result, allow_nan=False)


def test_date_bootstrap_uses_the_supplied_reference_for_every_paired_row():
    result = module().paired_date_bootstrap(
        ["2024-01-01", "2024-01-02", "2024-01-03"],
        [1, -2, 3],
        [2, -1, 4],
        reference=[3, 0, 5],
        block_length=1,
        repetitions=20,
    )

    assert result["estimate"] == -1
    assert result["interval"] == {"lower": -1, "upper": -1}
    assert result["reference_kind"] == "provided"


def test_date_bootstrap_is_reproducible_and_independent_of_row_order():
    dates = np.array(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-01"])
    target, prediction = np.zeros(4), np.array([4, 0, 2, 0])
    options = {"block_length": 2, "repetitions": 20, "seed": 42}
    expected = module().paired_date_bootstrap(dates, target, prediction, **options)
    actual = module().paired_date_bootstrap(dates[::-1], target[::-1], prediction[::-1], **options)

    assert actual == expected
    assert module().paired_date_bootstrap(dates, target, prediction, **options) == expected


@pytest.mark.parametrize("block_length", [2, 5])
def test_too_few_dates_for_a_block_report_no_interval(block_length):
    result = module().paired_date_bootstrap(
        ["2024-01-01", "2024-01-02"],
        [0, 0],
        [1, 3],
        block_length=block_length,
    )

    assert result["estimate"] == 2
    assert result["interval"] is None
    assert "fechas" in result["interval_reason"]
    assert result["completed_repetitions"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "dates",
    [
        [],
        ["2024-01-01"],
        [["2024-01-01", "2024-01-02"]],
        ["2024-01-01", "fecha incorrecta"],
        ["2024-01-01", "NaT"],
        ["2024-01-01", None],
        [1, 2],
    ],
)
def test_date_bootstrap_rejects_missing_or_invalid_dates(dates):
    with pytest.raises(ValueError, match="fechas|dates"):
        module().paired_date_bootstrap(dates, [0, 0], [1, 1])


@pytest.mark.parametrize(
    "options",
    [
        {"block_length": 0},
        {"block_length": 1.5},
        {"block_length": True},
        {"repetitions": 1},
        {"repetitions": 10001},
        {"repetitions": 2.5},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_date_bootstrap_rejects_invalid_execution_limits(options):
    with pytest.raises(ValueError, match="block_length|repetitions|seed"):
        module().paired_date_bootstrap(["2024-01-01", "2024-01-02"], [0, 0], [1, 1], **options)


def test_date_bootstrap_bounds_total_resampling_work():
    dates = np.arange("2020-01-01", "2023-01-01", dtype="datetime64[D]")

    with pytest.raises(ValueError, match="límite"):
        module().paired_date_bootstrap(
            dates, np.zeros(dates.size), np.ones(dates.size), repetitions=10000
        )
