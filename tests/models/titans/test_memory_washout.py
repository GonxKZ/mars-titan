"""La duración de un calentamiento no se deduce de la semivida de una puerta."""

import json
import sys
from fractions import Fraction as F

import pytest

from reports.research import check_memory_washout as check


@pytest.mark.parametrize(
    ("rho", "tolerance", "prefactor", "steps"),
    [
        (F(1, 2), F(1, 8), F(1), 3),
        (F(1, 2), F(1, 8), F(2), 4),
        (F(1, 2), F(2), F(1), 0),
        (F(0), F(1, 8), F(1), 1),
    ],
)
def test_horizon_is_the_first_integer_that_meets_the_bound(rho, tolerance, prefactor, steps):
    assert check.contraction_steps(rho, tolerance, prefactor=prefactor) == steps


@pytest.mark.parametrize("rho", [F(1), F(-1), F(3, 2)])
def test_a_noncontractive_factor_is_rejected(rho):
    with pytest.raises(ValueError):
        check.contraction_steps(rho, F(1, 100))


def test_the_exact_search_has_a_limit_and_refuses_nonpositive_bounds():
    with pytest.raises(ValueError, match="límite"):
        check.contraction_steps(F(1, 2), F(1, 100), max_steps=2)
    for tolerance, prefactor in [(F(0), F(1)), (F(1), F(0))]:
        with pytest.raises(ValueError):
            check.contraction_steps(F(1, 2), tolerance, prefactor=prefactor)


@pytest.mark.parametrize("limit", [0, True, -1, 100_001, 2.5])
def test_the_search_budget_cannot_be_unbounded_or_fractional(limit):
    with pytest.raises(ValueError, match="límite"):
        check.contraction_steps(F(1, 2), F(1, 10), max_steps=limit)


def test_floating_point_inputs_cannot_claim_exact_arithmetic():
    with pytest.raises(ValueError, match="fracciones"):
        check.contraction_steps(0.5, F(1, 10))


def test_pt1_horizons_are_exact_and_do_not_claim_to_cover_the_complete_predictor():
    result = check.check_washout()
    assert result["certificate_valid"] and result["euclidean_prefactor_valid"]
    rho, target = F(result["rho"]), F(result["relative_tolerance"])
    for name, prefactor in [("p_norm", F(1)), ("euclidean", F(result["euclidean_prefactor"]))]:
        steps = result[name]["steps"]
        assert prefactor * rho**steps <= target < prefactor * rho ** (steps - 1)
        assert steps > 252
    assert result["full_mac_covered"] is False
    assert result["two_layer_memory_covered"] is False
    assert result["protocol_changed"] is False


def test_feedback_can_preserve_two_fixed_points_inside_the_gate_box():
    result = check.check_feedback_counterexample()
    assert result["gates_inside_pt1_box"]
    assert result["fixed_points_verified_exactly"]
    assert result["fixed_points"] == [["1/9", "1/18"], ["-1/9", "-1/18"]]
    assert F(result["local_jacobian_frobenius_squared"]) == F(27, 100) < 1
    assert result["global_incremental_contraction"] is False


def test_the_command_records_the_bound_and_refuses_to_replace_a_receipt(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    monkeypatch.setattr(sys, "argv", ["check_memory_washout", "--output", str(path)])
    check.main()
    saved = path.read_bytes()
    result = json.loads(saved)
    assert result["p_norm"]["steps"] == 15349
    assert result["euclidean"]["steps"] == 19173
    assert result["feedback_counterexample"]["fixed_points_verified_exactly"]
    with pytest.raises(FileExistsError):
        check.main()
    assert path.read_bytes() == saved
