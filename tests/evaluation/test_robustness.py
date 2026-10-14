"""Análisis de robustez ante perturbaciones de las entradas (#37), sin modelos ni datos reales."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.data import input_perturbations as perturbations
from mars_titan.data.input_policy import MODALITIES
from mars_titan.evaluation import robustness

DAY = 86_400_000_000
ORIGIN = 946_684_800_000_000


def brute_since(asset, days, length, share, horizon):
    """Días desde la última ráfaga recorriendo cada día natural desde 2000."""
    calendar = np.arange(0, int(days.max()) + 1)
    flags = perturbations.burst_rows(asset, ORIGIN + calendar * DAY, length, share)
    result = np.full(len(days), np.nan)
    for index, day in enumerate(days):
        if flags[day]:
            continue
        before = np.flatnonzero(flags[:day])
        if len(before) and day - (before[-1] + 1) < horizon:
            result[index] = day - (before[-1] + 1)
    return result


@pytest.mark.parametrize("variant", robustness.BURSTS)
@pytest.mark.parametrize("asset", ["US/A0000", "CN/600000"])
def test_days_since_the_last_burst_match_a_day_by_day_walk(variant, asset):
    spec = perturbations.PERTURBATIONS[variant]
    rng = np.random.default_rng(7)
    days = np.sort(rng.choice(np.arange(0, 2200), 400, replace=False))
    # La hora del día no cambia el día ni el bloque.
    moments = ORIGIN + days * DAY + rng.integers(0, DAY, len(days))
    found = perturbations.days_since_burst(asset, moments, spec["length_days"], spec["share"], 252)
    expected = brute_since(asset, days, spec["length_days"], spec["share"], 252)
    np.testing.assert_array_equal(found, expected)
    inside = perturbations.burst_rows(asset, moments, spec["length_days"], spec["share"])
    assert inside.any() and np.isnan(found[inside]).all()
    assert np.isfinite(found).any()


def test_days_since_a_burst_reject_samples_before_2000_and_a_bad_horizon():
    with pytest.raises(ValueError, match="anterior a 2000"):
        perturbations.days_since_burst("US/A", np.array([ORIGIN - DAY]), 21, 0.1, 10)
    with pytest.raises(ValueError, match="horizonte"):
        perturbations.days_since_burst("US/A", np.array([ORIGIN]), 21, 0.1, 0)
    assert perturbations.days_since_burst("US/A", np.zeros(0, np.int64), 21, 0.1, 5).shape == (0,)


def test_the_burst_context_is_computed_asset_by_asset_in_panel_order():
    assets = np.array(["US/B", "US/A", "US/B", "US/A"] * 50)
    moments = ORIGIN + np.repeat(np.arange(100) * 7, 2) * DAY
    context = robustness.burst_context(assets, moments, 252)
    assert set(context) == set(robustness.BURSTS)
    for variant in robustness.BURSTS:
        spec = perturbations.PERTURBATIONS[variant]
        inside, since = context[variant]
        for asset in ("US/A", "US/B"):
            rows = assets == asset
            arguments = (asset, moments[rows], spec["length_days"], spec["share"])
            np.testing.assert_array_equal(inside[rows], perturbations.burst_rows(*arguments))
            np.testing.assert_array_equal(
                since[rows], perturbations.days_since_burst(*arguments, 252)
            )
    with pytest.raises(ValueError, match="mercado/símbolo"):
        robustness.burst_context(np.array(["A"]), moments[:1], 252)


def test_affected_rows_need_a_present_modality_and_a_burst():
    bits = np.zeros((4, len(MODALITIES)), dtype=bool)
    bits[0, MODALITIES.index("news")] = True
    bits[1, MODALITIES.index("fundamentals")] = True
    bits[3, MODALITIES.index("news")] = True
    bits[:, MODALITIES.index("macro")] = True
    variant = robustness.BURSTS[0]
    bursts = {variant: (np.array([True, True, True, False]), np.zeros(4))}
    np.testing.assert_array_equal(
        robustness.affected_rows(bits, variant, bursts), [True, True, False, False]
    )
    # El ruido toca el macro, presente en todas las filas.
    assert robustness.affected_rows(bits, "noise_10pct", bursts).all()


def panel(prediction, target):
    return SimpleNamespace(prediction=np.asarray(prediction), target=np.asarray(target))


def test_drift_sums_unchanged_rows_by_days_since_the_last_burst():
    original = panel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0] * 6)
    perturbed = panel([0.5, 0.2, 0.0, 0.1, -0.3, 0.4], [1.0] * 6)
    rows = np.array([True, False, False, False, False, False])
    since = np.array([np.nan, 0.0, 4.0, 5.0, np.nan, 300.0])
    inside = np.zeros(6, dtype=bool)
    found = robustness.drift(original, perturbed, rows, inside, since, [5, 21])
    assert list(found) == ["0-5", "5-21", "inside", "far"]
    assert found["inside"]["rows"] == 0
    # |1 − 0,2| − |1 − 0| en la fila de 0 días, que es la única que cambia.
    assert found["0-5"] == dict(
        rows=2, changed=1, sum_abs_difference=0.2, sum_error_difference=pytest.approx(-0.2)
    )
    # 300 días queda fuera de los tramos y no se cuenta en ninguno.
    assert found["5-21"]["rows"] == 1 and found["5-21"]["changed"] == 1
    assert found["far"]["rows"] == 1 and found["far"]["sum_abs_difference"] == pytest.approx(0.3)
    assert found["far"]["sum_error_difference"] == pytest.approx(0.3)
    # Una fila sin modalidad dentro de una ráfaga no cambia su entrada, pero tampoco está
    # lejos de la ráfaga: va a su propio tramo y sale de `far`.
    inside[4] = True
    found = robustness.drift(original, perturbed, rows, inside, since, [5, 21])
    assert found["inside"] == dict(
        rows=1, changed=1, sum_abs_difference=0.3, sum_error_difference=pytest.approx(0.3)
    )
    assert found["far"]["rows"] == 0


def test_drift_summary_adds_windows_before_dividing():
    parts = [
        {"0-5": dict(rows=2, changed=1, sum_abs_difference=0.4, sum_error_difference=0.2)},
        {"0-5": dict(rows=0, changed=0, sum_abs_difference=0.0, sum_error_difference=0.0)},
        {"0-5": dict(rows=2, changed=2, sum_abs_difference=0.4, sum_error_difference=-0.6)},
    ]
    assert robustness._drift_summary(parts) == {
        "0-5": dict(
            rows=4, changed=3, mean_abs_difference=0.2, mean_error_difference=pytest.approx(-0.1)
        )
    }
    empty = robustness._drift_summary([parts[1]])
    assert empty["0-5"]["mean_abs_difference"] is None


def test_the_declared_analysis_is_valid_and_its_path_stays_out_of_the_identity():
    declared = robustness.declaration()
    assert declared["recovery_bins_days"] == [5, 21, 63, 252]
    assert declared["metric"] == robustness.METRIC and "path" not in declared
    assert len(declared["sha256"]) == 64


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(use="model_selection"), "secundario y descriptivo"),
        (dict(recovery_bins_days=[21, 5]), "crecientes"),
        (dict(recovery_bins_days=[]), "crecientes"),
        (dict(min_rows=0), "min_rows"),
        (dict(extra=True), "exactamente"),
    ],
)
def test_a_changed_analysis_is_rejected(tmp_path, change, message):
    document = json.loads(robustness.DECLARATION.read_text())
    document["analysis"].update(change)
    path = tmp_path / "robustness.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        robustness.declaration(path)
