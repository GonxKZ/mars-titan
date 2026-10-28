"""Regiones por correlación comunes a C3 y C4 (#751), con series sintéticas de dos factores."""

import numpy as np
import pytest

from mars_titan.data import correlation_regions as regions

ASSETS = [f"A{i}" for i in range(6)] + [f"B{i}" for i in range(6)]


def two_factor_series(length=60, *, seed=3):
    """Los A cargan en el primer factor y los B en el segundo."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(2, length))
    loading = np.array([0 if name.startswith("A") else 1 for name in ASSETS])
    return factors[loading] + 0.3 * rng.normal(size=(len(ASSETS), length))


def test_correlation_matches_pearson_when_fully_observed_and_ignores_thin_pairs():
    rng = np.random.default_rng(11)
    series = rng.normal(size=(4, 50))
    series = (series - series.mean(1, keepdims=True)) / series.std(1, keepdims=True)
    mask = np.ones_like(series, dtype=bool)
    np.testing.assert_allclose(
        regions.correlation(series, mask, min_observations=10), np.corrcoef(series), atol=1e-12
    )
    mask[3, 5:] = False
    mask[2, 10:] = False
    result = regions.correlation(series, mask, min_observations=10)
    assert result[3, 0] == result[0, 3] == 0.0 and result[3, 3] == 1.0
    # Un par con exactamente el mínimo de observaciones comunes sí aporta su correlación.
    assert result[2, 0] != 0.0
    assert regions.correlation(series, mask, min_observations=11)[2, 0] == 0.0


def test_the_planted_groups_are_recovered_and_numbered_by_their_first_asset():
    result = regions.correlation_regions(
        ASSETS, two_factor_series(), np.ones((12, 60), dtype=bool), clusters=2, min_observations=30
    )
    assert {result[name] for name in ASSETS[:6]} == {"1"}
    assert {result[name] for name in ASSETS[6:]} == {"2"}


def test_unobserved_positions_do_not_count_whatever_they_hold():
    series, mask = two_factor_series(), np.ones((12, 60), dtype=bool)
    mask[:, ::3] = False
    expected = regions.correlation_regions(ASSETS, series, mask, clusters=2, min_observations=30)
    noisy = np.where(mask, series, 1e6)
    assert (
        regions.correlation_regions(ASSETS, noisy, mask, clusters=2, min_observations=30)
        == expected
    )


def test_short_or_constant_series_go_to_region_zero():
    series, mask = two_factor_series(), np.ones((12, 60), dtype=bool)
    mask[0, 31:] = False
    series[11] = 0.25
    result = regions.correlation_regions(ASSETS, series, mask, clusters=2, min_observations=32)
    assert result["A0"] == result["B5"] == regions.NO_HISTORY
    # Con exactamente el mínimo de observaciones, A0 ya tiene historia suficiente.
    result = regions.correlation_regions(ASSETS, series, mask, clusters=2, min_observations=31)
    assert result["A0"] == "1" and result["B5"] == regions.NO_HISTORY


@pytest.mark.parametrize("usable", [0, 1])
def test_without_two_usable_series_there_is_nothing_to_cluster(usable):
    series, mask = two_factor_series(), np.zeros((12, 60), dtype=bool)
    mask[:usable] = True
    result = regions.correlation_regions(ASSETS, series, mask, clusters=2, min_observations=30)
    assert result == {name: ("1" if i < usable else "0") for i, name in enumerate(ASSETS)}


@pytest.mark.parametrize(
    ("assets", "series", "mask"),
    [
        (ASSETS[::-1], np.zeros((12, 4)), np.ones((12, 4), dtype=bool)),
        (ASSETS[:2] * 6, np.zeros((12, 4)), np.ones((12, 4), dtype=bool)),
        (ASSETS, np.zeros((12, 4)), np.ones((12, 5), dtype=bool)),
        (ASSETS[:11], np.zeros((12, 4)), np.ones((12, 4), dtype=bool)),
        (ASSETS, np.zeros(12), np.ones(12, dtype=bool)),
    ],
)
def test_the_assets_must_be_sorted_and_name_one_masked_series_each(assets, series, mask):
    with pytest.raises(ValueError, match="activos ordenados"):
        regions.correlation_regions(assets, series, mask, clusters=2, min_observations=2)


def test_observed_values_must_be_finite():
    series, mask = two_factor_series(), np.ones((12, 60), dtype=bool)
    series[3, 7] = np.nan
    with pytest.raises(ValueError, match="no finitos"):
        regions.correlation_regions(ASSETS, series, mask, clusters=2, min_observations=30)
    mask[3, 7] = False
    regions.correlation_regions(ASSETS, series, mask, clusters=2, min_observations=30)


def test_the_permutation_keeps_labels_and_sizes_and_leaves_region_zero_alone():
    variant = {**dict.fromkeys(ASSETS[:6], "1"), **dict.fromkeys(ASSETS[6:], "2"), "Z": "0"}
    permuted = regions.permuted_regions(variant, [7, 0])
    assert sorted(permuted.values()) == sorted(variant.values()) and permuted != variant
    assert permuted["Z"] == "0" and permuted == regions.permuted_regions(variant, [7, 0])
    assert regions.permuted_regions(variant, [8, 0]) != permuted
