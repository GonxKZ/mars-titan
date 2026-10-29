"""Medidas de la ventana pasada y pertenencias fraccionarias de H4 (#783).

Las ventanas son sintéticas y solo fijan formas y casos de respuesta conocida. No hay
modelos, optimizadores ni ajustes.
"""

import math

import numpy as np
import pytest
import torch
from scipy import stats

from mars_titan.data import asset_groups as ag

SESSIONS = 64


def window(close, *, half_range=0.004, present=None):
    """Ventana [filas, 64, 6] con el cierre dado y un rango simétrico alrededor."""
    close = torch.as_tensor(close, dtype=torch.float64)
    rows = len(close)
    present = torch.ones(rows, SESSIONS, dtype=torch.bool) if present is None else present
    half = torch.as_tensor(half_range, dtype=torch.float64).expand(rows, SESSIONS)
    opened = torch.cat([close[:, :1], close[:, :-1]], dim=1)
    values = torch.stack(
        [opened, close + half, close - half, close, torch.zeros_like(close)], dim=-1
    )
    values = torch.where(present[..., None], values, 0.0)
    return torch.cat([values, present[..., None].to(torch.float64)], dim=-1)


def sectors(rows=48, groups=4, seed=3):
    """Rendimientos con un factor de mercado, uno por sector y ruido propio."""
    generator = torch.Generator().manual_seed(seed)
    market = torch.randn(SESSIONS, generator=generator, dtype=torch.float64) * 0.01
    factors = torch.randn(groups, SESSIONS, generator=generator, dtype=torch.float64) * 0.01
    labels = torch.arange(rows) % groups
    betas = 0.5 + torch.rand(rows, generator=generator, dtype=torch.float64)
    noise = torch.randn(rows, SESSIONS, generator=generator, dtype=torch.float64) * 0.004
    returns = betas[:, None] * market + factors[labels] + noise
    returns[:, 0] = 0
    close = returns.cumsum(1)
    half = 0.001 + torch.rand(rows, SESSIONS, generator=generator, dtype=torch.float64) * 0.01
    return window(close - close[:, :1], half_range=half), labels


def direct_spread(close, high, low, present):
    """Diferencial CHL corregido por dos días con un bucle sobre los pares de sesiones."""
    terms = []
    for t in range(len(close) - 1):
        if not (present[t] and present[t + 1]) or high[t] <= low[t] or high[t + 1] <= low[t + 1]:
            continue
        eta, following = (high[t] + low[t]) / 2, (high[t + 1] + low[t + 1]) / 2
        terms.append(math.sqrt(max(4 * (close[t] - eta) * (close[t] - following), 0.0)))
    return sum(terms) / len(terms), len(terms)


def test_each_measure_matches_a_direct_calculation():
    prices, _ = sectors(rows=6)
    # Una sesión ausente en todo el mercado y un día sin rango en la fila 2.
    prices[:, 20] = 0
    prices[2, 30, 1] = prices[2, 30, 2] = prices[2, 30, 3]
    measures = ag.window_measures(prices, 36)
    data = prices.numpy()
    present = data[..., 5] > 0
    valid = present[:, 1:] & present[:, :-1]
    returns = np.diff(data[..., 3], axis=1)
    for row in range(len(data)):
        step = returns[row, valid[row]]
        assert len(step) == 61
        assert measures["volatility"][row].item() == pytest.approx(step.std(ddof=1), rel=1e-12)
        assert measures["momentum"][row].item() == pytest.approx(
            data[row, 63, 3] - data[row, 0, 3], rel=1e-12
        )
        spread, pairs = direct_spread(
            data[row, :, 3], data[row, :, 1], data[row, :, 2], present[row]
        )
        assert measures["liquidity_spread"][row].item() == pytest.approx(spread, rel=1e-12)
        assert pairs == (59 if row == 2 else 61)
    # Todas las filas comparten las sesiones válidas, así que el mercado es su media.
    assert (valid == valid[0]).all()
    market = returns[:, valid[0]].mean(axis=0)
    for row in range(len(data)):
        slope = np.polyfit(market, returns[row, valid[0]], 1)[0]
        assert measures["beta"][row].item() == pytest.approx(slope, rel=1e-10)
    assert all(bool(mask.all()) for mask in measures["valid"].values())


def test_the_spread_sets_each_negative_two_day_estimate_to_zero_before_the_square_root():
    close = torch.zeros(1, SESSIONS, dtype=torch.float64)
    # Cierres en el máximo dos días seguidos dan 4 (c − η)(c − η') > 0. Un cierre en el máximo
    # seguido de un rango por encima da un término negativo, que cuenta como cero.
    high = torch.full((1, SESSIONS), 0.01, dtype=torch.float64)
    close[0, ::2], close[0, 1::2] = 0.01, 0.03
    high[0, 1::2] = 0.03
    low = high - 0.02
    prices = window(close)
    prices[..., 1], prices[..., 2] = high, low
    expected, pairs = direct_spread(
        close[0], high[0], low[0], torch.ones(SESSIONS, dtype=torch.bool)
    )
    measures = ag.window_measures(prices, 36)
    assert measures["liquidity_spread"].item() == pytest.approx(expected, rel=1e-12)
    terms = (
        4
        * (close[0, :-1] - (high[0, :-1] + low[0, :-1]) / 2)
        * (close[0, :-1] - (high[0, 1:] + low[0, 1:]) / 2)
    )
    assert (terms < 0).any() and (terms > 0).any() and pairs == 63
    # La versión mensual, sqrt(max(media, 0)), daría otra cifra.
    assert expected != pytest.approx(math.sqrt(max(terms.mean().item(), 0.0)))


def test_rows_with_too_few_observations_are_not_valid():
    prices, _ = sectors(rows=5)
    prices[0, :30] = 0
    prices[1, 10:50, 1] = prices[1, 10:50, 2] = prices[1, 10:50, 3]
    measures = ag.window_measures(prices, 36)
    valid = measures["valid"]
    assert valid["volatility"].tolist() == [False, True, True, True, True]
    assert valid["momentum"].tolist() == valid["volatility"].tolist()
    assert valid["liquidity_spread"].tolist() == [False, False, True, True, True]
    assert valid["beta"].tolist() == valid["correlation"].tolist() == [False] + [True] * 4
    assert torch.count_nonzero(measures["residual"][0]) == 0
    # La fila sin historia no entra en el mercado de las demás.
    reduced = ag.window_measures(prices[1:], 36)
    torch.testing.assert_close(reduced["beta"], measures["beta"][1:], rtol=1e-12, atol=0)


def test_a_session_with_one_valid_row_has_no_beta_or_correlation():
    prices, _ = sectors(rows=3)
    prices[1:, :40] = 0
    measures = ag.window_measures(prices, 36)
    # Con una sola fila, el mercado es la propia fila y el residuo es nulo.
    assert measures["valid"]["beta"].tolist() == [True, False, False]
    assert measures["valid"]["correlation"].tolist() == [False, False, False]
    assert measures["beta"][0].item() == pytest.approx(1.0, rel=1e-12)


def test_midranks_match_the_average_ranks_and_ignore_invalid_rows():
    values = torch.tensor([0.3, 0.1, 0.3, 0.7, 0.2, 9.0], dtype=torch.float64)
    valid = torch.tensor([True, True, True, True, True, False])
    ranks = ag.midranks(values, valid)
    expected = (stats.rankdata(values[:5].numpy(), method="average") - 0.5) / 5
    np.testing.assert_allclose(ranks[:5].numpy(), expected, rtol=0, atol=1e-15)
    assert ranks[5] == 0
    assert (
        ag.midranks(torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=bool)).tolist()
        == [0.5] * 4
    )


@pytest.mark.parametrize("buckets", [2, 5, 10])
def test_fuzzy_buckets_are_a_ruspini_partition_between_neighbours(buckets):
    ranks = torch.linspace(0.001, 0.999, 101, dtype=torch.float64)
    valid = torch.ones(101, dtype=torch.bool)
    valid[7] = False
    membership = ag.fuzzy_buckets(ranks, valid, buckets)
    totals = membership.sum(1)
    torch.testing.assert_close(totals[valid], torch.ones(100, dtype=torch.float64))
    assert torch.count_nonzero(membership[7]) == 0
    assert ((membership > 0).sum(1)[valid] <= 2).all()
    for row in torch.nonzero(valid).squeeze(1).tolist():
        position = min(max(ranks[row].item() * buckets - 0.5, 0.0), buckets - 1)
        lower = math.floor(position)
        expected = torch.zeros(buckets, dtype=torch.float64)
        expected[lower] += 1 - (position - lower)
        expected[min(lower + 1, buckets - 1)] += position - lower
        torch.testing.assert_close(membership[row], expected, rtol=0, atol=1e-15)


def test_a_rank_on_a_centre_belongs_to_one_bucket_and_the_edges_saturate():
    ranks = torch.tensor([0.1, 0.3, 0.02, 0.98, 0.2], dtype=torch.float64)
    membership = ag.fuzzy_buckets(ranks, torch.ones(5, dtype=torch.bool), 5)
    assert membership[0].tolist() == [1, 0, 0, 0, 0]
    assert membership[1].tolist() == [0, 1, 0, 0, 0]
    assert membership[2].tolist() == [1, 0, 0, 0, 0]
    assert membership[3].tolist() == [0, 0, 0, 0, 1]
    torch.testing.assert_close(
        membership[4], torch.tensor([0.5, 0.5, 0, 0, 0], dtype=torch.float64)
    )


def test_farthest_first_starts_far_from_the_mean_and_skips_invalid_rows():
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [-2.0, 0.0], [9.0, 9.0]], dtype=torch.float64
    )
    valid = torch.tensor([True, True, True, True, False])
    centers = ag.farthest_first(points, valid, 3)
    # La media válida es (−0,25, 0,75): el más lejano es (0, 3), luego (−2, 0) y luego (1, 0).
    assert centers.tolist() == [[0.0, 3.0], [-2.0, 0.0], [1.0, 0.0]]


def test_fuzzy_clusters_recover_separated_sectors():
    prices, labels = sectors()
    measures = ag.window_measures(prices, 36)
    points, kept = ag.correlation_embedding(
        measures["residual"], measures["valid"]["correlation"], 4
    )
    assert kept.all()
    torch.testing.assert_close(points.norm(dim=1), torch.ones(48, dtype=torch.float64))
    membership = ag.fuzzy_clusters(points, kept, 4, 32)
    torch.testing.assert_close(membership.sum(1), torch.ones(48, dtype=torch.float64))
    hard = membership.argmax(1)
    # Cada sector sintético cae entero en un grupo distinto y con pertenencia alta.
    pairs = {(label, group) for label, group in zip(labels.tolist(), hard.tolist(), strict=True)}
    assert len(pairs) == 4 and len({group for _, group in pairs}) == 4
    assert membership.max(1).values.mean() > 0.9


def test_the_embedding_does_not_depend_on_the_sign_of_each_component(monkeypatch):
    prices, _ = sectors()
    measures = ag.window_measures(prices, 36)
    residual, valid = measures["residual"], measures["valid"]["correlation"]
    points, kept = ag.correlation_embedding(residual, valid, 4)
    expected = ag.fuzzy_clusters(points, kept, 4, 32)
    original = torch.linalg.eigh

    def flipped(matrix):
        values, vectors = original(matrix)
        signs = torch.tensor([(-1.0) ** index for index in range(len(values))], dtype=vectors.dtype)
        return values, vectors * signs

    monkeypatch.setattr(torch.linalg, "eigh", flipped)
    points, kept = ag.correlation_embedding(residual, valid, 4)
    torch.testing.assert_close(ag.fuzzy_clusters(points, kept, 4, 32), expected)


def test_the_embedding_distance_is_the_mantegna_distance_of_the_reconstructed_correlation():
    prices, _ = sectors(rows=24)
    measures = ag.window_measures(prices, 36)
    residual, valid = measures["residual"], measures["valid"]["correlation"]
    assert valid.all()
    data = residual.numpy()
    # Con las mismas sesiones presentes, z_i · z_j es la correlación de Pearson de los residuos.
    np.testing.assert_allclose(data @ data.T, np.corrcoef(data), rtol=0, atol=1e-12)
    points, kept = ag.correlation_embedding(residual, valid, 4)
    assert kept.all()
    values, vectors = np.linalg.eigh(data @ data.T)
    reconstructed = (vectors[:, -4:] * values[-4:]) @ vectors[:, -4:].T
    scale = np.sqrt(np.diag(reconstructed))
    correlation = reconstructed / np.outer(scale, scale)
    embedded = points.numpy()
    distances = ((embedded[:, None, :] - embedded[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(distances, 2 * (1 - correlation), rtol=0, atol=1e-10)


def test_memberships_follow_a_permutation_of_the_rows():
    prices, _ = sectors()
    options = dict(buckets=5, clusters=4, min_observations=36, iterations=32)
    expected = ag.session_memberships(prices, **options)
    order = torch.randperm(48, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(ag.session_memberships(prices[order], **options), expected[order])


def test_session_memberships_sum_to_one_in_each_family_of_a_valid_row():
    prices, _ = sectors(rows=10)
    prices[0, :40] = 0
    membership = ag.session_memberships(
        prices, buckets=5, clusters=4, min_observations=36, iterations=8
    )
    assert membership.shape == (10, 24)
    for part in ag.family_slices(5, 4):
        totals = membership[:, part].sum(1)
        assert totals[0] == 0
        torch.testing.assert_close(totals[1:], torch.ones(9, dtype=torch.float64))


def test_the_fill_of_an_absent_session_does_not_reach_the_memberships():
    prices, _ = sectors(rows=8)
    prices[:, 12] = 0
    options = dict(buckets=5, clusters=4, min_observations=36, iterations=16)
    expected = ag.session_memberships(prices, **options)
    for fill in (3.75, -1e3):
        changed = prices.clone()
        changed[:, 12, :5] = fill
        assert torch.equal(ag.session_memberships(changed, **options), expected)


def test_fewer_valid_rows_than_clusters_put_each_row_in_its_own_group():
    prices, _ = sectors(rows=3, groups=3)
    measures = ag.window_measures(prices, 36)
    points, kept = ag.correlation_embedding(
        measures["residual"], measures["valid"]["correlation"], 4
    )
    membership = ag.fuzzy_clusters(points, kept, 4, 16)
    assert membership.max(1).values.min() > 0.49
    torch.testing.assert_close(membership.sum(1), torch.ones(3, dtype=torch.float64))


@pytest.mark.parametrize(
    "prices",
    [
        torch.zeros(2, 64, 5, dtype=torch.float64),
        torch.zeros(2, 32, 6, dtype=torch.float64),
        torch.zeros(0, 64, 6, dtype=torch.float64),
        torch.zeros(2, 64, 6, dtype=torch.bfloat16),
        np.zeros((2, 64, 6)),
    ],
)
def test_the_measures_refuse_other_windows(prices):
    with pytest.raises(ValueError, match="bit de presencia"):
        ag.window_measures(prices, 36)


def test_the_family_slices_cover_the_columns_in_order():
    slices = ag.family_slices(5, 8)
    assert [(part.start, part.stop) for part in slices] == [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 28),
    ]
    assert ag.FAMILIES[-1] == "correlation" and len(slices) == len(ag.FAMILIES)
