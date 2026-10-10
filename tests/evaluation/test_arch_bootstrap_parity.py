"""Paridad índice a índice del bootstrap circular propio con el de ``arch``.

``paired_comparisons`` sortea los bloques de días del contraste principal y ``arch`` los
del SPA, el StepM y el MCS de ``predictive_ability``. Con el mismo ``numpy.random.Generator``
los dos deben remuestrear exactamente los mismos días, de modo que los análisis
secundarios ven las mismas réplicas que el contraste principal. Solo se generan índices,
no se ajusta ningún modelo.
"""

import numpy as np
import pytest
from arch.bootstrap import MCS, CircularBlockBootstrap

from mars_titan.evaluation.paired_comparisons import (
    circular_block_counts,
    circular_block_indices,
)

# Días, longitud de bloque y réplicas: bloques que dividen o no a los días, un bloque que
# cubre toda la serie, bloques de un día y muchas réplicas cortas.
SHAPES = [(250, 5, 64), (251, 7, 33), (1000, 20, 16), (37, 37, 4), (10, 3, 1000), (17, 1, 50)]


def arch_indices(periods, block_length, replicates, seed):
    bootstrap = CircularBlockBootstrap(
        block_length, np.arange(periods), seed=np.random.default_rng(seed)
    )
    return np.stack([bootstrap.update_indices() for _ in range(replicates)])


@pytest.mark.parametrize(("periods", "block_length", "replicates"), SHAPES)
def test_circular_block_indices_match_arch_exactly(periods, block_length, replicates):
    ours = circular_block_indices(
        np.random.default_rng(20261009), replicates, periods, block_length
    )
    theirs = arch_indices(periods, block_length, replicates, 20261009)
    assert ours.dtype == theirs.dtype == np.int64
    assert ours.shape == theirs.shape == (replicates, periods)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize(("periods", "block_length", "replicates"), SHAPES)
def test_circular_block_counts_match_the_days_drawn_by_arch(periods, block_length, replicates):
    counts = circular_block_counts(np.random.default_rng(7), replicates, periods, block_length)
    theirs = arch_indices(periods, block_length, replicates, 7)
    expected = np.stack([np.bincount(row, minlength=periods) for row in theirs])
    np.testing.assert_array_equal(counts, expected)
    assert np.all(counts.sum(axis=1) == periods)


def test_drawing_in_chunks_keeps_the_same_stream():
    """El contraste principal sortea por tandas de réplicas y ``arch`` réplica a réplica."""
    rng = np.random.default_rng(11)
    chunks = [circular_block_indices(rng, size, 300, 16) for size in (256, 256, 88)]
    np.testing.assert_array_equal(np.concatenate(chunks), arch_indices(300, 16, 600, 11))


def test_the_mcs_resamples_the_same_days_as_the_primary_bootstrap():
    """El MCS de ``arch`` con la semilla de la comparación ve los bloques del contraste."""
    rng = np.random.default_rng(3)
    losses = rng.gamma(2.0, 1.0, (120, 3)) + np.array([0.0, 0.05, 0.1])
    mcs = MCS(losses, size=0.1, reps=40, block_size=8, method="R", bootstrap="circular", seed=5)
    mcs.compute()
    ours = circular_block_indices(np.random.default_rng(5), 40, 120, 8)
    np.testing.assert_array_equal(np.stack(mcs._bootstrap_indices), ours)


@pytest.mark.parametrize("change", ["start", "length", "wrap"])
def test_the_parity_detects_a_changed_start_length_or_wrap(change):
    """Las variantes con otro inicio, otra longitud o sin recorrido circular difieren."""
    periods, block_length, replicates, seed = 251, 7, 33, 20261009
    rng = np.random.default_rng(seed)
    blocks = -(-periods // block_length)
    starts = rng.integers(0, periods, size=(replicates, blocks))
    if change == "start":
        starts = (starts + 1) % periods
    length = block_length + 1 if change == "length" else block_length
    index = starts[:, :, None] + np.arange(length)
    index = np.minimum(index, periods - 1) if change == "wrap" else index % periods
    changed = index.reshape(replicates, -1)[:, :periods]
    assert not np.array_equal(changed, arch_indices(periods, block_length, replicates, seed))
