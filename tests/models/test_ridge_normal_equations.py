"""Álgebra de la Gram de Ridge por bloques en CPU, sin ajustar ninguna señal.

Las matrices y objetivos son aleatorios e independientes. Se comprueba la
acumulación de Z'Z y Z'y, su centrado y el sistema normal frente a una referencia
densa en float64, no la capacidad predictiva de un modelo.
"""

import gc
import weakref

import numpy as np
import pytest
import torch

from mars_titan.models.baselines.ridge import centered_normal_equations

# 64 sesiones por 5 precios, noticias, gráficos, 26 conceptos y 140 indicadores por
# triplete, más los cinco bits de presencia de la edición con máscaras.
HISTORICAL_FEATURES = 64 * 5 + 384 + 512 + 3 * 26 + 3 * 140 + 5


def statistics(x, y):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    return mean, scale, float(y.mean())


def dense_reference(x, y, mean, scale, target_mean):
    z = (x - mean) / scale
    z -= z.mean(axis=0)
    centered = y - target_mean
    centered -= centered.mean()
    return z.T @ z, z.T @ centered


def blocks(x, y, size):
    return ((x[i : i + size], y[i : i + size]) for i in range(0, len(x), size))


@pytest.mark.parametrize("size", [1, 7, 256])
@pytest.mark.parametrize("offset", [0.0, 0.5])
def test_blocked_gram_matches_dense_float64_reference(size, offset):
    """Con medias desplazadas, el centrado por sumas es imprescindible."""
    rng = np.random.default_rng(20261009)
    x = rng.normal(loc=3.0, scale=[0.5, 2.0, 10.0, 1e-3, 1.0, 4.0], size=(300, 6))
    x[:, 4] = 7.0
    y = rng.normal(size=300)
    mean, scale, target_mean = statistics(x, y)
    # Desplazamiento de media unidad tipificada en cada columna y en el objetivo.
    mean, target_mean = mean + offset * scale, target_mean - offset
    gram, rhs, sum_x, _ = centered_normal_equations(
        blocks(x, y, size), mean, scale, target_mean, len(x), device="cpu"
    )
    expected_gram, expected_rhs = dense_reference(x, y, mean, scale, target_mean)
    assert gram.dtype == rhs.dtype == sum_x.dtype == torch.float64
    np.testing.assert_allclose(gram.numpy(), expected_gram, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(rhs.numpy(), expected_rhs, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(gram.numpy(), gram.numpy().T, rtol=0, atol=1e-10)
    np.testing.assert_allclose(sum_x.numpy() / len(x), -offset, atol=1e-12)


def test_real_historical_width_keeps_float64_and_solves_the_normal_system():
    """La dimensión real con bits cabe en una Gram float64 de unos 23,6 MB."""
    rng = np.random.default_rng(7)
    rows = 600
    x = rng.normal(size=(rows, HISTORICAL_FEATURES))
    x[:, -5:] = rng.integers(0, 2, size=(rows, 5))
    x[:, -5] = x[:, -3] = 1.0
    y = rng.normal(size=rows)
    mean, scale, target_mean = statistics(x, y)
    gram, rhs, _, _ = centered_normal_equations(
        blocks(x, y, 256), mean, scale, target_mean, rows, device="cpu"
    )
    assert gram.shape == (HISTORICAL_FEATURES, HISTORICAL_FEATURES) == (1719, 1719)
    assert gram.dtype == torch.float64 and gram.element_size() * gram.numel() < 24 * 1024**2
    expected_gram, expected_rhs = dense_reference(x, y, mean, scale, target_mean)
    np.testing.assert_allclose(gram.numpy(), expected_gram, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(rhs.numpy(), expected_rhs, rtol=1e-10, atol=1e-8)
    # Los bits constantes quedan como columnas nulas tras centrar y escalar.
    for column in (-5, -3):
        assert not gram[:, column].any() and not rhs[column]
    alpha = 1.0
    gram.diagonal().add_(alpha)
    solution = torch.linalg.solve(gram, rhs).numpy()
    reference = np.linalg.solve(expected_gram + alpha * np.eye(len(expected_gram)), expected_rhs)
    np.testing.assert_allclose(solution, reference, rtol=1e-8, atol=1e-10)
    residual = gram.numpy() @ solution - rhs.numpy()
    assert np.linalg.norm(residual) <= 1e-9 * np.linalg.norm(rhs.numpy())


def test_accumulation_consumes_blocks_without_retaining_the_corpus():
    alive = []
    seen = []

    def stream():
        rng = np.random.default_rng(3)
        for _ in range(40):
            gc.collect()
            # Solo puede seguir vivo el bloque anterior, retenido por la variable del bucle.
            seen.append(sum(reference() is not None for reference in alive))
            x = rng.normal(size=(64, 5))
            alive.append(weakref.ref(x))
            yield x, rng.normal(size=64)

    mean, scale = np.zeros(5), np.ones(5)
    centered_normal_equations(stream(), mean, scale, 0.0, 40 * 64, device="cpu")
    assert len(seen) == 40 and max(seen) <= 1


def test_digest_and_dimensions_protect_the_second_pass():
    rng = np.random.default_rng(11)
    x, y = rng.normal(size=(20, 3)), rng.normal(size=20)
    mean, scale, target_mean = statistics(x, y)
    *_, first = centered_normal_equations(
        blocks(x, y, 5), mean, scale, target_mean, 20, device="cpu"
    )
    changed = x.copy()
    changed[3, 1] += 1e-12
    *_, second = centered_normal_equations(
        blocks(changed, y, 5), mean, scale, target_mean, 20, device="cpu"
    )
    assert first != second
    with pytest.raises(ValueError, match="dimensiones"):
        centered_normal_equations(
            blocks(x[:, :2], y, 5), mean, scale, target_mean, 20, device="cpu"
        )
