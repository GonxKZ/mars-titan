"""Estadísticas Ridge en cuda:0 frente a la ruta anterior, sin resolver ningún alpha.

La referencia es la versión anterior escrita aquí: estandarización en numpy float64 y
copia paginable de cada bloque. La nueva copia float32 fijada por turnos y estandariza en
la GPU. Las dos deben dar la misma Gram, el mismo Z'y y las mismas sumas bit a bit, con
bloques de distinto tamaño que obligan a reutilizar y a ampliar los búferes fijados.
"""

import hashlib

import numpy as np
import pytest
import torch

from mars_titan.models.baselines import ridge

DEVICE = "cuda:0"


@pytest.fixture(autouse=True)
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA no disponible, sin alternativa CPU para la ruta acelerada")


def blocks(seed=11, sizes=(64, 64, 64, 17, 96, 64, 5)):
    """Bloques float32 con columnas constantes, bits y medias desplazadas."""
    rng = np.random.default_rng(seed)
    result = []
    for size in sizes:
        x = rng.normal(loc=3.0, scale=2.0, size=(size, 40)).astype(np.float32)
        x[rng.random(x.shape) < 0.2] = 0.0
        x[:, 5] = 7.0
        x[:, -3:] = rng.integers(0, 2, size=(size, 3))
        result.append((x, rng.normal(size=size)))
    return result


def previous(pairs, mean, scale, target_mean, count):
    """Ruta anterior, literal salvo el dispositivo fijo."""
    gram = torch.zeros((len(mean), len(mean)), dtype=torch.float64, device=DEVICE)
    rhs = torch.zeros(len(mean), dtype=torch.float64, device=DEVICE)
    sum_x = torch.zeros_like(rhs)
    sum_y = torch.zeros((), dtype=torch.float64, device=DEVICE)
    for x, y in pairs:
        x = x.astype(np.float64)
        features = torch.as_tensor((x - mean) / scale, dtype=torch.float64, device=DEVICE)
        target = torch.as_tensor(y - target_mean, dtype=torch.float64, device=DEVICE)
        gram.addmm_(features.T, features)
        rhs.addmv_(features.T, target)
        sum_x += features.sum(dim=0)
        sum_y += target.sum()
    gram -= torch.outer(sum_x, sum_x) / count
    rhs -= sum_x * sum_y / count
    return gram, rhs, sum_x


def previous_statistics(pairs):
    count, mean, m2, target_mean = 0, None, None, 0.0
    for x, y in pairs:
        x = x.astype(np.float64)
        n = len(x)
        block_mean = x.mean(axis=0)
        block_m2 = np.square(x - block_mean).sum(axis=0)
        if mean is None:
            mean, m2 = block_mean, block_m2
        else:
            delta = block_mean - mean
            m2 += block_m2 + np.square(delta) * count * n / (count + n)
            mean += delta * n / (count + n)
        target_mean += (float(y.mean()) - target_mean) * n / (count + n)
        count += n
    variance = m2 / count
    epsilon = np.finfo(np.float64).eps
    constant = variance <= count * epsilon * variance + np.square(count * mean * epsilon)
    scale = np.sqrt(variance)
    scale[constant] = 1.0
    return count, mean, scale, target_mean


def same(a, b):
    return np.array_equal(a.cpu().numpy(), b.cpu().numpy())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_staged_gram_matches_the_previous_route_bit_for_bit(dtype):
    pairs = blocks()
    count, mean, scale, target_mean = previous_statistics(pairs)
    expected = previous(pairs, mean, scale, target_mean, count)
    gram, rhs, sum_x, digest = ridge.centered_normal_equations(
        ((x.astype(dtype), y) for x, y in pairs), mean, scale, target_mean, count, device=DEVICE
    )
    assert same(gram, expected[0]) and same(rhs, expected[1]) and same(sum_x, expected[2])
    reference = hashlib.sha256()
    for x, y in pairs:
        reference.update(x.astype(dtype).tobytes())
        reference.update(y.tobytes())
    assert digest == reference.digest()


def test_statistics_keep_every_value_of_the_previous_two_passes():
    pairs = blocks(seed=3)
    count, mean, scale, target_mean = previous_statistics(pairs)
    statistics = ridge.ridge_statistics(lambda: iter(pairs))
    assert statistics.count == count and statistics.target_mean == target_mean
    np.testing.assert_array_equal(statistics.mean, mean)
    np.testing.assert_array_equal(statistics.scale, scale)
    assert statistics.scale[5] == 1.0 and statistics.scale[-1] != 1.0
    expected = previous(pairs, mean, scale, target_mean, count)
    assert same(statistics.gram, expected[0]) and same(statistics.rhs, expected[1])
    assert statistics.sha256() == ridge.ridge_statistics(lambda: iter(pairs)).sha256()


def test_changed_rows_between_the_two_passes_are_rejected():
    pairs, calls = blocks(seed=5), []

    def factory():
        calls.append(1)
        for index, (x, y) in enumerate(pairs):
            if len(calls) == 2 and index == 3:
                x = x.copy()
                x[0, 0] = np.nextafter(x[0, 0], np.float32(np.inf))
            yield x, y

    with pytest.raises(ValueError, match="pasadas"):
        ridge.ridge_statistics(factory)


def test_solving_one_alpha_leaves_the_shared_gram_untouched(learning_doubles, monkeypatch):
    """Se registra el sistema que recibiría el solver, sin resolverlo."""
    statistics = ridge.ridge_statistics(lambda: iter(blocks(seed=8)))
    before = statistics.gram.clone()
    systems = []

    def recorded(matrix, rhs):
        systems.append(matrix.clone())
        return torch.zeros_like(rhs)

    monkeypatch.setattr(torch.linalg, "solve", recorded)
    for alpha in (0.1, 10.0):
        ridge.solve_ridge(statistics, alpha)
    assert same(statistics.gram, before)
    for alpha, system in zip((0.1, 10.0), systems, strict=True):
        expected = before.clone()
        expected.diagonal().add_(alpha)
        assert same(system, expected)


def test_device_standardization_predicts_like_numpy_for_an_unfitted_model():
    rng = np.random.default_rng(2)
    pairs = blocks(seed=13)
    _, mean, scale, _ = previous_statistics(pairs)
    model = ridge.RidgeModel(mean, scale, rng.normal(size=len(mean)), 0.25)

    def reference(values):
        return (
            torch.as_tensor((values - mean) / scale, dtype=torch.float64, device=DEVICE)
            @ torch.as_tensor(model.coefficient, device=DEVICE)
            + model.intercept
        ).cpu()

    for x, _ in pairs:
        np.testing.assert_array_equal(model.predict(x), reference(x.astype(np.float64)))
        # Valores float64 que no caben en float32: no pueden redondearse por el camino.
        wide = x.astype(np.float64) + 1e-9
        np.testing.assert_array_equal(model.predict(wide), reference(wide))
    with pytest.raises(ValueError, match="predicción"):
        model.predict(np.full((2, len(mean)), np.nan, dtype=np.float32))
