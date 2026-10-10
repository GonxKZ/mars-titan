"""Caché externa de XGBoost en disco medida en CUDA, sin ejecutar rondas de boosting.

La construcción de ExtMemQuantileDMatrix solo calcula cuantiles y páginas. xgb.train
se sustituye por una excepción, de modo que ninguna prueba puede ajustar árboles. Por eso
las pruebas usan `learning_doubles`: sin él, la protección del aprendizaje detiene
`fit_external_boosting` en su entrada y la caché nunca llega a construirse.
"""

import importlib

import numpy as np
import pytest

from mars_titan.models.baselines import external_boosting as external

pytestmark = [
    pytest.mark.skipif(
        any(importlib.util.find_spec(name) is None for name in ("cupy", "xgboost")),
        reason="La comprobación CUDA requiere el extra boosting",
    ),
    pytest.mark.usefixtures("learning_doubles"),
]

ROWS, FEATURES, MAX_BIN = 20_000, 1719, 128


class TrainingBlocked(Exception):
    """La construcción terminó y se iba a entrenar."""


def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA no disponible, sin alternativa CPU para la ruta externa")


def factory():
    # Valores continuos aleatorios: cada característica ocupa todos sus bins.
    rng = np.random.default_rng(20261009)
    for _ in range(0, ROWS, 2000):
        yield rng.random((2000, FEATURES), dtype=np.float32), np.zeros(2000, dtype=np.float32)


def blocked_training(monkeypatch, cache, observed):
    xgb = pytest.importorskip("xgboost")

    def train(params, data, *args, **kwargs):
        observed.update(bytes=external.directory_bytes(cache), rows=data.num_row())
        raise TrainingBlocked

    monkeypatch.setattr(xgb, "train", train)


def test_disk_pages_fit_the_declared_budget_and_the_dense_estimate(tmp_path, monkeypatch):
    cuda()
    cache, observed = tmp_path / "cache", {}
    blocked_training(monkeypatch, cache, observed)
    with pytest.raises(TrainingBlocked):
        external.fit_external_boosting(
            factory,
            cache,
            expected_rows=ROWS,
            rounds=1,
            max_bin=MAX_BIN,
            on_host=False,
            max_disk_cache_bytes=4 * 1024**3,
        )
    plan = external.external_cache_plan(
        rows=ROWS,
        features=FEATURES,
        max_bin=MAX_BIN,
        on_host=False,
        max_host_cache_bytes=1,
        max_disk_cache_bytes=4 * 1024**3,
        available_ram=0,
        free_disk=2**62,
    )
    assert observed["rows"] == ROWS
    assert 0 < observed["bytes"] <= plan["disk_cache_bytes_global_bins_bound"]
    # Contraste de la hipótesis ELLPACK densa con un margen fijo para cabeceras.
    assert observed["bytes"] <= plan["disk_cache_bytes_estimate"] + 16 * 1024**2


def test_disk_guard_stops_the_construction_before_any_round(tmp_path, monkeypatch):
    cuda()
    cache, observed = tmp_path / "cache", {}
    blocked_training(monkeypatch, cache, observed)
    with pytest.raises(ValueError, match="presupuesto de disco"):
        external.fit_external_boosting(
            factory,
            cache,
            expected_rows=ROWS,
            rounds=1,
            max_bin=MAX_BIN,
            on_host=False,
            max_disk_cache_bytes=1024,
        )
    assert not observed
