"""Las alphas de una ventana comparten una pasada de estadísticas Ridge, sin resolver alphas.

`solve_ridge` se sustituye por un doble que devuelve un modelo nulo sin ajustar. Se comprueba
que la segunda alpha reutiliza la misma Gram sin leer el corpus y que otro reparto de lotes
obliga a recalcularla.
"""

import numpy as np
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.models.baselines.ridge import RidgeModel
from mars_titan.training import tabular_corpus
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.historical_temporal_fixture import historical_temporal_fixture


@pytest.fixture
def solved(monkeypatch, learning_doubles):
    if not torch.cuda.is_available():
        pytest.skip("CUDA no disponible, sin alternativa CPU para la ruta acelerada")
    calls = []

    def solve(statistics, alpha):
        calls.append((statistics, alpha))
        return RidgeModel(statistics.mean, statistics.scale, np.zeros(len(statistics.mean)), 0.0)

    monkeypatch.setattr(tabular_corpus, "solve_ridge", solve)
    tabular_corpus.RIDGE_STATISTICS.clear()
    yield calls
    tabular_corpus.RIDGE_STATISTICS.clear()


def test_alphas_share_the_statistics_and_another_batch_recomputes(tmp_path, monkeypatch, solved):
    manifest = historical_temporal_fixture(tmp_path / "fixture").parent
    reads = []
    original = CorpusDataset.batches

    def counted(self, *, partition, **options):
        reads.append(partition)
        return original(self, partition=partition, **options)

    monkeypatch.setattr(CorpusDataset, "batches", counted)

    def run(name, **changes):
        options = dict(alpha=0.1, batch_size=3, input_policy=HISTORICAL_MASKED)
        return tabular_corpus.run_tabular_reference(
            manifest, tmp_path / name, kind="ridge", **(options | changes)
        )

    first = run("a")
    fitting_reads = reads.count("train")
    second = run("b", alpha=10.0)
    assert first["statistics"]["computed"] and first["fit_passes"] == 2
    assert not second["statistics"]["computed"] and second["fit_passes"] == 0
    assert first["statistics"]["sha256"] == second["statistics"]["sha256"]
    assert solved[0][0] is solved[1][0] and [alpha for _, alpha in solved] == [0.1, 10.0]
    assert first["fitted_rows"] == second["fitted_rows"] == 4
    # Anchura, dos pasadas de estadísticas y predicciones. La segunda alpha omite las pasadas.
    assert fitting_reads == 4 and reads.count("train") - fitting_reads == 2
    assert first["status"] == second["status"] == "completed"
    third = run("c", batch_size=2)
    assert third["statistics"]["computed"] and third["fit_passes"] == 2


def test_after_the_hold_shared_statistics_fit_the_same_ridge(tmp_path):
    """Solo sin bloqueo: una alpha con estadísticas compartidas frente a una propia.

    Bajo el bloqueo, `run_tabular_reference` se detiene antes de abrir fuentes y la
    prueba se omite sin resolver ningún sistema.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA no disponible, sin alternativa CPU para la ruta acelerada")
    manifest = historical_temporal_fixture(tmp_path / "fixture").parent

    def run(name, alpha):
        return tabular_corpus.run_tabular_reference(
            manifest,
            tmp_path / name,
            kind="ridge",
            alpha=alpha,
            batch_size=3,
            input_policy=HISTORICAL_MASKED,
        )

    tabular_corpus.RIDGE_STATISTICS.clear()
    try:
        run("first", 0.1)
        shared = run("shared", 10.0)
        tabular_corpus.RIDGE_STATISTICS.clear()
        own = run("own", 10.0)
    finally:
        tabular_corpus.RIDGE_STATISTICS.clear()
    assert not shared["statistics"]["computed"] and own["statistics"]["computed"]
    assert shared["statistics"]["sha256"] == own["statistics"]["sha256"]
    # El zip de numpy guarda la hora de escritura: se comparan los valores cargados.
    left, right = (RidgeModel.load(tmp_path / name / "model.npz") for name in ("shared", "own"))
    for name in ("mean", "scale", "coefficient", "intercept"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    for partition, record in own["predictions"].items():
        assert shared["predictions"][partition]["sha256"] == record["sha256"]
