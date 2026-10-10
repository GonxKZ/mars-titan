"""Ventana compartida de XGBoost en CUDA con el lector real, sin ajustar ningún árbol.

`xgb.train` se sustituye por un doble que anota la matriz recibida y se detiene. Se
comprueba qué configuraciones reutilizan la matriz cuantizada y la validación, cuántas
veces se recorre el corpus, que las claves de las filas siguen el orden del lector y que
no quedan páginas en disco al liberar la ventana ni tras un intento interrumpido.
"""

import importlib
import json

import numpy as np
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.training import external_corpus
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.historical_temporal_fixture import historical_temporal_fixture

pytestmark = [
    pytest.mark.skipif(
        any(importlib.util.find_spec(name) is None for name in ("cupy", "xgboost")),
        reason="La comprobación CUDA requiere el extra boosting",
    ),
    # XGBoost avisa si sus páginas desaparecen antes de liberar la matriz.
    pytest.mark.filterwarnings("error::UserWarning"),
]
SELECTION = dict(schema_version=1, minimum_rounds=2, patience_rounds=2, min_delta=1e-5)


class TrainingReached(Exception):
    """La matriz y la validación están listas y se iba a ajustar."""


@pytest.fixture
def window(tmp_path, monkeypatch, learning_doubles):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA no disponible, sin alternativa CPU para la ruta externa")
    import xgboost as xgb

    manifest = historical_temporal_fixture(tmp_path / "fixture").parent
    received, reads = [], []

    def train(params, data, *args, **kwargs):
        received.append(dict(data=data, params=params))
        raise TrainingReached

    original = CorpusDataset.batches

    def counted(self, *, partition, **options):
        reads.append(partition)
        return original(self, partition=partition, **options)

    monkeypatch.setattr(xgb, "train", train)
    monkeypatch.setattr(CorpusDataset, "batches", counted)
    external_corpus.SHARED.release()
    yield manifest, tmp_path, received, reads
    external_corpus.SHARED.release()


def run(manifest, output, shared, **changes):
    options = dict(
        rounds=6,
        max_depth=2,
        max_bin=16,
        batch_size=3,
        max_batch_bytes=1024**2,
        on_host=False,
        max_disk_cache_bytes=1024**3,
        selection=SELECTION,
        max_validation_cache_bytes=1024**2,
        input_policy=HISTORICAL_MASKED,
        prediction_retention="heldout_full_train_sessions_v1",
        shared_directory=shared,
    )
    with pytest.raises(TrainingReached):
        external_corpus.run_external_reference(manifest, output, **(options | changes))
    return json.loads((output / "run.json").read_text())["attempts"][-1]


def test_trees_with_the_same_bins_reuse_the_matrix_and_another_bin_rebuilds(window):
    manifest, root, received, reads = window
    shared = root / "jobs" / ".shared-matrix"
    first = run(manifest, root / "a", shared)
    construction = reads.count("train")
    second = run(manifest, root / "b", shared, max_depth=3, learning_rate=0.1)
    assert received[0]["data"] is received[1]["data"]
    assert received[0]["params"]["max_depth"] != received[1]["params"]["max_depth"]
    # La segunda configuración solo abre el lector para medir la anchura de las filas.
    assert reads.count("train") == construction + 1
    assert reads.count("validation") == 1
    assert first["matrix"]["constructed"] and not second["matrix"]["constructed"]
    assert first["matrix"]["construction_passes"] == second["matrix"]["construction_passes"]
    assert first["validation_cache_bytes"] == second["validation_cache_bytes"] > 0
    assert shared.is_dir() and external_corpus.SHARED.constructions == 1
    third = run(manifest, root / "c", shared, max_bin=8)
    assert third["matrix"]["constructed"] and received[2]["data"] is not received[0]["data"]
    assert external_corpus.SHARED.constructions == 2 and reads.count("validation") == 1
    external_corpus.SHARED.release()
    assert not shared.exists()


def test_without_sharing_every_run_builds_and_removes_its_own_pages(window):
    manifest, root, received, _ = window
    first = run(manifest, root / "a", None)
    second = run(manifest, root / "b", None)
    assert first["matrix"]["constructed"] and second["matrix"]["constructed"]
    assert received[0]["data"] is not received[1]["data"]
    assert not list((root / "a").glob("external-*")) and not list((root / "b").glob("external-*"))


def test_recorded_row_keys_follow_the_reader_order(window):
    manifest, root, _, _ = window
    run(manifest, root / "a", root / "shared")
    rows = external_corpus.SHARED.matrix.rows
    dataset = CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)
    markets, moments, targets = [], [], []
    for batch in dataset.batches(partition="train", batch_size=3, epoch=0, seed=0):
        markets.extend(batch["market"])
        moments.append(batch["prediction_at"].astype("datetime64[us]").astype(np.int64))
        targets.append(batch["target"])
    np.testing.assert_array_equal(np.where(rows.china, "CN", "US"), markets)
    np.testing.assert_array_equal(rows.moments, np.concatenate(moments))
    np.testing.assert_array_equal(rows.target, np.concatenate(targets))
    assert rows.expected == len(markets) == dataset.manifest["counts"]["train"]


def test_an_interrupted_attempt_leaves_no_pages_for_the_next_one(window):
    manifest, root, _, _ = window
    output = root / "a"
    run(manifest, output, None)
    stale = output / "external-stale" / "pages"
    stale.mkdir(parents=True)
    (stale / "pages.ellpack").write_bytes(b"x" * 4096)
    resumed = run(manifest, output, None, resume=True)
    assert resumed["stale_temporary_bytes_removed"] == 4096
    assert not (output / "external-stale").exists()


def test_after_the_hold_a_shared_matrix_fits_the_same_model_as_its_own(tmp_path):
    """Solo sin bloqueo: ajusta el fixture con matriz propia y compartida y compara bits.

    Bajo el bloqueo, `run_external_reference` se detiene antes de abrir fuentes y la
    prueba se omite sin ajustar nada. La segunda configuración reutiliza la matriz de la
    primera y debe dar el mismo modelo y las mismas predicciones que un ajuste con su
    propia matriz, repetido dos veces para comprobar también el determinismo.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA no disponible, sin alternativa CPU para la ruta externa")
    manifest = historical_temporal_fixture(tmp_path / "fixture").parent
    shared = tmp_path / "jobs" / ".shared-matrix"
    options = dict(
        rounds=6,
        max_depth=3,
        max_bin=16,
        batch_size=3,
        max_batch_bytes=1024**2,
        on_host=False,
        max_disk_cache_bytes=1024**3,
        selection=SELECTION,
        max_validation_cache_bytes=1024**2,
        input_policy=HISTORICAL_MASKED,
        prediction_retention="heldout_full_train_sessions_v1",
    )

    def fit(name, directory, **changes):
        return external_corpus.run_external_reference(
            manifest, tmp_path / name, shared_directory=directory, **(options | changes)
        )

    external_corpus.SHARED.release()
    try:
        own = fit("own", None)
        repeated = fit("repeated", None)
        fit("first", shared, max_depth=2)
        reused = fit("reused", shared)
    finally:
        external_corpus.SHARED.release()
    assert not reused["attempts"][-1]["matrix"]["constructed"]
    for report in (repeated, reused):
        assert report["status"] == own["status"] == "completed"
        assert report["checkpoint"]["sha256"] == own["checkpoint"]["sha256"]
        assert report["train_metrics"] == own["train_metrics"]
        assert report["selection"] == own["selection"]
        for partition, record in own["predictions"].items():
            assert report["predictions"][partition]["sha256"] == record["sha256"]
