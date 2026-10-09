"""Retención de predicciones reservadas en Ridge, HistGradientBoosting y XGBoost.

El estimador se sustituye por una función fija de las entradas, de modo que se
recorren lectura, predicción por fila y recibo sin estimar ningún parámetro. Por eso los
puntos de entrada se admiten con una protección temporal permitida (`learning_doubles`).
"""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.training import external_corpus, tabular_corpus
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.reference_run import FULL_TRAIN_VALIDATION, HELDOUT_FULL_TRAIN_SESSIONS
from tests.training.test_masked_tabular import LibrariesReached, external
from tests.training.test_walk_forward_v2_views import PROTOCOLS, fixture, prepare


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    root = tmp_path_factory.mktemp("retention")
    prepare(fixture(root / "data", ("US",)), PROTOCOLS["US"], root / "views")
    return root / "views/fold-018/manifest.json"


class Fixed:
    """Suma de los bits de presencia: sin parámetros ajustados."""

    estimator = SimpleNamespace(get_params=lambda: dict(fixed="presence_sum"))

    def predict(self, values):
        return values[:, -5:].sum(axis=1)

    def save(self, path):
        path.write_bytes(b"fixed")


@pytest.fixture
def fixed(monkeypatch, learning_doubles):
    calls = []

    def fit(factory, **_):
        calls.append(sum(len(y) for _, y in factory()))
        return Fixed()

    monkeypatch.setattr(tabular_corpus, "fit_boosting_batches", fit)
    monkeypatch.setattr(
        tabular_corpus.BoostingModel, "load_local", classmethod(lambda cls, *_: Fixed())
    )
    return calls


def rows(view, partition):
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    ids, presence = [], []
    for batch in dataset.batches(partition=partition, batch_size=2, epoch=0, seed=0):
        ids += list(batch["sample_ids"])
        presence.append(batch["presence"])
    return ids, np.concatenate(presence)


def test_heldout_retention_writes_calibration_and_evaluation_rows(view, tmp_path, fixed):
    output = tmp_path / "run"
    report = tabular_corpus.run_tabular_reference(
        view,
        output,
        kind="boosting",
        batch_size=2,
        input_policy=HISTORICAL_MASKED,
        prediction_retention=HELDOUT_FULL_TRAIN_SESSIONS,
    )
    counts = json.loads(view.read_text())["counts"]
    assert report["status"] == "completed" and fixed == [counts["train"]]
    assert report["prediction_retention"] == HELDOUT_FULL_TRAIN_SESSIONS
    assert set(report["predictions"]) == {"validation", "calibration", "evaluation"}
    assert report["train_metrics"]["samples"] == counts["train"]
    assert not (output / "train-predictions.parquet").exists()
    for partition in ("validation", "calibration", "evaluation"):
        ids, presence = rows(view, partition)
        table = pq.read_table(output / report["predictions"][partition]["path"])
        assert table["sample_id"].to_pylist() == ids and len(ids) == counts[partition]
        np.testing.assert_array_equal(table["prediction"].to_numpy(), presence.sum(axis=1))
        assert report["predictions"][partition]["metrics"]["samples"] == counts[partition]
    assert json.loads((output / "run.json").read_text()) == report


def test_default_retention_keeps_the_previous_receipt(view, tmp_path, fixed):
    report = tabular_corpus.run_tabular_reference(
        view, tmp_path / "run", kind="boosting", batch_size=2, input_policy=HISTORICAL_MASKED
    )
    assert set(report["predictions"]) == {"train", "validation"}
    assert "prediction_retention" not in report and "train_metrics" not in report


def test_unknown_retention_is_rejected_before_creating_outputs(view, tmp_path, fixed):
    with pytest.raises(ValueError, match="retención"):
        tabular_corpus.run_tabular_reference(
            view, tmp_path / "run", kind="ridge", prediction_retention="everything"
        )
    assert not (tmp_path / "run").exists() and fixed == []


def test_external_partitions_follow_the_declared_retention(
    view, tmp_path, monkeypatch, learning_doubles
):
    assert external_corpus._partitions({}) == ("train", "validation")
    assert external_corpus._partitions(dict(prediction_retention=HELDOUT_FULL_TRAIN_SESSIONS)) == (
        "validation",
        "calibration",
        "evaluation",
    )
    # La retención no es un parámetro del ajuste externo.
    assert "prediction_retention" in external_corpus._READER_OPTIONS
    external(monkeypatch)
    with pytest.raises(ValueError, match="retención"):
        external_corpus.run_external_reference(
            view, tmp_path / "xgb", prediction_retention="everything"
        )
    with pytest.raises(LibrariesReached):
        external_corpus.run_external_reference(
            view,
            tmp_path / "xgb",
            on_host=False,
            input_policy=HISTORICAL_MASKED,
            max_disk_cache_bytes=1024**3,
            prediction_retention=HELDOUT_FULL_TRAIN_SESSIONS,
        )
    assert not (tmp_path / "xgb").exists()
    assert tabular_corpus.retained_partitions(FULL_TRAIN_VALIDATION) == ("train", "validation")
