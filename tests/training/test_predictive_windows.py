"""Normalizador del adaptador predictivo ajustado solo con el tramo de cada ventana.

Usa una caché técnica de predicciones del padre, sin ajustar Ridge ni ninguna red.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.corpus_source import prepare_causal_corpus
from mars_titan.training.predictive_inputs import PredictiveDataset, fit_standardizer
from tests.training.test_temporal_corpus import inputs as inputs
from tests.training.test_temporal_corpus import prepare


def aligned_cache(ordered, folder):
    meta, digest = read_manifest(ordered)
    folder.mkdir()
    partitions = {}
    for partition in ("train", "validation"):
        path = folder / f"{partition}.npy"
        np.save(path, np.linspace(-0.01, 0.01, meta["counts"][partition]))
        content = sha256(path)
        path.rename(folder / f"{partition}-{content}.npy")
        partitions[partition] = dict(path=f"{partition}-{content}.npy", sha256=content)
    atomic_json(
        folder / "manifest.json",
        dict(
            kind="aligned_parent_predictions",
            status="completed",
            identity=dict(
                ordered_manifest_sha256=digest,
                source_sha256=meta["source_sha256"],
                cohort_id=meta.get("cohort_id"),
                weighting="natural",
                market_weights={"US": 1.0},
            ),
            counts=meta["counts"],
            checkpoint_sha256="a" * 64,
            partitions=partitions,
        ),
    )
    return folder / "manifest.json"


def window(inputs, root, *, change_later=False):
    prepare(inputs, root / "views")
    view = root / "views/fold-000/manifest.json"
    if change_later:
        meta = json.loads(view.read_text())
        labels = Path(meta["roots"]["labels"]) / "US/A0000/labels.parquet"
        table = pq.read_table(labels)
        rows = table.to_pylist()
        for row in rows:
            if row["partition"] in {"validation", "calibration", "evaluation"}:
                row["target"] += 1000.0
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), labels)
        meta["assets"][0]["labels_sha256"] = sha256(labels)
        view.write_text(json.dumps(meta))
    report = prepare_causal_corpus(view, root / "ordered", batch_size=2)
    ordered = root / "ordered/manifest.json"
    return report, PredictiveDataset(ordered, aligned_cache(ordered, root / "cache"))


def test_window_standardizer_ignores_later_partitions(inputs, tmp_path):
    before, first = window(inputs, tmp_path / "before")
    after, second = window(inputs, tmp_path / "after", change_later=True)
    with first, second:
        assert before["partitions"]["train"] == after["partitions"]["train"]
        assert before["partitions"]["validation"] != after["partitions"]["validation"]
        statistics = [fit_standardizer(data, batch_size=2) for data in (first, second)]
        assert statistics[0]["fit_partition"] == "train"
        assert statistics[0]["samples"] == before["counts"]["train"]
        for key in ("mean", "scale", "variance", "samples", "weight_sum", "feature_order"):
            assert statistics[0][key] == statistics[1][key], key
        # El recibo sigue ligado al corpus ordenado de su ventana.
        assert statistics[0]["ordered_manifest_sha256"] != statistics[1]["ordered_manifest_sha256"]


class Sealed:
    """Fuente de validación que falla si alguien lee una de sus cohortes."""

    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __len__(self):
        return len(self.source)

    def __call__(self, position):
        raise AssertionError("El normalizador ha leído validación")


def test_window_standardizer_never_reads_validation(inputs, tmp_path):
    _, data = window(inputs, tmp_path)
    with data:
        data.sources["validation"] = Sealed(data.sources["validation"])
        fit_standardizer(data, batch_size=2)
        with pytest.raises(AssertionError, match="validación"):
            next(data.batches(partition="validation", batch_size=2, epoch=0, seed=0))


def test_predictive_reader_rejects_the_masked_edition(tmp_path):
    from tests.posttraining.masked_fixture import masked_ordered

    _, ordered, _ = masked_ordered(tmp_path)
    cache = aligned_cache(ordered, tmp_path / "cache")
    with pytest.raises(ValueError, match="política"):
        PredictiveDataset(ordered, cache)
    assert json.loads(ordered.read_text())["input_policy"] == HISTORICAL_MASKED
