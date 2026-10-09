"""Sesiones de hasta 8.192 activos en el corpus ordenado y en los padres congelados."""

import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments import corpus_source
from mars_titan.environments.cohorts import MAX_COHORT_ASSETS, VALIDATION_START_US, shapes_contract
from mars_titan.environments.corpus_source import MAX_BLOCK_BYTES, ParquetCohortSource
from mars_titan.episodes.parents import ParentCache
from mars_titan.models.baselines.multimodal import MultimodalReference
from mars_titan.posttraining import parents
from mars_titan.posttraining.inputs import PairedInputs
from mars_titan.posttraining.parents import FrozenParent
from mars_titan.training.predictive_parents import ParentPredictions

SHAPES = dict(prices=[4, 5], news=[2], charts=[2], fundamentals=[3], macro=[3])
# Formas de la edición desde 2000: 64 sesiones de precios y 1.714 valores por fila.
HISTORICAL = dict(prices=[64, 5], news=[384], charts=[512], fundamentals=[78], macro=[420])
DAY = 86_400_000_000


def test_cohort_check_admits_the_shared_cap_and_rejects_one_more_asset():
    bounds = {"US": (0, VALIDATION_START_US, VALIDATION_START_US)}
    at = VALIDATION_START_US - 10 * DAY

    def row(rows, distinct=None):
        return ("US", at, rows, rows if distinct is None else distinct, 1, at, at + 1, at + DAY)

    corpus_source._check_cohorts([row(MAX_COHORT_ASSETS)], bounds)
    assert MAX_COHORT_ASSETS == 8192
    for invalid in (row(MAX_COHORT_ASSETS + 1), row(5000, 4999), row(0)):
        with pytest.raises(ValueError, match="cohorte"):
            corpus_source._check_cohorts([invalid], bounds)


def test_memory_budgets_cover_a_full_session_of_the_historical_edition():
    width = sum(math.prod(shape) for shape in HISTORICAL.values())
    assert width == 1714
    shapes_contract(HISTORICAL, MAX_COHORT_ASSETS, MAX_BLOCK_BYTES)
    assert MAX_COHORT_ASSETS * width * 4 <= parents.MAX_INPUT_BYTES
    # Con más columnas el presupuesto falla de forma explícita en lugar de recortar filas.
    wider = dict(HISTORICAL, charts=[1024])
    with pytest.raises(ValueError, match="presupuesto"):
        shapes_contract(wider, MAX_COHORT_ASSETS, MAX_BLOCK_BYTES)


def partition(root, name, at, count):
    generator = np.random.default_rng(count)
    ids = [f"US/A{i:05d}" for i in range(count)]
    columns = dict(
        cohort_id=pa.array([None] * count, pa.string()),
        sample_id=[f"{asset}/{at}" for asset in ids],
        asset_id=ids,
        prediction_at=np.full(count, at, dtype=np.int64),
        available_at=np.full(count, at - 1, dtype=np.int64),
        target_available_at=np.full(count, at + DAY, dtype=np.int64),
        target=generator.normal(scale=0.01, size=count),
    )
    for key, shape in SHAPES.items():
        width = math.prod(shape)
        values = generator.normal(size=count * width).astype(np.float32)
        columns[key] = pa.FixedSizeListArray.from_arrays(pa.array(values), width)
    path = root / "pending.parquet"
    pq.write_table(pa.table(columns), path, row_group_size=2048)
    digest = sha256(path)
    target = root / f"{name}-{digest}.parquet"
    path.rename(target)
    return dict(
        path=target.name,
        sha256=digest,
        size_bytes=target.stat().st_size,
        rows=count,
        cohorts=[[at, count]],
        max_assets=count,
        market_rows={"US": count},
    )


def legacy_ordered(root, rows):
    """Corpus ordenado de dos particiones con una sesión de ajuste de `rows` activos."""
    train_at, validation_at = VALIDATION_START_US - 30 * DAY, VALIDATION_START_US + 30 * DAY
    records = dict(
        train=partition(root, "train", train_at, rows),
        validation=partition(root, "validation", validation_at, 3),
    )
    counts = {name: record["rows"] for name, record in records.items()}
    meta = dict(
        schema_version=1,
        kind="causal_prediction_corpus",
        status="completed",
        final_test_opened=False,
        identity=dict(cohort_id=None, news_content_policy=None),
        source_sha256="a" * 64,
        counts=counts,
        cohort_id=None,
        news_content_policy=None,
        scope="full_corpus",
        cohort_complete=True,
        shapes=SHAPES,
        partitions=records,
    )
    atomic_json(root / "manifest.json", meta)
    return root / "manifest.json"


@pytest.fixture
def crowded(tmp_path):
    return legacy_ordered(tmp_path, MAX_COHORT_ASSETS)


def test_a_full_session_reaches_the_parent_and_the_adapter_without_losing_rows(crowded, tmp_path):
    torch.manual_seed(1)
    model = MultimodalReference(
        "gru", {k: v[-1] for k, v in SHAPES.items()}, context=4, hidden_size=32
    )
    parent = FrozenParent(model, dict(model="gru", checkpoint_sha256="b" * 64), SHAPES, "cpu")
    with (
        ParquetCohortSource(crowded, partition="train") as train,
        ParquetCohortSource(crowded, partition="validation") as validation,
        ParentCache(tmp_path / "cache.sqlite", "b" * 64, "capacity", parent.predict) as cache,
    ):
        raw = train(0)
        assert len(raw["asset_ids"]) == MAX_COHORT_ASSETS
        data = PairedInputs(train, validation, cache)
        seen, ids = 0, set()
        for batch in data.batches(partition="train", condition="real", batch_size=4096):
            seen += len(batch["target"])
            ids.update(batch["sample_ids"])
        assert seen == len(ids) == MAX_COHORT_ASSETS
        assert data.budget("real", 4096)["updates"] == 2
        predictions = parent.predict(raw["inputs"])
        with torch.inference_mode():
            direct = model({k: torch.tensor(v) for k, v in raw["inputs"].items()})
        np.testing.assert_allclose(predictions, direct.double().numpy(), rtol=0, atol=1e-6)
        extended = {k: np.concatenate([v, v[:1]]) for k, v in raw["inputs"].items()}
        with pytest.raises(ValueError, match="dimensiones"):
            parent.predict(extended)


def test_aligned_parent_values_read_a_full_session(crowded, tmp_path):
    values = np.linspace(-1, 1, MAX_COHORT_ASSETS)
    np.save(tmp_path / "train.npy", values)
    digest = sha256(tmp_path / "train.npy")
    (tmp_path / "train.npy").rename(tmp_path / f"train-{digest}.npy")
    meta = json.loads(crowded.read_text())
    cache = dict(
        kind="aligned_parent_predictions",
        status="completed",
        identity=dict(
            ordered_manifest_sha256=sha256(crowded),
            source_sha256=meta["source_sha256"],
            cohort_id=None,
        ),
        counts=meta["counts"],
        partitions=dict(train=dict(path=f"train-{digest}.npy", sha256=digest)),
    )
    atomic_json(tmp_path / "cache.json", cache)
    with ParquetCohortSource(crowded, partition="train") as source:
        aligned = ParentPredictions(tmp_path / "cache.json", source)
        np.testing.assert_array_equal(aligned.values(0, MAX_COHORT_ASSETS), values)
        with pytest.raises(ValueError, match="tramo"):
            aligned.values(0, MAX_COHORT_ASSETS + 1)
        aligned.close()
