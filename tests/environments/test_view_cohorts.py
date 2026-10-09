"""Cohortes leídas por bloques desde la vista frente al corpus ordenado, sin ajustar nada.

La vista es la primera ventana conjunta US+CN de un corpus técnico con tres activos por
mercado, noticias y fundamentales parciales. Cada prueba compara la lectura por bloques
con `ParquetCohortSource` sobre el corpus ordenado de la misma vista: cohortes, bits de
presencia, rejilla, lotes del adaptador, normalización y cursores.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.environments.corpus_source import ParquetCohortSource, prepare_causal_corpus
from mars_titan.environments.view_cohorts import (
    PARTITIONS,
    ViewCohortSource,
    _SessionRows,
    prepare_cohort_index,
)
from mars_titan.episodes.parents import ParentCache
from mars_titan.posttraining.inputs import PairedInputs, fit_normalization
from mars_titan.training import masked_campaign as engine
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.historical_temporal_fixture import historical_temporal_fixture
from tests.training.test_masked_campaign import write_campaign
from tests.training.test_walk_forward_v2_views import sessions

POLICY = HISTORICAL_MASKED
BATCH = 7
LARGE = 64 * 1024**2


@pytest.fixture(scope="module")
def joint(tmp_path_factory):
    root = tmp_path_factory.mktemp("view-cohorts")
    data = historical_temporal_fixture(
        root / "data",
        markets=("US", "CN"),
        days={market: sessions(market) for market in ("US", "CN")},
        assets=3,
        presence=lambda market, symbol, row: (row % 4 == 1, row % 5 == 2, row % 3 == 0),
    )
    campaign = write_campaign(root / "config", scopes=("US", "US+CN"))
    prepared = engine.prepare_views(campaign, data.parent, root / "views")
    view = Path(prepared["US+CN"]["windows"]["fold-000"]["path"])
    ordered = prepare_causal_corpus(view, root / "ordered", batch_size=BATCH, input_policy=POLICY)
    dataset = CorpusDataset(view, input_policy=POLICY)
    index = prepare_cohort_index(dataset, root / "index", input_policy=POLICY)
    return SimpleNamespace(
        root=root,
        view=view,
        dataset=dataset,
        ordered=ordered,
        index=index,
        ordered_path=root / "ordered/manifest.json",
        index_path=root / "index/manifest.json",
    )


def streamed(joint, partition, budget=LARGE):
    return ViewCohortSource(
        joint.index_path,
        joint.dataset,
        partition=partition,
        max_block_bytes=budget,
        input_policy=POLICY,
    )


def same_cohort(left, right):
    assert left.keys() == right.keys()
    assert left["prediction_at"] == right["prediction_at"]
    assert left["asset_ids"] == right["asset_ids"]
    for name in ("available_at", "target_available_at", "target", "presence"):
        assert left[name].dtype == right[name].dtype
        assert left[name].tobytes() == right[name].tobytes(), name
    assert left["inputs"].keys() == right["inputs"].keys()
    for name, value in left["inputs"].items():
        assert (
            value.dtype == right["inputs"][name].dtype
            and value.shape == right["inputs"][name].shape
        )
        assert value.tobytes() == right["inputs"][name].tobytes(), name


def test_index_keeps_the_population_markets_and_grid_of_the_ordered_corpus(joint):
    ordered, index = joint.ordered, joint.index
    assert index["kind"] == "causal_prediction_index" and ordered["status"] == "completed"
    for key in ("counts", "shapes", "source_sha256", "scope", "cohort_complete", "grid"):
        assert index[key] == ordered[key], key
    for partition in PARTITIONS:
        left, right = index["partitions"][partition], ordered["partitions"][partition]
        for key in ("rows", "cohorts", "max_assets", "market_rows"):
            assert left[key] == right[key], (partition, key)
    assert set(index["partitions"]["train"]["market_rows"]) == {"US", "CN"}
    # Sin filas: solo el manifiesto del índice queda en disco.
    assert [path.name for path in joint.index_path.parent.iterdir()] == ["manifest.json"]


@pytest.mark.parametrize("budget", [LARGE, 60_000, 20_000])
@pytest.mark.parametrize("partition", PARTITIONS)
def test_cohorts_match_the_ordered_source_bit_for_bit(joint, partition, budget):
    source = streamed(joint, partition, budget)
    order = np.random.default_rng(budget).permutation(len(source)).tolist()
    with ParquetCohortSource(joint.ordered_path, partition=partition, input_policy=POLICY) as ref:
        assert source.index == ref.index and source.shapes == ref.shapes
        assert source.market_bounds == ref.market_bounds
        count = 0
        for position, cohort in zip(order, source.cohorts(order), strict=True):
            same_cohort(cohort, ref(position))
            count += 1
    assert count == len(order)
    statistics = source.statistics
    assert statistics["peak_block_bytes"] <= budget
    # El primer bloque supone filas completas. Si el tramo no cabe así, hay varias lecturas.
    assert (statistics["passes"] > 1) == (budget < source.row_bytes * int(source.offsets[-1]))


def test_an_underestimated_block_evicts_its_last_sessions_and_stays_exact(joint):
    source = streamed(joint, "train", 40_000)
    source.estimate = 1
    order = np.random.default_rng(1).permutation(len(source)).tolist()
    order = order + order[:3]
    with ParquetCohortSource(joint.ordered_path, partition="train", input_policy=POLICY) as ref:
        for position, cohort in zip(order, source.cohorts(order), strict=True):
            same_cohort(cohort, ref(position))
    assert source.statistics["evicted"] > 0
    assert source.statistics["peak_block_bytes"] <= 40_000


class Predictor:
    """Padre técnico que depende de las entradas y de los bits de presencia."""

    def __call__(self, inputs, presence=None):
        return inputs["charts"].mean(axis=1) + 0.01 * presence.sum(axis=1)


def paired(joint, folder, sources):
    cache = ParentCache(folder / "parent.sqlite", "a" * 64, "masked", Predictor())
    return PairedInputs(*sources, cache), cache


def comparable(batch):
    cursor = {k: v for k, v in batch["confirmed_cursor"].items() if k != "data_sha256"}
    arrays = {
        name: (value.dtype.str, value.shape, value.tobytes())
        for name, value in batch.items()
        if isinstance(value, np.ndarray)
    }
    inputs = {name: value.tobytes() for name, value in batch["inputs"].items()}
    lists = {name: batch[name] for name in ("sample_ids", "market", "origin")}
    return arrays, inputs, lists, cursor


@pytest.fixture(scope="module")
def pairs(joint):
    ordered = [
        ParquetCohortSource(joint.ordered_path, partition=name, input_policy=POLICY)
        for name in PARTITIONS
    ]
    folder = joint.root / "pairs"
    (folder / "ordered").mkdir(parents=True)
    (folder / "streamed").mkdir()
    left, left_cache = paired(joint, folder / "ordered", ordered)
    right, right_cache = paired(
        joint, folder / "streamed", [streamed(joint, p, 30_000) for p in PARTITIONS]
    )
    yield left, right
    for handle in (*ordered, left_cache, right_cache):
        handle.close()


@pytest.mark.parametrize(
    ("partition", "epoch", "batch_size"),
    [("train", 0, 3), ("train", 1, 3), ("train", 0, 64), ("validation", 0, 5)],
)
def test_adapter_batches_order_and_cursors_match_the_ordered_path(
    pairs, partition, epoch, batch_size
):
    left, right = pairs
    options = dict(
        partition=partition, condition="real", batch_size=batch_size, epoch=epoch, seed=42
    )
    expected = [comparable(batch) for batch in left.batches(**options)]
    observed = [comparable(batch) for batch in right.batches(**options)]
    assert len(observed) == len(expected) > 1
    assert observed == expected
    assert left.budget("real", batch_size) == right.budget("real", batch_size)


def test_normalization_matches_the_ordered_path(pairs):
    left, right = pairs
    first, second = (fit_normalization(data, batch_size=4) for data in (left, right))
    assert first["mean"] == second["mean"] and first["scale"] == second["scale"]
    assert first["samples"] == second["samples"]


def test_resumed_batches_continue_after_the_confirmed_cursor(pairs):
    _, right = pairs
    options = dict(partition="train", condition="real", batch_size=2, epoch=3, seed=7)
    batches = list(right.batches(**options))
    middle = next(
        i for i, batch in enumerate(batches) if batch["confirmed_cursor"]["offset"] and i > 2
    )
    cursor = batches[middle]["confirmed_cursor"]
    resumed = [comparable(batch) for batch in right.batches(**options, cursor=cursor)]
    assert resumed == [comparable(batch) for batch in batches[middle + 1 :]]


def test_a_session_larger_than_the_budget_is_rejected(joint):
    source = streamed(joint, "train", 64)
    with pytest.raises(ValueError, match="no cabe"):
        next(source.cohorts([0, 1]))


def test_the_index_rejects_another_code_view_or_altered_record(joint, tmp_path):
    document = json.loads(joint.index_path.read_text())
    stale = json.loads(json.dumps(document))
    stale["identity"]["code"]["environments/view_cohorts.py"] = "0" * 64
    (tmp_path / "stale").mkdir()
    (tmp_path / "stale/manifest.json").write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="otra vista, política o código"):
        prepare_cohort_index(joint.dataset, tmp_path / "stale", input_policy=POLICY)
    document["partitions"]["train"]["cohorts"][0][1] += 1
    altered = tmp_path / "manifest.json"
    altered.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="huella"):
        ViewCohortSource(
            altered, joint.dataset, partition="train", max_block_bytes=LARGE, input_policy=POLICY
        )
    other = SimpleNamespace(identity="0" * 64, masked=True, cohort=joint.dataset.cohort)
    with pytest.raises(ValueError, match="vista abierta"):
        ViewCohortSource(
            joint.index_path, other, partition="train", max_block_bytes=LARGE, input_policy=POLICY
        )


def test_session_rows_keep_zero_repeated_and_signed_values_bit_for_bit():
    rng = np.random.default_rng(0)
    shapes = {"news": (4,), "charts": (3,), "prices": (2, 5)}
    rows = 9
    inputs = {
        name: rng.normal(size=(rows, *shape)).astype(np.float32) for name, shape in shapes.items()
    }
    inputs["news"][[0, 3, 4]] = 0.0
    inputs["news"][5] = -0.0
    inputs["news"][[6, 7]] = inputs["news"][1]
    inputs["charts"][:] = inputs["charts"][2]
    inputs["charts"][8, 1] = -0.0
    batch = dict(
        inputs=inputs,
        sample_ids=[f"US/A{i}/7" for i in range(rows)],
        input_available_at=np.arange(rows).astype("datetime64[us]"),
        target_available_at=np.arange(rows, 2 * rows).astype("datetime64[us]"),
        target=rng.normal(size=rows),
        presence=rng.random((rows, 5)) < 0.5,
    )
    session = _SessionRows(shapes, masked=True)
    order = [4, 0, 1, 8, 7, 6, 5, 3, 2]
    stored = session.add(batch, np.array(order[:3])) + session.add(batch, np.array(order[3:]))
    raw, presence = session.raw(7)
    assert stored == session.bytes and session.rows == rows
    assert raw["asset_ids"] == [f"US/A{i}" for i in order]
    assert presence.tobytes() == batch["presence"][order].tobytes()
    for name in shapes:
        assert raw["inputs"][name].tobytes() == inputs[name][order].tobytes(), name
    assert raw["target"].tobytes() == batch["target"][order].tobytes()
    # Los ceros y las filas repetidas no se copian. Los ceros negativos sí.
    kinds = {
        name: np.concatenate([c["inputs"][name][0] for c in session.chunks]) for name in shapes
    }
    assert (kinds["news"] == 0).sum() == 3 and (kinds["news"] == 1).sum() == 3
    assert (kinds["charts"] == 1).sum() == rows - 1
