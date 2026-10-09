"""Lectores de muestras con grupos Parquet vacíos, comparados con el mismo archivo compacto."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.corpus_inputs import CorpusDataset, _populated_groups, _random
from mars_titan.training.corpus_targets import _label_batches
from mars_titan.training.temporal_corpus import _masked_sample_state, prepare_temporal_corpus
from tests.training.chronological_fixture import chronological_corpus
from tests.training.empty_groups_fixture import (
    canonical,
    final_empty,
    inner_empty,
    regroup,
    regroup_samples,
)
from tests.training.historical_temporal_fixture import historical_temporal_fixture
from tests.training.temporal_fixture import temporal_fixture
from tests.training.test_historical_temporal import prepare

START = 946_684_800_000_000
END = 1_704_067_200_000_000
LAYOUTS = [
    pytest.param(final_empty, id="final"),
    pytest.param(inner_empty, id="intermedio"),
]


def masked(manifest):
    return CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)


def supervised(dataset):
    return [
        canonical(list(dataset.batches(partition=p, batch_size=5, epoch=e, seed=s)))
        for p in ("train", "validation")
        for e in (0, 1)
        for s in (0, 7, 42)
    ]


def observations(dataset):
    return list(dataset.observation_batches(start=START, end=END, batch_size=4))


def corpus(tmp_path):
    manifest = chronological_corpus(tmp_path / "corpus", group_size=8)
    sizes = regroup_samples(manifest, 8)
    assert all(0 not in groups for groups in sizes.values())
    return manifest


@pytest.mark.parametrize("layout", LAYOUTS)
def test_supervised_batches_and_cursors_match_the_compact_file(tmp_path, layout):
    manifest = corpus(tmp_path)
    expected = supervised(masked(manifest))
    sizes = regroup_samples(manifest, 8, layout)
    assert all(groups.count(0) == len(layout(0)) for groups in sizes.values())
    assert supervised(masked(manifest)) == expected


def test_cursor_resumes_the_same_suffix_after_empty_groups(tmp_path):
    manifest = corpus(tmp_path)
    regroup_samples(manifest, 8, inner_empty)
    dataset = masked(manifest)
    for partition in ("train", "validation"):
        complete = list(dataset.batches(partition=partition, batch_size=3, epoch=1, seed=42))
        assert len(complete) > 3
        for index, batch in enumerate(complete):
            resumed = dataset.batches(
                partition=partition,
                batch_size=3,
                epoch=1,
                seed=42,
                cursor=batch["confirmed_cursor"],
            )
            assert canonical(list(resumed)) == canonical(complete[index + 1 :])


def previous_order(dataset, partition, epoch, seed):
    """Orden anterior por índice físico, válido para archivos sin grupos vacíos."""
    identities = []
    for position in _random(seed, epoch, "assets").permutation(len(dataset.assets)):
        asset = dataset.assets[int(position)]
        key = f"{asset['market']}/{asset['symbol']}"
        with pq.ParquetFile(dataset._file(asset, "samples")) as file:
            positions, prediction, _, _ = dataset._labels(asset, partition, file.metadata.num_rows)
            offsets = np.cumsum(
                [0] + [file.metadata.row_group(g).num_rows for g in range(file.num_row_groups)]
            )
            for group in _random(seed, epoch, key + "/groups").permutation(file.num_row_groups):
                first, end = np.searchsorted(positions, offsets[group : group + 2])
                indexes = np.arange(first, end)
                if partition == "train":
                    indexes = _random(seed, epoch, key + f"/{group}").permutation(indexes)
                identities += [f"{key}/{prediction[i]}" for i in indexes]
    return identities


def test_files_without_empty_groups_keep_the_previous_order(tmp_path):
    dataset = masked(corpus(tmp_path))
    for partition in ("train", "validation"):
        for seed in (0, 42):
            batches = dataset.batches(partition=partition, batch_size=4, epoch=2, seed=seed)
            actual = [identity for batch in batches for identity in batch["sample_ids"]]
            assert actual == previous_order(dataset, partition, 2, seed)
            assert len(actual) == dataset.manifest["counts"][partition] > 4


@pytest.mark.parametrize("layout", LAYOUTS)
def test_observations_match_the_compact_file_and_keep_physical_groups(tmp_path, layout):
    manifest = corpus(tmp_path)
    expected = observations(masked(manifest))
    regroup_samples(manifest, 8, layout)
    dataset = masked(manifest)
    actual = observations(dataset)
    assert len(actual) == len(expected) > 0
    populated = {}
    for asset in dataset.assets:
        with pq.ParquetFile(dataset._file(asset, "samples")) as file:
            populated[f"{asset['market']}/{asset['symbol']}"] = _populated_groups(file)
    for left, right in zip(expected, actual, strict=True):
        key = left["sample_ids"][0].rsplit("/", 1)[0]
        assert right["source_group"] == populated[key][left["source_group"]]
        assert canonical({**right, "source_group": None}) == canonical(
            {**left, "source_group": None}
        )
    with pq.ParquetFile(dataset._file(dataset.assets[0], "samples")) as file:
        assert len(_populated_groups(file)) < file.num_row_groups


def views(folder, reader):
    """Recuentos, etiquetas y lotes de cada ventana mientras el padre conserva su huella."""
    result = {}
    for manifest in sorted(Path(folder).glob("*/manifest.json")):
        view = reader(manifest)
        labels = {
            f"{a['market']}/{a['symbol']}": pq.read_table(
                view.roots["labels"] / a["market"] / a["symbol"] / "labels.parquet"
            )
            for a in view.assets
        }
        batches = {
            partition: canonical(
                list(view.batches(partition=partition, batch_size=2, epoch=0, seed=3))
            )
            for partition in view.partitions
        }
        result[manifest.parent.name] = view.manifest["counts"], labels, batches
    assert result
    return result


def assert_same_views(expected, actual):
    assert expected.keys() == actual.keys()
    for fold, (counts, labels, batches) in expected.items():
        assert actual[fold][0] == counts
        assert labels.keys() == actual[fold][1].keys()
        assert all(table.equals(actual[fold][1][key]) for key, table in labels.items())
        assert actual[fold][2] == batches
        assert all(bool(batches[name]) == (count > 0) for name, count in counts.items())


def test_masked_view_with_an_empty_final_group_matches_the_compact_edition(tmp_path):
    fixture = historical_temporal_fixture(tmp_path / "source")
    regroup_samples(fixture.parent, 3)
    parent = masked(fixture.parent)
    state = _masked_sample_state(parent, parent.assets[0])
    prepare(fixture, tmp_path / "compact")
    expected = views(tmp_path / "compact", masked)
    sizes = regroup_samples(fixture.parent, 3, final_empty)
    assert list(sizes.values()) == [[3, 3, 3, 3, 1, 0]]
    parent = masked(fixture.parent)
    for left, right in zip(state, _masked_sample_state(parent, parent.assets[0]), strict=True):
        np.testing.assert_array_equal(left, right)
    assert len(prepare(fixture, tmp_path / "padded")["folds"]) == len(expected) == 10
    assert_same_views(expected, views(tmp_path / "padded", masked))


def test_strict_view_with_an_empty_final_group_matches_the_compact_edition(tmp_path):
    fixture = temporal_fixture(tmp_path / "strict")
    root = tmp_path / "strict"
    expected = views(fixture.views, CorpusDataset)
    sizes = regroup_samples(fixture.parent, 4, final_empty)
    assert all(groups[-1] == 0 and 0 not in groups[:-1] for groups in sizes.values())
    output = root / "padded-views"
    prepare_temporal_corpus(
        fixture.parent, fixture.protocol, root / "macro/macro.parquet", fixture.admission, output
    )
    assert_same_views(expected, views(output, CorpusDataset))


def test_strict_state_rejects_an_asset_whose_groups_have_no_rows(tmp_path):
    fixture = temporal_fixture(tmp_path / "strict")
    root = tmp_path / "strict"
    meta = json.loads(fixture.parent.read_text())
    asset = meta["assets"][0]
    path = Path(meta["roots"]["samples"]) / asset["market"] / asset["symbol"] / "samples.parquet"
    table = pq.read_table(path)
    with pq.ParquetWriter(path, table.schema) as writer:
        writer.write_table(table.slice(0, 0))
    asset["samples_sha256"] = sha256(path)
    atomic_json(fixture.parent, meta)
    with pytest.raises(ValueError, match="no contiene muestras"):
        prepare_temporal_corpus(
            fixture.parent,
            fixture.protocol,
            root / "macro/macro.parquet",
            fixture.admission,
            root / "empty-views",
        )


def test_label_preparation_reads_the_same_positions_with_empty_groups(tmp_path):
    manifest = corpus(tmp_path)
    dataset = masked(manifest)
    path = dataset._file(dataset.assets[0], "samples")
    calculated = pd.DataFrame(columns=["prediction_at", "reason", "target", "target_available_at"])

    def labels():
        audit = Counter()
        tables = list(_label_batches(path, calculated, audit, "original_audited"))
        return pa.concat_tables(tables), audit

    expected, expected_audit = labels()
    assert regroup(path, 8, [3, -(-expected.num_rows // 8)]).count(0) == 2
    actual, actual_audit = labels()
    assert actual.equals(expected) and actual_audit == expected_audit
    assert actual["sample_row"].to_pylist() == list(range(expected.num_rows))
