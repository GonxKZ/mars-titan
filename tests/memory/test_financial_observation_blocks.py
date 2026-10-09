"""Lectura por bloques de activos con el mismo conjunto y orden que la lectura por fila."""

import numpy as np
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.memory import financial_observations as api
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.chronological_fixture import chronological_corpus, phases


def sources(tmp_path, **options):
    dataset = CorpusDataset(
        chronological_corpus(tmp_path / "corpus", **options), input_policy=HISTORICAL_MASKED
    )
    result = {}
    for phase in phases():
        manifest = api.prepare_observation_index(
            dataset, tmp_path / f"index-{phase.partition}", phase=phase
        )
        result[phase.partition] = api.FinancialObservationSource(dataset, manifest)
    return dataset, result


def flatten(event):
    rows = []
    for batch in event.inputs:
        for index, identity in enumerate(batch["sample_ids"]):
            rows.append(
                (
                    identity,
                    batch["market"][index],
                    int(batch["prediction_at"][index].astype(np.int64)),
                    int(batch["input_available_at"][index].astype(np.int64)),
                    batch["presence"][index].tobytes(),
                    {name: value[index].tobytes() for name, value in batch["inputs"].items()},
                )
            )
    return rows


def assert_same(expected, actual):
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual, strict=True):
        assert (left.at, left.labels, left.close_phase) == (
            right.at,
            right.labels,
            right.close_phase,
        )
        assert flatten(left) == flatten(right)


@pytest.mark.parametrize("block_rows", [1, 2, 256])
def test_blocks_preserve_rows_order_bytes_and_labels(tmp_path, block_rows):
    _, streams = sources(tmp_path)
    for stream in streams.values():
        expected = list(stream.events())
        assert all(len(b["sample_ids"]) == 1 for event in expected for b in event.inputs)
        actual = list(stream.batched_events(block_rows=block_rows))
        assert_same(expected, actual)
        for event in actual:
            sizes = [len(batch["sample_ids"]) for batch in event.inputs]
            assert len(sizes) == -(-sum(sizes) // block_rows)
            assert all(size == block_rows for size in sizes[:-1])
            identities = [i for batch in event.inputs for i in batch["sample_ids"]]
            assert identities == sorted(identities)
            assert all({"target", "target_available_at"}.isdisjoint(b) for b in event.inputs)
        assert any(len(event.inputs) > 1 for event in actual) == (block_rows < 3)
        assert any(event.inputs and event.labels for event in actual)


def test_cursor_resumes_the_same_suffix(tmp_path):
    _, streams = sources(tmp_path)
    stream = streams["train"]
    complete = list(stream.batched_events(block_rows=2))
    for cursor in (0, 5, len(complete) - 1, len(complete)):
        assert_same(
            complete[cursor:], list(stream.batched_events(start_cursor=cursor, block_rows=2))
        )


def test_each_group_and_label_file_is_decoded_once_per_pass(tmp_path, monkeypatch):
    dataset, streams = sources(tmp_path, group_size=4)
    stream = streams["validation"]
    calls, labels = [], []
    decode, read = dataset._sample_group, dataset._labels
    monkeypatch.setattr(
        dataset, "_sample_group", lambda a, f, g: calls.append((a["symbol"], g)) or decode(a, f, g)
    )
    monkeypatch.setattr(
        dataset, "_labels", lambda a, p, n: labels.append(a["symbol"]) or read(a, p, n)
    )
    events = list(stream.batched_events(block_rows=3))
    assert len(calls) == len(set(calls))
    rows = sum(len(b["sample_ids"]) for e in events for b in e.inputs)
    assert len(calls) < rows
    assert sorted(labels) == sorted(set(labels))
    calls.clear()
    list(stream.events())
    assert len(calls) == rows


def test_eviction_redecodes_without_changing_the_stream(tmp_path):
    _, streams = sources(tmp_path, group_size=4)
    stream = streams["train"]
    reader = api._BlockReader(stream, 2, 1024**2)
    reader.limit = 1
    evicted, visited = [], set()
    for at, rows in stream._logical_events(0):
        visited |= {(identity, group) for _, kind, identity, group, *_ in rows if kind == 0}
        evicted.append(stream._event(at, rows, reader))
    assert_same(list(stream.events()), evicted)
    assert len(reader.groups) == 1 and reader.decoded_groups > len(visited)


@pytest.mark.parametrize(
    "options",
    [dict(block_rows=0), dict(block_rows=257), dict(block_rows=True), dict(max_cached_bytes=1)],
)
def test_invalid_block_or_cache_budget_is_rejected_before_reading(tmp_path, options):
    _, streams = sources(tmp_path)
    with pytest.raises(ValueError):
        next(streams["train"].batched_events(**options))


def test_changed_sample_file_is_rejected_by_the_block_reader(tmp_path):
    dataset, streams = sources(tmp_path)
    path = dataset._file(dataset.assets[1], "samples")
    path.write_bytes(path.read_bytes() + b"changed")
    assert sha256(path) != dataset.assets[1]["samples_sha256"]
    with pytest.raises(ValueError):
        list(streams["train"].batched_events(block_rows=2))
