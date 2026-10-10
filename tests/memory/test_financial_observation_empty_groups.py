"""Índice de observaciones con grupos Parquet vacíos en las muestras históricas."""

import json

import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory import financial_observations as api
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.chronological_fixture import chronological_corpus, phases
from tests.training.empty_groups_fixture import (
    canonical,
    final_empty,
    inner_empty,
    regroup_samples,
)


def corpus(tmp_path):
    manifest = chronological_corpus(tmp_path / "corpus", group_size=4)
    regroup_samples(manifest, 4)
    return manifest


def dataset(manifest):
    return CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)


def records(manifest):
    return [
        [table.to_pydict() for table in api._records(dataset(manifest), phase)]
        for phase in phases()
    ]


def streams(manifest, folder):
    reader, result = dataset(manifest), {}
    for phase in phases():
        path = api.prepare_observation_index(reader, folder / phase.partition, phase=phase)
        stream = api.FinancialObservationSource(reader, path)
        events = [[e.at, e.inputs, e.labels, e.close_phase] for e in stream.events()]
        blocks = [[e.at, e.inputs, e.labels, e.close_phase] for e in stream.batched_events()]
        metadata = json.loads(path.read_text())
        result[phase.partition] = (
            canonical(events),
            canonical(blocks),
            metadata["groups"],
            metadata["events_sha256"],
        )
    return result


def test_index_records_match_the_compact_file(tmp_path):
    manifest = corpus(tmp_path)
    expected = records(manifest)
    regroup_samples(manifest, 4, final_empty)
    assert records(manifest) == expected


@pytest.mark.parametrize(
    "layout", [pytest.param(final_empty, id="final"), pytest.param(inner_empty, id="intermedio")]
)
def test_events_and_blocks_match_the_compact_file(tmp_path, layout):
    manifest = corpus(tmp_path)
    expected = streams(manifest, tmp_path / "compact")
    sizes = regroup_samples(manifest, 4, layout)
    assert all(groups.count(0) == len(layout(0)) for groups in sizes.values())
    actual = streams(manifest, tmp_path / "padded")
    for partition, (events, blocks, groups, digest) in expected.items():
        assert actual[partition][:3] == (events, blocks, groups)
        # El índice guarda el grupo físico. Solo el caso real conserva sus bytes.
        assert (actual[partition][3] == digest) == (layout is final_empty)
