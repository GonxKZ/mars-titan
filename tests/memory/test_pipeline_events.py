"""Eventos cronológicos idénticos bit a bit con y sin la tubería de lectura."""

import pytest

from mars_titan.memory import financial_observations as api
from mars_titan.training.input_pipeline import PipelineOptions
from tests.training.chronological_fixture import phases
from tests.training.empty_groups_fixture import canonical
from tests.training.test_pipeline_batches import (
    ABLATIONS,
    CORRUPTIONS,
    PIPELINES,
    SEQUENTIAL,
    corpus,
    dataset,
)


def sources(tmp_path, pipeline, *, corrupt=None, ablation=None, **options):
    """Índices de cada fase preparados con la ruta secuencial y leídos con `pipeline`."""
    manifest = corpus(tmp_path, **options)
    if corrupt is not None:
        corrupt(manifest, "A0001", 7)
    prepared = dataset(manifest, SEQUENTIAL, ablation)
    reader = dataset(manifest, pipeline, ablation)
    result = {}
    for phase in phases():
        index = api.prepare_observation_index(
            prepared, tmp_path / f"index-{phase.partition}", phase=phase
        )
        result[phase.partition] = api.FinancialObservationSource(reader, index)
    return reader, result


def flatten(events):
    rows = []
    for event in events:
        rows.append(("event", event.at, event.labels, event.close_phase, len(event.inputs)))
        for batch in event.inputs:
            rows.append(canonical(batch))
    return rows


def per_row(events):
    rows = []
    for event in events:
        rows.append(("event", event.at, event.labels, event.close_phase))
        for batch in event.inputs:
            for index, identity in enumerate(batch["sample_ids"]):
                rows.append(
                    (
                        identity,
                        batch["market"][index],
                        batch["prediction_at"][index].tobytes(),
                        batch["input_available_at"][index].tobytes(),
                        batch["presence"][index].tobytes(),
                        {k: v[index].tobytes() for k, v in batch["inputs"].items()},
                    )
                )
    return rows


@pytest.mark.parametrize("pipeline", [SEQUENTIAL, *[p.values[0] for p in PIPELINES]])
@pytest.mark.parametrize("block_rows", [1, 2, 256])
def test_batched_events_match_the_per_observation_reader(tmp_path, pipeline, block_rows):
    _, streams = sources(tmp_path, pipeline, group_size=3)
    for source in streams.values():
        expected = per_row(source.events())
        assert per_row(source.batched_events(block_rows=block_rows)) == expected


@pytest.mark.parametrize("ablation", ABLATIONS)
def test_ablated_events_match_the_per_observation_reader(tmp_path, ablation):
    _, plain = sources(tmp_path / "plain", SEQUENTIAL, group_size=3)
    for index, options in enumerate([SEQUENTIAL, *(p.values[0] for p in PIPELINES)]):
        _, streams = sources(tmp_path / str(index), options, ablation=ablation, group_size=3)
        for name, source in streams.items():
            expected = per_row(source.events())
            assert expected != per_row(plain[name].events())
            assert per_row(source.batched_events(block_rows=2)) == expected


@pytest.mark.parametrize("pipeline", [p.values[0] for p in PIPELINES])
def test_pipeline_events_and_cursors_are_identical(tmp_path, pipeline):
    _, sequential = sources(tmp_path / "a", SEQUENTIAL, group_size=3)
    _, piped = sources(tmp_path / "b", pipeline, group_size=3)
    for name, source in sequential.items():
        complete = list(source.batched_events(block_rows=2))
        assert flatten(piped[name].batched_events(block_rows=2)) == flatten(complete)
        for cursor in (0, 3, len(complete) - 1, len(complete)):
            assert flatten(
                piped[name].batched_events(start_cursor=cursor, block_rows=2)
            ) == flatten(complete[cursor:])


@pytest.mark.parametrize("group_size", [3, 8])
def test_lookahead_decodes_each_group_once_and_the_reader_reports_it(
    tmp_path, monkeypatch, group_size
):
    # Con grupos mayores que el adelanto, el grupo pedido ya está en la caché.
    pipeline = PipelineOptions(decode_workers=2, prefetch_batches=2)
    reader_dataset, streams = sources(tmp_path, pipeline, group_size=group_size)
    source = streams["train"]
    calls, original = [], reader_dataset._sample_group
    monkeypatch.setattr(
        reader_dataset,
        "_sample_group",
        lambda asset, file, group: (
            calls.append((asset["symbol"], group)) or original(asset, file, group)
        ),
    )
    events = list(source.batched_events(block_rows=2))
    assert len(calls) == len(set(calls))
    reader = source.last_reader
    visited = {
        (identity, group)
        for _, rows in source._logical_events(0)
        for _, kind, identity, group, *_ in rows
        if kind == 0
    }
    assert reader.decoded_groups == len(visited) and reader.redecoded_groups == 0
    assert reader._pending == {}
    assert per_row(events) == per_row(source.events())


@pytest.mark.parametrize("pipeline", [SEQUENTIAL, PipelineOptions(decode_workers=2)])
def test_undersized_group_cache_redecodes_without_changing_the_events(tmp_path, pipeline):
    _, streams = sources(tmp_path, pipeline, assets=4, group_size=3)
    source = streams["train"]
    expected = per_row(source.events())
    assert per_row(source.batched_events(block_rows=2, max_cached_bytes=1024**2)) == expected
    assert source.last_reader.redecoded_groups == 0
    events = source.batched_events(block_rows=2, max_cached_bytes=1024**2)
    first = next(events)
    reader = source.last_reader
    reader.limit = 1
    assert per_row([first, *events]) == expected
    assert reader.redecoded_groups > 0 and len(reader.groups) == 1


def test_a_cache_smaller_than_the_active_assets_still_reuses_groups(tmp_path):
    # Un grupo por activo: cada instante pide los mismos seis grupos en el mismo orden.
    _, streams = sources(tmp_path, SEQUENTIAL, assets=6, group_size=128)
    source = streams["train"]
    expected = per_row(source.events())
    events = source.batched_events(block_rows=1, max_cached_bytes=1024**2)
    first = next(events)
    reader = source.last_reader
    # Sitio para cuatro de los seis grupos, antes de que entre el último activo.
    assert len(reader.groups) < 6
    reader.limit = 4 * max(entry[2] for entry in reader.groups.values())
    rest = list(events)
    assert per_row([first, *rest]) == expected
    requests = sum(len(batch["sample_ids"]) for event in rest for batch in event.inputs)
    # Descartar siempre el más antiguo volvería a decodificar casi todos los grupos pedidos.
    assert requests >= 24 and 0 < reader.redecoded_groups <= 0.6 * requests
    assert reader.cached_bytes <= reader.limit


def test_absent_assets_leave_the_cache_before_the_recent_ones():
    reader = api._BlockReader(None, 1, 1024**2)
    reader.limit, reader.event = 300, 40
    # El activo 0 no aparece desde hace más de STALE_EVENTS instantes.
    for identity, used in ((0, 40 - reader.STALE_EVENTS - 1), (1, 39), (2, 40)):
        reader.groups[identity] = (0, None, 100)
        reader._used[identity] = used
    reader.cached_bytes = 300
    reader._evict(100)
    assert list(reader.groups) == [1, 2] and reader.cached_bytes == 200
    # Sin activos ausentes, sale el usado más recientemente.
    reader._evict(200)
    assert list(reader.groups) == [1] and reader.cached_bytes == 100


def delivered_before_error(events):
    received = []
    try:
        for event in events:
            received.append(event)
    except ValueError as error:
        return per_row(received), str(error)
    return per_row(received), None


@pytest.mark.parametrize("corrupt", CORRUPTIONS)
def test_corrupt_group_fails_in_the_same_event(tmp_path, corrupt):
    _, streams = sources(tmp_path / "seq", SEQUENTIAL, corrupt=corrupt, assets=4)
    expected = delivered_before_error(streams["train"].events())
    assert expected[1] is not None and any(row[0] == "event" for row in expected[0])
    for index, options in enumerate([SEQUENTIAL, *(p.values[0] for p in PIPELINES)]):
        _, streams = sources(tmp_path / str(index), options, corrupt=corrupt, assets=4)
        assert delivered_before_error(streams["train"].batched_events(block_rows=2)) == expected
