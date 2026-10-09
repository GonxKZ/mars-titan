"""Lotes, cursores y eventos idénticos bit a bit con y sin la tubería de lectura."""

import json
import subprocess
import types
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.training import corpus_inputs
from mars_titan.training.corpus_inputs import CorpusDataset, _price_contexts, _window_contexts
from mars_titan.training.input_pipeline import PipelineOptions
from tests.training.chronological_fixture import chronological_corpus
from tests.training.empty_groups_fixture import canonical, inner_empty, regroup_samples

SEQUENTIAL = PipelineOptions()
PIPELINES = [
    pytest.param(PipelineOptions(decode_workers=1), id="1hilo"),
    pytest.param(PipelineOptions(decode_workers=3, prefetch_batches=2), id="3hilos-prefetch"),
    pytest.param(PipelineOptions(prefetch_batches=1), id="solo-prefetch"),
    pytest.param(PipelineOptions(decode_workers=8, prefetch_batches=16), id="8hilos"),
]
# Revisión anterior a la tubería: su lector es la referencia de la ruta secuencial.
REFERENCE = "14c6b1dd"
ABLATIONS = ["mask_news", "mask_fundamentals", "mask_news_and_fundamentals"]
ROOT = Path(__file__).resolve().parents[2]


def dataset(manifest, pipeline, ablation=None):
    return CorpusDataset(
        manifest, input_policy=HISTORICAL_MASKED, pipeline=pipeline, modality_ablation=ablation
    )


def corpus(tmp_path, **options):
    return chronological_corpus(tmp_path / "corpus", **({"group_size": 4} | options))


def stream(reader, partition, batch_size, epoch, seed, cursor=None):
    return canonical(
        list(
            reader.batches(
                partition=partition, batch_size=batch_size, epoch=epoch, seed=seed, cursor=cursor
            )
        )
    )


def passes(reader):
    return [
        stream(reader, partition, batch_size, epoch, seed)
        for partition in ("train", "validation")
        for batch_size in (1, 3, 7, 4096)
        for epoch in (0, 3)
        for seed in (0, 42)
    ]


def reference_reader():
    """Lector de la revisión anterior, cargado desde git sin tocar el árbol de trabajo."""
    try:
        source = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{REFERENCE}:src/mars_titan/training/corpus_inputs.py",
            ],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("La revisión de referencia no está disponible en este clon")
    module = types.ModuleType("mars_titan.training._reader_before_pipeline")
    module.__package__ = "mars_titan.training"
    module.__file__ = f"{REFERENCE}:corpus_inputs.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102
    return module.CorpusDataset


def test_sequential_route_matches_the_reader_before_the_pipeline(tmp_path):
    manifest = corpus(tmp_path)
    regroup_samples(manifest, 4, inner_empty)
    before = reference_reader()(manifest, input_policy=HISTORICAL_MASKED)
    assert passes(dataset(manifest, SEQUENTIAL)) == passes(before)


@pytest.mark.parametrize("ablation", ABLATIONS)
def test_ablated_reading_is_identical_to_the_reader_before_the_pipeline(tmp_path, ablation):
    manifest = corpus(tmp_path)
    regroup_samples(manifest, 4, inner_empty)
    before = reference_reader()(
        manifest, input_policy=HISTORICAL_MASKED, modality_ablation=ablation
    )
    expected = passes(before)
    assert expected != passes(dataset(manifest, SEQUENTIAL))
    for options in [SEQUENTIAL, *(p.values[0] for p in PIPELINES)]:
        assert passes(dataset(manifest, options, ablation)) == expected


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_pipeline_batches_and_cursors_are_identical(tmp_path, pipeline):
    manifest = corpus(tmp_path)
    regroup_samples(manifest, 4, inner_empty)
    assert passes(dataset(manifest, pipeline)) == passes(dataset(manifest, SEQUENTIAL))


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_resuming_from_every_cursor_gives_the_same_suffix(tmp_path, pipeline):
    manifest = corpus(tmp_path)
    sequential, piped = dataset(manifest, SEQUENTIAL), dataset(manifest, pipeline)
    for partition in ("train", "validation"):
        complete = list(sequential.batches(partition=partition, batch_size=3, epoch=1, seed=7))
        assert len(complete) > 3
        for index, batch in enumerate(complete):
            resumed = stream(piped, partition, 3, 1, 7, batch["confirmed_cursor"])
            assert resumed == canonical(complete[index + 1 :])


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_abandoned_pass_does_not_disturb_the_next_one(tmp_path, pipeline):
    reader = dataset(corpus(tmp_path), pipeline)
    expected = stream(reader, "train", 2, 0, 42)
    for taken in (1, 2, 5):
        iterator = reader.batches(partition="train", batch_size=2, epoch=0, seed=42)
        assert canonical([next(iterator) for _ in range(taken)]) == expected[:taken]
        iterator.close()
    assert stream(reader, "train", 2, 0, 42) == expected


def rewrite_samples(manifest, symbol, change):
    """Reescribir las muestras de un activo con sus grupos y su huella actualizada."""
    meta = json.loads(Path(manifest).read_text())
    asset = next(a for a in meta["assets"] if a["symbol"] == symbol)
    path = Path(meta["roots"]["samples"]) / asset["market"] / symbol / "samples.parquet"
    table = pq.read_table(path)
    groups = pq.ParquetFile(path).metadata.num_row_groups
    size = -(-table.num_rows // groups)
    pq.write_table(change(table), path, row_group_size=size)
    asset["samples_sha256"] = sha256(path)
    Path(manifest).write_text(json.dumps(meta))


def _replace(table, name, column):
    return table.set_column(table.schema.get_field_index(name), name, column)


def corrupt_vector(manifest, symbol, row):
    """Poner un valor no finito en una fila con presencia. Falla al copiar el lote."""

    def change(table):
        values = table["charts"].combine_chunks().flatten().to_numpy().copy()
        width = len(values) // table.num_rows
        values[row * width] = np.nan
        column = pa.FixedSizeListArray.from_arrays(pa.array(values, type=pa.float32()), width)
        return _replace(table, "charts", column)

    rewrite_samples(manifest, symbol, change)


def delay_availability(manifest, symbol, row):
    """Publicar los precios de una fila una hora después de su predicción."""

    def change(table):
        column = table["input_availability"].combine_chunks()
        values = column.to_pylist()
        values[row] = values[row] | dict(prices=values[row]["prices"] + timedelta(hours=1))
        return _replace(table, "input_availability", pa.array(values, type=column.type))

    rewrite_samples(manifest, symbol, change)


def misplace_price_end(manifest, symbol, row):
    """Situar la ventana de precios de una fila antes del contexto. Falla al montar bloques."""

    def change(table):
        ends = table["price_end_index"].to_numpy().copy()
        ends[row] = 0
        return _replace(
            table, "price_end_index", pa.array(ends, type=table["price_end_index"].type)
        )

    rewrite_samples(manifest, symbol, change)


CORRUPTIONS = [
    pytest.param(corrupt_vector, id="vector"),
    pytest.param(delay_availability, id="disponibilidad"),
    pytest.param(misplace_price_end, id="ventana"),
]


def delivered_before_error(reader, partition, batch_size):
    received = []
    try:
        for batch in reader.batches(partition=partition, batch_size=batch_size, epoch=0, seed=3):
            received.append(batch)
    except ValueError as error:
        return canonical(received), str(error)
    return canonical(received), None


@pytest.mark.parametrize("corrupt", CORRUPTIONS)
def test_a_corrupt_group_fails_at_the_same_position(tmp_path, corrupt):
    manifest = corpus(tmp_path, assets=4)
    corrupt(manifest, "A0002", 5)
    before = reference_reader()(manifest, input_policy=HISTORICAL_MASKED)
    readers = [dataset(manifest, SEQUENTIAL), *(dataset(manifest, p.values[0]) for p in PIPELINES)]
    failures = 0
    for partition, size in (("train", 1), ("train", 3), ("validation", 2)):
        expected = delivered_before_error(before, partition, size)
        failures += expected[1] is not None
        for reader in readers:
            assert delivered_before_error(reader, partition, size) == expected
    assert failures


def test_pipeline_decodes_at_most_its_lookahead_beyond_the_consumer(tmp_path, monkeypatch):
    manifest = corpus(tmp_path, assets=6, group_size=4)
    pipeline = PipelineOptions(decode_workers=2, prefetch_batches=1)
    reader = dataset(manifest, pipeline)
    started, original = [], reader._asset_blocks

    def counted(item):
        started.append(item[0][0]["cursor"][0])
        return original(item)

    monkeypatch.setattr(reader, "_asset_blocks", counted)
    consumed, ahead = set(), []
    for batch in reader.batches(partition="train", batch_size=1, epoch=0, seed=0):
        consumed.add(batch["confirmed_cursor"]["asset"])
        ahead.append(len(started) - len(consumed))
    # Activos decodificados por adelantado, más el que retiene el productor.
    assert max(ahead) <= pipeline.decode_workers + 1 + 1
    assert len(started) == len(consumed)


@pytest.mark.parametrize("options", [SEQUENTIAL, PipelineOptions(decode_workers=2)])
def test_each_asset_is_read_once_and_the_group_reader_stays_unused(tmp_path, monkeypatch, options):
    manifest = corpus(tmp_path, assets=3)
    regroup_samples(manifest, 4, inner_empty)
    expected = stream(dataset(manifest, options), "train", 3, 0, 1)
    reader, reads = dataset(manifest, options), []
    original = pq.ParquetFile.read_row_groups

    def recorded(self, groups, **options):
        reads.append(tuple(groups))
        return original(self, groups, **options)

    monkeypatch.setattr(pq.ParquetFile, "read_row_groups", recorded)
    monkeypatch.setattr(reader, "_group_blocks", None)
    assert stream(reader, "train", 3, 0, 1) == expected
    assert len(reads) == len(reader.assets) and all(r == tuple(sorted(r)) for r in reads)


def test_window_contexts_match_each_asset_window():
    rng = np.random.default_rng(5)
    prices = [rng.uniform(1, 100, size=(80 + i, 5)) for i in range(6)]
    for table in prices:
        table[:, 4] = np.where(rng.random(len(table)) < 0.2, 0, table[:, 4])
    ends = np.array([63, 70, 79, 64, 66, 84])
    observed = _window_contexts(prices, ends, 64)
    expected = np.concatenate(
        [_price_contexts(t, ends[i : i + 1], 64) for i, t in enumerate(prices)]
    )
    assert observed.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(observed, expected)
    mixed = [table.astype(np.float32) if i % 2 else table for i, table in enumerate(prices)]
    np.testing.assert_array_equal(
        _window_contexts(mixed, ends, 64),
        np.concatenate([_price_contexts(t, ends[i : i + 1], 64) for i, t in enumerate(mixed)]),
    )


@pytest.mark.parametrize("problem", ["early", "late", "nan", "zero", "float", "width"])
def test_window_contexts_reject_the_same_invalid_windows(problem):
    prices = [np.ones((70, 5)), np.ones((70, 5))]
    ends = np.array([63, 69])
    if problem == "early":
        ends[0] = 62
    elif problem == "late":
        ends[1] = 70
    elif problem == "nan":
        prices[1][40, 1] = np.nan
    elif problem == "zero":
        prices[0][10, 3] = 0
    elif problem == "float":
        ends = ends.astype(float)
    else:
        prices[0] = np.ones((70, 4))
    with pytest.raises(ValueError):
        _window_contexts(prices, ends, 64)


def test_changed_file_is_hashed_again_despite_the_process_digests(tmp_path):
    manifest = corpus(tmp_path)
    reader = dataset(manifest, SEQUENTIAL)
    asset = reader.assets[0]
    path = reader._file(asset, "samples")
    assert any(key[0] == str(path) for key in corpus_inputs._DIGESTS)
    content = path.read_bytes()
    path.write_bytes(content[:-9] + b"X" + content[-8:])
    with pytest.raises(ValueError):
        reader._file(asset, "samples")
    with pytest.raises(ValueError):
        dataset(manifest, SEQUENTIAL)


def test_parallel_hashing_reports_the_first_invalid_asset_in_order(tmp_path, monkeypatch):
    manifest = corpus(tmp_path, assets=4)
    meta = json.loads(Path(manifest).read_text())
    for asset in meta["assets"][1:]:
        asset["labels_sha256"] = "0" * 64
    Path(manifest).write_text(json.dumps(meta))
    checked = []
    original = CorpusDataset._file

    def recorded(self, asset, kind):
        checked.append((asset["symbol"], kind))
        return original(self, asset, kind)

    monkeypatch.setattr(CorpusDataset, "_file", recorded)
    with pytest.raises(ValueError, match="ha cambiado"):
        dataset(manifest, SEQUENTIAL)
    assert checked[-1] == (meta["assets"][1]["symbol"], "labels")


def test_shared_digests_spare_rehashing_and_still_detect_changes(tmp_path, monkeypatch):
    manifest = corpus(tmp_path, assets=3)
    shared = tmp_path / "shared" / "file-digests.json"
    monkeypatch.setenv(corpus_inputs.DIGEST_CACHE_ENV, str(shared))
    monkeypatch.setattr(corpus_inputs, "_DIGESTS", {})
    dataset(manifest, SEQUENTIAL)
    assert shared.is_file()
    # Otro proceso: memoria vacía. Las huellas compartidas evitan leer los archivos.
    monkeypatch.setattr(corpus_inputs, "_DIGESTS", {})
    hashed, original = [], corpus_inputs.sha256
    monkeypatch.setattr(corpus_inputs, "sha256", lambda path: hashed.append(path) or original(path))
    reader = dataset(manifest, SEQUENTIAL)
    assert hashed == []
    # Un archivo reescrito cambia de firma: se lee de nuevo y se rechaza.
    path = reader._file(reader.assets[0], "samples")
    content = path.read_bytes()
    path.write_bytes(content[:-9] + b"X" + content[-8:])
    monkeypatch.setattr(corpus_inputs, "_DIGESTS", {})
    with pytest.raises(ValueError, match="ha cambiado"):
        dataset(manifest, SEQUENTIAL)
    assert path in hashed


def test_an_unreadable_shared_digest_file_only_costs_a_rehash(tmp_path, monkeypatch):
    manifest = corpus(tmp_path, assets=2)
    shared = tmp_path / "file-digests.json"
    shared.write_text("{no es json")
    monkeypatch.setenv(corpus_inputs.DIGEST_CACHE_ENV, str(shared))
    monkeypatch.setattr(corpus_inputs, "_DIGESTS", {})
    expected = stream(dataset(manifest, SEQUENTIAL), "train", 3, 0, 1)
    assert json.loads(shared.read_text())["kind"] == "mars_titan_file_digests"
    monkeypatch.setenv(corpus_inputs.DIGEST_CACHE_ENV, "relativa.json")
    with pytest.raises(ValueError, match="ruta absoluta"):
        dataset(manifest, SEQUENTIAL)
    monkeypatch.delenv(corpus_inputs.DIGEST_CACHE_ENV)
    assert stream(dataset(manifest, SEQUENTIAL), "train", 3, 0, 1) == expected
