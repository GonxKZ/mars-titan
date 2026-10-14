"""Registros del índice de observaciones frente a la lectura grupo a grupo (#622)."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.memory import financial_observations as api
from mars_titan.training.corpus_inputs import CorpusDataset, _historical_times, _populated_groups
from tests.training.chronological_fixture import chronological_corpus, phases
from tests.training.empty_groups_fixture import final_empty, inner_empty, regroup, regroup_samples

LAYOUTS = [
    pytest.param(None, id="compacto"),
    pytest.param(final_empty, id="vacio-final"),
    pytest.param(inner_empty, id="vacios-intermedios"),
]


def grouped_records(dataset, phase):
    """La lectura anterior: `prediction_at` grupo a grupo y una tabla por grupo."""
    tables, total = [], 0
    for identity, asset in enumerate(api._assets(dataset)):
        path = dataset._file(asset, "samples")
        with pq.ParquetFile(path) as file:
            previous = None
            for group in _populated_groups(file):
                stamps = _historical_times(
                    file.read_row_group(group, columns=["prediction_at"], use_threads=False)
                )
                if np.any(np.diff(stamps) <= 0) or (previous is not None and stamps[0] <= previous):
                    raise ValueError("Las muestras necesitan orden temporal único")
                previous = stamps[-1]
                rows = np.flatnonzero(
                    (stamps >= phase.warmup_start) & (stamps < phase.decision_end)
                )
                total += len(rows)
                tables.append(
                    api._table(
                        dict(
                            event_at=stamps[rows],
                            kind=np.zeros(len(rows), np.int64),
                            asset_id=np.full(len(rows), identity, np.int64),
                            source_group=np.full(len(rows), group, np.int64),
                            source_row=rows.astype(np.int64),
                            sample_at=stamps[rows],
                        )
                    )
                )
            positions, stamps, _, maturity = dataset._labels(
                asset, phase.partition, file.metadata.num_rows
            )
            selected = np.flatnonzero(
                (stamps >= phase.decision_start)
                & (stamps < phase.decision_end)
                & (maturity < phase.decision_end)
            )
            tables.append(
                api._table(
                    dict(
                        event_at=maturity[selected],
                        kind=np.ones(len(selected), np.int64),
                        asset_id=np.full(len(selected), identity, np.int64),
                        source_group=np.full(len(selected), -1, np.int64),
                        source_row=positions[selected],
                        sample_at=stamps[selected],
                    )
                )
            )
    close = dict(event_at=[phase.close_at], kind=[2], asset_id=[-1])
    close |= dict(source_group=[-1], source_row=[-1], sample_at=[-1])
    tables.append(api._table(close))
    return pa.concat_tables(tables)


def corpus(tmp_path, layout=None):
    manifest = chronological_corpus(tmp_path / "corpus", group_size=4)
    regroup_samples(manifest, 4, layout)
    return manifest


def dataset(manifest):
    return CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)


def rewrite_stamps(manifest, change):
    """Cambiar `prediction_at` del primer activo con `change` y actualizar su huella."""
    meta = json.loads(Path(manifest).read_text())
    asset = meta["assets"][0]
    path = Path(meta["roots"]["samples"]) / asset["market"] / asset["symbol"] / "samples.parquet"
    table = pq.read_table(path)
    column = table["prediction_at"]
    values = change(column.cast(pa.int64()).to_numpy().copy())
    index = table.schema.get_field_index("prediction_at")
    table = table.set_column(index, "prediction_at", pa.array(values).cast(column.type))
    pq.write_table(table, path)
    regroup(path, 4)
    asset["samples_sha256"] = sha256(path)
    Path(manifest).write_text(json.dumps(meta))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("raw_rows", [1, 7, 1 << 18])
def test_records_keep_the_rows_and_order_of_the_grouped_reading(
    tmp_path, monkeypatch, layout, raw_rows
):
    manifest = corpus(tmp_path, layout)
    monkeypatch.setattr(api, "_RAW_ROWS", raw_rows)
    reader = dataset(manifest)
    for phase in phases():
        tables = list(api._records(reader, phase))
        assert pa.concat_tables(tables).equals(grouped_records(reader, phase)), phase
        # Cada tabla salvo la última reúne al menos `_RAW_ROWS` filas y ninguna está vacía.
        assert all(table.num_rows >= raw_rows for table in tables[:-1])
        assert all(table.num_rows for table in tables)
        assert tables[-1].column("kind").to_pylist()[-1] == 2


def test_inner_empty_groups_shift_the_physical_group_of_later_rows(tmp_path):
    compact, padded = corpus(tmp_path / "compacto"), corpus(tmp_path / "vacios", inner_empty)
    phase = phases()[0]
    left = pa.concat_tables(api._records(dataset(compact), phase)).to_pydict()
    right = pa.concat_tables(api._records(dataset(padded), phase)).to_pydict()
    samples = [i for i, kind in enumerate(left["kind"]) if kind == 0]
    assert samples and left["source_row"] == right["source_row"]
    # `inner_empty` inserta dos grupos vacíos antes del segundo grupo con filas.
    for i in samples:
        assert right["source_group"][i] == left["source_group"][i] + (
            2 if left["source_group"][i] else 0
        )


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda v: np.concatenate([v[:1], v[:1], v[2:]]), id="repetida-en-un-grupo"),
        pytest.param(lambda v: np.concatenate([v[:4], v[3:4], v[5:]]), id="repetida-entre-grupos"),
        pytest.param(lambda v: np.concatenate([v[:4], v[2:3], v[5:]]), id="atras-entre-grupos"),
    ],
)
def test_records_reject_samples_out_of_order(tmp_path, change):
    manifest = corpus(tmp_path)
    rewrite_stamps(manifest, change)
    reader = dataset(manifest)
    for read in (api._records, grouped_records):
        with pytest.raises(ValueError, match="orden temporal único"):
            list(read(reader, phases()[0]))


def test_records_reject_samples_outside_the_historical_cut(tmp_path):
    manifest = corpus(tmp_path)
    before = np.datetime64("1999-12-31", "us").astype(np.int64)
    rewrite_stamps(manifest, lambda v: np.concatenate([[before], v[1:]]))
    with pytest.raises(ValueError, match="fuera del corte 2000"):
        list(api._records(dataset(manifest), phases()[0]))


@pytest.mark.parametrize("kind", [0, 1])
def test_records_stop_at_the_record_limit(tmp_path, monkeypatch, kind):
    manifest = corpus(tmp_path)
    reader, phase = dataset(manifest), phases()[0]
    rows = pa.concat_tables(api._records(reader, phase)).to_pydict()
    first = [k for a, k in zip(rows["asset_id"], rows["kind"], strict=True) if a == 0]
    assert first.count(0) and first.count(1)
    # El límite cae justo en las muestras o en las maduraciones del primer activo, y la
    # lectura se detiene ahí: sin leer sus etiquetas o sin pasar al segundo activo.
    monkeypatch.setattr(api, "_MAX_ROWS", first.count(0) + kind)
    read, labels = [], reader._labels
    samples = api._asset_samples
    monkeypatch.setattr(api, "_asset_samples", lambda *a: read.append(a[2]) or samples(*a))
    monkeypatch.setattr(reader, "_labels", lambda *a: read.append("etiquetas") or labels(*a))
    with pytest.raises(ValueError, match="número máximo de registros"):
        list(api._records(reader, phase))
    assert read == ([0] if kind == 0 else [0, "etiquetas"])
