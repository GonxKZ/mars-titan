"""Validación de etiquetas por columnas, idéntica a la lectura fila a fila anterior (#625)."""

import subprocess
import threading
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.training import corpus_inputs
from mars_titan.training.cohort_contract import validate_cohort_rows
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.temporal_corpus import prepare_temporal_corpus
from tests.training.historical_temporal_fixture import historical_temporal_fixture

# Revisión de develop anterior a #625: su `_labels` recorre motivos, particiones y cohortes
# fila a fila en Python y es la referencia de los resultados y de los mensajes de error.
REFERENCE = "dd0a5d68"
ROOT = Path(__file__).resolve().parents[2]
COHORT = "us_historical_masked"
BASE = np.datetime64("2021-01-01", "D")
# Inicio del tramo reservado de la vista temporal de prueba, como `final_test_start`.
RESERVED = np.datetime64("2023-10-01", "us")
SPLITS = {
    False: ("train", "validation"),
    True: ("train", "validation", "calibration", "evaluation"),
}
ERRORS = {
    "Hay filas ausentes o de otra cohorte",
    "Una etiqueta no corresponde a una posición del archivo de muestras",
    "Hay etiquetas duplicadas para la misma muestra o decisión",
    "Falta una etiqueta o una exclusión explícita de una muestra",
    "Hay un motivo de exclusión de etiqueta desconocido",
    "Una etiqueta excluida no puede asignarse a una partición",
    "La partición no concilia con sus etiquetas",
    "La etiqueta debe madurar después de su predicción",
    "Hay etiquetas ausentes o no finitas",
    "Una etiqueta cruza la partición o sus recuentos no concilian",
}


def reference_module(name="corpus_inputs"):
    try:
        source = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{REFERENCE}:src/mars_titan/training/{name}.py",
            ],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(
            f"La revisión de referencia {REFERENCE} no está en este clon. Hace falta un clon con "
            "el historial completo de develop (git fetch --unshallow)"
        )
    module = types.ModuleType(f"mars_titan.training._{name}_before_625")
    module.__package__ = "mars_titan.training"
    module.__file__ = f"{REFERENCE}:{name}.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def bucket(prediction, maturity, temporal):
    """Partición esperada de cada fila, o None si la etiqueta cruza la frontera."""
    years = [
        np.asarray(t).astype("datetime64[Y]").astype(int) + 1970 for t in (prediction, maturity)
    ]
    if not temporal:
        names = [np.where(y <= 2022, "train", np.where(y == 2023, "validation", "")) for y in years]
    else:
        halves = [
            np.asarray(t).astype("datetime64[M]").astype(int) % 12 >= 6
            for t in (prediction, maturity)
        ]
        names = [
            np.select(
                [y == 2021, y == 2022, (y == 2023) & ~h, (y == 2023) & h],
                ["train", "validation", "calibration", "evaluation"],
                "",
            )
            for y, h in zip(years, halves, strict=True)
        ]
    return np.where((names[0] == names[1]) & (names[0] != ""), names[0], None)


class Calendar:
    """Vista temporal mínima: asigna por año y semestre como `bucket`."""

    def lookup(self, prediction):
        return None, prediction - 1, np.ones(len(prediction), dtype=bool)

    def partition_reasons(self, prediction, available, maturity, eligible):
        names = bucket(prediction.astype("datetime64[us]"), maturity.astype("datetime64[us]"), True)
        accepted = names != None  # noqa: E711
        partition = np.where(accepted, names, "excluded").astype("U13")
        reason = np.where(accepted, "accepted", "label_crosses_boundary").astype("U32")
        # Como la vista real, el final queda reservado con un tramo propio y sin aceptar.
        reserved = prediction.astype("datetime64[us]") >= RESERVED
        partition[reserved], reason[reserved] = "test_reserved", "final_test_reserved"
        return partition, reason

    def assign(self, prediction, available, maturity, eligible):
        # La revisión anterior pedía la asignación con los motivos como objetos.
        partition, reason = self.partition_reasons(prediction, available, maturity, eligible)
        return dict(partition=partition, reason=reason.astype(object))


def reader(module, path, temporal, cache_limit=1 << 30):
    """Lo que `_labels` usa de `CorpusDataset`, con su caché real y sin abrir un manifiesto."""
    dataset = object.__new__(module.CorpusDataset)
    calendar = Calendar() if temporal else None
    dataset.__dict__.update(
        _file=lambda asset, kind: path,
        verified={path: "huella"},
        cohort=COHORT,
        temporal=calendar,
        temporals={"US": calendar},
        partitions=SPLITS[temporal],
        _lock=threading.Lock(),
        _cache=OrderedDict(),
        cached_bytes=0,
        cache_limit=cache_limit,
        cache_entry_limit=8192,
    )
    return dataset


def label_case(rng, temporal, path):
    """Tabla de etiquetas válida y, la mitad de las veces, con un único defecto."""
    rows = int(rng.integers(0, 12))
    days = np.sort(rng.choice(3 * 365, rows, replace=False))
    prediction = (BASE + days).astype("datetime64[us]")
    maturity = prediction + rng.integers(1, 40, rows).astype("timedelta64[D]")
    partition = bucket(prediction, maturity, temporal).astype(object)
    reason = np.where(partition == None, "target_crosses_partition_boundary", "accepted")  # noqa: E711
    reason = reason.astype(object)
    history = rng.random(rows) < 0.2
    reason[history], partition[history] = "insufficient_history", None
    if temporal:
        view = rng.random(rows) < 0.1
        reason[view], partition[view] = "outside_window", None
        reserved = prediction >= RESERVED
        reason[reserved], partition[reserved] = "final_test_reserved", None
    columns = dict(
        sample_row=np.arange(rows, dtype=np.int64),
        prediction_at=prediction,
        target_available_at=maturity,
        target=rng.normal(size=rows),
        partition=partition,
        reason=reason,
        cohort_id=np.full(rows, COHORT, dtype=object),
    )
    sample_rows = rows
    counts = {p: int(np.sum(partition == p)) for p in SPLITS[temporal]}
    accepted = np.flatnonzero(reason == "accepted")
    excluded = np.flatnonzero(reason != "accepted")
    defect = rng.integers(0, 20) if rng.random() < 0.5 else None
    if defect == 0 and rows:
        columns["reason"][rng.integers(rows)] = "desconocido"
    elif defect == 1 and rows:
        columns["reason"][rng.integers(rows)] = None
    elif defect == 2 and len(excluded):
        columns["partition"][rng.choice(excluded)] = "train"
    elif defect == 3 and len(accepted):
        columns["partition"][rng.choice(accepted)] = None
    elif defect == 4 and len(accepted):
        row = rng.choice(accepted)
        columns["partition"][row] = "validation" if partition[row] == "train" else "train"
    elif defect == 5 and rows > 1:
        columns["sample_row"][1] = columns["sample_row"][0]
    elif defect == 6 and rows > 1:
        columns["prediction_at"][-1] = columns["prediction_at"][0]
    elif defect == 7 and len(accepted):
        row = rng.choice(accepted)
        columns["target_available_at"][row] = columns["prediction_at"][row]
    elif defect == 8 and len(accepted):
        columns["target"][rng.choice(accepted)] = np.nan
    elif defect == 9 and rows:
        columns["cohort_id"][rng.integers(rows)] = "otra_cohorte"
    elif defect == 10 and rows:
        columns["cohort_id"][rng.integers(rows)] = None
    elif defect == 11:
        del columns["reason"]
    elif defect == 12:
        counts[SPLITS[temporal][rng.integers(len(counts))]] += 1
    elif defect == 13:
        sample_rows += 1
    elif defect == 14 and rows:
        columns["sample_row"][rng.integers(rows)] = rng.choice([-1, rows])
    elif defect == 15 and rows:
        columns["reason"][:] = "insufficient_history"
        columns["partition"][:] = None
    elif defect == 16 and len(accepted):
        columns["partition"][rng.choice(accepted)] = "test_reserved"
    elif defect == 17 and temporal and np.any(prediction >= RESERVED):
        # Coincide con el tramo de la vista, pero la vista no acepta la fila.
        row = rng.choice(np.flatnonzero(prediction >= RESERVED))
        columns["reason"][row], columns["partition"][row] = "accepted", "test_reserved"
    # Las filas se escriben a veces desordenadas, y en grupos que dan varios fragmentos.
    order = rng.permutation(rows) if rng.random() < 0.5 else np.arange(rows)
    arrays = {
        "sample_row": pa.array(columns["sample_row"][order]),
        "prediction_at": pa.array(columns["prediction_at"][order], pa.timestamp("us", tz="UTC")),
        "target_available_at": pa.array(
            columns["target_available_at"][order], pa.timestamp("us", tz="UTC")
        ),
        "target": pa.array(columns["target"][order]),
        "partition": pa.array(columns["partition"][order].tolist(), pa.string()),
        "cohort_id": pa.array(columns["cohort_id"][order].tolist(), pa.string()),
    }
    if "reason" in columns:
        kind = pa.large_string() if rng.random() < 0.3 else pa.string()
        arrays["reason"] = pa.array(columns["reason"][order].tolist(), kind)
    if rng.random() < 0.3:
        arrays["partition"] = arrays["partition"].dictionary_encode()
    pq.write_table(pa.table(arrays), path, row_group_size=int(rng.integers(1, rows + 2)))
    return dict(market="US", counts=counts), sample_rows


def outcome(dataset, asset, rows, partitions):
    try:
        return [
            [(a.dtype.str, a.tolist()) for a in dataset._labels(asset, p, rows)]
            for p in (*partitions, "partición_inexistente")
        ]
    except Exception as error:
        return type(error).__name__, str(error)


@pytest.mark.parametrize("temporal", [False, True], ids=["fija", "temporal"])
def test_labels_match_the_row_by_row_reading(tmp_path, temporal):
    before = reference_module()
    seen = set()
    for seed in range(400):
        rng = np.random.default_rng(seed)
        path = tmp_path / f"labels-{seed}.parquet"
        asset, rows = label_case(rng, temporal, path)
        splits = SPLITS[temporal]
        expected = outcome(reader(before, path, temporal), asset, rows, splits)
        # Lectura que llena la caché y sigue desde ella, y lecturas sin caché por partición.
        assert outcome(reader(corpus_inputs, path, temporal), asset, rows, splits) == expected, seed
        for split in splits:
            alone = outcome(
                reader(corpus_inputs, path, temporal, cache_limit=0), asset, rows, [split]
            )
            assert alone == outcome(reader(before, path, temporal, 0), asset, rows, [split]), seed
        seen.add(expected[1] if isinstance(expected, tuple) else "ok")
    # El sorteo recorre los casos válidos y cada mensaje de error de la validación.
    assert seen == ERRORS | {"ok"}


def test_cached_labels_take_less_than_half_the_memory(tmp_path):
    before = reference_module()
    path = tmp_path / "labels.parquet"
    rows = 2000
    prediction = (BASE + np.arange(rows) // 2 * 1).astype("datetime64[us]")
    prediction += (np.arange(rows) % 2).astype("timedelta64[h]")
    maturity = prediction + np.timedelta64(1, "D")
    partition = bucket(prediction, maturity, False)
    reason = np.where(partition == None, "target_crosses_partition_boundary", "accepted")  # noqa: E711
    pq.write_table(
        pa.table(
            dict(
                sample_row=pa.array(np.arange(rows)),
                prediction_at=pa.array(prediction, pa.timestamp("us", tz="UTC")),
                target_available_at=pa.array(maturity, pa.timestamp("us", tz="UTC")),
                target=pa.array(np.linspace(-1, 1, rows)),
                partition=pa.array(partition.tolist(), pa.string()),
                reason=pa.array(reason.tolist()),
                cohort_id=pa.array([COHORT] * rows),
            )
        ),
        path,
    )
    asset = dict(market="US", counts={p: int(np.sum(partition == p)) for p in SPLITS[False]})
    old, new = reader(before, path, False), reader(corpus_inputs, path, False)
    assert outcome(new, asset, rows, SPLITS[False]) == outcome(old, asset, rows, SPLITS[False])
    *_, names, codes = next(iter(new._cache.values()))[1]
    assert codes.dtype == np.uint8 and set(names.tolist()) == {"train", "validation"}
    assert 0 < new.cached_bytes < old.cached_bytes / 2


@pytest.mark.parametrize(
    "column",
    [
        None,
        pa.array([], pa.string()),
        pa.array([COHORT] * 3),
        pa.array([COHORT, None, COHORT]),
        pa.array([None, None], pa.null()),
        pa.array([COHORT, "otra"]),
        pa.array([COHORT, COHORT]).dictionary_encode(),
        pa.array([COHORT, "otra"]).dictionary_encode(),
        pa.array([COHORT] * 2, pa.large_string()),
        pa.array([1, 2]),
        pa.array([COHORT.encode()]),
    ],
)
@pytest.mark.parametrize("cohort", [COHORT, None])
def test_cohort_rows_match_the_row_by_row_check(column, cohort):
    table = pa.table({"otra": pa.array([0])}) if column is None else pa.table({"cohort_id": column})
    expected = cohort is not None and (
        "cohort_id" not in table.column_names
        or any(value != cohort for value in table["cohort_id"].to_pylist())
    )
    if expected:
        with pytest.raises(ValueError, match="otra cohorte"):
            validate_cohort_rows(table, cohort)
    else:
        validate_cohort_rows(table, cohort)


def test_repeated_values_match_unique():
    rng = np.random.default_rng(7)
    cases = [np.arange(0), np.arange(1), np.arange(5), np.arange(5)[::-1], np.array([3, 3])]
    cases += [rng.integers(0, size, size) for size in range(1, 40) for _ in range(5)]
    cases += [np.sort(case) for case in cases]
    for values in cases:
        assert corpus_inputs._repeated(values) == (len(np.unique(values)) != len(values))


def test_an_empty_label_file_keeps_any_reason_type(tmp_path):
    """Sin filas no hay motivos que comparar, aunque la columna no sea de texto."""
    before = reference_module()
    path = tmp_path / "labels.parquet"
    stamp = pa.timestamp("us", tz="UTC")
    empty = dict(sample_row=pa.int64(), prediction_at=stamp, target_available_at=stamp)
    empty |= dict(target=pa.float64(), partition=pa.string(), reason=pa.int64())
    empty |= dict(cohort_id=pa.string())
    pq.write_table(pa.table({name: pa.array([], kind) for name, kind in empty.items()}), path)
    asset = dict(market="US", counts={"train": 0, "validation": 0})
    expected = outcome(reader(before, path, False), asset, 0, SPLITS[False])
    assert not isinstance(expected, tuple)
    assert outcome(reader(corpus_inputs, path, False), asset, 0, SPLITS[False]) == expected


def test_temporal_assignment_matches_the_object_reasons_before(tmp_path):
    """Los motivos de anchura fija coinciden con los objetos que devolvía `assign`."""
    before = reference_module("temporal_corpus")
    fixture = historical_temporal_fixture(tmp_path)
    views = tmp_path / "views"
    prepare_temporal_corpus(
        fixture.parent,
        fixture.protocols["US"],
        None,
        None,
        views,
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
    )
    rng = np.random.default_rng(3)
    checked = set()
    for manifest in sorted(views.glob("fold-*/manifest.json")):
        temporal = CorpusDataset(manifest, input_policy=HISTORICAL_MASKED).temporals["US"]
        sessions = temporal.partitioner.sessions
        for _ in range(20):
            chosen = np.sort(rng.choice(len(sessions) - 3, size=min(40, len(sessions) - 3)))
            prediction = sessions[chosen]
            maturity = sessions[chosen + rng.integers(1, 3, len(chosen))]
            available = prediction - rng.integers(-1, 2, len(chosen))
            eligible = rng.random(len(chosen)) < 0.9
            args = prediction, available, maturity, eligible
            expected = before.TemporalInputs.assign(temporal, *args)
            assigned = temporal.assign(*args)
            partition, reason = temporal.partition_reasons(*args)
            assert assigned["partition"].dtype == partition.dtype == expected["partition"].dtype
            assert assigned["reason"].dtype == expected["reason"].dtype == object
            assert assigned["partition"].tolist() == partition.tolist()
            assert partition.tolist() == expected["partition"].tolist()
            assert assigned["reason"].tolist() == reason.tolist() == expected["reason"].tolist()
            checked |= set(reason.tolist())
    # Las comparaciones recorren motivos aceptados, excluidos y reservados.
    assert {"accepted", "outside_window", "final_test_reserved"} <= checked


@pytest.mark.parametrize(
    "column",
    [
        pa.chunked_array([["train", "validation"], ["train"]]),
        pa.chunked_array([["calibration"] * 3], pa.large_string()),
        pa.chunked_array(
            [
                pa.DictionaryArray.from_arrays(
                    pa.array([0, 1, 2, 1], pa.int32()), pa.array(["train", "evaluation", "train"])
                )
            ]
        ),
        pa.chunked_array([], pa.string()),
    ],
    ids=["fragmentos", "texto-largo", "diccionario-repetido", "vacia"],
)
def test_dictionary_codes_rebuild_the_column(column):
    names, codes = corpus_inputs._dictionary(column)
    assert len(set(names.tolist())) == len(names)
    assert names[codes].tolist() == column.to_pylist()
    if len(column):
        assert names[codes].dtype == np.asarray(column.to_pylist()).dtype
