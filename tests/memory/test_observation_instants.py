"""Agrupación por instante de las filas del índice de observaciones (`_instants`).

La referencia es la agrupación fila a fila anterior, copiada aquí. Se comparan los instantes
entregados, los grupos anotados y el error, también cuando el error llega a mitad de un lote,
en el borde entre dos lotes o después de varios instantes ya entregados.
"""

import numpy as np
import pyarrow as pa
import pytest

from mars_titan.memory import financial_observations as api

FIELDS = api._SCHEMA.names


def reference(batches, start_cursor, groups):
    rows, previous = [], None
    for batch in batches:
        if any(column.null_count for column in batch.columns):
            raise ValueError("El índice contiene posiciones ausentes")
        columns = [column.to_numpy(zero_copy_only=False) for column in batch.columns]
        for values in zip(*columns, strict=True):
            item = tuple(int(value) for value in values)
            at = item[0]
            if previous is not None and at < previous:
                raise ValueError("El índice ha perdido su orden temporal")
            if previous is not None and at != previous:
                groups.append([previous, len(rows)])
                if len(groups) > start_cursor:
                    yield previous, rows
                rows = []
            rows.append(item)
            if len(rows) > 16385:
                raise ValueError("El evento lógico supera su presupuesto")
            previous = at
    if rows:
        groups.append([previous, len(rows)])
        if len(groups) > start_cursor:
            yield previous, rows


def batches(moments, sizes, *, null_at=None):
    """Lotes con esas fechas por fila, cortados con esos tamaños, y valores distintos por fila."""
    moments = np.asarray(moments, dtype=np.int64)
    total = len(moments)
    columns = [moments] + [
        np.arange(total, dtype=np.int64) * (k + 2) - 3 * k for k in range(len(FIELDS) - 1)
    ]
    result, start = [], 0
    for size in sizes:
        arrays = []
        for k, values in enumerate(columns):
            part = values[start : start + size]
            mask = None
            if null_at is not None and k == 3 and start <= null_at < start + size:
                mask = np.arange(start, start + size) == null_at
            arrays.append(pa.array(part, type=pa.int64(), mask=mask))
        result.append(pa.RecordBatch.from_arrays(arrays, names=FIELDS))
        start += size
    assert start == total
    return result


def run(function, items, start_cursor):
    groups, delivered = [], []
    try:
        for at, rows in function(iter(items), start_cursor, groups):
            delivered.append((at, list(rows)))
    except ValueError as error:
        return delivered, groups, str(error)
    return delivered, groups, None


def assert_same(items, start_cursor):
    expected = run(reference, items, start_cursor)
    found = run(api._instants, items, start_cursor)
    assert found == expected
    for _, rows in found[0]:
        assert all(type(value) is int for row in rows for value in row)
    return found


# Instantes de 3, 1, 5, 2 y 4 filas, con lotes que cortan instantes por la mitad y uno vacío.
MOMENTS = [10] * 3 + [20] + [30] * 5 + [40] * 2 + [55] * 4
CUTS = [2, 0, 3, 4, 1, 5]


@pytest.mark.parametrize("start_cursor", [0, 1, 3, 5])
def test_instants_match_the_row_by_row_grouping_across_batch_borders(start_cursor):
    delivered, groups, error = assert_same(batches(MOMENTS, CUTS), start_cursor)
    assert error is None
    assert groups == [[10, 3], [20, 1], [30, 5], [40, 2], [55, 4]]
    assert [at for at, _ in delivered] == [10, 20, 30, 40, 55][start_cursor:]


@pytest.mark.parametrize(
    ("moments", "cuts"),
    [
        # Retrocede dentro de un lote, después de dos instantes completos.
        ([10, 10, 20, 30, 25, 40], [6]),
        # Retrocede justo en el borde entre dos lotes.
        ([10, 20, 20, 30, 15, 40], [4, 2]),
        # Retrocede en la primera fila de un instante que sigue en el lote siguiente.
        ([10, 10, 30, 30, 20, 20], [3, 3]),
    ],
)
def test_a_step_back_fails_after_the_same_complete_instants(moments, cuts):
    delivered, _, error = assert_same(batches(moments, cuts), 0)
    assert error == "El índice ha perdido su orden temporal"
    assert delivered


@pytest.mark.parametrize("cuts", [[16386 + 3], [2, 8192, 8192, 3], [16384, 5]])
def test_an_instant_over_its_budget_fails_after_the_previous_ones(cuts):
    moments = [5, 5] + [9] * 16386 + [12]
    _, groups, error = assert_same(batches(moments, cuts), 0)
    assert error == "El evento lógico supera su presupuesto"
    assert groups == [[5, 2]]


def test_an_instant_at_its_budget_is_delivered_whole():
    moments = [9] * 16385 + [12]
    delivered, groups, error = assert_same(batches(moments, [8192, 8192, 2]), 0)
    assert error is None and groups == [[9, 16385], [12, 1]]
    assert len(delivered[0][1]) == 16385


@pytest.mark.parametrize("null_at", [0, 4, 9])
def test_a_missing_position_fails_in_its_batch(null_at):
    _, _, error = assert_same(batches(MOMENTS, CUTS, null_at=null_at), 0)
    assert error == "El índice contiene posiciones ausentes"


def test_no_rows_deliver_nothing():
    assert assert_same(batches([], [0]), 0) == ([], [], None)
