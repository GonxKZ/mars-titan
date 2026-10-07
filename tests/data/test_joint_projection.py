"""Proyección semántica exacta de valores, máscaras y edades numéricas."""

import importlib

import numpy as np
import pyarrow as pa
import pytest


def module():
    try:
        return importlib.import_module("mars_titan.data.joint_projection")
    except ModuleNotFoundError:
        pytest.fail("Falta la proyección de contextos numéricos")


def array(values, size=None):
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), size or values.shape[1])


def bits(column):
    return column.values.to_numpy().view(np.uint32).reshape(len(column), column.type.list_size)


def reference(values, source, target):
    """Referencia por filas para fixtures pequeños, sin reutilizar la proyección."""
    result = np.zeros((len(values), 3 * len(target)), dtype=np.float32)
    for row, vector in enumerate(values):
        for channel in range(3):
            for index, concept in enumerate(source):
                result[row, channel * len(target) + target.index(concept)] = vector[
                    channel * len(source) + index
                ]
    return result.view(np.uint32)


def test_permutation_preserves_bits_and_separates_known_zero_from_structural_absence():
    source = ["us-gaap:Assets:USD", "us-gaap:Assets:CAD", "cn-reported:Assets:CNY"]
    target = [source[2], "other:Assets:EUR", source[0], source[1]]
    values = np.array(
        [[-0.0, -0.0, 1e-40, 1, 0, 1, 3, -0.0, 0], [2.5, 7, 0, 1, 1, 0, 1, 2, 0]],
        dtype=np.float32,
    )
    initial = values.view(np.uint32).copy()
    result = module().project_numeric_context(array(values), source, target)
    assert result.type == pa.list_(pa.float32(), 12)
    np.testing.assert_array_equal(bits(result), reference(values, source, target))
    np.testing.assert_array_equal(values.view(np.uint32), initial)
    assert result.to_pylist()[0][6] == 1
    assert result.to_pylist()[0][7] == 0
    assert np.all(bits(result)[:, [1, 5, 9]] == 0)


def test_identity_projection_does_not_alias_a_mutable_source_buffer():
    values = np.array([[2, 1, 3]], dtype=np.float32)
    result = module().project_numeric_context(array(values), ["A"], ["A"])
    values[:] = 0
    assert result.to_pylist() == [[2, 1, 3]]


def test_slice_ignores_nulls_and_nonfinite_values_outside_its_logical_rows():
    source = pa.FixedSizeListArray.from_arrays(
        pa.array([None, 0, 0, 5, 1, 2, float("nan"), 0, 0], type=pa.float32()), 3
    ).slice(1, 1)
    result = module().project_numeric_context(source, ["A"], ["B", "A"])
    assert result.to_pylist() == [[0, 5, 0, 1, 0, 2]]


def test_sliced_child_values_and_chunked_columns_keep_row_order():
    primitive = pa.array([999, 7, 1, 2, 8, 1, 3, 999], type=pa.float32()).slice(1, 6)
    source = pa.FixedSizeListArray.from_arrays(primitive, 3)
    chunks = pa.chunked_array([source.slice(1, 1), source.slice(0, 0), source.slice(0, 1)])
    result = module().project_numeric_context(chunks, ("A",), ("B", "A"))
    assert isinstance(result, pa.FixedSizeListArray)
    assert result.to_pylist() == [[0, 8, 0, 1, 0, 3], [0, 7, 0, 1, 0, 2]]


@pytest.mark.parametrize("chunked", [False, True])
def test_empty_input_has_the_target_type(chunked):
    source = array(np.empty((0, 3), dtype=np.float32))
    if chunked:
        source = pa.chunked_array([], type=source.type)
    result = module().project_numeric_context(source, ["A"], ["B", "A"])
    assert len(result) == 0 and result.type == pa.list_(pa.float32(), 6)


@pytest.mark.parametrize(
    "source,target",
    [
        ([], ["A"]),
        (["A"], []),
        (["A", "A"], ["A"]),
        (["A"], ["A", "A"]),
        (["A"], ["a"]),
        (["us-gaap:Assets:USD"], ["us-gaap:Assets:CAD"]),
        (["us-gaap:Assets:CNY"], ["cn-reported:Assets:CNY"]),
        ([""], [""]),
        ([" "], [" "]),
        ([None], ["A"]),
        ([1], ["A"]),
        ("A", ["A"]),
        (["A"], "A"),
        (["x" * 257], ["x" * 257]),
        (["A"], ["A"] + [f"B{i}" for i in range(512)]),
    ],
)
def test_invalid_or_aliased_concepts_are_rejected(source, target):
    with pytest.raises(ValueError):
        module().project_numeric_context(array([[0, 0, 0]]), source, target)


@pytest.mark.parametrize(
    "source",
    [
        pa.array([[0.0, 0.0, 0.0]], type=pa.list_(pa.float32())),
        pa.array([[0.0, 0.0, 0.0]], type=pa.list_(pa.float64(), 3)),
        pa.array([[0, 0, 0]], type=pa.list_(pa.int32(), 3)),
        pa.array([[0.0, 0.0]], type=pa.list_(pa.float32(), 2)),
        pa.array([None], type=pa.list_(pa.float32(), 3)),
        pa.array([[0.0, None, 0.0]], type=pa.list_(pa.float32(), 3)),
        np.zeros((1, 3), dtype=np.float32),
    ],
)
def test_wrong_arrow_type_width_and_nulls_are_rejected(source):
    with pytest.raises(ValueError):
        module().project_numeric_context(source, ["A"], ["A"])


@pytest.mark.parametrize(
    "values",
    [
        [float("nan"), 1, 0],
        [float("inf"), 1, 0],
        [-float("inf"), 1, 0],
        [1, float("nan"), 0],
        [1, 1, float("inf")],
        [1, 0.5, 0],
        [1, -1, 0],
        [1, 2, 0],
        [1, 1, -1],
        [1, 0, 0],
        [0, 0, 1],
    ],
)
def test_invalid_numeric_context_is_rejected(values):
    with pytest.raises(ValueError):
        module().project_numeric_context(array([values]), ["A"], ["A", "B"])


def test_all_chunks_and_blocks_are_validated(monkeypatch):
    api = module()
    monkeypatch.setattr(api, "_BLOCK_BYTES", 24)
    values = np.tile(np.array([[5, 1, 2]], dtype=np.float32), (7, 1))
    source = array(values)
    result = api.project_numeric_context(source, ["A"], ["B", "A"])
    np.testing.assert_array_equal(bits(result), reference(values, ["A"], ["B", "A"]))
    corrupted = pa.chunked_array([source, array([[0, 0, 1]])])
    with pytest.raises(ValueError):
        api.project_numeric_context(corrupted, ["A"], ["B", "A"])


def test_budget_covers_input_and_output_before_allocating(monkeypatch):
    api = module()
    source = array([[2, 1, 3]])
    monkeypatch.setattr(api, "_MAX_BYTES", 36)
    assert api.project_numeric_context(source, ["A"], ["B", "A"]).to_pylist() == [
        [0, 2, 0, 1, 0, 3]
    ]
    monkeypatch.setattr(api, "_MAX_BYTES", 35)

    def forbidden(*args, **kwargs):
        pytest.fail("Se reservó una salida fuera del presupuesto")

    monkeypatch.setattr(api.np, "zeros", forbidden)
    with pytest.raises(ValueError, match="64 MiB"):
        api.project_numeric_context(source, ["A"], ["B", "A"])


def test_actual_64_mib_limit_rejects_excessive_logical_input_and_output():
    rows = (64 * 1024**2) // 24 + 1
    source = array(np.zeros((rows, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="64 MiB"):
        module().project_numeric_context(source, ["A"], ["A"])


def test_slice_budget_also_counts_the_buffers_it_keeps_alive(monkeypatch):
    api = module()
    source = array(np.zeros((5, 3), dtype=np.float32)).slice(2, 1)
    monkeypatch.setattr(api, "_MAX_BYTES", 83)
    with pytest.raises(ValueError, match="64 MiB"):
        api.project_numeric_context(source, ["A"], ["A", "B"])


def test_empty_fragments_cannot_bypass_the_metadata_budget():
    empty = array(np.empty((0, 3), dtype=np.float32))
    source = pa.chunked_array([empty] * 1025)
    with pytest.raises(ValueError, match="fragmentos"):
        module().project_numeric_context(source, ["A"], ["A"])
