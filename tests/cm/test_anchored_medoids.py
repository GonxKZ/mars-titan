"""Objetivo completo con centros fijos, sin consultar etiquetas ni fechas."""

import dataclasses
import itertools
import math

import numpy as np
import pytest

from mars_titan.cm.anchored_medoids import select_anchored_medoids
from mars_titan.cm.medoids import MedoidBudgetExceeded, select_medoids


def objective(points, centers, metric):
    if metric == "l1" and points.dtype.kind == "i":
        return sum(
            min(
                sum(abs(int(a) - int(b)) for a, b in zip(row, center, strict=True))
                for center in centers
            )
            for row in points
        )
    return math.fsum(
        min(
            math.dist(row, center)
            if metric == "euclidean"
            else math.fsum(abs(float(a) - float(b)) for a, b in zip(row, center, strict=True))
            for center in centers
        )
        for row in points
    )


@pytest.mark.parametrize("metric", ["euclidean", "l1"])
def test_enumeration_matches_full_objective_with_real_fixed_centers(metric):
    points = np.array([[0, 0], [0, 0], [3, 4], [4, 4], [8, 0], [10, 0]], dtype=np.int64)
    if metric == "euclidean":
        points = points.astype(float)
    ids = list("abcdef")
    candidates = list("cdef")
    expected = min(
        (objective(points, points[[0, *[ids.index(i) for i in selected]]], metric), selected)
        for selected in itertools.combinations(candidates, 2)
    )
    result = select_anchored_medoids(
        points, ids, ["a"], candidates, 3, algorithm="enumeration", metric=metric
    )
    assert result.objective == pytest.approx(expected[0])
    assert result.variable_ids == expected[1]
    assert result.fixed_ids == ("a",)
    assert result.retained_ids == tuple(ids[i] for i in result.retained_indices)
    assert len(result.retained_ids) == 3
    assert result.status == "enumerated_restricted"
    assert result.background_pairs == 5
    assert result.distance_pairs == result.background_pairs + result.variable_pairs


def test_fixed_center_cost_zero_survives_variable_state_recomputation():
    points = np.array([[0.0], [0.0], [9.0], [10.0], [11.0]])
    result = select_anchored_medoids(points, list("abcde"), ["a"], list("cde"), 2)
    assert result.fixed_ids == ("a",) and result.variable_ids == ("d",)
    assert result.objective == 2
    assert result.status == "one_swap_local_restricted"


def test_clients_outside_candidates_still_contribute_to_objective():
    points = np.array([[0.0], [5.0], [6.0], [7.0], [10.0]])
    result = select_anchored_medoids(points, list("abcde"), ["a"], ["e"], 2)
    assert result.retained_ids == ("a", "e")
    assert result.objective == 12


def test_no_fixed_centers_matches_existing_selector_and_external_rng():
    points = np.array([[7, -10], [1, -9], [-9, 1], [3, 12], [-20, 5], [12, 7], [13, -6]])
    ids = list("abcdefg")
    state = np.random.get_state()
    expected = select_medoids(points, ids, points, ids, 2, metric="l1")
    result = select_anchored_medoids(points, ids, [], ids, 2, metric="l1")
    assert result.retained_ids == expected.candidate_ids
    assert result.retained_indices == expected.candidate_indices
    assert result.objective == expected.objective
    assert result.variable_pairs == expected.distance_pairs
    assert result.swaps == expected.swaps
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


@pytest.mark.parametrize("backend", ["numpy", "scipy_cdist_fp32_exploratory"])
def test_permutations_and_geometry_duplicates_preserve_real_representatives(backend):
    points = np.array([[0, 0], [0, 0], [4, 0], [4, 0], [8, 0], [9, 0]], dtype=np.float32)
    ids = np.array(list("abcdef"))
    baseline = select_anchored_medoids(
        points, ids, ["a"], list("cdef"), 2, background_backend=backend
    )
    result = select_anchored_medoids(
        points[::-1],
        ids[::-1],
        ["a"],
        list("fedc"),
        2,
        background_backend=backend,
        client_block_size=2,
        fixed_block_size=1,
        candidate_block_size=1,
    )
    assert baseline.retained_ids == result.retained_ids
    assert baseline.objective == result.objective
    assert tuple(ids[::-1][list(result.retained_indices)]) == result.retained_ids
    assert "c" in result.variable_ids and "d" not in result.variable_ids


def test_capacity_contracts_and_fixed_only_objective():
    points = np.array([[0.0], [3.0], [4.0]])
    fixed = select_anchored_medoids(points, list("abc"), ["a"], ["b", "c"], 1)
    assert fixed.retained_ids == ("a",) and fixed.objective == 7
    assert fixed.status == "fixed_only" and fixed.variable_pairs == 0
    all_fixed = select_anchored_medoids(
        points, list("abc"), list("abc"), [], 3, max_distance_pairs=1
    )
    assert all_fixed.objective == all_fixed.distance_pairs == 0
    empty = select_anchored_medoids(np.empty((0, 1)), [], [], [], 1)
    assert empty.retained_ids == () and empty.objective == 0
    with pytest.raises(ValueError, match="capacity"):
        select_anchored_medoids(points, list("abc"), ["a", "b"], ["c"], 1)
    with pytest.raises(ValueError):
        select_anchored_medoids(points, list("abc"), [], [], 2)
    with pytest.raises(ValueError, match="capacity"):
        select_anchored_medoids(np.empty((0, 1)), [], [], [], 0)


def test_excess_capacity_keeps_all_authorized_centers():
    points = np.array([[0.0], [3.0], [4.0]])
    result = select_anchored_medoids(points, list("abc"), ["a"], ["c"], 8)
    assert result.retained_ids == ("a", "c") and result.objective == 1
    assert result.status == "all_candidates"


@pytest.mark.parametrize(
    "fixed,candidates",
    [
        (["missing"], ["a"]),
        (["a"], ["missing"]),
        (["a", "a"], ["b"]),
        (["a"], ["b", "b"]),
        (["a"], ["a", "b"]),
    ],
)
def test_invalid_center_identity_is_rejected(fixed, candidates):
    with pytest.raises(ValueError):
        select_anchored_medoids(np.zeros((2, 1)), ["a", "b"], fixed, candidates, 2)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coordinates_rejected_before_library(value):
    with pytest.raises(ValueError):
        select_anchored_medoids(
            np.array([[0], [value]], dtype=np.float32),
            ["a", "b"],
            ["a"],
            ["b"],
            2,
            background_backend="scipy_cdist_fp32_exploratory",
        )


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.int64])
def test_specialized_backend_rejects_other_coordinate_types(dtype):
    with pytest.raises(TypeError, match="FP32"):
        select_anchored_medoids(
            np.zeros((2, 1), dtype=dtype),
            ["a", "b"],
            ["a"],
            ["b"],
            2,
            background_backend="scipy_cdist_fp32_exploratory",
        )


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_general_reference_retains_fp64_dynamic_range(scale):
    points = np.array([[0.0, 0.0], [3 * scale, 4 * scale]])
    result = select_anchored_medoids(points, ["a", "b"], ["a"], [], 1)
    assert result.objective == pytest.approx(5 * scale, rel=1e-14, abs=0)


@pytest.mark.parametrize(
    "scale", [0.0, np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).max]
)
def test_specialized_backend_matches_reference_at_fp32_extremes(scale):
    points = np.array([[0, 0], [scale, scale], [-scale, scale], [scale, -scale]], dtype=np.float32)
    for algorithm in ("enumeration", "greedy_swap"):
        first = select_anchored_medoids(
            points, list("abcd"), ["a"], list("bcd"), 2, algorithm=algorithm
        )
        second = select_anchored_medoids(
            points,
            list("abcd"),
            ["a"],
            list("bcd"),
            2,
            algorithm=algorithm,
            background_backend="scipy_cdist_fp32_exploratory",
        )
        assert second.objective == pytest.approx(first.objective, rel=1e-14, abs=0)
        assert second.retained_ids == first.retained_ids
        assert second.background_backend == "scipy_cdist_fp32_exploratory"
        assert second.coordinate_dtype == np.dtype(np.float32).str
        assert second.distance_arithmetic == "float64"


def test_distance_budget_includes_fixed_background_and_exact_boundary():
    points = np.array([[0.0], [1.0], [3.0], [7.0]])
    result = select_anchored_medoids(points, list("abcd"), ["a", "b"], [], 2, max_distance_pairs=4)
    assert result.background_pairs == result.distance_pairs == 4
    with pytest.raises(MedoidBudgetExceeded, match="pares"):
        select_anchored_medoids(points, list("abcd"), ["a", "b"], [], 2, max_distance_pairs=3)
    complete = select_anchored_medoids(points, list("abcd"), ["a"], list("bcd"), 2)
    with pytest.raises(MedoidBudgetExceeded, match="pares"):
        select_anchored_medoids(
            points,
            list("abcd"),
            ["a"],
            list("bcd"),
            2,
            max_distance_pairs=complete.distance_pairs - 1,
        )


def test_memory_and_enumeration_budgets_fail_explicitly():
    points = np.arange(9, dtype=float)[:, None]
    ids = list("abcdefghi")
    with pytest.raises(MedoidBudgetExceeded, match="memoria"):
        select_anchored_medoids(points, ids, ["a"], ids[1:], 4, max_working_bytes=1)
    with pytest.raises(MedoidBudgetExceeded, match="combinaciones"):
        select_anchored_medoids(
            points, ids, ["a"], ids[1:], 4, algorithm="enumeration", max_combinations=1
        )


def test_work_limit_result_stays_labelled_partial():
    points = np.array([[0.0], [1.0], [4.0], [8.0], [9.0]])
    result = select_anchored_medoids(points, list("abcde"), ["a"], list("bcde"), 3, max_swaps=0)
    assert result.status == "swap_limit" and result.swaps == 0


def test_integer_l1_preserves_python_total_and_checks_distance_overflow():
    limit = np.iinfo(np.int64).max
    points = np.array([[0], [limit], [limit]], dtype=np.int64)
    result = select_anchored_medoids(points, list("abc"), ["a"], [], 1, metric="l1")
    assert result.objective == 2 * int(limit) and isinstance(result.objective, int)
    with pytest.raises(ValueError, match="int64"):
        select_anchored_medoids(np.array([[-1], [limit]]), ["a", "b"], ["a"], [], 1, metric="l1")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity": True},
        {"metric": "sqeuclidean"},
        {"background_backend": "automatic"},
        {"algorithm": "published"},
        {"fixed_block_size": 0},
        {"client_block_size": 0},
        {"candidate_block_size": False},
        {"max_swaps": -1},
        {"reuse_distances": 1},
        {"bounded_background": "yes"},
    ],
)
def test_bad_configuration_rejected(kwargs):
    config = {"capacity": 2, **kwargs}
    with pytest.raises((ValueError, TypeError)):
        select_anchored_medoids(np.zeros((2, 1)), ["a", "b"], ["a"], ["b"], **config)


def test_missing_optional_backend_fails_without_fallback(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "scipy":
            raise ImportError("fixture")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match="SciPy"):
        select_anchored_medoids(
            np.zeros((2, 1), dtype=np.float32),
            ["a", "b"],
            ["a"],
            ["b"],
            2,
            background_backend="scipy_cdist_fp32_exploratory",
        )


def test_specialized_metric_and_dimension_contracts():
    with pytest.raises(ValueError, match="euclídea"):
        select_anchored_medoids(
            np.zeros((2, 1), dtype=np.float32),
            ["a", "b"],
            ["a"],
            ["b"],
            2,
            metric="l1",
            background_backend="scipy_cdist_fp32_exploratory",
        )
    with pytest.raises(ValueError, match="dimensiones"):
        select_anchored_medoids(
            np.zeros((2, 4097), dtype=np.float32),
            ["a", "b"],
            ["a"],
            ["b"],
            2,
            background_backend="scipy_cdist_fp32_exploratory",
        )


def test_traced_peak_includes_unicode_ids_and_inputs_remain_unchanged():
    import tracemalloc

    points = np.zeros((2000, 1), dtype=np.float32)
    ids = np.array(["a" * 247 + f"{index:05d}" + "\U0001f600" for index in range(len(points))])
    before = points.copy()
    tracemalloc.start()
    try:
        result = select_anchored_medoids(points, ids, ids[:1999], ids[1999:], 2000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= result.estimated_peak_bytes
    assert result.background_tile_shape[0] == 1
    np.testing.assert_array_equal(points, before)


@pytest.mark.parametrize("metric", ["l1", "euclidean"])
def test_many_small_restricted_problems_match_independent_enumeration(metric):
    rng = np.random.default_rng(302)
    for case in range(12):
        points = rng.integers(-8, 9, (9, 3), dtype=np.int64)
        if metric == "euclidean":
            points = points.astype(np.float32)
        if case % 3 == 0:
            points[5] = points[0]
            points[8] = points[7]
        ids = list("abcdefghi")
        fixed, candidates = ids[:4], ids[4:]
        expected = min(
            (
                objective(points, points[[0, 1, 2, 3, *chosen]], metric),
                tuple(ids[i] for i in chosen),
            )
            for chosen in itertools.combinations(range(4, 9), 2)
        )
        result = select_anchored_medoids(
            points, ids, fixed, candidates, 6, algorithm="enumeration", metric=metric
        )
        assert result.objective == pytest.approx(expected[0], rel=1e-14, abs=0)
        if metric == "l1":
            assert result.variable_ids == expected[1]
        else:
            accelerated = select_anchored_medoids(
                points,
                ids,
                fixed,
                candidates,
                6,
                algorithm="enumeration",
                background_backend="scipy_cdist_fp32_exploratory",
            )
            assert accelerated.objective == pytest.approx(result.objective, rel=1e-14, abs=0)
            assert accelerated.retained_ids == result.retained_ids


def test_exploratory_background_can_change_ids_at_a_geometric_tie():
    points = np.array(
        [
            [-100, -100, -100],
            [0, 0, 0],
            [-100, -100, -100],
            [0, 0, 0],
            [0.3654440641403198, 0.4127326011657715, 0.4308210015296936],
        ],
        dtype=np.float32,
    )
    reference = select_anchored_medoids(
        points, list("abcde"), list("ab"), list("cd"), 3, algorithm="enumeration"
    )
    exploratory = select_anchored_medoids(
        points,
        list("abcde"),
        list("ab"),
        list("cd"),
        3,
        algorithm="enumeration",
        background_backend="scipy_cdist_fp32_exploratory",
    )
    assert reference.retained_ids == ("a", "b", "c")
    assert exploratory.retained_ids == ("a", "b", "d")
    assert reference.objective == exploratory.objective


def test_unlabelled_scipy_backend_is_rejected():
    with pytest.raises(ValueError, match="Backend"):
        select_anchored_medoids(
            np.zeros((2, 1), dtype=np.float32),
            ["a", "b"],
            ["a"],
            ["b"],
            2,
            background_backend="scipy_cdist_fp32",
        )


def bank_problem(seed, *, clients=700, dimensions=5, kind="float"):
    """Retención restringida con varios bloques de clientes que necesita un intercambio.

    Dos grupos en 0 y 10 sobre la primera coordenada y un candidato intermedio: la elección
    greedy empieza por el centro y un intercambio lo sustituye. Los fijos quedan lejos.
    """
    rng = np.random.default_rng(seed)
    half = clients // 2
    first = np.concatenate([rng.normal(0, 1, half), rng.normal(10, 1, clients - half)])
    points = np.column_stack([first, rng.normal(0, 0.3, (clients, dimensions - 1))])
    points[0] = [5] + [0] * (dimensions - 1)
    points[1:21, 0] += 1000
    if kind == "ties":
        points = np.round(points)
    points = np.round(points).astype(np.int64) if kind == "integer" else points.astype(np.float32)
    ids = [f"{index:05d}" for index in range(clients)]
    candidates = [ids[0], *ids[30:36], *ids[half + 30 : half + 36]]
    return points, ids, ids[1:21], candidates, 22


@pytest.mark.parametrize(
    ("case", "metric"),
    [("float", "euclidean"), ("ties", "euclidean"), ("integer", "l1"), ("ties", "l1")],
)
def test_reused_distances_give_the_reference_bits(case, metric):
    points, ids, fixed, candidates, capacity = bank_problem(11, kind=case)
    options = dict(client_block_size=64, fixed_block_size=32, candidate_block_size=3, max_swaps=8)
    reference = select_anchored_medoids(
        points, ids, fixed, candidates, capacity, metric=metric, **options
    )
    reused = select_anchored_medoids(
        points, ids, fixed, candidates, capacity, metric=metric, reuse_distances=True, **options
    )
    assert reused.reused_distances and not reference.reused_distances
    assert reference.swaps > 0
    assert type(reused.objective) is type(reference.objective)
    assert repr(reused.objective) == repr(reference.objective)
    for field in dataclasses.fields(reference):
        if field.name not in {"reused_distances", "estimated_peak_bytes"}:
            assert getattr(reused, field.name) == getattr(reference, field.name), field.name
    # La tabla de distancias se suma a la memoria declarada.
    assert reused.estimated_peak_bytes > reference.estimated_peak_bytes


def test_reused_distances_compute_each_variable_pair_once(monkeypatch):
    from mars_titan.cm import medoids

    points, ids, fixed, candidates, capacity = bank_problem(5)
    options = dict(client_block_size=64, fixed_block_size=32, candidate_block_size=3, max_swaps=8)
    compute = medoids._Costs._compute

    def counted(work):
        def wrapper(self, start, end, indices, distance):
            work.append(((end - start) * len(indices), len(self.candidates)))
            return compute(self, start, end, indices, distance)

        return wrapper

    runs = {}
    for reuse in (False, True):
        work = runs[reuse] = []
        monkeypatch.setattr(medoids._Costs, "_compute", counted(work))
        result = select_anchored_medoids(
            points, ids, fixed, candidates, capacity, reuse_distances=reuse, **options
        )
        # Los pares se cuentan igual con y sin reutilización.
        assert result.distance_pairs == sum(pairs for pairs, _ in runs[False])

    def variable(work):
        return sum(pairs for pairs, size in work if size == len(candidates))

    clients = len(points) - len(fixed)
    assert variable(runs[True]) == clients * len(candidates)
    assert variable(runs[False]) > 3 * clients * len(candidates)


def test_budget_failures_and_tight_memory_keep_the_reference_behaviour():
    points, ids, fixed, candidates, capacity = bank_problem(3)
    options = dict(client_block_size=64, fixed_block_size=32, candidate_block_size=3, max_swaps=8)
    reference = select_anchored_medoids(points, ids, fixed, candidates, capacity, **options)
    both = dict(reuse_distances=True, bounded_background=True)
    for limit in (reference.background_pairs - 1, reference.distance_pairs - 1):
        with pytest.raises(MedoidBudgetExceeded, match="pares"):
            select_anchored_medoids(
                points,
                ids,
                fixed,
                candidates,
                capacity,
                max_distance_pairs=limit,
                **both,
                **options,
            )
    # Sin memoria para la cota ni la tabla se usa la ruta de referencia, y la salida lo dice.
    tight = select_anchored_medoids(
        points,
        ids,
        fixed,
        candidates,
        capacity,
        max_working_bytes=reference.estimated_peak_bytes,
        **both,
        **options,
    )
    assert tight == reference
    # Con memoria solo para la cota, la tabla se descarta y la cota se conserva.
    bounded = select_anchored_medoids(
        points, ids, fixed, candidates, capacity, bounded_background=True, **options
    )
    partial = select_anchored_medoids(
        points,
        ids,
        fixed,
        candidates,
        capacity,
        max_working_bytes=bounded.estimated_peak_bytes,
        **both,
        **options,
    )
    assert (partial.bounded_background, partial.reused_distances) == (True, False)
    assert partial == bounded


def traced_problem(shape):
    """Problemas grandes, mínimos, con un solo centro fijo, densos y de muchas dimensiones."""
    if shape == "bank":
        return (*bank_problem(9, clients=4000, dimensions=8), {})
    if shape == "minimal":
        points = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
        return points, ["a", "b", "c"], ["a"], ["b"], 2, {}
    if shape == "one_fixed":
        points = np.random.default_rng(6).standard_normal((300, 1)).astype(np.float32)
        ids = [f"{index:03d}" for index in range(300)]
        return points, ids, ids[:1], ids[1:5], 3, dict(client_block_size=7)
    if shape == "dense":
        points = adversarial_points("constant")
        return (points, *retention(points), dict(fixed_block_size=96))
    points = adversarial_points("normal", count=400, dimensions=300)
    return (points, *retention(points, fixed=200), dict(fixed_block_size=96))


@pytest.mark.parametrize("reuse", [False, True], ids=["bound", "bound_and_reuse"])
@pytest.mark.parametrize("shape", ["bank", "minimal", "one_fixed", "dense", "wide"])
def test_traced_peak_with_the_options_stays_within_the_estimate(shape, reuse):
    import tracemalloc

    points, ids, fixed, candidates, capacity, options = traced_problem(shape)
    tracemalloc.start()
    try:
        result = select_anchored_medoids(
            points,
            ids,
            fixed,
            candidates,
            capacity,
            reuse_distances=reuse,
            bounded_background=True,
            **options,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert (result.reused_distances, result.bounded_background) == (reuse, True)
    assert peak <= result.estimated_peak_bytes


def adversarial_points(kind, count=900, dimensions=16, seed=4):
    """Coordenadas FP32 donde la cota del fondo es floja o hay empates exactos."""
    rng = np.random.default_rng(seed)
    if kind == "normal":
        points = rng.standard_normal((count, dimensions))
    elif kind == "duplicates":
        points = np.repeat(rng.standard_normal((count // 6 + 1, dimensions)), 6, axis=0)[:count]
    elif kind == "far_cloud":
        # Una nube lejana del origen y muy estrecha: el producto de matrices pierde casi todo.
        points = 1e4 + 1e-3 * rng.standard_normal((count, dimensions))
    elif kind == "mixed_scales":
        points = rng.standard_normal((count, dimensions)) * 10.0 ** rng.integers(-8, 8, (count, 1))
    elif kind == "grid":
        points = rng.integers(-2, 3, (count, dimensions)).astype(float)
    elif kind == "constant":
        # Todas las distancias valen cero: ningún par se puede descartar.
        points = np.full((count, dimensions), 3.0)
    else:
        points = rng.standard_normal((count, dimensions)) * 1e37
    return points.astype(np.float32)


KINDS = ["normal", "duplicates", "far_cloud", "mixed_scales", "grid", "constant", "extremes"]


def retention(points, seed=5, fixed=500):
    rng = np.random.default_rng(seed)
    ids = [f"{index:05d}" for index in range(len(points))]
    order = [ids[index] for index in rng.permutation(len(points))]
    return ids, order[:fixed], order[fixed : fixed + 16], fixed + 8


@pytest.mark.parametrize("reuse", [False, True], ids=["bound", "bound_and_reuse"])
@pytest.mark.parametrize("kind", KINDS)
def test_bounded_background_gives_the_reference_bits(kind, reuse):
    assert_reference_bits(adversarial_points(kind), reuse)


@pytest.mark.parametrize("kind", ["normal", "duplicates", "mixed_scales"])
def test_bounded_background_keeps_the_reference_bits_beyond_255_dimensions(kind):
    # El margen del factor final crece con D. Con uno fijo solo cubriría hasta 255 hypot.
    assert_reference_bits(adversarial_points(kind, count=400, dimensions=300), True, fixed=200)


def assert_reference_bits(points, reuse, fixed=500):
    ids, fixed, candidates, capacity = retention(points, fixed=fixed)
    options = dict(client_block_size=64, fixed_block_size=128, max_swaps=8)
    reference = select_anchored_medoids(points, ids, fixed, candidates, capacity, **options)
    bounded = select_anchored_medoids(
        points,
        ids,
        fixed,
        candidates,
        capacity,
        bounded_background=True,
        reuse_distances=reuse,
        **options,
    )
    assert bounded.bounded_background and not reference.bounded_background
    assert bounded.reused_distances is reuse
    assert repr(bounded.objective) == repr(reference.objective)
    changed = {"bounded_background", "reused_distances", "estimated_peak_bytes"}
    for field in dataclasses.fields(reference):
        if field.name not in changed:
            assert getattr(bounded, field.name) == getattr(reference, field.name), field.name


def test_bounded_background_computes_few_reference_distances_and_falls_back_on_dense_blocks(
    monkeypatch,
):
    from mars_titan.cm import anchored_medoids, medoids

    chain, compute = anchored_medoids._reference_chain, medoids._Costs._compute
    work = {"pairs": 0, "tiles": 0}

    def counted_chain(xt, yt, rows, columns, buffers):
        work["pairs"] += len(rows)
        return chain(xt, yt, rows, columns, buffers)

    def counted_compute(self, start, end, indices, distance):
        if len(self.candidates) == 500:
            work["tiles"] += (end - start) * len(indices)
        return compute(self, start, end, indices, distance)

    monkeypatch.setattr(anchored_medoids, "_reference_chain", counted_chain)
    monkeypatch.setattr(medoids._Costs, "_compute", counted_compute)
    # Centrar las coordenadas mantiene útil la cota en una nube lejana y estrecha.
    for kind, sparse in (("normal", True), ("far_cloud", True), ("constant", False)):
        work.update(pairs=0, tiles=0)
        points = adversarial_points(kind)
        ids, fixed, candidates, capacity = retention(points)
        result = select_anchored_medoids(
            points, ids, fixed, candidates, capacity, bounded_background=True
        )
        pairs = (len(points) - len(fixed)) * len(fixed)
        # Los pares se cuentan como en la referencia aunque se calculen menos.
        assert result.background_pairs == pairs
        if sparse:
            assert work["tiles"] == 0 and work["pairs"] < pairs / 50
        else:
            # Sin una cota útil, los bloques se calculan enteros como en la referencia.
            assert work["tiles"] == pairs


def test_bounded_background_only_applies_to_euclidean_fp32_and_fits_its_memory():
    points = adversarial_points("normal")
    ids, fixed, candidates, capacity = retention(points)
    for other, metric in ((points.astype(np.float64), "euclidean"), (points, "l1")):
        result = select_anchored_medoids(
            other, ids, fixed, candidates, capacity, metric=metric, bounded_background=True
        )
        assert not result.bounded_background
    reference = select_anchored_medoids(points, ids, fixed, candidates, capacity)
    tight = select_anchored_medoids(
        points,
        ids,
        fixed,
        candidates,
        capacity,
        bounded_background=True,
        max_working_bytes=reference.estimated_peak_bytes,
    )
    assert tight == reference
    import tracemalloc

    tracemalloc.start()
    try:
        result = select_anchored_medoids(
            points, ids, fixed, candidates, capacity, bounded_background=True
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.bounded_background and peak <= result.estimated_peak_bytes
