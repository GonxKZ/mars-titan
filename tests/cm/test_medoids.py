"""Oráculos pequeños y límites de la selección de representantes reales."""

import itertools
import math

import numpy as np
import pytest

from mars_titan.cm.medoids import MedoidBudgetExceeded, select_medoids


def oracle(clients, candidates, ids, capacity, metric):
    def cost(indices):
        total = 0.0
        for client in clients:
            distances = []
            for index in indices:
                difference = client - candidates[index]
                distances.append(
                    math.dist(client, candidates[index])
                    if metric == "euclidean"
                    else sum(abs(int(x)) for x in difference)
                )
            total += min(distances)
        return total

    options = itertools.combinations(range(len(candidates)), min(capacity, len(candidates)))
    return min((cost(indices), tuple(sorted(ids[i] for i in indices))) for indices in options)


@pytest.mark.parametrize("metric", ["euclidean", "l1"])
def test_enumeration_matches_independent_oracle(metric):
    clients = np.array([[0, 1], [4, 2], [5, 9], [12, 8]], dtype=np.int64)
    candidates = np.array([[1, 1], [4, 3], [8, 7], [10, 10]], dtype=np.int64)
    if metric == "euclidean":
        clients, candidates = clients.astype(float), candidates.astype(float)
    ids = ["d", "b", "c", "a"]
    result = select_medoids(
        clients,
        ["u", "v", "w", "x"],
        candidates,
        ids,
        2,
        backend="enumeration",
        metric=metric,
        client_block_size=2,
    )
    expected_cost, expected_ids = oracle(clients, candidates, ids, 2, metric)
    assert result.objective == pytest.approx(expected_cost)
    assert result.candidate_ids == expected_ids
    assert tuple(ids[index] for index in result.candidate_indices) == expected_ids
    assert result.status == "enumerated"
    assert result.objective_evaluations == 6
    assert result.arithmetic == ("int64_distances_python_int_sum" if metric == "l1" else "float64")


@pytest.mark.parametrize("backend", ["greedy_swap", "enumeration"])
def test_duplicates_keep_multiplicity_and_canonical_real_representative(backend):
    clients = np.array([[0], [0], [10]], dtype=np.int64)
    candidates = np.array([[10], [0], [0]], dtype=np.int64)
    result = select_medoids(
        clients, ["e0", "e1", "e2"], candidates, ["z", "b", "a"], 1, backend=backend, metric="l1"
    )
    assert result.candidate_ids == ("a",)
    assert result.candidate_indices == (2,)
    assert result.objective == 10
    assert isinstance(result.objective, int)


@pytest.mark.parametrize("backend", ["greedy_swap", "enumeration"])
def test_permutations_and_blocks_preserve_selection(backend):
    points = np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 8.0], [9.0, 2.0], [9.0, 3.0]])
    ids = np.array(["c", "a", "e", "b", "d"])
    baseline = select_medoids(points, ids, points, ids, 2, backend=backend)
    for permutation in ([4, 2, 0, 3, 1], [3, 2, 1, 4, 0]):
        result = select_medoids(
            points[permutation],
            ids[permutation],
            points[::-1],
            ids[::-1],
            2,
            backend=backend,
            client_block_size=2,
            candidate_block_size=1,
        )
        assert result.candidate_ids == baseline.candidate_ids
        assert result.objective == pytest.approx(baseline.objective, abs=1e-14)
        assert tuple(ids[::-1][list(result.candidate_indices)]) == result.candidate_ids


def test_shared_id_must_have_identical_representation():
    with pytest.raises(ValueError, match="representación"):
        select_medoids(np.array([[1.0]]), ["same"], np.array([[2.0]]), ["same"], 1)
    result = select_medoids(np.array([[1.0]]), ["same"], np.array([[1.0]]), ["same"], 1)
    assert result.objective == 0


def test_capacities_and_empty_sets_have_explicit_contracts():
    empty = np.empty((0, 2))
    points = np.ones((2, 2))
    for clients, ids in ((empty, []), (points, ["a", "b"])):
        with pytest.raises(ValueError, match="capacity"):
            select_medoids(clients, ids, points, ["a", "b"], 0)
    result = select_medoids(empty, [], empty, [], 3)
    assert result.candidate_ids == () and result.objective == 0
    assert result.status == "empty_clients"
    with pytest.raises(ValueError, match="candidatos"):
        select_medoids(points, ["a", "b"], empty, [], 1)
    result = select_medoids(points, ["a", "b"], points, ["a", "b"], 20)
    assert result.candidate_ids == ("a", "b") and result.objective == 0
    assert result.status == "all_candidates"


@pytest.mark.parametrize("metric", ["squared_euclidean", "cosine", "published"])
def test_unknown_metric_rejected(metric):
    with pytest.raises(ValueError):
        select_medoids(np.zeros((1, 1)), ["a"], np.zeros((1, 1)), ["a"], 1, metric=metric)


@pytest.mark.parametrize(
    "points", [np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-np.inf]]), np.array([[1j]])]
)
def test_nonfinite_or_complex_coordinates_rejected(points):
    with pytest.raises((TypeError, ValueError)):
        select_medoids(points, ["a"], points, ["a"], 1)


def test_integer_l1_checks_distance_overflow_but_not_python_total():
    limit = np.iinfo(np.int64).max
    with pytest.raises(ValueError, match="int64"):
        select_medoids(
            np.array([[-1]], dtype=np.int64),
            ["a"],
            np.array([[limit]], dtype=np.int64),
            ["b"],
            1,
            metric="l1",
        )
    result = select_medoids(
        np.array([[limit], [limit]], dtype=np.int64),
        ["a", "b"],
        np.array([[0]], dtype=np.int64),
        ["c"],
        1,
        metric="l1",
    )
    assert result.objective == 2 * int(limit)
    assert isinstance(result.objective, int)


def test_euclidean_is_not_squared_and_does_not_square_overflow_or_underflow():
    for scale in (1.0, 1e200, 1e-200):
        result = select_medoids(
            np.array([[3.0 * scale, 4.0 * scale]]), ["a"], np.zeros((1, 2)), ["b"], 1
        )
        assert result.objective == pytest.approx(5 * scale, rel=1e-14, abs=0)
    with pytest.raises(ValueError, match="finito"):
        select_medoids(np.array([[1e308]]), ["a"], np.array([[-1e308]]), ["b"], 1)


def test_distance_and_combination_budgets_fail_explicitly():
    points = np.arange(8, dtype=np.float64)[:, None]
    ids = list("abcdefgh")
    with pytest.raises(MedoidBudgetExceeded, match="pares"):
        select_medoids(points, ids, points, ids, 2, max_distance_pairs=10)
    with pytest.raises(MedoidBudgetExceeded, match="combinaciones"):
        select_medoids(points, ids, points, ids, 4, backend="enumeration", max_combinations=10)
    with pytest.raises(MedoidBudgetExceeded, match="memoria"):
        select_medoids(points, ids, points, ids, 2, max_working_bytes=1)


def test_two_axes_are_blocked_to_fit_buffer_budget():
    clients = np.arange(2000, dtype=float)[:, None]
    candidates = np.arange(4, dtype=float)[:, None]
    result = select_medoids(
        clients,
        [f"e{i:05d}" for i in range(2000)],
        candidates,
        list("abcd"),
        4,
        client_block_size=2000,
        max_working_bytes=650_000,
    )
    assert result.distance_tile_shape[0] < 2000
    assert result.distance_tile_shape[1] <= 4
    assert result.estimated_peak_bytes <= 650_000
    assert result.distance_pairs == 8000


def test_input_arrays_and_external_rng_are_unchanged():
    points = np.array([[2.0], [0.0], [1.0]])
    before = points.copy()
    state = np.random.get_state()
    select_medoids(points, list("abc"), points, list("abc"), 1)
    np.testing.assert_array_equal(points, before)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity": True},
        {"capacity": -1},
        {"backend": "pam"},
        {"candidate_block_size": 0},
        {"client_block_size": 0},
        {"max_swaps": -1},
        {"max_distance_pairs": True},
    ],
)
def test_bad_configuration_rejected(kwargs):
    parameters = {"capacity": 1, **kwargs}
    with pytest.raises((TypeError, ValueError)):
        select_medoids(np.zeros((1, 1)), ["a"], np.zeros((1, 1)), ["a"], **parameters)


def test_bad_id_and_shape_contracts_rejected():
    points = np.zeros((2, 1))
    for ids in (["a", "a"], ["a"], ["a", ""], ["a", 1]):
        with pytest.raises((TypeError, ValueError)):
            select_medoids(points, ids, points, ["a", "b"], 1)
    with pytest.raises(ValueError):
        select_medoids(points, ["a", "b"], np.zeros((2, 2)), ["a", "b"], 1)


def test_integer_to_float_conversion_cannot_silently_lose_coordinates():
    points = np.array([[2**53], [2**53 + 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="float64"):
        select_medoids(points, ["a", "b"], points, ["a", "b"], 1)


def test_swaps_improve_greedy_and_partial_result_is_identified():
    points = np.array(
        [[7, -10], [1, -9], [-9, 1], [3, 12], [-20, 5], [12, 7], [13, -6]], dtype=np.int64
    )
    ids = list("abcdefg")
    initial = select_medoids(points, ids, points, ids, 2, metric="l1", max_swaps=0)
    result = select_medoids(points, ids, points, ids, 2, metric="l1")
    optimum = select_medoids(points, ids, points, ids, 2, metric="l1", backend="enumeration")
    assert initial.status == "swap_limit" and initial.objective == 86
    assert result.status == "one_swap_local" and result.objective == 77
    assert result.swaps == 2 and result.candidate_ids == ("a", "c")
    assert result.objective >= optimum.objective
    for outgoing in result.candidate_ids:
        for incoming in set(ids) - set(result.candidate_ids):
            trial = tuple(sorted((set(result.candidate_ids) - {outgoing}) | {incoming}))
            cost, _ = oracle(points, points[[ids.index(item) for item in trial]], trial, 2, "l1")
            assert cost >= result.objective


def test_symmetric_ties_and_zero_optimum_use_canonical_ids_without_ratios():
    clients = np.array([[0, 0]], dtype=np.int64)
    candidates = np.array([[1, 0], [-1, 0]], dtype=np.int64)
    for backend in ("greedy_swap", "enumeration"):
        result = select_medoids(
            clients, ["e"], candidates, ["z", "a"], 1, metric="l1", backend=backend
        )
        assert result.candidate_ids == ("a",) and result.objective == 1
        zero = select_medoids(
            clients,
            ["e"],
            np.zeros((3, 2), dtype=np.int64),
            list("zba"),
            2,
            metric="l1",
            backend=backend,
        )
        assert zero.candidate_ids == ("a", "b") and zero.objective == 0


def test_float_l1_and_integer_euclidean_have_correct_costs():
    result = select_medoids(
        np.array([[0.5, 1.5]]), ["e"], np.array([[2.5, 4.5]]), ["f"], 1, metric="l1"
    )
    assert result.objective == 5.0 and result.arithmetic == "float64"
    integer = select_medoids(np.array([[3, 4]]), ["e"], np.array([[0, 0]]), ["f"], 1)
    assert integer.objective == 5.0 and integer.arithmetic == "float64"


def test_mixed_coordinate_kinds_and_unsupported_inputs_rejected():
    with pytest.raises(ValueError):
        select_medoids(np.array([[1]]), ["a"], np.array([[1.0]]), ["a"], 1)
    with pytest.raises(TypeError):
        select_medoids([[1.0]], ["a"], np.array([[1.0]]), ["a"], 1)
    with pytest.raises(ValueError):
        select_medoids(np.array([1.0]), ["a"], np.array([[1.0]]), ["a"], 1)


def test_cost_overflow_is_rejected_even_when_each_distance_is_finite():
    with pytest.raises(ValueError, match="finito"):
        select_medoids(np.full((4, 1), 1e308), list("abcd"), np.zeros((1, 1)), ["e"], 1)


def test_integer_coordinate_extent_is_checked_across_dimensions():
    edge = np.iinfo(np.int64).max // 2 + 1
    with pytest.raises(ValueError, match="int64"):
        select_medoids(
            np.array([[edge, edge]], dtype=np.int64),
            ["a"],
            np.zeros((1, 2), dtype=np.int64),
            ["b"],
            1,
            metric="l1",
        )


def test_enumeration_limit_is_checked_without_forming_huge_combination_count(monkeypatch):
    monkeypatch.setattr(math, "comb", lambda *args: pytest.fail("Conteo sin límite"))
    points = np.zeros((100, 1))
    with pytest.raises(MedoidBudgetExceeded, match="combinaciones"):
        select_medoids(
            points,
            [f"e{i}" for i in range(100)],
            points,
            [f"f{i}" for i in range(100)],
            50,
            backend="enumeration",
        )


def test_oversized_id_is_rejected_before_encoding_it():
    class LargeId(str):
        def encode(self, *args, **kwargs):
            pytest.fail("No debe reservarse una copia UTF-8 de un ID sobredimensionado")

    with pytest.raises(ValueError, match="ID"):
        select_medoids(np.zeros((1, 1)), [LargeId("x" * 257)], np.zeros((1, 1)), ["a"], 1)


def test_empty_clients_do_not_spend_the_enumeration_budget():
    result = select_medoids(
        np.empty((0, 1)),
        [],
        np.zeros((4, 1)),
        list("abcd"),
        2,
        backend="enumeration",
        max_combinations=1,
        max_distance_pairs=1,
    )
    assert result.status == "empty_clients"
    assert result.objective == 0
    assert result.distance_pairs == result.objective_evaluations == 0


def test_buffer_estimate_counts_numpy_unicode_id_objects():
    import tracemalloc

    count = 2000
    clients = np.zeros((count, 1))
    ids = np.array(["a" * 247 + f"{index:05d}" + "\U0001f600" for index in range(count)])
    tracemalloc.start()
    try:
        result = select_medoids(clients, ids, np.zeros((1, 1)), ["f"], 1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= result.estimated_peak_bytes
