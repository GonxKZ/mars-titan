"""Retención con centros fijos reales y candidatos variables restringidos.

El objetivo incluye a todos los clientes de entrada. La elegibilidad temporal,
la representación congelada y la publicación del banco son externas a esta API.
"""

import itertools
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mars_titan.cm.medoids import (
    MedoidBudgetExceeded,
    _check_combinations,
    _Costs,
    _enumerate,
    _greedy_swap,
    _id_bytes,
    _integer,
    _integer_geometry,
    _ordered,
    _point_shape,
    _workspace,
)


@dataclass(frozen=True)
class AnchoredMedoidSelection:
    """Representantes reales, objetivo completo y coste de la búsqueda restringida."""

    fixed_ids: tuple[str, ...]
    variable_ids: tuple[str, ...]
    retained_ids: tuple[str, ...]
    retained_indices: tuple[int, ...]
    objective: int | float
    algorithm: str
    metric: str
    status: str
    swaps: int
    background_pairs: int
    variable_pairs: int
    distance_pairs: int
    objective_evaluations: int
    estimated_peak_bytes: int
    background_tile_shape: tuple[int, int]
    variable_tile_shape: tuple[int, int]
    coordinate_dtype: str
    distance_arithmetic: str
    background_backend: str
    numpy_version: str
    scipy_version: str | None


def _indices(ids: Sequence[str], ordered_ids: tuple[str, ...]) -> tuple[int, ...]:
    requested = tuple(sorted(ids))
    if any(left == right for left, right in itertools.pairwise(requested)):
        raise ValueError("Los IDs de centros no pueden repetirse")
    result = []
    for identifier in requested:
        index = bisect_left(ordered_ids, identifier)
        if index == len(ordered_ids) or ordered_ids[index] != identifier:
            raise ValueError("Cada centro debe ser un cliente real de E")
        result.append(index)
    return tuple(result)


def _scipy_backend(points: np.ndarray, metric: str, backend: str):
    if backend == "numpy":
        return None, None
    if backend != "scipy_cdist_fp32":
        raise ValueError("Backend de fondo no admitido")
    if points.dtype.kind != "f" or points.dtype.itemsize != 4:
        raise TypeError("El fondo cdist exige coordenadas FP32, sin conversión implícita")
    if metric != "euclidean":
        raise ValueError("El fondo cdist FP32 solo admite distancia euclídea")
    try:
        import scipy
        from scipy.spatial.distance import cdist
    except ImportError as error:
        raise ImportError("El backend solicitado necesita SciPy del extra research") from error
    return cdist, scipy.__version__


def _background(clients, fixed, metric, integer, rows, columns, max_pairs, cdist):
    """Calcula b por dos ejes, contabilizando todos los pares frente a centros fijos."""
    pairs = len(clients) * len(fixed)
    if pairs > max_pairs:
        raise MedoidBudgetExceeded("El fondo supera el presupuesto de pares de distancia")
    nearest = np.full(
        len(clients),
        np.iinfo(np.int64).max if integer else np.inf,
        dtype=np.int64 if integer else np.float64,
    )
    if cdist is None:
        engine = _Costs(clients, fixed, metric, integer, rows, columns, max_pairs)
    else:
        buffer = np.empty(rows * columns, dtype=np.float64)
    for start in range(0, len(clients), rows):
        end = min(start + rows, len(clients))
        for offset in range(0, len(fixed), columns):
            stop = min(offset + columns, len(fixed))
            if cdist is None:
                tile = engine._distance(start, end, tuple(range(offset, stop)))
            else:
                tile = buffer[: (end - start) * (stop - offset)].reshape(end - start, stop - offset)
                cdist(clients[start:end], fixed[offset:stop], metric="euclidean", out=tile)
                if not np.isfinite(tile).all():
                    raise ValueError("Una distancia del fondo produjo un resultado no finito")
            np.minimum(nearest[start:end], tile.min(axis=1), out=nearest[start:end])
    return nearest, pairs


def select_anchored_medoids(
    points: np.ndarray,
    ids: Sequence[str],
    fixed_ids: Sequence[str],
    candidate_ids: Sequence[str],
    capacity: int,
    *,
    algorithm: str = "greedy_swap",
    metric: str = "euclidean",
    background_backend: str = "numpy",
    client_block_size: int = 256,
    fixed_block_size: int = 256,
    candidate_block_size: int = 16,
    max_working_bytes: int = 64 * 1024**2,
    max_distance_pairs: int = 50_000_000,
    max_combinations: int = 10_000,
    max_swaps: int = 100,
) -> AnchoredMedoidSelection:
    """Selecciona hasta capacity centros, incluidos los fijos, sobre el E completo.

    Los centros fijos y candidatos son IDs distintos contenidos en E. Los
    clientes fijos aportan cero. Los demás minimizan su distancia al conjunto
    fijo y al variable. La enumeración solo es óptima sobre esta restricción.

    El fondo NumPy conserva la referencia general. cdist exige FP32 y calcula
    distancias euclídeas FP64. La selección variable mantiene la referencia
    NumPy. La salida identifica ambos tipos y las versiones de bibliotecas.
    Los límites de pares y buffers son conjuntos para fondo y selección.
    """
    for value, name in (
        (capacity, "capacity"),
        (client_block_size, "client_block_size"),
        (fixed_block_size, "fixed_block_size"),
        (candidate_block_size, "candidate_block_size"),
        (max_working_bytes, "max_working_bytes"),
        (max_distance_pairs, "max_distance_pairs"),
        (max_combinations, "max_combinations"),
    ):
        _integer(value, name)
    _integer(max_swaps, "max_swaps", 0)
    if algorithm not in ("greedy_swap", "enumeration") or metric not in ("euclidean", "l1"):
        raise ValueError("Algoritmo o métrica no admitidos")
    n, dimensions = _point_shape(points)
    cdist, scipy_version = _scipy_backend(points, metric, background_backend)
    if cdist is not None and dimensions > 4096:
        raise ValueError("El backend cdist admite como máximo 4096 dimensiones FP32")
    if len(fixed_ids) > capacity:
        raise ValueError("capacity es inferior al número de centros fijos")
    centers = len(fixed_ids) + len(candidate_ids)
    # Partición de E, correspondencias y vector de fondo, además de los buffers compartidos.
    extra_bytes = 40 * n
    remaining_bytes = max_working_bytes - extra_bytes
    _workspace(
        n,
        centers,
        dimensions,
        client_block_size,
        max(fixed_block_size, candidate_block_size),
        remaining_bytes,
        0,
    )
    id_bytes = (
        _id_bytes(ids, n)
        + _id_bytes(fixed_ids, len(fixed_ids))
        + _id_bytes(candidate_ids, len(candidate_ids))
    )
    rows, columns, estimated = _workspace(
        n,
        centers,
        dimensions,
        client_block_size,
        max(fixed_block_size, candidate_block_size),
        remaining_bytes,
        id_bytes,
    )
    integer = _integer_geometry(points, points, metric)
    normalized, ordered_ids, original = _ordered(
        points, ids, np.dtype(np.int64 if integer else np.float64), rows
    )
    fixed_indices = _indices(fixed_ids, ordered_ids)
    candidates = _indices(candidate_ids, ordered_ids)
    if not set(fixed_indices).isdisjoint(candidates):
        raise ValueError("Los centros fijos y candidatos deben ser disjuntos")
    if n and not centers:
        raise ValueError("No hay centros admisibles para los clientes")
    variable_capacity = min(capacity - len(fixed_indices), len(candidates))
    if algorithm == "enumeration" and n and variable_capacity:
        _check_combinations(len(candidates), variable_capacity, max_combinations)
    mask = np.ones(n, dtype=bool)
    mask[list(fixed_indices)] = False
    clients = normalized[mask]
    fixed = normalized[list(fixed_indices)]
    variable = normalized[list(candidates)]
    rows = max(1, min(rows, len(clients)))
    fixed_columns = max(1, min(columns, fixed_block_size, len(fixed)))
    variable_columns = max(1, min(columns, candidate_block_size, len(variable)))
    background, background_pairs = (None, 0)
    if len(fixed) and len(clients):
        background, background_pairs = _background(
            clients, fixed, metric, integer, rows, fixed_columns, max_distance_pairs, cdist
        )
    costs = _Costs(
        clients,
        variable,
        metric,
        integer,
        rows,
        variable_columns,
        max_distance_pairs - background_pairs,
        background=background,
    )
    swaps = 0
    if not n:
        selected, objective, status = (), 0, "empty_clients"
    elif not len(clients):
        selected, objective, status = (), 0, "fixed_only"
    elif variable_capacity == 0:
        selected, objective, status = (), costs._sum(background), "fixed_only"
    elif variable_capacity == len(candidates):
        selected = tuple(range(len(candidates)))
        objective, status = costs.state(selected), "all_candidates"
    elif algorithm == "enumeration":
        selected, objective, swaps, _ = _enumerate(costs, len(candidates), variable_capacity)
        status = "enumerated_restricted"
    else:
        selected, objective, swaps, status = _greedy_swap(
            costs, len(candidates), variable_capacity, max_swaps
        )
        if status == "one_swap_local":
            status = "one_swap_local_restricted"
    chosen = tuple(candidates[index] for index in selected)
    retained = tuple(sorted((*fixed_indices, *chosen)))
    return AnchoredMedoidSelection(
        tuple(ordered_ids[index] for index in fixed_indices),
        tuple(ordered_ids[index] for index in chosen),
        tuple(ordered_ids[index] for index in retained),
        tuple(original[index] for index in retained),
        objective,
        algorithm,
        metric,
        status,
        swaps,
        background_pairs,
        costs.pairs,
        background_pairs + costs.pairs,
        costs.evaluations,
        estimated + extra_bytes,
        (rows, fixed_columns),
        (rows, variable_columns),
        points.dtype.str,
        "int64_distances_python_int_sum" if integer else "float64",
        background_backend,
        np.__version__,
        scipy_version,
    )
