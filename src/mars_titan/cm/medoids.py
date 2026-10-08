"""Selección determinista de representantes reales con distancias por bloques.

La heurística greedy con intercambios no tiene un factor de aproximación
atribuido. La enumeración solo es un oráculo pequeño. La admisión temporal,
las etiquetas y la versión de la representación pertenecen al consumidor.
"""

import itertools
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


class MedoidBudgetExceeded(RuntimeError):
    """El cálculo no ha terminado dentro de su presupuesto explícito."""


@dataclass(frozen=True)
class MedoidSelection:
    """Representantes, coste calculado y trabajo realizado, sin promediar episodios."""

    candidate_ids: tuple[str, ...]
    candidate_indices: tuple[int, ...]
    objective: int | float
    backend: str
    metric: str
    arithmetic: str
    status: str
    swaps: int
    distance_pairs: int
    objective_evaluations: int
    estimated_peak_bytes: int
    distance_tile_shape: tuple[int, int]


def _integer(value: int, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} debe ser un entero mayor o igual que {minimum}")


def _point_shape(points: np.ndarray) -> tuple[int, int]:
    if not isinstance(points, np.ndarray):
        raise TypeError(
            "Las coordenadas deben ser matrices NumPy, sin conversión implícita de tensores"
        )
    if points.ndim != 2 or points.shape[1] < 1:
        raise ValueError("Las coordenadas necesitan dos ejes y al menos una dimensión")
    if points.dtype.kind not in ("i", "f") or points.dtype.itemsize > 8:
        raise TypeError("Las coordenadas deben ser reales o enteros con signo de hasta 64 bits")
    return points.shape


def _check_combinations(count: int, capacity: int, limit: int) -> None:
    combinations = 1
    for index in range(1, min(capacity, count - capacity) + 1):
        combinations = combinations * (count - index + 1) // index
        if combinations > limit:
            raise MedoidBudgetExceeded("El número de combinaciones supera el presupuesto")


def _workspace(
    n: int,
    f: int,
    dimensions: int,
    client_block: int,
    candidate_block: int,
    budget: int,
    id_bytes: int,
) -> tuple[int, int, int]:
    """Estima copias, índices, metadatos, estado y temporales antes de reservarlos."""
    rows, columns = max(1, min(n, client_block)), max(1, min(f, candidate_block))
    # Indexar IDs NumPy crea escalares. Se cuenta su tamaño y el de los índices.
    fixed = 16 * (n + f) * dimensions + 128 * (n + f) + id_bytes + 24 * n + 128 * f + 65536
    while True:
        parts = math.ceil(n / rows)
        estimated = (
            fixed
            + 24 * rows * columns
            + 12 * rows
            + rows * dimensions
            + 8 * columns * dimensions
            + 64 * columns * (parts + 1)
        )
        if estimated <= budget:
            return rows, columns, estimated
        if rows > 1:
            rows = max(1, rows // 2)
        elif columns > 1:
            columns = max(1, columns // 2)
            rows = max(1, min(n, client_block))
        else:
            raise MedoidBudgetExceeded("El presupuesto de memoria no permite los buffers propios")


def _id_bytes(ids: Sequence[str], count: int) -> int:
    if len(ids) != count:
        raise ValueError("El número de IDs no coincide con las coordenadas")
    size = 0
    for identifier in ids:
        if (
            not isinstance(identifier, str)
            or not identifier
            or len(identifier) > 256
            or len(identifier.encode("utf-8")) > 256
        ):
            raise ValueError("Cada ID debe ser una cadena no vacía de hasta 256 bytes")
        size += sys.getsizeof(identifier)
    return size


def _ordered(
    points: np.ndarray, ids: Sequence[str], dtype: np.dtype, block: int
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
    order = tuple(sorted(range(len(ids)), key=ids.__getitem__))
    ordered_ids = tuple(ids[index] for index in order)
    if any(first == second for first, second in itertools.pairwise(ordered_ids)):
        raise ValueError("Los IDs deben ser únicos dentro de cada conjunto")
    for start in range(0, len(points), block):
        if not np.isfinite(points[start : start + block]).all():
            raise ValueError("Las coordenadas contienen NaN o infinito")
    return points[list(order)].astype(dtype, copy=False), ordered_ids, order


def _check_shared_ids(
    clients: np.ndarray,
    client_ids: tuple[str, ...],
    candidates: np.ndarray,
    candidate_ids: tuple[str, ...],
) -> None:
    left = right = 0
    while left < len(client_ids) and right < len(candidate_ids):
        if client_ids[left] == candidate_ids[right]:
            if not np.array_equal(clients[left], candidates[right]):
                raise ValueError("Un ID compartido debe tener la misma representación")
            left += 1
            right += 1
        elif client_ids[left] < candidate_ids[right]:
            left += 1
        else:
            right += 1


def _integer_geometry(clients: np.ndarray, candidates: np.ndarray, metric: str) -> bool:
    if clients.dtype.kind != candidates.dtype.kind:
        raise ValueError("Clientes y candidatos deben usar la misma clase de coordenadas")
    integer = clients.dtype.kind == "i"
    if not integer or not len(clients) or not len(candidates):
        return integer and metric == "l1"
    if metric == "euclidean":
        for points in (clients, candidates):
            if int(points.min()) < -(2**53) or int(points.max()) > 2**53:
                raise ValueError("La conversión a float64 podría perder coordenadas enteras")
        return False
    extent = 0
    for dimension in range(clients.shape[1]):
        low = min(int(clients[:, dimension].min()), int(candidates[:, dimension].min()))
        high = max(int(clients[:, dimension].max()), int(candidates[:, dimension].max()))
        extent += high - low
        if extent > np.iinfo(np.int64).max:
            raise ValueError("La distancia L1 podría desbordar int64")
    return True


class _Costs:
    """Estado lineal en clientes y dos buffers de distancia, sin tabla global."""

    def __init__(
        self, clients, candidates, metric, integer, rows, columns, max_pairs, *, background=None
    ):
        self.clients, self.candidates = clients, candidates
        self.metric, self.integer = metric, integer
        self.rows, self.columns, self.max_pairs = rows, columns, max_pairs
        self.pairs = self.evaluations = 0
        self.background = background
        dtype = np.int64 if integer else np.float64
        self.distances = np.empty((rows, columns), dtype=dtype)
        self.scratch = np.empty_like(self.distances)
        self.nearest = np.empty(len(clients), dtype=dtype)
        self.second = np.empty_like(self.nearest)
        self.owner = np.empty(len(clients), dtype=np.int64)
        self.base = np.empty(rows, dtype=dtype)
        self.mask = np.empty(rows, dtype=bool)

    def _sum(self, values):
        if self.integer:
            return sum(int(value) for value in values)
        try:
            total = math.fsum(values)
        except OverflowError as error:
            raise ValueError("El coste agregado no es finito") from error
        if not math.isfinite(total):
            raise ValueError("El coste agregado no es finito")
        return total

    def _distance(self, start, end, indices):
        count = (end - start) * len(indices)
        if self.pairs + count > self.max_pairs:
            raise MedoidBudgetExceeded("Se ha agotado el presupuesto de pares de distancia")
        self.pairs += count
        target = self.candidates[list(indices)]
        distance = self.distances[: end - start, : len(indices)]
        scratch = self.scratch[: end - start, : len(indices)]
        distance.fill(0)
        try:
            with np.errstate(over="raise", invalid="raise"):
                for dimension in range(self.clients.shape[1]):
                    np.subtract(
                        self.clients[start:end, dimension, None],
                        target[None, :, dimension],
                        out=scratch,
                    )
                    if self.metric == "euclidean":
                        np.hypot(distance, scratch, out=distance)
                    else:
                        np.abs(scratch, out=scratch)
                        np.add(distance, scratch, out=distance)
        except FloatingPointError as error:
            raise ValueError("Una distancia produjo un resultado no finito") from error
        if not np.isfinite(distance).all():
            raise ValueError("Una distancia produjo un resultado no finito")
        return distance

    def reset_nearest(self):
        if self.background is None:
            self.nearest.fill(np.iinfo(np.int64).max if self.integer else np.inf)
        else:
            self.nearest[:] = self.background

    def state(self, selected):
        """Recalcula exactamente el conjunto aceptado, incluidos sus empates."""
        self.evaluations += 1
        self.reset_nearest()
        self.second[:] = self.nearest
        # El propietario fijo no se sustituye por un candidato de mayor distancia.
        self.owner.fill(-1 if self.background is None else -2)
        parts = []
        for start in range(0, len(self.clients), self.rows):
            end = min(start + self.rows, len(self.clients))
            nearest, second, owner = (
                self.nearest[start:end],
                self.second[start:end],
                self.owner[start:end],
            )
            base, mask = self.base[: end - start], self.mask[: end - start]
            for offset in range(0, len(selected), self.columns):
                indices = selected[offset : offset + self.columns]
                distance = self._distance(start, end, indices)
                for column, index in enumerate(indices):
                    value = distance[:, column]
                    np.less(value, nearest, out=mask)
                    np.logical_or(mask, owner == -1, out=mask)
                    np.minimum(second, value, out=base)
                    np.copyto(base, nearest, where=mask)
                    np.copyto(second, base)
                    np.copyto(nearest, value, where=mask)
                    np.copyto(owner, index, where=mask)
            parts.append(self._sum(nearest))
        return self._sum(parts)

    def proposals(self, candidates, outgoing=None):
        """Costes al añadir un candidato, opcionalmente retirando otro."""
        for offset in range(0, len(candidates), self.columns):
            indices = candidates[offset : offset + self.columns]
            parts = [[] for _ in indices]
            for start in range(0, len(self.clients), self.rows):
                end = min(start + self.rows, len(self.clients))
                base = self.nearest[start:end]
                if outgoing is not None:
                    base = self.base[: end - start]
                    np.copyto(base, self.nearest[start:end])
                    np.equal(self.owner[start:end], outgoing, out=self.mask[: end - start])
                    np.copyto(base, self.second[start:end], where=self.mask[: end - start])
                distance = self._distance(start, end, indices)
                temporary = self.scratch[: end - start, : len(indices)]
                np.minimum(distance, base[:, None], out=temporary)
                for column in range(len(indices)):
                    parts[column].append(self._sum(temporary[:, column]))
            self.evaluations += len(indices)
            for index, values in zip(indices, parts, strict=True):
                yield index, self._sum(values)


def _greedy_swap(costs, count, capacity, max_swaps):
    selected = ()
    costs.reset_nearest()
    for _ in range(capacity):
        available = tuple(index for index in range(count) if index not in selected)
        incoming, _ = min(costs.proposals(available), key=lambda pair: (pair[1], pair[0]))
        selected = tuple(sorted((*selected, incoming)))
        objective = costs.state(selected)
    for swaps in range(max_swaps):
        available = tuple(index for index in range(count) if index not in selected)
        best = (objective, selected)
        for outgoing in selected:
            for incoming, value in costs.proposals(available, outgoing):
                trial = tuple(sorted(index for index in (*selected, incoming) if index != outgoing))
                if value < objective and (value, trial) < best:
                    best = (value, trial)
        if best[1] == selected:
            return selected, objective, swaps, "one_swap_local"
        selected = best[1]
        objective = costs.state(selected)
        if objective != best[0]:
            raise ArithmeticError("El coste del intercambio no coincide con su recálculo")
    return selected, objective, max_swaps, "swap_limit"


def _enumerate(costs, count, capacity):
    best = None
    for selected in itertools.combinations(range(count), capacity):
        candidate = (costs.state(selected), selected)
        if best is None or candidate < best:
            best = candidate
    return best[1], best[0], 0, "enumerated"


def select_medoids(
    client_points: np.ndarray,
    client_ids: Sequence[str],
    candidate_points: np.ndarray,
    candidate_ids: Sequence[str],
    capacity: int,
    *,
    backend: str = "greedy_swap",
    metric: str = "euclidean",
    candidate_block_size: int = 16,
    client_block_size: int = 4096,
    max_working_bytes: int = 64 * 1024**2,
    max_distance_pairs: int = 50_000_000,
    max_combinations: int = 10_000,
    max_swaps: int = 100,
) -> MedoidSelection:
    """Minimiza distancias a candidatos reales, sin reglas de admisión temporal.

    Capacidad cero siempre falla. Clientes vacíos producen selección vacía.
    Con clientes y sin candidatos se falla. Si la capacidad alcanza el número
    de candidatos, se conservan todos. Un presupuesto agotado lanza excepción.
    Alcanzar max_swaps devuelve el estado parcial identificado como swap_limit.

    L1 entera usa distancias int64 sin desbordamiento y sumas Python exactas.
    El resto usa float64, con sumas fsum dentro y entre bloques canónicos de
    clientes. Cambiar los bloques puede cambiar el último bit del coste.
    La memoria indicada estima buffers y metadatos propios, no el RSS total.
    """
    for value, name in (
        (capacity, "capacity"),
        (candidate_block_size, "candidate_block_size"),
        (client_block_size, "client_block_size"),
        (max_working_bytes, "max_working_bytes"),
        (max_distance_pairs, "max_distance_pairs"),
        (max_combinations, "max_combinations"),
    ):
        _integer(value, name)
    _integer(max_swaps, "max_swaps", 0)
    if backend not in ("greedy_swap", "enumeration") or metric not in ("euclidean", "l1"):
        raise ValueError("Backend o distancia no admitidos")
    n, dimensions = _point_shape(client_points)
    count, candidate_dimensions = _point_shape(candidate_points)
    if dimensions != candidate_dimensions:
        raise ValueError("Clientes y candidatos deben compartir dimensiones")
    if n and not count:
        raise ValueError("No hay candidatos para representar a los clientes")
    effective = min(capacity, count)
    if n and backend == "enumeration":
        _check_combinations(count, effective, max_combinations)
    id_bytes = _id_bytes(client_ids, n) + _id_bytes(candidate_ids, count)
    rows, columns, estimated = _workspace(
        n, count, dimensions, client_block_size, candidate_block_size, max_working_bytes, id_bytes
    )
    integer = _integer_geometry(client_points, candidate_points, metric)
    dtype = np.dtype(np.int64 if integer else np.float64)
    clients, ordered_clients, _ = _ordered(client_points, client_ids, dtype, rows)
    candidates, ordered_candidates, original = _ordered(
        candidate_points, candidate_ids, dtype, rows
    )
    _check_shared_ids(clients, ordered_clients, candidates, ordered_candidates)
    costs = _Costs(clients, candidates, metric, integer, rows, columns, max_distance_pairs)
    if not n:
        selected, objective, swaps, status = (), 0, 0, "empty_clients"
    elif effective == count:
        selected = tuple(range(count))
        objective, swaps, status = costs.state(selected), 0, "all_candidates"
    elif backend == "enumeration":
        selected, objective, swaps, status = _enumerate(costs, count, effective)
    else:
        selected, objective, swaps, status = _greedy_swap(costs, count, effective, max_swaps)
    return MedoidSelection(
        tuple(ordered_candidates[index] for index in selected),
        tuple(original[index] for index in selected),
        objective,
        backend,
        metric,
        "int64_distances_python_int_sum" if integer else "float64",
        status,
        swaps,
        costs.pairs,
        costs.evaluations,
        estimated,
        (rows, columns),
    )
