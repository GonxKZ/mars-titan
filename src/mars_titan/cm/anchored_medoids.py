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
    _reuse_bytes,
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
    reused_distances: bool = False
    bounded_background: bool = False


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
    if backend != "scipy_cdist_fp32_exploratory":
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


# Unidad de redondeo de float64 para las cotas de `_bounded_minimum`.
_UNIT = 2.0**-53
# Fracción de pares supervivientes de un bloque a partir de la cual se calcula el bloque entero.
_DENSE = 8


def _reference_chain(xt, yt, rows, columns, buffers):
    """Distancia de referencia de los pares (rows, columns): resta y hypot por dimensión.

    Repite las operaciones de `_Costs._compute` en el mismo orden. Las coordenadas se reúnen
    en `buffers` (distancia, diferencia, filas y columnas) para no reservar memoria por par.
    Los índices deben ser contiguos para que `take` no los copie en cada dimensión.
    """
    distance, scratch, left, right = (buffer[: len(rows)] for buffer in buffers)
    distance.fill(0)
    for dimension in range(len(xt)):
        np.take(xt[dimension], rows, out=left, mode="clip")
        np.take(yt[dimension], columns, out=right, mode="clip")
        np.subtract(left, right, out=scratch)
        np.hypot(distance, scratch, out=distance)
    return distance


def _bounded_bytes(clients, fixed, dimensions, rows, columns):
    """Memoria que reserva `_bounded_minimum`, con el peor caso de pares supervivientes.

    Cuenta las copias centradas y traspuestas, las normas, los tres bloques FP64 y la
    máscara, los índices de los supervivientes (como mucho un octavo del bloque, porque con
    más se calcula entero), los buffers de la cadena de referencia, los vectores por fila,
    la tupla de columnas del bloque denso, los temporales de NumPy y las cabeceras. El
    vector de fondo ya está en la memoria de la referencia.

    Las reservas internas de la BLAS no entran. OpenBLAS guarda un área de trabajo por
    proceso desde su primer producto de matrices y, si reparte uno entre varios hilos,
    reserva durante la llamada su tabla de tareas. Su medida va aparte en
    `benchmarks/anchored_medoid_memory.py`.
    """
    tile = rows * columns
    pairs = max(rows, tile // _DENSE)
    # Temporales de NumPy que no coinciden en el tiempo: los buffers del iterador de una
    # ufunc con dos operandos difundidos (`getbufsize()` elementos cada uno) y la copia
    # contigua que hace `argmin` de un bloque parcial.
    transient = max(16 * min(tile, np.getbufsize()), 8 * tile)
    return (
        8 * dimensions
        + 16 * (clients + fixed) * dimensions
        + 16 * (clients + fixed)
        + 25 * tile
        + 16 * (tile // _DENSE)
        + 48 * pairs
        + 56 * rows
        + 48 * columns
        + transient
        + 16384
    )


def _bounded_minimum(clients, fixed, rows, columns, engine):
    """Mínimo por cliente de la distancia de referencia a los centros fijos.

    Solo se calcula la distancia de referencia de los pares que pueden ser el mínimo. Un
    producto de matrices sobre coordenadas centradas da una cota inferior de cada distancia
    de referencia, y el par se descarta si su cota supera una distancia de referencia ya
    calculada para ese cliente. El mínimo es exacto, así que coincide bit a bit con el de la
    referencia aunque se calcule sobre menos pares.

    La cota se obtiene así, con u = 2^-53 y D dimensiones. Al centrar, x' = fl(x - c) cambia
    cada coordenada como mucho en u|x'|, y la distancia exacta queda por encima de
    ||x' - y'|| - 2u(||x'|| + ||y'||). El cuadrado s = q + p - 2 x'·y' calculado en float64
    difiere del exacto como mucho en (D + 2)u(||x'|| + ||y'||)² con cualquier orden de suma,
    así que se resta 4(D + 4)u(√q + √p)². La referencia resta cada coordenada con un error
    relativo de u y encadena D llamadas a `hypot`, que en glibc tienen un error de como
    mucho una ulp. Con cuatro ulp por llamada su resultado no baja de la distancia exacta
    por (1 - (8D + 1)u). El factor final (1 - (8D + 2048)u) cubre ese error y deja 2047u
    para el redondeo de la propia cota. Si un bloque conserva más de un octavo de sus pares,
    se calcula entero como en la referencia.

    Los buffers de trabajo se reservan una vez al principio. `_bounded_bytes` cuenta además
    los índices de `nonzero` y los temporales de NumPy. El producto usa la BLAS de NumPy con
    los hilos que fije el proceso.
    """
    count, dimensions = clients.shape
    center = fixed.mean(axis=0)
    # Las coordenadas centradas solo sirven a la cota. La referencia resta las originales,
    # que se trasponen para leer cada dimensión de forma contigua.
    x, y = clients - center, fixed - center
    ot, ft = np.ascontiguousarray(clients.T), np.ascontiguousarray(fixed.T)
    q, p = np.einsum("ij,ij->i", x, x), np.einsum("ij,ij->i", y, y)
    rq, rp = np.sqrt(q), np.sqrt(p)
    relative = 4 * (dimensions + 4) * _UNIT
    absolute = 4 * _UNIT
    shrink = 1 - (8 * dimensions + 2048) * _UNIT
    nearest = np.empty(count)
    rows, columns = min(rows, count), min(columns, len(fixed))
    square, bound, scratch = (np.empty((rows, columns)) for _ in range(3))
    mask = np.empty((rows, columns), dtype=bool)
    best_square, value, best = np.empty(rows), np.empty(rows), np.empty(rows)
    best_index, index = np.empty(rows, dtype=np.int64), np.empty(rows, dtype=np.int64)
    better = np.empty(rows, dtype=bool)
    pairs = max(rows, rows * columns // _DENSE)
    buffers = (np.empty(pairs), np.empty(pairs), np.empty(pairs), np.empty(pairs))
    pair_rows, pair_columns = np.empty(pairs, dtype=np.int64), np.empty(pairs, dtype=np.int64)
    for start in range(0, count, rows):
        end = min(start + rows, count)
        width = end - start
        best_square[:width] = np.inf
        best_index[:width] = 0
        for offset in range(0, len(fixed), columns):
            stop = min(offset + columns, len(fixed))
            tile = square[:width, : stop - offset]
            np.matmul(x[start:end], y[offset:stop].T, out=tile)
            tile *= -2
            tile += q[start:end, None]
            tile += p[None, offset:stop]
            # El mínimo de cada fila es el valor de su primer argmin, como en la referencia.
            np.argmin(tile, axis=1, out=index[:width])
            np.min(tile, axis=1, out=value[:width])
            np.less(value[:width], best_square[:width], out=better[:width])
            np.copyto(best_square[:width], value[:width], where=better[:width])
            index[:width] += offset
            np.copyto(best_index[:width], index[:width], where=better[:width])
        result = nearest[start:end]
        rows_index = np.arange(start, end)
        result[:] = _reference_chain(ot, ft, rows_index, best_index[:width], buffers)
        np.copyto(best[:width], result)
        for offset in range(0, len(fixed), columns):
            stop = min(offset + columns, len(fixed))
            tile = square[:width, : stop - offset]
            limit = bound[:width, : stop - offset]
            margin = scratch[:width, : stop - offset]
            np.matmul(x[start:end], y[offset:stop].T, out=tile)
            tile *= -2
            tile += q[start:end, None]
            tile += p[None, offset:stop]
            np.add(rq[start:end, None], rp[None, offset:stop], out=limit)
            np.multiply(limit, limit, out=margin)
            margin *= relative
            tile -= margin
            np.maximum(tile, 0, out=tile)
            np.sqrt(tile, out=tile)
            np.multiply(limit, absolute, out=margin)
            tile -= margin
            tile *= shrink
            kept = mask[:width, : stop - offset]
            np.less_equal(tile, best[:width, None], out=kept)
            survivors = np.count_nonzero(kept)
            if survivors * _DENSE > kept.size:
                distance = engine._distance(start, end, tuple(range(offset, stop)))
                np.min(distance, axis=1, out=value[:width])
                np.minimum(result, value[:width], out=result)
            elif survivors:
                local, remote = np.nonzero(kept)
                chain_rows, chain_columns = pair_rows[:survivors], pair_columns[:survivors]
                np.add(local, start, out=chain_rows)
                np.add(remote, offset, out=chain_columns)
                values = _reference_chain(ot, ft, chain_rows, chain_columns, buffers)
                np.minimum.at(result, local, values)
    return nearest


def _background(clients, fixed, metric, integer, rows, columns, max_pairs, cdist, bounded=False):
    """Calcula b por dos ejes, contabilizando todos los pares frente a centros fijos."""
    pairs = len(clients) * len(fixed)
    if pairs > max_pairs:
        raise MedoidBudgetExceeded("El fondo supera el presupuesto de pares de distancia")
    if bounded:
        engine = _Costs(clients, fixed, metric, integer, rows, columns, max_pairs)
        return _bounded_minimum(clients, fixed, rows, columns, engine), pairs
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
    reuse_distances: bool = False,
    bounded_background: bool = False,
) -> AnchoredMedoidSelection:
    """Selecciona hasta capacity centros, incluidos los fijos, sobre el E completo.

    Los centros fijos y candidatos son IDs distintos contenidos en E. Los
    clientes fijos aportan cero. Los demás minimizan su distancia al conjunto
    fijo y al variable. La enumeración solo es óptima sobre esta restricción.

    El fondo NumPy conserva la referencia general. cdist es exploratorio,
    exige FP32 y calcula distancias euclídeas FP64. Puede cambiar los IDs en
    empates de redondeo. La selección variable mantiene la referencia NumPy.
    La salida identifica ambos tipos y las versiones de bibliotecas.
    Los límites de pares y buffers son conjuntos para fondo y selección.

    `reuse_distances` calcula una sola vez cada distancia de un bloque de
    clientes a un candidato variable y `bounded_background` descarta en el
    fondo los pares que una cota inferior excluye del mínimo (solo con
    distancia euclídea y coordenadas FP32). Las dos dan los mismos bits que la
    referencia. Su memoria se suma a la estimada solo si cabe sin cambiar los
    bloques, primero la de la cota y después la de la tabla. La que no cabe se
    descarta y la salida indica cuáles se aplicaron.
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
    if type(reuse_distances) is not bool or type(bounded_background) is not bool:
        raise ValueError("Las opciones de reutilización y de cota deben ser booleanas")
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
    # La cota solo se demuestra para la distancia euclídea sobre coordenadas FP32, las del banco.
    bounded_background = bool(
        bounded_background
        and cdist is None
        and metric == "euclidean"
        and points.dtype == np.float32
        and len(fixed)
        and len(clients)
    )
    # La cota y la tabla reutilizada no cambian los bloques. La que no cabe se descarta.
    if bounded_background:
        extra = _bounded_bytes(len(clients), len(fixed), dimensions, rows, fixed_columns)
        bounded_background = estimated + extra <= remaining_bytes
        estimated += extra if bounded_background else 0
    if reuse_distances:
        extra = _reuse_bytes(len(clients), len(variable), rows)
        reuse_distances = estimated + extra <= remaining_bytes
        estimated += extra if reuse_distances else 0
    background, background_pairs = (None, 0)
    if len(fixed) and len(clients):
        background, background_pairs = _background(
            clients,
            fixed,
            metric,
            integer,
            rows,
            fixed_columns,
            max_distance_pairs,
            cdist,
            bounded_background,
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
        reuse=reuse_distances,
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
        reuse_distances,
        bounded_background,
    )
