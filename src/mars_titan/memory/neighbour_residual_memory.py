"""Memoria por vecinos del residuo de rango maduro, sin parámetros que ajustar.

Generaliza la memoria por activo (`asset_residual_memory`) de «el mismo activo» a «filas con
entradas parecidas», como la lectura episódica por vecinos que el laboratorio tabular midió sobre
la red estática del origen 2013 (+0,0016 de IC de evaluación en dos semillas, placebo negativo).

Para una cohorte (mercado g, instante de decisión t) con n filas y M modelos:

- clave de cada fila: q_i = x_i P / ‖x_i P‖, con P ∈ {−1, +1}^{d×m} fija (proyección aleatoria de
  signos de las entradas ya normalizadas, Achlioptas 2003); una fila con x_i P = 0 tiene clave 0;
- banco B_t: las filas de las W cohortes más recientes del mercado cuya etiqueta maduró como mucho
  en t, con su clave y su valor v_jm = recorte_c(E_jm), donde E = G − (z_m·G/n) z_m es el residuo
  de los rangos gaussianos centrados de la etiqueta frente a la predicción tipificada que emitió
  el modelo m en esa cohorte (la misma definición que la memoria por activo);
- lectura: N_k(i) = los k vecinos de mayor coseno q_i·q_j en B_t (a igual coseno, primero la fila
  más antigua del banco), w_ij = softmax_j(q_i·q_j / τ) sobre N_k(i) y r_im = Σ_j w_ij v_jm,
  sumada de mayor a menor coseno; con el banco vacío r = 0;
- placebo: la misma lectura con los valores del banco permutados entre sus filas, con una
  permutación propia de cada cohorte derivada de la semilla declarada, del mercado y del instante.

Los vecinos y sus pesos solo dependen de las claves, así que se calculan una vez para todos los
modelos. El consumidor emite z_m + κ · tipificación(r_m) en la sesión.

Contrato causal: la lectura en t solo usa cohortes cuya etiqueta maduró como mucho en t (target
available at ≤ decisión), escritas una vez y en orden de emisión, con reloj monótono por mercado.
Las claves solo usan entradas de la propia decisión. Falla con mercado, fila e instante ante un
valor no finito, una cohorte fuera de orden, una etiqueta que madura antes de su decisión o
fechas del test final. Cosenos en FP32 (como las claves); pesos, valores y sumas en FP64.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.memory import _fast_kernels as fk
from mars_titan.memory import asset_residual_memory as arm

KIND = "neighbour_residual_memory"
# v2: los empates en el k-ésimo coseno van a la fila más antigua del banco y la suma ponderada
# sigue el orden de la definición (v1 dejaba ambos al orden interno de np.argpartition).
METHOD = "cosine_topk_softmax_matured_rank_residual_v2"
SCHEMA_VERSION = 1
MAX_WINDOW = 512
MAX_K = 4096
MAX_MODELS = 64
MAX_PENDING = 64
CHUNK = 128
MAX_THREADS = 16


def _top(sims, k, mask=None):
    """Posiciones de las k mayores similitudes de cada fila, en el orden estable del laboratorio.

    El orden es el de `np.argsort(-sims, kind="stable")[:, :k]`: de mayor a menor coseno y, a
    igual coseno, primero la fila más antigua del banco. Es exacto sin ordenar la fila entera: el
    k-ésimo mayor de los máximos de bloques consecutivos es una cota inferior del k-ésimo mayor
    valor de la fila (al menos k bloques tienen un elemento que la alcanza), así que las filas que
    la alcanzan contienen a las k mayores. Con unos 2k bloques quedan del orden de 1,4k
    candidatas por fila, que se ordenan; el resultado no depende del tamaño de los bloques.
    `mask`, si se da, es un búfer booleano con la forma de `sims` donde se marcan las candidatas.
    """
    rows, size = sims.shape
    width = max(1, size // (2 * k))
    whole = size // width * width
    maxima = sims[:, :whole].reshape(rows, whole // width, width).max(axis=2)
    if whole < size:
        maxima = np.concatenate([maxima, sims[:, whole:].max(axis=1, keepdims=True)], axis=1)
    blocks = maxima.shape[1]
    low = np.partition(maxima, blocks - k, axis=1)[:, blocks - k]
    if mask is None:
        mask = np.empty(sims.shape, dtype=bool)
    row, column = np.nonzero(np.greater_equal(sims, low[:, None], out=mask))
    counts = np.bincount(row, minlength=rows)
    _require(bool(np.all(counts >= k)), "La cota de los k vecinos dejó una fila sin k candidatas")
    order = np.lexsort((column, -sims[row, column], row))
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return column[order][first[:, None] + np.arange(k)]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _is_instant(value):
    """Un instante de desarrollo: entero (no booleano) de microsegundos UTC anterior a 2024."""
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and 0 <= int(value) < FINAL_TEST_START_US
    )


def _instant(value, name):
    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool),
        f"{name} debe ser un entero de microsegundos UTC",
    )
    value = int(value)
    _require(0 <= value < FINAL_TEST_START_US, f"{name} está fuera del periodo de desarrollo")
    return value


@dataclass(frozen=True)
class NeighbourConfig:
    """Cohortes maduras en el banco, vecinos, temperatura y recorte del valor guardado."""

    window: int
    k: int
    tau: float
    clip: float = 3.0
    # Con exclude_own, una consulta no lee las filas de su propio activo: mide la parte de la
    # lectura que viene de activos parecidos y no de la historia del mismo activo.
    exclude_own: bool = False
    # Decaimiento temporal causal opcional en cohortes dentro de la ventana del banco:
    # con None (por omisión) conserva literalmente la identidad y las lecturas v2.
    recency_half_life: float | None = None

    def __post_init__(self):
        _require(
            type(self.window) is int and 1 <= self.window <= MAX_WINDOW,
            f"La ventana del banco es un entero entre 1 y {MAX_WINDOW}",
        )
        _require(
            type(self.k) is int and 1 <= self.k <= MAX_K,
            f"El número de vecinos es un entero entre 1 y {MAX_K}",
        )
        for name in ("tau", "clip"):
            value = getattr(self, name)
            _require(
                type(value) in (int, float) and math.isfinite(value) and value > 0,
                f"{name} es un número finito y positivo",
            )
        _require(type(self.exclude_own) is bool, "exclude_own es un booleano")
        if self.recency_half_life is not None:
            _require(
                type(self.recency_half_life) in (int, float)
                and math.isfinite(self.recency_half_life)
                and self.recency_half_life > 0,
                "recency_half_life es un número finito y positivo o None",
            )

    def identity(self):
        identity = dict(window=self.window, k=self.k, tau=float(self.tau), clip=float(self.clip))
        # Sin exclusión ni decaimiento temporal la identidad conserva literalmente su forma anterior.
        if self.exclude_own:
            identity["exclude_own"] = True
        if self.recency_half_life is not None:
            identity["recency_half_life"] = float(self.recency_half_life)
        return identity


def projection(inputs, dim, *, seed):
    """Matriz de signos ±1 en FP32 con forma [entradas, dim], reproducible con la semilla."""
    _require(
        type(inputs) is int and type(dim) is int and inputs >= 1 and dim >= 1,
        "La proyección necesita dimensiones enteras positivas",
    )
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=(inputs, dim)) * 2 - 1).astype(np.float32)


def keys_from_inputs(inputs, matrix):
    """Claves de norma uno en FP32; una fila cuya proyección es nula queda en cero."""
    inputs = np.asarray(inputs)
    _require(
        inputs.ndim == 2
        and inputs.dtype == np.float32
        and matrix.dtype == np.float32
        and matrix.shape[0] == inputs.shape[1],
        "Las entradas son una matriz FP32 con tantas columnas como filas tiene la proyección",
    )
    bad = np.flatnonzero(~np.isfinite(inputs).all(axis=1))
    _require(bad.size == 0, f"Entradas no finitas en {bad.size} filas; la primera es {bad[:1]}")
    keys = inputs @ matrix
    _require(np.isfinite(keys).all(), "La proyección de las entradas desborda FP32")
    with np.errstate(over="ignore", under="ignore"):
        norms = np.linalg.norm(keys, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, np.float32(1.0))
    result = np.where(norms > 0, keys / safe, np.float32(0.0)).astype(np.float32)
    # Una fila no nula cuya norma FP32 desborda o se anula (sus cuadrados salen del rango de FP32)
    # se normaliza escalada por su máximo absoluto. Las demás filas no cambian (H02 de la
    # auditoría): antes quedaban en cero, como si su proyección fuera nula.
    peak = np.abs(keys).max(axis=1)
    odd = (~np.isfinite(norms[:, 0]) | (norms[:, 0] == 0)) & (peak > 0)
    if odd.any():
        scaled = keys[odd] / peak[odd, None]
        result[odd] = scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
    return result


def _array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _market_seed(market):
    return int.from_bytes(hashlib.sha256(market.encode()).digest()[:8], "big")


class _Market:
    def __init__(self):
        self.bank = []
        self.pending = []
        self.clock = None
        self.matured = 0


class _Scratch:
    """Búferes de un hilo de lectura: similitudes y máscara de un bloque de consultas.

    Se reservan una vez y solo crecen. Reservarlos en cada bloque (87 MB con el banco de 63
    cohortes) costaba un fallo de página por cada 4 KiB tocado y era la mayor parte del tiempo.
    """

    def __init__(self):
        self._sims = np.empty(0, dtype=np.float32)
        self._mask = np.empty(0, dtype=bool)

    def views(self, rows, size):
        need = rows * size
        if self._sims.size < need:
            self._sims = np.empty(need, dtype=np.float32)
            self._mask = np.empty(need, dtype=bool)
        return self._sims[:need].reshape(rows, size), self._mask[:need].reshape(rows, size)


class _HeapScratch:
    """Búferes reutilizables para lectura streaming por cohortes sin concatenar claves."""

    def __init__(self):
        self._sims = np.empty(0, dtype=np.float32)
        self._heap_v = np.empty(0, dtype=np.float32)
        self._heap_i = np.empty(0, dtype=np.int64)
        self._heap_f = np.empty(0, dtype=np.int64)

    def views(self, rows, max_tile, k):
        need_s = rows * max_tile
        if self._sims.size < need_s:
            self._sims = np.empty(need_s, dtype=np.float32)
        need_h = rows * k
        if self._heap_v.size < need_h:
            self._heap_v = np.empty(need_h, dtype=np.float32)
            self._heap_i = np.empty(need_h, dtype=np.int64)
        if self._heap_f.size < rows:
            self._heap_f = np.empty(rows, dtype=np.int64)
        return (
            self._sims[:need_s].reshape(rows, max_tile),
            self._heap_v[:need_h].reshape(rows, k),
            self._heap_i[:need_h].reshape(rows, k),
            self._heap_f[:rows],
        )


def _recency_bias(bank, half_life):
    """Sesgo logarítmico causal −edad · ln(2) / h por fila del banco (edad 0 = cohorte más reciente)."""
    total = sum(len(c["keys"]) for c in bank)
    bias = np.empty(total, dtype=np.float64)
    rate = math.log(2.0) / float(half_life)
    n_cohorts = len(bank)
    offset = 0
    for idx, cohort in enumerate(bank):
        count = len(cohort["keys"])
        age = float(n_cohorts - 1 - idx)
        bias[offset : offset + count] = -age * rate
        offset += count
    return bias


def _torch_module():
    try:
        import torch

        return torch
    except ImportError:
        return None


class NeighbourMemory:
    """Banco causal por mercado y lectura por vecinos para M modelos a la vez."""

    def __init__(self, config, models, *, threads=1, device="cpu"):
        _require(isinstance(config, NeighbourConfig), "La configuración es NeighbourConfig")
        _require(
            type(models) is int and 1 <= models <= MAX_MODELS,
            f"El número de modelos es un entero entre 1 y {MAX_MODELS}",
        )
        # Los hilos reparten los bloques de consultas de una cohorte; no cambian el resultado ni
        # forman parte del estado.
        _require(
            type(threads) is int and 1 <= threads <= MAX_THREADS,
            f"Los hilos de la lectura son un entero entre 1 y {MAX_THREADS}",
        )
        _require(device in ("cpu", "cuda"), "device debe ser 'cpu' o 'cuda'")
        self.config = config
        self.models = models
        self.threads = threads
        self.device = device
        self._markets = {}
        # Búferes reutilizados (no son estado): uno por hilo de lectura y el banco contiguo de
        # cada mercado.
        self._scratch = queue.SimpleQueue()
        for _ in range(threads):
            self._scratch.put(_Scratch())
        self._heap_scratch = _HeapScratch()
        self._joined = {}

    def _join(self, market, name, parts):
        """`np.concatenate(parts)` escrito en un búfer del mercado que solo crece."""
        size = sum(len(part) for part in parts)
        tail, dtype = parts[0].shape[1:], parts[0].dtype
        buffer = self._joined.get((market, name))
        fits = buffer is not None and buffer.dtype == dtype and buffer.shape[1:] == tail
        if not fits or len(buffer) < size:
            buffer = np.empty((size + size // 2, *tail), dtype=dtype)
            self._joined[(market, name)] = buffer
        return np.concatenate(parts, out=buffer[:size])

    def _market(self, market, create=False):
        _require(isinstance(market, str) and market, "El mercado es un texto no vacío")
        if market not in self._markets:
            _require(create, f"El mercado {market!r} no tiene memoria")
            self._markets[market] = _Market()
        return self._markets[market]

    def _neighbours_streaming_native(
        self, lib, state, keys, values, shuffled, k, tau, recency, assets
    ):
        """Lectura exacta en streaming por cohortes con torch.mm + min-heap C99/OpenMP.

        Evita concatenar `bank_keys` (42 MB por sesión) y evita la matriz de similitudes de
        [CHUNK, total_bank] (84 MB), manteniendo la matriz de similitudes acotada al tamaño de
        un bloque de consultas por una cohorte, con desempate exacto por fila más antigua y
        paridad independiente del número de hilos.
        """
        rows = len(keys)
        reads = np.empty((rows, self.models), dtype=np.float64)
        placebo = None if shuffled is None else np.empty((rows, self.models), dtype=np.float64)
        max_tile = max(len(c["keys"]) for c in state.bank)
        q_chunk = min(rows, CHUNK)
        sims_full, heap_v, heap_i, heap_f = self._heap_scratch.views(q_chunk, max_tile, k)
        torch = _torch_module()
        use_torch_cpu = torch is not None and self.device == "cpu"
        use_torch_cuda = torch is not None and self.device == "cuda" and torch.cuda.is_available()
        if use_torch_cpu:
            prev_threads = torch.get_num_threads()
            torch.set_num_threads(self.threads)
        else:
            prev_threads = None
        values_c = np.ascontiguousarray(values, dtype=np.float64)
        shuffled_c = None if shuffled is None else np.ascontiguousarray(shuffled, dtype=np.float64)
        recency_c = None if recency is None else np.ascontiguousarray(recency, dtype=np.float64)
        exclude_own = 1 if self.config.exclude_own else 0
        q_assets_all = None if assets is None else np.ascontiguousarray(assets, dtype=np.int64)
        try:
            if use_torch_cuda:
                # Ruta opcional en GPU (p. ej. Colab RTX PRO 6000 / DGX GB10 libre):
                # SGEMM FP32 determinista en GPU y min-heap exacto en C99.
                q_gpu = torch.from_numpy(np.ascontiguousarray(keys)).to("cuda")
                bank_gpu = [
                    torch.from_numpy(np.ascontiguousarray(c["keys"])).to("cuda")
                    for c in state.bank
                ]
            else:
                q_gpu = bank_gpu = None
            for start in range(0, rows, q_chunk):
                end = min(start + q_chunk, rows)
                nr = end - start
                heap_f[:nr] = 0
                q_sub = np.ascontiguousarray(keys[start:end], dtype=np.float32)
                qa_ptr = (
                    None
                    if q_assets_all is None
                    else q_assets_all[start:end].ctypes.data_as(ctypes.c_void_p)
                )
                col_offset = 0
                for c_idx, cohort in enumerate(state.bank):
                    ck = cohort["keys"]
                    ns = len(ck)
                    sims_view = sims_full[:nr, :ns]
                    if use_torch_cuda:
                        prod = torch.mm(q_gpu[start:end], bank_gpu[c_idx].T)
                        sims_view[...] = prod.cpu().numpy()
                    elif use_torch_cpu:
                        torch.mm(
                            torch.from_numpy(q_sub),
                            torch.from_numpy(ck).T,
                            out=torch.from_numpy(sims_view),
                        )
                    else:
                        np.matmul(q_sub, ck.T, out=sims_view)
                    ba_ptr = (
                        None
                        if not exclude_own
                        else np.ascontiguousarray(cohort["assets"], dtype=np.int64).ctypes.data_as(
                            ctypes.c_void_p
                        )
                    )
                    lib.nrm_heap_update_tile_f32(
                        ctypes.c_int64(nr),
                        ctypes.c_int64(ns),
                        ctypes.c_int64(col_offset),
                        ctypes.c_int64(k),
                        sims_view.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(exclude_own),
                        qa_ptr,
                        ba_ptr,
                        heap_v[:nr].ctypes.data_as(ctypes.c_void_p),
                        heap_i[:nr].ctypes.data_as(ctypes.c_void_p),
                        heap_f[:nr].ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(self.threads),
                    )
                    col_offset += ns
                if recency_c is None:
                    lib.nrm_heap_finalize_gather_f64(
                        ctypes.c_int64(nr),
                        ctypes.c_int64(k),
                        ctypes.c_int64(self.models),
                        heap_v[:nr].ctypes.data_as(ctypes.c_void_p),
                        heap_i[:nr].ctypes.data_as(ctypes.c_void_p),
                        heap_f[:nr].ctypes.data_as(ctypes.c_void_p),
                        values_c.ctypes.data_as(ctypes.c_void_p),
                        None if shuffled_c is None else shuffled_c.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_double(tau),
                        reads[start:end].ctypes.data_as(ctypes.c_void_p),
                        None if placebo is None else placebo[start:end].ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(self.threads),
                    )
                else:
                    lib.nrm_heap_finalize_gather_recency_f64(
                        ctypes.c_int64(nr),
                        ctypes.c_int64(k),
                        ctypes.c_int64(self.models),
                        heap_v[:nr].ctypes.data_as(ctypes.c_void_p),
                        heap_i[:nr].ctypes.data_as(ctypes.c_void_p),
                        heap_f[:nr].ctypes.data_as(ctypes.c_void_p),
                        values_c.ctypes.data_as(ctypes.c_void_p),
                        None if shuffled_c is None else shuffled_c.ctypes.data_as(ctypes.c_void_p),
                        recency_c.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_double(tau),
                        reads[start:end].ctypes.data_as(ctypes.c_void_p),
                        None if placebo is None else placebo[start:end].ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(self.threads),
                    )
        finally:
            if prev_threads is not None:
                torch.set_num_threads(prev_threads)
        return reads, placebo

    def _neighbours(self, state, keys, placebo_seed, market, prediction_at, assets=None):
        rows = len(keys)
        reads = np.zeros((rows, self.models))
        placebo = None if placebo_seed is None else np.zeros((rows, self.models))
        if not state.bank:
            return reads, placebo
        _require(
            state.bank[0]["keys"].shape[1] == keys.shape[1],
            f"Las claves de {market} en {prediction_at} no tienen la dimensión del banco",
        )
        values = self._join(market, "values", [c["values"] for c in state.bank])
        shuffled = None
        if placebo_seed is not None:
            sequence = np.random.SeedSequence([placebo_seed, _market_seed(market), prediction_at])
            shuffled = values[np.random.default_rng(sequence).permutation(len(values))]
        size = len(values)
        k = min(self.config.k, size)
        tau = float(self.config.tau)
        recency = (
            None
            if self.config.recency_half_life is None
            else _recency_bias(state.bank, self.config.recency_half_life)
        )

        lib = fk.load_library()
        if lib is not None and size >= 32:
            return self._neighbours_streaming_native(
                lib, state, keys, values, shuffled, k, tau, recency, assets
            )

        bank_keys = self._join(market, "keys", [c["keys"] for c in state.bank])
        bank_assets = None
        if self.config.exclude_own:
            bank_assets = self._join(market, "assets", [c["assets"] for c in state.bank])

        def read(start):
            scratch = self._scratch.get()
            try:
                _read(start, scratch)
            finally:
                self._scratch.put(scratch)

        def _read(start, scratch):
            block = keys[start : start + CHUNK]
            sims, mask = scratch.views(len(block), size)
            np.matmul(block, bank_keys.T, out=sims)
            if bank_assets is not None:
                np.equal(assets[start : start + CHUNK, None], bank_assets[None, :], out=mask)
                sims[mask] = -np.inf
            if k < size:
                index = _top(sims, k, mask=mask)
            else:
                index = np.broadcast_to(np.arange(size), sims.shape)
            top = np.take_along_axis(sims, index, axis=1).astype(np.float64)
            logits = top / tau if recency is None else top / tau + recency[index]
            if bank_assets is None:
                weights = np.exp(logits - logits.max(axis=1, keepdims=True))
                weights /= weights.sum(axis=1, keepdims=True)
            else:
                # Las filas del propio activo pesan cero. Una consulta sin otro activo en el
                # banco lee cero, como con el banco vacío.
                best = logits.max(axis=1, keepdims=True)
                found = np.isfinite(best)
                weights = np.where(
                    np.isfinite(logits), np.exp(logits - np.where(found, best, 0.0)), 0.0
                )
                total = weights.sum(axis=1, keepdims=True)
                weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
            reads[start : start + CHUNK] = np.einsum("ik,ikm->im", weights, values[index])
            if shuffled is not None:
                placebo[start : start + CHUNK] = np.einsum("ik,ikm->im", weights, shuffled[index])

        starts = range(0, rows, CHUNK)
        if self.threads == 1 or len(starts) == 1:
            for start in starts:
                read(start)
        else:
            # Cada bloque escribe solo sus filas y no depende de los demás, así que el resultado
            # es el mismo con cualquier número de hilos (NumPy suelta el GIL en el trabajo pesado).
            with ThreadPoolExecutor(min(self.threads, len(starts))) as pool:
                for _ in pool.map(read, starts):
                    pass
        return reads, placebo

    def emit(
        self,
        market,
        prediction_at,
        keys,
        predictions,
        *,
        matures_at,
        placebo_seed=None,
        assets=None,
    ):
        """Lecturas de una cohorte para cada modelo (y su placebo), sin ver su etiqueta.

        Con la exclusión del propio activo, `assets` es el código entero del activo de cada fila.
        """
        prediction_at = _instant(prediction_at, "La fecha de decisión")
        _require(
            isinstance(matures_at, (int, np.integer)) and not isinstance(matures_at, bool),
            "La madurez debe ser un entero de microsegundos UTC",
        )
        # Una decisión de 2023 puede madurar en el test final: se emite y no se escribe nunca,
        # porque ni mature() ni el vaciado final llegan a 2024 (como en la memoria por activo).
        matures_at = int(matures_at)
        _require(matures_at > prediction_at, "Cada etiqueta madura después de su decisión")
        keys = np.asarray(keys)
        predictions = np.asarray(predictions)
        _require(
            keys.ndim == 2
            and keys.dtype == np.float32
            and len(keys) >= 1
            and predictions.shape == (len(keys), self.models)
            and predictions.dtype.kind == "f",
            "Una cohorte necesita claves FP32 y una predicción real por fila y modelo",
        )
        for values, name in ((keys, "La clave"), (predictions, "La predicción")):
            bad = np.flatnonzero(~np.isfinite(values).all(axis=1))
            _require(
                bad.size == 0,
                f"{name} no es finita en {bad.size} filas de {market}; la primera es la fila"
                f" {bad[0] if bad.size else ''} en el instante {prediction_at}",
            )
        _require(
            placebo_seed is None or (type(placebo_seed) is int and placebo_seed >= 0),
            "La semilla del placebo es un entero no negativo",
        )
        if self.config.exclude_own:
            assets = None if assets is None else np.asarray(assets)
            _require(
                assets is not None and assets.shape == (len(keys),) and assets.dtype.kind in "iu",
                "La exclusión del propio activo necesita el activo de cada fila como código entero",
            )
            assets = assets.astype(np.int64)
        else:
            _require(
                assets is None, "Los activos se declaran solo con la exclusión del propio activo"
            )
        state = self._markets.get(market)
        if state is not None:
            _require(
                state.clock is None or prediction_at >= state.clock,
                "El reloj del mercado no puede retroceder",
            )
            _require(
                not state.pending or prediction_at > state.pending[-1]["prediction_at"],
                "Las cohortes de un mercado se emiten en orden y una sola vez",
            )
            _require(len(state.pending) < MAX_PENDING, "Demasiadas cohortes pendientes")
        state = self._market(market, create=True)
        reads, placebo = self._neighbours(state, keys, placebo_seed, market, prediction_at, assets)
        z = np.column_stack(
            [arm.zscore(predictions[:, m].astype(np.float64)) for m in range(self.models)]
        )
        pending = dict(
            prediction_at=prediction_at,
            matures_at=matures_at,
            keys=np.array(keys, copy=True),
            z=z,
        )
        if assets is not None:
            pending["assets"] = np.array(assets, copy=True)
        state.pending.append(pending)
        state.clock = prediction_at
        return reads, placebo

    def pending(self, market):
        """Instantes de decisión y de madurez de las cohortes pendientes del mercado."""
        state = self._markets.get(market)
        if state is None:
            return []
        return [(c["prediction_at"], c["matures_at"]) for c in state.pending]

    def mature(self, market, prediction_at, target, *, at):
        """Escribir en el banco la cohorte pendiente más antigua con su etiqueta madura."""
        state = self._market(market)
        prediction_at = _instant(prediction_at, "La fecha de decisión")
        at = _instant(at, "El instante de actualización")
        _require(state.pending, f"{market} no tiene cohortes pendientes")
        oldest = state.pending[0]
        _require(
            prediction_at == oldest["prediction_at"],
            "La cohorte madura fuera de orden: antes debe madurar la emitida en "
            f"{oldest['prediction_at']}",
        )
        _require(at >= oldest["matures_at"], "La etiqueta de la cohorte todavía no es madura")
        _require(
            state.clock is None or at >= state.clock, "El reloj del mercado no puede retroceder"
        )
        target = np.asarray(target)
        _require(
            target.shape == (len(oldest["keys"]),) and target.dtype.kind == "f",
            "Se necesita una etiqueta real por fila de la cohorte emitida",
        )
        target = target.astype(np.float64)
        bad = np.flatnonzero(~np.isfinite(target))
        _require(
            bad.size == 0,
            f"La etiqueta madura no es finita en {bad.size} filas de {market}; la primera es la"
            f" fila {bad[0] if bad.size else ''} de la cohorte {prediction_at}",
        )
        clip = float(self.config.clip)
        values = np.column_stack(
            [
                np.clip(arm.stored_value(oldest["z"][:, m], target, "residual"), -clip, clip)
                for m in range(self.models)
            ]
        )
        cohort = dict(prediction_at=prediction_at, keys=oldest["keys"], values=values)
        if "assets" in oldest:
            cohort["assets"] = oldest["assets"]
        state.bank.append(cohort)
        while len(state.bank) > self.config.window:
            state.bank.pop(0)
        state.pending.pop(0)
        state.clock = at
        state.matured += 1

    def export(self):
        """Estado completo (banco, cohortes pendientes y relojes) con su huella."""
        markets = {}
        for name in sorted(self._markets):
            state = self._markets[name]
            markets[name] = dict(
                clock=state.clock,
                matured=state.matured,
                bank=[{k: _copied(v) for k, v in c.items()} for c in state.bank],
                pending=[{k: _copied(v) for k, v in c.items()} for c in state.pending],
            )
        payload = dict(
            schema_version=SCHEMA_VERSION,
            kind=KIND,
            method=METHOD,
            config=self.config.identity(),
            models=self.models,
            markets=markets,
        )
        payload["state_sha256"] = _state_sha256(payload)
        return payload

    @classmethod
    def restore(cls, payload, *, threads=1):
        """Reconstruir la memoria tras validar sus campos, sus invariantes y su huella."""
        _require(
            isinstance(payload, dict)
            and set(payload)
            == {"schema_version", "kind", "method", "config", "models", "markets", "state_sha256"}
            and type(payload["schema_version"]) is int
            and payload["schema_version"] == SCHEMA_VERSION
            and payload["kind"] == KIND
            and payload["method"] == METHOD,
            "El estado pertenece a otra memoria o versión",
        )
        _require(
            _state_sha256(payload) == payload["state_sha256"],
            "El estado exportado no coincide con su huella",
        )
        memory = cls(NeighbourConfig(**payload["config"]), payload["models"], threads=threads)
        clip = float(memory.config.clip)
        for name, saved in payload["markets"].items():
            # La huella no basta: un estado firmado debe cumplir también los invariantes (H02).
            _require(
                isinstance(name, str)
                and name
                and isinstance(saved, dict)
                and set(saved) == {"clock", "matured", "bank", "pending"},
                f"El estado guardado del mercado {name!r} no es coherente",
            )
            clock, matured = saved["clock"], saved["matured"]
            bank, pending = saved["bank"], saved["pending"]
            _require(
                (clock is None or _is_instant(clock))
                and (clock is not None or not (bank or pending))
                and type(matured) is int
                and matured >= 0
                and len(bank) <= memory.config.window,
                f"El reloj, las maduradas o el banco guardados de {name} no son coherentes",
            )
            _require(len(pending) <= MAX_PENDING, f"Demasiadas cohortes pendientes en {name}")
            state = memory._market(name, create=True)
            state.clock, state.matured = clock, matured
            previous = None
            for cohort in bank:
                _require(
                    memory._valid_cohort(cohort, "values", clock, previous)
                    and np.all(np.abs(cohort["values"]) <= clip),
                    f"Una cohorte del banco de {name} no es coherente",
                )
                previous = cohort["prediction_at"]
                state.bank.append({k: _copied(v) for k, v in cohort.items()})
            for cohort in pending:
                matures = cohort.get("matures_at")
                _require(
                    memory._valid_cohort(cohort, "z", clock, previous)
                    and isinstance(matures, (int, np.integer))
                    and not isinstance(matures, bool)
                    and matures > cohort["prediction_at"],
                    f"Una cohorte pendiente de {name} no es coherente",
                )
                previous = cohort["prediction_at"]
                state.pending.append({k: _copied(v) for k, v in cohort.items()})
        return memory


def _carries_assets(self, cohort):
    """Una cohorte lleva el activo de cada fila si y solo si la memoria excluye el propio."""
    if not self.config.exclude_own:
        return "assets" not in cohort
    assets = cohort.get("assets")
    return (
        isinstance(assets, np.ndarray)
        and assets.dtype == np.int64
        and assets.shape == (len(cohort["keys"]),)
    )


NeighbourMemory._carries_assets = _carries_assets


def _valid_cohort(self, cohort, field, clock, previous):
    """Una cohorte guardada: instante de desarrollo en orden, y claves y tabla finitas."""
    if not isinstance(cohort, dict):
        return False
    expected = {"prediction_at", "keys", field}
    expected |= {"matures_at"} if field == "z" else set()
    expected |= {"assets"} if self.config.exclude_own else set()
    keys, table, decided = cohort.get("keys"), cohort.get(field), cohort.get("prediction_at")
    return (
        set(cohort) == expected
        and _is_instant(decided)
        and decided <= clock
        and (previous is None or decided > previous)
        and isinstance(keys, np.ndarray)
        and keys.dtype == np.float32
        and keys.ndim == 2
        and np.isfinite(keys).all()
        and isinstance(table, np.ndarray)
        and table.shape == (len(keys), self.models)
        and np.isfinite(table).all()
        and self._carries_assets(cohort)
    )


NeighbourMemory._valid_cohort = _valid_cohort


def _copied(value):
    return value.copy() if isinstance(value, np.ndarray) else value


def _state_sha256(payload):
    def plain(value):
        if isinstance(value, np.ndarray):
            return dict(array=_array_sha256(value))
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items() if k != "state_sha256"}
        if isinstance(value, list):
            return [plain(v) for v in value]
        return value

    text = json.dumps(plain(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _cohorts(markets, prediction_at):
    order = np.lexsort((prediction_at, markets))
    order = order[np.lexsort((markets[order], prediction_at[order]))]
    keys = list(zip(prediction_at[order].tolist(), markets[order].tolist(), strict=True))
    bounds = [0] + [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]] + [len(keys)]
    return [
        (keys[start][1], keys[start][0], order[start:end])
        for start, end in zip(bounds[:-1], bounds[1:], strict=True)
    ]


def replay(
    memory,
    *,
    markets,
    prediction_at,
    matures_at,
    keys,
    predictions,
    target,
    placebo_seed=None,
    flush=True,
    pending_targets=None,
    assets=None,
):
    """Recorrer un panel en orden de decisión con la memoria por vecinos.

    Dentro de cada cohorte las filas conservan el orden del panel. Antes de emitir una cohorte
    se escriben, en orden de emisión, las cohortes de su mercado cuya madurez (la mayor de sus
    filas) no es posterior a la decisión. Con `flush` se escriben al final las que maduran antes
    del test final; sin él quedan pendientes para una reanudación, cuyas etiquetas se buscan en
    `pending_targets` (mercados, instantes y objetivos en el orden en que se emitieron). Devuelve
    las lecturas [filas, modelos] y su placebo (None sin semilla) en el orden original. Con la
    exclusión del propio activo, `assets` es el código entero del activo de cada fila.
    """
    markets, prediction_at = np.asarray(markets), np.asarray(prediction_at)
    matures_at, target = np.asarray(matures_at), np.asarray(target)
    keys, predictions = np.asarray(keys), np.asarray(predictions)
    rows = len(markets)
    _require(
        rows >= 1
        and all(a.shape == (rows,) for a in (prediction_at, matures_at, target))
        and len(keys) == rows
        and predictions.shape == (rows, memory.models)
        and prediction_at.dtype.kind == "i"
        and matures_at.dtype.kind == "i",
        "El panel necesita columnas alineadas e instantes enteros",
    )
    _require(
        bool(np.all(matures_at > prediction_at)), "Cada etiqueta madura después de su decisión"
    )
    if assets is not None:
        assets = np.asarray(assets)
        _require(
            assets.shape == (rows,),
            "La exclusión del propio activo necesita el activo de cada fila del panel",
        )
    sources = [(markets, prediction_at, target)]
    if pending_targets is not None:
        sources.append(
            (
                np.asarray(pending_targets["markets"]),
                np.asarray(pending_targets["prediction_at"]),
                np.asarray(pending_targets["target"]),
            )
        )
    lookups = [
        {(m, t): rows_ for m, t, rows_ in _cohorts(src_markets, src_moments)}
        for src_markets, src_moments, _ in sources
    ]

    def labels(market, moment):
        for lookup, (_, _, values) in zip(lookups, sources, strict=True):
            found = lookup.get((market, moment))
            if found is not None:
                return values[found]
        raise ValueError(f"No se encuentran las etiquetas de la cohorte {market} {moment}")

    def settle(market, now):
        state = memory._markets.get(market)
        while state is not None and state.pending and state.pending[0]["matures_at"] <= now:
            moment = state.pending[0]["prediction_at"]
            memory.mature(market, moment, labels(market, moment), at=now)

    reads = np.full((rows, memory.models), np.nan)
    placebo = None if placebo_seed is None else np.full((rows, memory.models), np.nan)
    for market, moment, chosen in _cohorts(markets, prediction_at):
        settle(market, moment)
        read, fake = memory.emit(
            market,
            moment,
            keys[chosen],
            predictions[chosen],
            matures_at=int(matures_at[chosen].max()),
            placebo_seed=placebo_seed,
            assets=None if assets is None else assets[chosen],
        )
        reads[chosen] = read
        if placebo is not None:
            placebo[chosen] = fake
    if flush:
        for market in sorted(memory._markets):
            state = memory._markets[market]
            while state.pending and state.pending[0]["matures_at"] < FINAL_TEST_START_US:
                settle(market, max(state.pending[0]["matures_at"], state.clock or 0))
    return reads, placebo
