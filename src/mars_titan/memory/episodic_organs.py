"""Órganos nuevos de memoria episódica sobre residuos maduros del modelo padre (WP5).

Tres memorias independientes, causales y deterministas para construir señales propias del
«cerebro holográfico» sobre las emisiones de un modelo padre:

1. **Banco denso con temperatura** (`DenseBankMemory`): vecinos por coseno sobre todas las filas
   maduras de las últimas `window` sesiones (hasta 126), con `k` hasta 256 y pesos
   `softmax(coseno / τ)` sobre una rejilla declarada de temperaturas `τ`. Guarda el residuo
   maduro del padre `recorte_c(y - p)` y devuelve la media ponderada y, aparte, la dispersión
   ponderada `√(Σ w_j (v_j - μ)²)` como medida de incertidumbre.
2. **Memoria holográfica reducida con regla delta** (`HolographicDeltaMemory`): claves
   compuestas por convolución circular (`torch.fft.rfft` / `irfft`) de un vector base del activo
   o de su sector y un vector base de régimen; vectores base aleatorios fijos y unitarios en
   frecuencia generados con semilla declarada; superposición en dimensión `D ∈ [1024, 4096]`,
   decaimiento por sesión madura, escritura por regla delta (solo el residuo que la memoria aún
   no predice) y lectura por correlación circular.
3. **Traza por activo** (`AssetResidualTrace`): media exponencial con semivida declarada de los
   residuos maduros `y - p` (opcionalmente recortados) de cada activo, por mercado.

Contrato causal común: una cohorte emitida en `prediction_at` solo puede leer cohortes de su
mismo mercado cuya etiqueta ya maduró (`matures_at <= prediction_at`, con `matures_at >
prediction_at` para toda cohorte, por lo que jamás se leen filas de la misma sesión ni de
sesiones futuras). El test final de 2024 (`FINAL_TEST_START_US`) permanece sellado. Cada órgano
incluye su placebo reproducible y un estado exportable con huella SHA-256 y validación completa
de invariantes al restaurar.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import torch

from mars_titan.data.sector_map import SectorMap
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.memory import neighbour_residual_memory as nrm
from mars_titan.memory import view_keys as vk

SCHEMA_VERSION = 1
EPISODIC_ARMS = ("dense_bank", "holographic_delta", "asset_trace")
MAX_GRID = 16
MAX_MODELS = 64
MAX_PENDING = 64
MAX_ASSETS = 100_000
MAX_THREADS = 16

DENSE_BANK_KIND = "dense_bank_temperature_memory"
DENSE_BANK_METHOD = "cosine_topk_softmax_parent_residual_uncertainty_v1"
DENSE_MAX_WINDOW = 126
DENSE_MAX_K = 256
DENSE_MAX_TEMPERATURES = 16
DENSE_CHUNK = 128

HOLOGRAPHIC_KIND = "holographic_delta_memory"
HOLOGRAPHIC_METHOD = "hrr_circular_conv_delta_superposition_v1"
HOLO_MIN_DIM = 1024
HOLO_MAX_DIM = 4096
HOLO_ENTITIES = ("asset", "sector")
HOLO_DTYPES = ("float32", "float64")
HOLO_CHUNK = 64

ASSET_TRACE_KIND = "asset_residual_trace_memory"
ASSET_TRACE_METHOD = "asset_ewma_matured_parent_residual_v1"

REGIME_RULE = "observable_volatility_trend_keys_v1"
REGIME_LABELS = ("unclassified", "calm_up", "calm_down", "turbulent_up", "turbulent_down")


# -------------------------------------------------------------------------------------------------
# Validación y utilidades comunes
# -------------------------------------------------------------------------------------------------


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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


def _matures_instant(value):
    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool),
        "La madurez debe ser un entero de microsegundos UTC",
    )
    return int(value)


def _market_seed(market):
    return int.from_bytes(hashlib.sha256(market.encode("utf-8")).digest()[:8], "big")


def _permute_rows(values, rng):
    """Permutación uniforme reproducible de filas (esperanza exacta igual a la media)."""
    if len(values) >= 2:
        return values[rng.permutation(len(values))]
    sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
    return values * sign


def _array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_sha256(payload):
    def plain(value):
        if isinstance(value, np.ndarray):
            return dict(array=_array_sha256(value))
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items() if k != "state_sha256"}
        if isinstance(value, list):
            return [plain(v) for v in value]
        return value

    return hashlib.sha256(_canonical(plain(payload)).encode()).hexdigest()


def _copied(value):
    return value.copy() if isinstance(value, np.ndarray) else value


def _cohorts(markets, prediction_at):
    """Filas de cada cohorte (mercado, instante) en orden de decisión y conservando su orden."""
    order = np.lexsort((markets, prediction_at))
    keys = list(zip(prediction_at[order].tolist(), markets[order].tolist(), strict=True))
    bounds = [0] + [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]] + [len(keys)]
    return [
        (keys[start][1], keys[start][0], order[start:end])
        for start, end in zip(bounds[:-1], bounds[1:], strict=True)
    ]


def _normalize_keys(keys):
    """Normaliza filas por norma L2 en FP32 o FP64; una fila nula permanece en cero."""
    keys = np.asarray(keys)
    _require(
        keys.ndim == 2
        and keys.dtype in (np.float32, np.float64)
        and len(keys) >= 1
        and keys.shape[1] >= 1,
        "Las claves deben ser una matriz 2D no vacía en FP32 o FP64",
    )
    bad = np.flatnonzero(~np.isfinite(keys).all(axis=1))
    _require(
        bad.size == 0,
        f"Clave no finita en {bad.size} filas; la primera es la fila {bad[0] if bad.size else ''}",
    )
    dtype = keys.dtype
    with np.errstate(over="ignore", under="ignore"):
        norms = np.linalg.norm(keys, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, dtype.type(1.0))
    result = np.where(norms > 0, keys / safe, dtype.type(0.0)).astype(dtype)
    peak = np.abs(keys).max(axis=1)
    odd = (~np.isfinite(norms[:, 0]) | (norms[:, 0] == 0)) & (peak > 0)
    if odd.any():
        scaled = keys[odd] / peak[odd, None]
        result[odd] = (scaled / np.linalg.norm(scaled, axis=1, keepdims=True)).astype(dtype)
    return result


def _check_finite_matrix(values, market, at, name):
    bad = np.flatnonzero(~np.isfinite(values).all(axis=1))
    _require(
        bad.size == 0,
        f"{name} no es finita en {bad.size} filas de {market}; la primera es la fila"
        f" {bad[0] if bad.size else ''} en el instante {at}",
    )


def _check_finite_vector(values, market, at, name):
    bad = np.flatnonzero(~np.isfinite(values))
    _require(
        bad.size == 0,
        f"{name} no es finita en {bad.size} filas de {market}; la primera es la fila"
        f" {bad[0] if bad.size else ''} en el instante {at}",
    )


# -------------------------------------------------------------------------------------------------
# 1. Banco denso con temperatura (media ponderada y dispersión ponderada)
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseBankConfig:
    """Ventana de sesiones maduras (≤ 126), vecinos (≤ 256), temperaturas y recorte del residuo."""

    window: int
    k: int
    temperatures: tuple[float, ...] = (0.05, 0.1, 0.2)
    clip: float = 0.1

    def __post_init__(self):
        _require(
            type(self.window) is int and 1 <= self.window <= DENSE_MAX_WINDOW,
            f"La ventana del banco denso es un entero entre 1 y {DENSE_MAX_WINDOW}",
        )
        _require(
            type(self.k) is int and 1 <= self.k <= DENSE_MAX_K,
            f"El número de vecinos del banco denso es un entero entre 1 y {DENSE_MAX_K}",
        )
        _require(
            isinstance(self.temperatures, tuple)
            and 1 <= len(self.temperatures) <= DENSE_MAX_TEMPERATURES
            and all(
                type(t) in (int, float) and math.isfinite(t) and t > 0 for t in self.temperatures
            ),
            f"Las temperaturas son una tupla de 1 a {DENSE_MAX_TEMPERATURES} números positivos",
        )
        temps = tuple(float(t) for t in self.temperatures)
        _require(
            temps == tuple(sorted(set(temps))),
            "Las temperaturas deben ser distintas y estar en orden creciente",
        )
        object.__setattr__(self, "temperatures", temps)
        _require(
            type(self.clip) in (int, float) and math.isfinite(self.clip) and self.clip > 0,
            "El recorte del banco denso es un número finito y positivo",
        )
        object.__setattr__(self, "clip", float(self.clip))

    def identity(self):
        return dict(
            window=self.window,
            k=self.k,
            temperatures=list(self.temperatures),
            clip=float(self.clip),
        )

    def fingerprint(self):
        return hashlib.sha256(_canonical(self.identity()).encode()).hexdigest()


class _DenseMarket:
    __slots__ = ("bank", "pending", "clock", "matured")

    def __init__(self):
        self.bank = []
        self.pending = []
        self.clock = None
        self.matured = 0


class _DenseScratch:
    """Búferes reutilizables por hilo para similitudes y máscara de candidatas."""

    def __init__(self):
        self._sims = np.empty(0, dtype=np.float32)
        self._mask = np.empty(0, dtype=bool)

    def views(self, rows, size, dtype):
        need = rows * size
        if self._sims.size < need or self._sims.dtype != dtype:
            self._sims = np.empty(need, dtype=dtype)
        if self._mask.size < need:
            self._mask = np.empty(need, dtype=bool)
        return self._sims[:need].reshape(rows, size), self._mask[:need].reshape(rows, size)


class DenseBankMemory:
    """Banco denso causal con temperatura: devuelve media y dispersión ponderadas de residuos."""

    def __init__(self, config, models, *, threads=1):
        _require(isinstance(config, DenseBankConfig), "La configuración es DenseBankConfig")
        _require(
            type(models) is int and 1 <= models <= MAX_MODELS,
            f"El número de modelos es un entero entre 1 y {MAX_MODELS}",
        )
        _require(
            type(threads) is int and 1 <= threads <= MAX_THREADS,
            f"Los hilos de la lectura son un entero entre 1 y {MAX_THREADS}",
        )
        self.config = config
        self.models = models
        self.threads = threads
        self._markets = {}
        self._scratch = queue.SimpleQueue()
        for _ in range(threads):
            self._scratch.put(_DenseScratch())
        self._joined = {}

    def _join(self, market, name, parts):
        size = sum(len(part) for part in parts)
        tail, dtype = parts[0].shape[1:], parts[0].dtype
        buffer = self._joined.get((market, name))
        fits = buffer is not None and buffer.dtype == dtype and buffer.shape[1:] == tail
        if not fits or len(buffer) < size:
            buffer = np.empty((size + size // 2, *tail), dtype=dtype)
            self._joined[(market, name)] = buffer
        return np.concatenate(parts, out=buffer[:size])

    def _market(self, market, *, create=False):
        _require(isinstance(market, str) and market, "El mercado es un texto no vacío")
        if market not in self._markets:
            _require(create, f"El mercado {market!r} no tiene memoria")
            self._markets[market] = _DenseMarket()
        return self._markets[market]

    def _read_bank(self, state, keys, placebo_seed, market, prediction_at):
        rows = len(keys)
        taus = self.config.temperatures
        means = {tau: np.zeros((rows, self.models), dtype=np.float64) for tau in taus}
        disps = {tau: np.zeros((rows, self.models), dtype=np.float64) for tau in taus}
        p_means = (
            None
            if placebo_seed is None
            else {tau: np.zeros((rows, self.models), dtype=np.float64) for tau in taus}
        )
        p_disps = (
            None
            if placebo_seed is None
            else {tau: np.zeros((rows, self.models), dtype=np.float64) for tau in taus}
        )
        if not state.bank:
            return means, disps, p_means, p_disps
        bank_keys = self._join(market, "keys", [c["keys"] for c in state.bank])
        values = self._join(market, "values", [c["values"] for c in state.bank])
        _require(
            bank_keys.shape[1] == keys.shape[1] and bank_keys.dtype == keys.dtype,
            f"Las claves de {market} en {prediction_at} no coinciden en forma o tipo con el banco",
        )
        shuffled = None
        if placebo_seed is not None:
            sequence = np.random.SeedSequence([placebo_seed, _market_seed(market), prediction_at])
            shuffled = _permute_rows(values, np.random.default_rng(sequence))
        size = len(bank_keys)
        k_eff = min(self.config.k, size)

        def read_block(start):
            scratch = self._scratch.get()
            try:
                _read_block(start, scratch)
            finally:
                self._scratch.put(scratch)

        def _read_block(start, scratch):
            block = keys[start : start + DENSE_CHUNK]
            end = start + len(block)
            sims, mask = scratch.views(len(block), size, keys.dtype)
            np.matmul(block, bank_keys.T, out=sims)
            if k_eff < size:
                index = nrm._top(sims, k_eff, mask=mask)
            else:
                index = np.argsort(-sims, axis=1, kind="stable")[:, :k_eff]
            top = np.take_along_axis(sims, index, axis=1).astype(np.float64)
            top_max = top.max(axis=1, keepdims=True)
            gathered = values[index]
            gathered_p = None if shuffled is None else shuffled[index]
            for tau in taus:
                weights = np.exp((top - top_max) / tau)
                weights /= weights.sum(axis=1, keepdims=True)
                mu = np.einsum("ik,ikm->im", weights, gathered)
                diff = gathered - mu[:, None, :]
                var = np.einsum("ik,ikm->im", weights, diff * diff)
                means[tau][start:end] = mu
                disps[tau][start:end] = np.sqrt(np.maximum(var, 0.0))
                if gathered_p is not None:
                    mu_p = np.einsum("ik,ikm->im", weights, gathered_p)
                    diff_p = gathered_p - mu_p[:, None, :]
                    var_p = np.einsum("ik,ikm->im", weights, diff_p * diff_p)
                    p_means[tau][start:end] = mu_p
                    p_disps[tau][start:end] = np.sqrt(np.maximum(var_p, 0.0))

        starts = range(0, rows, DENSE_CHUNK)
        if self.threads == 1 or len(starts) == 1:
            for start in starts:
                read_block(start)
        else:
            with ThreadPoolExecutor(min(self.threads, len(starts))) as pool:
                for _ in pool.map(read_block, starts):
                    pass
        return means, disps, p_means, p_disps

    def emit(self, market, prediction_at, keys, predictions, *, matures_at, placebo_seed=None):
        """Lecturas (media y dispersión) y placebos de una cohorte sin ver su etiqueta."""
        _require(isinstance(market, str) and market, "El mercado es un texto no vacío")
        prediction_at = _instant(prediction_at, "La fecha de decisión")
        matures_at = _matures_instant(matures_at)
        _require(matures_at > prediction_at, "Cada etiqueta madura después de su decisión")
        unit_keys = _normalize_keys(keys)
        predictions = np.asarray(predictions)
        _require(
            predictions.shape == (len(unit_keys), self.models) and predictions.dtype.kind == "f",
            "Una cohorte necesita una predicción real por fila y modelo",
        )
        predictions = predictions.astype(np.float64, copy=True)
        _check_finite_matrix(predictions, market, prediction_at, "La predicción")
        _require(
            placebo_seed is None or (type(placebo_seed) is int and placebo_seed >= 0),
            "La semilla del placebo es un entero no negativo",
        )
        existing = self._markets.get(market)
        if existing is not None:
            _require(
                existing.clock is None or prediction_at >= existing.clock,
                "El reloj del mercado no puede retroceder",
            )
            _require(
                not existing.pending or prediction_at > existing.pending[-1]["prediction_at"],
                "Las cohortes de un mercado se emiten en orden y una sola vez",
            )
            _require(len(existing.pending) < MAX_PENDING, "Demasiadas cohortes pendientes")
            if existing.pending:
                ref = existing.pending[-1]["keys"]
                _require(
                    ref.shape[1] == unit_keys.shape[1] and ref.dtype == unit_keys.dtype,
                    f"Las claves de {market} en {prediction_at} cambian de forma o tipo",
                )
        state = existing if existing is not None else _DenseMarket()
        means, disps, p_means, p_disps = self._read_bank(
            state, unit_keys, placebo_seed, market, prediction_at
        )
        if existing is None:
            self._markets[market] = state
        state.pending.append(
            dict(
                prediction_at=prediction_at,
                matures_at=matures_at,
                keys=np.array(unit_keys, copy=True),
                predictions=predictions,
            )
        )
        state.clock = prediction_at
        return means, disps, p_means, p_disps

    def pending(self, market):
        state = self._markets.get(market)
        if state is None:
            return []
        return [(c["prediction_at"], c["matures_at"]) for c in state.pending]

    def mature(self, market, prediction_at, target, *, at):
        """Escribir en el banco la cohorte más antigua con el residuo recortado `y - p`."""
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
        _check_finite_vector(target, market, prediction_at, "La etiqueta madura")
        clip = self.config.clip
        values = np.clip(target[:, None] - oldest["predictions"], -clip, clip)
        state.bank.append(dict(prediction_at=prediction_at, keys=oldest["keys"], values=values))
        while len(state.bank) > self.config.window:
            state.bank.pop(0)
        state.pending.pop(0)
        state.clock = at
        state.matured += 1

    def export(self):
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
            kind=DENSE_BANK_KIND,
            method=DENSE_BANK_METHOD,
            config=self.config.identity(),
            models=self.models,
            markets=markets,
        )
        payload["state_sha256"] = _state_sha256(payload)
        return payload

    @classmethod
    def restore(cls, payload, *, threads=1):
        _require(
            isinstance(payload, dict)
            and set(payload)
            == {"schema_version", "kind", "method", "config", "models", "markets", "state_sha256"}
            and type(payload["schema_version"]) is int
            and payload["schema_version"] == SCHEMA_VERSION
            and payload["kind"] == DENSE_BANK_KIND
            and payload["method"] == DENSE_BANK_METHOD,
            "El estado pertenece a otra memoria o versión",
        )
        _require(
            _state_sha256(payload) == payload["state_sha256"],
            "El estado exportado no coincide con su huella",
        )
        cfg_data = dict(payload["config"])
        cfg_data["temperatures"] = tuple(cfg_data["temperatures"])
        memory = cls(DenseBankConfig(**cfg_data), payload["models"], threads=threads)
        clip = memory.config.clip
        for name, saved in payload["markets"].items():
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
                and len(bank) <= memory.config.window
                and len(pending) <= MAX_PENDING,
                f"El reloj, las maduradas o las colas guardadas de {name} no son coherentes",
            )
            state = memory._market(name, create=True)
            state.clock, state.matured = clock, matured
            previous, ref_shape = None, None
            for cohort in bank:
                _require(
                    _valid_dense_cohort(cohort, "values", memory.models, clock, previous)
                    and np.all(np.abs(cohort["values"]) <= clip),
                    f"Una cohorte del banco de {name} no es coherente",
                )
                key_spec = (cohort["keys"].shape[1], cohort["keys"].dtype)
                _require(
                    ref_shape is None or ref_shape == key_spec,
                    f"Las claves del banco de {name} no tienen forma y tipo homogéneos",
                )
                ref_shape = key_spec
                previous = cohort["prediction_at"]
                state.bank.append({k: _copied(v) for k, v in cohort.items()})
            for cohort in pending:
                matures = cohort.get("matures_at")
                _require(
                    _valid_dense_cohort(cohort, "predictions", memory.models, clock, previous)
                    and isinstance(matures, (int, np.integer))
                    and not isinstance(matures, bool)
                    and matures > cohort["prediction_at"],
                    f"Una cohorte pendiente de {name} no es coherente",
                )
                key_spec = (cohort["keys"].shape[1], cohort["keys"].dtype)
                _require(
                    ref_shape is None or ref_shape == key_spec,
                    f"Las claves pendientes de {name} no tienen forma y tipo homogéneos",
                )
                ref_shape = key_spec
                previous = cohort["prediction_at"]
                state.pending.append({k: _copied(v) for k, v in cohort.items()})
        return memory


def _valid_dense_cohort(cohort, field, models, clock, previous):
    if not isinstance(cohort, dict):
        return False
    expected = {"prediction_at", "keys", field}
    if field == "predictions":
        expected |= {"matures_at"}
    keys, table, decided = cohort.get("keys"), cohort.get(field), cohort.get("prediction_at")
    return (
        set(cohort) == expected
        and _is_instant(decided)
        and decided <= clock
        and (previous is None or decided > previous)
        and isinstance(keys, np.ndarray)
        and keys.dtype in (np.float32, np.float64)
        and keys.ndim == 2
        and len(keys) >= 1
        and keys.shape[1] >= 1
        and np.isfinite(keys).all()
        and isinstance(table, np.ndarray)
        and table.dtype == np.float64
        and table.shape == (len(keys), models)
        and np.isfinite(table).all()
    )


def replay_dense_bank(
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
):
    """Recorrer un panel en orden de decisión con el banco denso con temperatura."""
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
        {(m, t): rows_ for m, t, rows_ in _cohorts(src_m, src_t)} for src_m, src_t, _ in sources
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

    taus = memory.config.temperatures
    means = {tau: np.full((rows, memory.models), np.nan, dtype=np.float64) for tau in taus}
    disps = {tau: np.full((rows, memory.models), np.nan, dtype=np.float64) for tau in taus}
    p_means = (
        None
        if placebo_seed is None
        else {tau: np.full((rows, memory.models), np.nan, dtype=np.float64) for tau in taus}
    )
    p_disps = (
        None
        if placebo_seed is None
        else {tau: np.full((rows, memory.models), np.nan, dtype=np.float64) for tau in taus}
    )
    for market, moment, chosen in _cohorts(markets, prediction_at):
        settle(market, moment)
        m_out, d_out, pm_out, pd_out = memory.emit(
            market,
            moment,
            keys[chosen],
            predictions[chosen],
            matures_at=int(matures_at[chosen].max()),
            placebo_seed=placebo_seed,
        )
        for tau in taus:
            means[tau][chosen] = m_out[tau]
            disps[tau][chosen] = d_out[tau]
            if p_means is not None:
                p_means[tau][chosen] = pm_out[tau]
                p_disps[tau][chosen] = pd_out[tau]
    if flush:
        for market in sorted(memory._markets):
            state = memory._markets[market]
            while state.pending and state.pending[0]["matures_at"] < FINAL_TEST_START_US:
                before = len(state.pending)
                settle(market, max(state.pending[0]["matures_at"], state.clock or 0))
                _require(
                    len(state.pending) < before,
                    f"El vaciado causal de {market} no pudo madurar la cohorte pendiente",
                )
    return means, disps, p_means, p_disps


# -------------------------------------------------------------------------------------------------
# 2. Memoria holográfica reducida con regla delta (HRR con FFT)
# -------------------------------------------------------------------------------------------------


def _check_no_autocast():
    _require(
        not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
        "La memoria holográfica exige FP32 o FP64 explícitos, sin autocast",
    )


def _torch_dtype(name):
    return torch.float32 if name == "float32" else torch.float64


def unitary_base_vector(dim, seed, namespace, label, *, dtype=torch.float64):
    """Vector aleatorio real, determinista y unitario en frecuencia (|rfft(u)| = 1, ‖u‖₂ = 1)."""
    _check_no_autocast()
    _require(type(dim) is int and dim >= 2, "La dimensión del vector unitario es un entero >= 2")
    _require(type(seed) is int and 0 <= seed < 2**32, "La semilla es un entero uint32")
    _require(
        isinstance(namespace, str) and namespace and isinstance(label, str) and label,
        "El espacio de nombres y la etiqueta del vector base son textos no vacíos",
    )
    _require(
        dtype in (torch.float32, torch.float64),
        "El tipo del vector unitario debe ser torch.float32 o torch.float64",
    )
    digest = hashlib.sha256(f"hrr_unitary_v1|{seed}|{namespace}|{label}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))
    bins = dim // 2 + 1
    spectrum = np.empty(bins, dtype=np.complex128)
    spectrum[0] = 1.0 if int(rng.integers(0, 2)) else -1.0
    if dim % 2 == 0:
        spectrum[-1] = 1.0 if int(rng.integers(0, 2)) else -1.0
        inner = bins - 2
    else:
        inner = bins - 1
    if inner > 0:
        angles = rng.uniform(-math.pi, math.pi, size=inner)
        spectrum[1 : 1 + inner] = np.cos(angles) + 1j * np.sin(angles)
    vector = torch.fft.irfft(torch.from_numpy(spectrum), n=dim)
    return vector.to(dtype=dtype)


def circular_convolution(left, right):
    """Convolución circular vía `torch.fft.rfft` e `irfft` a lo largo de la última dimensión."""
    _check_no_autocast()
    _require(
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.dtype in (torch.float32, torch.float64)
        and left.dtype == right.dtype
        and left.shape[-1] == right.shape[-1]
        and left.shape[-1] >= 2,
        "La convolución circular exige tensores FP32 o FP64 con la misma última dimensión (>= 2)",
    )
    dim = left.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(left, n=dim) * torch.fft.rfft(right, n=dim), n=dim)


def circular_correlation(key, memory):
    """Correlación circular vía `torch.fft.rfft` e `irfft`: `irfft(conj(rfft(k)) * rfft(m))`."""
    _check_no_autocast()
    _require(
        isinstance(key, torch.Tensor)
        and isinstance(memory, torch.Tensor)
        and key.dtype in (torch.float32, torch.float64)
        and key.dtype == memory.dtype
        and key.shape[-1] == memory.shape[-1]
        and key.shape[-1] >= 2,
        "La correlación circular exige tensores FP32 o FP64 con la misma última dimensión (>= 2)",
    )
    dim = key.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(key, n=dim).conj() * torch.fft.rfft(memory, n=dim), n=dim)


@dataclass(frozen=True)
class HolographicDeltaConfig:
    """Dimensión (1024..4096), decaimiento, semilla, entidad (activo/sector), tasa y precisión."""

    dim: int = 1024
    decay: float = 0.95
    seed: int = 0
    entity: str = "asset"
    rate: float = 1.0
    clip: float | None = 0.1
    dtype: str = "float64"

    def __post_init__(self):
        _require(
            type(self.dim) is int and HOLO_MIN_DIM <= self.dim <= HOLO_MAX_DIM,
            f"La dimensión holográfica es un entero entre {HOLO_MIN_DIM} y {HOLO_MAX_DIM}",
        )
        _require(
            type(self.decay) in (int, float)
            and math.isfinite(self.decay)
            and 0.0 < self.decay <= 1.0,
            "El decaimiento holográfico es un número finito en (0, 1]",
        )
        object.__setattr__(self, "decay", float(self.decay))
        _require(
            type(self.seed) is int and 0 <= self.seed < 2**32,
            "La semilla holográfica es un entero uint32",
        )
        _require(
            type(self.entity) is str and self.entity in HOLO_ENTITIES,
            f"La entidad holográfica es una de {HOLO_ENTITIES}",
        )
        _require(
            type(self.rate) in (int, float) and math.isfinite(self.rate) and 0.0 < self.rate <= 1.0,
            "La tasa de la regla delta es un número finito en (0, 1]",
        )
        object.__setattr__(self, "rate", float(self.rate))
        if self.clip is not None:
            _require(
                type(self.clip) in (int, float) and math.isfinite(self.clip) and self.clip > 0.0,
                "El recorte holográfico es None o un número finito y positivo",
            )
            object.__setattr__(self, "clip", float(self.clip))
        _require(
            type(self.dtype) is str and self.dtype in HOLO_DTYPES,
            f"El tipo numérico holográfico es uno de {HOLO_DTYPES}",
        )

    def identity(self):
        return dict(
            dim=self.dim,
            decay=self.decay,
            seed=self.seed,
            entity=self.entity,
            rate=self.rate,
            clip=self.clip,
            dtype=self.dtype,
        )

    def fingerprint(self):
        return hashlib.sha256(_canonical(self.identity()).encode()).hexdigest()


class _HolographicMarket:
    __slots__ = ("memory", "placebo_memory", "pending", "clock", "matured")

    def __init__(self, models, dim, dtype, with_placebo):
        self.memory = torch.zeros((models, dim), dtype=dtype)
        self.placebo_memory = torch.zeros((models, dim), dtype=dtype) if with_placebo else None
        self.pending = []
        self.clock = None
        self.matured = 0


class HolographicDeltaMemory:
    """Memoria holográfica por superposición con claves compuestas y escritura por regla delta."""

    def __init__(self, config, models, *, placebo_seed=None):
        _require(
            isinstance(config, HolographicDeltaConfig),
            "La configuración es HolographicDeltaConfig",
        )
        _require(
            type(models) is int and 1 <= models <= MAX_MODELS,
            f"El número de modelos es un entero entre 1 y {MAX_MODELS}",
        )
        _require(
            placebo_seed is None or (type(placebo_seed) is int and placebo_seed >= 0),
            "La semilla del placebo es un entero no negativo",
        )
        self.config = config
        self.models = models
        self.placebo_seed = placebo_seed
        self._dtype = _torch_dtype(config.dtype)
        self._base_cache = {}
        self._value_vec = unitary_base_vector(
            config.dim, config.seed, "value", "residual", dtype=self._dtype
        )
        self._markets = {}

    def _base(self, namespace, label):
        key = (namespace, label)
        cached = self._base_cache.get(key)
        if cached is None:
            cached = unitary_base_vector(
                self.config.dim, self.config.seed, namespace, label, dtype=self._dtype
            )
            self._base_cache[key] = cached
        return cached

    def _composite_keys(self, entities, regimes):
        e_vecs = torch.stack([self._base(self.config.entity, str(e)) for e in entities])
        r_vecs = torch.stack([self._base("regime", str(r)) for r in regimes])
        return circular_convolution(e_vecs, r_vecs)

    def _read_tensor(self, memory, entities, regimes):
        rows = len(entities)
        output = np.zeros((rows, self.models), dtype=np.float64)
        if not bool(torch.any(memory != 0)):
            return output
        # Deduplicamos pares (entidad, régimen) dentro de la cohorte para acelerar modo sector.
        unique_pairs = {}
        inverse = np.empty(rows, dtype=np.int64)
        ordered = []
        for idx, pair in enumerate(zip(entities.tolist(), regimes.tolist(), strict=True)):
            pos = unique_pairs.get(pair)
            if pos is None:
                pos = len(ordered)
                unique_pairs[pair] = pos
                ordered.append(pair)
            inverse[idx] = pos
        u_entities = [p[0] for p in ordered]
        u_regimes = [p[1] for p in ordered]
        u_out = np.empty((len(ordered), self.models), dtype=np.float64)
        for start in range(0, len(ordered), HOLO_CHUNK):
            end = min(start + HOLO_CHUNK, len(ordered))
            keys = self._composite_keys(u_entities[start:end], u_regimes[start:end])
            decoded = circular_correlation(keys[:, None, :], memory[None, :, :])
            projected = torch.einsum("cmd,d->cm", decoded, self._value_vec)
            u_out[start:end] = projected.to(torch.float64).numpy()
        return u_out[inverse]

    def _market(self, market, *, create=False):
        _require(isinstance(market, str) and market, "El mercado es un texto no vacío")
        if market not in self._markets:
            _require(create, f"El mercado {market!r} no tiene memoria")
            self._markets[market] = _HolographicMarket(
                self.models, self.config.dim, self._dtype, self.placebo_seed is not None
            )
        return self._markets[market]

    def emit(self, market, prediction_at, entities, regimes, predictions, *, matures_at):
        """Leer la superposición por correlación circular sin ver la etiqueta de la cohorte."""
        _check_no_autocast()
        _require(isinstance(market, str) and market, "El mercado es un texto no vacío")
        prediction_at = _instant(prediction_at, "La fecha de decisión")
        matures_at = _matures_instant(matures_at)
        _require(matures_at > prediction_at, "Cada etiqueta madura después de su decisión")
        entities = np.asarray(entities)
        regimes = np.asarray(regimes)
        predictions = np.asarray(predictions)
        _require(
            entities.ndim == 1
            and entities.dtype.kind == "U"
            and len(entities) >= 1
            and regimes.shape == entities.shape
            and regimes.dtype.kind == "U"
            and predictions.shape == (len(entities), self.models)
            and predictions.dtype.kind == "f",
            "Una cohorte holográfica necesita entidades, regímenes de texto y predicciones reales",
        )
        _require(
            all(e != "" for e in entities.tolist()) and all(r != "" for r in regimes.tolist()),
            "Las entidades y los regímenes deben ser textos no vacíos",
        )
        predictions = predictions.astype(np.float64, copy=True)
        _check_finite_matrix(predictions, market, prediction_at, "La predicción")
        existing = self._markets.get(market)
        if existing is not None:
            _require(
                existing.clock is None or prediction_at >= existing.clock,
                "El reloj del mercado no puede retroceder",
            )
            _require(
                not existing.pending or prediction_at > existing.pending[-1]["prediction_at"],
                "Las cohortes de un mercado se emiten en orden y una sola vez",
            )
            _require(len(existing.pending) < MAX_PENDING, "Demasiadas cohortes pendientes")
        state = (
            existing
            if existing is not None
            else _HolographicMarket(
                self.models, self.config.dim, self._dtype, self.placebo_seed is not None
            )
        )
        reads = self._read_tensor(state.memory, entities, regimes)
        placebo = (
            None
            if state.placebo_memory is None
            else self._read_tensor(state.placebo_memory, entities, regimes)
        )
        if existing is None:
            self._markets[market] = state
        state.pending.append(
            dict(
                prediction_at=prediction_at,
                matures_at=matures_at,
                entities=np.array(entities, copy=True),
                regimes=np.array(regimes, copy=True),
                predictions=predictions,
            )
        )
        state.clock = prediction_at
        return reads, placebo

    def pending(self, market):
        state = self._markets.get(market)
        if state is None:
            return []
        return [(c["prediction_at"], c["matures_at"]) for c in state.pending]

    def _delta_write(self, memory, bound, values):
        updated = memory * self.config.decay
        rate = self.config.rate
        vals = torch.from_numpy(values).to(dtype=self._dtype)
        for idx in range(len(bound)):
            vec = bound[idx]
            pred = updated @ vec
            delta = vals[idx] - pred
            updated = updated + rate * delta[:, None] * vec[None, :]
        return updated

    def mature(self, market, prediction_at, target, *, at):
        """Aplicar decaimiento y regla delta con los residuos maduros de la cohorte más antigua."""
        _check_no_autocast()
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
            target.shape == (len(oldest["entities"]),) and target.dtype.kind == "f",
            "Se necesita una etiqueta real por fila de la cohorte emitida",
        )
        target = target.astype(np.float64)
        _check_finite_vector(target, market, prediction_at, "La etiqueta madura")
        residuals = target[:, None] - oldest["predictions"]
        if self.config.clip is not None:
            residuals = np.clip(residuals, -self.config.clip, self.config.clip)
        # Agrupación canónica por (entidad, régimen) ordenada lexicográficamente: invariante al
        # orden de las filas dentro de la cohorte (crucial en modo sector).
        grouped = {}
        for row_idx, pair in enumerate(
            zip(oldest["entities"].tolist(), oldest["regimes"].tolist(), strict=True)
        ):
            grouped.setdefault(pair, []).append(row_idx)
        pairs = sorted(grouped)
        mean_values = np.vstack([residuals[grouped[pair]].mean(axis=0) for pair in pairs])
        keys = self._composite_keys([p[0] for p in pairs], [p[1] for p in pairs])
        bound = circular_convolution(keys, self._value_vec[None, :].expand_as(keys))
        state.memory = self._delta_write(state.memory, bound, mean_values)
        if state.placebo_memory is not None:
            sequence = np.random.SeedSequence(
                [self.placebo_seed, _market_seed(market), prediction_at]
            )
            p_values = _permute_rows(mean_values, np.random.default_rng(sequence))
            state.placebo_memory = self._delta_write(state.placebo_memory, bound, p_values)
        state.pending.pop(0)
        state.clock = at
        state.matured += 1

    def export(self):
        markets = {}
        for name in sorted(self._markets):
            state = self._markets[name]
            entry = dict(
                clock=state.clock,
                matured=state.matured,
                memory=state.memory.detach().cpu().numpy().copy(),
                pending=[
                    dict(
                        prediction_at=c["prediction_at"],
                        matures_at=c["matures_at"],
                        entities=c["entities"].tolist(),
                        regimes=c["regimes"].tolist(),
                        predictions=np.array(c["predictions"], dtype=np.float64, copy=True),
                    )
                    for c in state.pending
                ],
            )
            if state.placebo_memory is not None:
                entry["placebo_memory"] = state.placebo_memory.detach().cpu().numpy().copy()
            markets[name] = entry
        payload = dict(
            schema_version=SCHEMA_VERSION,
            kind=HOLOGRAPHIC_KIND,
            method=HOLOGRAPHIC_METHOD,
            config=self.config.identity(),
            models=self.models,
            placebo_seed=self.placebo_seed,
            markets=markets,
        )
        payload["state_sha256"] = _state_sha256(payload)
        return payload

    @classmethod
    def restore(cls, payload):
        _require(
            isinstance(payload, dict)
            and set(payload)
            == {
                "schema_version",
                "kind",
                "method",
                "config",
                "models",
                "placebo_seed",
                "markets",
                "state_sha256",
            }
            and type(payload["schema_version"]) is int
            and payload["schema_version"] == SCHEMA_VERSION
            and payload["kind"] == HOLOGRAPHIC_KIND
            and payload["method"] == HOLOGRAPHIC_METHOD,
            "El estado pertenece a otra memoria o versión",
        )
        _require(
            _state_sha256(payload) == payload["state_sha256"],
            "El estado exportado no coincide con su huella",
        )
        memory = cls(
            HolographicDeltaConfig(**payload["config"]),
            payload["models"],
            placebo_seed=payload["placebo_seed"],
        )
        np_dtype = np.float32 if memory.config.dtype == "float32" else np.float64
        expected_fields = {"clock", "matured", "memory", "pending"}
        if memory.placebo_seed is not None:
            expected_fields |= {"placebo_memory"}
        for name, saved in payload["markets"].items():
            _require(
                isinstance(name, str)
                and name
                and isinstance(saved, dict)
                and set(saved) == expected_fields,
                f"El estado guardado del mercado {name!r} no es coherente",
            )
            clock, matured = saved["clock"], saved["matured"]
            arr = saved["memory"]
            _require(
                (clock is None or _is_instant(clock))
                and (clock is not None or not saved["pending"])
                and type(matured) is int
                and matured >= 0
                and isinstance(arr, np.ndarray)
                and arr.dtype == np_dtype
                and arr.shape == (memory.models, memory.config.dim)
                and np.isfinite(arr).all()
                and len(saved["pending"]) <= MAX_PENDING,
                f"El estado holográfico guardado de {name} no es coherente",
            )
            state = memory._market(name, create=True)
            state.clock, state.matured = clock, matured
            state.memory = torch.from_numpy(arr.copy())
            if memory.placebo_seed is not None:
                p_arr = saved["placebo_memory"]
                _require(
                    isinstance(p_arr, np.ndarray)
                    and p_arr.dtype == np_dtype
                    and p_arr.shape == (memory.models, memory.config.dim)
                    and np.isfinite(p_arr).all(),
                    f"La memoria placebo guardada de {name} no es coherente",
                )
                state.placebo_memory = torch.from_numpy(p_arr.copy())
            previous = None
            for cohort in saved["pending"]:
                _require(
                    isinstance(cohort, dict)
                    and set(cohort)
                    == {"prediction_at", "matures_at", "entities", "regimes", "predictions"},
                    f"Una cohorte pendiente de {name} no es coherente",
                )
                decided, matures = cohort["prediction_at"], cohort["matures_at"]
                ents, regs, preds = cohort["entities"], cohort["regimes"], cohort["predictions"]
                _require(
                    _is_instant(decided)
                    and isinstance(matures, (int, np.integer))
                    and not isinstance(matures, bool)
                    and matures > decided
                    and decided <= clock
                    and (previous is None or decided > previous)
                    and isinstance(ents, list)
                    and isinstance(regs, list)
                    and len(ents) >= 1
                    and len(ents) == len(regs)
                    and all(isinstance(e, str) and e for e in ents)
                    and all(isinstance(r, str) and r for r in regs)
                    and isinstance(preds, np.ndarray)
                    and preds.dtype == np.float64
                    and preds.shape == (len(ents), memory.models)
                    and np.isfinite(preds).all(),
                    f"Una cohorte pendiente de {name} no es coherente",
                )
                state.pending.append(
                    dict(
                        prediction_at=decided,
                        matures_at=matures,
                        entities=np.array(ents),
                        regimes=np.array(regs),
                        predictions=preds.copy(),
                    )
                )
                previous = decided
        return memory


def replay_holographic_delta(
    memory,
    *,
    markets,
    prediction_at,
    matures_at,
    entities,
    regimes,
    predictions,
    target,
    flush=True,
    pending_targets=None,
):
    """Recorrer un panel en orden de decisión con la memoria holográfica con regla delta."""
    markets, prediction_at = np.asarray(markets), np.asarray(prediction_at)
    matures_at, target = np.asarray(matures_at), np.asarray(target)
    entities, regimes = np.asarray(entities), np.asarray(regimes)
    predictions = np.asarray(predictions)
    rows = len(markets)
    _require(
        rows >= 1
        and all(a.shape == (rows,) for a in (prediction_at, matures_at, entities, regimes, target))
        and predictions.shape == (rows, memory.models)
        and prediction_at.dtype.kind == "i"
        and matures_at.dtype.kind == "i",
        "El panel holográfico necesita columnas alineadas e instantes enteros",
    )
    _require(
        bool(np.all(matures_at > prediction_at)), "Cada etiqueta madura después de su decisión"
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
        {(m, t): rows_ for m, t, rows_ in _cohorts(src_m, src_t)} for src_m, src_t, _ in sources
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

    reads = np.full((rows, memory.models), np.nan, dtype=np.float64)
    placebo = (
        None
        if memory.placebo_seed is None
        else np.full((rows, memory.models), np.nan, dtype=np.float64)
    )
    for market, moment, chosen in _cohorts(markets, prediction_at):
        settle(market, moment)
        out, fake = memory.emit(
            market,
            moment,
            entities[chosen],
            regimes[chosen],
            predictions[chosen],
            matures_at=int(matures_at[chosen].max()),
        )
        reads[chosen] = out
        if placebo is not None:
            placebo[chosen] = fake
    if flush:
        for market in sorted(memory._markets):
            state = memory._markets[market]
            while state.pending and state.pending[0]["matures_at"] < FINAL_TEST_START_US:
                before = len(state.pending)
                settle(market, max(state.pending[0]["matures_at"], state.clock or 0))
                _require(
                    len(state.pending) < before,
                    f"El vaciado causal de {market} no pudo madurar la cohorte pendiente",
                )
    return reads, placebo


# -------------------------------------------------------------------------------------------------
# 3. Traza por activo (media exponencial de residuos maduros del padre)
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetTraceConfig:
    """Semivida en sesiones maduras del mercado y recorte opcional del residuo `y - p`."""

    half_life: float
    clip: float | None = None

    def __post_init__(self):
        _require(
            type(self.half_life) in (int, float)
            and math.isfinite(self.half_life)
            and self.half_life > 0.0,
            "La semivida de la traza por activo es un número finito y positivo",
        )
        object.__setattr__(self, "half_life", float(self.half_life))
        if self.clip is not None:
            _require(
                type(self.clip) in (int, float) and math.isfinite(self.clip) and self.clip > 0.0,
                "El recorte de la traza por activo es None o un número finito y positivo",
            )
            object.__setattr__(self, "clip", float(self.clip))

    @property
    def decay(self):
        return 0.5 ** (1.0 / self.half_life)

    def identity(self):
        return dict(half_life=self.half_life, clip=self.clip)

    def fingerprint(self):
        return hashlib.sha256(_canonical(self.identity()).encode()).hexdigest()


class _TraceMarket:
    __slots__ = (
        "index",
        "assets",
        "total",
        "weight",
        "last_session",
        "clock",
        "matured",
        "pending",
    )

    def __init__(self, models):
        self.index, self.assets = {}, []
        self.total = np.zeros((0, models), dtype=np.float64)
        self.weight = np.zeros(0, dtype=np.float64)
        self.last_session = np.zeros(0, dtype=np.int64)
        self.clock, self.matured, self.pending = None, 0, []

    def positions(self, assets, *, grow, models):
        missing = [a for a in dict.fromkeys(assets.tolist()) if a not in self.index]
        if missing and grow:
            _require(
                len(self.assets) + len(missing) <= MAX_ASSETS,
                "La traza por activo supera el máximo de activos por mercado",
            )
            for asset in missing:
                self.index[asset] = len(self.assets)
                self.assets.append(asset)
            self.total = np.vstack([self.total, np.zeros((len(missing), models), dtype=np.float64)])
            self.weight = np.concatenate([self.weight, np.zeros(len(missing), dtype=np.float64)])
            self.last_session = np.concatenate(
                [self.last_session, np.full(len(missing), -1, dtype=np.int64)]
            )
        return np.array([self.index.get(a, -1) for a in assets.tolist()], dtype=np.int64)


class AssetResidualTrace:
    """Media exponencial causal de los residuos maduros `y - p` de cada activo."""

    def __init__(self, config, models):
        _require(isinstance(config, AssetTraceConfig), "La configuración es AssetTraceConfig")
        _require(
            type(models) is int and 1 <= models <= MAX_MODELS,
            f"El número de modelos es un entero entre 1 y {MAX_MODELS}",
        )
        self.config = config
        self.models = models
        self._markets = {}

    def _market(self, market, *, create=False):
        _require(isinstance(market, str) and market, "El mercado es un texto no vacío")
        if market not in self._markets:
            _require(create, f"El mercado {market!r} no tiene memoria")
            self._markets[market] = _TraceMarket(self.models)
        return self._markets[market]

    def read(self, market, assets):
        """Media exponencial y peso efectivo de cada activo; cero si no tiene historia."""
        assets = np.asarray(assets)
        state = self._markets.get(market)
        if state is None:
            return np.zeros((len(assets), self.models), dtype=np.float64), np.zeros(
                len(assets), dtype=np.float64
            )
        positions = state.positions(assets, grow=False, models=self.models)
        known = positions >= 0
        safe_pos = np.maximum(positions, 0)
        weight = np.where(known, state.weight[safe_pos], 0.0)
        total = np.where(known[:, None], state.total[safe_pos], 0.0)
        mean = np.divide(
            total,
            weight[:, None],
            out=np.zeros((len(assets), self.models), dtype=np.float64),
            where=weight[:, None] > 0.0,
        )
        return mean, weight

    def emit(self, market, prediction_at, assets, predictions, *, matures_at, placebo_seed=None):
        """Leer las trazas de la cohorte (y su placebo por permutación) sin ver su etiqueta."""
        _require(isinstance(market, str) and market, "El mercado es un texto no vacío")
        prediction_at = _instant(prediction_at, "La fecha de decisión")
        matures_at = _matures_instant(matures_at)
        _require(matures_at > prediction_at, "Cada etiqueta madura después de su decisión")
        assets = np.asarray(assets)
        predictions = np.asarray(predictions)
        _require(
            assets.ndim == 1
            and assets.dtype.kind == "U"
            and len(assets) >= 1
            and predictions.shape == (len(assets), self.models)
            and predictions.dtype.kind == "f",
            "Una cohorte necesita activos de texto y una predicción real por activo y modelo",
        )
        _require(
            len(set(assets.tolist())) == len(assets),
            f"Hay un activo repetido en la sesión {market} {prediction_at}",
        )
        predictions = predictions.astype(np.float64, copy=True)
        _check_finite_matrix(predictions, market, prediction_at, "La predicción")
        _require(
            placebo_seed is None or (type(placebo_seed) is int and placebo_seed >= 0),
            "La semilla del placebo es un entero no negativo",
        )
        existing = self._markets.get(market)
        if existing is not None:
            _require(
                existing.clock is None or prediction_at >= existing.clock,
                "El reloj del mercado no puede retroceder",
            )
            _require(
                not existing.pending or prediction_at > existing.pending[-1]["prediction_at"],
                "Las cohortes de un mercado se emiten en orden y una sola vez",
            )
            _require(len(existing.pending) < MAX_PENDING, "Demasiadas cohortes pendientes")
        reads, _ = self.read(market, assets)
        placebo = None
        if placebo_seed is not None:
            sequence = np.random.SeedSequence([placebo_seed, _market_seed(market), prediction_at])
            placebo = _permute_rows(reads, np.random.default_rng(sequence))
        state = existing if existing is not None else _TraceMarket(self.models)
        state.positions(assets, grow=True, models=self.models)
        if existing is None:
            self._markets[market] = state
        state.pending.append(
            dict(
                prediction_at=prediction_at,
                matures_at=matures_at,
                assets=np.array(assets, copy=True),
                predictions=predictions,
            )
        )
        state.clock = prediction_at
        return reads, placebo

    def pending(self, market):
        state = self._markets.get(market)
        if state is None:
            return []
        return [(c["prediction_at"], c["matures_at"]) for c in state.pending]

    def mature(self, market, prediction_at, target, *, at):
        """Actualizar la media exponencial con los residuos maduros de la cohorte más antigua."""
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
            target.shape == oldest["assets"].shape and target.dtype.kind == "f",
            "Se necesita una etiqueta real por activo de la cohorte emitida",
        )
        target = target.astype(np.float64)
        _check_finite_vector(target, market, prediction_at, "La etiqueta madura")
        residuals = target[:, None] - oldest["predictions"]
        if self.config.clip is not None:
            residuals = np.clip(residuals, -self.config.clip, self.config.clip)
        positions = state.positions(oldest["assets"], grow=False, models=self.models)
        _require(bool(np.all(positions >= 0)), "Un activo de la cohorte no tiene posición")
        step = state.matured + 1
        decay = self.config.decay
        prev = state.last_session[positions]
        seen = prev >= 0
        factors = np.where(seen, decay ** np.where(seen, step - prev, 0), 0.0)
        state.total[positions] = factors[:, None] * state.total[positions] + residuals
        state.weight[positions] = factors * state.weight[positions] + 1.0
        state.last_session[positions] = step
        state.pending.pop(0)
        state.matured = step
        state.clock = at

    def export(self):
        markets = {}
        for name in sorted(self._markets):
            state = self._markets[name]
            markets[name] = dict(
                assets=list(state.assets),
                total=np.array(state.total, dtype=np.float64, copy=True),
                weight=np.array(state.weight, dtype=np.float64, copy=True),
                last_session=np.array(state.last_session, dtype=np.int64, copy=True),
                clock=state.clock,
                matured=state.matured,
                pending=[
                    dict(
                        prediction_at=c["prediction_at"],
                        matures_at=c["matures_at"],
                        assets=c["assets"].tolist(),
                        predictions=np.array(c["predictions"], dtype=np.float64, copy=True),
                    )
                    for c in state.pending
                ],
            )
        payload = dict(
            schema_version=SCHEMA_VERSION,
            kind=ASSET_TRACE_KIND,
            method=ASSET_TRACE_METHOD,
            config=self.config.identity(),
            models=self.models,
            markets=markets,
        )
        payload["state_sha256"] = _state_sha256(payload)
        return payload

    @classmethod
    def restore(cls, payload):
        _require(
            isinstance(payload, dict)
            and set(payload)
            == {"schema_version", "kind", "method", "config", "models", "markets", "state_sha256"}
            and type(payload["schema_version"]) is int
            and payload["schema_version"] == SCHEMA_VERSION
            and payload["kind"] == ASSET_TRACE_KIND
            and payload["method"] == ASSET_TRACE_METHOD,
            "El estado pertenece a otra memoria o versión",
        )
        _require(
            _state_sha256(payload) == payload["state_sha256"],
            "El estado exportado no coincide con su huella",
        )
        memory = cls(AssetTraceConfig(**payload["config"]), payload["models"])
        for name, saved in copy.deepcopy(payload["markets"]).items():
            _require(
                isinstance(name, str)
                and name
                and isinstance(saved, dict)
                and set(saved)
                == {
                    "assets",
                    "total",
                    "weight",
                    "last_session",
                    "clock",
                    "matured",
                    "pending",
                },
                f"El estado guardado del mercado {name!r} no es coherente",
            )
            assets, clock, matured = saved["assets"], saved["clock"], saved["matured"]
            total, weight, last = saved["total"], saved["weight"], saved["last_session"]
            _require(
                isinstance(assets, list)
                and len(set(assets)) == len(assets) <= MAX_ASSETS
                and all(isinstance(a, str) and a for a in assets)
                and isinstance(total, np.ndarray)
                and total.dtype == np.float64
                and total.shape == (len(assets), memory.models)
                and np.isfinite(total).all()
                and isinstance(weight, np.ndarray)
                and weight.dtype == np.float64
                and weight.shape == (len(assets),)
                and np.isfinite(weight).all()
                and np.all(weight >= 0.0)
                and isinstance(last, np.ndarray)
                and last.dtype == np.int64
                and last.shape == (len(assets),)
                and np.all((last == -1) | ((last >= 1) & (last <= matured)))
                and (clock is None or _is_instant(clock))
                and (clock is not None or not saved["pending"])
                and type(matured) is int
                and matured >= 0
                and len(saved["pending"]) <= MAX_PENDING,
                f"La traza guardada de {name} no es coherente",
            )
            state = memory._market(name, create=True)
            state.assets = list(assets)
            state.index = {asset: i for i, asset in enumerate(assets)}
            state.total, state.weight, state.last_session = total, weight, last
            state.clock, state.matured = clock, matured
            previous = None
            for cohort in saved["pending"]:
                _require(
                    isinstance(cohort, dict)
                    and set(cohort) == {"prediction_at", "matures_at", "assets", "predictions"},
                    f"Una cohorte pendiente de {name} no es coherente",
                )
                decided, matures = cohort["prediction_at"], cohort["matures_at"]
                preds = cohort["predictions"]
                _require(
                    _is_instant(decided)
                    and isinstance(matures, (int, np.integer))
                    and not isinstance(matures, bool)
                    and matures > decided
                    and decided <= clock
                    and (previous is None or decided > previous)
                    and isinstance(cohort["assets"], list)
                    and len(cohort["assets"]) >= 1
                    and len(set(cohort["assets"])) == len(cohort["assets"])
                    and all(a in state.index for a in cohort["assets"])
                    and isinstance(preds, np.ndarray)
                    and preds.dtype == np.float64
                    and preds.shape == (len(cohort["assets"]), memory.models)
                    and np.isfinite(preds).all(),
                    f"Una cohorte pendiente de {name} no es coherente",
                )
                state.pending.append(
                    dict(
                        prediction_at=decided,
                        matures_at=matures,
                        assets=np.array(cohort["assets"]),
                        predictions=preds.copy(),
                    )
                )
                previous = decided
        return memory


def _trace_targets(sources, market, moment, assets):
    for lookup, src_assets, src_target in sources:
        rows = lookup.get((market, moment))
        if rows is None:
            continue
        found = dict(zip(src_assets[rows].tolist(), src_target[rows], strict=True))
        _require(
            all(a in found for a in assets.tolist()),
            f"Falta la etiqueta de algún activo de la cohorte {market} {moment}",
        )
        return np.array([found[a] for a in assets.tolist()], dtype=np.float64)
    raise ValueError(f"No se encuentran las etiquetas de la cohorte {market} {moment}")


def replay_asset_trace(
    memory,
    *,
    markets,
    assets,
    prediction_at,
    matures_at,
    predictions,
    target,
    placebo_seed=None,
    flush=True,
    pending_targets=None,
):
    """Recorrer un panel en orden de decisión con la traza exponencial por activo."""
    markets, assets = np.asarray(markets), np.asarray(assets)
    prediction_at, matures_at = np.asarray(prediction_at), np.asarray(matures_at)
    predictions, target = np.asarray(predictions), np.asarray(target)
    rows = len(markets)
    _require(
        rows >= 1
        and all(a.shape == (rows,) for a in (assets, prediction_at, matures_at, target))
        and predictions.shape == (rows, memory.models)
        and prediction_at.dtype.kind == "i"
        and matures_at.dtype.kind == "i",
        "El panel de la traza necesita columnas alineadas e instantes enteros",
    )
    _require(
        bool(np.all(matures_at > prediction_at)), "Cada etiqueta madura después de su decisión"
    )
    sources = [
        (
            {(m, t): rows_ for m, t, rows_ in _cohorts(markets, prediction_at)},
            assets,
            target,
        )
    ]
    if pending_targets is not None:
        pt_m = np.asarray(pending_targets["markets"])
        pt_t = np.asarray(pending_targets["prediction_at"])
        sources.append(
            (
                {(m, t): rows_ for m, t, rows_ in _cohorts(pt_m, pt_t)},
                np.asarray(pending_targets["assets"]),
                np.asarray(pending_targets["target"]),
            )
        )

    def settle(market, now):
        state = memory._markets.get(market)
        while state is not None and state.pending and state.pending[0]["matures_at"] <= now:
            cohort = state.pending[0]
            labels = _trace_targets(sources, market, cohort["prediction_at"], cohort["assets"])
            memory.mature(market, cohort["prediction_at"], labels, at=now)

    reads = np.full((rows, memory.models), np.nan, dtype=np.float64)
    placebo = (
        None if placebo_seed is None else np.full((rows, memory.models), np.nan, dtype=np.float64)
    )
    for market, moment, chosen in _cohorts(markets, prediction_at):
        settle(market, moment)
        out, fake = memory.emit(
            market,
            moment,
            assets[chosen],
            predictions[chosen],
            matures_at=int(matures_at[chosen].max()),
            placebo_seed=placebo_seed,
        )
        reads[chosen] = out
        if placebo is not None:
            placebo[chosen] = fake
    if flush:
        for market in sorted(memory._markets):
            state = memory._markets[market]
            while state.pending and state.pending[0]["matures_at"] < FINAL_TEST_START_US:
                before = len(state.pending)
                settle(market, max(state.pending[0]["matures_at"], state.clock or 0))
                _require(
                    len(state.pending) < before,
                    f"El vaciado causal de {market} no pudo madurar la cohorte pendiente",
                )
    return reads, placebo


# -------------------------------------------------------------------------------------------------
# 4. Régimen causal observable por sesión a partir de las columnas escaladas de view_keys
# -------------------------------------------------------------------------------------------------


def session_regimes_from_inputs(raw, slices, *, min_assets=1):
    """Régimen observable por sesión (tendencia 63 × volatilidad 21/252) solo con rasgos pasados.

    Usa exclusivamente las columnas escaladas con filas de ajuste de `view_keys` (`return_63` y
    `volatility_21_over_252`). La mediana transversal no depende del orden de las filas dentro de
    la sesión y ninguna etiqueta ni sesión futura interviene.
    """
    raw = np.asarray(raw)
    _require(
        raw.ndim == 2 and raw.shape[1] >= len(vk.COLUMNS),
        f"Se necesitan al menos {len(vk.COLUMNS)} columnas escaladas de view_keys",
    )
    _require(type(min_assets) is int and min_assets >= 1, "min_assets es un entero positivo")
    idx_ret = vk.COLUMNS.index("return_63")
    idx_vol = vk.COLUMNS.index("volatility_21_over_252")
    regimes = np.empty(len(raw), dtype="<U16")
    covered = np.zeros(len(raw), dtype=bool)
    for _, _, _, start, end in slices:
        _require(
            0 <= start < end <= len(raw) and not covered[start:end].any(),
            "Las sesiones de régimen deben particionar las filas sin huecos ni solapes",
        )
        covered[start:end] = True
        if end - start < min_assets:
            label = "unclassified"
        else:
            ret_slice = raw[start:end, idx_ret].astype(np.float64)
            vol_slice = raw[start:end, idx_vol].astype(np.float64)
            if not (np.isfinite(ret_slice).all() and np.isfinite(vol_slice).all()):
                label = "unclassified"
            else:
                trend = float(np.median(ret_slice))
                vol = float(np.median(vol_slice))
                label = ("calm" if vol <= 0.0 else "turbulent") + (
                    "_up" if trend > 0.0 else "_down"
                )
        regimes[start:end] = label
    _require(
        bool(covered.all()),
        "Las sesiones de régimen deben cubrir todas las filas de entrada",
    )
    return regimes


# -------------------------------------------------------------------------------------------------
# 5. Validación de declaración y estudio de los órganos nuevos para memory_organs
# -------------------------------------------------------------------------------------------------


def _sorted_grid(values, low, high, label, *, strict_low=False):
    _require(
        isinstance(values, list)
        and 1 <= len(values) <= MAX_GRID
        and all(
            type(v) in (int, float)
            and math.isfinite(v)
            and ((low < v <= high) if strict_low else (low <= v <= high))
            for v in values
        )
        and values == sorted(set(values)),
        f"{label} es una lista creciente, sin repetir, de 1 a {MAX_GRID} valores válidos",
    )
    return [float(v) for v in values]


def checked_episodic_organs(section):
    """Validar la sección opcional `episodic_organs` de una declaración de memory_organs."""
    _require(
        isinstance(section, dict) and bool(section) and set(section) <= set(EPISODIC_ARMS),
        f"episodic_organs declara un subconjunto no vacío de {list(EPISODIC_ARMS)}",
    )
    if "dense_bank" in section:
        cfg = section["dense_bank"]
        expected = {"window", "k", "temperatures", "clip", "kappas"}
        _require(
            isinstance(cfg, dict) and set(cfg) == expected,
            f"dense_bank declara exactamente los campos {sorted(expected)}",
        )
        _require(
            isinstance(cfg["temperatures"], list),
            "dense_bank.temperatures es una lista creciente de temperaturas positivas",
        )
        DenseBankConfig(
            window=cfg["window"],
            k=cfg["k"],
            temperatures=tuple(cfg["temperatures"]),
            clip=cfg["clip"],
        )
        kappas = _sorted_grid(cfg["kappas"], 0.0, 10.0, "Los pesos κ de dense_bank")
        _require(any(k > 0.0 for k in kappas), "dense_bank necesita al menos un peso κ > 0")

    if "holographic_delta" in section:
        cfg = section["holographic_delta"]
        required = {"dim", "decay", "seed", "entity", "kappas"}
        optional = {"rate", "clip", "dtype", "sector_source"}
        _require(
            isinstance(cfg, dict) and required <= set(cfg) <= required | optional,
            f"holographic_delta declara {sorted(required)} y opcionalmente {sorted(optional)}",
        )
        HolographicDeltaConfig(
            dim=cfg["dim"],
            decay=cfg["decay"],
            seed=cfg["seed"],
            entity=cfg["entity"],
            rate=cfg.get("rate", 1.0),
            clip=cfg.get("clip", 0.1),
            dtype=cfg.get("dtype", "float64"),
        )
        sector_source = cfg.get("sector_source")
        if cfg["entity"] == "asset":
            _require(
                sector_source is None,
                "sector_source solo se declara con entity='sector'",
            )
        elif sector_source is not None:
            _require(
                isinstance(sector_source, str) and bool(sector_source),
                "sector_source es una ruta no vacía a las fichas de sectores",
            )
        kappas = _sorted_grid(cfg["kappas"], 0.0, 10.0, "Los pesos κ de holographic_delta")
        _require(any(k > 0.0 for k in kappas), "holographic_delta necesita al menos un peso κ > 0")

    if "asset_trace" in section:
        cfg = section["asset_trace"]
        required = {"half_lives", "kappas"}
        optional = {"clip"}
        _require(
            isinstance(cfg, dict) and required <= set(cfg) <= required | optional,
            f"asset_trace declara {sorted(required)} y opcionalmente {sorted(optional)}",
        )
        half_lives = _sorted_grid(
            cfg["half_lives"], 0.0, 10_000.0, "Las semividas de asset_trace", strict_low=True
        )
        clip = cfg.get("clip")
        for half_life in half_lives:
            AssetTraceConfig(half_life=half_life, clip=clip)
        kappas = _sorted_grid(cfg["kappas"], 0.0, 10.0, "Los pesos κ de asset_trace")
        _require(any(k > 0.0 for k in kappas), "asset_trace necesita al menos un peso κ > 0")
    return section


def resolve_sector_entities(assets, *, sector_map=None, sector_source=None):
    """Mapear cada activo a su grupo sectorial con un SectorMap o desde `sector_source`."""
    if sector_map is None:
        _require(
            bool(sector_source),
            "El modo entity='sector' necesita sector_source en la declaración o sector_map",
        )
        sector_map = SectorMap.from_finmultitime(sector_source)
    if isinstance(sector_map, SectorMap):
        return np.array([sector_map.groups[sector_map.group(a)] for a in assets.tolist()])
    if isinstance(sector_map, dict):
        missing = [a for a in assets.tolist() if a not in sector_map]
        _require(not missing, f"Falta el sector de {missing[0] if missing else ''}")
        return np.array([str(sector_map[a]) for a in assets.tolist()])
    raise ValueError("sector_map debe ser un SectorMap o un diccionario {activo: sector}")


def _uncertainty_summary(splits, bounds, dispersion, residual, module):
    """Resumen de incertidumbre en calibración y evaluación: media, mediana y correlación."""
    summary = {}
    for idx, name in enumerate(("validation", "calibration", "evaluation")):
        if name == "validation":
            continue
        start, end = int(bounds[idx]), int(bounds[idx + 1])
        unc = dispersion[start:end]
        abs_err = np.abs(residual[start:end])
        corrs = []
        for _, _, s_start, s_end in splits[name].sessions:
            if s_end - s_start < module.MIN_ASSETS:
                continue
            val = module.correlation(
                module.average_ranks(unc[s_start:s_end]),
                module.average_ranks(abs_err[s_start:s_end]),
            )
            if math.isfinite(val):
                corrs.append(val)
        summary[name] = dict(
            mean=float(np.mean(unc)),
            median=float(np.median(unc)),
            abs_error_rank_corr=float(np.mean(corrs)) if corrs else None,
        )
    return summary


def episodic_study(
    splits,
    predictions,
    keys,
    section,
    *,
    placebo_seed,
    regimes=None,
    sector_map=None,
    threads=1,
    log=print,
):
    """Medir y preparar la emisión de los órganos episódicos declarados en `section`.

    Toda selección de hiperparámetros (temperatura, semivida, κ) usa exclusivamente el IC medio
    de calibración; la evaluación jamás interviene en ninguna elección.
    """
    from mars_titan.evaluation import posttraining_pilots as pilots

    checked_episodic_organs(section)
    _require(type(placebo_seed) is int and placebo_seed >= 0, "placebo_seed es un entero >= 0")
    parts = [splits[name] for name in pilots.SPLITS]
    joined = dict(
        markets=np.concatenate([p.market for p in parts]),
        assets=np.concatenate([p.asset for p in parts]),
        prediction_at=np.concatenate([p.prediction_at for p in parts]),
        matures_at=np.concatenate([p.matures_at for p in parts]),
        target=np.concatenate([p.target for p in parts]),
    )
    rows = len(joined["target"])
    _require(len(keys) == rows, "Las claves no tienen las filas de las emisiones")
    bounds = np.cumsum([0] + [p.rows for p in parts])
    slices = pilots._session_slices(splits)
    models = list(predictions)
    matrix = np.column_stack([np.asarray(predictions[m], dtype=np.float64) for m in models])
    _require(matrix.shape == (rows, len(models)), "Cada modelo emite una predicción por fila")
    module = pilots.screening()

    def ics(values):
        return {
            name: pilots.session_ic(splits[name], values[bounds[i] : bounds[i + 1]])
            for i, name in enumerate(pilots.SPLITS)
            if name != "validation"
        }

    def mean_ic(series_dict):
        return float(np.mean(list(series_dict.values()))) if series_dict else None

    z_by_model = {m: pilots._zscore_sessions(matrix[:, j], slices) for j, m in enumerate(models)}
    parent_ics = {m: ics(matrix[:, j]) for j, m in enumerate(models)}
    results = {m: {} for m in models}
    arms_by_model = {m: {} for m in models}
    shared_states = {}
    series = []
    declared_arms = [arm_name for arm_name in EPISODIC_ARMS if arm_name in section]

    # --- 1. Banco denso con temperatura ---
    if "dense_bank" in section:
        started = time.monotonic()
        cfg_doc = section["dense_bank"]
        cfg = DenseBankConfig(
            window=cfg_doc["window"],
            k=cfg_doc["k"],
            temperatures=tuple(cfg_doc["temperatures"]),
            clip=cfg_doc["clip"],
        )
        kappas = [float(k) for k in cfg_doc["kappas"] if k > 0]
        dense_mem = DenseBankMemory(cfg, models=len(models), threads=threads)
        d_means, d_disps, d_pmeans, _ = replay_dense_bank(
            dense_mem,
            markets=joined["markets"],
            prediction_at=joined["prediction_at"],
            matures_at=joined["matures_at"],
            keys=keys,
            predictions=matrix,
            target=joined["target"],
            placebo_seed=placebo_seed,
        )
        shared_states["dense_bank"] = dense_mem.export()
        log(f"banco denso ({len(models)} modelos): {time.monotonic() - started:.0f} s")
        for j, model in enumerate(models):
            z = z_by_model[model]
            u_by_tau = {
                tau: pilots._zscore_sessions(d_means[tau][:, j], slices) for tau in cfg.temperatures
            }
            grid = {
                (tau, kappa): ics(z + kappa * u_by_tau[tau])
                for tau in cfg.temperatures
                for kappa in kappas
            }
            best_tau, best_kappa = pilots._best(grid, list(grid))
            real_var = grid[(best_tau, best_kappa)]
            u_fake = pilots._zscore_sessions(d_pmeans[best_tau][:, j], slices)
            placebo_var = ics(z + best_kappa * u_fake)
            selection = dict(
                selected_on="calibration",
                window=cfg.window,
                k=cfg.k,
                tau=best_tau,
                clip=cfg.clip,
                kappa=best_kappa,
            )
            unc = d_disps[best_tau][:, j]
            results[model]["dense_bank"] = dict(
                selection=selection,
                calibration=mean_ic(real_var["calibration"]),
                evaluation=mean_ic(real_var["evaluation"]),
                placebo_calibration=mean_ic(placebo_var["calibration"]),
                placebo_evaluation=mean_ic(placebo_var["evaluation"]),
                uncertainty=_uncertainty_summary(
                    splits, bounds, unc, joined["target"] - matrix[:, j], module
                ),
                contrasts=dict(
                    vs_parent=pilots._contrast(
                        real_var["evaluation"], parent_ics[model]["evaluation"]
                    ),
                    placebo_vs_parent=pilots._contrast(
                        placebo_var["evaluation"], parent_ics[model]["evaluation"]
                    ),
                    vs_placebo=pilots._contrast(real_var["evaluation"], placebo_var["evaluation"]),
                ),
            )
            arms_by_model[model]["dense_bank"] = dict(
                signal=best_kappa * u_by_tau[best_tau],
                selection=selection,
                ic={s: mean_ic(real_var[s]) for s in ("calibration", "evaluation")},
                placebo_ic={s: mean_ic(placebo_var[s]) for s in ("calibration", "evaluation")},
                extra_columns=dict(organ_uncertainty=unc),
                uncertainty_column="organ_uncertainty",
            )
            for label, var in (("dense_bank", real_var), ("dense_bank_placebo", placebo_var)):
                for split_name, vals in var.items():
                    series += [
                        (model, split_name, market, instant, label, val)
                        for (market, instant), val in vals.items()
                    ]

    # --- 2. Memoria holográfica reducida con regla delta ---
    if "holographic_delta" in section:
        started = time.monotonic()
        _require(
            regimes is not None and len(regimes) == rows,
            "holographic_delta necesita el régimen observable de cada fila",
        )
        cfg_doc = section["holographic_delta"]
        cfg = HolographicDeltaConfig(
            dim=cfg_doc["dim"],
            decay=cfg_doc["decay"],
            seed=cfg_doc["seed"],
            entity=cfg_doc["entity"],
            rate=cfg_doc.get("rate", 1.0),
            clip=cfg_doc.get("clip", 0.1),
            dtype=cfg_doc.get("dtype", "float64"),
        )
        kappas = [float(k) for k in cfg_doc["kappas"] if k > 0]
        if cfg.entity == "asset":
            entities = joined["assets"]
        else:
            entities = resolve_sector_entities(
                joined["assets"],
                sector_map=sector_map,
                sector_source=cfg_doc.get("sector_source"),
            )
        holo_mem = HolographicDeltaMemory(cfg, models=len(models), placebo_seed=placebo_seed)
        h_reads, h_placebo = replay_holographic_delta(
            holo_mem,
            markets=joined["markets"],
            prediction_at=joined["prediction_at"],
            matures_at=joined["matures_at"],
            entities=entities,
            regimes=regimes,
            predictions=matrix,
            target=joined["target"],
        )
        shared_states["holographic_delta"] = holo_mem.export()
        log(f"memoria holográfica ({len(models)} modelos): {time.monotonic() - started:.0f} s")
        for j, model in enumerate(models):
            z = z_by_model[model]
            u = pilots._zscore_sessions(h_reads[:, j], slices)
            u_fake = pilots._zscore_sessions(h_placebo[:, j], slices)
            grid = {kappa: ics(z + kappa * u) for kappa in kappas}
            best_kappa = pilots._best(grid, list(grid))
            real_var = grid[best_kappa]
            placebo_var = ics(z + best_kappa * u_fake)
            selection = dict(
                selected_on="calibration",
                dim=cfg.dim,
                decay=cfg.decay,
                rate=cfg.rate,
                clip=cfg.clip,
                entity=cfg.entity,
                seed=cfg.seed,
                dtype=cfg.dtype,
                regime_rule=REGIME_RULE,
                kappa=best_kappa,
            )
            results[model]["holographic_delta"] = dict(
                selection=selection,
                calibration=mean_ic(real_var["calibration"]),
                evaluation=mean_ic(real_var["evaluation"]),
                placebo_calibration=mean_ic(placebo_var["calibration"]),
                placebo_evaluation=mean_ic(placebo_var["evaluation"]),
                contrasts=dict(
                    vs_parent=pilots._contrast(
                        real_var["evaluation"], parent_ics[model]["evaluation"]
                    ),
                    placebo_vs_parent=pilots._contrast(
                        placebo_var["evaluation"], parent_ics[model]["evaluation"]
                    ),
                    vs_placebo=pilots._contrast(real_var["evaluation"], placebo_var["evaluation"]),
                ),
            )
            arms_by_model[model]["holographic_delta"] = dict(
                signal=best_kappa * u,
                selection=selection,
                ic={s: mean_ic(real_var[s]) for s in ("calibration", "evaluation")},
                placebo_ic={s: mean_ic(placebo_var[s]) for s in ("calibration", "evaluation")},
            )
            for label, var in (
                ("holographic_delta", real_var),
                ("holographic_delta_placebo", placebo_var),
            ):
                for split_name, vals in var.items():
                    series += [
                        (model, split_name, market, instant, label, val)
                        for (market, instant), val in vals.items()
                    ]

    # --- 3. Traza exponencial por activo ---
    if "asset_trace" in section:
        started = time.monotonic()
        cfg_doc = section["asset_trace"]
        half_lives = [float(h) for h in cfg_doc["half_lives"]]
        clip = cfg_doc.get("clip")
        kappas = [float(k) for k in cfg_doc["kappas"] if k > 0]
        trace_reads, trace_placebo = {}, {}
        for half_life in half_lives:
            mem = AssetResidualTrace(AssetTraceConfig(half_life=half_life, clip=clip), len(models))
            trace_reads[half_life], trace_placebo[half_life] = replay_asset_trace(
                mem,
                markets=joined["markets"],
                assets=joined["assets"],
                prediction_at=joined["prediction_at"],
                matures_at=joined["matures_at"],
                predictions=matrix,
                target=joined["target"],
                placebo_seed=placebo_seed,
            )
        log(f"traza por activo ({len(models)} modelos): {time.monotonic() - started:.0f} s")
        for j, model in enumerate(models):
            z = z_by_model[model]
            u_by_h = {h: pilots._zscore_sessions(trace_reads[h][:, j], slices) for h in half_lives}
            grid = {(h, kappa): ics(z + kappa * u_by_h[h]) for h in half_lives for kappa in kappas}
            best_h, best_kappa = pilots._best(grid, list(grid))
            real_var = grid[(best_h, best_kappa)]
            u_fake = pilots._zscore_sessions(trace_placebo[best_h][:, j], slices)
            placebo_var = ics(z + best_kappa * u_fake)
            # Estado individual de 1 modelo para el brazo elegido (idéntico solo o con otros).
            single_mem = AssetResidualTrace(AssetTraceConfig(half_life=best_h, clip=clip), 1)
            replay_asset_trace(
                single_mem,
                markets=joined["markets"],
                assets=joined["assets"],
                prediction_at=joined["prediction_at"],
                matures_at=joined["matures_at"],
                predictions=matrix[:, [j]],
                target=joined["target"],
            )
            selection = dict(
                selected_on="calibration",
                half_life=best_h,
                clip=None if clip is None else float(clip),
                kappa=best_kappa,
            )
            results[model]["asset_trace"] = dict(
                selection=selection,
                calibration=mean_ic(real_var["calibration"]),
                evaluation=mean_ic(real_var["evaluation"]),
                placebo_calibration=mean_ic(placebo_var["calibration"]),
                placebo_evaluation=mean_ic(placebo_var["evaluation"]),
                contrasts=dict(
                    vs_parent=pilots._contrast(
                        real_var["evaluation"], parent_ics[model]["evaluation"]
                    ),
                    placebo_vs_parent=pilots._contrast(
                        placebo_var["evaluation"], parent_ics[model]["evaluation"]
                    ),
                    vs_placebo=pilots._contrast(real_var["evaluation"], placebo_var["evaluation"]),
                ),
            )
            arms_by_model[model]["asset_trace"] = dict(
                signal=best_kappa * u_by_h[best_h],
                selection=selection,
                ic={s: mean_ic(real_var[s]) for s in ("calibration", "evaluation")},
                placebo_ic={s: mean_ic(placebo_var[s]) for s in ("calibration", "evaluation")},
                state=single_mem.export(),
            )
            for label, var in (("asset_trace", real_var), ("asset_trace_placebo", placebo_var)):
                for split_name, vals in var.items():
                    series += [
                        (model, split_name, market, instant, label, val)
                        for (market, instant), val in vals.items()
                    ]

    model_, split_c, market_c, instant_c, variant_c, value_c = zip(*series, strict=True)
    table = pa.table(
        dict(
            model=pa.array(model_, type=pa.string()),
            split=pa.array(split_c, type=pa.string()),
            market=pa.array(market_c, type=pa.string()),
            instant=pa.array(instant_c, type=pa.int64()),
            variant=pa.array(variant_c, type=pa.string()),
            value=pa.array(value_c, type=pa.float64()),
        )
    )
    exported = dict(
        arms=declared_arms,
        models={
            model: dict(
                dispersion=pilots._session_dispersion(matrix[:, j], slices),
                arms=arms_by_model[model],
            )
            for j, model in enumerate(models)
        },
        order=models,
        shared_states=shared_states,
    )
    return results, table, exported
