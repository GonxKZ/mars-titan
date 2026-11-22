#!/usr/bin/env python3
"""Inferencia estadística robusta para series diarias del coeficiente de información (IC).

Implementa estimación de varianza HAC de Newey-West (1987), selección automática de longitud
de bloque de Politis y White (2004) con la corrección de Patton, Politis y White (2009),
bootstrap estacionario de Politis y Romano (1994), contraste de Diebold y Mariano (1995),
contraste de capacidad predictiva superior (SPA) de Hansen (2005) y Model Confidence Set
(MCS) de Hansen, Lunde y Nason (2011).

El módulo emplea únicamente la biblioteca estándar de Python y NumPy, garantiza resultados
deterministas dada una semilla y remuestrea siempre los mismos índices temporales para todos
los modelos a la vez para preservar la correlación contemporánea.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_CHUNK_BOOTSTRAP = 512
_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=UTC)
_NS_PER_SECOND = 1_000_000_000
_INT_RE = re.compile(r"^[+-]?\d+$")
_FRAC_SEC_RE = re.compile(r"[Tt ]\d{2}:\d{2}:\d{2}[.,](\d+)")


def automatic_newey_west_lags(n_obs: int) -> int:
    """Regla automática de Newey y West (1994): ``floor(4 * (T / 100) ** (2 / 9))``."""
    return min(int(math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))), n_obs - 1)


def _parse_instant_ns(instant: str) -> int:
    """Convierte un instante textual (ISO 8601 o época en s/ms/us/ns) a nanosegundos UTC."""
    text = instant.strip()
    if not text:
        raise ValueError("El instante no puede estar vacío.")
    if _INT_RE.fullmatch(text):
        raw = int(text)
        magnitude = abs(raw)
        if magnitude >= 10**17:
            return raw
        if magnitude >= 10**14:
            return raw * 1_000
        if magnitude >= 10**11:
            return raw * 1_000_000
        if magnitude >= 10**8:
            return raw * _NS_PER_SECOND
        return raw
    dt = datetime.fromisoformat(text)
    dt_utc = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    extra_ns = 0
    frac_match = _FRAC_SEC_RE.search(text)
    if frac_match is not None:
        sub_micro = frac_match.group(1)[6:9]
        if sub_micro:
            extra_ns = int(sub_micro.ljust(3, "0"))
    delta = dt_utc - _EPOCH_UTC
    total_seconds = delta.days * 86_400 + delta.seconds
    return total_seconds * _NS_PER_SECOND + delta.microseconds * 1_000 + extra_ns


def _parse_session_stratum(session: str) -> tuple[int, str, int] | None:
    """Extrae ``(origin_year, market, instant_ns)`` si ``session`` sigue ``year:market:instant``."""
    parts = session.split(":", 2)
    if len(parts) != 3:
        return None
    year_str, market, instant_str = parts
    if not year_str.isdigit() or not market.strip():
        return None
    try:
        instant_ns = _parse_instant_ns(instant_str)
    except ValueError:
        return None
    return int(year_str), market.strip(), instant_ns


def _partition_blocks(n_obs: int, blocks: Sequence[Any] | None) -> list[np.ndarray]:
    """Agrupa los índices ``0..n_obs-1`` por estrato preservando el orden dentro de cada bloque."""
    if blocks is None:
        return [np.arange(n_obs, dtype=np.int64)]
    if isinstance(blocks, (str, bytes)) or len(blocks) != n_obs:
        raise ValueError(f"blocks debe tener longitud {n_obs}.")
    grouped: dict[Any, list[int]] = {}
    for idx, blk in enumerate(blocks):
        grouped.setdefault(blk, []).append(idx)
    return [np.asarray(idxs, dtype=np.int64) for idxs in grouped.values()]


def _validar_entero(valor: Any, nombre: str, minimo: int) -> int:
    """Valida que un parámetro sea un entero (no booleano) mayor o igual que ``minimo``."""
    if isinstance(valor, bool) or not isinstance(valor, (int, np.integer)):
        raise ValueError(f"{nombre} debe ser un entero; se recibió {type(valor).__name__}.")
    entero = int(valor)
    if entero < minimo:
        raise ValueError(f"{nombre} debe ser >= {minimo}; se recibió {entero}.")
    return entero


def _validar_flotante_finito(valor: Any, nombre: str) -> float:
    """Valida que un parámetro sea un número real finito."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float, np.number)):
        raise ValueError(f"{nombre} debe ser numérico; se recibió {type(valor).__name__}.")
    flotante = float(valor)
    if not math.isfinite(flotante):
        raise ValueError(f"{nombre} debe ser finito; se recibió {flotante}.")
    return flotante


def _validar_serie_1d(x: Any, nombre: str = "x", min_len: int = 2) -> np.ndarray:
    """Convierte y valida una serie unidimensional finita con longitud mínima."""
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{nombre} no se pudo convertir a arreglo numérico 1D.") from exc
    if arr.ndim != 1:
        raise ValueError(f"{nombre} debe ser un arreglo 1D; se recibió ndim={arr.ndim}.")
    if arr.size < min_len:
        raise ValueError(
            f"{nombre} debe tener al menos {min_len} observaciones; se recibieron {arr.size}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{nombre} contiene valores no finitos (NaN o infinito).")
    return arr


def _validar_matriz_2d(
    x: Any,
    nombre: str = "x",
    min_t: int = 2,
    min_m: int = 1,
) -> np.ndarray:
    """Convierte y valida una matriz 2D [T, M] finita con dimensiones mínimas."""
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{nombre} no se pudo convertir a matriz numérica 2D.") from exc
    if arr.ndim != 2:
        raise ValueError(f"{nombre} debe ser una matriz 2D [T, M]; se recibió ndim={arr.ndim}.")
    n_obs, n_mod = arr.shape
    if n_obs < min_t:
        raise ValueError(
            f"{nombre} debe tener al menos {min_t} filas temporales; se recibieron {n_obs}."
        )
    if n_mod < min_m:
        raise ValueError(
            f"{nombre} debe tener al menos {min_m} columnas de modelos; se recibieron {n_mod}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{nombre} contiene valores no finitos (NaN o infinito).")
    return arr


def _validar_bloque(block: Any) -> float:
    """Valida que la longitud media de bloque sea finita y mayor o igual que 1."""
    valor = _validar_flotante_finito(block, "block")
    if valor < 1.0:
        raise ValueError(f"block debe ser >= 1.0; se recibió {valor}.")
    return valor


def _es_constante(arr: np.ndarray) -> bool:
    """Comprueba si todos los elementos de un vector son exactamente iguales."""
    return bool(np.all(arr == arr[0]))


def _medias_bootstrap(datos: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Calcula medias bootstrap por bloques de memoria usando los mismos índices temporales.

    Parámetros
    ----------
    datos : np.ndarray
        Arreglo 1D de forma ``[T]`` o 2D de forma ``[T, M]``.
    indices : np.ndarray
        Matriz entera de índices de forma ``[B, T]``.

    Devuelve
    --------
    np.ndarray
        Medias remuestreadas de forma ``[B]`` o ``[B, M]``.
    """
    n_boot = indices.shape[0]
    forma_salida = (n_boot,) if datos.ndim == 1 else (n_boot, datos.shape[1])
    salida = np.empty(forma_salida, dtype=np.float64)
    for inicio in range(0, n_boot, _CHUNK_BOOTSTRAP):
        fin = min(inicio + _CHUNK_BOOTSTRAP, n_boot)
        salida[inicio:fin] = datos[indices[inicio:fin]].mean(axis=1)
    return salida


def _longitud_bloque_multivariante(matriz: np.ndarray) -> float:
    """Calcula una longitud de bloque común para una matriz ``[T, M]``.

    Toma el máximo de las longitudes óptimas univariantes por columna para respetar
    la persistencia de la serie más autocorrelacionada al remuestrear todos los modelos
    simultáneamente con los mismos índices temporales.
    """
    bloques = [optimal_block_length(matriz[:, col]) for col in range(matriz.shape[1])]
    return float(max(bloques))


def _t_simple(x: np.ndarray) -> float:
    """Calcula el estadístico t clásico de la media (desviación típica con ddof=1)."""
    media = float(np.mean(x))
    if _es_constante(x):
        if media == 0.0:
            return 0.0
        return math.copysign(math.inf, media)
    desv = float(np.std(x, ddof=1))
    if desv <= 0.0:
        if media == 0.0:
            return 0.0
        return math.copysign(math.inf, media)
    return float(media / (desv / math.sqrt(x.size)))


def newey_west_hac(
    x: Sequence[float] | np.ndarray,
    lags: int | None = None,
    *,
    blocks: Sequence[Any] | None = None,
) -> tuple[float, float, int]:
    """Calcula ``(t_hac, se_hac, max_lags)`` de la media con varianza HAC de Newey-West.

    Si se pasa ``blocks`` (p. ej. ``(origin_year, market)`` por observación), las
    autocovarianzas de rezago ``k >= 1`` se suman estrictamente dentro de cada bloque
    contiguo sin cruzar fronteras entre orígenes no contiguos ni entre mercados.
    """
    arr = _validar_serie_1d(x, nombre="x", min_len=2)
    n_obs = arr.size

    rezagos_fijos: int | None
    if lags is None:
        rezagos_fijos = None
    else:
        rezagos_fijos = _validar_entero(lags, "lags", minimo=0)
        if rezagos_fijos >= n_obs:
            raise ValueError(f"lags debe ser menor que T ({n_obs}); se recibió {rezagos_fijos}.")

    media = float(np.mean(arr))
    if _es_constante(arr):
        if media == 0.0:
            return 0.0, 0.0, 0
        return math.copysign(math.inf, media), 0.0, 0

    centrada = arr - media
    particiones = _partition_blocks(n_obs, blocks)
    suma_varianzas = 0.0
    max_rezagos = 0
    for idxs in particiones:
        u = centrada[idxs]
        tb = u.size
        if tb == 0:
            continue
        if rezagos_fijos is None:
            l_b = automatic_newey_west_lags(tb) if tb > 1 else 0
        else:
            l_b = min(rezagos_fijos, tb - 1)
        max_rezagos = max(max_rezagos, l_b)
        v_b = float(np.dot(u, u))
        for k in range(1, l_b + 1):
            peso = 1.0 - k / (l_b + 1.0)
            v_b += 2.0 * peso * float(np.dot(u[k:], u[:-k]))
        suma_varianzas += max(v_b, 0.0)

    lrv = suma_varianzas / n_obs
    if lrv <= 0.0:
        if media == 0.0:
            return 0.0, 0.0, max_rezagos
        return math.copysign(math.inf, media), 0.0, max_rezagos

    error_estandar = math.sqrt(lrv / n_obs)
    return float(media / error_estandar), float(error_estandar), int(max_rezagos)


def newey_west_t(
    x: Sequence[float] | np.ndarray,
    lags: int | None = None,
    *,
    blocks: Sequence[Any] | None = None,
) -> float:
    """Calcula el estadístico t de la media con varianza HAC de Newey-West.

    Emplea el núcleo de Bartlett:
        omega_hat = gamma_0 + 2 * sum_{k=1}^{L} (1 - k / (L + 1)) * gamma_k,
    donde ``gamma_k = (1 / T) * sum_{t=k}^{T-1} (x_t - x_bar) * (x_{t-k} - x_bar)``.
    Si ``lags`` es ``None``, aplica la regla de Newey y West (1994):
    ``L = floor(4 * (T / 100) ** (2 / 9))``.
    Si se proporciona ``blocks``, las autocovarianzas se calculan por bloque sin
    cruzar fronteras entre estratos.
    """
    t_stat, _, _ = newey_west_hac(x, lags=lags, blocks=blocks)
    return t_stat


def optimal_block_length(x: Sequence[float] | np.ndarray) -> float:
    """Calcula la longitud media de bloque óptima para el bootstrap estacionario.

    Implementa el selector automático de Politis y White (2004) con la corrección de
    Patton, Politis y White (2009) para el bootstrap estacionario:
        b_opt = ((2 * G_hat^2) / D_SB_hat)^(1/3) * T^(1/3),
    con ``D_SB_hat = 2 * g_hat(0)^2``, de modo que
    ``b_opt = (G_hat^2 / g_hat(0)^2)^(1/3) * T^(1/3)``.
    Los estimadores espectrales ``g_hat(0)`` y ``G_hat`` usan la ventana plana trapezoidal
    (flat-top) de Politis y Romano (1995) con ancho de banda automático ``M = 2 * m_hat``.

    Parámetros
    ----------
    x : array_like
        Serie temporal 1D de longitud ``T >= 4``.

    Devuelve
    --------
    float
        Longitud media de bloque óptima ``>= 1.0``, acotada superiormente por
        ``min(3 * sqrt(T), T / 3)`` (al menos 1.0) según Patton, Politis y White (2009).
    """
    arr = _validar_serie_1d(x, nombre="x", min_len=4)
    n_obs = arr.size
    if _es_constante(arr):
        return 1.0

    centrada = arr - float(np.mean(arr))
    r0 = float(np.dot(centrada, centrada) / n_obs)
    if r0 <= 0.0:
        return 1.0

    # Parámetros del selector automático de ancho de banda (Politis 2003; PPW 2009).
    const_c = 2.0
    k_n = max(5, int(math.ceil(math.sqrt(math.log10(n_obs)))))
    m_cap = int(math.ceil(math.sqrt(n_obs)))
    max_rezago = min(max(2 * m_cap, m_cap + k_n), n_obs - 1)

    # Autocovarianzas muestrales R(k) y autocorrelaciones rho(k) para k = 0..max_rezago.
    autocov = np.empty(max_rezago + 1, dtype=np.float64)
    autocov[0] = r0
    for k in range(1, max_rezago + 1):
        autocov[k] = float(np.dot(centrada[k:], centrada[:-k]) / n_obs)
    autocorr = autocov / r0

    umbral = const_c * math.sqrt(math.log10(n_obs) / n_obs)
    insig = np.abs(autocorr[1:]) < umbral  # insig[j] corresponde al rezago j + 1

    # Búsqueda del menor rezago con k_eff autocorrelaciones consecutivas no significativas.
    k_eff = min(k_n, max_rezago)
    limite_busqueda = min(m_cap, max_rezago - k_eff + 1)
    m_hat: int | None = None
    for rezago in range(1, limite_busqueda + 1):
        ventana = insig[rezago - 1 : rezago - 1 + k_eff]
        if bool(np.all(ventana)):
            m_hat = 1 if rezago == 1 else rezago - 1
            break

    if m_hat is None:
        sig_indices = np.flatnonzero(~insig[: min(m_cap, max_rezago)])
        m_hat = int(sig_indices[-1] + 1) if sig_indices.size > 0 else 1

    ancho_m = min(2 * m_hat, max_rezago)

    # Ventana trapezoidal (flat-top) de Politis y Romano (1995).
    rezagos = np.arange(1, ancho_m + 1, dtype=np.float64)
    fraccion = rezagos / float(ancho_m)
    pesos = np.where(fraccion <= 0.5, 1.0, 2.0 * (1.0 - fraccion))
    r_pos = autocov[1 : ancho_m + 1]

    g0_hat = r0 + 2.0 * float(np.dot(pesos, r_pos))
    g_deriv_hat = 2.0 * float(np.dot(pesos * rezagos, r_pos))

    # Respaldo con núcleo de Bartlett (estrictamente positivo) si la ventana plana
    # produce una densidad espectral en cero no positiva en muestra finita.
    if g0_hat <= 1e-12 * r0:
        pesos_bartlett = 1.0 - rezagos / (ancho_m + 1.0)
        g0_hat = r0 + 2.0 * float(np.dot(pesos_bartlett, r_pos))
        if g0_hat <= 0.0:
            return 1.0

    # Corrección de Patton, Politis y White (2009): D_SB = 2 * g(0)^2.
    d_sb = 2.0 * (g0_hat**2)
    b_opt = ((2.0 * (g_deriv_hat**2)) / d_sb) ** (1.0 / 3.0) * (n_obs ** (1.0 / 3.0))

    b_max = max(1.0, min(3.0 * math.sqrt(n_obs), n_obs / 3.0))
    return float(max(1.0, min(b_opt, b_max)))


def stationary_bootstrap_indices(
    T: int,
    block: float,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera índices ``[B, T]`` del bootstrap estacionario de Politis y Romano (1994).

    En cada réplica, el primer índice se sortea uniformemente en ``0..T-1``. En cada paso
    posterior, con probabilidad ``p = 1 / block`` se reinicia en un índice uniforme en
    ``0..T-1``, y con probabilidad ``1 - p`` se continúa en el índice siguiente con
    envoltura circular ``(indice_anterior + 1) % T``.

    Parámetros
    ----------
    T : int
        Longitud de la serie temporal (``T >= 1``).
    block : float
        Longitud media esperada de bloque (``block >= 1.0``).
    B : int
        Número de réplicas bootstrap (``B >= 1``).
    rng : np.random.Generator
        Generador pseudoaleatorio de NumPy.

    Devuelve
    --------
    np.ndarray
        Matriz entera (``int64``) de forma ``[B, T]`` con valores en ``[0, T - 1]``.
    """
    n_obs = _validar_entero(T, "T", minimo=1)
    n_boot = _validar_entero(B, "B", minimo=1)
    bloque = _validar_bloque(block)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng debe ser una instancia de numpy.random.Generator.")

    inicios = rng.integers(0, n_obs, size=(n_boot, n_obs), dtype=np.int64)
    if bloque == 1.0 or n_obs == 1:
        return inicios

    prob_reinicio = 1.0 / bloque
    reinicios = rng.random(size=(n_boot, n_obs)) < prob_reinicio
    reinicios[:, 0] = True

    posiciones = np.arange(n_obs, dtype=np.int64)
    ultimo_reinicio = np.maximum.accumulate(np.where(reinicios, posiciones, 0), axis=1)
    desplazamiento = posiciones - ultimo_reinicio
    inicio_bloque = np.take_along_axis(inicios, ultimo_reinicio, axis=1)
    return (inicio_bloque + desplazamiento) % n_obs


def stratified_stationary_bootstrap_indices(
    blocks: Sequence[Any],
    block_lengths: float | Sequence[float],
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera índices ``[B, T]`` de bootstrap estacionario estratificado por bloque.

    Remuestrea independientemente cada estrato ``(origin_year, market)`` con envoltura
    circular estrictamente interna, de modo que nunca se enlazan orígenes no contiguos
    ni mercados distintos.
    """
    n_obs = len(blocks)
    if n_obs < 1:
        raise ValueError("blocks no puede estar vacío.")
    n_boot = _validar_entero(B, "B", minimo=1)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng debe ser una instancia de numpy.random.Generator.")

    particiones = _partition_blocks(n_obs, blocks)
    if isinstance(block_lengths, (int, float, np.number)) and not isinstance(block_lengths, bool):
        longitudes = [_validar_bloque(block_lengths)] * len(particiones)
    else:
        longitudes = [_validar_bloque(b) for b in block_lengths]
        if len(longitudes) != len(particiones):
            raise ValueError("block_lengths debe coincidir con el número de estratos.")

    partes: list[np.ndarray] = []
    for idxs, longitud in zip(particiones, longitudes, strict=True):
        tb = idxs.size
        if tb == 1:
            partes.append(np.full((n_boot, 1), int(idxs[0]), dtype=np.int64))
        else:
            locales = stationary_bootstrap_indices(tb, longitud, n_boot, rng)
            partes.append(idxs[locales])
    return np.concatenate(partes, axis=1)


def _resolver_bloques_bootstrap(
    datos: np.ndarray,
    blocks: Sequence[Any] | None,
    block: float | None,
) -> tuple[list[np.ndarray], list[float], float]:
    """Calcula particiones, longitudes por estrato y bloque medio ponderado."""
    n_obs = datos.shape[0]
    particiones = _partition_blocks(n_obs, blocks)
    longitudes: list[float] = []
    for idxs in particiones:
        if block is not None:
            longitudes.append(_validar_bloque(block))
        elif idxs.size < 4:
            longitudes.append(1.0)
        elif datos.ndim == 1:
            longitudes.append(optimal_block_length(datos[idxs]))
        else:
            longitudes.append(_longitud_bloque_multivariante(datos[idxs]))
    bloque_medio = float(
        sum(idxs.size * longitud for idxs, longitud in zip(particiones, longitudes, strict=True))
        / n_obs
    )
    return particiones, longitudes, bloque_medio


def bootstrap_mean_interval(
    x: Sequence[float] | np.ndarray,
    level: float = 0.95,
    B: int = 5000,
    block: float | None = None,
    seed: int = 0,
    *,
    blocks: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Calcula el intervalo percentil de la media mediante bootstrap estacionario."""
    min_len = 4 if (block is None and blocks is None) else 2
    arr = _validar_serie_1d(x, nombre="x", min_len=min_len)
    nivel = _validar_flotante_finito(level, "level")
    if not (0.0 < nivel < 1.0):
        raise ValueError(f"level debe estar en (0, 1); se recibió {nivel}.")
    n_boot = _validar_entero(B, "B", minimo=1)
    semilla = _validar_entero(seed, "seed", minimo=0)

    rng = np.random.default_rng(semilla)
    if blocks is None:
        bloque = optimal_block_length(arr) if block is None else _validar_bloque(block)
        indices = stationary_bootstrap_indices(arr.size, bloque, n_boot, rng)
    else:
        _, longitudes, bloque = _resolver_bloques_bootstrap(arr, blocks, block)
        indices = stratified_stationary_bootstrap_indices(blocks, longitudes, n_boot, rng)
    medias_boot = _medias_bootstrap(arr, indices)

    alfa = 1.0 - nivel
    inferior, superior = np.quantile(medias_boot, [alfa / 2.0, 1.0 - alfa / 2.0])
    return {
        "mean": float(np.mean(arr)),
        "low": float(inferior),
        "high": float(superior),
        "block": float(bloque),
    }


def diebold_mariano(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    lags: int | None = None,
    *,
    blocks: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Contraste de Diebold y Mariano (1995) de igualdad de medias pareadas con HAC."""
    serie_a = _validar_serie_1d(a, nombre="a", min_len=2)
    serie_b = _validar_serie_1d(b, nombre="b", min_len=2)
    if serie_a.size != serie_b.size:
        raise ValueError(
            f"a y b deben tener la misma longitud; se recibieron {serie_a.size} y {serie_b.size}."
        )

    diferencia = serie_a - serie_b
    media = float(np.mean(diferencia))
    estadistico_t = newey_west_t(diferencia, lags=lags, blocks=blocks)
    if math.isinf(estadistico_t):
        p_valor = 0.0
    else:
        p_valor = float(math.erfc(abs(estadistico_t) / math.sqrt(2.0)))

    return {
        "mean": media,
        "t": float(estadistico_t),
        "p_value": p_valor,
    }


def _varianza_bootstrap_estacionario(diferencias: np.ndarray, bloque: float) -> np.ndarray:
    """Calcula la varianza exacta del bootstrap estacionario (Hansen 2005, ec. 6).

    Para cada columna ``k`` de ``diferencias`` ``[T, M]``, evalúa:
        omega_k^2 = gamma_0 + 2 * sum_{i=1}^{T-1} ((T - i)/T * (1 - q)^i
                    + (i/T) * (1 - q)^(T - i)) * gamma_i,
    con ``q = 1 / bloque``, de forma exacta en ``O(T log T)`` mediante autocovarianza
    circular por FFT.
    """
    n_obs = diferencias.shape[0]
    centrada = diferencias - np.mean(diferencias, axis=0, keepdims=True)
    if bloque == 1.0:
        return np.mean(centrada * centrada, axis=0)

    q = 1.0 / bloque
    fft_u = np.fft.rfft(centrada, axis=0)
    rcirc = np.fft.irfft(fft_u * np.conj(fft_u), n=n_obs, axis=0) / float(n_obs)

    rezagos = np.arange(1, n_obs, dtype=np.float64)
    pesos = (1.0 - rezagos / float(n_obs)) * ((1.0 - q) ** rezagos)
    omega2 = rcirc[0] + 2.0 * (pesos @ rcirc[1:])
    return np.maximum(omega2, 0.0)


def _varianza_bootstrap_estacionario_estratificada(
    diferencias: np.ndarray,
    particiones: list[np.ndarray],
    longitudes: list[float],
) -> np.ndarray:
    """Varianza exacta del bootstrap estacionario estratificado por bloques independientes."""
    n_obs = diferencias.shape[0]
    omega2 = np.zeros(diferencias.shape[1], dtype=np.float64)
    for idxs, longitud in zip(particiones, longitudes, strict=True):
        tb = idxs.size
        if tb <= 1:
            continue
        omega2 += (tb / float(n_obs)) * _varianza_bootstrap_estacionario(
            diferencias[idxs], longitud
        )
    return np.maximum(omega2, 0.0)


def spa_test(
    candidates: Sequence[Sequence[float]] | np.ndarray,
    benchmark: Sequence[float] | np.ndarray,
    B: int = 5000,
    block: float | None = None,
    seed: int = 0,
    *,
    blocks: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Contraste SPA (Superior Predictive Ability) de Hansen (2005).

    Soporta remuestreo estratificado opcional por ``blocks`` (p. ej. ``(origin_year, market)``).
    """
    min_t = 4 if (block is None and blocks is None) else 2
    cand = _validar_matriz_2d(candidates, nombre="candidates", min_t=min_t, min_m=1)
    ref = _validar_serie_1d(benchmark, nombre="benchmark", min_len=min_t)
    if cand.shape[0] != ref.size:
        raise ValueError(
            "candidates y benchmark deben tener el mismo número de sesiones; "
            f"se recibieron {cand.shape[0]} y {ref.size}."
        )
    n_boot = _validar_entero(B, "B", minimo=1)
    semilla = _validar_entero(seed, "seed", minimo=0)

    n_obs, n_mod = cand.shape
    diferencias = cand - ref[:, np.newaxis]
    medias_d = np.mean(diferencias, axis=0)
    mejor_candidato = int(np.argmax(np.mean(cand, axis=0)))

    rng = np.random.default_rng(semilla)
    if blocks is None:
        bloque = (
            _longitud_bloque_multivariante(diferencias) if block is None else _validar_bloque(block)
        )
        omega2 = _varianza_bootstrap_estacionario(diferencias, bloque)
        indices = stationary_bootstrap_indices(n_obs, bloque, n_boot, rng)
    else:
        particiones, longitudes, _ = _resolver_bloques_bootstrap(diferencias, blocks, block)
        omega2 = _varianza_bootstrap_estacionario_estratificada(
            diferencias, particiones, longitudes
        )
        indices = stratified_stationary_bootstrap_indices(blocks, longitudes, n_boot, rng)

    # Detección segura de columnas constantes (varianza nula).
    escala_col = np.maximum(1.0, np.mean(diferencias * diferencias, axis=0))
    constantes = np.array([_es_constante(diferencias[:, k]) for k in range(n_mod)], dtype=bool)
    var_nula = constantes | (omega2 <= 1e-24 * escala_col)
    omega = np.where(var_nula, 1.0, np.sqrt(np.maximum(omega2, 0.0)))

    raiz_t = math.sqrt(n_obs)
    t_obs_col = np.empty(n_mod, dtype=np.float64)
    for k in range(n_mod):
        if var_nula[k]:
            if medias_d[k] > 0.0:
                t_obs_col[k] = math.inf
            elif medias_d[k] < 0.0:
                t_obs_col[k] = -math.inf
            else:
                t_obs_col[k] = 0.0
        else:
            t_obs_col[k] = raiz_t * medias_d[k] / omega[k]

    estadistico = float(max(float(np.max(t_obs_col)), 0.0))

    # Remuestreo simultáneo de todos los candidatos con los mismos índices temporales.
    medias_boot = _medias_bootstrap(diferencias, indices)  # [B, M]
    centradas_boot = medias_boot - medias_d  # [B, M]

    # Umbral de ley del logaritmo iterado de Hansen (2005, ec. 8).
    umbral_lil = math.sqrt(max(2.0 * math.log(math.log(max(n_obs, 3))), 0.0))
    mu_lower = np.minimum(medias_d, 0.0)
    mu_consistent = np.where(t_obs_col <= -umbral_lil, medias_d, 0.0)
    mu_upper = np.zeros(n_mod, dtype=np.float64)

    def _calcular_p_valor(desplazamiento: np.ndarray) -> float:
        if estadistico == 0.0:
            return 1.0
        z_boot = centradas_boot + desplazamiento
        t_boot = np.empty_like(z_boot)
        activas = ~var_nula
        if np.any(activas):
            t_boot[:, activas] = raiz_t * z_boot[:, activas] / omega[activas]
        for k in np.flatnonzero(var_nula):
            # En columnas constantes, centradas_boot[:, k] == 0; solo cuenta desplazamiento[k].
            if desplazamiento[k] < 0.0:
                t_boot[:, k] = -math.inf
            else:
                t_boot[:, k] = 0.0
        t_spa_boot = np.maximum(np.max(t_boot, axis=1), 0.0)
        return float(np.mean(t_spa_boot >= estadistico))

    return {
        "statistic": estadistico,
        "p_lower": _calcular_p_valor(mu_lower),
        "p_consistent": _calcular_p_valor(mu_consistent),
        "p_upper": _calcular_p_valor(mu_upper),
        "best_candidate": mejor_candidato,
    }


def model_confidence_set(
    scores: Sequence[Sequence[float]] | np.ndarray,
    alpha: float = 0.10,
    B: int = 5000,
    block: float | None = None,
    seed: int = 0,
    *,
    blocks: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Calcula el Model Confidence Set (MCS) de Hansen, Lunde y Nason (2011).

    Soporta remuestreo estratificado opcional por ``blocks`` (p. ej. ``(origin_year, market)``).
    """
    min_t = 4 if (block is None and blocks is None) else 2
    puntuaciones = _validar_matriz_2d(scores, nombre="scores", min_t=min_t, min_m=1)
    nivel_alfa = _validar_flotante_finito(alpha, "alpha")
    if not (0.0 < nivel_alfa < 1.0):
        raise ValueError(f"alpha debe estar en (0, 1); se recibió {nivel_alfa}.")
    n_boot = _validar_entero(B, "B", minimo=1)
    semilla = _validar_entero(seed, "seed", minimo=0)

    n_obs, n_mod = puntuaciones.shape
    if n_mod == 1:
        return {"included": [0], "order": [0], "p_values": [1.0]}

    perdidas = -puntuaciones
    perdidas_rel = perdidas - np.mean(perdidas, axis=1, keepdims=True)
    rng = np.random.default_rng(semilla)
    if blocks is None:
        bloque = (
            _longitud_bloque_multivariante(perdidas_rel)
            if block is None
            else _validar_bloque(block)
        )
        indices = stationary_bootstrap_indices(n_obs, bloque, n_boot, rng)
    else:
        _, longitudes, _ = _resolver_bloques_bootstrap(perdidas_rel, blocks, block)
        indices = stratified_stationary_bootstrap_indices(blocks, longitudes, n_boot, rng)

    medias_perd = np.mean(perdidas, axis=0)  # [M]
    medias_boot = _medias_bootstrap(perdidas, indices)  # [B, M]

    activos = list(range(n_mod))
    orden: list[int] = []
    p_pasos: list[float] = []

    while len(activos) > 1:
        l_bar = medias_perd[activos]
        l_boot = medias_boot[:, activos]

        d_bar = l_bar - float(np.mean(l_bar))
        d_boot = l_boot - np.mean(l_boot, axis=1, keepdims=True)
        d_boot_cent = d_boot - d_bar

        var_d = np.mean(d_boot_cent * d_boot_cent, axis=0)
        escala_cuad = max(1.0, float(np.mean(perdidas[:, activos] ** 2)))
        var_nula = var_d <= 1e-24 * escala_cuad
        media_nula = np.abs(d_bar) <= 1e-12 * math.sqrt(escala_cuad)

        n_activos = len(activos)
        t_obs = np.empty(n_activos, dtype=np.float64)
        t_boot = np.empty((n_boot, n_activos), dtype=np.float64)

        con_var = ~var_nula
        if np.any(con_var):
            desv = np.sqrt(var_d[con_var])
            t_obs[con_var] = d_bar[con_var] / desv
            t_boot[:, con_var] = d_boot_cent[:, con_var] / desv

        for j in np.flatnonzero(var_nula):
            t_boot[:, j] = 0.0
            if media_nula[j]:
                t_obs[j] = 0.0
            elif d_bar[j] > 0.0:
                t_obs[j] = math.inf
            else:
                t_obs[j] = -math.inf

        t_max_obs = float(np.max(t_obs))
        t_max_boot = np.max(t_boot, axis=1)
        p_paso = float(np.mean(t_max_boot >= t_max_obs))

        idx_local = int(np.argmax(t_obs))
        eliminado = activos.pop(idx_local)
        orden.append(eliminado)
        p_pasos.append(p_paso)

    orden.append(activos[0])
    p_pasos.append(1.0)

    # P-valores MCS mediante máximo acumulado en el orden de eliminación (HLN 2011, def. 2).
    p_mcs = [0.0] * n_mod
    max_acum = 0.0
    for modelo_idx, p_paso in zip(orden, p_pasos, strict=False):
        max_acum = max(max_acum, p_paso)
        p_mcs[modelo_idx] = float(max_acum)
    p_mcs[orden[-1]] = 1.0

    incluidos = [i for i in range(n_mod) if p_mcs[i] >= nivel_alfa]
    return {
        "included": incluidos,
        "order": orden,
        "p_values": p_mcs,
    }


def _cargar_tabla_csv(ruta: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Lee un CSV con columna ``session`` y una columna numérica por modelo sin huecos."""
    if not ruta.is_file():
        raise ValueError(f"No existe el archivo CSV especificado: {ruta}")

    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if lector.fieldnames is None:
            raise ValueError("El archivo CSV está vacío o no tiene cabecera.")

        columnas = [col.strip() for col in lector.fieldnames]
        if len(set(columnas)) != len(columnas):
            raise ValueError("El archivo CSV contiene nombres de columna duplicados.")
        if "session" not in columnas:
            raise ValueError("El archivo CSV debe incluir una columna llamada 'session'.")

        nombres_modelos = [col for col in columnas if col != "session"]
        if not nombres_modelos:
            raise ValueError("El archivo CSV no contiene columnas de modelos.")

        sesiones: list[str] = []
        filas_valores: list[list[float]] = []
        for num_fila, fila in enumerate(lector, start=2):
            fila_limpia = {(k.strip() if k is not None else ""): v for k, v in fila.items()}
            sesion = (fila_limpia.get("session") or "").strip()
            if not sesion:
                raise ValueError(f"Valor vacío en la columna 'session' en la fila {num_fila}.")
            sesiones.append(sesion)

            valores_fila: list[float] = []
            for modelo in nombres_modelos:
                texto = fila_limpia.get(modelo)
                if texto is None or not texto.strip():
                    raise ValueError(
                        f"Hueco o valor faltante para '{modelo}' en la fila {num_fila}."
                    )
                try:
                    val = float(texto.strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Valor no numérico '{texto}' para '{modelo}' en la fila {num_fila}."
                    ) from exc
                if not math.isfinite(val):
                    raise ValueError(f"Valor no finito para '{modelo}' en la fila {num_fila}.")
                valores_fila.append(val)
            filas_valores.append(valores_fila)

    if len(filas_valores) < 4:
        raise ValueError(
            "El archivo CSV debe tener al menos 4 sesiones para la inferencia automática; "
            f"se encontraron {len(filas_valores)}."
        )

    matriz = np.asarray(filas_valores, dtype=np.float64)
    return sesiones, nombres_modelos, matriz


def _analizar_matriz(
    nombres_modelos: list[str],
    matriz: np.ndarray,
    benchmark: str,
    nivel_alfa: float,
    n_boot: int,
    semilla: int,
    *,
    blocks: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Ejecuta intervalos bootstrap, HAC, DM, SPA y MCS sobre una matriz ``[T, M]``."""
    idx_ref = nombres_modelos.index(benchmark)
    serie_ref = matriz[:, idx_ref]
    indices_cand = [i for i in range(len(nombres_modelos)) if i != idx_ref]
    if not indices_cand:
        raise ValueError(
            "El CSV debe contener al menos un modelo candidato además de la referencia."
        )

    nombres_cand = [nombres_modelos[i] for i in indices_cand]
    matriz_cand = matriz[:, indices_cand]

    # Intervalos bootstrap al 95 % con remuestreo simultáneo compartido entre modelos.
    rng_ic = np.random.default_rng(semilla)
    if blocks is None:
        bloque_comun = _longitud_bloque_multivariante(matriz)
        indices_boot = stationary_bootstrap_indices(matriz.shape[0], bloque_comun, n_boot, rng_ic)
        n_strata = 1
    else:
        particiones, longitudes, bloque_comun = _resolver_bloques_bootstrap(matriz, blocks, None)
        indices_boot = stratified_stationary_bootstrap_indices(blocks, longitudes, n_boot, rng_ic)
        n_strata = len(particiones)
    medias_boot = _medias_bootstrap(matriz, indices_boot)  # [B, M]
    ic_inf = np.quantile(medias_boot, 0.025, axis=0)
    ic_sup = np.quantile(medias_boot, 0.975, axis=0)

    resumen_modelos: dict[str, dict[str, Any]] = {}
    for col, nombre in enumerate(nombres_modelos):
        serie = matriz[:, col]
        resumen_modelos[nombre] = {
            "mean": float(np.mean(serie)),
            "t_simple": _t_simple(serie),
            "t_newey_west": newey_west_t(serie, blocks=blocks),
            "bootstrap_ci_95": {
                "mean": float(np.mean(serie)),
                "low": float(ic_inf[col]),
                "high": float(ic_sup[col]),
                "block": float(bloque_comun),
            },
            "diebold_mariano": diebold_mariano(serie, serie_ref, blocks=blocks),
        }

    res_spa = spa_test(matriz_cand, serie_ref, B=n_boot, block=None, seed=semilla, blocks=blocks)
    res_spa_salida = {
        "statistic": res_spa["statistic"],
        "p_lower": res_spa["p_lower"],
        "p_consistent": res_spa["p_consistent"],
        "p_upper": res_spa["p_upper"],
        "best_candidate": nombres_cand[res_spa["best_candidate"]],
        "best_candidate_index": res_spa["best_candidate"],
    }

    res_mcs = model_confidence_set(
        matriz, alpha=nivel_alfa, B=n_boot, block=None, seed=semilla, blocks=blocks
    )
    res_mcs_salida = {
        "alpha": nivel_alfa,
        "included": [nombres_modelos[i] for i in res_mcs["included"]],
        "included_indices": res_mcs["included"],
        "order": [nombres_modelos[i] for i in res_mcs["order"]],
        "order_indices": res_mcs["order"],
        "p_values": {
            nombres_modelos[i]: res_mcs["p_values"][i] for i in range(len(nombres_modelos))
        },
    }

    return {
        "metadata": {
            "n_sessions": int(matriz.shape[0]),
            "n_models": len(nombres_modelos),
            "benchmark": benchmark,
            "alpha": nivel_alfa,
            "bootstrap": n_boot,
            "seed": semilla,
            "bootstrap_block": float(bloque_comun),
            "stratified": blocks is not None,
            "n_strata": n_strata,
        },
        "models": resumen_modelos,
        "spa": res_spa_salida,
        "mcs": res_mcs_salida,
    }


def analizar_tabla_ic(
    csv_path: str | Path,
    benchmark: str,
    alpha: float = 0.10,
    bootstrap: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Ejecuta el análisis completo de inferencia IC sobre un archivo CSV.

    Si las sesiones siguen ``<year>:<market>:<instant>``, estratifica HAC, bootstrap
    estacionario, SPA y MCS por ``(origin_year, market)`` en orden cronológico y,
    cuando hay más de un mercado, añade la clave ``by_market`` con inferencia por mercado.
    """
    nivel_alfa = _validar_flotante_finito(alpha, "alpha")
    if not (0.0 < nivel_alfa < 1.0):
        raise ValueError(f"alpha debe estar en (0, 1); se recibió {nivel_alfa}.")
    n_boot = _validar_entero(bootstrap, "bootstrap", minimo=1)
    semilla = _validar_entero(seed, "seed", minimo=0)

    sesiones, nombres_modelos, matriz = _cargar_tabla_csv(Path(csv_path))
    if benchmark not in nombres_modelos:
        raise ValueError(
            f"El modelo de referencia '{benchmark}' no existe en el CSV. "
            f"Modelos disponibles: {', '.join(nombres_modelos)}."
        )

    parsed = [_parse_session_stratum(s) for s in sesiones]
    if all(item is not None for item in parsed):
        parsed_typed = [item for item in parsed if item is not None]
        order = sorted(range(len(parsed_typed)), key=lambda idx: parsed_typed[idx])
        matriz_ord = matriz[order]
        parsed_ord = [parsed_typed[idx] for idx in order]
        blocks = [(year, market) for year, market, _ in parsed_ord]
        resultado = _analizar_matriz(
            nombres_modelos,
            matriz_ord,
            benchmark,
            nivel_alfa,
            n_boot,
            semilla,
            blocks=blocks,
        )
        unique_markets = sorted({market for _, market, _ in parsed_ord})
        if len(unique_markets) > 1:
            by_market: dict[str, Any] = {}
            for market in unique_markets:
                m_indices = [i for i, (_, m, _) in enumerate(parsed_ord) if m == market]
                if len(m_indices) < 2:
                    by_market[market] = {
                        "metadata": {"n_sessions": len(m_indices), "market": market},
                        "error": "Menos de 2 sesiones en el mercado.",
                    }
                    continue
                m_blocks = [(parsed_ord[i][0], market) for i in m_indices]
                by_market[market] = _analizar_matriz(
                    nombres_modelos,
                    matriz_ord[m_indices],
                    benchmark,
                    nivel_alfa,
                    n_boot,
                    semilla,
                    blocks=m_blocks,
                )
            resultado["by_market"] = by_market
        return resultado

    return _analizar_matriz(
        nombres_modelos,
        matriz,
        benchmark,
        nivel_alfa,
        n_boot,
        semilla,
        blocks=None,
    )


def _construir_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos de línea de órdenes."""
    parser = argparse.ArgumentParser(
        description=(
            "Inferencia estadística robusta (Newey-West, bootstrap estacionario, "
            "Diebold-Mariano, SPA y MCS) sobre series diarias de IC."
        )
    )
    parser.add_argument(
        "csv_path",
        help="Ruta al archivo CSV con columna 'session' y una columna por modelo.",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Nombre de la columna del modelo de referencia.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta del archivo JSON de salida.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Nivel de significación del Model Confidence Set (por defecto: 0.10).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Número de réplicas del bootstrap estacionario (por defecto: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Semilla determinista para el bootstrap (por defecto: 0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de la interfaz de línea de órdenes."""
    parser = _construir_parser()
    args = parser.parse_args(argv)

    resultado = analizar_tabla_ic(
        csv_path=args.csv_path,
        benchmark=args.benchmark,
        alpha=args.alpha,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    ruta_salida = Path(args.output)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with ruta_salida.open("w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, indent=2, ensure_ascii=False)
        archivo.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
