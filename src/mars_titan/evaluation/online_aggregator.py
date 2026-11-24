#!/usr/bin/env python3
# ruff: noqa: E501, B023
"""Agregación causal en línea sobre emisiones de varios modelos en formato Parquet.

Procedencia: entrega «arbitraje-agente» del tercer enjambre de Gemini (instantánea de las
19:22 UTC del 22-11-2026), adoptada tras revisarla.
- Revisión estática: solo usa numpy, pyarrow y la biblioteca estándar, y solo escribe
  report.json y report.md en --output.
- Sus 22 pruebas pasan con el intérprete de la rama, y la paridad se comprueba contra
  scripts/dgx_screening_report.py de esta rama.
- B023 está silenciado porque los cierres de la abstención por activo se usan dentro de la
  misma vuelta.
- Lo que se usa para medir: la barrera de madurez, la media de rangos, ridge_cov_active, los
  placebos, el modo operable y los contrastes. La abstención y los pesos por régimen se
  calculan, pero no se adoptan para decidir nada.

Sigue exactamente el esquema y las convenciones de `load_split`, `average_ranks`,
`session_ic`, `paired` y Newey-West HAC de `material/repo-v11/scripts/dgx_screening_report.py`,
añadiendo la columna `asset_id` y soporte opcional para el objetivo operable (`alpha`,
apagado por defecto).

Restricciones de diseño (invariantes comprobadas por pruebas y mutaciones):
1. Causalidad estricta con etiquetas maduras: los pesos y decisiones de la sesión `t`
   dependen exclusivamente de sesiones del mismo mercado cuya etiqueta haya madurado
   en o antes del instante de decisión `t` (`matures_at <= prediction_at[t]`, o bien
   índice `<= t - maturity_lag` con `maturity_lag >= 1`). Jamás se lee la etiqueta de `t`
   ni de ninguna sesión futura.
2. Cero fuga de evaluación: la selección del mejor componente individual, todos los
   hiperparámetros de los agregadores y los factores de escala `k_m` del modo operable se
   fijan exclusivamente en calibración (o validación + calibración si se declara) antes de
   abrir el tramo de evaluación.
3. Sello de 2024: se rechaza cualquier marca temporal >= 2024-01-01T00:00:00Z.
4. Determinismo IEEE FP32 en la ruta de predicciones y combinaciones (ordenación estable
   `mergesort` y reducción secuencial por modelo sin variabilidad BLAS), con acumuladores
   estadísticos estables (Newey-West HAC, potencial de AdaNormalHedge y resolución Ridge)
   protegidos en FP64.

Uso por línea de órdenes:
    ./py entregas/arbitraje-agente/online_aggregator.py \\
        --model titans_nogain=/ruta/runs/titans_mac_nogain/2013/seed-42 \\
        --model xattn=/ruta/runs/cross_asset_attention/2013/seed-42 \\
        --model lgbm=/ruta/runs/lightgbm_technical/2013/seed-42 \\
        --output /ruta/salida [--operable --alpha-col alpha]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.training.learning_hold import require_learning_allowed

SPLITS = ("validation", "calibration", "evaluation")
DECISION_SPLITS = ("validation", "calibration")
CALIBRATION_SPLIT = ("calibration",)
REQUIRED_COLUMNS = ("sample_id", "asset_id", "market", "prediction_at", "target", "prediction")
MIN_ASSETS = 30
# 2024-01-01 00:00:00 UTC en microsegundos (reserva sellada del proyecto).
FINAL_TEST_START_US = 1_704_067_200_000_000
OPERABLE_SCALES = (
    0.01,
    0.02,
    0.03,
    0.05,
    0.07,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    0.7,
    1.0,
    1.5,
    2.0,
    3.0,
)

# Rejillas compactas declaradas ex-ante ordenadas de mayor a menor contracción
# (en caso de empate exacto en calibración se retiene siempre la opción más conservadora).
RIDGE_LAMBDAS = (2.0, 1.0, 0.5, 0.1)
RIDGE_WINDOWS = (126, 63, 21)
RIDGE_HALF_LIVES = (None, 63.0, 21.0)
HEDGE_ETAS = (0.5, 1.0, 2.0, 4.0, 8.0)
REGIME_TAUS = (100.0, 30.0, 10.0)
ABSTENTION_THRESHOLDS = (-0.02, 0.0, 0.02)
MIN_CALL_COVERAGE = 0.50


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# ---------------------------------------------------------------------------
# Rangos, IC en FP32 determinista y Newey-West HAC por bloques de mercado
# ---------------------------------------------------------------------------


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Rangos medios (empates comparten el rango medio), idéntico a `dgx_screening_report.py`."""
    arr = np.asarray(values)
    count = len(arr)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ordered = arr[order]
    first = np.empty(count, dtype=bool)
    first[:1] = True
    first[1:] = ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(first)
    ends = np.empty(len(starts), dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1:] = count
    ranks = np.empty(count, dtype=np.float64)
    ranks[order] = ((starts + 1 + ends) / 2.0)[np.cumsum(first) - 1]
    return ranks


def standardized_ranks(values: np.ndarray) -> np.ndarray:
    """Rangos medios tipificados (media 0, desviación 1) en FP32 determinista."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.empty(0, dtype=np.float32)
    ranks = average_ranks(arr).astype(np.float32)
    mean = np.float32(np.mean(ranks, dtype=np.float32))
    centered = ranks - mean
    var = np.float32(np.mean(centered * centered, dtype=np.float32))
    scale = np.float32(math.sqrt(float(var)))
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        return np.zeros(arr.size, dtype=np.float32)
    return (centered / scale).astype(np.float32)


def _prep_target_ranks(target: np.ndarray) -> tuple[np.ndarray, float]:
    """Precalcula rangos centrados y norma L2² en FP32 del objetivo de una sesión."""
    targ = np.asarray(target, dtype=np.float32)
    if targ.size < 2:
        return np.empty(0, dtype=np.float32), 0.0
    r_y = average_ranks(targ).astype(np.float32)
    ly = r_y - np.float32(np.mean(r_y, dtype=np.float32))
    n_y = float(np.sum(ly * ly, dtype=np.float32))
    return ly, n_y


def _rank_ic_prepped(prediction: np.ndarray, ly: np.ndarray, n_y: float) -> float:
    """IC de rangos (Spearman) en FP32 bit-exacto contra rangos del objetivo precalculados."""
    pred = np.asarray(prediction, dtype=np.float32)
    if pred.size < 2 or pred.size != ly.size or n_y <= 0.0:
        return float("nan")
    r_p = average_ranks(pred).astype(np.float32)
    lp = r_p - np.float32(np.mean(r_p, dtype=np.float32))
    n_p = float(np.sum(lp * lp, dtype=np.float32))
    denom = math.sqrt(n_p * n_y)
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan")
    num = float(np.sum(lp * ly, dtype=np.float32))
    val = num / denom
    return float(max(-1.0, min(1.0, val))) if math.isfinite(val) else float("nan")


def rank_ic_vector(prediction: np.ndarray, target: np.ndarray) -> float:
    """IC de rangos (Spearman) en FP32 determinista entre dos vectores de una sesión."""
    pred = np.asarray(prediction, dtype=np.float32)
    targ = np.asarray(target, dtype=np.float32)
    if pred.size < 2 or pred.size != targ.size:
        return float("nan")
    ly, n_y = _prep_target_ranks(targ)
    return _rank_ic_prepped(pred, ly, n_y)


def combine_session_scores(z_session: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combinación lineal secuencial en FP32 determinista (sin no-determinismo BLAS)."""
    n_rows, n_models = z_session.shape
    out = np.zeros(n_rows, dtype=np.float32)
    w32 = np.asarray(weights, dtype=np.float32)
    for m in range(n_models):
        wm = np.float32(w32[m])
        if wm != np.float32(0.0):
            out = out + wm * z_session[:, m].astype(np.float32, copy=False)
    return out


def session_rank_ic(
    data: dict[str, Any],
    prediction: np.ndarray | None = None,
    *,
    min_assets: int = MIN_ASSETS,
    abstained: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
    _target_cache: list[tuple[np.ndarray, float] | None] | None = None,
) -> dict[tuple[str, int], float]:
    """IC de rangos por sesión `{(mercado, instante): IC}`.

    - Si `abstained[s]` es True en una sesión puntuable (`>= min_assets` y objetivo no constante),
      la sesión cuenta con IC = 0,0 (sin trampas de evaluación condicional).
    - Si se pasa `exposure[s] in [0, 1]`, la contribución incondicional de la sesión es
      `exposure[s] * IC_s` (de modo que una rampa continua `0 < e_t < 1` se pondera de forma
      exacta y no queda neutralizada por la invarianza de escala de Spearman).
    """
    pred = data["prediction"] if prediction is None else np.asarray(prediction, dtype=np.float32)
    target = data["target"]
    result: dict[tuple[str, int], float] = {}
    for s_idx, (market, instant, start, end) in enumerate(data["sessions"]):
        if end - start < min_assets:
            continue
        if _target_cache is not None and _target_cache[s_idx] is not None:
            ly, n_y = _target_cache[s_idx]  # type: ignore[misc]
            if n_y <= 0.0:
                continue
        else:
            y_seg = target[start:end]
            if float(np.ptp(y_seg)) == 0.0:
                continue
            ly, n_y = _prep_target_ranks(y_seg)
        if abstained is not None and bool(abstained[s_idx]):
            result[(market, instant)] = 0.0
            continue
        exp_val = 1.0
        if exposure is not None:
            exp_val = float(exposure[s_idx])
            _require(
                math.isfinite(exp_val) and 0.0 <= exp_val <= 1.0,
                f"exposure debe estar en [0, 1]; se recibió {exp_val!r}",
            )
            if exp_val <= 0.0:
                result[(market, instant)] = 0.0
                continue
        value = _rank_ic_prepped(pred[start:end], ly, n_y)
        if not math.isnan(value):
            result[(market, instant)] = float(exp_val * value) if exp_val != 1.0 else value
    return result


def automatic_newey_west_lags(n_obs: int) -> int:
    """Regla automática de Newey y West (1994): `floor(4 * (T / 100) ** (2 / 9))`."""
    return min(int(math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))), max(n_obs - 1, 0))


def newey_west_hac(
    values: np.ndarray,
    lags: int | None = None,
    *,
    blocks: list[Any] | None = None,
) -> tuple[float | None, float | None, int]:
    """Devuelve `(t_hac, se_hac, max_lags)` con núcleo de Bartlett segmentado por bloques.

    Idéntico a `_newey_west_hac` en `dgx_screening_report.py:762-805`: las autocovarianzas
    nunca cruzan fronteras entre bloques (mercados u orígenes).
    """
    arr = np.asarray(values, dtype=np.float64)
    count = int(arr.size)
    if count < 2:
        return None, None, 0
    _require(bool(np.isfinite(arr).all()), "newey_west_hac requiere observaciones finitas")
    if lags is not None and (lags < 0 or lags >= count):
        raise ValueError(f"lags debe estar en [0, {count - 1}]; se recibió {lags}.")
    mean = float(np.mean(arr))
    if bool(np.all(arr == arr[0])):
        return None, 0.0, 0

    centered = arr - mean
    if blocks is None:
        partitions = [np.arange(count, dtype=np.int64)]
    else:
        grouped: dict[Any, list[int]] = {}
        for idx, blk in enumerate(blocks):
            grouped.setdefault(blk, []).append(idx)
        partitions = [np.asarray(idxs, dtype=np.int64) for idxs in grouped.values()]

    total_v = 0.0
    max_lags = 0
    for idxs in partitions:
        u = centered[idxs]
        tb = int(u.size)
        if tb == 0:
            continue
        l_b = (
            (automatic_newey_west_lags(tb) if tb > 1 else 0) if lags is None else min(lags, tb - 1)
        )
        max_lags = max(max_lags, l_b)
        v_b = float(np.dot(u, u))
        for k in range(1, l_b + 1):
            weight = 1.0 - k / (l_b + 1.0)
            v_b += 2.0 * weight * float(np.dot(u[k:], u[:-k]))
        total_v += max(v_b, 0.0)

    lrv = total_v / count
    if lrv <= 0.0:
        return None, 0.0, max_lags
    se = math.sqrt(lrv / count)
    return float(mean / se), float(se), int(max_lags)


def summarize_session_dict(ics: dict[tuple[str, int], float]) -> dict[str, Any]:
    """Resumen agregado y por mercado de un diccionario `{(mercado, instante): IC}`."""

    def _one(sub: dict[tuple[str, int], float]) -> dict[str, Any]:
        if not sub:
            return dict(sessions=0, mean=None, std=None, icir=None, se_hac=None, t_nw=None)
        keys = sorted(sub)
        arr = np.asarray([sub[k] for k in keys], dtype=np.float64)
        blocks = [k[:-1] for k in keys]
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        mean = float(arr.mean())
        t_nw, se_hac, _ = newey_west_hac(arr, blocks=blocks)
        return dict(
            sessions=int(len(arr)),
            mean=mean,
            std=std,
            icir=float(mean / std) if std > 0.0 else None,
            se_hac=se_hac,
            t_nw=t_nw,
        )

    markets = sorted({k[0] for k in ics})
    by_market = {mkt: _one({k: v for k, v in ics.items() if k[0] == mkt}) for mkt in markets}
    return dict(all=_one(ics), by_market=by_market)


def paired_contrast(
    candidate: dict[tuple[str, int], float],
    control: dict[tuple[str, int], float],
    *,
    lags: int | None = None,
) -> dict[str, Any]:
    """Contraste pareado por sesión `candidate - control` con error estándar HAC Newey-West."""
    keys = sorted(set(candidate) & set(control))
    if len(keys) < 2:
        return dict(
            sessions=len(keys),
            mean=None,
            se_hac=None,
            se_iid=None,
            t_hac=None,
            t_iid=None,
            hac_lags=0,
            by_market={},
        )
    blocks = (
        [k[:-1] for k in keys] if all(isinstance(k, tuple) and len(k) >= 2 for k in keys) else None
    )
    diff = np.asarray([candidate[k] - control[k] for k in keys], dtype=np.float64)
    mean = float(diff.mean())
    se_iid = float(diff.std(ddof=1) / math.sqrt(len(keys)))
    t_iid = float(mean / se_iid) if se_iid > 0.0 else None
    t_hac, se_hac, hac_lags = newey_west_hac(diff, lags=lags, blocks=blocks)

    by_market = {}
    markets = sorted({k[0] for k in keys})
    if len(markets) > 1:
        for mkt in markets:
            m_cand = {k: candidate[k] for k in keys if k[0] == mkt}
            m_ctrl = {k: control[k] for k in keys if k[0] == mkt}
            sub = paired_contrast(m_cand, m_ctrl, lags=lags)
            sub.pop("by_market", None)
            by_market[mkt] = sub

    return dict(
        sessions=len(keys),
        mean=mean,
        se_hac=se_hac,
        se_iid=se_iid,
        t_hac=t_hac,
        t_iid=t_iid,
        hac_lags=hac_lags,
        by_market=by_market,
    )


# ---------------------------------------------------------------------------
# Lectura de Parquet (`load_split`) y alineación multimodelo
# ---------------------------------------------------------------------------


def sample_digest(column: pa.ChunkedArray | pa.Array) -> str:
    """Huella SHA-256 del orden de `sample_id`, idéntica a `dgx_screening_report.py:698-705`."""
    digest = hashlib.sha256()
    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
    for chunk in chunks:
        for buf in chunk.buffers():
            if buf is not None:
                digest.update(memoryview(buf))
    return digest.hexdigest()


def _to_int64_us(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if pa.types.is_timestamp(arr.type):
        return arr.cast(pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_numpy()
    if pa.types.is_integer(arr.type):
        return arr.cast(pa.int64()).to_numpy()
    raise ValueError(f"Tipo de marca temporal no soportado: {arr.type}")


def load_split(
    path_or_run: Path | str,
    split: str | None = None,
    *,
    alpha_col: str | None = None,
    maturity_col: str | None = None,
) -> dict[str, Any]:
    """Lee un fichero Parquet siguiendo exactamente el esquema y orden de `dgx_screening_report.load_split`.

    Acepta tanto la ruta directa a un `.parquet` como una carpeta de ejecución + nombre del tramo
    (`validation`, `calibration`, `evaluation`), leyendo `<run>/<split>-predictions.parquet`.
    """
    base = Path(path_or_run)
    path = base if split is None else base / f"{split}-predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Falta el archivo de emisiones: {path}")

    available = set(pq.read_schema(path).names)
    missing = [c for c in REQUIRED_COLUMNS if c not in available]
    _require(not missing, f"Faltan columnas requeridas en {path}: {missing}")

    read_cols = list(REQUIRED_COLUMNS)
    if alpha_col is not None:
        _require(
            alpha_col in available,
            f"El modo operable solicita la columna '{alpha_col}', ausente en {path}",
        )
        read_cols.append(alpha_col)
    if maturity_col is not None and maturity_col in available:
        read_cols.append(maturity_col)

    table = pq.read_table(path, columns=read_cols).sort_by(
        [("market", "ascending"), ("prediction_at", "ascending"), ("sample_id", "ascending")]
    )
    _require(table.num_rows > 0, f"El archivo {path} está vacío")
    for col in REQUIRED_COLUMNS:
        _require(table.column(col).null_count == 0, f"La columna '{col}' contiene nulos en {path}")

    instant = _to_int64_us(table.column("prediction_at"))
    _require(
        int(instant.max()) < FINAL_TEST_START_US,
        f"Violación del sello de 2024 en {path}: prediction_at >= 2024-01-01 UTC",
    )

    matures_at = None
    if maturity_col is not None and maturity_col in table.column_names:
        _require(
            table.column(maturity_col).null_count == 0,
            f"La columna '{maturity_col}' contiene nulos en {path}",
        )
        matures_at = _to_int64_us(table.column(maturity_col))
        _require(
            bool(np.all(matures_at > instant)),
            f"Etiquetas con madurez <= instante de decisión en {path}",
        )
        _require(
            int(matures_at.max()) < FINAL_TEST_START_US,
            f"Violación del sello de 2024 en {path}: matures_at >= 2024-01-01 UTC",
        )

    target32 = table.column("target").to_numpy().astype(np.float32)
    pred32 = table.column("prediction").to_numpy().astype(np.float32)
    _require(bool(np.isfinite(target32).all()), f"Valores no finitos en 'target' de {path}")
    _require(bool(np.isfinite(pred32).all()), f"Valores no finitos en 'prediction' de {path}")

    alpha32 = None
    if alpha_col is not None:
        alpha32 = table.column(alpha_col).to_numpy().astype(np.float32)

    market_col = table.column("market").combine_chunks().dictionary_encode()
    codes = market_col.indices.to_numpy(zero_copy_only=False)
    market_names = market_col.dictionary.to_pylist()
    sample_ids = table.column("sample_id").to_pylist()
    asset_ids = table.column("asset_id").to_pylist()

    # Comprobar unicidad de (market, prediction_at, sample_id)
    if len(sample_ids) > 1:
        same_key = (codes[1:] == codes[:-1]) & (instant[1:] == instant[:-1])
        dup_positions = np.flatnonzero(same_key)
        for pos in dup_positions:
            if sample_ids[pos] == sample_ids[pos + 1]:
                raise ValueError(
                    f"Clave duplicada ({market_names[codes[pos]]}, {instant[pos]}, {sample_ids[pos]}) en {path}"
                )

    change = (instant[1:] != instant[:-1]) | (codes[1:] != codes[:-1])
    starts = np.r_[0, np.flatnonzero(change) + 1]
    ends = np.r_[starts[1:], len(instant)]
    sessions = [
        (str(market_names[codes[int(s)]]), int(instant[int(s)]), int(s), int(e))
        for s, e in zip(starts, ends, strict=True)
    ]

    return dict(
        sample=sample_digest(table.column("sample_id")),
        sample_id=sample_ids,
        asset_id=asset_ids,
        market_codes=codes,
        market_names=market_names,
        prediction_at=instant,
        target=target32,
        prediction=pred32,
        alpha=alpha32,
        matures_at=matures_at,
        sessions=sessions,
    )


def build_raw_split_base(
    table: pa.Table,
    *,
    alpha: np.ndarray | None = None,
    matures_at: np.ndarray | None = None,
    already_sorted: bool = False,
) -> dict[str, Any]:
    """Construye la estructura compartida de un tramo en memoria, idéntica a `load_split`.

    Permite compartir metadatos, sesiones, objetivo, `alpha` y `matures_at` entre múltiples
    componentes alineados sin escribir ni releer ficheros Parquet temporales.
    """
    if not already_sorted:
        table = table.sort_by(
            [("market", "ascending"), ("prediction_at", "ascending"), ("sample_id", "ascending")]
        )
    _require(table.num_rows > 0, "La tabla del tramo está vacía")
    for col in ("sample_id", "asset_id", "market", "prediction_at", "target"):
        _require(table.column(col).null_count == 0, f"La columna '{col}' contiene nulos")

    instant = _to_int64_us(table.column("prediction_at"))
    _require(
        int(instant.max()) < FINAL_TEST_START_US,
        "Violación del sello de 2024: prediction_at >= 2024-01-01 UTC",
    )

    mat_arr = None
    if matures_at is not None:
        mat_arr = np.asarray(matures_at, dtype=np.int64)
        _require(len(mat_arr) == len(instant), "Longitud de matures_at incompatible")
        _require(bool(np.all(mat_arr > instant)), "Etiquetas con madurez <= instante de decisión")
        _require(
            int(mat_arr.max()) < FINAL_TEST_START_US,
            "Violación del sello de 2024: matures_at >= 2024-01-01 UTC",
        )

    target32 = table.column("target").to_numpy().astype(np.float32)
    _require(bool(np.isfinite(target32).all()), "Valores no finitos en 'target'")

    alpha32 = None
    if alpha is not None:
        alpha32 = np.asarray(alpha, dtype=np.float64).astype(np.float32)
        _require(len(alpha32) == len(instant), "Longitud de alpha incompatible")

    market_col = table.column("market").combine_chunks().dictionary_encode()
    codes = market_col.indices.to_numpy(zero_copy_only=False)
    market_names = market_col.dictionary.to_pylist()
    sample_ids = table.column("sample_id").to_pylist()
    asset_ids = table.column("asset_id").to_pylist()

    if len(sample_ids) > 1:
        same_key = (codes[1:] == codes[:-1]) & (instant[1:] == instant[:-1])
        dup_positions = np.flatnonzero(same_key)
        for pos in dup_positions:
            if sample_ids[pos] == sample_ids[pos + 1]:
                raise ValueError(
                    f"Clave duplicada ({market_names[codes[pos]]}, {instant[pos]}, {sample_ids[pos]})"
                )

    change = (instant[1:] != instant[:-1]) | (codes[1:] != codes[:-1])
    starts = np.r_[0, np.flatnonzero(change) + 1]
    ends = np.r_[starts[1:], len(instant)]
    sessions = [
        (str(market_names[codes[int(s)]]), int(instant[int(s)]), int(s), int(e))
        for s, e in zip(starts, ends, strict=True)
    ]

    return dict(
        sample=sample_digest(table.column("sample_id")),
        sample_id=sample_ids,
        asset_id=asset_ids,
        market_codes=codes,
        market_names=market_names,
        prediction_at=instant,
        target=target32,
        alpha=alpha32,
        matures_at=mat_arr,
        sessions=sessions,
    )


def build_raw_split(
    base: dict[str, Any],
    prediction: np.ndarray,
    *,
    include_alpha: bool = True,
) -> dict[str, Any]:
    """Adjunta una predicción en FP32 a una base creada por `build_raw_split_base`."""
    pred32 = np.asarray(prediction, dtype=np.float32)
    _require(len(pred32) == len(base["target"]), "Longitud de prediction incompatible")
    _require(bool(np.isfinite(pred32).all()), "Valores no finitos en 'prediction'")
    out = dict(base, prediction=pred32)
    if not include_alpha:
        out["alpha"] = None
    return out


def _rebuild_sessions(
    markets: list[str],
    instants: np.ndarray,
) -> list[tuple[str, int, int, int]]:
    n = len(markets)
    if n == 0:
        return []
    m_arr = np.asarray(markets)
    change = (instants[1:] != instants[:-1]) | (m_arr[1:] != m_arr[:-1])
    starts = np.r_[0, np.flatnonzero(change) + 1]
    ends = np.r_[starts[1:], n]
    return [
        (str(m_arr[int(s)]), int(instants[int(s)]), int(s), int(e))
        for s, e in zip(starts, ends, strict=True)
    ]


def align_models_split(
    models_split: dict[str, dict[str, Any]],
    *,
    align: str = "strict",
) -> dict[str, Any]:
    """Alinea las emisiones de varios modelos en un tramo (`strict` o `inner`)."""
    _require(align in ("strict", "inner"), f"Modo de alineación desconocido: {align!r}")
    names = sorted(models_split)
    _require(len(names) >= 2, "La agregación requiere al menos 2 modelos")
    has_mat = [models_split[n]["matures_at"] is not None for n in names]
    _require(
        all(has_mat) or not any(has_mat),
        "Presencia asimétrica de 'matures_at' entre modelos: todos deben incluirla o ninguno",
    )
    has_alpha = [models_split[n]["alpha"] is not None for n in names]
    _require(
        all(has_alpha) or not any(has_alpha),
        "Presencia asimétrica de 'alpha' entre modelos: todos deben incluirla o ninguno",
    )
    first = models_split[names[0]]

    if align == "strict":
        for name in names[1:]:
            other = models_split[name]
            _require(
                other["sample"] == first["sample"]
                and len(other["sample_id"]) == len(first["sample_id"])
                and other["sessions"] == first["sessions"]
                and np.array_equal(other["prediction_at"], first["prediction_at"]),
                f"Las muestras de '{name}' no están alineadas con '{names[0]}' (use align='inner' si faltan activos)",
            )
            _require(
                other["asset_id"] == first["asset_id"],
                f"Los asset_id de '{name}' difieren de '{names[0]}'",
            )
            _require(
                np.array_equal(other["target"], first["target"]),
                f"Los objetivos (target) de '{name}' difieren de '{names[0]}'",
            )
            if first["matures_at"] is not None:
                _require(
                    other["matures_at"] is not None
                    and np.array_equal(other["matures_at"], first["matures_at"]),
                    f"La madurez (matures_at) de '{name}' difiere de '{names[0]}'",
                )
            if first["alpha"] is not None:
                _require(
                    other["alpha"] is not None
                    and np.array_equal(other["alpha"], first["alpha"], equal_nan=True),
                    f"La columna alpha de '{name}' difiere de '{names[0]}'",
                )
        preds = np.column_stack([models_split[n]["prediction"] for n in names]).astype(np.float32)
        return dict(
            model_names=names,
            sample=first["sample"],
            sample_id=first["sample_id"],
            asset_id=first["asset_id"],
            prediction_at=first["prediction_at"],
            target=first["target"],
            alpha=first["alpha"],
            matures_at=first["matures_at"],
            predictions=preds,
            sessions=first["sessions"],
            dropped_rows={n: 0 for n in names},
            common_rows=len(first["target"]),
        )

    # Modo align="inner": intersección exacta sobre (market, prediction_at, sample_id)
    key_maps: dict[str, dict[tuple[str, int, str], int]] = {}
    for name in names:
        d = models_split[name]
        mkts = [d["market_names"][c] for c in d["market_codes"]]
        kmap = {
            (mkts[i], int(d["prediction_at"][i]), d["sample_id"][i]): i
            for i in range(len(d["sample_id"]))
        }
        key_maps[name] = kmap

    common_keys = sorted(set.intersection(*(set(m) for m in key_maps.values())))
    _require(len(common_keys) > 0, "La intersección de muestras entre modelos está vacía")

    idxs_by_model = {
        name: np.fromiter(
            (key_maps[name][k] for k in common_keys), dtype=np.int64, count=len(common_keys)
        )
        for name in names
    }
    base_idx = idxs_by_model[names[0]]
    target = first["target"][base_idx]
    alpha = None if first["alpha"] is None else first["alpha"][base_idx]
    matures = None if first["matures_at"] is None else first["matures_at"][base_idx]

    for name in names[1:]:
        idx = idxs_by_model[name]
        _require(
            [models_split[name]["asset_id"][int(i)] for i in idx]
            == [first["asset_id"][int(i)] for i in base_idx],
            f"Los asset_id de '{name}' difieren de '{names[0]}' en filas comunes",
        )
        _require(
            np.array_equal(models_split[name]["target"][idx], target),
            f"Los objetivos (target) de '{name}' difieren de '{names[0]}' en filas comunes",
        )
        if matures is not None:
            _require(
                models_split[name]["matures_at"] is not None
                and np.array_equal(models_split[name]["matures_at"][idx], matures),
                f"La madurez (matures_at) de '{name}' difiere de '{names[0]}' en filas comunes",
            )
        if alpha is not None:
            _require(
                models_split[name]["alpha"] is not None
                and np.array_equal(models_split[name]["alpha"][idx], alpha, equal_nan=True),
                f"La columna alpha de '{name}' difiere de '{names[0]}' en filas comunes",
            )

    markets_common = [k[0] for k in common_keys]
    instants_common = np.fromiter(
        (k[1] for k in common_keys), dtype=np.int64, count=len(common_keys)
    )
    sample_ids_common = [k[2] for k in common_keys]
    asset_ids_common = [first["asset_id"][int(i)] for i in base_idx]
    preds = np.column_stack(
        [models_split[n]["prediction"][idxs_by_model[n]] for n in names]
    ).astype(np.float32)

    return dict(
        model_names=names,
        sample=sample_digest(pa.array(sample_ids_common, type=pa.string())),
        sample_id=sample_ids_common,
        asset_id=asset_ids_common,
        prediction_at=instants_common,
        target=target,
        alpha=alpha,
        matures_at=matures,
        predictions=preds,
        sessions=_rebuild_sessions(markets_common, instants_common),
        dropped_rows={n: len(models_split[n]["sample_id"]) - len(common_keys) for n in names},
        common_rows=len(common_keys),
    )


def _validate_chronological_splits(aligned_splits: dict[str, dict[str, Any]]) -> None:
    """Exige validación < calibración < evaluación por mercado, simetría de matures_at y frontera causal."""
    present = [s for s in SPLITS if s in aligned_splits]
    mat_flags = [aligned_splits[s]["matures_at"] is not None for s in present]
    _require(
        all(mat_flags) or not any(mat_flags),
        "Presencia asimétrica de 'matures_at' entre tramos: todos los tramos deben incluirla o ninguno",
    )
    for earlier, later in zip(present, present[1:], strict=False):
        e_by_m: dict[str, list[int]] = {}
        l_by_m: dict[str, list[int]] = {}
        for m, t, _, _ in aligned_splits[earlier]["sessions"]:
            e_by_m.setdefault(m, []).append(t)
        for m, t, _, _ in aligned_splits[later]["sessions"]:
            l_by_m.setdefault(m, []).append(t)
        for m in set(e_by_m) & set(l_by_m):
            _require(
                max(e_by_m[m]) < min(l_by_m[m]),
                f"Orden cronológico violado en mercado {m}: {earlier} llega hasta {max(e_by_m[m])} >= {later} ({min(l_by_m[m])})",
            )
    # Frontera causal de madurez explícita entre calibración y evaluación por mercado
    if "calibration" in aligned_splits and "evaluation" in aligned_splits:
        cal, ev = aligned_splits["calibration"], aligned_splits["evaluation"]
        if cal["matures_at"] is not None:
            ev_first: dict[str, int] = {}
            for m, t, _, _ in ev["sessions"]:
                ev_first[m] = min(ev_first.get(m, t), t)
            for m, _, s, e in cal["sessions"]:
                if m in ev_first:
                    max_mat = int(np.max(cal["matures_at"][s:e]))
                    _require(
                        max_mat <= ev_first[m],
                        f"Frontera causal violada en {m}: etiqueta de calibración madura en {max_mat} > primera decisión evaluada {ev_first[m]}",
                    )


def _calibration_embargo_keys(
    cal_aligned: dict[str, Any],
    *,
    maturity_lag: int,
) -> set[tuple[str, int]]:
    """Sesiones finales de calibración que aún no han madurado al abrir evaluación cuando
    no hay `matures_at` explícito y `maturity_lag > 1` (se purgan de selección y ajuste de `k_m`)."""
    if cal_aligned["matures_at"] is not None or maturity_lag <= 1:
        return set()
    by_mkt: dict[str, list[int]] = {}
    for m, t, _, _ in cal_aligned["sessions"]:
        by_mkt.setdefault(m, []).append(t)
    embargoed: set[tuple[str, int]] = set()
    tail = maturity_lag - 1
    for m, times in by_mkt.items():
        for t in sorted(times)[-tail:]:
            embargoed.add((m, t))
    return embargoed


# ---------------------------------------------------------------------------
# Barrera causal de madurez y utilidades algebraicas
# ---------------------------------------------------------------------------


def mature_session_indices(
    session_times: np.ndarray,
    session_matures: np.ndarray | None,
    current_idx: int,
    *,
    maturity_lag: int = 1,
) -> np.ndarray:
    """Índices estrictamente causales de sesiones ya maduras al decidir `current_idx`.

    - Con `session_matures` explícito (en microsegundos UTC): exige `matures > times` en todas
      las sesiones y devuelve `{j < current_idx : session_matures[j] <= session_times[current_idx]}`.
    - Sin `session_matures`: exige `maturity_lag >= 1` y devuelve `{0, ..., current_idx - maturity_lag}`.
    """
    if type(maturity_lag) is not int or maturity_lag < 1:
        raise ValueError(f"maturity_lag debe ser un entero >= 1; se recibió {maturity_lag!r}")
    times = np.asarray(session_times, dtype=np.int64)
    n = int(times.size)
    if not (0 <= current_idx < n):
        raise ValueError(f"current_idx={current_idx} fuera de rango [0, {n - 1}]")
    if session_matures is not None:
        matures = np.asarray(session_matures, dtype=np.int64)
        _require(
            matures.shape == times.shape and bool(np.all(matures > times)),
            "Toda etiqueta debe madurar estrictamente después de su instante de decisión",
        )
        return np.flatnonzero(matures[:current_idx] <= times[current_idx])
    cutoff = current_idx - maturity_lag
    if cutoff < 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, cutoff + 1, dtype=np.int64)


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Proyección euclídea exacta sobre el símplex {w >= 0, sum w = 1} (Duchi et al., 2008)."""
    x = np.asarray(v, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if not np.isfinite(x).all():
        return np.full(n, 1.0 / n, dtype=np.float64)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=np.float64)
    cond = u - cssv / ind > 0.0
    if not cond.any():
        return np.full(n, 1.0 / n, dtype=np.float64)
    rho = int(np.flatnonzero(cond)[-1])
    theta = cssv[rho] / (rho + 1.0)
    w = np.maximum(x - theta, 0.0)
    s = float(w.sum())
    return (w / s) if s > 0.0 else np.full(n, 1.0 / n, dtype=np.float64)


# ---------------------------------------------------------------------------
# Algoritmos causales de pesos por sesión (sobre un único mercado cronológico)
# ---------------------------------------------------------------------------


def rank_average_weights(n_sessions: int, n_models: int) -> np.ndarray:
    """Pesos uniformes 1/M para la media simple de rangos."""
    return np.full((n_sessions, n_models), 1.0 / n_models, dtype=np.float64)


def ridge_stacking_weights(
    session_ics: np.ndarray,
    session_covs: np.ndarray,
    *,
    lam: float = 0.5,
    window: int | None = None,
    half_life: float | None = None,
    session_times: np.ndarray | None = None,
    session_matures: np.ndarray | None = None,
    maturity_lag: int = 1,
    scored: np.ndarray | None = None,
    use_current_cov: bool = False,
    se_shrinkage: float = 0.0,
    shrink_to_uniform: bool = True,
) -> np.ndarray:
    """Apilado por regresión Ridge con contracción en ventana móvil o exponencial (EWMA).

    En cada sesión `t` resuelve:
        (A_t + lam * I) w = b_t + (lam * w_0 si shrink_to_uniform)
    donde `b_t` es la media (móvil o EWMA) de los IC de rangos de las sesiones maduras
    `M(t) = mature_session_indices(...)`, y `A_t` es la covarianza causal de rangos
    estandarizados (madura si `use_current_cov=False` o de la propia sesión `t` conocida
    al decidir sin etiquetas si `use_current_cov=True`).
    Con `se_shrinkage > 0`, restringe la inversión al conjunto activo `{m : b_m - c*SE_m > 0}`
    para anular a cero expertos placebo y rescatar órganos débiles ortogonales.
    """
    ics = np.asarray(session_ics, dtype=np.float64)
    covs = np.asarray(session_covs, dtype=np.float64)
    n_sessions, n_models = ics.shape
    _require(covs.shape == (n_sessions, n_models, n_models), "Forma de covarianzas incompatible")
    _require(lam > 0.0 and math.isfinite(lam), "lam debe ser positivo y finito")
    if window is not None:
        _require(
            type(window) is int and window >= 1,
            f"window debe ser un entero >= 1 o None; se recibió {window!r}",
        )
    if half_life is not None:
        _require(
            isinstance(half_life, (int, float))
            and not isinstance(half_life, bool)
            and math.isfinite(float(half_life))
            and float(half_life) > 0.0,
            f"half_life debe ser un número positivo y finito o None; se recibió {half_life!r}",
        )
    _require(
        math.isfinite(se_shrinkage) and se_shrinkage >= 0.0,
        f"se_shrinkage debe ser finito y >= 0; se recibió {se_shrinkage!r}",
    )
    times = np.arange(n_sessions, dtype=np.int64) if session_times is None else np.asarray(session_times, dtype=np.int64)
    sc = np.ones(n_sessions, dtype=bool) if scored is None else np.asarray(scored, dtype=bool)
    if type(maturity_lag) is not int or maturity_lag < 1:
        raise ValueError(f"maturity_lag debe ser un entero >= 1; se recibió {maturity_lag!r}")
    mat_arr = None
    if session_matures is not None:
        mat_arr = np.asarray(session_matures, dtype=np.int64)
        _require(
            mat_arr.shape == times.shape and bool(np.all(mat_arr > times)),
            "Toda etiqueta debe madurar estrictamente después de su instante de decisión",
        )

    uniform = np.full(n_models, 1.0 / n_models, dtype=np.float64)
    out = np.empty((n_sessions, n_models), dtype=np.float64)
    eye = np.eye(n_models, dtype=np.float64)
    ics_clean = np.nan_to_num(ics, nan=0.0)
    covs_clean = np.nan_to_num(covs, nan=0.0)

    for t in range(n_sessions):
        if mat_arr is not None:
            matured = np.flatnonzero((mat_arr[:t] <= times[t]) & sc[:t])
        else:
            cutoff = t - maturity_lag
            matured = np.flatnonzero(sc[: cutoff + 1]) if cutoff >= 0 else np.empty(0, dtype=np.int64)
        if window is not None and matured.size > window:
            matured = matured[-window:]
        if matured.size == 0:
            out[t] = uniform
            continue

        k = matured.size
        if half_life is not None:
            ages = np.arange(k - 1, -1, -1, dtype=np.float64)
            decay = np.power(2.0, -ages / float(half_life))
        else:
            decay = np.ones(k, dtype=np.float64)
        sw = float(decay.sum())
        norm_w = decay / sw
        n_eff = float((sw * sw) / np.sum(decay * decay))

        hist_ic = ics_clean[matured]
        b_mean = (norm_w[:, None] * hist_ic).sum(axis=0)

        if use_current_cov:
            cov_t = covs_clean[t]
        else:
            cov_t = np.tensordot(norm_w, covs_clean[matured], axes=(0, 0))

        if se_shrinkage > 0.0:
            if k >= 2:
                diff = hist_ic - b_mean[None, :]
                var = (norm_w[:, None] * (diff * diff)).sum(axis=0)
                se = np.sqrt(np.maximum(var, 0.0) / max(n_eff, 1.0))
            else:
                se = np.zeros(n_models, dtype=np.float64)
            b_adj = np.maximum(b_mean - se_shrinkage * se, 0.0)
            active = np.flatnonzero(b_adj > 0.0)
            if active.size == 0:
                out[t] = uniform
                continue
            sub_a = cov_t[np.ix_(active, active)] + lam * np.eye(active.size, dtype=np.float64)
            sub_rhs = b_adj[active]
            raw_sub = np.linalg.solve(sub_a, sub_rhs)
            raw_sub = np.maximum(raw_sub, 0.0)
            if float(raw_sub.sum()) <= 0.0:
                out[t] = uniform
            else:
                w_full = np.zeros(n_models, dtype=np.float64)
                w_full[active] = raw_sub / float(raw_sub.sum())
                out[t] = w_full
        else:
            # Escala consistente con ||b_mean|| para que lam regule la contracción hacia 1/M
            # sin desbalancear amplitudes (b_mean ~ 0,10 frente a lam ~ 1,0) ni inflar placebos.
            scale_b = float(np.mean(np.maximum(b_mean, 0.0)))
            if scale_b <= 1e-12:
                out[t] = uniform
                continue
            rhs = b_mean + (lam * scale_b * uniform if shrink_to_uniform else 0.0)
            raw = np.linalg.solve(cov_t + lam * eye, rhs)
            raw_pos = np.maximum(raw, 0.0)
            s_raw = float(raw_pos.sum())
            out[t] = (raw_pos / s_raw) if (s_raw > 0.0 and math.isfinite(s_raw)) else uniform
    return out


def hedge_weights(
    losses: np.ndarray,
    *,
    eta: float = 2.0,
    fixed_share_alpha: float = 0.0,
    prior: np.ndarray | None = None,
    session_times: np.ndarray | None = None,
    session_matures: np.ndarray | None = None,
    maturity_lag: int = 1,
    scored: np.ndarray | None = None,
) -> np.ndarray:
    """Hedge / Multiplicative Weights Update (Freund y Schapire, 1997) con Fixed-Share opcional.

    Opera estrictamente sobre sesiones maduras `M(t)`. Con `fixed_share_alpha > 0`
    implementa seguimiento de expertos cambiantes (Herbster y Warmuth, 1998; Adamskiy et al., 2016).
    """
    ell = np.asarray(losses, dtype=np.float64)
    n_sessions, n_models = ell.shape
    _require(eta > 0.0 and math.isfinite(eta), "eta debe ser positivo y finito")
    _require(0.0 <= fixed_share_alpha < 1.0, "fixed_share_alpha debe estar en [0, 1)")
    q = (
        np.full(n_models, 1.0 / n_models, dtype=np.float64)
        if prior is None
        else np.asarray(prior, dtype=np.float64)
    )
    q = q / float(q.sum())
    times = np.arange(n_sessions, dtype=np.int64) if session_times is None else session_times
    sc = np.ones(n_sessions, dtype=bool) if scored is None else np.asarray(scored, dtype=bool)

    out = np.empty((n_sessions, n_models), dtype=np.float64)
    applied = np.zeros(n_sessions, dtype=bool)
    w = q.copy()

    for t in range(n_sessions):
        matured = mature_session_indices(times, session_matures, t, maturity_lag=maturity_lag)
        newly = matured[~applied[matured]]
        for s in newly:
            applied[s] = True
            if not sc[s]:
                continue
            row = ell[s]
            _require(
                bool(np.isfinite(row).all() and (row >= 0.0).all() and (row <= 1.0).all()),
                "Las pérdidas de Hedge deben ser finitas y pertenecer a [0, 1]",
            )
            log_w = np.log(np.maximum(w, 1e-300)) - eta * row
            log_w -= float(np.max(log_w))
            w = np.exp(log_w)
            w /= float(w.sum())
            if fixed_share_alpha > 0.0:
                w = (1.0 - fixed_share_alpha) * w + fixed_share_alpha * q
        out[t] = w
    return out


def _anh_potential(regret: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    pos = np.maximum(regret, 0.0)
    exp = np.divide(pos * pos, 3.0 * magnitude, out=np.zeros_like(pos), where=pos > 0.0)
    return np.exp(exp)


def _anh_weights(regret: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    scale = 3.0 * (magnitude + 1.0)
    upper = np.maximum(regret + 1.0, 0.0) ** 2 / scale
    lower = np.maximum(regret - 1.0, 0.0) ** 2 / scale
    return 0.5 * np.exp(lower) * np.expm1(upper - lower)


@dataclass
class AdaNormalHedgeState:
    """Estado de AdaNormalHedge sin parámetros (Luo y Schapire, 2015, Algoritmo 1)."""

    prior: np.ndarray
    regret: np.ndarray
    magnitude: np.ndarray
    rounds: int = 0

    @classmethod
    def start(cls, n_models: int, prior: np.ndarray | None = None) -> AdaNormalHedgeState:
        q = (
            np.full(n_models, 1.0 / n_models, dtype=np.float64)
            if prior is None
            else np.asarray(prior, dtype=np.float64)
        )
        _require(q.ndim == 1 and q.size >= 1 and bool(np.all(q > 0.0)), "Prior inválido")
        q = q / float(q.sum())
        return cls(q, np.zeros(q.size, dtype=np.float64), np.zeros(q.size, dtype=np.float64), 0)

    def distribution(self) -> np.ndarray:
        mass = self.prior * _anh_weights(self.regret, self.magnitude)
        total = float(mass.sum())
        if total <= 0.0 or not math.isfinite(total):
            return self.prior.copy()
        return mass / total

    def update(self, losses: np.ndarray, emitted_weights: np.ndarray | None = None) -> float:
        ell = np.asarray(losses, dtype=np.float64)
        _require(
            ell.shape == self.prior.shape
            and bool(np.isfinite(ell).all() and (ell >= 0.0).all() and (ell <= 1.0).all()),
            "Las pérdidas de AdaNormalHedge deben ser finitas y pertenecer a [0, 1]",
        )
        p = (
            self.distribution()
            if emitted_weights is None
            else np.asarray(emitted_weights, dtype=np.float64)
        )
        mixed = float(p @ ell)
        inst = mixed - ell
        self.regret = self.regret + inst
        self.magnitude = self.magnitude + np.abs(inst)
        self.rounds += 1
        return mixed

    def potential_bound(self) -> tuple[float, float]:
        """Devuelve `(sum q_i Phi(R_i, C_i), B)` del Lema 2 de Luo y Schapire (2015)."""
        total = math.fsum(self.prior * _anh_potential(self.regret, self.magnitude))
        bound = 1.0 + 1.5 * math.fsum(self.prior * (1.0 + np.log1p(self.magnitude)))
        return total, bound


def adanormalhedge_weights(
    losses: np.ndarray,
    *,
    prior: np.ndarray | None = None,
    session_times: np.ndarray | None = None,
    session_matures: np.ndarray | None = None,
    maturity_lag: int = 1,
    scored: np.ndarray | None = None,
    use_emitted_weights: bool = True,
    return_state: bool = False,
) -> np.ndarray | tuple[np.ndarray, AdaNormalHedgeState]:
    """Pesos causales por sesión de AdaNormalHedge (Luo y Schapire, 2015).

    Por defecto (`use_emitted_weights=True`), cada sesión que madura se actualiza respecto a
    los pesos realmente emitidos `p_{t_0}` al decidir esa sesión (idéntico a `False` para
    retardo `d = 1`, y libre de deriva de estado intermedio para `d >= 2`)."""
    ell = np.asarray(losses, dtype=np.float64)
    n_sessions, n_models = ell.shape
    times = np.arange(n_sessions, dtype=np.int64) if session_times is None else session_times
    sc = np.ones(n_sessions, dtype=bool) if scored is None else np.asarray(scored, dtype=bool)
    state = AdaNormalHedgeState.start(n_models, prior=prior)

    out = np.empty((n_sessions, n_models), dtype=np.float64)
    applied = np.zeros(n_sessions, dtype=bool)

    for t in range(n_sessions):
        matured = mature_session_indices(times, session_matures, t, maturity_lag=maturity_lag)
        newly = matured[~applied[matured]]
        for s in newly:
            applied[s] = True
            if not sc[s]:
                continue
            em = out[s] if use_emitted_weights else None
            state.update(ell[s], emitted_weights=em)
        out[t] = state.distribution()
    return (out, state) if return_state else out


def regime_conditioned_weights(
    session_ics: np.ndarray,
    regime_observable: np.ndarray,
    *,
    window: int = 10,
    tau: float = 20.0,
    eta: float = 4.0,
    session_times: np.ndarray | None = None,
    session_matures: np.ndarray | None = None,
    maturity_lag: int = 1,
    scored: np.ndarray | None = None,
    min_history: int = 20,
    observable_is_target_based: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pesos condicionados causalmente al régimen observable con contracción jerárquica.

    - Si `observable_is_target_based=True` (p. ej. dispersión transversal de la etiqueta),
      en la sesión `t` SOLO se leen valores de `regime_observable` en `matured = M(t)` (maduras
      hasta `t - d`), comparando la media móvil reciente de `window` sesiones maduras frente a
      la mediana expansiva causal de todas las sesiones maduras hasta `t - d`.
    - Si `observable_is_target_based=False` (p. ej. desacuerdo entre modelos conocido en `t`
      sin etiquetas), se compara el valor actual `regime_observable[t]` frente a la mediana
      expansiva de `0 .. t - 1`.
    - Los pesos del régimen `r` se contraen de forma continua hacia los pesos globales causales
      con factor `alpha = n_r / (n_r + tau)`, evitando colapsos por muestra pequeña y el
      secuestro de peso de especialistas vírgenes.
    """
    ics = np.asarray(session_ics, dtype=np.float64)
    obs = np.asarray(regime_observable, dtype=np.float64)
    n_sessions, n_models = ics.shape
    _require(obs.shape == (n_sessions,), "regime_observable debe tener una entrada por sesión")
    _require(window >= 1 and tau > 0.0 and eta > 0.0, "Parámetros de régimen inválidos")
    times = np.arange(n_sessions, dtype=np.int64) if session_times is None else session_times
    sc = np.ones(n_sessions, dtype=bool) if scored is None else np.asarray(scored, dtype=bool)

    losses = (1.0 - np.nan_to_num(ics, nan=0.0)) / 2.0
    uniform = np.full(n_models, 1.0 / n_models, dtype=np.float64)
    weights_out = np.empty((n_sessions, n_models), dtype=np.float64)
    regimes_out = np.full(n_sessions, -1, dtype=np.int64)

    def _exp_weights(idxs: np.ndarray) -> np.ndarray:
        if idxs.size == 0:
            return uniform.copy()
        cum_loss = losses[idxs].sum(axis=0)
        log_w = -eta * cum_loss
        log_w -= float(np.max(log_w))
        w = np.exp(log_w)
        return w / float(w.sum())

    for t in range(n_sessions):
        matured = mature_session_indices(times, session_matures, t, maturity_lag=maturity_lag)
        matured_sc = matured[sc[matured]] if matured.size > 0 else matured

        if observable_is_target_based:
            if matured_sc.size >= max(min_history, window):
                recent_obs = float(np.mean(obs[matured_sc[-window:]]))
                med_obs = float(np.median(obs[matured_sc]))
                regimes_out[t] = 1 if recent_obs > med_obs else 0
        else:
            if t >= min_history:
                med_obs = float(np.median(obs[:t]))
                regimes_out[t] = 1 if float(obs[t]) > med_obs else 0

        w_global = _exp_weights(matured_sc)
        r_t = int(regimes_out[t])
        if r_t < 0 or matured_sc.size == 0:
            weights_out[t] = w_global
            continue

        same_regime = matured_sc[regimes_out[matured_sc] == r_t]
        n_r = int(same_regime.size)
        if n_r == 0:
            weights_out[t] = w_global
            continue
        w_reg = _exp_weights(same_regime)
        alpha = n_r / (n_r + float(tau))
        weights_out[t] = (1.0 - alpha) * w_global + alpha * w_reg

    return weights_out, regimes_out


# ---------------------------------------------------------------------------
# Abstención por sesión y abstención selectiva por activo (sin trampas)
# ---------------------------------------------------------------------------


def evaluate_session_abstention(
    base_session_ics: np.ndarray,
    *,
    window: int = 21,
    threshold: float = 0.0,
    ramp_width: float = 0.0,
    session_times: np.ndarray | None = None,
    session_matures: np.ndarray | None = None,
    maturity_lag: int = 1,
    cost_per_session: float = 0.0,
) -> dict[str, Any]:
    """Abstención o escalado continuo causal por sesión medido sin trampas.

    - `unconditional_ic`: media de `e_t * IC_t` (una sesión con `e_t = 0` vale `0,0`).
    - `coverage`: media de `e_t` (se marca `eligible_coverage = coverage >= MIN_CALL_COVERAGE`).
    - `active_ic`: media ponderada `sum(e_t * IC_t) / sum(e_t)` (solo interpretable junto a `coverage`).
    - `net_utility`: media de `e_t * (IC_t - cost_per_session)`.
    """
    ics = np.asarray(base_session_ics, dtype=np.float64)
    _require(
        type(window) is int and window >= 1,
        f"window debe ser un entero >= 1; se recibió {window!r}",
    )
    _require(
        math.isfinite(threshold) and math.isfinite(ramp_width) and ramp_width >= 0.0,
        "threshold debe ser finito y ramp_width finito >= 0",
    )
    _require(math.isfinite(cost_per_session), "cost_per_session debe ser finito")
    n = ics.size
    times = np.arange(n, dtype=np.int64) if session_times is None else session_times
    exposure = np.ones(n, dtype=np.float64)
    reliability = np.full(n, np.nan, dtype=np.float64)

    for t in range(n):
        matured = mature_session_indices(times, session_matures, t, maturity_lag=maturity_lag)
        if matured.size == 0:
            # Sin historia madura: exposición neutral completa (no sesga el arranque)
            exposure[t] = 1.0
            continue
        recent = matured[-window:]
        valid = ics[recent][np.isfinite(ics[recent])]
        if valid.size == 0:
            exposure[t] = 1.0
            continue
        rel = float(np.mean(valid))
        reliability[t] = rel
        if ramp_width > 0.0:
            exposure[t] = float(np.clip((rel - threshold) / ramp_width, 0.0, 1.0))
        else:
            exposure[t] = 1.0 if rel > threshold else 0.0

    valid_mask = np.isfinite(ics)
    if not valid_mask.any():
        return dict(
            exposure=exposure,
            reliability=reliability,
            coverage=0.0,
            eligible_coverage=False,
            unconditional_ic=None,
            active_ic=None,
            net_utility=None,
            abstained_sessions=0,
        )

    e_v = exposure[valid_mask]
    ic_v = ics[valid_mask]
    cov = float(np.mean(e_v))
    uncond = float(np.mean(e_v * ic_v))
    active = float(np.sum(e_v * ic_v) / np.sum(e_v)) if float(np.sum(e_v)) > 0.0 else None
    net_util = float(np.mean(e_v * (ic_v - cost_per_session)))
    return dict(
        exposure=exposure,
        reliability=reliability,
        coverage=cov,
        eligible_coverage=bool(cov >= MIN_CALL_COVERAGE),
        unconditional_ic=uncond,
        active_ic=active,
        net_utility=net_util,
        abstained_sessions=int(np.sum(e_v == 0.0)),
    )


def evaluate_asset_abstention(
    session_model_preds: list[np.ndarray],
    session_targets: list[np.ndarray],
    *,
    keep_fraction: float = 0.5,
    eps: float = 1e-3,
    seed: int = 0,
    min_assets: int = MIN_ASSETS,
) -> dict[str, Any]:
    """Abstención selectiva intrasesión por activo con control emparejado por |predicción| y placebo.

    Compara la selección por razón señal/desacuerdo `|z_bar| / (std_ep + eps)` frente a:
    1. Control obligatorio anti-*range-enhancement*: mantener exactamente la misma fracción
       `keep_fraction` según únicamente la magnitud puntual `|z_bar|`.
    2. Placebo aleatorio emparejado en cobertura.
    Informa tanto el IC en el subconjunto activo como el IC de sesión completa poniendo a 0
    los activos abstenidos.
    """
    _require(0.0 < keep_fraction <= 1.0, "keep_fraction debe pertenecer a (0, 1]")
    rng = np.random.default_rng(seed)

    ic_s2d_active, ic_mag_active, ic_rnd_active = [], [], []
    ic_s2d_full, ic_mag_full, ic_rnd_full, ic_base_full = [], [], [], []
    coverages = []

    for preds, targ in zip(session_model_preds, session_targets, strict=True):
        n_assets, n_models = preds.shape
        if n_assets < min_assets:
            continue
        z = np.column_stack([standardized_ranks(preds[:, m]) for m in range(n_models)]).astype(
            np.float64
        )
        z_bar = z.mean(axis=1)
        ep_std = z.std(axis=1)
        score_s2d = np.abs(z_bar) / (ep_std + eps)
        score_mag = np.abs(z_bar)
        score_rnd = rng.random(n_assets)

        k = max(2, int(math.ceil(keep_fraction * n_assets)))
        coverages.append(k / n_assets)

        def _top_mask(sc: np.ndarray) -> np.ndarray:
            order = np.argsort(-sc, kind="mergesort")
            mask = np.zeros(n_assets, dtype=bool)
            mask[order[:k]] = True
            return mask

        m_s2d = _top_mask(score_s2d)
        m_mag = _top_mask(score_mag)
        m_rnd = _top_mask(score_rnd)

        ic_base_full.append(rank_ic_vector(z_bar, targ))
        ic_s2d_active.append(rank_ic_vector(z_bar[m_s2d], targ[m_s2d]))
        ic_mag_active.append(rank_ic_vector(z_bar[m_mag], targ[m_mag]))
        ic_rnd_active.append(rank_ic_vector(z_bar[m_rnd], targ[m_rnd]))

        ic_s2d_full.append(rank_ic_vector(np.where(m_s2d, z_bar, 0.0), targ))
        ic_mag_full.append(rank_ic_vector(np.where(m_mag, z_bar, 0.0), targ))
        ic_rnd_full.append(rank_ic_vector(np.where(m_rnd, z_bar, 0.0), targ))

    def _m(vals: list[float]) -> float | None:
        clean = [v for v in vals if math.isfinite(v)]
        return float(np.mean(clean)) if clean else None

    return dict(
        sessions=len(coverages),
        coverage=float(np.mean(coverages)) if coverages else 0.0,
        base_full_ic=_m(ic_base_full),
        active_ic_signal_to_disagreement=_m(ic_s2d_active),
        active_ic_magnitude_control=_m(ic_mag_active),
        active_ic_random_placebo=_m(ic_rnd_active),
        full_session_ic_signal_to_disagreement=_m(ic_s2d_full),
        full_session_ic_magnitude_control=_m(ic_mag_full),
        full_session_ic_random_placebo=_m(ic_rnd_full),
    )


# ---------------------------------------------------------------------------
# Modo operable (opcional, apagado por defecto)
# ---------------------------------------------------------------------------


def _fit_operable_scales_on_calibration(
    cal_aligned: dict[str, Any],
    *,
    min_assets: int = MIN_ASSETS,
    scales: tuple[float, ...] = OPERABLE_SCALES,
    exclude_keys: set[tuple[str, int]] | None = None,
) -> dict[str, float]:
    """Elige el factor `k_m` óptimo para `k_m * p_m + alpha` frente a `target + alpha`
    ESTRICTAMENTE en las sesiones ya maduras de calibración (idéntico a `dgx_screening_report.best_scale`)."""
    alpha = cal_aligned["alpha"]
    _require(alpha is not None, "Modo operable requiere columna alpha en calibración")
    target = cal_aligned["target"]
    names = cal_aligned["model_names"]
    preds = cal_aligned["predictions"]
    skip = exclude_keys or set()

    chosen: dict[str, float] = {}
    # Precalcular por sesión los activos con alfa finito y los rangos centrados de `y + alpha`.
    prepped_sessions: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for mkt, inst, start, end in cal_aligned["sessions"]:
        if (mkt, inst) in skip:
            continue
        a_seg = alpha[start:end]
        known = np.isfinite(a_seg)
        if int(known.sum()) < min_assets:
            continue
        a_k = a_seg[known]
        y_op = target[start:end][known] + a_k
        ly, n_y = _prep_target_ranks(y_op)
        prepped_sessions.append((start, end, known, a_k, ly, n_y))

    for m_idx, name in enumerate(names):
        p_col = preds[:, m_idx]
        p_segs = [p_col[start:end][known] for start, end, known, _, _, _ in prepped_sessions]
        best_k = scales[0]
        best_score = -2.0
        for factor in scales:
            f32 = np.float32(factor)
            vals = []
            for p_k, (_, _, _, a_k, ly, n_y) in zip(p_segs, prepped_sessions, strict=True):
                p_op = f32 * p_k + a_k
                v = _rank_ic_prepped(p_op, ly, n_y)
                if math.isfinite(v):
                    vals.append(v)
            score = float(np.mean(vals)) if vals else -2.0
            if score > best_score:
                best_score = score
                best_k = factor
        chosen[name] = float(best_k)
    return chosen


def _transform_split_to_operable(
    aligned: dict[str, Any],
    scales: dict[str, float],
) -> dict[str, Any]:
    """Convierte un tramo alineado al objetivo operable filtrando filas con `alpha` finito."""
    alpha = aligned["alpha"]
    _require(alpha is not None, "Modo operable requiere columna alpha en todos los tramos")
    known = np.isfinite(alpha)
    _require(bool(known.any()), "No hay filas con alpha finito en el tramo")

    idx = np.flatnonzero(known)
    mkts = []
    for m, _, s, e in aligned["sessions"]:
        mkts.extend([m] * (e - s))
    mkts_f = [mkts[int(i)] for i in idx]
    instants_f = aligned["prediction_at"][idx]
    alpha_f = alpha[idx]
    target_op = (aligned["target"][idx] + alpha_f).astype(np.float32)

    names = aligned["model_names"]
    preds_op = np.column_stack(
        [
            (np.float32(scales[name]) * aligned["predictions"][idx, m_idx] + alpha_f).astype(
                np.float32
            )
            for m_idx, name in enumerate(names)
        ]
    )
    return dict(
        model_names=names,
        sample=sample_digest(
            pa.array([aligned["sample_id"][int(i)] for i in idx], type=pa.string())
        ),
        sample_id=[aligned["sample_id"][int(i)] for i in idx],
        asset_id=[aligned["asset_id"][int(i)] for i in idx],
        prediction_at=instants_f,
        target=target_op,
        alpha=alpha_f,
        matures_at=None if aligned["matures_at"] is None else aligned["matures_at"][idx],
        predictions=preds_op,
        sessions=_rebuild_sessions(mkts_f, instants_f),
        dropped_rows=aligned["dropped_rows"],
        common_rows=int(idx.size),
    )


# ---------------------------------------------------------------------------
# Pre-cómputo por mercado y orquestación causal completa sin fuga de evaluación
# ---------------------------------------------------------------------------


def _build_market_streams(
    aligned_splits: dict[str, dict[str, Any]],
    splits_order: tuple[str, ...],
    *,
    min_assets: int,
) -> dict[str, dict[str, Any]]:
    """Construye series cronológicas por mercado a lo largo de `splits_order`."""
    markets = sorted({m for sp in splits_order for (m, _, _, _) in aligned_splits[sp]["sessions"]})
    streams: dict[str, dict[str, Any]] = {}

    for mkt in markets:
        z_sessions: list[np.ndarray] = []
        y_sessions: list[np.ndarray] = []
        y_ranks: list[tuple[np.ndarray, float] | None] = []
        raw_p_sessions: list[np.ndarray] = []
        times_list: list[int] = []
        matures_list: list[int] | None = []
        split_tags: list[str] = []
        session_indices_in_split: list[int] = []
        ic_rows: list[list[float]] = []
        cov_rows: list[np.ndarray] = []
        disp_list: list[float] = []
        disagree_list: list[float] = []
        scored_list: list[bool] = []
        mat_flags = [aligned_splits[sp]["matures_at"] is not None for sp in splits_order]
        _require(
            all(mat_flags) or not any(mat_flags),
            "Presencia asimétrica de 'matures_at' entre tramos en _build_market_streams",
        )
        has_explicit_mat = all(mat_flags)

        for sp in splits_order:
            d = aligned_splits[sp]
            n_models = len(d["model_names"])
            for s_idx, (m, inst, start, end) in enumerate(d["sessions"]):
                if m != mkt:
                    continue
                y_seg = d["target"][start:end]
                p_seg = d["predictions"][start:end]
                z_seg = np.column_stack(
                    [standardized_ranks(p_seg[:, col]) for col in range(n_models)]
                ).astype(np.float32)

                n_rows = end - start
                is_scored = bool(n_rows >= min_assets and float(np.ptp(y_seg)) > 0.0)
                if is_scored:
                    ly, n_y = _prep_target_ranks(y_seg)
                    y_rank_entry: tuple[np.ndarray, float] | None = (ly, n_y)
                    row_ics = [_rank_ic_prepped(p_seg[:, col], ly, n_y) for col in range(n_models)]
                    row_ics = [0.0 if math.isnan(v) else v for v in row_ics]
                else:
                    y_rank_entry = None
                    row_ics = [0.0] * n_models

                z64 = z_seg.astype(np.float64)
                cov = (z64.T @ z64) / max(n_rows, 1)
                # Desacuerdo entre modelos conocido en t (sin etiquetas)
                disagree = float(np.sqrt(np.mean(np.var(z64, axis=1))))

                z_sessions.append(z_seg)
                y_sessions.append(y_seg)
                y_ranks.append(y_rank_entry)
                raw_p_sessions.append(p_seg)
                times_list.append(inst)
                if has_explicit_mat:
                    matures_list.append(int(np.max(d["matures_at"][start:end])))
                split_tags.append(sp)
                session_indices_in_split.append(s_idx)
                ic_rows.append(row_ics)
                cov_rows.append(cov)
                disp_list.append(float(np.std(y_seg.astype(np.float64))))
                disagree_list.append(disagree)
                scored_list.append(is_scored)

        ics_arr = np.asarray(ic_rows, dtype=np.float64)
        streams[mkt] = dict(
            z_sessions=z_sessions,
            y_sessions=y_sessions,
            y_ranks=y_ranks,
            raw_p_sessions=raw_p_sessions,
            times=np.asarray(times_list, dtype=np.int64),
            matures=np.asarray(matures_list, dtype=np.int64) if has_explicit_mat else None,
            split_tags=split_tags,
            session_indices_in_split=session_indices_in_split,
            ics=ics_arr,
            losses=(1.0 - ics_arr) / 2.0,
            covs=np.asarray(cov_rows, dtype=np.float64),
            target_dispersion=np.asarray(disp_list, dtype=np.float64),
            model_disagreement=np.asarray(disagree_list, dtype=np.float64),
            scored=np.asarray(scored_list, dtype=bool),
        )
    return streams


def _extend_market_streams(
    base_streams: dict[str, dict[str, Any]],
    aligned_splits: dict[str, dict[str, Any]],
    extra_splits: tuple[str, ...],
    *,
    min_assets: int,
) -> dict[str, dict[str, Any]]:
    """Añade tramos posteriores a `base_streams` sin recalcular rangos de los tramos previos."""
    if not extra_splits:
        return base_streams
    extra = _build_market_streams(aligned_splits, extra_splits, min_assets=min_assets)
    markets = sorted(set(base_streams) | set(extra))
    merged: dict[str, dict[str, Any]] = {}
    for mkt in markets:
        if mkt not in base_streams:
            merged[mkt] = extra[mkt]
            continue
        if mkt not in extra:
            merged[mkt] = base_streams[mkt]
            continue
        a, b = base_streams[mkt], extra[mkt]
        ics_arr = np.concatenate([a["ics"], b["ics"]], axis=0)
        merged[mkt] = dict(
            z_sessions=a["z_sessions"] + b["z_sessions"],
            y_sessions=a["y_sessions"] + b["y_sessions"],
            y_ranks=a["y_ranks"] + b["y_ranks"],
            raw_p_sessions=a["raw_p_sessions"] + b["raw_p_sessions"],
            times=np.concatenate([a["times"], b["times"]]),
            matures=(
                np.concatenate([a["matures"], b["matures"]])
                if a["matures"] is not None and b["matures"] is not None
                else None
            ),
            split_tags=a["split_tags"] + b["split_tags"],
            session_indices_in_split=a["session_indices_in_split"] + b["session_indices_in_split"],
            ics=ics_arr,
            losses=(1.0 - ics_arr) / 2.0,
            covs=np.concatenate([a["covs"], b["covs"]], axis=0),
            target_dispersion=np.concatenate([a["target_dispersion"], b["target_dispersion"]]),
            model_disagreement=np.concatenate([a["model_disagreement"], b["model_disagreement"]]),
            scored=np.concatenate([a["scored"], b["scored"]]),
        )
    return merged


def _slice_streams_for_models(
    streams: dict[str, dict[str, Any]],
    col_indices: list[int],
    *,
    allowed_splits: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Extrae un subconjunto de modelos (y opcionalmente de tramos) desde un stream compartido."""
    idx_arr = np.asarray(col_indices, dtype=np.int64)
    out: dict[str, dict[str, Any]] = {}
    for mkt, st in streams.items():
        if allowed_splits is None:
            row_idx = list(range(len(st["split_tags"])))
        else:
            row_idx = [i for i, sp in enumerate(st["split_tags"]) if sp in allowed_splits]
        if not row_idx:
            continue
        r_arr = np.asarray(row_idx, dtype=np.int64)
        ics_sub = st["ics"][np.ix_(r_arr, idx_arr)]
        covs_sub = st["covs"][r_arr][:, idx_arr, :][:, :, idx_arr]
        out[mkt] = dict(
            z_sessions=[st["z_sessions"][i][:, idx_arr] for i in row_idx],
            y_sessions=[st["y_sessions"][i] for i in row_idx],
            y_ranks=[st["y_ranks"][i] for i in row_idx],
            raw_p_sessions=[st["raw_p_sessions"][i][:, idx_arr] for i in row_idx],
            times=st["times"][r_arr],
            matures=None if st["matures"] is None else st["matures"][r_arr],
            split_tags=[st["split_tags"][i] for i in row_idx],
            session_indices_in_split=[st["session_indices_in_split"][i] for i in row_idx],
            ics=ics_sub,
            losses=(1.0 - ics_sub) / 2.0,
            covs=covs_sub,
            target_dispersion=st["target_dispersion"][r_arr],
            model_disagreement=st["model_disagreement"][r_arr],
            scored=st["scored"][r_arr],
        )
    return out


def _target_cache_for_split(
    aligned_split: dict[str, Any],
    streams: dict[str, dict[str, Any]],
    split_name: str,
) -> list[tuple[np.ndarray, float] | None]:
    """Construye la lista `_target_cache` alineada con `aligned_split['sessions']`."""
    cache: list[tuple[np.ndarray, float] | None] = [None] * len(aligned_split["sessions"])
    for st in streams.values():
        for local_idx, sp in enumerate(st["split_tags"]):
            if sp == split_name:
                cache[st["session_indices_in_split"][local_idx]] = st["y_ranks"][local_idx]
    return cache


def build_shared_panel_cache(
    raw_by_split: dict[str, dict[str, dict[str, Any]]],
    *,
    splits: tuple[str, ...] = SPLITS,
    maturity_lag: int = 1,
    min_assets: int = MIN_ASSETS,
    align: str = "strict",
    operable: bool = False,
    operable_scales: tuple[float, ...] = OPERABLE_SCALES,
) -> dict[str, Any]:
    """Precalcula alineación, transformación operable, rangos por jugador e IC una sola vez.

    Reutilizable en todas las coaliciones de `agent_core`, `organ_aggregation` y `reliability`
    sin repetir `standardized_ranks`, `_fit_operable_scales_on_calibration` ni lecturas Parquet.
    """
    _require(align == "strict", "build_shared_panel_cache solo soporta align='strict'")
    aligned_all = {sp: align_models_split(raw_by_split[sp], align=align) for sp in splits}
    _validate_chronological_splits(aligned_all)
    cal_embargo = _calibration_embargo_keys(aligned_all["calibration"], maturity_lag=maturity_lag)

    scales: dict[str, float] | None = None
    baseline_alpha_ic: dict[str, Any] | None = None
    if operable:
        _require(
            len(operable_scales) >= 2
            and all(isinstance(s, float) and math.isfinite(s) and s > 0 for s in operable_scales)
            and list(operable_scales) == sorted(set(operable_scales)),
            "La rejilla de factores operables son reales positivos, distintos y crecientes",
        )
        scales = _fit_operable_scales_on_calibration(
            aligned_all["calibration"],
            min_assets=min_assets,
            exclude_keys=cal_embargo,
            scales=tuple(operable_scales),
        )
        eval_raw = aligned_all["evaluation"]
        alpha_eval = eval_raw["alpha"]
        known_eval = np.isfinite(alpha_eval)
        base_plus_op = {}
        base_minus_lbl = {}
        for m, inst, s, e in eval_raw["sessions"]:
            k_seg = known_eval[s:e]
            if int(k_seg.sum()) < min_assets:
                continue
            a_seg = alpha_eval[s:e][k_seg]
            y_seg = eval_raw["target"][s:e][k_seg]
            v_op = rank_ic_vector(a_seg, y_seg + a_seg)
            v_lbl = rank_ic_vector(-a_seg, y_seg)
            if math.isfinite(v_op):
                base_plus_op[(m, inst)] = v_op
            if math.isfinite(v_lbl):
                base_minus_lbl[(m, inst)] = v_lbl
        baseline_alpha_ic = dict(
            plus_alpha_on_operable=summarize_session_dict(base_plus_op),
            minus_alpha_on_label=summarize_session_dict(base_minus_lbl),
        )
        aligned_all = {
            sp: _transform_split_to_operable(aligned_all[sp], scales) for sp in splits
        }

    full_streams = _build_market_streams(aligned_all, splits, min_assets=min_assets)
    all_names = aligned_all["evaluation"]["model_names"]
    target_caches = {
        sp: _target_cache_for_split(aligned_all[sp], full_streams, sp) for sp in splits
    }
    individual_ics: dict[str, dict[str, dict[tuple[str, int], float]]] = {}
    individual_summary: dict[str, dict[str, Any]] = {}
    for m_idx, name in enumerate(all_names):
        individual_ics[name] = {}
        individual_summary[name] = {}
        for sp in splits:
            ics_sp = session_rank_ic(
                aligned_all[sp],
                aligned_all[sp]["predictions"][:, m_idx],
                min_assets=min_assets,
                _target_cache=target_caches[sp],
            )
            individual_ics[name][sp] = ics_sp
            individual_summary[name][sp] = summarize_session_dict(ics_sp)

    return dict(
        splits=tuple(splits),
        maturity_lag=maturity_lag,
        min_assets=min_assets,
        align=align,
        operable=operable,
        operable_scales=tuple(operable_scales),
        model_names=all_names,
        aligned_all=aligned_all,
        full_streams=full_streams,
        target_caches=target_caches,
        cal_embargo=cal_embargo,
        scales=scales,
        baseline_alpha_ic=baseline_alpha_ic,
        individual_ics=individual_ics,
        individual_summary=individual_summary,
    )


def _emit_split_predictions_from_weights(
    aligned_splits: dict[str, dict[str, Any]],
    streams: dict[str, dict[str, Any]],
    weights_by_market: dict[str, np.ndarray],
    target_split: str,
    *,
    exposure_by_market: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ensambla predicciones FP32, pesos, máscara de abstención y exposición por sesión de `target_split`."""
    d = aligned_splits[target_split]
    n_models = len(d["model_names"])
    n_sess = len(d["sessions"])
    pred_out = np.zeros(len(d["target"]), dtype=np.float32)
    weights_out = np.zeros((n_sess, n_models), dtype=np.float64)
    abstained_out = np.zeros(n_sess, dtype=bool)
    exposure_out = np.ones(n_sess, dtype=np.float64)

    for mkt, st in streams.items():
        w_m = weights_by_market[mkt]
        e_m = None if exposure_by_market is None else exposure_by_market[mkt]
        for local_idx, sp in enumerate(st["split_tags"]):
            if sp != target_split:
                continue
            s_idx = st["session_indices_in_split"][local_idx]
            _, _, start, end = d["sessions"][s_idx]
            weights_out[s_idx] = w_m[local_idx]
            exp = 1.0 if e_m is None else float(e_m[local_idx])
            exposure_out[s_idx] = exp
            if exp <= 0.0:
                abstained_out[s_idx] = True
                pred_out[start:end] = np.float32(0.0)
            else:
                combined = combine_session_scores(st["z_sessions"][local_idx], w_m[local_idx])
                pred_out[start:end] = (
                    combined if exp == 1.0 else (np.float32(exp) * combined).astype(np.float32)
                )
    return pred_out, weights_out, abstained_out, exposure_out


def _mean_ic_on_splits(
    aligned_splits: dict[str, dict[str, Any]],
    streams: dict[str, dict[str, Any]],
    weights_by_market: dict[str, np.ndarray],
    target_splits: tuple[str, ...],
    *,
    min_assets: int,
    exposure_by_market: dict[str, np.ndarray] | None = None,
    exclude_keys_by_split: dict[str, set[tuple[str, int]]] | None = None,
) -> float:
    vals: list[float] = []
    if exposure_by_market is None:
        # Ruta rápida sin asignar `pred_out` completo ni reordenar los rangos del objetivo.
        for sp in target_splits:
            excl = (exclude_keys_by_split or {}).get(sp, set())
            for mkt in sorted(streams):
                st = streams[mkt]
                w_m = weights_by_market[mkt]
                times = st["times"]
                scored = st["scored"]
                z_sessions = st["z_sessions"]
                y_ranks = st["y_ranks"]
                for local_idx, tag in enumerate(st["split_tags"]):
                    if tag != sp or not scored[local_idx]:
                        continue
                    if (mkt, int(times[local_idx])) in excl:
                        continue
                    yr = y_ranks[local_idx]
                    if yr is None:
                        continue
                    comb = combine_session_scores(z_sessions[local_idx], w_m[local_idx])
                    v = _rank_ic_prepped(comb, yr[0], yr[1])
                    if not math.isnan(v):
                        vals.append(v)
        return float(np.mean(vals)) if vals else -2.0

    for sp in target_splits:
        pred, _, abst, exp = _emit_split_predictions_from_weights(
            aligned_splits,
            streams,
            weights_by_market,
            sp,
            exposure_by_market=exposure_by_market,
        )
        ics = session_rank_ic(
            aligned_splits[sp],
            pred,
            min_assets=min_assets,
            abstained=abst,
            exposure=exp,
        )
        excl = (exclude_keys_by_split or {}).get(sp, set())
        for k, v in ics.items():
            if k not in excl:
                vals.append(v)
    return float(np.mean(vals)) if vals else -2.0


ALL_AGGREGATORS = (
    "rank_average",
    "ridge_rolling",
    "ridge_ewma",
    "ridge_cov_active",
    "hedge",
    "hedge_fixed_share",
    "adanormalhedge",
    "regime_weights",
    "session_abstention",
)


def run_aggregation(
    model_runs: dict[str, Path | str],
    *,
    splits: tuple[str, ...] = SPLITS,
    selection_split: str = "calibration",
    maturity_lag: int = 1,
    min_assets: int = MIN_ASSETS,
    align: str = "strict",
    operable: bool = False,
    alpha_col: str = "alpha",
    maturity_col: str | None = "matures_at",
    placebo_seed: int = 20261122,
    abstention_cost: float = 0.02,
    operable_scales: tuple[float, ...] = OPERABLE_SCALES,
    aggregators: tuple[str, ...] | None = None,
    include_placebos: bool = True,
    include_asset_abstention: bool = True,
    preloaded_raw: dict[str, dict[str, dict[str, Any]]] | None = None,
    precomputed_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ejecuta la agregación causal completa sobre los tramos declarados sin fuga de evaluación.

    - Carga y alinea todos los modelos por tramo.
    - Si `operable=True`, ajusta `k_m` estrictamente en `calibration` y transforma objetivo y
      predicciones a `y + alpha` y `k_m * p_m + alpha`.
    - Elige el mejor componente individual y todos los hiperparámetros de los agregadores
      EXCLUSIVAMENTE sobre los tramos previos a `evaluation` (`validation` + `calibration`,
      puntuando en `selection_split`, que nunca puede ser `"evaluation"`).
    - Evalúa todos los agregadores y placebos en `evaluation` con contrastes pareados HAC
      Newey-West frente al mejor componente individual y frente a la media simple de rangos.
    """
    require_learning_allowed("la agregación en línea de modelos")
    _require(
        selection_split in ("calibration", "validation", "pooled"),
        f"selection_split no puede usar evaluación; se recibió {selection_split!r}",
    )
    _require(
        "calibration" in splits and "evaluation" in splits, "Se requieren calibration y evaluation"
    )
    if selection_split == "validation":
        _require(
            "validation" in splits,
            "selection_split='validation' requiere el tramo 'validation' (incompatible con --no-validation)",
        )
    _require(type(maturity_lag) is int and maturity_lag >= 1, "maturity_lag debe ser entero >= 1")

    if aggregators is None:
        active_aggs = list(ALL_AGGREGATORS)
    else:
        unknown_aggs = [a for a in aggregators if a not in ALL_AGGREGATORS]
        _require(not unknown_aggs, f"Agregadores desconocidos: {unknown_aggs}")
        # Mantener `rank_average` primero como referencia y el orden canónico
        req_set = set(aggregators) | {"rank_average"}
        if include_placebos:
            req_set.add("ridge_cov_active")
        active_aggs = [a for a in ALL_AGGREGATORS if a in req_set]
    active_set = set(active_aggs)

    req_names = sorted(model_runs)
    use_cache = (
        precomputed_cache is not None
        and align == "strict"
        and tuple(splits) == precomputed_cache["splits"]
        and maturity_lag == precomputed_cache["maturity_lag"]
        and min_assets == precomputed_cache["min_assets"]
        and operable == precomputed_cache["operable"]
        and (not operable or tuple(operable_scales) == precomputed_cache["operable_scales"])
        and all(n in precomputed_cache["model_names"] for n in req_names)
    )

    if use_cache:
        cache = precomputed_cache  # type: ignore[assignment]
        all_names = cache["model_names"]
        col_indices = [all_names.index(n) for n in req_names]
        idx_arr = np.asarray(col_indices, dtype=np.int64)
        aligned_splits = {
            sp: dict(
                cache["aligned_all"][sp],
                model_names=req_names,
                predictions=cache["aligned_all"][sp]["predictions"][:, idx_arr],
                dropped_rows={n: 0 for n in req_names},
            )
            for sp in splits
        }
        alignment_info = {
            sp: dict(
                common_rows=aligned_splits[sp]["common_rows"],
                dropped_rows=aligned_splits[sp]["dropped_rows"],
                sessions=len(aligned_splits[sp]["sessions"]),
                scorable_sessions=sum(
                    1 for _, _, s, e in aligned_splits[sp]["sessions"] if e - s >= min_assets
                ),
            )
            for sp in splits
        }
        cal_embargo = cache["cal_embargo"]
        if operable:
            scales = {n: cache["scales"][n] for n in req_names}
            operable_info: dict[str, Any] = dict(
                enabled=True,
                scales=scales,
                scales_on_boundary=sorted(
                    name
                    for name, factor in scales.items()
                    if factor in (operable_scales[0], operable_scales[-1])
                ),
                baseline_alpha_ic=cache["baseline_alpha_ic"],
            )
        else:
            operable_info = dict(enabled=False, scales=None, baseline_alpha_ic=None)
        model_names = req_names
        n_models = len(model_names)
        individual_ics = {n: cache["individual_ics"][n] for n in model_names}
        individual_summary = {n: cache["individual_summary"][n] for n in model_names}
        target_caches = cache["target_caches"]
        pre_eval_splits = tuple(sp for sp in splits if sp != "evaluation")
        pre_streams = _slice_streams_for_models(
            cache["full_streams"], col_indices, allowed_splits=set(pre_eval_splits)
        )
        full_streams = _slice_streams_for_models(cache["full_streams"], col_indices)
    else:
        raw_by_split: dict[str, dict[str, dict[str, Any]]] = {}
        if preloaded_raw is not None:
            for sp in splits:
                raw_by_split[sp] = {name: preloaded_raw[sp][name] for name in req_names}
        else:
            for sp in splits:
                raw_by_split[sp] = {
                    name: load_split(
                        run_path,
                        sp,
                        alpha_col=alpha_col if operable else None,
                        maturity_col=maturity_col,
                    )
                    for name, run_path in sorted(model_runs.items())
                }

        aligned_splits = {sp: align_models_split(raw_by_split[sp], align=align) for sp in splits}
        _validate_chronological_splits(aligned_splits)

        alignment_info = {
            sp: dict(
                common_rows=aligned_splits[sp]["common_rows"],
                dropped_rows=aligned_splits[sp]["dropped_rows"],
                sessions=len(aligned_splits[sp]["sessions"]),
                scorable_sessions=sum(
                    1 for _, _, s, e in aligned_splits[sp]["sessions"] if e - s >= min_assets
                ),
            )
            for sp in splits
        }

        # Embargo de las últimas (maturity_lag - 1) sesiones de calibración si no hay matures_at explícito
        cal_embargo = _calibration_embargo_keys(
            aligned_splits["calibration"], maturity_lag=maturity_lag
        )

        operable_info = dict(enabled=False, scales=None, baseline_alpha_ic=None)
        if operable:
            _require(
                len(operable_scales) >= 2
                and all(isinstance(s, float) and math.isfinite(s) and s > 0 for s in operable_scales)
                and list(operable_scales) == sorted(set(operable_scales)),
                "La rejilla de factores operables son reales positivos, distintos y crecientes",
            )
            scales = _fit_operable_scales_on_calibration(
                aligned_splits["calibration"],
                min_assets=min_assets,
                exclude_keys=cal_embargo,
                scales=tuple(operable_scales),
            )
            # Baseline +alpha sobre lo operable y -alpha sobre la etiqueta en evaluación
            eval_raw = aligned_splits["evaluation"]
            alpha_eval = eval_raw["alpha"]
            known_eval = np.isfinite(alpha_eval)
            base_plus_op = {}
            base_minus_lbl = {}
            for m, inst, s, e in eval_raw["sessions"]:
                k_seg = known_eval[s:e]
                if int(k_seg.sum()) < min_assets:
                    continue
                a_seg = alpha_eval[s:e][k_seg]
                y_seg = eval_raw["target"][s:e][k_seg]
                v_op = rank_ic_vector(a_seg, y_seg + a_seg)
                v_lbl = rank_ic_vector(-a_seg, y_seg)
                if math.isfinite(v_op):
                    base_plus_op[(m, inst)] = v_op
                if math.isfinite(v_lbl):
                    base_minus_lbl[(m, inst)] = v_lbl

            aligned_splits = {
                sp: _transform_split_to_operable(aligned_splits[sp], scales) for sp in splits
            }
            operable_info = dict(
                enabled=True,
                scales=scales,
                # Componentes cuyo factor elegido cae en un borde de la rejilla: la rejilla es corta.
                scales_on_boundary=sorted(
                    name
                    for name, factor in scales.items()
                    if factor in (operable_scales[0], operable_scales[-1])
                ),
                baseline_alpha_ic=dict(
                    plus_alpha_on_operable=summarize_session_dict(base_plus_op),
                    minus_alpha_on_label=summarize_session_dict(base_minus_lbl),
                ),
            )

        model_names = aligned_splits["evaluation"]["model_names"]
        n_models = len(model_names)

        # Construir streams previos a evaluación y extender con evaluación sin reordenar dos veces.
        pre_eval_splits = tuple(sp for sp in splits if sp != "evaluation")
        eval_only_splits = tuple(sp for sp in splits if sp == "evaluation")
        pre_streams = _build_market_streams(aligned_splits, pre_eval_splits, min_assets=min_assets)
        full_streams = _extend_market_streams(
            pre_streams, aligned_splits, eval_only_splits, min_assets=min_assets
        )
        target_caches = {
            sp: _target_cache_for_split(aligned_splits[sp], full_streams, sp) for sp in splits
        }

        # 1) IC individual por modelo y tramo (reutilizando rangos de objetivo cacheados)
        individual_ics = {}
        individual_summary = {}
        for m_idx, name in enumerate(model_names):
            individual_ics[name] = {}
            individual_summary[name] = {}
            for sp in splits:
                ics_sp = session_rank_ic(
                    aligned_splits[sp],
                    aligned_splits[sp]["predictions"][:, m_idx],
                    min_assets=min_assets,
                    _target_cache=target_caches[sp],
                )
                individual_ics[name][sp] = ics_sp
                individual_summary[name][sp] = summarize_session_dict(ics_sp)

    decision_splits = (
        tuple(s for s in ("validation", "calibration") if s in splits)
        if selection_split == "pooled"
        else (selection_split,)
    )
    decision_scores: dict[str, float] = {}
    for name in model_names:
        vals = [
            v
            for sp in decision_splits
            for k, v in individual_ics[name][sp].items()
            if not (sp == "calibration" and k in cal_embargo)
        ]
        decision_scores[name] = float(np.mean(vals)) if vals else -2.0
    # Desempate determinista por nombre ordenado
    best_model_name = max(model_names, key=lambda nm: (decision_scores[nm], -model_names.index(nm)))

    # 2) Selección de hiperparámetros EXCLUSIVAMENTE en tramos previos a `evaluation`
    tune_splits = decision_splits
    tune_exclude_map = {"calibration": cal_embargo}

    def _tune_grid(grid: list[dict[str, Any]], build_weights_fn) -> dict[str, Any]:
        best_cfg = grid[0]
        best_val = -2.0
        for cfg in grid:
            w_by_mkt = {mkt: build_weights_fn(st, cfg) for mkt, st in pre_streams.items()}
            score = _mean_ic_on_splits(
                aligned_splits,
                pre_streams,
                w_by_mkt,
                tune_splits,
                min_assets=min_assets,
                exclude_keys_by_split=tune_exclude_map,
            )
            if score > best_val + 1e-12:
                best_val = score
                best_cfg = cfg
        return dict(best_cfg)

    selected_hyperparams: dict[str, Any] = {}
    weights_map: dict[str, dict[str, np.ndarray]] = {
        "rank_average": {
            mkt: rank_average_weights(len(st["times"]), n_models)
            for mkt, st in full_streams.items()
        }
    }

    if "ridge_rolling" in active_set:
        cfg_ridge_rolling = _tune_grid(
            [dict(lam=lam, window=win) for lam in RIDGE_LAMBDAS for win in RIDGE_WINDOWS],
            lambda st, cfg: ridge_stacking_weights(
                st["ics"],
                st["covs"],
                lam=cfg["lam"],
                window=cfg["window"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
                use_current_cov=False,
                se_shrinkage=0.0,
            ),
        )
        selected_hyperparams["ridge_rolling"] = cfg_ridge_rolling
        weights_map["ridge_rolling"] = {
            mkt: ridge_stacking_weights(
                st["ics"],
                st["covs"],
                lam=cfg_ridge_rolling["lam"],
                window=cfg_ridge_rolling["window"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
                use_current_cov=False,
                se_shrinkage=0.0,
            )
            for mkt, st in full_streams.items()
        }

    if "ridge_ewma" in active_set:
        cfg_ridge_ewma = _tune_grid(
            [dict(lam=lam, half_life=hl) for lam in RIDGE_LAMBDAS for hl in RIDGE_HALF_LIVES],
            lambda st, cfg: ridge_stacking_weights(
                st["ics"],
                st["covs"],
                lam=cfg["lam"],
                half_life=cfg["half_life"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
                use_current_cov=False,
                se_shrinkage=0.0,
            ),
        )
        selected_hyperparams["ridge_ewma"] = cfg_ridge_ewma
        weights_map["ridge_ewma"] = {
            mkt: ridge_stacking_weights(
                st["ics"],
                st["covs"],
                lam=cfg_ridge_ewma["lam"],
                half_life=cfg_ridge_ewma["half_life"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
                use_current_cov=False,
                se_shrinkage=0.0,
            )
            for mkt, st in full_streams.items()
        }

    cfg_ridge_cov_active: dict[str, Any] | None = None
    if "ridge_cov_active" in active_set:
        cfg_ridge_cov_active = _tune_grid(
            [dict(lam=lam, half_life=hl) for lam in RIDGE_LAMBDAS for hl in RIDGE_HALF_LIVES],
            lambda st, cfg: ridge_stacking_weights(
                st["ics"],
                st["covs"],
                lam=cfg["lam"],
                half_life=cfg["half_life"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
                use_current_cov=True,
                se_shrinkage=1.0,
            ),
        )
        selected_hyperparams["ridge_cov_active"] = cfg_ridge_cov_active
        weights_map["ridge_cov_active"] = {
            mkt: ridge_stacking_weights(
                st["ics"],
                st["covs"],
                lam=cfg_ridge_cov_active["lam"],
                half_life=cfg_ridge_cov_active["half_life"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
                use_current_cov=True,
                se_shrinkage=1.0,
            )
            for mkt, st in full_streams.items()
        }

    if "hedge" in active_set:
        cfg_hedge = _tune_grid(
            [dict(eta=eta) for eta in HEDGE_ETAS],
            lambda st, cfg: hedge_weights(
                st["losses"],
                eta=cfg["eta"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
            ),
        )
        selected_hyperparams["hedge"] = cfg_hedge
        weights_map["hedge"] = {
            mkt: hedge_weights(
                st["losses"],
                eta=cfg_hedge["eta"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
            )
            for mkt, st in full_streams.items()
        }

    if "hedge_fixed_share" in active_set:
        cfg_hedge_fs = _tune_grid(
            [
                dict(eta=eta, alpha=alpha)
                for eta in (2.0, 4.0, 8.0)
                for alpha in (1.0 / 63.0, 1.0 / 21.0)
            ],
            lambda st, cfg: hedge_weights(
                st["losses"],
                eta=cfg["eta"],
                fixed_share_alpha=cfg["alpha"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
            ),
        )
        selected_hyperparams["hedge_fixed_share"] = cfg_hedge_fs
        weights_map["hedge_fixed_share"] = {
            mkt: hedge_weights(
                st["losses"],
                eta=cfg_hedge_fs["eta"],
                fixed_share_alpha=cfg_hedge_fs["alpha"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
            )
            for mkt, st in full_streams.items()
        }

    if "adanormalhedge" in active_set:
        weights_map["adanormalhedge"] = {
            mkt: adanormalhedge_weights(
                st["losses"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
            )
            for mkt, st in full_streams.items()
        }

    if "regime_weights" in active_set:
        cfg_regime = _tune_grid(
            [dict(tau=tau, eta=eta) for tau in REGIME_TAUS for eta in (2.0, 4.0)],
            lambda st, cfg: regime_conditioned_weights(
                st["ics"],
                st["target_dispersion"],
                window=10,
                tau=cfg["tau"],
                eta=cfg["eta"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
                min_history=15,
                observable_is_target_based=True,
            )[0],
        )
        selected_hyperparams["regime_weights"] = cfg_regime
        weights_map["regime_weights"] = {
            mkt: regime_conditioned_weights(
                st["ics"],
                st["target_dispersion"],
                window=10,
                tau=cfg_regime["tau"],
                eta=cfg_regime["eta"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
                min_history=15,
                observable_is_target_based=True,
            )[0]
            for mkt, st in full_streams.items()
        }

    abs_exposure_map: dict[str, np.ndarray] = {}
    abs_diag_eval: dict[str, Any] = {}
    if "session_abstention" in active_set:
        w_ra_pre = {
            mkt: rank_average_weights(len(st["times"]), n_models)
            for mkt, st in pre_streams.items()
        }
        ra_ics_pre: dict[str, np.ndarray] = {}
        for mkt, st in pre_streams.items():
            ra_ics_pre[mkt] = np.array(
                [
                    _rank_ic_prepped(
                        combine_session_scores(st["z_sessions"][i], w_ra_pre[mkt][i]),
                        st["y_ranks"][i][0],
                        st["y_ranks"][i][1],
                    )
                    if st["scored"][i] and st["y_ranks"][i] is not None
                    else float("nan")
                    for i in range(len(st["times"]))
                ],
                dtype=np.float64,
            )

        best_abs_cfg: dict[str, Any] | None = None
        best_abs_util = -1e9
        for thr in ABSTENTION_THRESHOLDS:
            e_vals: list[float] = []
            u_vals: list[float] = []
            for mkt, st in pre_streams.items():
                exp_vec = evaluate_session_abstention(
                    ra_ics_pre[mkt],
                    window=21,
                    threshold=thr,
                    session_times=st["times"],
                    session_matures=st["matures"],
                    maturity_lag=maturity_lag,
                    cost_per_session=abstention_cost,
                )["exposure"]
                for i, sp in enumerate(st["split_tags"]):
                    if sp not in tune_splits:
                        continue
                    if sp == "calibration" and (mkt, int(st["times"][i])) in cal_embargo:
                        continue
                    ic_i = float(ra_ics_pre[mkt][i])
                    if not math.isfinite(ic_i):
                        continue
                    e_i = float(exp_vec[i])
                    e_vals.append(e_i)
                    u_vals.append(e_i * (ic_i - abstention_cost))
            cov_cal = float(np.mean(e_vals)) if e_vals else 1.0
            util_cal = float(np.mean(u_vals)) if u_vals else -2.0
            if cov_cal < MIN_CALL_COVERAGE:
                continue
            if util_cal > best_abs_util + 1e-12:
                best_abs_util = util_cal
                best_abs_cfg = dict(
                    window=21,
                    threshold=thr,
                    calibration_coverage=cov_cal,
                    calibration_net_utility=util_cal,
                )
        if best_abs_cfg is None:
            u_always = [
                float(ra_ics_pre[mkt][i]) - abstention_cost
                for mkt, st in pre_streams.items()
                for i, sp in enumerate(st["split_tags"])
                if sp in tune_splits
                and not (sp == "calibration" and (mkt, int(st["times"][i])) in cal_embargo)
                and math.isfinite(float(ra_ics_pre[mkt][i]))
            ]
            best_abs_cfg = dict(
                window=21,
                threshold=-2.0,
                calibration_coverage=1.0,
                calibration_net_utility=float(np.mean(u_always)) if u_always else None,
            )
        selected_hyperparams["session_abstention"] = best_abs_cfg

        for mkt, st in full_streams.items():
            w_ra = weights_map["rank_average"][mkt]
            ra_ics = np.array(
                [
                    _rank_ic_prepped(
                        combine_session_scores(st["z_sessions"][i], w_ra[i]),
                        st["y_ranks"][i][0],
                        st["y_ranks"][i][1],
                    )
                    if st["scored"][i] and st["y_ranks"][i] is not None
                    else float("nan")
                    for i in range(len(st["times"]))
                ],
                dtype=np.float64,
            )
            res_abs = evaluate_session_abstention(
                ra_ics,
                window=best_abs_cfg["window"],
                threshold=best_abs_cfg["threshold"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                cost_per_session=abstention_cost,
            )
            abs_exposure_map[mkt] = res_abs["exposure"]
            eval_mask = np.array([sp == "evaluation" for sp in st["split_tags"]], dtype=bool)
            if eval_mask.any():
                e_ev = res_abs["exposure"][eval_mask]
                ic_ev = ra_ics[eval_mask]
                valid_ev = np.isfinite(ic_ev)
                e_v, ic_v = e_ev[valid_ev], ic_ev[valid_ev]
                cov = float(np.mean(e_v)) if e_v.size else 0.0
                abs_diag_eval[mkt] = dict(
                    coverage=cov,
                    eligible_coverage=bool(cov >= MIN_CALL_COVERAGE),
                    unconditional_ic=float(np.mean(e_v * ic_v)) if e_v.size else None,
                    active_ic=(
                        float(np.sum(e_v * ic_v) / np.sum(e_v))
                        if e_v.size and float(np.sum(e_v)) > 0.0
                        else None
                    ),
                    net_utility=float(np.mean(e_v * (ic_v - abstention_cost)))
                    if e_v.size
                    else None,
                    abstained_sessions=int(np.sum(e_v == 0.0)),
                )
        weights_map["session_abstention"] = weights_map["rank_average"]

    # Evaluar los agregadores pedidos y contrastes pareados
    best_ind_eval_ics = individual_ics[best_model_name]["evaluation"]
    best_ind_cal_ics = individual_ics[best_model_name]["calibration"]

    agg_results: dict[str, Any] = {}
    ra_eval_ics: dict[tuple[str, int], float] = {}
    ra_cal_ics: dict[tuple[str, int], float] = {}

    for agg_name in active_aggs:
        exp_map = abs_exposure_map if agg_name == "session_abstention" else None
        per_split_summary = {}
        per_split_ics = {}
        per_split_predictions = {}
        eval_pred = None
        eval_w = None
        eval_abst = None
        for sp in splits:
            pred_sp, w_sp, abst_sp, exp_sp = _emit_split_predictions_from_weights(
                aligned_splits, full_streams, weights_map[agg_name], sp, exposure_by_market=exp_map
            )
            ics_sp = session_rank_ic(
                aligned_splits[sp],
                pred_sp,
                min_assets=min_assets,
                abstained=abst_sp,
                exposure=exp_sp if exp_map is not None else None,
                _target_cache=target_caches[sp],
            )
            per_split_ics[sp] = ics_sp
            per_split_summary[sp] = summarize_session_dict(ics_sp)
            per_split_predictions[sp] = pred_sp
            if sp == "evaluation":
                eval_pred, eval_w, eval_abst = pred_sp, w_sp, abst_sp
        if agg_name == "rank_average":
            ra_eval_ics = per_split_ics["evaluation"]
            ra_cal_ics = per_split_ics["calibration"]

        mean_eval_weights = {model_names[m]: float(np.mean(eval_w[:, m])) for m in range(n_models)}
        entry = dict(
            summary=per_split_summary,
            contrasts=dict(
                calibration=dict(
                    vs_best_individual=paired_contrast(
                        per_split_ics["calibration"], best_ind_cal_ics
                    ),
                    vs_rank_average=paired_contrast(per_split_ics["calibration"], ra_cal_ics),
                ),
                evaluation=dict(
                    vs_best_individual=paired_contrast(
                        per_split_ics["evaluation"], best_ind_eval_ics
                    ),
                    vs_rank_average=paired_contrast(per_split_ics["evaluation"], ra_eval_ics),
                ),
            ),
            mean_evaluation_weights=mean_eval_weights,
            evaluation_prediction=eval_pred,
            evaluation_weights=eval_w,
            evaluation_abstained=eval_abst,
            evaluation_session_ics=per_split_ics["evaluation"],
            # Predicción emitida en cada tramo, en el orden de las filas alineadas del tramo.
            split_predictions=per_split_predictions,
            split_session_ics=per_split_ics,
        )
        if agg_name == "session_abstention":
            entry["abstention_diagnostics"] = abs_diag_eval
        agg_results[agg_name] = entry

    # Abstención selectiva intrasesión por activo en evaluación (con control |z| y placebo)
    asset_abstention_eval: dict[str, Any] | None = None
    if include_asset_abstention:
        eval_preds_by_sess = [
            aligned_splits["evaluation"]["predictions"][s:e]
            for _, _, s, e in aligned_splits["evaluation"]["sessions"]
        ]
        eval_targs_by_sess = [
            aligned_splits["evaluation"]["target"][s:e]
            for _, _, s, e in aligned_splits["evaluation"]["sessions"]
        ]
        asset_abstention_eval = evaluate_asset_abstention(
            eval_preds_by_sess,
            eval_targs_by_sess,
            keep_fraction=0.5,
            seed=placebo_seed,
            min_assets=min_assets,
        )

    # 4) Placebos obligatorios: (a) pesos barajados en el tiempo, (b) etiquetas maduras permutadas entre expertos
    placebos: dict[str, Any] = {}
    if include_placebos and cfg_ridge_cov_active is not None:
        rng = np.random.default_rng(placebo_seed)
        shf_weights_map: dict[str, np.ndarray] = {}
        perm_weights_map: dict[str, np.ndarray] = {}
        for mkt, st in full_streams.items():
            w_src = weights_map["ridge_cov_active"][mkt].copy()
            eval_idxs = np.flatnonzero([sp == "evaluation" for sp in st["split_tags"]])
            if eval_idxs.size > 1:
                w_src[eval_idxs] = w_src[rng.permutation(eval_idxs)]
            shf_weights_map[mkt] = w_src

            # Permutar columnas de IC maduro por sesión (destruye qué modelo acertó preservando covarianza)
            perm_ics = st["ics"].copy()
            for row_i in range(perm_ics.shape[0]):
                perm_ics[row_i] = perm_ics[row_i, rng.permutation(n_models)]
            perm_weights_map[mkt] = ridge_stacking_weights(
                perm_ics,
                st["covs"],
                lam=cfg_ridge_cov_active["lam"],
                half_life=cfg_ridge_cov_active["half_life"],
                session_times=st["times"],
                session_matures=st["matures"],
                maturity_lag=maturity_lag,
                scored=st["scored"],
                use_current_cov=True,
                se_shrinkage=1.0,
            )

        for p_name, p_map in (
            ("placebo_shuffled_weights", shf_weights_map),
            ("placebo_permuted_labels", perm_weights_map),
        ):
            p_pred, p_w, _, _ = _emit_split_predictions_from_weights(
                aligned_splits, full_streams, p_map, "evaluation"
            )
            p_ics = session_rank_ic(
                aligned_splits["evaluation"],
                p_pred,
                min_assets=min_assets,
                _target_cache=target_caches["evaluation"],
            )
            placebos[p_name] = dict(
                summary=summarize_session_dict(p_ics),
                contrasts=dict(
                    vs_best_individual=paired_contrast(p_ics, best_ind_eval_ics),
                    vs_rank_average=paired_contrast(p_ics, ra_eval_ics),
                    vs_ridge_cov_active=paired_contrast(
                        p_ics, agg_results["ridge_cov_active"]["evaluation_session_ics"]
                    ),
                ),
                mean_evaluation_weights={
                    model_names[m]: float(np.mean(p_w[:, m])) for m in range(n_models)
                },
            )

    return dict(
        schema_version=1,
        kind="online_aggregator_report",
        model_names=model_names,
        splits=list(splits),
        maturity_lag=maturity_lag,
        min_assets=min_assets,
        align=align,
        alignment=alignment_info,
        operable=operable_info,
        best_individual=dict(
            model=best_model_name,
            selection_split=selection_split,
            decision_ic=decision_scores[best_model_name],
            decision_scores=decision_scores,
        ),
        selected_hyperparams=selected_hyperparams,
        individuals=individual_summary,
        aggregators=agg_results,
        asset_abstention_evaluation=asset_abstention_eval,
        placebos=placebos,
        aligned=aligned_splits,
    )


# ---------------------------------------------------------------------------
# Serialización JSON / Markdown y CLI
# ---------------------------------------------------------------------------


def _json_safe_report(result: dict[str, Any]) -> dict[str, Any]:
    """Elimina arrays pesados internos para guardar un `report.json` limpio."""
    doc = {k: v for k, v in result.items() if k not in ("aligned", "aggregators")}
    doc["aggregators"] = {
        name: {
            k: v
            for k, v in entry.items()
            if k
            not in (
                "evaluation_prediction",
                "evaluation_weights",
                "evaluation_abstained",
                "evaluation_session_ics",
                "split_predictions",
                "split_session_ics",
            )
        }
        for name, entry in result["aggregators"].items()
    }
    return doc


def _fmt(val: float | None, digits: int = 4) -> str:
    return "—" if val is None else f"{val:.{digits}f}".replace(".", ",")


def render_markdown(doc: dict[str, Any]) -> str:
    """Genera un informe Markdown reproducible con métricas por mercado y contrastes HAC."""
    lines = [
        "# Informe de agregación causal en línea",
        "",
        f"- Modelos: {', '.join(f'`{m}`' for m in doc['model_names'])}",
        f"- Mejor componente individual (elegido en `{doc['best_individual']['selection_split']}`, nunca en evaluación): "
        f"`{doc['best_individual']['model']}` (IC decisión {_fmt(doc['best_individual']['decision_ic'])})",
        f"- Retardo de maduración mínimo: `{doc['maturity_lag']}` sesión(es) | Mínimo de activos: `{doc['min_assets']}` | "
        f"Modo operable: `{'activado' if doc['operable']['enabled'] else 'apagado'}`",
        "",
        "## 1. Modelos individuales (IC de rangos por sesión)",
        "",
        "| Modelo | Validación | Calibración | Evaluación | ICIR eval. | t NW eval. |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in doc["model_names"]:
        ind = doc["individuals"][name]
        v_m = ind.get("validation", {}).get("all", {}).get("mean")
        c_m = ind["calibration"]["all"]["mean"]
        e_a = ind["evaluation"]["all"]
        lines.append(
            f"| `{name}` | {_fmt(v_m)} | {_fmt(c_m)} | {_fmt(e_a['mean'])} | "
            f"{_fmt(e_a['icir'], 2)} | {_fmt(e_a['t_nw'], 2)} |"
        )

    lines += [
        "",
        "## 2. Agregadores causales en evaluación y contrastes pareados HAC (Newey-West)",
        "",
        "| Agregador | IC eval. | ICIR | vs mejor individual (Δ / t HAC) | vs media rangos (Δ / t HAC) | Pesos medios eval. |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, entry in doc["aggregators"].items():
        ev = entry["summary"]["evaluation"]["all"]
        c_best = entry["contrasts"]["evaluation"]["vs_best_individual"]
        c_ra = entry["contrasts"]["evaluation"]["vs_rank_average"]
        w_str = ", ".join(f"{m}:{_fmt(w, 2)}" for m, w in entry["mean_evaluation_weights"].items())
        lines.append(
            f"| `{name}` | {_fmt(ev['mean'])} | {_fmt(ev['icir'], 2)} | "
            f"{_fmt(c_best['mean'])} ({_fmt(c_best['t_hac'], 2)}) | "
            f"{_fmt(c_ra['mean'])} ({_fmt(c_ra['t_hac'], 2)}) | {w_str} |"
        )

    if doc.get("placebos"):
        lines += [
            "",
            "## 3. Placebos obligatorios en evaluación",
            "",
            "| Placebo | IC eval. | vs mejor ind. (Δ / t HAC) | vs `ridge_cov_active` (Δ / t HAC) |",
            "| --- | ---: | ---: | ---: |",
        ]
        for p_name, p_entry in doc["placebos"].items():
            ev = p_entry["summary"]["all"]
            cb = p_entry["contrasts"]["vs_best_individual"]
            cr = p_entry["contrasts"]["vs_ridge_cov_active"]
            lines.append(
                f"| `{p_name}` | {_fmt(ev['mean'])} | {_fmt(cb['mean'])} ({_fmt(cb['t_hac'], 2)}) | "
                f"{_fmt(cr['mean'])} ({_fmt(cr['t_hac'], 2)}) |"
            )

    aa = doc.get("asset_abstention_evaluation")
    if aa is not None:
        lines += [
            "",
            "## 4. Abstención selectiva por activo en evaluación (cobertura emparejada)",
            "",
            f"- Cobertura efectiva: `{_fmt(100.0 * aa['coverage'], 1)} %` ({aa['sessions']} sesiones)",
            f"- IC activo señal/desacuerdo: `{_fmt(aa['active_ic_signal_to_disagreement'])}` vs control magnitud `|z|`: "
            f"`{_fmt(aa['active_ic_magnitude_control'])}` vs placebo aleatorio: `{_fmt(aa['active_ic_random_placebo'])}`",
            f"- IC sesión completa (abstenidos=0): señal/desacuerdo `{_fmt(aa['full_session_ic_signal_to_disagreement'])}`, "
            f"magnitud `{_fmt(aa['full_session_ic_magnitude_control'])}`, base sin abstener `{_fmt(aa['base_full_ic'])}`",
            "",
        ]
    return "\n".join(lines)


def _parse_model_arg(item: str) -> tuple[str, Path]:
    _require("=" in item, f"--model debe tener formato NOMBRE=RUTA; se recibió {item!r}")
    name, raw_path = item.split("=", 1)
    name = name.strip()
    _require(bool(name) and bool(raw_path.strip()), f"--model inválido: {item!r}")
    return name, Path(raw_path.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Agregación causal en línea sobre emisiones Parquet con contrastes HAC Newey-West."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Par NOMBRE=CARPETA_EJECUCION (contiene {validation,calibration,evaluation}-predictions.parquet). Repetible.",
    )
    parser.add_argument(
        "--selection-split",
        choices=("calibration", "validation", "pooled"),
        default="calibration",
        help="Tramo usado para elegir mejor componente e hiperparámetros (nunca evaluación). Por defecto: calibration.",
    )
    parser.add_argument(
        "--maturity-lag",
        type=int,
        default=1,
        help="Retardo mínimo de maduración en sesiones si el parquet no trae matures_at (por defecto: 1).",
    )
    parser.add_argument(
        "--min-assets",
        type=int,
        default=MIN_ASSETS,
        help=f"Activos mínimos por sesión para calcular IC (por defecto: {MIN_ASSETS}).",
    )
    parser.add_argument(
        "--align",
        choices=("strict", "inner"),
        default="strict",
        help="strict exige muestras idénticas; inner toma intersección exacta por (market, prediction_at, sample_id).",
    )
    parser.add_argument(
        "--operable",
        action="store_true",
        help="Activar objetivo operable (y + alpha vs k*p + alpha, con k elegido solo en calibración). Apagado por defecto.",
    )
    parser.add_argument(
        "--alpha-col",
        default="alpha",
        help="Nombre de la columna de desplazamiento conocido alpha cuando --operable está activo.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Ejecutar solo con calibration y evaluation (si un órgano de memoria no emite validation).",
    )
    parser.add_argument(
        "--placebo-seed",
        type=int,
        default=20261122,
        help="Semilla determinista para placebos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Carpeta de salida donde se escriben report.json y report.md.",
    )
    args = parser.parse_args(argv)
    model_runs = dict(_parse_model_arg(item) for item in args.model)
    splits = ("calibration", "evaluation") if args.no_validation else SPLITS

    result = run_aggregation(
        model_runs,
        splits=splits,
        selection_split=args.selection_split,
        maturity_lag=args.maturity_lag,
        min_assets=args.min_assets,
        align=args.align,
        operable=args.operable,
        alpha_col=args.alpha_col,
        placebo_seed=args.placebo_seed,
    )
    doc = _json_safe_report(result)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "report.md").write_text(render_markdown(doc), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
