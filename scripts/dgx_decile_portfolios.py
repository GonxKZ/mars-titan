#!/usr/bin/env python3
"""Evaluación económica de predicciones transversales diarias por deciles.

Este módulo construye carteras transversales por grupos de rango (deciles por defecto)
sin utilizar información futura y calcula rendimientos largo-corto, rotación por pata,
costes de equilibrio, Sharpe bruto y neto, estadísticos t (simple y Newey-West) y
monotonía entre grupos.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("session", "asset", "prediction", "target")
OPTIONAL_COLUMNS: tuple[str, ...] = ("liquidity",)
MIN_ASSETS_PER_SESSION: int = 30
TRADING_DAYS_PER_YEAR: float = 252.0
BPS_PER_UNIT: float = 10_000.0
VALID_WEIGHTINGS: tuple[str, ...] = ("equal", "liquidity")


def _validate_int_param(value: Any, name: str, minimum: int) -> int:
    """Valida un parámetro entero (rechaza booleanos) con cota inferior."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"El parámetro '{name}' debe ser un entero, recibido: {value!r}.")
    int_value = int(value)
    if int_value < minimum:
        raise ValueError(
            f"El parámetro '{name}' debe ser mayor o igual que {minimum}, recibido: {int_value}."
        )
    return int_value


def _validate_costs_bps(costs_bps: Iterable[float]) -> tuple[float, ...]:
    """Valida los costes en puntos básicos (finitos y no negativos)."""
    if isinstance(costs_bps, (str, bytes)):
        raise ValueError("El parámetro 'costs_bps' debe ser una secuencia numérica.")
    try:
        items = list(costs_bps)
    except TypeError as exc:
        raise ValueError("El parámetro 'costs_bps' debe ser iterable.") from exc

    validated: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float, np.number)):
            raise ValueError(f"Cada coste en 'costs_bps' debe ser numérico, recibido: {item!r}.")
        val = float(item)
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(
                f"Cada coste en 'costs_bps' debe ser finito y >= 0, recibido: {item!r}."
            )
        validated.append(val)
    return tuple(validated)


def _cost_key(cost_bps: float) -> int | float:
    """Normaliza la clave numérica del diccionario de Sharpe neto."""
    return int(cost_bps) if float(cost_bps).is_integer() else float(cost_bps)


def _load_source(frame: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Carga un DataFrame desde memoria o desde un archivo Parquet o CSV."""
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if isinstance(frame, (str, Path)):
        path = Path(frame)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de entrada: {path}")
        suffix = path.suffix.lower()
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(path, engine="pyarrow")
        if suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(
            f"Formato de archivo no soportado '{path.suffix}'. Use '.parquet' o '.csv'."
        )
    raise ValueError(
        "La entrada debe ser un pandas.DataFrame o la ruta a un archivo '.parquet' o '.csv'."
    )


def _validate_session_column(series: pd.Series) -> pd.Series:
    """Valida y normaliza la columna 'session' como entero o fecha sin nulos."""
    if series.isna().any():
        raise ValueError("La columna 'session' contiene valores nulos o no finitos.")
    if len(series) == 0:
        return series
    if pd.api.types.is_bool_dtype(series):
        raise ValueError("La columna 'session' no puede ser booleana; debe ser entero o fecha.")
    if pd.api.types.is_integer_dtype(series):
        return series.astype("int64")
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_float_dtype(series):
        raise ValueError("La columna 'session' no puede ser flotante; debe ser entero o fecha.")

    # Tipos fecha o marca temporal de PyArrow (p. ej. date32[day][pyarrow]).
    dtype_str = str(series.dtype).lower()
    if "date" in dtype_str or "timestamp" in dtype_str:
        return pd.to_datetime(series)

    # Cadenas (por ejemplo fechas leídas desde CSV) o valores Python homogéneos.
    if pd.api.types.is_string_dtype(series):
        try:
            parsed = pd.to_datetime(series, format="ISO8601")
        except (ValueError, TypeError):
            try:
                parsed = pd.to_datetime(series)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "La columna 'session' contiene texto no interpretable como fecha."
                ) from exc
        if parsed.isna().any():
            raise ValueError("La columna 'session' contiene fechas inválidas.")
        return parsed

    values = series.tolist()
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in values):
        return pd.Series([int(v) for v in values], index=series.index, dtype="int64")
    if all(isinstance(v, (dt.date, dt.datetime, pd.Timestamp)) for v in values):
        return pd.to_datetime(series)

    raise ValueError(
        "La columna 'session' debe contener enteros o fechas homogéneas (sin mezclar tipos)."
    )


def _validate_asset_column(series: pd.Series) -> pd.Series:
    """Valida que la columna 'asset' sea texto sin nulos ni cadenas vacías."""
    if series.isna().any():
        raise ValueError("La columna 'asset' contiene valores nulos.")
    if len(series) == 0:
        return series.astype("string")
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        raise ValueError("La columna 'asset' debe contener texto (cadenas de caracteres).")
    if not isinstance(series.dtype, pd.StringDtype):
        values = series.tolist()
        if not all(isinstance(v, str) for v in values):
            raise ValueError("Todos los valores de la columna 'asset' deben ser texto (str).")
    as_str = series.astype("string")
    if (as_str == "").any():
        raise ValueError("La columna 'asset' no puede contener cadenas vacías.")
    return as_str


def _validate_float_column(series: pd.Series, name: str) -> pd.Series:
    """Valida que una columna numérica sea finita (sin NaN ni infinitos)."""
    if pd.api.types.is_bool_dtype(series):
        raise ValueError(f"La columna '{name}' no puede ser booleana; debe ser numérica.")
    if len(series) == 0:
        return pd.Series(dtype="float64", index=series.index, name=name)
    if series.dtype == object and any(isinstance(v, bool) for v in series.tolist()):
        raise ValueError(f"La columna '{name}' contiene booleanos; debe ser numérica.")
    try:
        numeric = pd.to_numeric(series, errors="raise").astype("float64")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"La columna '{name}' debe ser numérica y convertible a float.") from exc
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"La columna '{name}' contiene valores no finitos (NaN, +inf o -inf).")
    return numeric


def _prepare_frame(
    frame: pd.DataFrame | str | Path,
    require_positive_liquidity: bool = False,
) -> pd.DataFrame:
    """Carga, valida y ordena canónicamente el DataFrame de entrada."""
    df = _load_source(frame)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en los datos: {', '.join(missing)}.")

    has_liquidity = "liquidity" in df.columns
    if require_positive_liquidity and not has_liquidity:
        raise ValueError("La ponderación por liquidez exige que exista la columna 'liquidity'.")

    cols = list(REQUIRED_COLUMNS) + (["liquidity"] if has_liquidity else [])
    clean = pd.DataFrame(index=df.index)
    clean["session"] = _validate_session_column(df["session"])
    clean["asset"] = _validate_asset_column(df["asset"])
    clean["prediction"] = _validate_float_column(df["prediction"], "prediction")
    clean["target"] = _validate_float_column(df["target"], "target")
    if has_liquidity:
        clean["liquidity"] = _validate_float_column(df["liquidity"], "liquidity")
        if require_positive_liquidity and (clean["liquidity"] <= 0.0).any():
            raise ValueError(
                "La ponderación por liquidez exige valores estrictamente positivos (liquidity > 0)."
            )

    if clean.duplicated(subset=["session", "asset"]).any():
        raise ValueError(
            "Existen activos duplicados ('asset') dentro de una misma sesión ('session')."
        )

    # Orden canónico determinista por sesión y activo (sin información futura).
    clean = clean[cols].sort_values(
        by=["session", "asset"], ascending=[True, True], kind="mergesort"
    )
    return clean.reset_index(drop=True)


def _assign_rank_groups(df: pd.DataFrame, groups: int) -> pd.DataFrame:
    """Asigna grupos 1..groups por rango medio de 'prediction' en cada sesión válida."""
    min_assets = max(MIN_ASSETS_PER_SESSION, groups)
    if df.empty:
        out = df.copy()
        out["group"] = pd.Series(dtype="int64")
        return out

    counts = df.groupby("session", sort=False)["asset"].transform("size")
    valid = df.loc[counts >= min_assets].copy()
    if valid.empty:
        valid["group"] = pd.Series(dtype="int64")
        return valid.reset_index(drop=True)

    # El rango medio de enteros es siempre un semientero: 2*r es un entero exacto.
    # Grupo = floor(((2*r - 1) * G) / (2*N)) + 1 en {1, ..., G}.
    # Garantiza grupos de tamaño floor(N/G) o ceil(N/G) y evita redondeos flotantes.
    ranks = valid.groupby("session", sort=False)["prediction"].rank(
        method="average", ascending=True
    )
    double_ranks = np.rint(2.0 * ranks.to_numpy(dtype=np.float64)).astype(np.int64)
    sizes = valid.groupby("session", sort=False)["asset"].transform("size").to_numpy(dtype=np.int64)
    group_ids = ((double_ranks - 1) * groups) // (2 * sizes) + 1
    valid["group"] = group_ids.astype(np.int64)
    return valid.reset_index(drop=True)


def deciles(frame: pd.DataFrame | str | Path, groups: int = 10) -> pd.DataFrame:
    """Asigna cada activo a un grupo transversal (1..groups) por sesión.

    Para cada sesión con al menos ``max(30, groups)`` activos, ordena las predicciones
    con rango medio en caso de empate y asigna cada activo al grupo:

        grupo = floor(((2 * rango - 1) * groups) / (2 * N)) + 1

    donde ``rango`` pertenece a ``[1, N]`` y ``N`` es el número de activos de la sesión.
    Las sesiones con menos de 30 activos (o menos de ``groups``) se excluyen.

    Args:
        frame: DataFrame o ruta a archivo ``.parquet`` o ``.csv``.
        groups: Número de grupos (por defecto 10).

    Returns:
        DataFrame ordenado por ``(session, asset)`` con las sesiones válidas y la
        columna entera ``group`` (de ``1`` a ``groups``).
    """
    n_groups = _validate_int_param(groups, "groups", minimum=2)
    clean = _prepare_frame(frame, require_positive_liquidity=False)
    return _assign_rank_groups(clean, n_groups)


def _compute_long_short_from_grouped(
    grouped: pd.DataFrame, groups: int, weighting: str
) -> pd.Series:
    """Calcula el rendimiento largo-corto diario a partir de un DataFrame ya agrupado."""
    if grouped.empty:
        return pd.Series(
            dtype="float64",
            index=pd.Index([], name="session"),
            name="long_short",
        )

    sessions = pd.Index(grouped["session"].drop_duplicates(), name="session")
    long_leg = grouped.loc[grouped["group"] == groups]
    short_leg = grouped.loc[grouped["group"] == 1]

    if weighting == "equal":
        long_ret = long_leg.groupby("session", sort=True)["target"].mean()
        short_ret = short_leg.groupby("session", sort=True)["target"].mean()
    else:
        long_num = (
            (long_leg["target"] * long_leg["liquidity"])
            .groupby(long_leg["session"], sort=True)
            .sum()
        )
        long_den = long_leg.groupby("session", sort=True)["liquidity"].sum()
        long_ret = long_num / long_den

        short_num = (
            (short_leg["target"] * short_leg["liquidity"])
            .groupby(short_leg["session"], sort=True)
            .sum()
        )
        short_den = short_leg.groupby("session", sort=True)["liquidity"].sum()
        short_ret = short_num / short_den

    ls = long_ret.reindex(sessions) - short_ret.reindex(sessions)
    ls.name = "long_short"
    return ls.astype("float64")


def long_short(
    frame: pd.DataFrame | str | Path,
    groups: int = 10,
    weighting: str = "equal",
) -> pd.Series:
    """Calcula el rendimiento diario del grupo más alto menos el del más bajo.

    Args:
        frame: DataFrame o ruta a archivo ``.parquet`` o ``.csv``.
        groups: Número de grupos (por defecto 10).
        weighting: ``"equal"`` para pesos iguales o ``"liquidity"`` para pesos
            proporcionales a la columna ``liquidity`` (exige ``liquidity > 0``).

    Returns:
        Serie indexada por ``session`` (en orden temporal) con el rendimiento
        largo-corto de cada sesión válida.
    """
    n_groups = _validate_int_param(groups, "groups", minimum=2)
    if weighting not in VALID_WEIGHTINGS:
        raise ValueError(
            f"El parámetro 'weighting' debe ser 'equal' o 'liquidity', recibido: {weighting!r}."
        )
    clean = _prepare_frame(frame, require_positive_liquidity=(weighting == "liquidity"))
    grouped = _assign_rank_groups(clean, n_groups)
    return _compute_long_short_from_grouped(grouped, n_groups, weighting)


def _equal_weight_leg_turnover(prev_assets: set[str], curr_assets: set[str]) -> float:
    """Calcula 0.5 * suma(|w_t - w_{t-1}|) para una pata con pesos iguales.

    Si una pata tiene a > 0 activos en t-1 y b > 0 activos en t con c activos en común,
    entonces 0.5 * suma(|w_t - w_{t-1}|) = 1 - c / max(a, b).
    """
    n_prev = len(prev_assets)
    n_curr = len(curr_assets)
    if n_prev == 0 or n_curr == 0:
        return float("nan")
    overlap = len(prev_assets & curr_assets)
    return 1.0 - (overlap / max(n_prev, n_curr))


def _leg_sets_by_session(leg_df: pd.DataFrame) -> dict[Any, set[str]]:
    """Construye el conjunto de activos de una pata para cada sesión."""
    sets_by_session: dict[Any, set[str]] = {}
    for sess, asset in zip(leg_df["session"].tolist(), leg_df["asset"].tolist(), strict=True):
        sets_by_session.setdefault(sess, set()).add(asset)
    return sets_by_session


def _compute_turnover_from_grouped(grouped: pd.DataFrame, groups: int) -> pd.DataFrame:
    """Calcula la rotación por pata y su media a partir de un DataFrame ya agrupado."""
    if grouped.empty:
        return pd.DataFrame(
            columns=["long", "short", "mean"],
            index=pd.Index([], name="session"),
            dtype="float64",
        )

    sessions = grouped["session"].drop_duplicates().tolist()
    long_sets = _leg_sets_by_session(grouped.loc[grouped["group"] == groups])
    short_sets = _leg_sets_by_session(grouped.loc[grouped["group"] == 1])

    n_sessions = len(sessions)
    long_to = np.full(n_sessions, np.nan, dtype=np.float64)
    short_to = np.full(n_sessions, np.nan, dtype=np.float64)

    prev_long = long_sets.get(sessions[0], set())
    prev_short = short_sets.get(sessions[0], set())
    for idx in range(1, n_sessions):
        sess = sessions[idx]
        curr_long = long_sets.get(sess, set())
        curr_short = short_sets.get(sess, set())
        long_to[idx] = _equal_weight_leg_turnover(prev_long, curr_long)
        short_to[idx] = _equal_weight_leg_turnover(prev_short, curr_short)
        prev_long = curr_long
        prev_short = curr_short

    mean_to = 0.5 * (long_to + short_to)
    return pd.DataFrame(
        {"long": long_to, "short": short_to, "mean": mean_to},
        index=pd.Index(sessions, name="session"),
        dtype="float64",
    )


def turnover(frame: pd.DataFrame | str | Path, groups: int = 10) -> pd.DataFrame:
    """Calcula la rotación diaria por pata (con pesos iguales) y su media.

    Para cada sesión válida ``t`` posterior a la primera, la rotación de cada pata es:

        rotación_pata(t) = 0,5 * suma_i |peso_{t,i} - peso_{t-1,i}|

    La primera sesión válida devuelve ``NaN`` al no existir sesión anterior.

    Args:
        frame: DataFrame o ruta a archivo ``.parquet`` o ``.csv``.
        groups: Número de grupos (por defecto 10).

    Returns:
        DataFrame indexado por ``session`` con columnas ``long``, ``short`` y ``mean``.
    """
    n_groups = _validate_int_param(groups, "groups", minimum=2)
    clean = _prepare_frame(frame, require_positive_liquidity=False)
    grouped = _assign_rank_groups(clean, n_groups)
    return _compute_turnover_from_grouped(grouped, n_groups)


def _newey_west_t_stat(values: np.ndarray, lags: int) -> float | None:
    """Calcula el estadístico t de Newey-West (núcleo de Bartlett) de la media.

    Usa el estimador de varianza de largo plazo con pesos de Bartlett
    ``w_l = 1 - l / (lags + 1)`` y corrección de muestra finita ``T / (T - 1)``,
    de modo que con ``lags = 0`` coincide exactamente con el estadístico t simple.
    """
    n = len(values)
    if n < 2:
        return None
    mean = float(np.mean(values))
    residuals = values - mean
    gamma_0 = float(np.dot(residuals, residuals)) / n
    long_run_var = gamma_0

    max_lag = min(lags, n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - (lag / (lags + 1))
        gamma_lag = float(np.dot(residuals[lag:], residuals[:-lag])) / n
        long_run_var += 2.0 * weight * gamma_lag

    if long_run_var <= 0.0 or not math.isfinite(long_run_var):
        return None
    # Con factor T/(T-1), Var(media) = (long_run_var / T) * (T / (T - 1)) = long_run_var / (T - 1).
    se = math.sqrt(long_run_var / (n - 1))
    if se <= 0.0 or not math.isfinite(se):
        return None
    return mean / se


def _spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """Calcula la correlación de Spearman usando rangos medios sin dependencias externas."""
    if len(x) < 2 or len(y) < 2:
        return None
    rx = pd.Series(x, dtype="float64").rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y, dtype="float64").rank(method="average").to_numpy(dtype=np.float64)
    dx = rx - float(np.mean(rx))
    dy = ry - float(np.mean(ry))
    denom = math.sqrt(float(np.dot(dx, dx)) * float(np.dot(dy, dy)))
    if denom <= 0.0 or not math.isfinite(denom):
        return None
    corr = float(np.dot(dx, dy)) / denom
    return max(-1.0, min(1.0, corr))


def _annualized_sharpe(values: np.ndarray) -> float | None:
    """Calcula el ratio de Sharpe anualizado (raíz de 252) con desviación muestral."""
    if len(values) < 2:
        return None
    std = float(np.std(values, ddof=1))
    if std <= 0.0 or not math.isfinite(std):
        return None
    return (float(np.mean(values)) / std) * math.sqrt(TRADING_DAYS_PER_YEAR)


def _filter_top_third_liquidity(clean: pd.DataFrame) -> pd.DataFrame:
    """Restringe cada sesión al tercio más líquido sin usar información futura.

    En una sesión con N activos, toma el umbral igual al elemento en posición
    ``(2 * N) // 3`` de las liquideces ordenadas de menor a mayor, y conserva los
    activos con ``liquidity >= umbral`` (exactamente ceil(N / 3) activos sin empates,
    y de forma determinista si existen empates en el umbral).
    """
    if clean.empty:
        return clean.copy()

    def _session_threshold(liq: pd.Series) -> float:
        arr = np.sort(liq.to_numpy(dtype=np.float64))
        idx = (2 * len(arr)) // 3
        return float(arr[idx])

    thresholds = clean.groupby("session", sort=False)["liquidity"].transform(_session_threshold)
    return clean.loc[clean["liquidity"] >= thresholds].reset_index(drop=True)


def _summarize_clean_frame(
    clean: pd.DataFrame,
    groups: int,
    costs_bps: tuple[float, ...],
    lags: int,
) -> dict[str, Any]:
    """Calcula el diccionario de métricas económicas sobre un DataFrame validado."""
    grouped = _assign_rank_groups(clean, groups)
    ls_series = _compute_long_short_from_grouped(grouped, groups, weighting="equal")
    to_df = _compute_turnover_from_grouped(grouped, groups)

    valid_ls = ls_series.dropna()
    n_sessions = int(len(valid_ls))
    net_sharpe: dict[int | float, float | None] = {_cost_key(c): None for c in costs_bps}

    # Media de target por grupo: media temporal de las medias transversales de cada sesión.
    group_means: dict[int, float] = {}
    if not grouped.empty:
        session_group_means = (
            grouped.groupby(["session", "group"], sort=True)["target"]
            .mean()
            .groupby("group", sort=True)
            .mean()
        )
        group_means = {int(g): float(val) for g, val in session_group_means.items()}

    monotonicity: float | None = None
    if len(group_means) >= 2:
        g_keys = np.array(list(group_means.keys()), dtype=np.float64)
        g_vals = np.array(list(group_means.values()), dtype=np.float64)
        monotonicity = _spearman_rank_corr(g_keys, g_vals)

    if n_sessions == 0:
        return {
            "sessions": 0,
            "mean": None,
            "std": None,
            "t_stat": None,
            "t_newey_west": None,
            "sharpe_gross": None,
            "hit_rate": None,
            "turnover_mean": None,
            "two_leg_turnover_mean": None,
            "breakeven_cost_bps": None,
            "net_sharpe": net_sharpe,
            "group_means": group_means,
            "monotonicity": monotonicity,
        }

    ls_vals = valid_ls.to_numpy(dtype=np.float64)
    mean_ret = float(np.mean(ls_vals))
    std_ret = float(np.std(ls_vals, ddof=1)) if n_sessions >= 2 else None
    t_stat: float | None = None
    if std_ret is not None and std_ret > 0.0 and math.isfinite(std_ret):
        t_stat = (mean_ret / std_ret) * math.sqrt(n_sessions)

    t_nw = _newey_west_t_stat(ls_vals, lags=lags)
    sharpe_gross = _annualized_sharpe(ls_vals)
    hit_rate = float(np.mean(ls_vals > 0.0))

    # Alineamos la rotación a las sesiones con rendimiento largo-corto válido.
    to_aligned = to_df.reindex(valid_ls.index)
    two_leg_to = to_aligned["long"] + to_aligned["short"]
    valid_mean_to = to_aligned["mean"].dropna()
    valid_two_leg_to = two_leg_to.dropna()

    turnover_mean: float | None = float(valid_mean_to.mean()) if len(valid_mean_to) > 0 else None
    two_leg_turnover_mean: float | None = (
        float(valid_two_leg_to.mean()) if len(valid_two_leg_to) > 0 else None
    )

    # Coste de equilibrio en bps por unidad de rotación de las dos patas:
    # cada sesión paga coste * (rotación_larga + rotación_corta).
    breakeven_cost_bps: float | None = None
    if two_leg_turnover_mean is not None and two_leg_turnover_mean > 0.0:
        breakeven_cost_bps = (mean_ret / two_leg_turnover_mean) * BPS_PER_UNIT

    # En la primera sesión no hay rebalanceo previo, por lo que el coste pagado es 0.
    paid_two_leg_to = two_leg_to.fillna(0.0).to_numpy(dtype=np.float64)
    for cost_bps in costs_bps:
        cost_rate = cost_bps / BPS_PER_UNIT
        net_vals = ls_vals - (cost_rate * paid_two_leg_to)
        net_sharpe[_cost_key(cost_bps)] = _annualized_sharpe(net_vals)

    return {
        "sessions": n_sessions,
        "mean": mean_ret,
        "std": std_ret,
        "t_stat": t_stat,
        "t_newey_west": t_nw,
        "sharpe_gross": sharpe_gross,
        "hit_rate": hit_rate,
        "turnover_mean": turnover_mean,
        "two_leg_turnover_mean": two_leg_turnover_mean,
        "breakeven_cost_bps": breakeven_cost_bps,
        "net_sharpe": net_sharpe,
        "group_means": group_means,
        "monotonicity": monotonicity,
    }


def summary(
    frame: pd.DataFrame | str | Path,
    groups: int = 10,
    costs_bps: Sequence[float] = (5, 10, 20),
    lags: int = 10,
) -> dict[str, Any]:
    """Evalúa económicamente las predicciones con carteras por grupos y costes.

    Devuelve un diccionario con métricas de la estrategia largo-corto diaria con
    pesos iguales: número de sesiones, media, desviación muestral, t simple, t de
    Newey-West (núcleo de Bartlett), Sharpe bruto anualizado (raíz de 252), tasa de
    acierto, rotación media diaria (por pata y suma de las dos patas), coste de
    equilibrio en puntos básicos por unidad de rotación de las dos patas, Sharpe neto
    para cada nivel de coste, media de ``target`` por grupo y monotonía (correlación
    de Spearman).

    Si la entrada incluye la columna ``liquidity``, añade la clave
    ``"liquid_top_third"`` con el mismo conjunto de métricas restringido al tercio más
    líquido de cada sesión.

    Args:
        frame: DataFrame o ruta a archivo ``.parquet`` o ``.csv``.
        groups: Número de grupos (por defecto 10).
        costs_bps: Niveles de coste en puntos básicos (por defecto ``(5, 10, 20)``).
        lags: Número de retardos del núcleo de Bartlett para Newey-West (por defecto 10).

    Returns:
        Diccionario serializable en JSON con los resultados de la evaluación.
    """
    n_groups = _validate_int_param(groups, "groups", minimum=2)
    n_lags = _validate_int_param(lags, "lags", minimum=0)
    validated_costs = _validate_costs_bps(costs_bps)

    clean = _prepare_frame(frame, require_positive_liquidity=False)
    result = _summarize_clean_frame(clean, n_groups, validated_costs, n_lags)

    if "liquidity" in clean.columns:
        liquid_clean = _filter_top_third_liquidity(clean)
        result["liquid_top_third"] = _summarize_clean_frame(
            liquid_clean, n_groups, validated_costs, n_lags
        )

    return result


def _build_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos de la interfaz de línea de órdenes."""
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa económicamente predicciones transversales diarias mediante "
            "carteras por deciles, rotación y costes."
        )
    )
    parser.add_argument(
        "input",
        help="Ruta al archivo de entrada (.parquet o .csv).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta del archivo JSON de salida.",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=10,
        help="Número de grupos por sesión (por defecto: 10).",
    )
    parser.add_argument(
        "--costs",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 20.0],
        help="Costes en puntos básicos por unidad de rotación (por defecto: 5 10 20).",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=10,
        help="Retardos de Bartlett para el t de Newey-West (por defecto: 10).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de la interfaz de línea de órdenes."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = summary(
            args.input,
            groups=args.groups,
            costs_bps=tuple(args.costs),
            lags=args.lags,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
