#!/usr/bin/env python3
"""Estabilidad del IC por sesión de los modelos de una campaña DGX.

La media del IC y su t dicen cuánto ordena un modelo, no cómo se reparte ese acierto en el
tiempo. Esta herramienta lee la tabla de IC por sesión que escribe el informe de la criba con
`--inference` (`sessions.csv`) y mide, para cada modelo: el ICIR, la proporción de sesiones
y de meses con IC positivo, el peor mes, la caída máxima del IC acumulado, la racha más larga
de sesiones sin acierto, la peor ventana de sesiones consecutivas y la diferencia entre las dos
mitades del tramo. Con más de un mercado mide además cada uno por separado.

    dgx_ic_stability.py sessions.csv [--window 21] [--json salida.json]

El informe de la criba la carga si está junto a él y añade su tabla. Solo usa numpy y la
biblioteca estándar.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__: Final[tuple[str, ...]] = (
    "longest_negative_run",
    "main",
    "markdown",
    "max_drawdown",
    "month_of",
    "read_sessions",
    "stability",
    "stability_table",
    "worst_window",
)

_EPOCH_UTC: Final[datetime] = datetime(1970, 1, 1, tzinfo=UTC)
_NS_PER_SECOND: Final[int] = 1_000_000_000
_US_THRESHOLD: Final[int] = 10**14
_MS_THRESHOLD: Final[int] = 10**11
_SEC_THRESHOLD: Final[int] = 10**8
_NS_THRESHOLD: Final[int] = 10**17
_INT_RE: Final[re.Pattern[str]] = re.compile(r"^[+-]?\d+$")
_MONTH_RE: Final[re.Pattern[str]] = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")
_FRAC_SEC_RE: Final[re.Pattern[str]] = re.compile(r"[Tt ]\d{2}:\d{2}:\d{2}[.,](\d+)")

_MD_HEADERS: Final[tuple[str, ...]] = (
    "Modelo",
    "Ámbito",
    "Sesiones",
    "IC medio",
    "ICIR",
    "% sesiones positivas",
    "% meses positivos",
    "Peor mes",
    "Caída máxima del IC acumulado",
    "Racha negativa más larga",
    "Peor ventana",
)
_MD_ALIGNMENTS: Final[tuple[str, ...]] = (
    "---",
    "---",
    "---:",
    "---:",
    "---:",
    "---:",
    "---:",
    "---",
    "---:",
    "---:",
    "---:",
)


def _parse_instant(instant: str) -> tuple[int, str]:
    """Convierte un instante textual en (nanosegundos_utc, mes_utc «YYYY-MM»)."""
    if not isinstance(instant, str):
        raise ValueError("El instante debe ser una cadena de texto.")
    text = instant.strip()
    if not text:
        raise ValueError("El instante no puede estar vacío.")

    if _INT_RE.fullmatch(text):
        raw = int(text)
        magnitude = abs(raw)
        if magnitude >= _NS_THRESHOLD:
            ns = raw
        elif magnitude >= _US_THRESHOLD:
            ns = raw * 1_000
        elif magnitude >= _MS_THRESHOLD:
            ns = raw * 1_000_000
        elif magnitude >= _SEC_THRESHOLD:
            ns = raw * _NS_PER_SECOND
        else:
            ns = raw
        seconds, _ = divmod(ns, _NS_PER_SECOND)
        try:
            dt_utc = _EPOCH_UTC + timedelta(seconds=seconds)
        except OverflowError as exc:
            raise ValueError(f"Instante de nanosegundos fuera de rango: {instant!r}.") from exc
        return ns, f"{dt_utc.year:04d}-{dt_utc.month:02d}"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Formato de instante no válido: {instant!r}.") from exc

    dt_utc = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)

    # Extrae los dígitos 7..9 de la fracción de segundos para conservar precisión de nanosegundos.
    extra_ns = 0
    frac_match = _FRAC_SEC_RE.search(text)
    if frac_match is not None:
        sub_micro = frac_match.group(1)[6:9]
        if sub_micro:
            extra_ns = int(sub_micro.ljust(3, "0"))

    delta = dt_utc - _EPOCH_UTC
    total_seconds = delta.days * 86_400 + delta.seconds
    ns = total_seconds * _NS_PER_SECOND + delta.microseconds * 1_000 + extra_ns
    return ns, f"{dt_utc.year:04d}-{dt_utc.month:02d}"


def _as_1d_finite_array(values: Sequence[float] | np.ndarray, *, allow_empty: bool) -> np.ndarray:
    """Valida y convierte una secuencia o array 1D en un ndarray float64 finito."""
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, (Sequence, np.ndarray)
    ):
        raise ValueError("values debe ser una secuencia o un array 1D de reales finitos.")

    if isinstance(values, np.ndarray):
        if values.dtype == np.bool_ or np.iscomplexobj(values):
            raise ValueError("values debe contener únicamente números reales finitos.")
        try:
            arr = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("values debe contener únicamente números reales finitos.") from exc
    else:
        if any(isinstance(item, (bool, np.bool_)) for item in values):
            raise ValueError("values no admite valores booleanos.")
        try:
            arr = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("values debe contener únicamente números reales finitos.") from exc

    if arr.ndim != 1:
        raise ValueError("values debe ser unidimensional (1D).")
    if not allow_empty and arr.size == 0:
        raise ValueError("values no puede estar vacío.")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("values contiene valores no finitos (NaN o infinito).")
    return arr


def _validate_window(window: int) -> int:
    """Comprueba que el tamaño de ventana sea un entero estrictamente positivo."""
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)):
        raise ValueError("El tamaño de ventana debe ser un número entero positivo.")
    win = int(window)
    if win < 1:
        raise ValueError("El tamaño de ventana debe ser mayor o igual que 1.")
    return win


def _validate_months(months: Sequence[str], expected_len: int) -> list[str]:
    """Comprueba que months sea una secuencia de cadenas «YYYY-MM» con la longitud esperada."""
    if isinstance(months, (str, bytes, bytearray)) or not isinstance(months, Sequence):
        raise ValueError("months debe ser una secuencia de cadenas con formato «YYYY-MM».")
    if len(months) != expected_len:
        raise ValueError(
            f"months debe tener la misma longitud que values ({expected_len}), "
            f"pero tiene {len(months)}."
        )
    validated: list[str] = []
    for month in months:
        if not isinstance(month, str) or not _MONTH_RE.fullmatch(month):
            raise ValueError(f"Mes no válido (se esperaba «YYYY-MM»): {month!r}.")
        validated.append(month)
    return validated


def _format_float(value: float | None) -> str:
    """Formatea un número real con coma decimal y cuatro decimales, o «—» si es None."""
    if value is None:
        return "—"
    formatted = f"{value:.4f}".replace(".", ",")
    return "0,0000" if formatted == "-0,0000" else formatted


def _format_pct(fraction: float | None) -> str:
    """Formatea una proporción en porcentaje con coma decimal y un decimal, o «—» si es None."""
    if fraction is None:
        return "—"
    formatted = f"{fraction * 100.0:.1f}".replace(".", ",")
    if formatted == "-0,0":
        formatted = "0,0"
    return f"{formatted} %"


def _escape_md(text: str) -> str:
    """Escapa barras verticales para no romper celdas en tablas Markdown."""
    return text.replace("|", "\\|")


def month_of(instant: str) -> str:
    """Devuelve el mes en UTC («YYYY-MM») correspondiente al instante indicado.

    Admite marcas temporales ISO 8601 (con «T» o espacio, con o sin desplazamiento
    horario, o solo fecha «YYYY-MM-DD») y enteros de nanosegundos desde la época Unix.
    """
    _, month = _parse_instant(instant)
    return month


def max_drawdown(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    """Calcula la caída máxima sobre la suma acumulada de la serie.

    Con C_0 = 0 y C_t igual a la suma de los t primeros valores, devuelve un diccionario
    con las claves «depth» (máximo de C_s - C_t para 0 <= s <= t <= n), «start» (índice s
    del máximo previo en la acumulada) y «end» (índice t donde se alcanza la mayor caída).
    Si depth == 0, start = end = 0. Ante empates devuelve la primera ocurrencia.
    """
    arr = _as_1d_finite_array(values, allow_empty=True)
    cum = np.empty(arr.size + 1, dtype=np.float64)
    cum[0] = 0.0
    if arr.size > 0:
        np.cumsum(arr, dtype=np.float64, out=cum[1:])

    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    best_end = int(np.argmax(drawdowns))
    best_depth = float(drawdowns[best_end])

    if best_depth <= 0.0:
        return {"depth": 0.0, "start": 0, "end": 0}

    peak_value = running_max[best_end]
    best_start = int(np.argmax(cum[:best_end] == peak_value))
    return {"depth": best_depth, "start": best_start, "end": best_end}


def longest_negative_run(values: Sequence[float] | np.ndarray) -> int:
    """Devuelve la longitud de la racha más larga de sesiones consecutivas con valor <= 0."""
    arr = _as_1d_finite_array(values, allow_empty=True)
    non_positive = arr <= 0.0
    if not np.any(non_positive):
        return 0

    padded = np.concatenate(([False], non_positive, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return int(np.max(ends - starts))


def worst_window(
    values: Sequence[float] | np.ndarray, window: int
) -> dict[str, float | int] | None:
    """Devuelve la media mínima entre ventanas consecutivas de tamaño «window» y su inicio.

    Retorna None si la serie tiene menos de «window» sesiones. Ante empates, devuelve
    el índice base 0 de la primera ventana que alcanza la media mínima.
    """
    win = _validate_window(window)
    arr = _as_1d_finite_array(values, allow_empty=True)
    if arr.size < win:
        return None

    windows = np.lib.stride_tricks.sliding_window_view(arr, win)
    means = np.mean(windows, axis=1, dtype=np.float64)
    best_start = int(np.argmin(means))
    return {"mean": float(means[best_start]), "start": best_start}


def stability(
    values: Sequence[float] | np.ndarray,
    months: Sequence[str] | None = None,
    window: int = 21,
    *,
    blocks: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Calcula el conjunto completo de métricas de estabilidad para una serie de IC.

    Si se suministra ``blocks`` (uno por sesión, p. ej. ``(origin_year, market)``),
    ``max_drawdown``, ``longest_negative_run`` y ``worst_window`` se calculan
    estrictamente dentro de cada bloque contiguo sin cruzar saltos de origen ni mercados.

    Lanza ValueError si values no es una secuencia o array 1D no vacío de reales finitos,
    si window no es un entero positivo o si months/blocks no coinciden en longitud o formato.
    """
    arr = _as_1d_finite_array(values, allow_empty=False)
    win = _validate_window(window)
    validated_months = _validate_months(months, arr.size) if months is not None else None
    if blocks is not None:
        if isinstance(blocks, (str, bytes, bytearray)) or not isinstance(blocks, Sequence):
            raise ValueError("blocks debe ser una secuencia con la misma longitud que values.")
        if len(blocks) != arr.size:
            raise ValueError(
                f"blocks debe tener la misma longitud que values ({arr.size}), "
                f"pero tiene {len(blocks)}."
            )

    n = int(arr.size)
    mean_val = float(np.mean(arr, dtype=np.float64))

    std_val: float | None
    icir_val: float | None
    if n < 2:
        std_val = None
        icir_val = None
    elif bool(np.all(arr == arr[0])):
        # Evita residuos numéricos de coma flotante en series constantes (p. ej. 0,1 repetido).
        std_val = 0.0
        icir_val = None
    else:
        std_val = float(np.std(arr, ddof=1, dtype=np.float64))
        icir_val = None if std_val == 0.0 else float(mean_val / std_val)

    positive_fraction = float(np.sum(arr > 0.0) / n)
    if blocks is None:
        dd = max_drawdown(arr)
        neg_run = longest_negative_run(arr)
        w_win = worst_window(arr, win)
    else:
        block_indices: dict[Any, list[int]] = {}
        for idx, blk in enumerate(blocks):
            block_indices.setdefault(blk, []).append(idx)
        dd = {"depth": 0.0, "start": 0, "end": 0}
        neg_run = 0
        w_win = None
        for idxs in block_indices.values():
            block_arr = arr[idxs]
            block_dd = max_drawdown(block_arr)
            if block_dd["depth"] > dd["depth"]:
                dd = block_dd
            block_run = longest_negative_run(block_arr)
            if block_run > neg_run:
                neg_run = block_run
            block_win = worst_window(block_arr, win)
            if block_win is not None:
                candidate = {
                    "mean": block_win["mean"],
                    "start": int(idxs[int(block_win["start"])]),
                }
                if w_win is None or candidate["mean"] < w_win["mean"]:
                    w_win = candidate

    halves: dict[str, float] | None
    if n < 2:
        halves = None
    else:
        mid = n // 2
        first_mean = float(np.mean(arr[:mid], dtype=np.float64))
        second_mean = float(np.mean(arr[mid:], dtype=np.float64))
        halves = {
            "first": first_mean,
            "second": second_mean,
            "difference": float(second_mean - first_mean),
        }

    result: dict[str, Any] = {
        "sessions": n,
        "mean": mean_val,
        "std": std_val,
        "icir": icir_val,
        "positive_fraction": positive_fraction,
        "max_drawdown": dd,
        "longest_negative_run": neg_run,
        "worst_window": w_win,
        "halves": halves,
    }

    if validated_months is not None:
        groups: dict[str, list[float]] = {}
        for month, value in zip(validated_months, arr.tolist(), strict=True):
            groups.setdefault(month, []).append(value)

        monthly: list[dict[str, Any]] = []
        for month in sorted(groups):
            month_vals = np.asarray(groups[month], dtype=np.float64)
            monthly.append(
                {
                    "month": month,
                    "sessions": int(month_vals.size),
                    "mean": float(np.mean(month_vals, dtype=np.float64)),
                }
            )

        months_pos_frac = float(sum(1 for item in monthly if item["mean"] > 0.0) / len(monthly))
        worst = min(monthly, key=lambda item: item["mean"])
        best = max(monthly, key=lambda item: item["mean"])

        result["monthly"] = monthly
        result["months_positive_fraction"] = months_pos_frac
        result["worst_month"] = {"month": worst["month"], "mean": worst["mean"]}
        result["best_month"] = {"month": best["month"], "mean": best["mean"]}

    return result


def read_sessions(
    path: str | os.PathLike[str],
) -> tuple[list[str], list[str], list[str], list[str], np.ndarray]:
    """Lee un CSV de IC por sesión, valida su integridad y ordena por (instante, mercado).

    Devuelve la tupla (sessions, months, markets, names, matrix), donde matrix tiene
    forma [sesiones, modelos] y tipo float64.
    """
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or not header:
            raise ValueError("El archivo CSV está vacío.")
        if header[0] != "session":
            raise ValueError(
                f"Cabecera incorrecta: la primera columna debe ser «session», no {header[0]!r}."
            )

        names = header[1:]
        if not names:
            raise ValueError("La cabecera debe incluir al menos un modelo después de «session».")
        if any(not name or not name.strip() for name in names):
            raise ValueError("La cabecera contiene un nombre de modelo vacío.")
        if len(set(names)) != len(names):
            raise ValueError("La cabecera contiene nombres de modelos repetidos.")

        expected_fields = len(header)
        seen_sessions: set[str] = set()
        seen_keys: set[tuple[int, str]] = set()
        records: list[tuple[int, str, str, str, list[float]]] = []

        for line_no, row in enumerate(reader, start=2):
            if len(row) != expected_fields:
                raise ValueError(
                    f"Número incorrecto de campos en la línea {line_no}: "
                    f"se esperaban {expected_fields}, se encontraron {len(row)}."
                )

            session_str = row[0]
            parts = session_str.split(":", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"Identificador de sesión mal formado en la línea {line_no}: {session_str!r}."
                )

            year_str, market, instant_str = parts
            if not year_str or not year_str.isdigit():
                raise ValueError(
                    f"Año no válido en la sesión de la línea {line_no}: {session_str!r}."
                )
            if not market or not market.strip():
                raise ValueError(
                    f"Mercado vacío en la sesión de la línea {line_no}: {session_str!r}."
                )

            try:
                instant_ns, month = _parse_instant(instant_str)
            except ValueError as exc:
                raise ValueError(
                    f"Instante no válido en la línea {line_no}: {instant_str!r}."
                ) from exc

            sort_key = (instant_ns, market)
            if session_str in seen_sessions or sort_key in seen_keys:
                raise ValueError(f"Sesión repetida en la línea {line_no}: {session_str!r}.")
            seen_sessions.add(session_str)
            seen_keys.add(sort_key)

            row_values: list[float] = []
            for col_idx, raw_val in enumerate(row[1:], start=1):
                try:
                    value = float(raw_val)
                except ValueError as exc:
                    raise ValueError(
                        f"Valor numérico no válido en la línea {line_no}, "
                        f"columna {header[col_idx]!r}: {raw_val!r}."
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"Valor no finito en la línea {line_no}, "
                        f"columna {header[col_idx]!r}: {raw_val!r}."
                    )
                row_values.append(value)

            records.append((instant_ns, market, session_str, month, row_values))

    if not records:
        raise ValueError("El archivo CSV no contiene filas de datos.")

    records.sort(key=lambda rec: (rec[0], rec[1]))

    sessions = [rec[2] for rec in records]
    months = [rec[3] for rec in records]
    markets = [rec[1] for rec in records]
    matrix = np.asarray([rec[4] for rec in records], dtype=np.float64)
    return sessions, months, markets, names, matrix


def stability_table(
    path: str | os.PathLike[str], window: int = 21
) -> dict[str, dict[str, dict[str, Any]]]:
    """Calcula la tabla de estabilidad por modelo y ámbito a partir de un CSV.

    Con un único mercado solo se incluye el ámbito «all». Con varios mercados se
    calculan «all» y cada mercado por separado en orden alfabético. Las métricas
    secuenciales (caída máxima, racha negativa y peor ventana) se acotan siempre
    a cada bloque ``(origin_year, market)`` para no cruzar saltos de origen ni mercados.
    """
    win = _validate_window(window)
    sessions, months, markets, names, matrix = read_sessions(path)
    years = [sess.split(":", 2)[0] for sess in sessions]
    all_blocks = list(zip(years, markets, strict=True))

    unique_markets = sorted(set(markets))
    if len(unique_markets) > 1 and "all" in unique_markets:
        raise ValueError(
            "El nombre de mercado «all» está reservado cuando el CSV contiene varios mercados."
        )
    market_indices: dict[str, list[int]] = {}
    if len(unique_markets) > 1:
        for idx, market in enumerate(markets):
            market_indices.setdefault(market, []).append(idx)

    table: dict[str, dict[str, dict[str, Any]]] = {}
    for col_idx, model_name in enumerate(names):
        column = matrix[:, col_idx]
        model_scopes: dict[str, dict[str, Any]] = {
            "all": stability(column, months=months, window=win, blocks=all_blocks)
        }
        if len(unique_markets) > 1:
            for market in unique_markets:
                indices = market_indices[market]
                scope_months = [months[i] for i in indices]
                scope_blocks = [(years[i], market) for i in indices]
                model_scopes[market] = stability(
                    column[indices], months=scope_months, window=win, blocks=scope_blocks
                )
        table[model_name] = model_scopes

    return table


def markdown(table: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> str:
    """Genera una tabla Markdown en español con las métricas de estabilidad por modelo y ámbito."""
    lines = [
        "| " + " | ".join(_MD_HEADERS) + " |",
        "| " + " | ".join(_MD_ALIGNMENTS) + " |",
    ]

    for model_name, scopes in table.items():
        for scope_name, stats in scopes.items():
            worst_month = stats.get("worst_month")
            if worst_month is None:
                worst_month_cell = "—"
            else:
                worst_month_cell = f"{worst_month['month']} ({_format_float(worst_month['mean'])})"

            w_win = stats["worst_window"]
            worst_window_cell = "—" if w_win is None else _format_float(w_win["mean"])

            row = (
                _escape_md(str(model_name)),
                _escape_md(str(scope_name)),
                str(stats["sessions"]),
                _format_float(stats["mean"]),
                _format_float(stats["icir"]),
                _format_pct(stats["positive_fraction"]),
                _format_pct(stats.get("months_positive_fraction")),
                worst_month_cell,
                _format_float(stats["max_drawdown"]["depth"]),
                str(stats["longest_negative_run"]),
                worst_window_cell,
            )
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada de línea de comandos para analizar un CSV de IC por sesión."""
    parser = argparse.ArgumentParser(
        prog="dgx_ic_stability.py",
        description="Mide la estabilidad temporal de series de IC por sesión.",
    )
    parser.add_argument("csv", help="Ruta al archivo CSV de entrada.")
    parser.add_argument(
        "--window",
        type=int,
        default=21,
        help="Tamaño de la ventana móvil de sesiones (por defecto: 21).",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Ruta opcional donde escribir el resultado en formato JSON.",
    )

    args = parser.parse_args(argv)
    try:
        table = stability_table(args.csv, window=args.window)
        md_output = markdown(table)
        if args.json_path is not None:
            json_text = json.dumps(table, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
            Path(args.json_path).write_text(json_text, encoding="utf-8")
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(md_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
