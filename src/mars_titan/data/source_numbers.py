"""Conversión exacta de los números que las fuentes escriben como texto.

El analizador por defecto del motor C de pandas no redondea siempre al double más cercano.
Con pandas 3.0.6 se aleja una unidad de la última cifra (1 ULP) en el 11,3 % de los precios
chinos de FinMultiTime. Las ediciones anteriores a la v3.1 conservan ese error. Desde la v3.1
todo número leído de un texto de la fuente pasa por `exact_floats`, que aplica `float()` a cada
valor y devuelve el double correctamente redondeado.
"""

import csv
import math
import multiprocessing
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

NUMBER_PARSING = "python_float_correctly_rounded_v1"
# Solo decimales ASCII con signo y exponente opcionales. `float()` aceptaría además guiones
# bajos, espacios, dígitos de otros alfabetos e «infinity», que la fuente no debe colar.
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def exact_floats(values) -> np.ndarray:
    """Convertir textos decimales con `float()` y dejar como NaN el resto y las ausencias.

    Igual que `pd.to_numeric(errors="coerce")`, lo que no es un número queda ausente, pero sin
    pasar por el analizador aproximado. Un valor que no sea texto ni ausencia es un error del
    lector, porque significaría que otro analizador ya lo convirtió.
    """
    result = np.empty(len(values), dtype=np.float64)
    for i, value in enumerate(values):
        if isinstance(value, str):
            result[i] = float(value) if _DECIMAL.fullmatch(value) else math.nan
        elif value is None or (isinstance(value, float) and math.isnan(value)):
            result[i] = math.nan
        else:
            raise TypeError("La conversión exacta solo admite texto o ausencias")
    return result


def read_source_csv(path: Path, numeric: Sequence[str]) -> pd.DataFrame:
    """Leer un CSV de la fuente como texto y convertir exactamente las columnas numéricas.

    Las demás columnas quedan como texto. Las columnas numéricas que el archivo no tenga no se
    crean, para que cada lector conserve su propia comprobación de esquema.
    """
    frame = pd.read_csv(path, dtype=str)
    for name in numeric:
        if name in frame:
            frame[name] = exact_floats(frame[name].to_numpy(dtype=object))
    return frame


def compare_number_parsers(path: Path, numeric: Sequence[str]) -> dict:
    """Comparar valor a valor la lectura exacta, `float(texto)` y el analizador de pandas.

    La referencia lee el CSV con el módulo `csv` y aplica `float()` a cada texto. La lectura
    exacta debe coincidir bit a bit. Los recuentos del analizador por defecto de pandas, el que
    usaban las ediciones hasta la v3, miden cuántos valores se alejaban de su texto.
    """
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    header, rows = rows[0], rows[1:]
    exact = read_source_csv(path, numeric)
    legacy = pd.read_csv(path, dtype=dict.fromkeys(["Date", "Dividends", "Stock Splits"], str))
    counts = dict(
        values=0,
        absent=0,
        exact_mismatches=0,
        pattern_rejected_float_accepted=0,
        legacy_mismatches=0,
        legacy_max_ulps=0,
    )
    for name in numeric:
        if name not in header:
            continue
        column = header.index(name)
        texts = [row[column] if column < len(row) else "" for row in rows]
        reference = np.array([_python_float(text) for text in texts], dtype=np.float64)
        got = exact[name].to_numpy(dtype=np.float64)
        old = pd.to_numeric(legacy[name], errors="coerce").to_numpy(dtype=np.float64)
        absent = np.isnan(reference)
        rejected = np.isnan(got) & ~absent
        counts["values"] += len(texts)
        counts["absent"] += int(absent.sum())
        counts["pattern_rejected_float_accepted"] += int(rejected.sum())
        counts["exact_mismatches"] += int((_differ(got, reference) & ~rejected).sum())
        moved = _differ(old, reference)
        counts["legacy_mismatches"] += int(moved.sum())
        if moved.any():
            same = (
                moved
                & np.isfinite(old)
                & np.isfinite(reference)
                & (np.sign(old) == np.sign(reference))
            )
            ulps = np.abs(old[same].view(np.int64) - reference[same].view(np.int64))
            counts["legacy_max_ulps"] = max(counts["legacy_max_ulps"], int(ulps.max(initial=0)))
    return counts


def _python_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def _differ(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    both_absent = np.isnan(left) & np.isnan(right)
    return ~both_absent & (left.view(np.int64) != right.view(np.int64))


PRICE_SOURCES = {"US": "S&P500_time_series", "CN": "HS300_time_series"}
PRICE_NUMBERS = ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")


def number_parsing_receipt(source: Path, *, workers: int) -> dict:
    """Recorrer todos los CSV de precios y sumar las comparaciones por mercado.

    Cada proceso lee un archivo cada vez, así que la memoria queda acotada por el CSV mayor.
    """
    import platform
    import time
    from concurrent.futures import ProcessPoolExecutor

    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("El número de procesos debe estar entre 1 y 16")
    started = time.perf_counter()
    markets = {}
    # Procesos nuevos en lugar de fork, que no es seguro con hilos de Arrow o BLAS ya activos.
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=spawn) as pool:
        for market, folder in PRICE_SOURCES.items():
            paths = sorted((source / "time_series" / folder).glob("*.csv"))
            totals = dict(files=len(paths))
            for counts in pool.map(
                compare_number_parsers, paths, [PRICE_NUMBERS] * len(paths), chunksize=8
            ):
                for key, value in counts.items():
                    if key == "legacy_max_ulps":
                        totals[key] = max(totals.get(key, 0), value)
                    else:
                        totals[key] = totals.get(key, 0) + value
            present = totals["values"] - totals["absent"]
            totals["legacy_mismatch_fraction"] = totals["legacy_mismatches"] / present
            markets[market] = totals
    return dict(
        number_parsing=NUMBER_PARSING,
        reference="csv.reader + float(text)",
        legacy="pandas.read_csv default C engine + pandas.to_numeric",
        columns=list(PRICE_NUMBERS),
        markets=markets,
        versions=dict(
            python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__
        ),
        workers=workers,
        elapsed_seconds=time.perf_counter() - started,
        # La orden termina con error si un valor no coincide con su texto.
        errors=[
            market
            for market, totals in markets.items()
            if totals["exact_mismatches"] or totals["pattern_rejected_float_accepted"]
        ],
    )
