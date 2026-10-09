"""Conversión exacta de los números que las fuentes escriben como texto.

El analizador por defecto del motor C de pandas no redondea siempre al double más cercano.
Con pandas 3.0.6 se aleja una unidad de la última cifra (1 ULP) en el 11,3 % de los precios
chinos de FinMultiTime. Las ediciones anteriores a la v3.1 conservan ese error. Desde la v3.1
todo número leído de un texto de la fuente pasa por `exact_floats`, que aplica `float()` a cada
valor y devuelve el double correctamente redondeado.
"""

import math
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
