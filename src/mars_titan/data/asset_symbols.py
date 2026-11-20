"""Símbolos de activos admitidos en las cohortes y los resultados de la comparación."""

import re

MAX_SYMBOL_LENGTH = 64
SYMBOL = re.compile(rf"[A-Z0-9.^_=-]{{1,{MAX_SYMBOL_LENGTH}}}")
RESERVED_SYMBOLS = (".", "..")
