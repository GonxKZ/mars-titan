"""Edición sin ajustar sintética con un activo de cada estrato de liquidez.

Los precios y volúmenes son técnicos, elegidos para que cada activo caiga en un estrato
conocido. No proceden de FinMultiTime ni de la edición real.
"""

import json
from pathlib import Path

from mars_titan.evaluation import walk_forward_comparison as walk
from tests.simulation.unadjusted_edition_fixture import Asset, tape_days, write_edition

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/evaluation/historical-masked-2000-comparison.json"
THIN = 500.0
# Estrato esperado de cada símbolo con la declaración de la campaña.
EXPECTED = dict(
    A="low_price",
    B="low_price_and_thin_volume",
    C="thin_volume",
    D="unclassified",
    E="liquid",
    F="liquid",
    G="unclassified",
)


def section(**changes):
    """Declaración de la campaña con los cambios indicados."""
    return dict(json.loads(CONFIG.read_text())[walk.LIQUIDITY_FIELD], **changes)


def thin(market):
    """Volumen por debajo del umbral en todas las sesiones de 2023."""
    return {index: dict(volume=THIN) for index in range(len(tape_days(market)))}


def study_assets(market):
    """A y B por debajo de 1, B y C con pocos títulos, D sin verificar y G fuera."""
    return [
        Asset("A", base=0.5),
        Asset("B", base=0.5, overrides=thin(market)),
        Asset("C", overrides=thin(market)),
        Asset("D", never_verified=True),
        Asset("E"),
        Asset("F", base=40.0),
    ]


def study_edition(root, markets=("US", "CN")):
    """Escribir la edición de los activos del estudio sintético y devolver su carpeta."""
    write_edition(root, {market: study_assets(market) for market in markets})
    return root
