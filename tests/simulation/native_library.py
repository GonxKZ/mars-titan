"""Variantes nativas de las pruebas de simulación, omitidas solo si no hay compilación."""

import os
from pathlib import Path

import pytest

from mars_titan.simulation.native_runtime import library_path

ROOT = Path(__file__).resolve().parents[2]
SIMULATOR = ROOT / "build" / "native" / "native-release" / "mars-titan-sim"

# Sin ruta declarada ni compilación native-release la prueba se omite con su motivo. Una ruta
# declarada inexistente o una biblioteca defectuosa no se omiten y siguen fallando al cargar.
requires_native_library = pytest.mark.skipif(
    library_path() is None,
    reason="Falta la biblioteca nativa de simulación (preset native-release o "
    "MARS_TITAN_NATIVE_LIBRARY)",
)
NATIVE_BACKEND = pytest.param("native", marks=requires_native_library)


def simulator_path():
    """mars-titan-sim declarado en el entorno o compilado con native-release.

    Sigue la convención de la biblioteca: sin ruta declarada ni compilación se omite la prueba,
    y una ruta declarada que no existe falla.
    """
    declared = os.environ.get("MARS_TITAN_SIM_EXECUTABLE")
    if declared:
        return Path(declared).resolve(strict=True)
    if not SIMULATOR.is_file():
        pytest.skip(
            "Falta el ejecutable mars-titan-sim (preset native-release o MARS_TITAN_SIM_EXECUTABLE)"
        )
    return SIMULATOR
