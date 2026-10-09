"""Variante nativa de las pruebas de simulación, omitida solo si no hay biblioteca compilada."""

import pytest

from mars_titan.simulation.native_runtime import library_path

# Sin ruta declarada ni compilación native-release la prueba se omite con su motivo. Una ruta
# declarada inexistente o una biblioteca defectuosa no se omiten y siguen fallando al cargar.
requires_native_library = pytest.mark.skipif(
    library_path() is None,
    reason="Falta la biblioteca nativa de simulación (preset native-release o "
    "MARS_TITAN_NATIVE_LIBRARY)",
)
NATIVE_BACKEND = pytest.param("native", marks=requires_native_library)
