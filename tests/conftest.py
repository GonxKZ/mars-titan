"""Configuración común de pytest."""

import pytest

from mars_titan.training.learning_hold import install_optimizer_guard


def _skip(reason: str) -> None:
    pytest.skip(reason)


# Mientras la protección local esté vigente, una prueba que intente un paso de optimizador
# de PyTorch se omite antes de modificar pesos. Sin protección, el gancho no se instala.
_GUARD = install_optimizer_guard(_skip)
