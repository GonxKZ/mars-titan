"""Rangos NVTX opcionales para leer por fase las trazas de Nsight Systems.

Están apagados por defecto. Con `MARS_TITAN_NVTX=1` al arrancar el proceso, cada fase queda
entre `range_push` y `range_pop` de `torch.cuda.nvtx`. Son marcas de tiempo del anfitrión:
no sincronizan la GPU, no crean tensores ni tocan el RNG, así que las salidas no cambian.
Apagados, `phase` devuelve siempre el mismo `nullcontext` e `iterate` devuelve el iterable
recibido, de modo que el coste es una consulta de un booleano por fase.

Un rango se abre y se cierra en el mismo hilo. Los del lector con prefetch aparecen en el
hilo productor y el tiempo que el entrenador espera al siguiente instante, en el principal.
"""

import os
from contextlib import nullcontext

ENVIRONMENT = "MARS_TITAN_NVTX"


def _enabled():
    value = os.environ.get(ENVIRONMENT, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{ENVIRONMENT} debe valer 0 o 1")
    if value == "1":
        import torch

        # Sin el backend CUDA de PyTorch no hay NVTX: se avisa al arrancar, no a mitad de fase.
        if torch.version.cuda is None:
            raise RuntimeError(f"{ENVIRONMENT}=1 necesita PyTorch con CUDA")
    return value == "1"


ENABLED = _enabled()
_OFF = nullcontext()


def _nvtx():
    import torch

    return torch.cuda.nvtx


class _Range:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        _nvtx().range_push(self.name)
        return self

    def __exit__(self, *_):
        _nvtx().range_pop()
        return False


def phase(name):
    """Contexto de una fase: un rango NVTX si están encendidos y nada si no."""
    return _Range(name) if ENABLED else _OFF


def iterate(name, iterable):
    """Cada `next` del iterable dentro de un rango. Apagados, devuelve el mismo iterable."""
    if not ENABLED:
        return iterable
    return _ranged(name, iter(iterable))


def _ranged(name, iterator):
    try:
        while True:
            with _Range(name):
                try:
                    item = next(iterator)
                except StopIteration:
                    return
            yield item
    finally:
        # Cerrar el envoltorio cierra el lector, como al abandonar un `for` sobre él.
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
