"""Enganchar las trazas comunes a los entrenadores cronológicos (#448).

Los entrenadores de Titans-MAC, de los lectores de MARS-TITAN y CM-v1 y de la GRU candidata
exponen `_trace_modules()`, `identity` y los atributos `trace` y `traces`. `attach` crea el
registrador en la carpeta de la ventana, añade el observador de la memoria neuronal cuando
el modelo tiene alguna y deja las trazas en el entrenador, que las usa en sus pasos, sus
validaciones y sus puntos de control. Sin configuración no toca el entrenador.
"""

from .hooks import open_traces, trace_identity
from .memory import MemoryObserver


def attach(trainer, folder, config, family, *, observers=()):
    """Trazas de `trainer` en `folder`, o None si la ejecución no declara trazas."""
    if config is None:
        return None
    modules = trainer._trace_modules()
    memory = MemoryObserver(modules)
    traces = open_traces(
        folder,
        config,
        trace_identity(family, trainer.identity),
        modules,
        observers=([memory] if memory else []) + list(observers),
    )
    trainer.traces = traces
    return traces
