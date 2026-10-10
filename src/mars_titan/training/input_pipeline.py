"""Tubería acotada de lectura: decodificación ordenada en hilos y prefetch en segundo plano.

Las dos piezas conservan el orden exacto de la ruta secuencial. `ordered_map` reparte una
función pura entre hilos y entrega sus resultados en el orden de entrada, con un número
máximo de tareas pendientes. `background` recorre un iterable en un hilo productor con una
cola acotada. Un error se entrega en la misma posición en que lo habría lanzado la ruta
secuencial, después de los elementos anteriores, y cerrar el consumidor detiene al productor.

Las opciones se declaran con `PipelineOptions` o con las variables de entorno
`MARS_TITAN_DECODE_WORKERS` y `MARS_TITAN_PREFETCH_BATCHES`. Sin ellas la lectura es la
secuencial. No cambian qué datos se leen ni en qué orden, así que no forman parte de la
identidad científica de ningún ajuste.
"""

import atexit
import os
import queue
import threading
from collections import deque
from dataclasses import dataclass

DECODE_WORKERS_ENV = "MARS_TITAN_DECODE_WORKERS"
PREFETCH_ENV = "MARS_TITAN_PREFETCH_BATCHES"
GROUP_CACHE_ENV = "MARS_TITAN_GROUP_CACHE_MIB"
MAX_DECODE_WORKERS = 16
MAX_PREFETCH = 64
# La lectura cronológica conserva un grupo decodificado por activo, unos 0,7 MiB con las
# anchuras de la edición v3. En 2023 hay 4.162 activos US con muestra en la misma sesión, así
# que 1 GiB obligaba a decodificar de nuevo cada grupo en cada instante. 4 GiB alcanzan los
# activos de US y CN juntos. Es un límite: la caché solo crece hasta los grupos en uso.
GROUP_CACHE_MIB = 4096
MAX_GROUP_CACHE_MIB = 8192


def _bounded_integer(value, upper, label):
    if type(value) is not int or not 0 <= value <= upper:
        raise ValueError(f"{label} debe ser un entero entre 0 y {upper}")
    return value


def _environment_integer(name, upper, default=0):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not 1 <= len(raw) <= len(str(upper)) or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{name} debe ser un entero entre 0 y {upper}")
    return _bounded_integer(int(raw), upper, name)


@dataclass(frozen=True)
class PipelineOptions:
    """Hilos de decodificación y lotes adelantados. Cero conserva la ruta secuencial.

    `lookahead` acota las tareas de decodificación pendientes: cada una retiene como mucho
    un grupo Parquet decodificado (unos 0,7 MiB con las anchuras de la edición v3).
    """

    decode_workers: int = 0
    prefetch_batches: int = 0
    group_cache_mib: int = GROUP_CACHE_MIB

    def __post_init__(self):
        _bounded_integer(self.decode_workers, MAX_DECODE_WORKERS, "decode_workers")
        _bounded_integer(self.prefetch_batches, MAX_PREFETCH, "prefetch_batches")
        _bounded_integer(self.group_cache_mib, MAX_GROUP_CACHE_MIB, "group_cache_mib")
        if self.group_cache_mib < 1:
            raise ValueError("La caché de grupos necesita al menos 1 MiB")

    @classmethod
    def from_environment(cls):
        return cls(
            decode_workers=_environment_integer(DECODE_WORKERS_ENV, MAX_DECODE_WORKERS),
            prefetch_batches=_environment_integer(PREFETCH_ENV, MAX_PREFETCH),
            group_cache_mib=_environment_integer(
                GROUP_CACHE_ENV, MAX_GROUP_CACHE_MIB, GROUP_CACHE_MIB
            ),
        )

    @property
    def group_cache_bytes(self):
        return self.group_cache_mib * 1024**2

    @property
    def lookahead(self):
        return 2 * self.decode_workers + 2 if self.decode_workers else 0

    def environment(self):
        """Variables que reproducen estas opciones en un proceso hijo."""
        return {
            DECODE_WORKERS_ENV: str(self.decode_workers),
            PREFETCH_ENV: str(self.prefetch_batches),
            GROUP_CACHE_ENV: str(self.group_cache_mib),
        }


class _Raised:
    """Error de la entrada, entregado después de los resultados anteriores."""

    def __init__(self, error):
        self.error = error

    def result(self):
        raise self.error

    def cancel(self):
        return False

    def done(self):
        return True


# Marca del hilo que ejecuta ahora una tarea de decodificación.
_TASK = threading.local()


def _marked(function, *args):
    _TASK.active = True
    try:
        return function(*args)
    finally:
        _TASK.active = False


def submit(executor, function, *args):
    """Enviar una tarea de decodificación que marca su hilo mientras se ejecuta."""
    return executor.submit(_marked, function, *args)


def in_task():
    """Si este hilo está ejecutando una tarea enviada con `submit`."""
    return getattr(_TASK, "active", False)


def drain(futures):
    """Cancelar las tareas que no han empezado y esperar a las que están en curso.

    Un recorrido abandonado sin cerrar puede finalizarlo el recolector de basura en
    cualquier hilo, también dentro de una de sus propias tareas. Esperar ahí bloquearía ese
    hilo para siempre, así que dentro de una tarea solo se cancela. Las tareas en curso
    terminan solas y su resultado se descarta.
    """
    for future in futures:
        future.cancel()
    if in_task():
        return
    for future in futures:
        if not future.done():
            try:
                future.result()
            except BaseException:  # El recorrido ya se ha cerrado y el error no importa.
                pass


def ordered_map(function, items, executor, lookahead):
    """Aplicar `function` en `executor` y entregar los resultados en el orden de `items`.

    Como máximo `lookahead` tareas quedan enviadas sin consumir. Un error al obtener el
    siguiente elemento se entrega tras los resultados ya enviados y detiene la lectura de
    `items`, igual que en un recorrido secuencial. Al cerrar el generador se cancelan las
    tareas que no han empezado y se espera a las que están en curso (`drain`), de modo que
    ningún hilo sigue usando los recursos del recorrido.
    """
    if type(lookahead) is not int or lookahead < 1:
        raise ValueError("El adelanto de la decodificación debe ser un entero positivo")
    pending, iterator, exhausted = deque(), iter(items), False
    try:
        while True:
            while not exhausted and len(pending) < lookahead:
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                except BaseException as error:  # Se entrega en su posición.
                    pending.append(_Raised(error))
                    exhausted = True
                else:
                    pending.append(submit(executor, function, item))
            if not pending:
                return
            yield pending.popleft().result()
    finally:
        drain(pending)
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


_END = object()
# Productores vivos. Si un consumidor abandona su generador sin cerrarlo, el hilo daemon
# seguiría dentro de PyArrow al terminar el intérprete y el proceso abortaría
# («terminate called without an active exception») sin cerrar sus archivos.
_PRODUCERS = set()
_STOP_TIMEOUT = 30.0


@atexit.register
def _stop_producers():
    """Detener y esperar a los productores abandonados antes de cerrar el intérprete."""
    producers = list(_PRODUCERS)
    for stop, _ in producers:
        stop.set()
    for _, worker in producers:
        worker.join(_STOP_TIMEOUT)


class _Failure:
    def __init__(self, error):
        self.error = error


def background(iterable, depth):
    """Recorrer `iterable` en un hilo productor con una cola de `depth` elementos.

    Los elementos llegan en el mismo orden y un error del productor se lanza en la posición
    en que se habría producido. Cerrar este generador detiene al productor en su siguiente
    elemento, cierra `iterable` en ese mismo hilo y espera a que termine.
    """
    if type(depth) is not int or depth < 1:
        raise ValueError("La profundidad del prefetch debe ser un entero positivo")
    channel, stop = queue.Queue(maxsize=depth), threading.Event()

    def offer(value):
        while not stop.is_set():
            try:
                channel.put(value, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        iterator = iter(iterable)
        try:
            for item in iterator:
                if not offer(item):
                    return
            offer(_END)
        except BaseException as error:  # Se entrega al consumidor.
            offer(_Failure(error))
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    worker = threading.Thread(target=produce, name="mars-titan-prefetch", daemon=True)
    producer = stop, worker
    _PRODUCERS.add(producer)
    worker.start()
    try:
        while True:
            item = channel.get()
            if item is _END:
                return
            if isinstance(item, _Failure):
                raise item.error
            yield item
    finally:
        stop.set()
        while worker.is_alive():
            try:
                channel.get(timeout=0.05)
            except queue.Empty:
                continue
        worker.join()
        _PRODUCERS.discard(producer)
