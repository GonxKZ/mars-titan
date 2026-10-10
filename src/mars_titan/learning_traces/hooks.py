"""Ganchos comunes de trazas para todos los entrenadores (#448).

Un entrenador con trazas recibe un `TrainingTraces` y lo usa en cuatro sitios, siempre los
mismos en cada familia:

- `update(step, optimizer, loss)` envuelve el recorte del gradiente y el paso del
  optimizador. En los pasos que tocan registro guarda la pérdida, las normas del gradiente
  y de los pesos por grupo de parámetros antes del recorte, la norma total que devuelve el
  recorte con su umbral, la tasa de aprendizaje de cada grupo del optimizador y la razón de
  actualización después del paso. En el resto de pasos no hace nada.
- `observing(step)` envuelve el recorrido hacia delante que lleva a ese paso. Si toca
  registro, activa los observadores internos (por ejemplo las puertas de la memoria de
  Titans) y los desactiva al salir.
- `validation(event, modules)` se llama tras cada validación completa. Guarda los campos
  numéricos del evento, las normas de los pesos por grupo y lo que añadan los observadores,
  y vuelca todo lo pendiente.
- `checkpoint()` vuelca lo pendiente y devuelve el cursor que se guarda con el estado.
  `restore(state)` lo recupera al empezar o reanudar y borra las partes posteriores.

Las estadísticas se quedan en el dispositivo hasta el volcado y no consumen números
aleatorios. Sin trazas, `traced_update(None, ...)` y `observing(None, ...)` son bloques
vacíos, de modo que el recorrido es el mismo bit a bit. Las trazas no entran en la
identidad de ninguna ejecución.
"""

import hashlib
import json
from contextlib import contextmanager, nullcontext

from .parameters import UpdateProbe, parameter_groups, record_gradients
from .recorder import STEP_UNITS, TraceConfig, TraceRecorder, trace_name

# Estadísticas pendientes a partir de las que se vuelca sin esperar al punto de control. Con
# siete valores en float64 por estadística, son unos 230 KB en el dispositivo.
MAX_PENDING = 4096
# Cursor de un registrador sin partes confirmadas.
EMPTY_CURSOR = dict(sequence=0, bytes=0, exhausted_at=None)


class TraceObserver:
    """Observador interno de una familia. Cada método es opcional y por defecto no hace nada.

    `start` y `stop` delimitan el recorrido hacia delante de un paso registrado, `record`
    guarda lo acumulado después del paso y `validation` añade medidas en cada validación.
    `reset` descarta lo acumulado al guardar un punto de control, porque ese acumulado no
    viaja con el estado y una reanudación no lo tendría.
    """

    def start(self):
        pass

    def stop(self):
        pass

    def record(self, recorder):
        pass

    def validation(self, recorder, modules):
        pass

    def reset(self):
        pass


class _Probe:
    """Recibe la norma total que devuelve el recorte y su umbral."""

    __slots__ = ("norm", "threshold")

    def __init__(self):
        self.norm, self.threshold = None, None

    def clip(self, norm, threshold=None):
        self.norm, self.threshold = norm, threshold


class _Skip:
    """Receptor del recorte en un paso sin registro."""

    def clip(self, norm, threshold=None):
        pass


SKIP = _Skip()


class TrainingTraces:
    """Trazas de un entrenador con el contrato común."""

    def __init__(self, recorder, modules, *, observers=()):
        if not isinstance(recorder, TraceRecorder) or not modules:
            raise ValueError("Las trazas necesitan su registrador y los módulos del ajuste")
        self.recorder = recorder
        self.modules = dict(modules)
        self.grouped = parameter_groups(self.modules)
        self.observers = list(observers)
        self._probe = UpdateProbe()

    @property
    def active(self):
        return self.recorder.active

    def due(self, step):
        return self.recorder.due(step)

    @contextmanager
    def update(self, step, optimizer, loss=None, *, phase="train"):
        if not self.recorder.due(step):
            yield SKIP
            return
        probe = _Probe()
        with self.recorder.at(step, phase):
            if loss is not None:
                self.recorder.scalar("loss", loss)
            record_gradients(self.recorder, self.grouped)
            self._probe.before(self.grouped)
        yield probe
        with self.recorder.at(step, phase):
            self._probe.after(self.recorder, self.grouped)
            if probe.norm is not None:
                self.recorder.scalar("grad_l2_total", probe.norm)
            if probe.threshold is not None:
                self.recorder.scalar("clip_threshold", probe.threshold)
            # Los optimizadores de PyTorch llevan `lr` en cada grupo. Uno sin tasa por grupo no
            # tiene nada que registrar aquí, y la traza no debe detener el ajuste por eso.
            for index, group in enumerate(getattr(optimizer, "param_groups", ())):
                if "lr" in group:
                    name = trace_name(group.get("role", f"group_{index}"))
                    self.recorder.scalar("learning_rate", group["lr"], group=name)
            for observer in self.observers:
                observer.record(self.recorder)
        if self.recorder.pending >= MAX_PENDING:
            self.recorder.flush()

    @contextmanager
    def observing(self, step):
        if not self.observers or not self.recorder.due(step):
            yield
            return
        for observer in self.observers:
            observer.start()
        try:
            yield
        finally:
            for observer in self.observers:
                observer.stop()

    def validation(self, event, modules=None, *, phase="validation"):
        """Medidas de una validación completa. Vuelca todo lo pendiente.

        Los campos numéricos del evento se guardan como `<fase>.<campo>`, por ejemplo
        `validation.score`.
        """
        if not self.recorder.active:
            return
        step = event["global_step"]
        with self.recorder.at(step, phase):
            for key, value in sorted(event.items()):
                if key != "global_step" and type(value) in (int, float):
                    self.recorder.scalar(f"{phase}.{trace_name(key)}", value)
            for group, parameters in self.grouped.items():
                weights = [p.detach() for _, p in parameters]
                total = sum(w.double().square().sum() for w in weights)
                self.recorder.scalar("weight_l2", total.sqrt(), group=group)
            for observer in self.observers:
                observer.validation(self.recorder, modules or self.modules)
        self.recorder.flush()

    def checkpoint(self):
        """Volcar lo pendiente y devolver el cursor que acompaña al estado guardado.

        Lo que los observadores hayan acumulado desde el último paso se descarta. Así un
        recorrido reanudado registra lo mismo que uno sin cortes con los mismos guardados.
        """
        for observer in self.observers:
            observer.reset()
        self.recorder.flush()
        return self.recorder.state_dict()

    def restore(self, state):
        """Recuperar el cursor del estado (o empezar de cero) y borrar las partes posteriores."""
        self.recorder.load_state_dict(EMPTY_CURSOR if state is None else state)


def trace_identity(family, identity, **fields):
    """Identidad de las trazas: la familia y la huella de la identidad de su ejecución."""
    encoded = json.dumps(identity, sort_keys=True, default=str, allow_nan=False)
    return dict(
        family=family,
        run_identity_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
        **fields,
    )


def unit_config(config, step_unit):
    """`TraceConfig` de una unidad de paso, o None sin trazas.

    La declaración es un `TraceConfig` o un diccionario con `every` y `max_bytes`, y puede
    añadir `every_by_unit`, que fija otra cadencia para las unidades que no son pasos del
    optimizador (por ejemplo transiciones u oleadas de la etapa RL nativa).
    """
    if config is None or isinstance(config, TraceConfig):
        return config
    fields = dict(config)
    by_unit = fields.pop("every_by_unit", None) or {}
    if not set(by_unit) <= set(STEP_UNITS):
        raise ValueError("every_by_unit declara unidades de paso desconocidas")
    fields["every"] = by_unit.get(step_unit, fields["every"])
    return TraceConfig(**fields)


def open_recorder(folder, config, identity, *, step_unit="optimizer_step"):
    """Registrador de una ejecución, o None si no declara trazas o las desactiva."""
    config = unit_config(config, step_unit)
    if config is None:
        return None
    recorder = TraceRecorder(folder, config, identity, step_unit=step_unit)
    return recorder if recorder.active else None


def open_traces(folder, config, identity, modules, *, observers=()):
    """Crear las trazas de un ajuste, o None si la ejecución no declara trazas."""
    recorder = open_recorder(folder, config, identity)
    return None if recorder is None else TrainingTraces(recorder, modules, observers=observers)


def traced_update(traces, step, optimizer, loss=None, *, phase="train"):
    """`traces.update` o un bloque vacío sin trazas."""
    if traces is None:
        return nullcontext(SKIP)
    return traces.update(step, optimizer, loss, phase=phase)


def observing(traces, step):
    """`traces.observing` o un bloque vacío sin trazas."""
    return nullcontext() if traces is None else traces.observing(step)
