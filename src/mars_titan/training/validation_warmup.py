"""Estado del padre congelado justo antes de la primera decisión de validación."""

import os
from bisect import bisect_left
from contextlib import closing

import torch

from .financial_run import Paused

SWITCH = "MARS_TITAN_RESIDENT_VALIDATION_WARMUP"
MAX_BYTES = 256 * 1024**2
RESERVE = 1024**3


def configured(device):
    value = os.environ.get(SWITCH, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{SWITCH} solo admite 0 o 1")
    return ValidationWarmup() if value == "1" and device.type == "cuda" else None


class ValidationWarmup:
    def __init__(self, limit=MAX_BYTES):
        if type(limit) is not int or not 0 < limit <= MAX_BYTES:
            raise ValueError("Presupuesto de calentamiento fuera de rango")
        self.limit = limit
        self.clear()
        self.hits = 0
        self.allocation_fallbacks = 0

    def clear(self):
        self.key = None
        self.states = ()
        self.events = self.observations = self.tensor_bytes = 0

    def receipt(self):
        return dict(
            hits=self.hits,
            events=self.events,
            tensor_bytes=self.tensor_bytes,
            limit_bytes=self.limit,
            allocation_fallbacks=self.allocation_fallbacks,
        )

    def _room(self, device, needed):
        return needed <= self.limit and (
            device.type != "cuda" or torch.cuda.mem_get_info(device)[0] >= RESERVE + 2 * needed
        )

    def _capture(self, trainer, run, key, cut, size):
        flows = list(run.flows)
        needed = len(flows) * trainer.predictor._tensor_bytes_per_flow()
        if not flows or not self._room(trainer.device, needed):
            return
        # Solo se copian las filas vivas. Cada repetición tendrá su propio estado mutable.
        try:
            self.states = tuple(
                trainer.predictor.export_state(run.flows.gather(flows[i : i + size]))
                for i in range(0, len(flows), size)
            )
        except torch.OutOfMemoryError:
            self.allocation_fallbacks += 1
            return
        self.events, self.observations = cut, run.counters["warmup_observations"]
        self.key, self.tensor_bytes = key, needed

    def _restore(self, trainer, run):
        try:
            for packed in self.states:
                run.flows.put(trainer.predictor.restore_state(packed))
        except torch.OutOfMemoryError:
            run.flows.clear()
            self.clear()
            self.allocation_fallbacks += 1
            return
        for name in ("observations", "mac_updates", "warmup_observations"):
            run.counters[name] = self.observations
        self.hits += 1

    def evaluate(self, trainer, source, *, stop=None, rows=None):
        # Solo el padre y sus flujos avanzan durante el prefijo sin etiquetas.
        # El lector y el banco se vuelven a ejecutar en todas las decisiones puntuadas.
        trainer._check_runtime()
        phase, size = source.phase, trainer.recipe.block_rows
        key = (
            source.identity,
            phase,
            trainer.predictor._parameter_id,
            size,
            trainer.dtype,
            trainer.device,
        )
        if key != self.key or not self._room(trainer.device, self.tensor_bytes):
            self.clear()
        cut = bisect_left(source.metadata["groups"], phase.decision_start, key=lambda row: row[0])
        run = trainer._new_pass(phase.partition)
        trainer.readout.eval()
        with torch.no_grad():
            if stop is not None and stop.requested:
                raise Paused
            if self.events:
                if self.events != cut:
                    raise ValueError("El corte del calentamiento ha cambiado")
                self._restore(trainer, run)
                with closing(
                    source.batched_events(start_cursor=self.events, block_rows=size)
                ) as events:
                    return trainer._pass(phase, events, stop=stop, rows=rows, run=run)
            with closing(source.batched_events(block_rows=size)) as events:
                for _ in range(cut):
                    if stop is not None and stop.requested:
                        raise Paused
                    event = next(events)
                    if event.at >= phase.decision_start or event.labels or event.close_phase:
                        raise ValueError("El prefijo de validación contiene decisiones o etiquetas")
                    if event.inputs:
                        trainer._observe(run, phase, event, train=False)
                self._capture(trainer, run, key, cut, size)
                return trainer._pass(phase, events, stop=stop, rows=rows, run=run)
