"""Registro de las etiquetas auxiliares de F5 (#749) durante un recorrido de ajuste.

Cada evento con entradas avanza el ordinal de decisión de sus mercados. Cada decisión emitida
guarda su ordinal, sus entradas CPU y, cuando madura, su etiqueta principal. Una ventana de h
ordinales consecutivos del mismo flujo forma una etiqueta auxiliar en el instante en que
madura su último sumando, si ese instante no supera en `maturity_slack` decisiones de mercado
el último ordinal de la ventana. Ninguna etiqueta entra antes de madurar y ninguna depende
del valor de otra para decidir si existe.

Las etiquetas auxiliares de una decisión se entregan juntas cuando ya no puede completarse
ninguna más: al madurar la ventana del horizonte más largo, que contiene a las demás, al
pasar su plazo o al cerrar el tramo. Así su token se calcula una sola vez para todos los
horizontes. Una etiqueta corta puede esperar hasta `horizonte máximo + maturity_slack`
decisiones, siempre después de madurar.

Por flujo hay como mucho `horizonte máximo + maturity_slack` decisiones guardadas. El
registro se reinicia en cada recorrido, así que una ventana nunca cruza la frontera de un
tramo.
"""

import math

import numpy as np
import torch

from mars_titan.data.input_policy import MODALITIES

_FIELDS = (*MODALITIES, "presence")
AT, LABEL, VALUES, COMPLETED = range(4)


def _market(flow):
    return flow.split("/", 1)[0]


class HorizonLabels:
    def __init__(self, config):
        self.config = config
        self.markets = {}
        # flujo -> {ordinal: [decisión, etiqueta, entradas, {cabeza: objetivo}]}. Las entradas
        # valen None cuando la decisión ya entregó sus etiquetas auxiliares.
        self.flows = {}
        self.keys = {}
        # (entradas, cabezas, objetivos) de cada decisión lista para el término auxiliar.
        self.ready = []

    def observe(self, markets):
        """Avanzar una decisión en cada mercado con entradas en este evento."""
        for market in sorted(set(markets)):
            self.markets[market] = self.markets.get(market, -1) + 1

    def remember(self, cpu):
        """Guardar las decisiones emitidas de un bloque CPU validado, fila a fila."""
        span = self.config.span
        # Una copia por bloque. Sus filas son vistas que se liberan juntas, porque comparten
        # ordinal y plazo.
        copied = {name: np.array(cpu.inputs[name]) for name in MODALITIES}
        copied["presence"] = np.array(cpu.presence)
        for row, (flow, at) in enumerate(zip(cpu.flow_ids, cpu.prediction_at, strict=True)):
            ordinal = self.markets[_market(flow)]
            entries = self.flows.setdefault(flow, {})
            if ordinal in entries:
                raise ValueError("El flujo ya decidió en este ordinal de su mercado")
            entries[ordinal] = [at, None, {name: copied[name][row] for name in _FIELDS}, {}]
            self.keys[flow, at] = ordinal
            # Pasado su plazo, una decisión ya no completa ninguna ventana.
            for old in [s for s in entries if s < ordinal - span]:
                entry = entries.pop(old)
                self._deliver(entry)
                self.keys.pop((flow, entry[AT]), None)

    def mature(self, flow, at, value):
        """Registrar una etiqueta principal y devolver cuántas etiquetas auxiliares completa."""
        ordinal = self.keys.pop((flow, at), None)
        if ordinal is None:
            return 0
        entries = self.flows[flow]
        entries[ordinal][LABEL] = value
        now, slack = self.markets[_market(flow)], self.config.maturity_slack
        horizons = self.config.horizons

        def labelled(position):
            entry = entries.get(position)
            return entry is not None and entry[LABEL] is not None

        # Etiquetas maduras consecutivas a cada lado, como mucho hasta el horizonte más largo.
        # Una ventana que contiene el ordinal está completa si cabe entre esos dos extremos.
        before = after = 0
        while before < horizons[-1] - 1 and labelled(ordinal - before - 1):
            before += 1
        while after < horizons[-1] - 1 and labelled(ordinal + after + 1):
            after += 1
        added = 0
        for head, horizon in enumerate(horizons):
            first = max(ordinal - before, ordinal - horizon + 1)
            for start in range(first, min(ordinal, ordinal + after - horizon + 1) + 1):
                if start + horizon - 1 + slack < now:
                    continue
                window = [entries[start + j] for j in range(horizon)]
                # Suma en orden de ordinal con fsum, igual en cualquier reanudación.
                window[0][COMPLETED][head] = math.fsum(entry[LABEL] for entry in window)
                added += 1
                if head == len(horizons) - 1:
                    # La ventana más larga contiene a las demás de su decisión.
                    self._deliver(window[0])
        return added

    def _deliver(self, entry):
        if entry[VALUES] is not None and entry[COMPLETED]:
            heads = sorted(entry[COMPLETED])
            self.ready.append((entry[VALUES], heads, [entry[COMPLETED][h] for h in heads]))
        entry[VALUES] = None

    def flush(self):
        """Entregar al cerrar el tramo las etiquetas auxiliares que ya no pueden crecer."""
        for flow in sorted(self.flows):
            for _, entry in sorted(self.flows[flow].items()):
                self._deliver(entry)

    def take(self):
        ready, self.ready = self.ready, []
        return ready

    def export(self):
        """Estado CPU en la barrera posterior a un paso, sin etiquetas auxiliares pendientes."""
        if self.ready:
            raise ValueError("Las etiquetas auxiliares se consumen antes de confirmar el estado")
        rows = [
            (flow, ordinal, entry)
            for flow in sorted(self.flows)
            for ordinal, entry in sorted(self.flows[flow].items())
        ]
        stored = [entry[VALUES] for _, _, entry in rows if entry[VALUES] is not None]
        payload = dict(
            schema_version=1,
            markets=[[market, ordinal] for market, ordinal in sorted(self.markets.items())],
            flows=[flow for flow, _, _ in rows],
            ordinals=torch.tensor([ordinal for _, ordinal, _ in rows], dtype=torch.int64),
            decision_at=torch.tensor([entry[AT] for _, _, entry in rows], dtype=torch.int64),
            matured=torch.tensor([entry[LABEL] is not None for _, _, entry in rows]),
            labels=torch.tensor(
                [math.nan if entry[LABEL] is None else entry[LABEL] for _, _, entry in rows],
                dtype=torch.float64,
            ),
            stored=torch.tensor([entry[VALUES] is not None for _, _, entry in rows]),
            completed=[sorted(entry[COMPLETED].items()) for _, _, entry in rows],
            pending=sorted([flow, at] for flow, at in self.keys),
        )
        if stored:
            payload["inputs"] = {
                name: torch.from_numpy(np.stack([values[name] for values in stored]))
                for name in _FIELDS
            }
        return payload

    @classmethod
    def restore(cls, config, payload):
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("El registro de horizontes guardado no corresponde a su esquema")
        result = cls(config)
        result.markets = {market: int(ordinal) for market, ordinal in payload["markets"]}
        inputs, position = payload.get("inputs", {}), 0
        for index, flow in enumerate(payload["flows"]):
            values = None
            if payload["stored"][index]:
                values = {name: inputs[name][position].numpy().copy() for name in _FIELDS}
                position += 1
            label = float(payload["labels"][index]) if payload["matured"][index] else None
            completed = {int(head): float(target) for head, target in payload["completed"][index]}
            ordinal = int(payload["ordinals"][index])
            at = int(payload["decision_at"][index])
            result.flows.setdefault(flow, {})[ordinal] = [at, label, values, completed]
        for flow, at in payload["pending"]:
            entries = result.flows.get(flow, {})
            ordinal = next((s for s, entry in entries.items() if entry[AT] == at), None)
            if ordinal is None or entries[ordinal][LABEL] is not None:
                raise ValueError("Una decisión pendiente no conserva su ordinal guardado")
            result.keys[flow, int(at)] = ordinal
        return result
