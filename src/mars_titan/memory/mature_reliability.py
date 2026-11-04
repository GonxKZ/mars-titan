"""Señal común de fiabilidad madura a varias velocidades (#791, `mature_reliability_signal_v1`).

Sea ŷ una predicción realmente emitida e y su resultado cuando madura. El acierto direccional
es h = sign(ŷ) · sign(y), que vale 0 si alguno de los dos es nulo. La puntuación de un evento
para el mercado m y el canal c es la media de los aciertos de sus etiquetas maduras,

    s_{m,c}(t) = (1/n) Σ_i h_i,

y cada velocidad k la promedia con λ_k = exp(−1/τ_k):

    v_{m,c,k}(t) = λ_k · v_{m,c,k}(t − 1) + (1 − λ_k) · s_{m,c}(t),        v = 0 al empezar.

La señal solo cambia con etiquetas maduras de predicciones emitidas. Un evento sin etiquetas del
mercado no la cambia, así que τ cuenta eventos de maduración y no sesiones del calendario. El
recorrido aplica las etiquetas de un instante antes de emitir sus decisiones, de modo que una
decisión del instante t lee lo madurado hasta t. Con un solo canal, todos los consumidores
leen la misma señal.

Controles declarados:

- `shuffled_in_time`. En la actualización n de un canal, s se sustituye por la puntuación de
  una de las n + 1 actualizaciones hechas hasta entonces, elegida al azar con una semilla que
  deriva de la declarada, el mercado, el canal y n. Conserva la distribución de las
  puntuaciones y rompe su orden sin usar el futuro.
- `titans_surprise`. La puntuación sale de la pérdida asociativa ℓ de Titans en las filas
  emitidas del mercado. Con ℓ̄ su media en el evento y r su promedio a la velocidad más lenta,
  s = (r − ℓ̄)/(r + ℓ̄). Vale 0 en el primer evento y es positiva cuando la sorpresa queda por
  debajo de su nivel habitual. Se registra al emitir, sin esperar a ninguna etiqueta, y la
  leen las decisiones del evento siguiente.
- `independent`. Cada consumidor tiene su canal y solo recibe las etiquetas de los flujos que
  le asigna el reparto de la semilla. La esperanza de las señales coincide y su ruido es
  independiente, aunque cada canal usa menos etiquetas.

Los valores se guardan en FP64 y las medias usan `math.fsum`, que redondea una sola vez. Por
eso no dependen del orden de las filas ni de su reparto en bloques.
"""

import hashlib
import math

import numpy as np

from mars_titan.models.titans.reliability_gates import MatureReliability

_FIELDS = {"values", "updates", "history", "reference"}


def _sign(value):
    return (value > 0) - (value < 0)


def _market(flow):
    return flow.split("/", 1)[0]


def _key(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def _real(value, message):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(message)
    return value


def _scores(values, length):
    return (
        isinstance(values, list)
        and len(values) == length
        and all(type(value) is float and -1.0 <= value <= 1.0 for value in values)
    )


class ReliabilitySignal:
    """Acumuladores por mercado y canal. Se reinicia con el estado rápido de cada recorrido."""

    def __init__(self, config):
        if type(config) is not MatureReliability:
            raise ValueError("La señal de fiabilidad necesita su configuración")
        self.config = config
        self._markets = {}

    def _channels(self, market):
        channels = self._markets.get(market)
        if channels is None:
            speeds = len(self.config.time_constants)
            channels = self._markets[market] = [
                dict(values=[0.0] * speeds, updates=0, history=[], reference=None)
                for _ in range(self.config.channels)
            ]
        return channels

    def channel(self, flow):
        """Canal que recibe las etiquetas del flujo. Con la señal compartida es siempre 0."""
        if not self.config.independent:
            return 0
        return _key(f"{self.config.seed}/{flow}") % self.config.channels

    def _groups(self, rows):
        groups = {}
        for flow, value in rows:
            if type(flow) is not str or "/" not in flow:
                raise ValueError("Cada fila de la señal necesita un flujo con su mercado")
            groups.setdefault((_market(flow), self.channel(flow)), []).append(value)
        return sorted(groups.items())

    def _update(self, market, channel, score):
        state = self._channels(market)[channel]
        if self.config.source == "shuffled_in_time":
            state["history"].append(score)
            count = state["updates"]
            seed = np.random.SeedSequence([self.config.seed, _key(market), channel, count])
            score = state["history"][int(np.random.default_rng(seed).integers(count + 1))]
        values = state["values"]
        for index, decay in enumerate(self.config.decays):
            values[index] = decay * values[index] + (1.0 - decay) * score
        state["updates"] += 1

    def settle(self, matured):
        """Aplicar las etiquetas maduras de un evento, cada una (flujo, emitida, resultado)."""
        if self.config.source == "titans_surprise":
            return
        hits = []
        for flow, issued, outcome in matured:
            issued = _real(issued, "La predicción emitida debe ser un real finito")
            outcome = _real(outcome, "El resultado maduro debe ser un real finito")
            hits.append((flow, float(_sign(issued) * _sign(outcome))))
        for (market, channel), values in self._groups(hits):
            self._update(market, channel, math.fsum(values) / len(values))

    def observe(self, flows, losses):
        """Registrar la pérdida asociativa de las filas emitidas de un evento en su control."""
        if self.config.source != "titans_surprise":
            raise ValueError("Solo el control de sorpresa registra la pérdida asociativa")
        if len(flows) != len(losses):
            raise ValueError("Cada fila emitida necesita su pérdida asociativa")
        for loss in losses:
            if _real(loss, "La pérdida asociativa debe ser un real finito") < 0:
                raise ValueError("La pérdida asociativa debe ser un real finito")
        slowest = self.config.decays[-1]
        for (market, channel), values in self._groups(zip(flows, losses, strict=True)):
            state = self._channels(market)[channel]
            mean = math.fsum(values) / len(values)
            reference = mean if state["reference"] is None else state["reference"]
            total = reference + mean
            score = 0.0 if total == 0.0 else (reference - mean) / total
            state["reference"] = slowest * reference + (1.0 - slowest) * mean
            self._update(market, channel, score)

    def values(self, flows):
        """Señal de cada fila como [filas][consumidores][velocidades] en FP64."""
        consumers = len(self.config.consumers)
        neutral = [0.0] * len(self.config.time_constants)
        rows = []
        for flow in flows:
            channels = self._markets.get(_market(flow))
            if channels is None:
                rows.append([list(neutral) for _ in range(consumers)])
            elif self.config.independent:
                rows.append([list(state["values"]) for state in channels])
            else:
                rows.append([list(channels[0]["values"]) for _ in range(consumers)])
        return rows

    def export(self):
        markets = {
            market: [
                dict(state, values=list(state["values"]), history=list(state["history"]))
                for state in channels
            ]
            for market, channels in sorted(self._markets.items())
        }
        return dict(schema_version=1, configuration=self.config.identity(), markets=markets)

    def _checked(self, state):
        config = self.config
        shuffled = config.source == "shuffled_in_time"
        surprise = config.source == "titans_surprise"
        if not isinstance(state, dict) or set(state) != _FIELDS:
            raise ValueError("El canal guardado no conserva sus campos")
        updates, reference = state["updates"], state["reference"]
        if (
            type(updates) is not int
            or updates < 0
            or not _scores(state["values"], len(config.time_constants))
            or not _scores(state["history"], updates if shuffled else 0)
            or (reference is not None) != (surprise and updates > 0)
        ):
            raise ValueError("El canal guardado no corresponde a la configuración de la señal")
        if reference is not None and _real(reference, "La referencia debe ser finita") < 0:
            raise ValueError("La referencia debe ser finita")
        return dict(
            values=list(state["values"]),
            updates=updates,
            history=list(state["history"]),
            reference=reference,
        )

    @classmethod
    def restore(cls, config, payload):
        signal = cls(config)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "configuration", "markets"}
            or payload["schema_version"] != 1
            or payload["configuration"] != config.identity()
            or not isinstance(payload["markets"], dict)
        ):
            raise ValueError("El estado guardado no corresponde a la señal de fiabilidad")
        for market, channels in payload["markets"].items():
            if (
                type(market) is not str
                or not market
                or not isinstance(channels, list)
                or len(channels) != config.channels
            ):
                raise ValueError("El mercado guardado no conserva sus canales")
            signal._markets[market] = [signal._checked(state) for state in channels]
        return signal
