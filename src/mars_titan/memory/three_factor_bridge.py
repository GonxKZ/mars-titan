"""Puente hebbiano de tres factores sobre las áreas de H1 (#784, P1 `three_factor_bridge_v1`).

Las aristas plásticas son los pares de áreas distintas (i, j) que une la topología de la red
holográfica. Cada mercado m tiene un peso rápido w_{m,ij} por arista, que empieza en cero en
cada recorrido y se suma al logit de atención de la consulta i sobre la clave j en todos los
pasos del puente de las filas del mercado. Al emitir la predicción de una fila se guarda su
traza de elegibilidad,

    e_{ij} = (1 / (C · L)) Σ_{c,l} a^{(c,l)}_{ij},

la media de las probabilidades de atención entre las dos áreas en los C ciclos y L pasos
locales. Solo es positiva si las dos áreas están presentes: un área ausente no envía mensajes
y la red anula la traza de su consulta, cuyo estado se descarta. La traza queda junto a la
predicción pendiente y no decae mientras espera (τ_e → ∞).

Cuando madura la etiqueta llega el tercer factor, de una de dos fuentes declaradas. Con la
brecha relativa del error (`relative_error_gap`), ε = |ŷ − y| es el error absoluto de la
predicción emitida y E_m la expectativa del mercado antes del evento,

    δ = clip((E_m − ε) / E_m, −1, 1),

positivo si la predicción falló menos de lo esperado. En el primer evento del mercado, o con
E_m = 0, δ = 0, y después

    E_m ← λ · E_m + (1 − λ) · (1/n) Σ_k ε_k,        λ = exp(−1/τ),   E_m = media en el primero.

Con la señal común de #791 (`mature_directional_hit`), h = sign(ŷ) · sign(y) es el acierto
direccional de la etiqueta y r_m la velocidad más lenta de la señal del mercado antes de que
la señal reciba las etiquetas del evento,

    δ = (h − r_m) / 2,

que ya está en [−1, 1]. En los dos casos, con las n etiquetas maduras del mercado en el evento,

    w_m ← w_m + η · (1/n) Σ_k δ_k · e_k.

Después, la normalización homeostática escala de forma proporcional las aristas que llegan a
cada área para que la suma de sus valores absolutos no pase de la cota B. Todas las etiquetas
del evento usan la expectativa anterior al evento, así que el resultado no depende de su
orden. Las sumas se hacen en FP64 en el orden de las claves (flujo, decisión).

El control `mean_third_factor_v1` usa el mismo puente y las mismas trazas, pero sustituye
cada δ_k por la media de todos los δ calculados hasta el evento en el mercado. Conserva la
escala y el signo medios del tercer factor y retira el mérito de cada predicción.

Es una adaptación propia. La forma de la regla viene de Gerstner y otros (2018,
https://doi.org/10.3389/fncir.2018.00053, ecuaciones 1 y 2), con el tercer factor definido
como desviación de una media móvil. La escala proporcional por neurona viene del escalado
sináptico (Turrigiano, 2008). La traza no es un producto de actividad presináptica y
postsináptica separadas sino la probabilidad de atención realizada, que depende de las dos.
"""

import math

import numpy as np
import torch

from mars_titan.models.titans.plastic_bridge import (
    MATURE_DIRECTIONAL_HIT,
    MEAN_THIRD_FACTOR,
    ThreeFactorBridge,
)

_MARKET_FIELDS = {"weights", "expected", "delta_sum", "delta_count", "events"}


def _real(value, message):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(message)
    return value


def _sign(value):
    return (value > 0) - (value < 0)


def _market(flow):
    if type(flow) is not str or "/" not in flow:
        raise ValueError("Cada fila del puente necesita un flujo con su mercado")
    return flow.split("/", 1)[0]


def _edge_list(edges, areas):
    if (
        type(areas) is not int
        or areas < 2
        or not isinstance(edges, tuple)
        or not edges
        or any(
            type(edge) is not tuple
            or len(edge) != 2
            or any(type(value) is not int or not 0 <= value < areas for value in edge)
            or edge[0] == edge[1]
            for edge in edges
        )
        or list(edges) != sorted(set(edges))
    ):
        raise ValueError("Las aristas plásticas deben ser pares ordenados de áreas distintas")
    return edges


class BridgePlasticity:
    """Pesos rápidos por mercado y trazas pendientes de un recorrido."""

    def __init__(self, config, edges, areas):
        if type(config) is not ThreeFactorBridge:
            raise ValueError("El puente de tres factores necesita su configuración")
        self.config = config
        self.edges = _edge_list(edges, areas)
        self.areas = areas
        self._targets = np.array([i for i, _ in edges], dtype=np.int64)
        self._markets = {}
        self._traces = {}

    def keys(self):
        """Claves (flujo, decisión) de las predicciones con traza pendiente, en orden."""
        return sorted(self._traces)

    def _state(self, market):
        state = self._markets.get(market)
        if state is None:
            state = self._markets[market] = dict(
                weights=np.zeros(len(self.edges)),
                expected=None,
                delta_sum=0.0,
                delta_count=0,
                events=0,
            )
        return state

    def biases(self, flows):
        """Pesos del mercado de cada fila, [filas, aristas] en FP64. Cero si no ha madurado."""
        rows = np.zeros((len(flows), len(self.edges)))
        for index, flow in enumerate(flows):
            state = self._markets.get(_market(flow))
            if state is not None:
                rows[index] = state["weights"]
        return rows

    def hold(self, keys, traces):
        """Guardar la traza de cada predicción emitida hasta que madure su etiqueta."""
        traces = np.asarray(traces, dtype=np.float64)
        if traces.shape != (len(keys), len(self.edges)):
            raise ValueError("Cada predicción emitida necesita su traza en todas las aristas")
        if not np.isfinite(traces).all() or (traces < 0).any() or (traces > 1).any():
            raise ValueError("La traza de elegibilidad debe estar en [0, 1]")
        keys = list(keys)
        for key in keys:
            if type(key) is not tuple or len(key) != 2 or type(key[1]) is not int:
                raise ValueError("Cada traza pendiente necesita su flujo y su instante de decisión")
            _market(key[0])
        # Se comprueba todo antes de guardar nada, para no dejar el estado a medias.
        if len(set(keys)) != len(keys) or any(key in self._traces for key in keys):
            raise ValueError("La predicción ya tiene una traza pendiente")
        for key, trace in zip(keys, traces, strict=True):
            self._traces[key] = trace.copy()

    def _checked_expectations(self, expectations, markets):
        common = self.config.source == MATURE_DIRECTIONAL_HIT
        if not common:
            if expectations is not None:
                raise ValueError("La brecha del error no recibe la señal común")
            return {}
        if (
            not isinstance(expectations, dict)
            or not markets <= set(expectations)
            or any(
                type(expectations[market]) is not float
                or not math.isfinite(expectations[market])
                or not -1.0 <= expectations[market] <= 1.0
                for market in markets
            )
        ):
            raise ValueError("La señal común debe dar el acierto esperado de cada mercado")
        return expectations

    def _third_factor(self, state, error, hit, expected_hit):
        if self.config.source == MATURE_DIRECTIONAL_HIT:
            return (hit - expected_hit) / 2.0
        expected = state["expected"]
        if expected is None or expected == 0.0:
            return 0.0
        return min(1.0, max(-1.0, (expected - error) / expected))

    def settle(self, matured, expectations=None):
        """Aplicar las etiquetas maduras de un evento, cada una (flujo, decisión, ŷ, y).

        Con la señal común, `expectations` da el acierto esperado de cada mercado antes de que
        la señal reciba estas etiquetas.
        """
        checked = []
        for flow, decision_at, issued, outcome in matured:
            if (flow, decision_at) not in self._traces:
                raise ValueError("La etiqueta madura no tiene la traza de su predicción emitida")
            issued = _real(issued, "La predicción emitida debe ser un real finito")
            outcome = _real(outcome, "El resultado maduro debe ser un real finito")
            error = _real(abs(issued - outcome), "El error maduro debe ser finito")
            checked.append(((flow, decision_at), error, float(_sign(issued) * _sign(outcome))))
        if len({key for key, _, _ in checked}) != len(checked):
            raise ValueError("Una predicción solo madura una vez")
        markets = {_market(key[0]) for key, _, _ in checked}
        expectations = self._checked_expectations(expectations, markets)
        # Las trazas se retiran después de comprobar todo el evento.
        groups = {}
        for key, error, hit in checked:
            groups.setdefault(_market(key[0]), []).append((key, error, hit, self._traces.pop(key)))
        for market, labels in sorted(groups.items()):
            labels.sort(key=lambda label: label[0])
            self._update(self._state(market), labels, expectations.get(market))

    def _update(self, state, labels, expected_hit):
        config, count = self.config, len(labels)
        deltas = [
            self._third_factor(state, error, hit, expected_hit) for _, error, hit, _ in labels
        ]
        traces = np.stack([trace for _, _, _, trace in labels])
        state["delta_sum"] = math.fsum([state["delta_sum"], *deltas])
        state["delta_count"] += count
        if config.kind == MEAN_THIRD_FACTOR:
            mean = state["delta_sum"] / state["delta_count"]
            step = mean * np.add.reduce(traces, axis=0) / count
        else:
            step = np.add.reduce(np.array(deltas)[:, None] * traces, axis=0) / count
        weights = state["weights"] + config.learning_rate * step
        # Escalado proporcional de las aristas que llegan a cada área.
        total = np.bincount(self._targets, weights=np.abs(weights), minlength=self.areas)
        factor = np.where(total > config.budget, config.budget / np.maximum(total, 1e-300), 1.0)
        state["weights"] = weights * factor[self._targets]
        if config.source != MATURE_DIRECTIONAL_HIT:
            error = math.fsum(error for _, error, _, _ in labels) / count
            if state["expected"] is None:
                state["expected"] = error
            else:
                decay = config.decay
                state["expected"] = decay * state["expected"] + (1.0 - decay) * error
        state["events"] += 1

    def export(self):
        keys = sorted(self._traces)
        return dict(
            schema_version=1,
            configuration=self.config.identity(),
            areas=self.areas,
            edges=[list(edge) for edge in self.edges],
            markets={
                market: dict(state, weights=torch.from_numpy(state["weights"].copy()))
                for market, state in sorted(self._markets.items())
            },
            pending=dict(
                flows=[flow for flow, _ in keys],
                decision_at=torch.tensor([at for _, at in keys], dtype=torch.int64),
                traces=torch.from_numpy(
                    np.stack([self._traces[key] for key in keys])
                    if keys
                    else np.zeros((0, len(self.edges)))
                ),
            ),
        )

    def _checked(self, state):
        if not isinstance(state, dict) or set(state) != _MARKET_FIELDS:
            raise ValueError("El mercado guardado no conserva sus campos")
        weights, expected = state["weights"], state["expected"]
        count, events = state["delta_count"], state["events"]
        if (
            not isinstance(weights, torch.Tensor)
            or weights.dtype != torch.float64
            or weights.shape != (len(self.edges),)
            or not torch.isfinite(weights).all()
            or type(count) is not int
            or type(events) is not int
            or not 0 < events <= count
            or (expected is None) != (self.config.source == MATURE_DIRECTIONAL_HIT)
        ):
            raise ValueError("El mercado guardado no corresponde al puente")
        delta_sum = _real(state["delta_sum"], "La suma de δ guardada debe ser finita")
        if expected is not None and _real(expected, "La expectativa debe ser finita") < 0:
            raise ValueError("La expectativa debe ser finita")
        if abs(delta_sum) > count:
            raise ValueError("La suma de δ guardada supera su número de etiquetas")
        values = weights.numpy().copy()
        total = np.bincount(self._targets, weights=np.abs(values), minlength=self.areas)
        if (total > self.config.budget * (1 + 1e-12)).any():
            raise ValueError("Los pesos guardados superan la cota de su área")
        return dict(
            weights=values,
            expected=expected,
            delta_sum=delta_sum,
            delta_count=count,
            events=events,
        )

    @classmethod
    def restore(cls, config, edges, areas, payload):
        bridge = cls(config, edges, areas)
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {"schema_version", "configuration", "areas", "edges", "markets", "pending"}
            or payload["schema_version"] != 1
            or payload["configuration"] != config.identity()
            or payload["areas"] != areas
            or payload["edges"] != [list(edge) for edge in bridge.edges]
            or not isinstance(payload["markets"], dict)
        ):
            raise ValueError("El estado guardado no corresponde al puente de tres factores")
        for market, state in payload["markets"].items():
            if type(market) is not str or not market or "/" in market:
                raise ValueError("El mercado guardado no tiene un nombre válido")
            bridge._markets[market] = bridge._checked(state)
        pending = payload["pending"]
        if not isinstance(pending, dict) or set(pending) != {"flows", "decision_at", "traces"}:
            raise ValueError("Las trazas pendientes guardadas no conservan sus campos")
        flows, moments, traces = pending["flows"], pending["decision_at"], pending["traces"]
        if (
            not isinstance(flows, list)
            or not isinstance(moments, torch.Tensor)
            or moments.dtype != torch.int64
            or moments.shape != (len(flows),)
            or not isinstance(traces, torch.Tensor)
            or traces.dtype != torch.float64
            or traces.shape != (len(flows), len(bridge.edges))
        ):
            raise ValueError("Las trazas pendientes guardadas no corresponden al puente")
        keys = list(zip(flows, moments.tolist(), strict=True))
        if keys != sorted(set(keys)):
            raise ValueError("Las trazas pendientes guardadas no están ordenadas sin repetir")
        bridge.hold(keys, traces.numpy())
        return bridge
