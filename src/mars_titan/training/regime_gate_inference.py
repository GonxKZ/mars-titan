"""Conectar la puerta E4 a las escrituras del recorrido cronológico de corrección.

La puerta de cada mercado se calcula con los resultados que maduran en el evento y regula la
tasa η de sus escrituras proximales en ese mismo evento. La regla proximal exige pesos que
sumen uno, así que con varios mercados en un evento la escritura usa η·ḡ, con ḡ la media de
las puertas de sus filas, y pesos proporcionales a cada puerta. Por fila equivale a η·w·g.
Con una sola puerta en el evento los pesos no cambian y la tasa es η·g. El punto que se
compara con la etiqueta es el emitido, y el valor escrito sigue siendo etiqueta menos
núcleo. La recuperación es por fase completa, como en B6 y E3.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from mars_titan.memory import regime_evidence_gate
from mars_titan.memory.regime_evidence_gate import (
    MAX_ROWS,
    RegimeEvidenceGate,
    RegimeGateConfig,
    check_correction,
    gated_write,
)

from .mars_titan_correction import CorrectionInference


def checked_configuration(config, correction):
    """E4 pondera escrituras proximales sin escalas, áreas, roles ni regla kalman."""
    return check_correction(config, correction)


def source_hashes():
    return {
        "mars_titan.training.regime_gate_inference": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "mars_titan.memory.regime_evidence_gate": hashlib.sha256(
            Path(regime_evidence_gate.__file__).read_bytes()
        ).hexdigest(),
    }


def restore_configuration(request):
    """Recuperar la puerta del ancla, sin ignorarla ni aceptar otras fuentes de cálculo."""
    value = request["regime_gate"]
    keys = RegimeGateConfig.__dataclass_fields__
    if type(value) is not dict or set(value) != set(RegimeGateConfig().identity()):
        raise ValueError("La configuración E4 no conserva su esquema")
    config = RegimeGateConfig(**{key: value[key] for key in keys})
    if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(
        config.identity(), sort_keys=True, allow_nan=False
    ):
        raise ValueError("La configuración E4 no conserva su identidad")
    code = request.get("code")
    if type(code) is not dict or any(
        code.get(key) != value for key, value in source_hashes().items()
    ):
        raise ValueError("Las fuentes E4 no coinciden con las del ancla")
    return config


def matched_control(config, report):
    """Control constante con la puerta media de la variante en la validación de su ventana.

    La validación precede a la calibración y a la evaluación, así que el control no usa
    información de los tramos que se comparan. `report` es el `run.json` de la ventana
    `cusum` con la misma configuración.
    """
    if type(config) is not RegimeGateConfig or config.mode != "cusum":
        raise ValueError("El control emparejado E4 parte de una configuración cusum")
    try:
        gate = report["predictions"]["validation"]["metrics"]["regime_gate"]
    except (KeyError, TypeError) as error:
        raise ValueError("La ventana E4 no registra la puerta de su validación") from error
    if gate.get("configuration") != config.identity() or gate.get("mean_gate") is None:
        raise ValueError("La validación E4 no corresponde a esta configuración o no escribió")
    return replace(config, mode="constant", constant=gate["mean_gate"])


class RegimeGateCorrectionInference(CorrectionInference):
    """Ponderar por mercado las escrituras de cada evento, sin reajustar al padre."""

    def __init__(self, predictor, recipe, correction, codec, *, regime_gate, **options):
        self.regime_gate_config = checked_configuration(regime_gate, correction)
        super().__init__(predictor, recipe, correction, codec, **options)
        self._reset_regime_gate()

    def _reset_regime_gate(self):
        self.regime_gate = RegimeEvidenceGate(self.regime_gate_config)
        self._gate_rows, self._gates = [], None
        self.last_gate_decision = None

    def _labels(self, run, event, *, train):
        if not isinstance(event.labels, (tuple, list)) or len(event.labels) > MAX_ROWS:
            raise ValueError("La publicación E4 supera su cohorte madura acotada")
        if self._gate_rows:
            raise ValueError("El evento anterior E4 no ha publicado sus escrituras")
        # run.pending contiene el punto emitido antes de que B6 lo retire al madurar.
        rows = []
        for flow, at, target in event.labels:
            if (flow, at) not in run.pending:
                raise ValueError("La etiqueta E4 no tiene una predicción emitida pendiente")
            rows.append((flow, at, event.at, run.pending[flow, at], target))
        super()._labels(run, event, train=train)
        self._gate_rows = rows

    def _write(self, at):
        if not self._gate_rows:
            return super()._write(at)
        decision, following = self.regime_gate.step(tuple(self._gate_rows), cutoff=at)
        self._gates = decision.by_market()
        try:
            super()._write(at)
        finally:
            self._gates = None
        self.regime_gate = following
        self.last_gate_decision = decision
        self._gate_rows = []

    def _write_memory(self, memory, feedback, staged, at):
        if self._gates is None:
            return super()._write_memory(memory, feedback, staged, at)
        # A conserva su configuración. Solo esta escritura usa la tasa η·g de cada fila.
        gates = [self._gates[flow.split("/", 1)[0]] for _, flow, *_ in staged]
        return gated_write(memory, feedback, gates, cutoff=at)

    def _pass(self, source, events, *, stop=None, rows=None):
        self._reset_regime_gate()
        result = super()._pass(source, events, stop=stop, rows=rows)
        result["regime_gate"] = dict(
            configuration=self.regime_gate_config.identity(),
            publications=self.regime_gate.publications,
            mean_gate=self.regime_gate.mean_gate(),
            state_sha256=self.regime_gate.export()["sha256"],
            reset="at_the_start_of_each_complete_phase",
        )
        return result
