"""Conectar E3 a emisiones y escrituras del recorrido cronológico de corrección.

Cada fase usa las cohortes completas que entrega FinancialObservationSource. La
recuperación es por fase completa, como en B6. No publica estados de FinancialSession.
"""

import hashlib
import json
from pathlib import Path

from mars_titan.memory import salience_budget
from mars_titan.memory.associative_memory import MatureCorrection
from mars_titan.memory.salience_budget import (
    MAX_ROWS,
    MatureWrite,
    MatureWriteBudget,
    SalienceBudgetConfig,
)

from .mars_titan_correction import CorrectionInference


def checked_configuration(config, correction):
    if (
        type(config) is not SalienceBudgetConfig
        or type(correction) is not MatureCorrection
        or correction.memory.rule not in ("delta", "proximal")
    ):
        raise ValueError("La escritura E3 necesita su configuración y un corrector delta/proximal")
    return config


def source_hashes():
    return {
        "mars_titan.training.salience_inference": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "mars_titan.memory.salience_budget": hashlib.sha256(
            Path(salience_budget.__file__).read_bytes()
        ).hexdigest(),
    }


def restore_configuration(request):
    """Recuperar el factor del ancla, sin ignorarlo ni aceptar otras fuentes de cálculo."""
    if "salience" not in request:
        return None
    value = request["salience"]
    keys = SalienceBudgetConfig.__dataclass_fields__
    if type(value) is not dict or set(value) != set(SalienceBudgetConfig().identity()):
        raise ValueError("La configuración E3 no conserva su esquema")
    config = SalienceBudgetConfig(**{key: value[key] for key in keys})
    if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(
        config.identity(), sort_keys=True, allow_nan=False
    ):
        raise ValueError("La configuración E3 no conserva su identidad")
    code = request.get("code")
    if type(code) is not dict or any(
        code.get(key) != value for key, value in source_hashes().items()
    ):
        raise ValueError("Las fuentes E3 no coinciden con las del ancla")
    return config


class SalienceCorrectionInference(CorrectionInference):
    """Aplicar el cupo E3 después de emitir cada evento, sin reajustar al padre."""

    def __init__(self, predictor, recipe, correction, codec, *, salience, **options):
        self.salience_config = checked_configuration(salience, correction)
        super().__init__(predictor, recipe, correction, codec, **options)
        self._reset_salience()

    def _reset_salience(self):
        self.salience_budget = MatureWriteBudget(self.salience_config)
        self._salience_rows = []
        self.last_salience_selection = None

    def _labels(self, run, event, *, train):
        if not isinstance(event.labels, (tuple, list)) or len(event.labels) > MAX_ROWS:
            raise ValueError("La publicación E3 supera su cohorte madura acotada")
        if self._salience_rows:
            raise ValueError("El evento anterior E3 no ha publicado sus escrituras")
        # run.pending contiene el punto final que se emitió, antes de que B6 lo retire.
        # _core conserva otra señal: la predicción sin corrección para el valor de escritura.
        rows = []
        for flow, at, target in event.labels:
            if (flow, at) not in run.pending:
                raise ValueError("La etiqueta E3 no tiene una predicción emitida pendiente")
            rows.append(MatureWrite(flow, at, event.at, run.pending[flow, at], target))
        super()._labels(run, event, train=train)
        self._salience_rows = rows

    def _write(self, at):
        if not self._salience_rows:
            return super()._write(at)
        selected, following = self.salience_budget.select(tuple(self._salience_rows), cutoff=at)
        keys = set(selected.selected)
        self._staged = [row for row in self._staged if (row[1], row[0]) in keys]
        # E1 retiró los componentes del pendiente al madurar. Los no elegidos tampoco
        # pueden quedarse retenidos indefinidamente en la cola de escritura.
        for key in set(self._mature_experts) - keys:
            del self._mature_experts[key]
        super()._write(at)
        self.salience_budget = following
        self.last_salience_selection = selected
        self._salience_rows = []

    def _pass(self, source, events, *, stop=None, rows=None):
        self._reset_salience()
        result = super()._pass(source, events, stop=stop, rows=rows)
        result["salience"] = dict(
            configuration=self.salience_config.identity(),
            observed=self.salience_budget.observed,
            selected=self.salience_budget.selected,
            publications=self.salience_budget.cohorts,
            state_sha256=self.salience_budget.export()["sha256"],
            reset="at_the_start_of_each_complete_phase",
        )
        return result
