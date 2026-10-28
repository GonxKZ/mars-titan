"""Corrección asociativa B6 de MARS-TITAN en las ventanas walk-forward.

B6 suma a la predicción del Titans-MAC `mac_online` elegido en la ventana la lectura kᵀA de
una memoria asociativa lineal escrita solo con resultados maduros
(`memory.associative_memory.MatureCorrection`). No tiene parámetros que ajustar. Con las reglas
delta y proximal, los casos de búsqueda de su receta fijan η, la receta fija λ y la campaña
elige el caso con el MAE por sesión de validación, igual que en los demás modelos.

La regla kalman (PT3, #455) tiene su propia receta. Sus casos fijan κ = q/σ² y la receta p0/σ²,
el tope de ϱ y los meses del tramo de estimación. σ² y ϱ se estiman en cada ventana con los
residuos etiqueta menos núcleo de los últimos meses del tramo de ajuste del padre
(`memory.cohort_noise`), antes de validación, calibración y evaluación. Como p0 y q escalan
con σ² y la ventana no usa Huber, la media de la corrección no depende de σ².

Cada ventana recibe la vista, la carpeta de la ventana Titans-MAC `mac_online` completada en
la misma vista y semilla (el padre), la receta de la corrección, la regla y la clave de
`associative_memory`, la semilla y el caso elegido.

Cada tramo medido repite la inferencia cronológica del padre con su receta, su calentamiento
y la memoria rápida inicial, y A empieza en cero en cada recorrido. Dentro de un evento se
emiten primero las predicciones con la A confirmada antes del evento y después se escriben
las etiquetas que maduran en él, con el valor etiqueta menos predicción del núcleo y en el
orden canónico de decisión y flujo. Es el mismo contrato que `FinancialSession`, y por eso
las dos rutas pueden compararse bit a bit.

La ventana guarda `run.json`, `selected.json` con el padre y la corrección, y un Parquet por
tramo con las columnas y cuantiles de `training.titans_walk_forward`. La corrección desplaza
todos los cuantiles emitidos en la misma cantidad que el punto, así que no cambia su anchura.

Con la regla kalman el Parquet lleva dos columnas más para la regla de decisión de PT3: la
varianza predictiva kᵀPk + σ² con el estado confirmado antes del evento, y la predicción que
daría otra memoria igual que escribe cada etiqueta una sesión más tarde (D3), en la primera
decisión de su mercado posterior a su madurez y después de emitirla. Las dos memorias se
recorren en la misma pasada, así que D3 no repite la inferencia del padre.

No hay tramo de ajuste, de modo que el índice de entrenamiento no se construye ni se lee.
Los errores que escriben A proceden siempre de predicciones emitidas en el propio tramo,
nunca del ajuste del padre. Un tramo interrumpido se repite desde su inicio, porque A y la
memoria rápida se reinician en cada recorrido, y los tramos confirmados conservan su huella.
"""

import hashlib
import importlib
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.learning_traces.hooks import (
    EMPTY_CURSOR,
    MAX_PENDING,
    open_recorder,
    trace_identity,
)
from mars_titan.memory.associative_memory import MatureCorrection
from mars_titan.memory.cohort_noise import estimate_cohort_noise
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_observations import FinancialObservationSource
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.memory.kernel_memory import KERNEL_RULE
from mars_titan.memory.lesion_ledger import LesionLedger
from mars_titan.memory.lesion_protocol import LesionProtocol
from mars_titan.memory.mars_titan_variant import (
    check_components,
    core_identity,
    load_declaration,
    select_variant,
)
from mars_titan.memory.regimes import SLOTS
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.financial_inputs import (
    MODALITIES,
    FinancialInputSpec,
    validated_cpu_batch,
)

from .checkpoints import StopRequest
from .corpus_inputs import CorpusDataset
from .financial_run import ChronologicalInference, ChronologicalRecipe, Paused, _compatible, _Pass
from .kernel_policy import declared_policy, require_policy
from .learning_hold import require_learning_allowed
from .search_cases import CASE_NAME
from .titans_walk_forward import (
    PREDICTED,
    PredictionRows,
    _check_view,
    _new_destination,
    _protocol,
    _require,
    _sources,
    _verify,
    case_recipe,
    checked_tables,
    memory_policy,
    view_protocol,
    walk_forward_options,
    window_phases,
)
from .walk_forward_phases import micros, months_before
from .window_indices import index_root

KIND = "mars_titan_correction_window"
CARRY_KIND = "mars_titan_correction_carried_predictions"
SELECTED_KIND = "mars_titan_correction_selected_state"
RECIPE = "mars_titan_mature_correction_v1"
KALMAN_RECIPE = "mars_titan_kalman_correction_v1"
# Las reglas delta y proximal solo dependen de η y λ. Cada valor se declara una sola vez, en la
# receta o en los casos de búsqueda, para que elegir un caso nunca contradiga a la receta.
SEARCHED = ("rate", "forgetting")
# La regla kalman busca κ = q/σ² y fija en la receta p0/σ², el tope de ϱ y los meses del tramo
# de estimación. σ² y ϱ no se declaran: se estiman en cada ventana.
KALMAN_SEARCHED = ("process_ratio",)
KALMAN_FIXED = ("prior_ratio", "correlation_cap", "estimation_months")
# El control de Nadaraya-Watson de B1 busca su ancho τ y fija en la receta λ y la capacidad.
KERNEL_RECIPE = "mars_titan_kernel_correction_v1"
KERNEL_SEARCHED = ("bandwidth",)
KERNEL_FIXED = ("forgetting", "capacity")
_RECIPES = {
    RECIPE: (SEARCHED, ()),
    KALMAN_RECIPE: (KALMAN_SEARCHED, KALMAN_FIXED),
    KERNEL_RECIPE: (KERNEL_SEARCHED, KERNEL_FIXED),
}
# Un modelo solo declara la regla y la clave. η y λ se añaden con el caso elegido, porque la
# campaña los busca y no forman parte de la definición del modelo. Un modelo kalman declara
# además si usa la ϱ estimada o la anula, que es la alternativa con la que se descarta.
ARM_FIELDS = ("rule", "key")
KALMAN_ARM_FIELDS = (*ARM_FIELDS, "cohort_correlation")
COHORT_CORRELATIONS = ("estimated", "zero")
RULES = ("delta", "proximal", "kalman")
# El control nuevo es explícito y no amplía las reglas de la campaña histórica.
CORRECTION_BACKENDS = (*RULES, KERNEL_RULE)
# σ² y ϱ con los que se comprueba una combinación kalman antes de abrir fuentes y se mide su
# caudal. La media de la corrección no depende de σ² y el coste no depende de ninguno.
UNIT_NOISE = dict(observation_noise=1.0, cohort_correlation=0.0)
# Columnas que añade la regla kalman a sus predicciones (PT3 y D3, #455).
KALMAN_COLUMNS = ("predictive_variance", "prediction_label_delay1")
KALMAN_DIAGNOSTICS = dict(
    predictive_variance="k_P_k_plus_sigma2_with_the_state_confirmed_before_the_event",
    label_delay="second_memory_writes_each_label_at_the_next_decision_of_its_market_after_"
    "maturity_after_emitting",
    label_delay_sessions=1,
)
MAX_ESTIMATION_MONTHS = 12
_RECIPE_FIELDS = {"schema_version", "recipe_name", "status", "recipe", "walk_forward", "pending"}
_CODE = (
    "mars_titan.training.mars_titan_correction",
    "mars_titan.training.financial_run",
    "mars_titan.training.titans_walk_forward",
    "mars_titan.training.mars_titan_walk_forward",
    "mars_titan.memory.mars_titan_variant",
    "mars_titan.memory.associative_memory",
    "mars_titan.memory.area_memory",
    "mars_titan.memory.area_query_bridge",
    "mars_titan.memory.paired_proximal",
    "mars_titan.memory.role_binding",
    "mars_titan.memory.multiscale_mixture",
    "mars_titan.memory.regimes",
    "mars_titan.memory.kernel_memory",
    "mars_titan.memory.cohort_noise",
    "mars_titan.memory.episodic_codec",
    "mars_titan.data.batches",
)


def _code():
    return {name: sha256(Path(importlib.import_module(name).__file__)) for name in _CODE}


def _lesion_code():
    return {
        kind.__module__: sha256(Path(importlib.import_module(kind.__module__).__file__))
        for kind in (LesionProtocol, LesionLedger)
    }


def diagnostic_protocol(request):
    """Restaurar B4 únicamente con la política y fuentes que registró el ancla."""
    if "lesions" not in request:
        return None
    protocol = LesionProtocol.from_identity(request["lesions"])
    _require(
        isinstance(request.get("code"), dict)
        and all(request["code"].get(name) == digest for name, digest in _lesion_code().items()),
        "Las fuentes del diagnóstico B4 han cambiado",
    )
    return protocol


def load_correction_recipe(path):
    """Leer la receta de B6 y comprobar cómo reparte sus valores entre la base y sus casos.

    La receta de las reglas delta y proximal declara η y λ. La de la regla kalman declara κ en
    sus casos y los valores de `KALMAN_FIXED` en su base. Todos los casos sustituyen los mismos
    valores buscados, que por eso no pueden aparecer también en `recipe`. La receta no tiene
    épocas ni optimizador porque B6 no ajusta parámetros.
    """
    path = Path(path)
    _require(
        not path.is_symlink() and path.is_file() and path.stat().st_size <= 64 * 1024,
        "La receta de la corrección no es un archivo regular de hasta 64 KiB",
    )
    document, _ = read_manifest(path, 64 * 1024)
    cases = (document.get("walk_forward") or {}).get("search_cases") if document else None
    base = document.get("recipe") if isinstance(document, dict) else None
    name = document.get("recipe_name") if isinstance(document, dict) else None
    searched, fixed = _RECIPES.get(name, ((), ()))
    valid = (
        isinstance(document, dict)
        and set(document) == _RECIPE_FIELDS
        and document["schema_version"] == 1
        and name in _RECIPES
        and isinstance(base, dict)
        and set(document["walk_forward"]) == {"search_cases"}
        and isinstance(cases, dict)
        and 1 <= len(cases) <= 3
        and all(isinstance(name, str) and CASE_NAME.fullmatch(name) for name in cases)
        and all(isinstance(case, dict) and case for case in cases.values())
    )
    keys = {frozenset(case) for case in cases.values()} if valid else set()
    _require(
        valid
        and len(keys) == 1
        and not next(iter(keys)) & set(base)
        and next(iter(keys)) <= set(searched)
        and next(iter(keys)) | set(base) == {*searched, *fixed}
        and len({canonical(case) for case in cases.values()}) == len(cases),
        "La receta de la corrección declara η y λ, o κ y los valores fijos de kalman, entre su "
        "base y de uno a tres casos distintos con nombre válido que sustituyen los mismos valores",
    )
    if name == KALMAN_RECIPE:
        for case in cases.values():
            _kalman_values(dict(base, **case))
    return document


def _kalman_values(values):
    """Comprobar κ, p0/σ², el tope de ϱ y los meses de estimación antes de abrir fuentes."""
    process, prior, cap = (
        values[name] for name in ("process_ratio", "prior_ratio", "correlation_cap")
    )
    months = values["estimation_months"]
    _require(
        all(type(value) is float and math.isfinite(value) for value in (process, prior, cap))
        and process >= 0.0
        and prior > 0.0
        and 0.0 < cap < 1.0
        and type(months) is int
        and 1 <= months <= MAX_ESTIMATION_MONTHS,
        "La receta kalman necesita κ ≥ 0, p0/σ² > 0, un tope de ϱ en (0, 1) y de 1 a "
        f"{MAX_ESTIMATION_MONTHS} meses de estimación",
    )


def case_values(document, search_case):
    """Devolver los valores del caso elegido y rechazar un caso que la receta no declare."""
    cases = document["walk_forward"]["search_cases"]
    _require(
        isinstance(search_case, str) and search_case in cases,
        "Elige uno de los casos de búsqueda que declara la receta de la corrección",
    )
    return dict(document["recipe"], **cases[search_case])


def arm_components(components, values, noise=None):
    """Completar la combinación de un modelo B6 con los valores del caso.

    El modelo solo puede declarar la memoria asociativa. Con un banco añadido la ventana
    mezclaría dos vías de memoria y A5 dejaría de aislar el efecto de la corrección. Las reglas
    delta y proximal reciben η y λ, también con las áreas de B1. El control de Nadaraya-Watson
    recibe τ, λ y la capacidad de su receta. La regla kalman recibe además `noise`, con la σ² y
    la ϱ estimadas en la ventana: q = κσ², p0 = (p0/σ²)σ² y ϱ estimada o nula según el modelo.
    """
    memory = components.get("associative_memory") if isinstance(components, dict) else None
    rule = memory.get("rule") if isinstance(memory, dict) else None
    fields = KALMAN_ARM_FIELDS if rule == "kalman" else ARM_FIELDS
    # Solo las reglas delta y proximal admiten el reparto por áreas de B1 en el modelo.
    optional = (
        {"areas", "query_bridge", "role_binding", "multiscale"}
        if rule in ("delta", "proximal")
        else set()
    )
    _require(
        isinstance(components, dict)
        and set(components) == {"associative_memory"}
        and isinstance(memory, dict)
        and set(fields) <= set(memory) <= set(fields) | optional,
        "Un modelo B6 declara solo associative_memory con su regla y su clave, y con la regla "
        "kalman también su correlación de cohorte, o con delta y proximal las áreas de B1",
    )
    _require(
        rule in CORRECTION_BACKENDS,
        "La ventana B6 solo admite las reglas delta, proximal, kalman o nadaraya_watson",
    )
    if rule == KERNEL_RULE:
        _require(
            set(values) == {*KERNEL_SEARCHED, *KERNEL_FIXED} and noise is None,
            "Nadaraya-Watson usa la receta de τ, λ y capacidad, sin varianzas estimadas",
        )
        return dict(associative_memory=dict(memory, **values))
    if rule != "kalman":
        _require(
            set(values) == set(SEARCHED) and noise is None,
            "Las reglas delta y proximal usan la receta de η y λ, sin varianzas estimadas",
        )
        return dict(associative_memory=dict(memory, **values))
    _require(
        set(values) == {*KALMAN_SEARCHED, *KALMAN_FIXED}
        and memory["cohort_correlation"] in COHORT_CORRELATIONS
        and isinstance(noise, dict)
        and set(UNIT_NOISE) <= set(noise),
        "La regla kalman usa la receta de sus varianzas, declara su correlación de cohorte "
        "estimada o nula y necesita la σ² y la ϱ de la ventana",
    )
    variance = noise["observation_noise"]
    estimated = memory["cohort_correlation"] == "estimated"
    return dict(
        associative_memory=dict(
            rule="kalman",
            key=memory["key"],
            process_noise=values["process_ratio"] * variance,
            observation_noise=variance,
            cohort_correlation=noise["cohort_correlation"] if estimated else 0.0,
            prior_variance=values["prior_ratio"] * variance,
            huber_threshold=None,
        )
    )


def kalman_arm(components):
    """Indicar si el modelo B6 usa la regla kalman, que estima sus varianzas en la ventana."""
    memory = components.get("associative_memory") if isinstance(components, dict) else None
    return isinstance(memory, dict) and memory.get("rule") == "kalman"


class CorrectionInference(ChronologicalInference):
    """Inferencia cronológica del padre congelado con la corrección B6.

    El cálculo del núcleo es exactamente el de `ChronologicalInference`. Tras emitir cada
    bloque se suma kᵀA con la A del evento anterior y, después de las predicciones del
    evento, se escriben las etiquetas que maduran en él. La memoria rápida del padre no
    recibe ninguna etiqueta, porque la sorpresa asociativa de Titans y el error financiero
    maduro son señales distintas.
    """

    def __init__(self, predictor, recipe, correction, codec, *, audit=False, lesions=None):
        if type(predictor) is not FinancialPredictor or type(recipe) is not ChronologicalRecipe:
            raise ValueError("La corrección necesita el padre Titans-MAC y su receta cronológica")
        if predictor.config.variant != "mac_online" or predictor.local_control is not None:
            raise ValueError("B6 parte de Titans-MAC mac_online sin control local")
        if predictor.training or any(value.requires_grad for value in predictor.parameters()):
            raise ValueError("El padre Titans-MAC debe llegar congelado, en eval y sin gradientes")
        if type(correction) is not MatureCorrection:
            raise ValueError("La corrección necesita su memoria asociativa identificada")
        if (
            type(codec) is not FrozenEpisodeCodec
            or codec.identity()["input_specification"] != predictor.config.inputs.identity()
        ):
            raise ValueError("El codec no comparte la entrada del padre")
        super().__init__(predictor, recipe, audit=audit)
        self.correction, self.codec = correction, codec
        if lesions is not None and type(lesions) is not LesionProtocol:
            raise ValueError("B4 necesita su protocolo de diagnósticos")
        self.lesion_plan = None if lesions is None else lesions.bind(correction)
        self.lesion_ledger = None if self.lesion_plan is None else LesionLedger(self.lesion_plan)
        self._lesion_resolved = {}
        # Registrador de las trazas de #448, o None. Solo lee A y el error escrito.
        self.trace_recorder = None
        self._trace_events, self._trace_phase = 0, "b6"
        self.memory = correction.empty()
        # Para cada decisión pendiente se guardan la predicción del núcleo, la clave del codec y
        # la ruta del régimen. Así A se escribe con el error del núcleo y no con el de la
        # emisión corregida, y en el compartimento del instante de la decisión.
        self._core = {}
        self._staged = []
        self._issued_experts, self._mature_experts = {}, {}
        self._routing = _RoutingLog()
        self.kalman = correction.memory.rule == "kalman"
        self._reset_delay()

    def _reset_delay(self):
        """Memoria de D3 y su cola: solo existen con la regla kalman.

        `_incoming` recibe las etiquetas que maduran en el evento, `_held` las guarda por
        mercado hasta la siguiente decisión de ese mercado y `_due` las que se escriben al
        final del evento actual. `extra` guarda por decisión la varianza y la predicción con
        retraso, que `KalmanPredictionRows` añade a su fila.
        """
        self.delayed = self.correction.empty() if self.kalman else None
        self._incoming, self._held, self._due, self.extra = [], {}, [], {}

    def _observe(self, run, source, event, *, differentiable, blocks=None):
        _require(not differentiable, "B6 no tiene parámetros que ajustar")
        super()._observe(run, source, event, differentiable=False, blocks=blocks)
        if event.at < source.phase.decision_start:
            return
        specification = self.predictor.config.inputs
        decisions, keys, prices = [], [], []
        for raw in event.inputs:
            cpu = validated_cpu_batch(raw, specification)
            decisions.extend(zip(cpu.flow_ids, cpu.prediction_at, strict=True))
            keys.append(self.codec.encode(cpu).key_inputs)
            prices.append(cpu.inputs["prices"])
        inputs = np.concatenate(keys)
        # El régimen se calcula una vez con todas las filas del evento, antes de leer A, así que
        # no depende de cómo se reparta el evento en bloques.
        routes, states = self.correction.routes(
            np.concatenate(prices) if self.correction.routing is not None else None,
            [flow for flow, _ in decisions],
            decisions[0][1],
        )
        self._routing.event(decisions[0][1], states, routes, self.correction.routing)
        # Todas las lecturas del evento usan la A confirmada antes de él, como FinancialSession.
        # Lo que se escriba en este evento solo puede afectar a los siguientes.
        keys = self.correction.keys(inputs, routes)
        parts = None
        if self.correction.multiscale is not None:
            corrections, parts = self.memory.predict(keys, at=event.at)
            corrections = corrections[:, 0].tolist()
        else:
            corrections = self.memory.read(keys)[:, 0].tolist()
        if self.lesion_plan is not None:
            diagnostics = self.lesion_plan.read(
                self.memory,
                keys,
                at=event.at,
                baseline=torch.tensor(corrections, dtype=torch.float64)[:, None],
            )
            core_values = torch.tensor(
                [run.pending[key] for key in decisions], dtype=torch.float64
            )[:, None]
            self.lesion_ledger.issue(decisions, diagnostics + core_values)
        if self.delayed is not None:
            extra = zip(
                self.memory.variance(keys).tolist(),
                self.delayed.read(keys)[:, 0].tolist(),
                strict=True,
            )
            # Las etiquetas retenidas de los mercados que deciden ahora se escriben al final de
            # este evento, una sesión después de la que las habría recibido sin retraso.
            for market in sorted({flow.split("/", 1)[0] for flow, _ in decisions}):
                self._due += self._held.pop(market, [])
        else:
            extra = [None] * len(decisions)
        rows = routes.tolist() if routes is not None else [None] * len(decisions)
        for index, (decision, key, route, correction, more) in enumerate(
            zip(decisions, inputs, rows, corrections, extra, strict=True)
        ):
            if decision in self._core:
                raise ValueError("La decisión ya tiene una predicción pendiente")
            core = run.pending[decision]
            if more is not None:
                variance, delayed = more
                self.extra[decision] = (variance, core + delayed)
            self._core[decision] = (core, key, route)
            if parts is not None:
                self._issued_experts[decision] = parts[index].detach().clone()
            run.pending[decision] = core + correction
            if decision in run.levels:
                run.levels[decision] = [level + correction for level in run.levels[decision]]
            if self.audit is not None:
                # La auditoría del núcleo ya registró la predicción sin corregir. Esta entrada
                # guarda la emisión corregida para poder comparar las dos.
                self.audit.append(("emitted", source.phase.partition, *decision, core + correction))

    def _labels(self, run, event, *, train):
        _require(not train, "B6 no tiene parámetros que ajustar")
        if self.lesion_ledger is not None:
            self._lesion_resolved = self.lesion_ledger.receive(
                [(flow, decision, event.at, value) for flow, decision, value in event.labels]
            )
        super()._labels(run, event, train=False)
        for flow, decision_at, value in event.labels:
            core, key, route = self._core.pop((flow, decision_at))
            self._staged.append((decision_at, flow, value - core, key, route))
            if self.correction.multiscale is not None:
                self._mature_experts[flow, decision_at] = self._issued_experts.pop(
                    (flow, decision_at)
                )
            if self.delayed is not None:
                self._incoming.append(self._staged[-1])

    def _write(self, at):
        """Escribir en A las etiquetas del evento cuando sus predicciones ya se han emitido.

        Las filas se ordenan por decisión y flujo antes de escribir. La regla delta escribe
        fila a fila, así que otro orden cambiaría A aunque las etiquetas fueran las mismas.
        """
        if self.delayed is not None:
            if self._due:
                due, self._due = self._due, []
                self.delayed, _ = self._written(self.delayed, due, at)
            for item in self._incoming:
                self._held.setdefault(item[1].split("/", 1)[0], []).append(item)
            self._incoming = []
        if not self._staged:
            return
        staged, self._staged = self._staged, []
        if self.correction.routing is not None:
            self._routing.writes(np.array([route for *_, route in staged], dtype=np.int64))
        self.memory, feedback = self._written(self.memory, staged, at)
        recorder = self.trace_recorder
        if recorder is not None:
            # Trazas de #448: B6 no tiene pasos de optimizador y su aprendizaje son las
            # escrituras de A. Se registran por evento de escritura con la cadencia declarada.
            self._trace_events += 1
            if recorder.due(self._trace_events):
                with recorder.at(self._trace_events, self._trace_phase):
                    recorder.scalar("associative.writes", self.memory.writes)
                    recorder.scalar(
                        "associative.matrix_l2", torch.linalg.vector_norm(self.memory.matrix)
                    )
                    recorder.tensor("associative.feedback", feedback.values)
                if recorder.pending >= MAX_PENDING:
                    recorder.flush()

    def _written(self, memory, staged, at):
        """Escribir en `memory` las filas `staged` en el orden canónico de decisión y flujo.

        Devuelve la memoria escrita y el resultado que ha recibido.
        """
        staged = sorted(staged, key=lambda item: item[:2])
        first = memory.writes + 1
        routes = None
        if self.correction.routing is not None:
            routes = np.array([route for *_, route in staged], dtype=np.int64)
        feedback = self.correction.feedback(
            ids=[first + offset for offset in range(len(staged))],
            decision_at=[decision_at for decision_at, *_ in staged],
            available_at=[at] * len(staged),
            keys=self.correction.keys(np.stack([key for _, _, _, key, _ in staged]), routes),
            values=[value for _, _, value, _, _ in staged],
        )
        if self.correction.multiscale is not None:
            decisions = [(flow, decision_at) for decision_at, flow, *_ in staged]
            issued = torch.stack([self._mature_experts[key] for key in decisions])
            updated = memory.write(feedback, cutoff=at, issued_predictions=issued)
            for key in decisions:
                del self._mature_experts[key]
            return updated, feedback
        return self._write_memory(memory, feedback, staged, at), feedback

    def _write_memory(self, memory, feedback, staged, at):
        """Escribir el resultado en A. E4 lo sustituye para regular la tasa por mercado."""
        return memory.write(feedback, cutoff=at)

    def _close(self, run):
        if self.lesion_ledger is not None:
            self.lesion_ledger.discard(list(self.lesion_ledger.pending), reason="finalized")
            self._lesion_resolved.clear()
        super()._close(run)
        self._core.clear()
        self._issued_experts.clear()
        self._mature_experts.clear()
        # Las decisiones cuya etiqueta no madura en el tramo no llegan a ser filas.
        self.extra.clear()

    def evaluate(self, source, *, stop=None, rows=None):
        """Recorrer un tramo medido empezando con A en cero y la memoria rápida inicial.

        El reinicio evita que un tramo dependa de los anteriores, y es lo que permite
        repetirlo desde el principio tras una interrupción.
        """
        if (
            type(source) is not FinancialObservationSource
            or source.phase.partition not in PREDICTED
            or not _compatible(self.predictor.config.inputs, source.specification())
        ):
            raise ValueError("El tramo no es un tramo medido de la entrada del padre")
        events = source.batched_events(block_rows=self.recipe.block_rows)
        return self._pass(source, events, stop=stop, rows=rows)

    def _pass(self, source, events, *, stop=None, rows=None):
        """Recorrer los eventos en su orden temporal.

        De `source` solo se usa la fase, así que las pruebas pueden entregar eventos
        construidos a mano con el mismo contrato.
        """
        run = _Pass(rows=rows)
        if self.lesion_plan is not None:
            if rows is not None and (
                type(rows) is not LesionPredictionRows or rows.inference is not self
            ):
                raise ValueError("Las filas B4 deben conservar los pronósticos de este diagnóstico")
            self.lesion_ledger = LesionLedger(self.lesion_plan)
            self._lesion_resolved = {}
        self.memory = self.correction.empty()
        self._core, self._staged = {}, []
        self._issued_experts, self._mature_experts = {}, {}
        self._routing = _RoutingLog()
        self._reset_delay()
        self._trace_events, self._trace_phase = 0, f"b6_{source.phase.partition}"
        self.predictor.eval()
        with torch.no_grad():
            for event in events:
                if stop is not None and stop.requested:
                    raise Paused
                if self.lesion_ledger is not None:
                    self.lesion_ledger.begin(event.at)
                self._labels(run, event, train=False)
                if event.inputs:
                    self._observe(run, source, event, differentiable=False)
                self._write(event.at)
                if event.close_phase:
                    self._close(run)
        if run.counters["labels"] == 0:
            raise ValueError("El tramo no contiene etiquetas maduras")
        exported = self.memory.export()
        metrics = dict(
            self._metrics(run),
            associative_writes=exported["writes"],
            associative_matrix_sha256=exported["matrix_sha256"],
        )
        # Sin enrutamiento las métricas conservan su forma anterior.
        if self.correction.routing is not None:
            metrics["routing"] = self._routing.summary(self.correction.routing)
        if self.delayed is not None:
            # Las etiquetas que siguen retenidas al cerrar el tramo no llegan a escribirse.
            metrics["label_delay_writes"] = self.delayed.writes
        if self.correction.multiscale is not None:
            metrics["multiscale"] = dict(
                weights=self.memory.weights.tolist(),
                clock=self.memory.clock,
                cohorts=self.memory.cohorts,
            )
        if self.lesion_ledger is not None:
            metrics["lesions"] = self.lesion_ledger.metrics()
        return metrics

    def predict(self, source, rows, *, stop=None):
        if not hasattr(rows, "append"):
            raise ValueError("La inferencia necesita un destino de filas")
        return self.evaluate(source, stop=stop, rows=rows)


# Factores opcionales que limitan (E3) o ponderan (E4) las escrituras de A. Cada uno tiene
# un módulo con checked_configuration, source_hashes, restore_configuration y su clase de
# inferencia, que recibe la configuración con el nombre del factor. No se combinan.
WRITE_FACTORS = {
    "salience": dict(
        label="E3",
        module="salience_inference",
        inference="SalienceCorrectionInference",
        policy="reset_scale_and_write_budget_at_each_phase_start",
    ),
    "regime_gate": dict(
        label="E4",
        module="regime_gate_inference",
        inference="RegimeGateCorrectionInference",
        policy="reset_scale_and_cusum_at_each_phase_start",
    ),
}


def _factor_module(name):
    return importlib.import_module(f"mars_titan.training.{WRITE_FACTORS[name]['module']}")


def _write_factor(**factors):
    """El factor de escritura declarado como (nombre, configuración), o None."""
    declared = [(name, config) for name, config in factors.items() if config is not None]
    _require(len(declared) <= 1, "Los factores de escritura E3 y E4 no se combinan")
    return declared[0] if declared else None


def _correction_inference(predictor, recipe, correction, codec, *, factor=None, **options):
    if factor is None:
        return CorrectionInference(predictor, recipe, correction, codec, **options)
    name, config = factor
    inference = getattr(_factor_module(name), WRITE_FACTORS[name]["inference"])
    return inference(predictor, recipe, correction, codec, **{name: config}, **options)


class KalmanPredictionRows(PredictionRows):
    """Filas de B6 kalman con la varianza predictiva y la predicción con una sesión más de retraso.

    Cada fila toma de la inferencia los dos valores que guardó al emitir su decisión.
    """

    def __init__(self, inference):
        super().__init__(inference.quantiles)
        self.inference, self.extra = inference, []

    def append(self, record):
        flow, at = record[:2]
        self.extra.append(self.inference.extra.pop((flow, at)))
        super().append(record)

    def _flush(self):
        if not self.buffer:
            return
        extra, self.extra = self.extra, []
        super()._flush()
        for name, values in zip(KALMAN_COLUMNS, zip(*extra, strict=True), strict=True):
            self.tables[-1] = self.tables[-1].append_column(
                name, pa.array(values, type=pa.float64())
            )


class LesionPredictionRows(PredictionRows):
    """Columnas de diagnósticos calculados al emitir, emparejadas con la fila original."""

    def __init__(self, inference):
        super().__init__(inference.quantiles)
        self.inference, self.extra = inference, []

    def append(self, record):
        self.extra.append(self.inference._lesion_resolved.pop(tuple(record[:2])).tolist())
        super().append(record)

    def _flush(self):
        if not self.buffer:
            return
        extra, self.extra = self.extra, []
        super()._flush()
        for name, values in zip(
            self.inference.lesion_plan.names, zip(*extra, strict=True), strict=True
        ):
            self.tables[-1] = self.tables[-1].append_column(
                "lesion_" + name, pa.array(values, type=pa.float64())
            )


def prediction_rows(inference):
    """Destino de las filas de un tramo, con las columnas de D3 y PT3 si la regla es kalman."""
    if inference.lesion_plan is not None:
        return LesionPredictionRows(inference)
    if inference.kalman:
        return KalmanPredictionRows(inference)
    return PredictionRows(inference.quantiles)


class _RoutingLog:
    """Ocupación, cambios y asignaciones del régimen en un recorrido.

    Cada evento emitido deja el estado de cada mercado con su corte, así que el informe
    registra cuándo y con qué datos se asignó cada ruta. Las lecturas cuentan predicciones
    emitidas por compartimento y las escrituras, etiquetas maduras por compartimento.
    """

    def __init__(self):
        self.assignments, self.reads, self.written = [], [0] * SLOTS, [0] * SLOTS
        self._last, self.transitions = {}, dict.fromkeys(("US", "CN"), 0)

    def event(self, at, states, routes, rule):
        if rule is None:
            return
        for state in states:
            previous = self._last.get(state.market)
            if previous is not None and previous != state.route:
                self.transitions[state.market] += 1
            self._last[state.market] = state.route
            self.assignments.append(state.record(at, rule.labels))
        for route in routes.tolist():
            self.reads[route] += 1

    def writes(self, routes):
        for route in routes.tolist():
            self.written[route] += 1

    def summary(self, rule):
        return dict(
            rule=rule.name,
            labels=list(rule.labels),
            reads_by_route=list(self.reads),
            writes_by_route=list(self.written),
            transitions_by_market={m: n for m, n in self.transitions.items() if m in self._last},
            assignments=self.assignments,
        )


def correction_memory_policy(warmup_months, *, kalman=False, kernel=False, fit_months=None):
    """Describir la política de memoria del padre ampliada con A.

    A también se reinicia en cada recorrido, y el informe lo deja escrito para que se vea
    que calibración y evaluación no heredan el estado de validación. Con la regla kalman A y
    P vuelven a su previa en cada recorrido y σ² y ϱ proceden del tramo de estimación.
    """
    policy = dict(
        memory_policy(warmup_months, fit_months),
        associative="zero_at_each_pass_then_mature_labels_after_the_event_predictions",
        associative_value="mature_label_minus_core_prediction",
        quantiles="all_levels_shifted_by_the_scalar_correction",
        fit="none_rate_and_forgetting_fixed_by_recipe_and_search_case",
    )
    if kalman:
        policy.update(
            associative="prior_mean_and_covariance_at_each_pass_then_mature_labels_after_the_"
            "event_predictions",
            fit="none_variances_estimated_on_the_train_tail_and_process_ratio_by_search_case",
        )
    if kernel:
        policy.update(
            associative="no_episodes_at_each_pass_then_mature_labels_after_the_event_predictions",
            fit="none_bandwidth_by_search_case_forgetting_and_capacity_fixed_by_recipe",
        )
    return policy


def estimation_phase(fold, months, warmup_months):
    """Tramo de estimación de σ² y ϱ: los últimos `months` meses del ajuste del padre.

    Empieza el primer día del mes situado `months` meses antes del final del ajuste y termina
    con él, que es también el corte de maduración de sus etiquetas, así que precede a
    validación, calibración y evaluación. Antes observa su calentamiento sin etiquetas, como
    los tramos medidos, sin empezar antes del origen de la ventana.
    """
    _require(
        type(months) is int and 1 <= months <= MAX_ESTIMATION_MONTHS,
        "Los meses de estimación no son válidos",
    )
    origin, end = (micros(day) for day in fold["train"])
    first = months_before(fold["train"][1], months)
    start = micros(first)
    _require(origin < start < end, "El tramo de estimación no cabe dentro del ajuste")
    warmup = max(origin, micros(months_before(first, warmup_months)))
    return FinancialPhase("train", warmup, start, end, end)


class _Residuals:
    """Destino de filas del tramo de estimación: residuo, mercado y sesión de cada etiqueta."""

    def __init__(self):
        self.values, self.markets, self.sessions = [], [], []

    def append(self, row):
        flow, decision_at, issued, value, _ = row
        self.values.append(value - issued)
        self.markets.append(flow.split("/", 1)[0])
        self.sessions.append(decision_at)


def window_cohort_noise(predictor, chronological, source, correlation_cap):
    """Estimar σ² y ϱ con los residuos etiqueta menos núcleo del tramo de estimación.

    El núcleo es el padre congelado con su receta cronológica y sin corrección, igual que en
    los tramos medidos, y cada residuo procede de una predicción emitida antes de conocer su
    etiqueta. La memoria rápida empieza en su estado inicial y observa el calentamiento.
    """
    _require(
        type(source) is FinancialObservationSource
        and source.phase.partition == "train"
        and _compatible(predictor.config.inputs, source.specification()),
        "El tramo de estimación no es un tramo de ajuste de la entrada del padre",
    )
    rows = _Residuals()
    metrics = ChronologicalInference(predictor, chronological).evaluate(source, rows=rows)
    estimate = estimate_cohort_noise(
        rows.values,
        rows.markets,
        np.asarray(rows.sessions, dtype=np.int64),
        correlation_cap=correlation_cap,
    )
    return dict(estimate, core_session_mae=metrics["session_mae"])


def window_noise(fold, values, warmup_months, dataset, root, core, stored=None):
    """σ² y ϱ de una ventana kalman, con su tramo de estimación y la huella de su índice.

    `core` es el padre congelado con su receta cronológica. Una ventana reanudada pasa en
    `stored` la estimación de su identidad, que se reutiliza si procede del mismo tramo y del
    mismo índice, porque repetirla en GPU podría no dar los mismos bits. Con otro tramo, otro
    índice o un registro incompleto se vuelve a estimar, la identidad deja de coincidir con la
    guardada y la reanudación se rechaza. La estimación no atiende la parada: recorre un tramo
    acotado y la parada pedida se atiende al empezar la validación, con la estimación ya en la
    identidad.
    """
    phase = estimation_phase(fold, values["estimation_months"], warmup_months)
    source = _sources(dataset, {"estimation": phase}, root)["estimation"]
    noise = dict(phase=asdict(phase), index=source.identity)
    if (
        isinstance(stored, dict)
        and set(stored) == {*noise, "estimate"}
        and all(stored[key] == value for key, value in noise.items())
    ):
        return stored
    estimate = window_cohort_noise(*core, source, values["correlation_cap"])
    return dict(noise, estimate=estimate)


def _parent_recipe(parent_report):
    """Recuperar la receta cronológica con la que predijo el padre.

    Bloques y truncamiento deben ser los mismos, porque con otros el núcleo de B6 dejaría de
    coincidir con el del modelo Titans-MAC.
    """
    identity, request = parent_report["identity"], parent_report["request"]
    return identity["recipe"], case_recipe(identity["recipe"], request.get("search_case"))


def _parent_specification(source, parent_report):
    """Contrato de entrada del padre sin construir el índice de entrenamiento.

    El padre se ajustó con la especificación de su tramo de entrenamiento, cuya única
    diferencia con la de un tramo medido es la huella del índice. Se toma de su informe.
    """
    measured = source.specification()
    trained = parent_report["identity"]["indices"].get("train")
    _require(isinstance(trained, str), "El padre no identifica su índice de entrenamiento")
    specification = FinancialInputSpec(
        source_sha256=measured.source_sha256,
        view_sha256=trained,
        representation=source.dataset.manifest["representation"],
        dimensions=dict(zip(MODALITIES, measured.widths, strict=True)),
        input_policy=measured.input_policy,
    )
    _require(_compatible(specification, measured), "El tramo no comparte la entrada del padre")
    return specification


def _variant(predictor, parent_report, components):
    request = parent_report["request"]
    base = core_identity(
        predictor,
        dict(
            recipe_sha256=request["recipe_sha256"],
            search_case=request.get("search_case"),
            window=request["window"],
        ),
    )
    variant = select_variant(load_declaration(), components, base=base)
    _require(
        variant.correction is not None and set(variant.components) == {"associative_memory"},
        "La ventana B6 solo admite la corrección asociativa, sin banco episódico",
    )
    return variant


def _prediction(output, rows, metrics, dataset, name):
    tables = checked_tables(rows, metrics, dataset, name)
    path = output / f"{name}-predictions.parquet"
    written = atomic_parquet_batches(path, tables)
    _require(written == rows.count, f"El Parquet de {name} no conserva sus filas")
    return dict(
        path=path.name,
        sha256=sha256(path),
        rows=written,
        bytes=path.stat().st_size,
        metrics=metrics,
    )


def run_correction_window(
    view,
    parent,
    recipe,
    *,
    components,
    seed,
    output,
    search_case,
    device="cuda:0",
    indices=None,
    stop=None,
    traces=None,
    lesions=None,
    salience=None,
    regime_gate=None,
):
    """Predecir los tramos medidos con el padre elegido y la corrección B6 del caso.

    La protección del aprendizaje se comprueba antes de leer ninguna fuente, porque la
    validación selecciona η, o κ con la regla kalman, entre los casos de la campaña. Con la
    regla kalman, σ² y ϱ se estiman antes de validar con `window_noise`. `traces` declara las
    trazas de #448: las escrituras de A de cada tramo, con la fase `b6_<tramo>`, en
    `output/traces`. Su cursor se guarda en `run.json` al confirmar cada tramo, porque B6 no
    tiene puntos de control y un tramo interrumpido se repite desde el principio.
    """
    from .mars_titan_walk_forward import _cuda, _frozen_parent, _parent

    require_learning_allowed("run_correction_window de MARS-TITAN")
    _require(type(seed) is int and 0 <= seed < 2**32, "La semilla no es válida")
    document = load_correction_recipe(recipe)
    values = case_values(document, search_case)
    kalman = kalman_arm(components)
    # Una combinación kalman se comprueba con σ² y ϱ de referencia antes de abrir fuentes. Las
    # estimadas solo cambian sus varianzas.
    combination = arm_components(components, values, UNIT_NOISE if kalman else None)
    # La combinación se valida contra la declaración antes de abrir ninguna fuente, para que
    # un modelo mal declarado falle sin crear la carpeta de salida.
    checked, correction = check_components(load_declaration(), combination)
    factor = _write_factor(salience=salience, regime_gate=regime_gate)
    if factor is not None:
        _factor_module(factor[0]).checked_configuration(factor[1], correction)
    if lesions is not None:
        _require(type(lesions) is LesionProtocol, "B4 necesita su protocolo identificado")
        lesions.bind(correction)
    _require(
        correction is not None and set(checked) == {"associative_memory"},
        "La ventana B6 solo admite la corrección asociativa, sin banco episódico",
    )
    _cuda(device)
    view, parent, output = Path(view), Path(parent), Path(output)
    parent_report, parent_sha = _parent(parent, view, seed)
    window = parent_report["request"]["window"]
    protocol, protocol_sha, _, fold = _protocol(view_protocol(view), window, seed)
    request = dict(
        view_sha256=sha256(view),
        protocol_sha256=protocol_sha,
        window=window,
        parent=dict(
            path=str(parent.resolve()),
            run_sha256=parent_sha,
            checkpoint_sha256=parent_report["checkpoint"]["sha256"],
        ),
        recipe_sha256=sha256(Path(recipe)),
        components=components,
        seed=seed,
        search_case=search_case,
        device=device,
        code=_code(),
    )
    if lesions is not None:
        request["lesions"] = lesions.identity()
        request["code"].update(_lesion_code())
    if factor is not None:
        request[factor[0]] = factor[1].identity()
        request["code"].update(_factor_module(factor[0]).source_hashes())
    safe_destination(output)
    report_path = output / "run.json"
    if report_path.exists():
        report, _ = read_manifest(report_path, 16 * 1024**2)
        _require(report.get("request") == request, "La petición de la ventana ha cambiado")
        if report["status"] == "completed":
            _verify(output, report)
            return report
    else:
        _require(
            not output.exists() or not any(output.iterdir()),
            "La salida contiene datos de otra ejecución",
        )
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    _check_view(dataset, protocol, fold)
    for protected in (*dataset.roots.values(), view.parent, parent):
        outside_source(protected, output)
    titans, chronological = _parent_recipe(parent_report)
    # Kalman conserva su rechazo hasta acotar también su tramo de estimación de ruido.
    options = walk_forward_options(titans, bounded_fit=not kalman)
    phases = window_phases(fold, options["warmup_months"], options.get("fit_months"))
    _require(
        {name: asdict(phase) for name, phase in phases.items()}
        == parent_report["identity"]["phases"],
        "Las fases de la ventana no coinciden con las del padre",
    )
    measured = {name: phases[name] for name in PREDICTED}
    root = Path(indices) if indices else index_root(dataset, output)
    sources = _sources(dataset, measured, root)
    specification = _parent_specification(sources["validation"], parent_report)
    predictor = _frozen_parent(titans, specification, seed, device, parent, parent_report)
    noise = None
    if kalman:
        stored = report["identity"].get("cohort_noise") if report_path.exists() else None
        noise = window_noise(
            fold,
            values,
            options["warmup_months"],
            dataset,
            root,
            (predictor, chronological),
            stored,
        )
        combination = arm_components(components, values, noise["estimate"])
    variant = _variant(predictor, parent_report, combination)
    codec = FrozenEpisodeCodec(specification)
    inference = _correction_inference(
        predictor, chronological, variant.correction, codec, lesions=lesions, factor=factor
    )
    identity = dict(
        schema_version=1,
        kind=KIND,
        request=request,
        window=fold,
        recipe=document,
        parent=dict(run_id=parent_report["run_id"], request=parent_report["request"]),
        chronological_recipe=chronological.identity(),
        variant=variant.identity(),
        variant_sha256=variant.fingerprint(),
        correction=variant.correction.identity(),
        codec=codec.identity(),
        memory_policy=correction_memory_policy(
            options["warmup_months"],
            kalman=kalman,
            kernel=variant.correction.memory.rule == KERNEL_RULE,
            fit_months=options.get("fit_months"),
        ),
        phases={name: asdict(phase) for name, phase in measured.items()},
        indices={name: source.identity for name, source in sources.items()},
        expected_rows={name: dataset.manifest["counts"][name] for name in PREDICTED},
        final_test_opened=False,
    )
    if noise is not None:
        # Solo kalman la lleva, así que las demás reglas conservan su identidad.
        identity["cohort_noise"] = noise
        identity["diagnostics"] = KALMAN_DIAGNOSTICS
    if factor is not None:
        name, config = factor
        identity[name] = config.identity()
        identity["memory_policy"][name] = WRITE_FACTORS[name]["policy"]
        identity["variant"] = {
            "kind": f"{name}_correction_variant_v1",
            "base": identity["variant"],
            name: config.identity(),
        }
        identity["variant_sha256"] = hashlib.sha256(
            canonical(identity["variant"]).encode()
        ).hexdigest()
    if inference.kernel_policy is not None:
        # Como en los lectores, la política de precisión de la receta del padre, que aplica
        # la inferencia al construirse, entra en la identidad. Sin precisión declarada la
        # identidad conserva su forma anterior.
        identity["kernel_policy"] = inference.kernel_policy
    run_id = hashlib.sha256(canonical(identity).encode()).hexdigest()
    if report_path.exists():
        _require(report["identity"] == identity, "La identidad de la ventana ha cambiado")
        _verify(output, report)
    else:
        output.mkdir(parents=True, exist_ok=True)
        selected = output / "selected.json"
        atomic_json(
            selected,
            dict(
                schema_version=1,
                kind=SELECTED_KIND,
                parent=request["parent"],
                correction=variant.correction.identity(),
                variant_sha256=identity["variant_sha256"],
                **({factor[0]: factor[1].identity()} if factor is not None else {}),
            ),
        )
        report = dict(
            schema_version=1,
            kind=KIND,
            run_id=run_id,
            request=request,
            identity=identity,
            status="running",
            started_at_utc=datetime.now(UTC).isoformat(),
            checkpoint=dict(path=selected.name, sha256=sha256(selected)),
            final_test_opened=False,
            predictions={},
            attempts=[],
        )
        atomic_json(report_path, report)
    stop = stop or StopRequest()
    started = time.perf_counter()
    recorder = open_recorder(
        output / "traces", traces, trace_identity("b6", identity), step_unit="write_event"
    )
    if recorder is not None:
        recorder.load_state_dict(report.get("traces") or EMPTY_CURSOR)
        inference.trace_recorder = recorder
    try:
        _require(
            sha256(output / "selected.json") == report["checkpoint"]["sha256"],
            "El estado elegido de la ventana ha cambiado",
        )
        for name in PREDICTED:
            if name in report["predictions"]:
                continue
            if stop.requested:
                raise Paused
            # Otro código del proceso podría haber cambiado la precisión entre tramos.
            require_policy(chronological.precision, identity.get("kernel_policy"))
            rows = prediction_rows(inference)
            metrics = inference.predict(sources[name], rows, stop=stop)
            report["predictions"][name] = _prediction(output, rows, metrics, dataset, name)
            if recorder is not None:
                if name == "validation":
                    # Solo la validación guarda su puntuación, como en los demás modelos.
                    with recorder.at(inference._trace_events, "validation"):
                        recorder.scalar("validation.score", metrics["session_mae"])
                recorder.flush()
                report["traces"] = recorder.state_dict()
            atomic_json(report_path, report)
        report.update(status="completed", finished_at_utc=datetime.now(UTC).isoformat())
        return report
    except Paused:
        report["status"] = "paused"
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["attempts"].append(dict(seconds=time.perf_counter() - started))
        atomic_json(report_path, report)


def carry_correction(
    anchor,
    anchor_view,
    view,
    output,
    *,
    device="cuda:0",
    stop=None,
    modality_ablation=None,
    regenerate=False,
    frozen_parent=False,
):
    """Predecir una ventana posterior con el padre y el caso elegidos en el ancla.

    Es la pieza de la variante B, así que no selecciona nada y usa el caso que eligió la
    validación del ancla. Cada tramo trasladado empieza con A en cero, la memoria rápida
    inicial y su propio calentamiento, como en el ancla. Con `regenerate` el ancla es el
    propio ajuste y se repiten su validación, su calibración y su evaluación en la misma
    vista. Como cada tramo empieza con A en cero, la repetición debe coincidir bit a bit con
    las tablas del ajuste antes de que la retención las libere.

    Con `modality_ablation` se vuelve a predecir la evaluación de la propia ventana con la
    variante aplicada en la lectura. El calentamiento, el núcleo y las claves del codec leen
    las mismas entradas ablacionadas, y A se escribe con los errores de ese recorrido.
    `frozen_parent` es el padre congelado de la cadena trivial de B6 en el walk-forward por
    etapas: el ancla es el estado elegido en k-1 y además se predice la validación de k, con
    la que la cadena puntúa.
    """
    from .carried_predictions import (
        ablation_record,
        carried_window,
        frozen_parent_record,
        predicted_partitions,
        regeneration_record,
        same_view,
    )
    from .mars_titan_walk_forward import _cuda, _frozen_parent

    require_learning_allowed("la predicción trasladada de B6")
    started = time.perf_counter()
    anchor, anchor_view, view, output = (Path(v) for v in (anchor, anchor_view, view, output))
    _cuda(device)
    report, report_sha = read_manifest(anchor / "run.json", 16 * 1024**2)
    _require(
        report.get("kind") == KIND
        and report.get("status") == "completed"
        and report.get("final_test_opened") is False
        and isinstance(report.get("checkpoint"), dict),
        "El ancla no es una ventana B6 completada",
    )
    _require(
        sha256(anchor_view) == report["request"]["view_sha256"],
        "La vista del ancla no es la de su ventana",
    )
    _verify(anchor, report)
    request, identity = report["request"], report["identity"]
    lesions = diagnostic_protocol(request)
    factors = {}
    for name, spec in WRITE_FACTORS.items():
        label = spec["label"]
        _require(
            (name in request) == (name in identity),
            f"El ancla no identifica {label} de forma coherente",
        )
        if name in request:
            factors[name] = _factor_module(name).restore_configuration(request)
            _require(
                canonical(identity[name]) == canonical(factors[name].identity()),
                f"La identidad {label} del ancla contradice a su petición",
            )
    factor = _write_factor(**factors)
    _require(
        sha256(anchor / "selected.json") == report["checkpoint"]["sha256"],
        "El estado elegido del ancla ha cambiado",
    )
    if factor is not None:
        name, config = factor
        label = WRITE_FACTORS[name]["label"]
        selected, _ = read_manifest(anchor / "selected.json", 64 * 1024)
        _require(
            canonical(selected.get(name)) == canonical(config.identity()),
            f"El estado seleccionado no conserva el factor {label} del ancla",
        )
    parent = Path(request["parent"]["path"])
    parent_report, parent_sha = read_manifest(parent / "run.json", 16 * 1024**2)
    _require(
        parent_sha == request["parent"]["run_sha256"]
        and parent_report["checkpoint"]["sha256"] == request["parent"]["checkpoint_sha256"]
        and parent_report["request"].get("local_control") is None,
        "El padre del ancla ha cambiado",
    )
    titans, chronological = _parent_recipe(parent_report)
    # El traslado aplica la política de la receta del padre, la misma que el ancla registró
    # en su identidad, y la compara antes de escribir nada.
    _require(
        declared_policy(chronological.precision) == identity.get("kernel_policy"),
        "La política de precisión no coincide con la del ancla",
    )
    options = walk_forward_options(titans)
    # La regla kalman usa la σ² y la ϱ estimadas en el ancla, sin volver a estimarlas.
    noise = identity.get("cohort_noise")
    combination = arm_components(
        request["components"],
        case_values(identity["recipe"], request["search_case"]),
        None if noise is None else noise["estimate"],
    )
    anchor_manifest, _ = read_manifest(anchor_view, 64 * 1024**2)
    partitions = predicted_partitions(modality_ablation, regenerate, frozen_parent)
    dataset = CorpusDataset(
        view, input_policy=HISTORICAL_MASKED, modality_ablation=modality_ablation
    )
    same_view(anchor_manifest, dataset.manifest, regenerate)
    anchor_fold, fold, age = carried_window(
        anchor_manifest,
        dataset.manifest,
        input_policy=HISTORICAL_MASKED,
        same_window=modality_ablation is not None or regenerate,
    )
    _check_view(dataset, view_protocol(view), fold)
    _new_destination(output, (*dataset.roots.values(), view.parent, anchor, parent))
    phases = window_phases(fold, options["warmup_months"])
    sources = _sources(
        dataset, {name: phases[name] for name in partitions}, index_root(dataset, output)
    )
    # La ablación solo predice la evaluación, así que la entrada se toma del primer tramo.
    specification = sources[partitions[0]].specification()
    seed = request["seed"]
    predictor = _frozen_parent(
        titans, specification, seed, device, parent, parent_report, carried=True
    )
    variant = _variant(predictor, parent_report, combination)
    _require(
        variant.correction.identity() == identity["correction"],
        "La corrección trasladada no es la elegida en el ancla",
    )
    codec = FrozenEpisodeCodec(specification)
    inference = _correction_inference(
        predictor, chronological, variant.correction, codec, lesions=lesions, factor=factor
    )
    predictions = {}
    for name in partitions:
        rows = prediction_rows(inference)
        metrics = inference.predict(sources[name], rows, stop=stop)
        predictions[name] = _prediction(output, rows, metrics, dataset, name)
    receipt = dict(
        schema_version=1,
        kind=CARRY_KIND,
        status="completed",
        anchor=dict(
            run_sha256=report_sha,
            checkpoint_sha256=report["checkpoint"]["sha256"],
            view_sha256=request["view_sha256"],
            fold=anchor_fold,
            components=request["components"],
            seed=seed,
            search_case=request["search_case"],
            variant_sha256=identity["variant_sha256"],
        ),
        view_sha256=sha256(view),
        fold=fold,
        months_since_anchor_information=age,
        memory_policy=dict(
            correction_memory_policy(
                options["warmup_months"],
                kalman=noise is not None,
                kernel=variant.correction.memory.rule == KERNEL_RULE,
            ),
            parameters="anchor_selected_parent_and_correction_without_further_selection",
            fast_state="never_transferred_from_the_anchor_reset_at_each_pass",
        ),
        phases={name: asdict(phases[name]) for name in partitions},
        indices={name: source.identity for name, source in sources.items()},
        device=device,
        code=_code(),
        predictions=predictions,
        final_test_opened=False,
        scientific_training_started=False,
        seconds=time.perf_counter() - started,
        finished_at_utc=datetime.now(UTC).isoformat(),
        **ablation_record(modality_ablation),
        **regeneration_record(regenerate),
        **frozen_parent_record(frozen_parent),
    )
    if lesions is not None:
        receipt["lesions"] = lesions.identity()
        receipt["code"].update(_lesion_code())
    if factor is not None:
        receipt[factor[0]] = factor[1].identity()
        receipt["code"].update(_factor_module(factor[0]).source_hashes())
    atomic_json(output / "carry.json", receipt)
    return receipt
