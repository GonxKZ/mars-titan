"""Corrección asociativa B6 de MARS-TITAN en las ventanas walk-forward.

B6 suma a la predicción del Titans-MAC `mac_online` elegido en la ventana la lectura kᵀA de
una memoria asociativa lineal escrita solo con resultados maduros
(`memory.associative_memory.MatureCorrection`). No tiene parámetros que ajustar. Los casos de
búsqueda de su receta fijan η, la receta fija λ y la campaña elige el caso con el MAE por
sesión de validación, igual que en los demás brazos.

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

No hay tramo de ajuste, de modo que el índice de entrenamiento no se construye ni se lee.
Los errores que escriben A proceden siempre de predicciones emitidas en el propio tramo,
nunca del ajuste del padre. Un tramo interrumpido se repite desde su inicio, porque A y la
memoria rápida se reinician en cada recorrido, y los tramos confirmados conservan su huella.
"""

import hashlib
import importlib
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.memory.associative_memory import AssociativeMemory, MatureCorrection
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_observations import FinancialObservationSource
from mars_titan.memory.mars_titan_variant import (
    check_components,
    core_identity,
    load_declaration,
    select_variant,
)
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

KIND = "mars_titan_correction_window"
CARRY_KIND = "mars_titan_correction_carried_predictions"
SELECTED_KIND = "mars_titan_correction_selected_state"
RECIPE = "mars_titan_mature_correction_v1"
# La corrección solo depende de η y λ. Cada valor se declara una sola vez, en la receta o en
# los casos de búsqueda, para que elegir un caso nunca contradiga a la receta.
SEARCHED = ("rate", "forgetting")
# Un brazo solo declara la regla y la clave. η y λ se añaden con el caso elegido, porque la
# campaña los busca y no forman parte de la definición del brazo.
ARM_FIELDS = ("rule", "key")
# Reglas que admite la ventana. La regla kalman (PT3) no usa η ni λ sino sus varianzas, que
# esta receta no declara ni busca, así que todavía no tiene ventana.
RULES = ("delta", "proximal")
_RECIPE_FIELDS = {"schema_version", "recipe_name", "status", "recipe", "walk_forward", "pending"}
_CODE = (
    "mars_titan.training.mars_titan_correction",
    "mars_titan.training.financial_run",
    "mars_titan.training.titans_walk_forward",
    "mars_titan.training.mars_titan_walk_forward",
    "mars_titan.memory.mars_titan_variant",
    "mars_titan.memory.associative_memory",
    "mars_titan.memory.episodic_codec",
    "mars_titan.data.batches",
)


def _code():
    return {name: sha256(Path(importlib.import_module(name).__file__)) for name in _CODE}


def load_correction_recipe(path):
    """Leer la receta de B6 y comprobar cómo reparte η y λ entre la base y sus casos.

    Todos los casos sustituyen los mismos valores de `SEARCHED`, que por eso no pueden
    aparecer también en `recipe`, y entre ambos deben declararse los dos. La receta no tiene
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
    valid = (
        isinstance(document, dict)
        and set(document) == _RECIPE_FIELDS
        and document["schema_version"] == 1
        and document["recipe_name"] == RECIPE
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
        and next(iter(keys)) | set(base) == set(SEARCHED)
        and len({canonical(case) for case in cases.values()}) == len(cases),
        "La receta de la corrección declara η y λ entre su base y de uno a tres casos "
        "distintos con nombre válido que sustituyen los mismos valores",
    )
    return document


def case_values(document, search_case):
    """Devolver η y λ del caso elegido y rechazar un caso que la receta no declare."""
    cases = document["walk_forward"]["search_cases"]
    _require(
        isinstance(search_case, str) and search_case in cases,
        "Elige uno de los casos de búsqueda que declara la receta de la corrección",
    )
    return dict(document["recipe"], **cases[search_case])


def arm_components(components, values):
    """Completar la combinación de un brazo B6 con el η y el λ del caso.

    El brazo solo puede declarar la memoria asociativa. Con un banco añadido la ventana
    mezclaría dos vías de memoria y A5 dejaría de aislar el efecto de la corrección.
    """
    memory = components.get("associative_memory") if isinstance(components, dict) else None
    _require(
        isinstance(components, dict)
        and set(components) == {"associative_memory"}
        and isinstance(memory, dict)
        and set(memory) == set(ARM_FIELDS),
        "Un brazo B6 declara solo associative_memory con su regla y su clave",
    )
    _require(
        memory["rule"] in RULES,
        "La ventana B6 solo admite las reglas delta y proximal. La regla kalman no usa η ni λ "
        "y todavía no tiene ventana",
    )
    return dict(associative_memory=dict(memory, **values))


class CorrectionInference(ChronologicalInference):
    """Inferencia cronológica del padre congelado con la corrección B6.

    El cálculo del núcleo es exactamente el de `ChronologicalInference`. Tras emitir cada
    bloque se suma kᵀA con la A del evento anterior y, después de las predicciones del
    evento, se escriben las etiquetas que maduran en él. La memoria rápida del padre no
    recibe ninguna etiqueta, porque la sorpresa asociativa de Titans y el error financiero
    maduro son señales distintas.
    """

    def __init__(self, predictor, recipe, correction, codec, *, audit=False):
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
        self.memory = AssociativeMemory(correction.memory)
        # Para cada decisión pendiente se guardan la predicción del núcleo y la clave del codec.
        # Así A se escribe con el error del núcleo y no con el de la emisión corregida.
        self._core = {}
        self._staged = []

    def _observe(self, run, source, event, *, differentiable):
        _require(not differentiable, "B6 no tiene parámetros que ajustar")
        super()._observe(run, source, event, differentiable=False)
        if event.at < source.phase.decision_start:
            return
        specification = self.predictor.config.inputs
        decisions, keys = [], []
        for raw in event.inputs:
            cpu = validated_cpu_batch(raw, specification)
            decisions.extend(zip(cpu.flow_ids, cpu.prediction_at, strict=True))
            keys.append(self.codec.encode(cpu).key_inputs)
        inputs = np.concatenate(keys)
        # Todas las lecturas del evento usan la A confirmada antes de él, como FinancialSession.
        # Lo que se escriba en este evento solo puede afectar a los siguientes.
        corrections = self.memory.read(self.correction.keys(inputs))[:, 0].tolist()
        for decision, key, correction in zip(decisions, inputs, corrections, strict=True):
            if decision in self._core:
                raise ValueError("La decisión ya tiene una predicción pendiente")
            core = run.pending[decision]
            self._core[decision] = (core, key)
            run.pending[decision] = core + correction
            if decision in run.levels:
                run.levels[decision] = [level + correction for level in run.levels[decision]]
            if self.audit is not None:
                # La auditoría del núcleo ya registró la predicción sin corregir. Esta entrada
                # guarda la emisión corregida para poder comparar las dos.
                self.audit.append(("emitted", source.phase.partition, *decision, core + correction))

    def _labels(self, run, event, *, train):
        _require(not train, "B6 no tiene parámetros que ajustar")
        super()._labels(run, event, train=False)
        for flow, decision_at, value in event.labels:
            core, key = self._core.pop((flow, decision_at))
            self._staged.append((decision_at, flow, value - core, key))

    def _write(self, at):
        """Escribir en A las etiquetas del evento cuando sus predicciones ya se han emitido.

        Las filas se ordenan por decisión y flujo antes de escribir. La regla delta escribe
        fila a fila, así que otro orden cambiaría A aunque las etiquetas fueran las mismas.
        """
        if not self._staged:
            return
        staged, self._staged = sorted(self._staged, key=lambda item: item[:2]), []
        first = self.memory.writes + 1
        feedback = self.correction.feedback(
            ids=[first + offset for offset in range(len(staged))],
            decision_at=[decision_at for decision_at, *_ in staged],
            available_at=[at] * len(staged),
            keys=self.correction.keys(np.stack([key for *_, key in staged])),
            values=[value for _, _, value, _ in staged],
        )
        self.memory = self.memory.write(feedback, cutoff=at)

    def _close(self, run):
        super()._close(run)
        self._core.clear()

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
        self.memory = AssociativeMemory(self.correction.memory)
        self._core, self._staged = {}, []
        self.predictor.eval()
        with torch.no_grad():
            for event in events:
                if stop is not None and stop.requested:
                    raise Paused
                self._labels(run, event, train=False)
                if event.inputs:
                    self._observe(run, source, event, differentiable=False)
                self._write(event.at)
                if event.close_phase:
                    self._close(run)
        if run.counters["labels"] == 0:
            raise ValueError("El tramo no contiene etiquetas maduras")
        exported = self.memory.export()
        return dict(
            self._metrics(run),
            associative_writes=exported["writes"],
            associative_matrix_sha256=exported["matrix_sha256"],
        )

    def predict(self, source, rows, *, stop=None):
        if not hasattr(rows, "append"):
            raise ValueError("La inferencia necesita un destino de filas")
        return self.evaluate(source, stop=stop, rows=rows)


def correction_memory_policy(warmup_months):
    """Describir la política de memoria del padre ampliada con A.

    A también se reinicia en cada recorrido, y el informe lo deja escrito para que se vea
    que calibración y evaluación no heredan el estado de validación.
    """
    return dict(
        memory_policy(warmup_months),
        associative="zero_at_each_pass_then_mature_labels_after_the_event_predictions",
        associative_value="mature_label_minus_core_prediction",
        quantiles="all_levels_shifted_by_the_scalar_correction",
        fit="none_rate_and_forgetting_fixed_by_recipe_and_search_case",
    )


def _parent_recipe(parent_report):
    """Recuperar la receta cronológica con la que predijo el padre.

    Bloques y truncamiento deben ser los mismos, porque con otros el núcleo de B6 dejaría de
    coincidir con el del brazo Titans-MAC.
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
):
    """Predecir los tramos medidos con el padre elegido y la corrección B6 del caso.

    La protección del aprendizaje se comprueba antes de leer ninguna fuente, porque la
    validación selecciona η entre los casos de la campaña.
    """
    from .mars_titan_walk_forward import _cuda, _frozen_parent, _parent

    require_learning_allowed("run_correction_window de MARS-TITAN")
    _require(type(seed) is int and 0 <= seed < 2**32, "La semilla no es válida")
    document = load_correction_recipe(recipe)
    combination = arm_components(components, case_values(document, search_case))
    # La combinación se valida contra la declaración antes de abrir ninguna fuente, para que
    # un brazo mal declarado falle sin crear la carpeta de salida.
    checked, correction = check_components(load_declaration(), combination)
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
    options = walk_forward_options(titans)
    phases = window_phases(fold, options["warmup_months"])
    _require(
        {name: asdict(phase) for name, phase in phases.items()}
        == parent_report["identity"]["phases"],
        "Las fases de la ventana no coinciden con las del padre",
    )
    measured = {name: phases[name] for name in PREDICTED}
    sources = _sources(dataset, measured, Path(indices) if indices else output / "indices")
    specification = _parent_specification(sources["validation"], parent_report)
    predictor = _frozen_parent(titans, specification, seed, device, parent, parent_report)
    variant = _variant(predictor, parent_report, combination)
    codec = FrozenEpisodeCodec(specification)
    inference = CorrectionInference(predictor, chronological, variant.correction, codec)
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
        memory_policy=correction_memory_policy(options["warmup_months"]),
        phases={name: asdict(phase) for name, phase in measured.items()},
        indices={name: source.identity for name, source in sources.items()},
        expected_rows={name: dataset.manifest["counts"][name] for name in PREDICTED},
        final_test_opened=False,
    )
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
                variant_sha256=variant.fingerprint(),
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
            rows = PredictionRows(inference.quantiles)
            metrics = inference.predict(sources[name], rows, stop=stop)
            report["predictions"][name] = _prediction(output, rows, metrics, dataset, name)
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
    """Predecir una ventana posterior con el padre y el η y λ elegidos en el ancla.

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
    _require(
        sha256(anchor / "selected.json") == report["checkpoint"]["sha256"],
        "El estado elegido del ancla ha cambiado",
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
    combination = arm_components(
        request["components"], case_values(identity["recipe"], request["search_case"])
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
    sources = _sources(dataset, {name: phases[name] for name in partitions}, output / "indices")
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
    inference = CorrectionInference(predictor, chronological, variant.correction, codec)
    predictions = {}
    for name in partitions:
        rows = PredictionRows(inference.quantiles)
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
            correction_memory_policy(options["warmup_months"]),
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
    atomic_json(output / "carry.json", receipt)
    return receipt
