"""Una ventana del postentrenamiento de Titans-MAC y de los brazos con lector episódico.

Cada ventana parte del estado elegido de un brazo de la campaña con máscaras y una semilla,
ajusta un caso de la matriz de versión 3 (`chronological_matrix`) y selecciona con la
validación, con el padre elegible en la época cero. Escribe validación, calibración y
evaluación con el esquema común y los cinco cuantiles.

Sin `parent_view`, el padre se ajustó en la misma vista y el caso usa su tramo de ajuste.
Con `parent_view` (walk-forward por etapas), el padre es el estado elegido en la ventana
anterior: se carga con el estado portable de los traslados, validación, calibración y
evaluación son las de esta ventana y el ajuste solo decide con las filas que el padre no
usó (`staged_rows.posttraining_rows`), tras el calentamiento sin etiquetas declarado.

Calentamiento, truncamiento, bloques, acumulación y política de memoria son los del ajuste
del padre, leídos de su identidad. Solo cambian el optimizador, el presupuesto y la
selección, que declara la matriz. Así un cambio de la regla de un brazo base (por ejemplo su
calentamiento) llega igual a sus adaptadores.

`frozen_titans` y `frozen_readout` predicen validación, calibración y evaluación de una
ventana posterior con el estado elegido del padre, sin ajustar nada: es el padre congelado
del walk-forward por etapas. La protección del aprendizaje se comprueba antes de leer
ninguna fuente.
"""

import hashlib
import importlib
import math
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.models.predictive_adaptation import (
    attach_adapters,
    base_digest,
    trainable_parameters,
)
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training.carried_predictions import carried_window
from mars_titan.training.checkpoints import StopRequest, load_training_state
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import (
    ChronologicalInference,
    ChronologicalTrainer,
    Paused,
)
from mars_titan.training.learning_hold import require_learning_allowed
from mars_titan.training.mars_titan_run import (
    MarsTitanInference,
    _parameters_digest,
)
from mars_titan.training.mars_titan_run import (
    case_recipe as readout_case_recipe,
)
from mars_titan.training.mars_titan_walk_forward import (
    _anchor_scalers,
    _carried_readout,
    _cuda,
    _frozen_parent,
    _mars_family,
    _native,
    _readout,
)
from mars_titan.training.titans_walk_forward import (
    DTYPES,
    PREDICTED,
    PredictionRows,
    _carried_parameters,
    _check_view,
    _new_destination,
    _require,
    _sources,
    _verify,
    case_recipe,
    checked_tables,
    control_config,
    view_protocol,
    walk_forward_options,
    window_phases,
)
from mars_titan.training.titans_walk_forward import KIND as TITANS_KIND
from mars_titan.training.walk_forward_phases import micros, months_before

from . import chronological_matrix as cm
from .readout_adapters import ReadoutAdapterTrainer
from .staged_rows import posttraining_rows

TITANS_WINDOW = "titans_mac_posttraining_window"
READOUT_WINDOW = "readout_posttraining_window"
TITANS_FROZEN = "titans_mac_frozen_parent"
READOUT_FROZEN = "readout_frozen_parent"
STAGED = "staged_previous_window_v1"
_CODE = (
    "mars_titan.posttraining.chronological_windows",
    "mars_titan.posttraining.chronological_matrix",
    "mars_titan.posttraining.readout_adapters",
    "mars_titan.posttraining.staged_rows",
    "mars_titan.models.predictive_adaptation",
    "mars_titan.training.financial_run",
    "mars_titan.training.mars_titan_run",
    "mars_titan.training.titans_walk_forward",
    "mars_titan.training.mars_titan_walk_forward",
)


def _code():
    return {name: sha256(Path(importlib.import_module(name).__file__)) for name in _CODE}


def _completed(folder, kinds):
    """Informe completado de una ventana, con sus predicciones intactas."""
    report, digest = read_manifest(folder / "run.json", 16 * 1024**2)
    _require(
        report.get("kind") in kinds
        and report.get("status") == "completed"
        and report.get("final_test_opened") is False
        and isinstance(report.get("checkpoint"), dict),
        f"{folder} no es una ventana completada de {', '.join(kinds)}",
    )
    _verify(folder, report)
    return report, digest


def _phases(identity, names):
    return {name: FinancialPhase(**identity["phases"][name]) for name in names}


def _attach(model, targets, seed, *, seal=False):
    """Añadir los adaptadores, comprobar su recuento y devolver su descripción."""
    attach_adapters(model, targets, seed=seed)
    if seal:
        model._seal_parameters()
    description = cm.describe(targets, model)
    _require(
        trainable_parameters(model) == description["trainable_parameters"],
        "El modelo adaptado no tiene exactamente los parámetros entrenables declarados",
    )
    return dict(description, seed=seed)


def _predictor(document, specification, request, device, state, *, carried=False):
    """Variante elegida del padre con su control C, cargada desde su mejor estado."""
    options = {key: value for key, value in document["predictor"].items() if key != "dtype"}
    predictor = FinancialPredictor(
        FinancialConfig(specification, variant=request["variant"], seed=request["seed"], **options),
        local_control=control_config(request.get("local_control")),
        device=device,
        dtype=DTYPES[document["predictor"]["dtype"]],
    )
    if carried:
        _carried_parameters(state, predictor)
    else:
        predictor.load_state_dict(state)
    return predictor


def _best(folder, report):
    fit, _ = read_manifest(folder / "fit/run.json", 16 * 1024**2)
    checkpoint = report.get("fit", {}).get("best_checkpoint", report["checkpoint"])
    return fit, load_training_state(
        folder / "fit/checkpoints",
        expected_identity=fit["identity"],
        selection="best",
        expected_sha256=checkpoint["sha256"],
    )


def _report(output, request, identity, kind):
    """Crear o reanudar el informe de la ventana con su petición e identidad."""
    path = output / "run.json"
    if path.exists():
        report, _ = read_manifest(path, 16 * 1024**2)
        _require(
            report.get("request") == request and report.get("identity") == identity,
            "La petición o la identidad de la ventana ha cambiado",
        )
        _verify(output, report)
        return report
    report = dict(
        schema_version=1,
        kind=kind,
        run_id=hashlib.sha256(canonical(identity).encode()).hexdigest(),
        request=request,
        identity=identity,
        status="running",
        started_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
        predictions={},
        attempts=[],
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(path, report)
    return report


def _resumed(output, request):
    """Ventana ya completada con la misma petición, o None si hay que recorrerla."""
    safe_destination(output)
    path = output / "run.json"
    if path.exists():
        report, _ = read_manifest(path, 16 * 1024**2)
        _require(report.get("request") == request, "La petición de la ventana ha cambiado")
        if report["status"] == "completed":
            _verify(output, report)
            return report
        return None
    _require(
        not output.exists() or not any(output.iterdir()),
        "La salida contiene datos de otra ejecución",
    )
    return None


def _write(output, name, rows, metrics, dataset):
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


def _fit_and_predict(trainer, predict, sources, dataset, output, report, stop, check=None):
    """Ajustar o reanudar, dejar el mejor estado y escribir los tres tramos conciliados."""
    report_path = output / "run.json"
    stop = stop or StopRequest()
    started = time.perf_counter()
    try:
        fit = trainer.run(resume=(output / "fit").exists(), stop=stop)
        if fit["status"] != "completed":
            report["status"] = "paused"
            return report
        if check is not None:
            check()
        best = dict(fit["best_checkpoint"], path="fit/" + fit["best_checkpoint"]["path"])
        report["fit"] = dict(
            path="fit/run.json",
            sha256=sha256(output / "fit/run.json"),
            run_id=trainer.run_id,
            best_epoch=fit["best_epoch"],
            best_score=fit["best_score"],
            best_checkpoint=best,
            plateau_epoch=fit["plateau_epoch"],
            global_step=trainer.global_step,
            selection=fit["selection"],
        )
        report["checkpoint"] = best
        for name in PREDICTED:
            if name in report["predictions"]:
                continue
            if stop.requested:
                raise Paused
            rows = PredictionRows(trainer.quantiles)
            metrics = predict(sources[name], rows, stop)
            if name == "validation":
                score, recomputed = fit["best_score"], metrics["session_mae"]
                report["validation_selection_check"] = dict(
                    best_score=score,
                    recomputed=recomputed,
                    absolute_difference=abs(score - recomputed),
                )
                _require(
                    math.isclose(score, recomputed, rel_tol=1e-6, abs_tol=1e-12),
                    "La validación del mejor estado no reproduce la puntuación seleccionada",
                )
            report["predictions"][name] = _write(output, name, rows, metrics, dataset)
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


def _open_view(view, identity, output, protected, indices, names=PREDICTED + ("train",)):
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    _check_view(dataset, view_protocol(view), identity["window"])
    for source in (*dataset.roots.values(), view.parent, *protected):
        outside_source(source, output)
    phases = _phases(identity, [name for name in ("train", *PREDICTED) if name in names])
    sources = _sources(dataset, phases, Path(indices) if indices else output / "indices")
    return dataset, sources


def _case_request(case, digest, matrix, parent, parent_sha, view, device, parent_view=None):
    cm.validate_case(case)
    _require(
        case["adapter"] is None or case["adapter"]["matrix_sha256"] == digest,
        "El caso no pertenece a la matriz declarada",
    )
    _require(matrix["input_policy"] == HISTORICAL_MASKED, "La matriz no usa la edición desde 2000")
    request = dict(
        view_sha256=sha256(view),
        matrix_sha256=digest,
        case=case,
        parent=dict(path=str(parent.resolve()), run_sha256=parent_sha),
        device=device,
        code=_code(),
    )
    if parent_view is not None:
        request["parent_view_sha256"] = sha256(parent_view)
    return request


def staged_window(parent_view, view, warmup_months):
    """Ventana, fases y colocación de un padre ajustado en la vista de la ventana anterior.

    Validación, calibración y evaluación son las de la ventana, con su calentamiento. El
    ajuste decide solo con las filas nuevas y lee antes, sin etiquetas, el mismo
    calentamiento que los tramos medidos, sin salir del tramo de ajuste de la vista.
    """
    parent_manifest, _ = read_manifest(parent_view, 64 * 1024**2)
    manifest, _ = read_manifest(view, 64 * 1024**2)
    parent_fold, fold, _ = carried_window(parent_manifest, manifest, input_policy=HISTORICAL_MASKED)
    start, end = posttraining_rows(parent_fold, fold)
    phases = window_phases(fold, warmup_months)
    origin, since, until = (micros(day) for day in (fold["train"][0], start, end))
    warmup = max(origin, micros(months_before(start, warmup_months)))
    phases["train"] = FinancialPhase("train", warmup, since, until, until)
    placement = dict(
        design=STAGED,
        parent_window=parent_fold["id"],
        parent_view_sha256=sha256(parent_view),
        fit_start=start,
        fit_end=end,
    )
    return fold, {name: asdict(phase) for name, phase in phases.items()}, placement


def _placement(identity, parent_view, view, titans):
    """Ventana y fases del ajuste: las del padre o, por etapas, las de la ventana nueva."""
    if parent_view is None:
        return identity["window"], identity["phases"], None
    return staged_window(parent_view, view, walk_forward_options(titans)["warmup_months"])


def run_titans_posttraining(
    parent,
    view,
    output,
    *,
    case,
    matrix,
    digest,
    device="cuda:0",
    indices=None,
    stop=None,
    optimizer_factory=None,
    parent_view=None,
):
    """Postentrenar el estado elegido de una variante de Titans-MAC.

    Con adaptadores solo cambian sus correcciones. La memoria neuronal, la persistente y
    los demás parámetros quedan congelados, y los pesos rápidos siguen su regla asociativa.
    La continuación completa ajusta los mismos papeles que el ajuste base. Con
    `parent_view`, el padre se ajustó en esa vista anterior (walk-forward por etapas).
    """
    require_learning_allowed("el postentrenamiento de Titans-MAC")
    _cuda(device)
    parent, view, output = Path(parent), Path(view), Path(output)
    origin = view if parent_view is None else Path(parent_view)
    parent_report, parent_sha = _completed(parent, (TITANS_KIND,))
    request = _case_request(case, digest, matrix, parent, parent_sha, view, device, parent_view)
    original = parent_report["request"]
    _require(
        original["view_sha256"] == sha256(origin) and original["seed"] == case["seed"],
        "El padre no se ajustó en su vista con la semilla del caso",
    )
    resumed = _resumed(output, request)
    if resumed is not None:
        return resumed
    identity = parent_report["identity"]
    document = identity["recipe"]
    recipe = replace(case_recipe(document, original.get("search_case")), **cm.recipe_options(case))
    window, phases, placement = _placement(identity, parent_view, view, document)
    dataset, sources = _open_view(
        view, dict(window=window, phases=phases), output, (parent,), indices
    )
    _, state = _best(parent, parent_report)
    predictor = _predictor(
        document,
        sources["train"].specification(),
        original,
        device,
        state["model"],
        carried=placement is not None,
    )
    adapter = case["adapter"]
    description = None
    if adapter is not None:
        allowed = matrix["architectures"]["chronological"][cm.TITANS]["variants"]
        _require(
            adapter["design"] == cm.TITANS
            and set(adapter["points"]) <= set(allowed[original["variant"]]),
            "El brazo no pertenece a los puntos de esta variante de Titans-MAC",
        )
        targets = cm.titans_targets(matrix, adapter["points"])
        description = _attach(predictor, targets, cm.component_seed(case, "core"), seal=True)
    base = base_digest(predictor)
    posttraining = dict(
        kind=TITANS_WINDOW,
        case=case,
        parent=dict(
            run_sha256=parent_sha,
            checkpoint_sha256=parent_report["checkpoint"]["sha256"],
            variant=original["variant"],
            seed=original["seed"],
            search_case=original.get("search_case"),
        ),
        adapter=description,
        base_parameters_sha256=base if adapter is not None else None,
    )
    if placement is not None:
        posttraining["placement"] = placement
    trainer = ChronologicalTrainer(
        predictor,
        recipe,
        train=sources["train"],
        validation=sources["validation"],
        output=output / "fit",
        optimizer_factory=optimizer_factory,
        posttraining=posttraining,
    )
    window = dict(
        schema_version=1,
        kind=TITANS_WINDOW,
        request=request,
        window=window,
        recipe=document,
        posttraining=posttraining,
        memory_policy=identity["memory_policy"],
        phases=phases,
        indices={name: source.identity for name, source in sources.items()},
        fit_run_id=trainer.run_id,
        expected_rows={name: dataset.manifest["counts"][name] for name in PREDICTED},
        final_test_opened=False,
    )
    report = _report(output, request, window, TITANS_WINDOW)

    def check():
        if adapter is not None:
            _require(
                base_digest(predictor) == base,
                "El ajuste cambió parámetros fuera de los adaptadores",
            )

    return _fit_and_predict(
        trainer,
        lambda source, rows, stop: trainer.predict_partition(source, rows, stop=stop),
        sources,
        dataset,
        output,
        report,
        stop,
        check,
    )


def readout_family_of(report):
    """Familia del lector de un brazo de MARS-TITAN o CM-v1 desde su informe."""
    from mars_titan.training.cm_v1_factorial import ARMS, load_declaration, readout_family

    request = report.get("request") if isinstance(report, dict) else None
    _require(isinstance(request, dict), "El informe no declara su petición")
    if "components" in request:
        return _mars_family(request["components"])
    _require(
        isinstance(request.get("declaration"), str) and request.get("arm") in ARMS,
        "El informe no es una ventana de MARS-TITAN ni de un brazo de CM-v1",
    )
    document = load_declaration(request["declaration"])
    _require(
        request.get("declaration_sha256") == document["sha256"],
        "La declaración de CM-v1 cambió después de ajustar el brazo",
    )
    return readout_family(document, request["arm"])


def _arm_parts(arm, view):
    """Informe del brazo, su familia, su padre Titans-MAC y el lector elegido."""
    report, digest = _completed(
        arm, ("mars_titan_walk_forward_window", "cm_v1_walk_forward_window")
    )
    family = readout_family_of(report)
    _require(report["kind"] == family.kind, "El brazo no corresponde a su familia de lector")
    request = report["request"]
    _require(
        request["view_sha256"] == sha256(view)
        and sha256(arm / "selected.json") == report["checkpoint"]["sha256"],
        "El brazo no se ajustó en esta vista o su estado elegido cambió",
    )
    parent = Path(request["parent"]["path"])
    parent_report, parent_sha = read_manifest(parent / "run.json", 16 * 1024**2)
    _require(
        parent_sha == request["parent"]["run_sha256"]
        and parent_report["checkpoint"]["sha256"] == request["parent"]["checkpoint_sha256"]
        and parent_report["request"].get("local_control") == family.control,
        "El padre del brazo ha cambiado",
    )
    return report, digest, family, parent, parent_report


def _readout_models(family, report, parent, parent_report, specification, device, *, carried):
    """Padre congelado, variante, codec y lector elegido del brazo."""
    request = report["request"]
    titans = parent_report["identity"]["recipe"]
    seed = request["seed"]
    frozen = _frozen_parent(
        titans, specification, seed, device, parent, parent_report, carried=carried
    )
    variant = family.variant(frozen, parent_report)
    if not carried:
        _require(
            variant.fingerprint() == report["identity"]["variant_sha256"],
            "La variante reconstruida no es la del brazo",
        )
    base_recipe = readout_case_recipe(report["identity"]["recipe"], request["search_case"])
    codec = FrozenEpisodeCodec(specification)
    readout = _readout(variant, codec, frozen, base_recipe, seed)
    return frozen, variant, codec, readout, base_recipe


def run_readout_posttraining(
    arm,
    view,
    output,
    *,
    case,
    matrix,
    digest,
    device="cuda:0",
    indices=None,
    stop=None,
    optimizer_factory=None,
    parent_view=None,
):
    """Postentrenar el núcleo, la lectura episódica o ambos de un brazo con lector.

    El núcleo adaptado parte del mismo padre con su control C declarado. Sin adaptadores
    en el núcleo, el padre se carga congelado como en el ajuste del brazo. Con
    `parent_view`, el brazo se ajustó en esa vista anterior (walk-forward por etapas) y,
    con M3, conserva las escalas congeladas de su propio ajuste.
    """
    require_learning_allowed("el postentrenamiento de un brazo con lector episódico")
    _cuda(device)
    arm, view, output = Path(arm), Path(view), Path(output)
    origin = view if parent_view is None else Path(parent_view)
    report, digest_arm, family, parent, parent_report = _arm_parts(arm, origin)
    request = _case_request(case, digest, matrix, arm, digest_arm, view, device, parent_view)
    _require(report["request"]["seed"] == case["seed"], "El brazo no tiene la semilla del caso")
    resumed = _resumed(output, request)
    if resumed is not None:
        return resumed
    identity = report["identity"]
    window, phases, placement = _placement(
        identity, parent_view, view, parent_report["identity"]["recipe"]
    )
    staged = placement is not None
    dataset, sources = _open_view(
        view, dict(window=window, phases=phases), output, (arm, parent), indices
    )
    specification = sources["train"].specification()
    frozen, variant, codec, readout, base_recipe = _readout_models(
        family, report, parent, parent_report, specification, device, carried=staged
    )
    fit, state = _best(arm, report)
    if staged:
        _carried_readout(state["model"], readout)
    else:
        readout.load_state_dict(state["model"])
    _require(
        _parameters_digest(readout) == state["readout_sha256"],
        "El lector elegido del brazo no conserva su huella",
    )
    adapter, description = case["adapter"], {}
    predictor, core_rows = frozen, None
    if adapter is not None:
        _require(adapter["design"] == cm.READOUT, "El caso no adapta un brazo con lector")
        _require(
            adapter["episodic_readout"] is None or variant.admission != "m0",
            "Sin banco la lectura episódica no recibe gradiente",
        )
        readout.requires_grad_(False)
        if adapter["core"] is not None:
            titans = parent_report["identity"]["recipe"]
            _, parent_state = _best(parent, parent_report)
            predictor = _predictor(
                titans,
                specification,
                parent_report["request"],
                device,
                parent_state["model"],
                carried=staged,
            ).eval()
            targets = cm.titans_targets(matrix, adapter["core"])
            description["core"] = _attach(
                predictor, targets, cm.component_seed(case, "core"), seal=True
            )
            core_rows = case_recipe(
                titans, parent_report["request"].get("search_case")
            ).accumulation_rows
        if adapter["episodic_readout"] is not None:
            targets = cm.readout_targets(matrix, adapter["episodic_readout"])
            description["episodic_readout"] = _attach(
                readout, targets, cm.component_seed(case, "episodic_readout")
            )
    recipe = replace(base_recipe, **cm.recipe_options(case))
    extra = {"scalers": _anchor_scalers(fit)} if variant.admission == "m3" else {}
    posttraining = dict(
        kind=READOUT_WINDOW,
        case=case,
        control=case["control"],
        arm=dict(
            run_sha256=digest_arm,
            checkpoint_sha256=report["checkpoint"]["sha256"],
            variant_sha256=identity["variant_sha256"],
            readout_sha256=state["readout_sha256"],
        ),
        adapter=description or None,
    )
    if staged:
        scalers = extra.get("scalers")
        posttraining["placement"] = dict(
            placement, scalers=None if scalers is None else asdict(scalers)
        )
    trainer = ReadoutAdapterTrainer(
        predictor,
        readout,
        recipe,
        posttraining=posttraining,
        core_rows=core_rows,
        admission=variant.admission,
        retention=family.retention(base_recipe, variant, **extra),
        native=_native(variant.admission),
        codec=codec,
        train=sources["train"],
        validation=sources["validation"],
        output=output / "fit",
        world=family.world,
        fold=window["id"],
        optimizer_factory=optimizer_factory,
    )
    record = dict(
        schema_version=1,
        kind=READOUT_WINDOW,
        request=request,
        window=window,
        recipe=identity["recipe"],
        family=family.request,
        posttraining=posttraining,
        memory_policy=identity["memory_policy"],
        phases=phases,
        indices={name: source.identity for name, source in sources.items()},
        fit_run_id=trainer.run_id,
        expected_rows={name: dataset.manifest["counts"][name] for name in PREDICTED},
        final_test_opened=False,
    )
    report_out = _report(output, request, record, READOUT_WINDOW)
    return _fit_and_predict(
        trainer,
        lambda source, rows, stop: trainer.evaluate(source, stop=stop, rows=rows),
        sources,
        dataset,
        output,
        report_out,
        stop,
    )


def _later_view(parent_view, view, output, protected, titans):
    """Vista de una ventana posterior a la del padre, con las fases de sus tramos medidos."""
    parent_manifest, _ = read_manifest(parent_view, 64 * 1024**2)
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    parent_fold, fold, age = carried_window(
        parent_manifest, dataset.manifest, input_policy=HISTORICAL_MASKED
    )
    _check_view(dataset, view_protocol(view), fold)
    _new_destination(output, (*dataset.roots.values(), view.parent, *protected))
    phases = window_phases(fold, walk_forward_options(titans)["warmup_months"])
    sources = _sources(dataset, {name: phases[name] for name in PREDICTED}, output / "indices")
    extra = dict(
        view_sha256=sha256(view),
        fold=fold,
        parent_fold=parent_fold,
        months_since_parent_information=age,
        phases={name: asdict(phases[name]) for name in PREDICTED},
    )
    return dataset, sources, fold, extra


def _frozen_receipt(output, kind, parent, report_sha, report, dataset, sources, inference, extra):
    """Predecir los tres tramos medidos sin ajustar y confirmar el recibo del padre congelado."""
    predictions = {}
    for name in PREDICTED:
        rows = PredictionRows(inference.quantiles)
        metrics = inference.evaluate(sources[name], rows=rows)
        predictions[name] = _write(output, name, rows, metrics, dataset)
    receipt = dict(
        schema_version=1,
        kind=kind,
        status="completed",
        parent=dict(
            path=str(parent.resolve()),
            run_sha256=report_sha,
            checkpoint_sha256=report["checkpoint"]["sha256"],
            view_sha256=report["request"]["view_sha256"],
        ),
        predictions=predictions,
        indices={name: source.identity for name, source in sources.items()},
        code=_code(),
        final_test_opened=False,
        scientific_training_started=False,
        finished_at_utc=datetime.now(UTC).isoformat(),
        **extra,
    )
    atomic_json(output / "frozen.json", receipt)
    return receipt


class _Evaluate:
    """Interfaz común de inferencia: `evaluate(source, rows=...)` y `quantiles`."""

    def __init__(self, inference):
        self.inference, self.quantiles = inference, inference.quantiles

    def evaluate(self, source, *, rows):
        return self.inference.predict(source, rows)


def frozen_titans(parent, parent_view, view, output, *, device="cuda:0", stop=None):
    """Padre congelado: el estado elegido de Titans-MAC en su ventana, aplicado a otra.

    No ajusta parámetros ni selección. Cada tramo empieza con la memoria rápida inicial y
    su propio calentamiento de entradas, como un traslado de la campaña base.
    """
    require_learning_allowed("la predicción del padre congelado de Titans-MAC")
    _cuda(device)
    parent, parent_view, view, output = (Path(v) for v in (parent, parent_view, view, output))
    report, report_sha = _completed(parent, (TITANS_KIND,))
    _require(
        sha256(parent_view) == report["request"]["view_sha256"],
        "La vista del padre no es la de su ajuste",
    )
    document, request = report["identity"]["recipe"], report["request"]
    dataset, sources, _, extra = _later_view(parent_view, view, output, (parent,), document)
    _, state = _best(parent, report)
    predictor = _predictor(
        document,
        sources["validation"].specification(),
        request,
        device,
        state["model"],
        carried=True,
    ).eval()
    inference = ChronologicalInference(predictor, case_recipe(document, request.get("search_case")))
    return _frozen_receipt(
        output,
        TITANS_FROZEN,
        parent,
        report_sha,
        report,
        dataset,
        sources,
        _Evaluate(inference),
        dict(extra, device=device),
    )


def frozen_readout(arm, arm_view, view, output, *, device="cuda:0", stop=None):
    """Padre congelado de un brazo con lector: núcleo y lector elegidos, aplicados a otra
    ventana. El núcleo predice con C en modo disabled, como en el ajuste del brazo."""
    require_learning_allowed("la predicción del padre congelado de un brazo con lector")
    _cuda(device)
    arm, arm_view, view, output = (Path(v) for v in (arm, arm_view, view, output))
    report, digest, family, parent, parent_report = _arm_parts(arm, arm_view)
    dataset, sources, fold, extra = _later_view(
        arm_view, view, output, (arm, parent), parent_report["identity"]["recipe"]
    )
    frozen, variant, codec, readout, base_recipe = _readout_models(
        family,
        report,
        parent,
        parent_report,
        sources["validation"].specification(),
        device,
        carried=True,
    )
    fit, state = _best(arm, report)
    _carried_readout(state["model"], readout)
    readout.eval().requires_grad_(False)
    scalers = {"scalers": _anchor_scalers(fit)} if variant.admission == "m3" else {}
    inference = MarsTitanInference(
        frozen,
        readout,
        base_recipe,
        admission=variant.admission,
        retention=family.retention(base_recipe, variant, **scalers),
        native=_native(variant.admission),
        codec=codec,
        world=family.world,
        fold=fold["id"],
    )
    return _frozen_receipt(
        output,
        READOUT_FROZEN,
        arm,
        digest,
        report,
        dataset,
        sources,
        inference,
        dict(extra, device=device, family=family.request),
    )
