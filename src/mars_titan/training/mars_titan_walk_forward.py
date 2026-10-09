"""Una ventana walk-forward de MARS-TITAN sobre el Titans-MAC elegido en esa ventana.

Contrato por ventana para el orquestador de la campaña con máscaras:

- Entradas: la vista de la ventana, la carpeta de la ventana Titans-MAC `mac_online`
  completada en la misma vista y semilla (el padre), la receta del lector con sus casos de
  búsqueda, la combinación de componentes de `configs/titans/mars-titan-extensions.json`,
  la semilla y el caso elegido.
- Salida: un directorio con `run.json`, `selected.json` con el estado elegido compuesto
  (checkpoint del padre y mejor lector) y un Parquet por fila para validación, calibración
  y evaluación con las columnas y cuantiles de `training.titans_walk_forward`.
- Reanudación: la misma llamada continúa desde el último checkpoint coherente del lector o
  desde el primer tramo sin confirmar. Una ventana completada se devuelve tras comprobar
  sus huellas.

El padre no cambia: se carga su mejor estado exacto, se congela y su memoria rápida avanza
con la regla asociativa de Titans en cada observación. Las fases son las del padre, con el
mismo calentamiento de entradas. Cada recorrido empieza con el banco vacío y la memoria
rápida inicial, así que una ventana trasladada aplica el mismo contrato que su ancla. Las
variantes sin banco episódico son el propio Titans-MAC y no se ajustan aquí. La corrección
asociativa B6 solo se emite en `FinancialSession`.
"""

import hashlib
import importlib
import math
import os
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.mars_titan_variant import (
    check_components,
    core_identity,
    load_declaration,
    select_variant,
)
from mars_titan.models.titans.config import canonical, require_identity
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor

from .checkpoints import StopRequest, load_training_state
from .corpus_inputs import CorpusDataset
from .financial_run import Paused
from .learning_hold import require_learning_allowed
from .mars_titan_run import MarsTitanInference, ReadoutTrainer, case_recipe, load_recipe
from .mars_titan_run import retention_config as readout_retention
from .titans_walk_forward import (
    DTYPES,
    PREDICTED,
    PredictionRows,
    _carried_parameters,
    _check_view,
    _new_destination,
    _protocol,
    _require,
    _sources,
    _verify,
    checked_tables,
    memory_policy,
    unfused_attention,
    view_protocol,
    walk_forward_options,
    window_phases,
)
from .titans_walk_forward import (
    KIND as TITANS_KIND,
)

KIND = "mars_titan_walk_forward_window"
CARRY_KIND = "mars_titan_carried_predictions"
CARRIED = ("calibration", "evaluation")
WORLD = "mars_titan_walk_forward"
# Campos del caso que la campaña declara para cada brazo de MARS-TITAN.
CASE_FIELDS = {"recipe", "recipe_sha256", "components", "seed", "search_case", "parent_arm"}
_CODE = (
    "mars_titan.training.mars_titan_walk_forward",
    "mars_titan.training.mars_titan_run",
    "mars_titan.training.titans_walk_forward",
    "mars_titan.memory.mars_titan_variant",
    "mars_titan.models.titans.episodic_readout",
    "mars_titan.data.batches",
)


def _code():
    return {name: sha256(Path(importlib.import_module(name).__file__)) for name in _CODE}


def bank_memory_policy(warmup_months):
    """Política del padre ampliada con el banco, que también se reinicia en cada recorrido."""
    return dict(
        memory_policy(warmup_months),
        bank="empty_at_each_pass_then_mature_labels_after_the_event_predictions",
        readout_working_state="discarded_after_each_prediction",
    )


def _cuda(device):
    _require(device in ("cpu", "cuda:0"), "El dispositivo debe ser cpu o cuda:0 explícitos")
    if device == "cuda:0":
        from mars_titan.data.embeddings import require_cuda

        require_cuda()
        _require(
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"},
            "Configura CUBLAS_WORKSPACE_CONFIG antes de iniciar PyTorch",
        )


def _parent(folder, view, seed):
    """Ventana Titans-MAC `mac_online` completada sobre la misma vista y semilla."""
    report, digest = read_manifest(folder / "run.json", 16 * 1024**2)
    request = report.get("request", {})
    _require(
        report.get("kind") == TITANS_KIND
        and report.get("status") == "completed"
        and report.get("final_test_opened") is False
        and isinstance(report.get("checkpoint"), dict)
        and request.get("variant") == "mac_online"
        and request.get("seed") == seed
        and request.get("view_sha256") == sha256(view),
        "El padre no es una ventana Titans-MAC mac_online completada en esta vista y semilla",
    )
    _verify(folder, report)
    return report, digest


def _frozen_parent(document, specification, seed, device, folder, report, *, carried=False):
    """Mejor estado del padre, exacto en su vista o portable en una ventana trasladada."""
    options = {key: value for key, value in document["predictor"].items() if key != "dtype"}
    predictor = FinancialPredictor(
        FinancialConfig(specification, variant="mac_online", seed=seed, **options),
        device=device,
        dtype=DTYPES[document["predictor"]["dtype"]],
    )
    fit, _ = read_manifest(folder / "fit/run.json", 16 * 1024**2)
    state = load_training_state(
        folder / "fit/checkpoints",
        expected_identity=fit["identity"],
        selection="best",
        expected_sha256=report["checkpoint"]["sha256"],
    )
    if carried:
        _carried_parameters(state["model"], predictor)
    else:
        predictor.load_state_dict(state["model"])
    return predictor.eval().requires_grad_(False)


def _components(components):
    """Combinación con lector que ajustar, comprobada antes de abrir ninguna fuente."""
    components, correction = check_components(load_declaration(), components)
    _require(
        correction is None,
        "La corrección B6 se emite en FinancialSession. Este recorrido todavía no la incluye",
    )
    _require(
        "episodic_bank" in components,
        "Sin banco episódico la variante es el propio Titans-MAC y no tiene lector que ajustar",
    )
    return components


def _variant(predictor, report, components):
    """Combinación de componentes sobre el núcleo elegido en la ventana del padre."""
    request = report["request"]
    base = core_identity(
        predictor,
        dict(
            recipe_sha256=request["recipe_sha256"],
            search_case=request.get("search_case"),
            window=request["window"],
        ),
    )
    return select_variant(load_declaration(), _components(components), base=base)


def _readout(variant, codec, predictor, recipe, seed):
    config = variant.readout_config(
        codec.fingerprint(),
        hidden_size=predictor.config.hidden_size,
        neighbors=recipe.neighbors,
        temperature=recipe.temperature,
        max_working_bytes=recipe.max_working_bytes,
        seed=seed,
    )
    weight = predictor.head.weight
    return EpisodicReadout(config, dtype=weight.dtype, device=weight.device)


def _native(admission):
    if admission == "m0":
        return None
    from mars_titan.memory.native_backend import load_native

    return load_native()


def run_mars_titan_window(
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
    optimizer_factory=None,
):
    """Ajustar el lector de una combinación sobre el padre elegido y escribir sus filas.

    La protección del aprendizaje se comprueba antes de leer ninguna fuente.
    `optimizer_factory` solo existe para comprobar el bucle con un optimizador que no
    modifica pesos.
    """
    require_learning_allowed("run_mars_titan_window de MARS-TITAN")
    _require(type(seed) is int and 0 <= seed < 2**32, "La semilla no es válida")
    components = _components(components)
    _cuda(device)
    view, parent, output = Path(view), Path(parent), Path(output)
    parent_report, parent_sha = _parent(parent, view, seed)
    window = parent_report["request"]["window"]
    protocol, protocol_sha, rule, fold = _protocol(view_protocol(view), window, seed)
    document = load_recipe(recipe)
    readout_recipe = case_recipe(document, search_case)
    selection = {key: value for key, value in rule.items() if key != "max_epochs"}
    _require(
        readout_recipe.epochs == rule["max_epochs"] and readout_recipe.selection == selection,
        "La receta del lector no aplica la regla de selección y parada del protocolo",
    )
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
    parent_identity = parent_report["identity"]
    titans = parent_identity["recipe"]
    options = walk_forward_options(titans)
    phases = window_phases(fold, options["warmup_months"])
    _require(
        {name: asdict(phase) for name, phase in phases.items()} == parent_identity["phases"],
        "Las fases de la ventana no coinciden con las del padre",
    )
    sources = _sources(dataset, phases, Path(indices) if indices else output / "indices")
    specification = sources["train"].specification()
    predictor = _frozen_parent(titans, specification, seed, device, parent, parent_report)
    variant = _variant(predictor, parent_report, components)
    codec = FrozenEpisodeCodec(specification)
    readout = _readout(variant, codec, predictor, readout_recipe, seed)
    trainer = ReadoutTrainer(
        predictor,
        readout,
        readout_recipe,
        admission=variant.admission,
        retention=readout_retention(readout_recipe, variant.admission),
        native=_native(variant.admission),
        codec=codec,
        train=sources["train"],
        validation=sources["validation"],
        output=output / "fit",
        world=WORLD,
        fold=fold["id"],
        optimizer_factory=optimizer_factory,
    )
    identity = dict(
        schema_version=1,
        kind=KIND,
        request=request,
        window=fold,
        stopping_rule=rule,
        recipe=document,
        parent=dict(run_id=parent_report["run_id"], request=parent_report["request"]),
        variant=variant.identity(),
        variant_sha256=variant.fingerprint(),
        memory_policy=bank_memory_policy(options["warmup_months"]),
        phases={name: asdict(phase) for name, phase in phases.items()},
        indices={name: source.identity for name, source in sources.items()},
        fit_run_id=trainer.run_id,
        expected_rows={name: dataset.manifest["counts"][name] for name in PREDICTED},
        final_test_opened=False,
    )
    run_id = hashlib.sha256(canonical(identity).encode()).hexdigest()
    if report_path.exists():
        _require(report["identity"] == identity, "La identidad de la ventana ha cambiado")
        _verify(output, report)
    else:
        output.mkdir(parents=True, exist_ok=True)
        report = dict(
            schema_version=1,
            kind=KIND,
            run_id=run_id,
            request=request,
            identity=identity,
            status="running",
            started_at_utc=datetime.now(UTC).isoformat(),
            final_test_opened=False,
            predictions={},
            attempts=[],
        )
        atomic_json(report_path, report)
    stop = stop or StopRequest()
    started = time.perf_counter()
    try:
        fit = trainer.run(resume=(output / "fit").exists(), stop=stop)
        if fit["status"] != "completed":
            report["status"] = "paused"
            return report
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
        )
        # Estado elegido compuesto: el padre congelado y el mejor lector de la ventana.
        selected = output / "selected.json"
        atomic_json(
            selected,
            dict(
                schema_version=1,
                kind="mars_titan_selected_state",
                parent=request["parent"],
                readout=best,
                variant_sha256=variant.fingerprint(),
            ),
        )
        report["checkpoint"] = dict(path=selected.name, sha256=sha256(selected))
        for name in PREDICTED:
            if name in report["predictions"]:
                continue
            if stop.requested:
                raise Paused
            rows = PredictionRows(trainer.quantiles)
            metrics = trainer.evaluate(sources[name], stop=stop, rows=rows)
            tables = checked_tables(rows, metrics, dataset, name)
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
            path = output / f"{name}-predictions.parquet"
            written = atomic_parquet_batches(path, tables)
            _require(written == rows.count, f"El Parquet de {name} no conserva sus filas")
            report["predictions"][name] = dict(
                path=path.name,
                sha256=sha256(path),
                rows=written,
                bytes=path.stat().st_size,
                metrics=metrics,
            )
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


def _carried_readout(state, target):
    """Cargar el lector del ancla en el de la ventana trasladada.

    Solo puede cambiar la huella del codec, que depende de la vista. Configuración,
    precisión, K, modo, vecinos y semilla deben coincidir.
    """

    def portable(extra):
        configuration = dict(extra["configuration"])
        configuration.pop("codec_id")
        return dict(extra, configuration=configuration)

    expected = target.get_extra_state()
    require_identity(portable(state["_extra_state"]), portable(expected))
    target.load_state_dict({**state, "_extra_state": expected})


def carry_mars_titan(anchor, anchor_view, view, output, *, device="cuda:0", stop=None):
    """Predecir una ventana posterior con el padre y el lector elegidos en el ancla.

    Es la pieza de la variante B. No ajusta parámetros ni selección. Cada tramo trasladado
    empieza con la memoria rápida inicial, el banco vacío y su propio calentamiento.
    """
    require_learning_allowed("la predicción trasladada de MARS-TITAN")
    from .carried_predictions import carried_window

    started = time.perf_counter()
    anchor, anchor_view, view, output = (Path(v) for v in (anchor, anchor_view, view, output))
    _cuda(device)
    report, report_sha = read_manifest(anchor / "run.json", 16 * 1024**2)
    _require(
        report.get("kind") == KIND
        and report.get("status") == "completed"
        and report.get("final_test_opened") is False
        and isinstance(report.get("checkpoint"), dict),
        "El ancla no es una ventana de MARS-TITAN completada",
    )
    _require(
        sha256(anchor_view) == report["request"]["view_sha256"],
        "La vista del ancla no es la de su ajuste",
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
        and parent_report["checkpoint"]["sha256"] == request["parent"]["checkpoint_sha256"],
        "El padre del ancla ha cambiado",
    )
    titans = parent_report["identity"]["recipe"]
    options = walk_forward_options(titans)
    readout_recipe = case_recipe(identity["recipe"], request["search_case"])
    anchor_manifest, _ = read_manifest(anchor_view, 64 * 1024**2)
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    anchor_fold, fold, age = carried_window(
        anchor_manifest, dataset.manifest, input_policy=HISTORICAL_MASKED
    )
    _check_view(dataset, view_protocol(view), fold)
    _new_destination(output, (*dataset.roots.values(), view.parent, anchor, parent))
    phases = window_phases(fold, options["warmup_months"])
    sources = _sources(dataset, {name: phases[name] for name in CARRIED}, output / "indices")
    specification = sources["calibration"].specification()
    seed = request["seed"]
    predictor = _frozen_parent(
        titans, specification, seed, device, parent, parent_report, carried=True
    )
    variant = _variant(predictor, parent_report, request["components"])
    codec = FrozenEpisodeCodec(specification)
    readout = _readout(variant, codec, predictor, readout_recipe, seed)
    fit, _ = read_manifest(anchor / "fit/run.json", 16 * 1024**2)
    state = load_training_state(
        anchor / "fit/checkpoints",
        expected_identity=fit["identity"],
        selection="best",
        expected_sha256=report["fit"]["best_checkpoint"]["sha256"],
    )
    _carried_readout(state["model"], readout)
    readout.eval().requires_grad_(False)
    inference = MarsTitanInference(
        predictor,
        readout,
        readout_recipe,
        admission=variant.admission,
        retention=readout_retention(readout_recipe, variant.admission),
        native=_native(variant.admission),
        codec=codec,
        world=WORLD,
        fold=fold["id"],
    )
    predictions = {}
    for name in CARRIED:
        rows = PredictionRows(inference.quantiles)
        metrics = inference.evaluate(sources[name], stop=stop, rows=rows)
        tables = checked_tables(rows, metrics, dataset, name)
        path = output / f"{name}-predictions.parquet"
        written = atomic_parquet_batches(path, tables)
        _require(written == rows.count, f"El Parquet de {name} no conserva sus filas")
        predictions[name] = dict(
            path=path.name,
            sha256=sha256(path),
            rows=written,
            bytes=path.stat().st_size,
            metrics=metrics,
        )
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
            bank_memory_policy(options["warmup_months"]),
            parameters="anchor_selected_parent_and_readout_without_further_fitting",
            fast_state="never_transferred_from_the_anchor_reset_at_each_pass",
        ),
        phases={name: asdict(phases[name]) for name in CARRIED},
        indices={name: source.identity for name, source in sources.items()},
        device=device,
        code=_code(),
        predictions=predictions,
        final_test_opened=False,
        scientific_training_started=False,
        seconds=time.perf_counter() - started,
        finished_at_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(output / "carry.json", receipt)
    return receipt


def _campaign_case(run):
    case = run.case
    _require(
        isinstance(case, dict)
        and set(case) == CASE_FIELDS
        and case["seed"] == run.job["seed"]
        and run.policy == HISTORICAL_MASKED
        and isinstance(run.parent, dict),
        "El trabajo no declara un caso de MARS-TITAN con su padre en la campaña con máscaras",
    )
    _require(
        sha256(Path(case["recipe"])) == case["recipe_sha256"],
        "La receta del lector cambió después de planificar la campaña",
    )
    return case


def mars_titan_fit(run, *, device="cuda:0", optimizer_factory=None):
    """Ejecutor de ajuste para `training.masked_campaign`, con el padre que resuelve la campaña."""
    from .masked_campaign import Paused as CampaignPaused

    case = _campaign_case(run)
    # El recorrido cronológico exige fastpath=False solo mientras dura el trabajo.
    with unfused_attention():
        report = run_mars_titan_window(
            run.view,
            run.parent["folder"],
            case["recipe"],
            components=case["components"],
            seed=case["seed"],
            output=run.folder,
            search_case=case["search_case"],
            device=device,
            stop=run.stop,
            optimizer_factory=optimizer_factory,
        )
    if report["status"] == "paused":
        raise CampaignPaused
    _require(
        report["status"] == "completed"
        and report["request"]["view_sha256"] == run.view_sha256
        and report["request"]["parent"]["checkpoint_sha256"] == run.parent["checkpoint_sha256"],
        "La ventana de MARS-TITAN no confirma la vista ni el padre del trabajo",
    )
    return report


def mars_titan_carry(run, *, device="cuda:0"):
    """Ejecutor de predicción trasladada para `training.masked_campaign` (variante B)."""
    from .masked_campaign import Paused as CampaignPaused

    try:
        with unfused_attention():
            return carry_mars_titan(
                run.anchor["folder"],
                run.anchor["view"],
                run.view,
                run.folder,
                device=device,
                stop=run.stop,
            )
    except Paused as error:
        raise CampaignPaused from error
