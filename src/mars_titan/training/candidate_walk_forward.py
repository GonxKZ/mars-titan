"""Ventanas walk-forward de la GRU candidata con banco episódico en la campaña con máscaras.

Un ajuste de ventana prepara desde la vista v2 el índice de observaciones de cada tramo
y el adaptador de su semilla, comprueba el bloqueo, ajusta con la receta declarada,
predice validación, calibración y evaluación con el contrato común de cuantiles,
comprueba que las filas son exactamente las de la vista y escribe un recibo walk-forward
por mercado. En la variante B, las ventanas intermedias se predicen con el estado elegido
en su ancla, sin ajustar pesos, selección ni normalizadores.

Al cruzar del ancla a una ventana trasladada, el banco y la cola de etiquetas siguen la
política `CARRY_POLICY`. Se conservan los parámetros elegidos en el ancla, las proyecciones fijas
del codec y la receta (admisión, K, capacidad, semilla y bloques del banco). Se reinician
el banco, la admisión pendiente, la cola de predicciones que esperan etiqueta, los errores
por sesión y el estado de la GRU. No hay calentamiento con etiquetas del ancla ni de tramos
anteriores: cada tramo empieza con el banco vacío y solo admite etiquetas que maduran
dentro del tramo después de emitir su predicción. Es la regla que ya siguen la validación,
la calibración y la evaluación de una ventana ajustada, así que una ventana trasladada
solo se diferencia de ella en los parámetros.

Las fases del calentamiento salen de `walk_forward_phases.window_phases` con los
`warmup_months` de la receta, como en Titans-MAC, MARS-TITAN y CM-v1: validación,
calibración y evaluación observan antes las entradas de esos meses, sin etiquetas ni
predicciones emitidas. La GRU no conserva estado entre instantes y el banco solo admite
etiquetas maduras de predicciones emitidas, así que el calentamiento no cambia el estado
de la candidata. Iguala la ventana de información observada por todos los brazos con
memoria y queda registrado en la identidad de cada fase.
"""

import time
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments.walk_forward_receipt import (
    RECEIPT_KIND,
    prediction_fingerprint,
    read_window_receipt,
)
from mars_titan.evaluation.splits import PARTITIONS
from mars_titan.memory.financial_observations import (
    FinancialObservationSource,
    prepare_observation_index,
)
from mars_titan.models.candidate.input_adapter import CandidateInputAdapter

from .candidate_run import (
    HELDOUT,
    CandidateChronologicalPredictor,
    CandidateChronologicalTrainer,
    CandidateRecipe,
    _Pause,
    load_recipe,
    restore_selected,
)
from .carried_predictions import (
    _destination,
    _receipt,
    ablation_record,
    carried_window,
    predicted_partitions,
    regeneration_record,
    same_view,
)
from .corpus_inputs import CorpusDataset
from .label_maturity import window_labels_until
from .learning_hold import require_learning_allowed
from .selection import AWAIT, campaign_rule, with_rule
from .temporal_contract import temporal_contracts
from .walk_forward_phases import checked_warmup, window_phases

MODEL = "candidate_gru_episodic"
WINDOW_KIND = "candidate_walk_forward_window"
WINDOW_REPORT = "window.json"
# Ámbito del banco. La ventana va en el fold y el tramo en la partición del banco.
WORLD = "gru_episodic_walk_forward"
# Tramos que publica el recibo de ventana, como `masked_campaign.publish`.
PUBLISHED = ("calibration", "evaluation")
CARRY_POLICY = dict(
    schema_version=2,
    kept=(
        "anchor_selected_parameters",
        "frozen_codec_projections",
        "recipe_admission_refinements_bank_capacity_seed_and_block_rows",
    ),
    reset=(
        "episodic_bank",
        "staged_admission",
        "pending_prediction_queue",
        "session_errors",
        "gru_state",
    ),
    warmup="inputs_in_the_months_before_the_measured_partition_without_labels",
    labels_from_anchor_window=False,
    parameters_updated=False,
    same_rule_as_fitted_window_heldout=True,
)


def bank_policy(warmup_months):
    """Devuelve la política de estado de la candidata con sus meses de calentamiento de entradas."""
    return dict(CARRY_POLICY, warmup_months=checked_warmup(warmup_months))


def anchor_warmup(window):
    """Devuelve los meses de calentamiento de la ventana del ancla, comprobados con sus fases.

    Un traslado, un adaptador o el padre congelado deben repetir exactamente el
    calentamiento con el que el ancla construyó sus índices. No hay valor por defecto: una
    ventana que no lo registra, o cuyas fases no corresponden a él, no se reutiliza.
    """
    policy = window.get("bank_policy")
    if not isinstance(policy, dict) or "warmup_months" not in policy:
        raise ValueError("La ventana del ancla no registra su calentamiento (warmup_months)")
    months = checked_warmup(policy["warmup_months"])
    expected = window_phases(window["fold"], months)
    declared = {name: record["phase"] for name, record in window["sources"].items()}
    if not declared or any(
        name not in expected or asdict(expected[name]) != phase for name, phase in declared.items()
    ):
        raise ValueError("Las fases de la ventana del ancla no corresponden a su calentamiento")
    return months


def _bounds(dataset):
    """Lee los límites de cada tramo de la vista y exige que coincidan en todos sus mercados."""
    declared = {}
    for temporal in dataset.temporals.values():
        bounds = {}
        for name, start, end, cutoff in temporal.partitioner.bounds:
            # En la versión 2 la purga es la maduración antes del final del tramo.
            if int(cutoff) != int(end):
                raise ValueError("La ventana no es una vista v2 con purga por intervalo")
            bounds[name] = (int(start), int(end))
        declared[tuple(sorted(bounds.items()))] = bounds
    if len(declared) != 1:
        raise ValueError("Los mercados de la vista no comparten los límites de la ventana")
    return next(iter(declared.values()))


def _phases(dataset, warmup_months):
    """Construye las fases de la ventana con el calentamiento común de los brazos con memoria.

    Exige que los tramos de decisión coincidan con los límites de la vista, de modo que
    el calentamiento solo añade entradas anteriores y nunca posteriores al tramo.
    """
    fold = next(iter(_window(dataset).values()))["fold"]
    phases = window_phases(fold, warmup_months)
    if {
        name: (phase.decision_start, phase.decision_end) for name, phase in phases.items()
    } != _bounds(dataset):
        raise ValueError("Las fases de la ventana no coinciden con los límites de la vista")
    return phases


def _window(dataset):
    contracts = temporal_contracts(dataset.manifest, input_policy=HISTORICAL_MASKED)
    if not contracts or set(dataset.temporals) != set(contracts):
        raise ValueError("La vista no declara su ventana walk-forward por mercado")
    folds = {view["fold"]["id"] for view in contracts.values()}
    if len(folds) != 1:
        raise ValueError("Los mercados de la vista pertenecen a ventanas distintas")
    return contracts


def window_sources(dataset, folder, partitions, warmup_months):
    """Preparar o reutilizar el índice de observaciones de cada tramo pedido."""
    phases = _phases(dataset, warmup_months)
    sources = {}
    for name in partitions:
        directory = Path(folder) / name
        manifest = prepare_observation_index(
            dataset, directory, phase=phases[name], resume=directory.exists()
        )
        sources[name] = FinancialObservationSource(dataset, manifest)
    return sources


def view_rows(dataset, partition):
    """Filas que la vista declara en un tramo: mercado, activo, instante y objetivo."""
    columns = dict(market=[], asset_id=[], prediction_at=[], target=[])
    for asset in sorted(dataset.assets, key=lambda row: (row["market"], row["symbol"])):
        with pq.ParquetFile(dataset._file(asset, "samples")) as file:
            rows = file.metadata.num_rows
        _, moments, values, _ = dataset._labels(asset, partition, rows)
        columns["market"] += [asset["market"]] * len(moments)
        columns["asset_id"] += [f"{asset['market']}/{asset['symbol']}"] * len(moments)
        columns["prediction_at"].append(np.asarray(moments, dtype=np.int64))
        columns["target"].append(np.asarray(values, dtype=np.float64))
    for name in ("prediction_at", "target"):
        columns[name] = np.concatenate(columns[name])
    if len(columns["target"]) != dataset.manifest["counts"][partition]:
        raise ValueError("Las etiquetas del tramo no concilian con los recuentos de la vista")
    return columns


def _table_rows(table):
    return dict(
        market=table["market"].to_pylist(),
        asset_id=table["asset_id"].to_pylist(),
        prediction_at=table["prediction_at"].cast(pa.int64()).to_numpy(),
        target=table["target"].to_numpy().astype(np.float64),
    )


def row_differences(observed, expected):
    """Contar filas de la vista sin predicción, predicciones ajenas, repetidas o con otro objetivo.

    Una fila se identifica por mercado, activo e instante, como en la comparación.
    """
    names = np.asarray(
        [
            f"{market}\n{asset}"
            for side in (observed, expected)
            for market, asset in zip(side["market"], side["asset_id"], strict=True)
        ],
        dtype=str,
    )
    sizes = len(observed["target"]), len(expected["target"])
    _, codes = np.unique(names, return_inverse=True)
    moments = np.concatenate([observed["prediction_at"], expected["prediction_at"]])
    targets = np.concatenate([observed["target"], expected["target"]])
    origin = np.repeat(np.array([0, 1]), sizes)
    order = np.lexsort((origin, moments, codes))
    codes, moments, targets, origin = codes[order], moments[order], targets[order], origin[order]
    same = (codes[1:] == codes[:-1]) & (moments[1:] == moments[:-1])
    repeated = same & (origin[1:] == origin[:-1])
    pairs = same & (origin[:-1] == 0) & (origin[1:] == 1)
    matched = int(np.count_nonzero(pairs))
    repeated_observed = int(np.count_nonzero(repeated & (origin[1:] == 0)))
    repeated_expected = int(np.count_nonzero(repeated & (origin[1:] == 1)))
    return dict(
        missing=sizes[1] - matched - repeated_expected,
        foreign=sizes[0] - matched - repeated_observed,
        repeated=repeated_observed + repeated_expected,
        target=int(np.count_nonzero(pairs & (targets[:-1] != targets[1:]))),
    )


def check_view_rows(table, dataset, partition):
    """Exigir exactamente las filas y objetivos de la vista o decir cuántas difieren."""
    differences = row_differences(_table_rows(table), view_rows(dataset, partition))
    total = sum(differences.values())
    if total:
        raise ValueError(
            f"{partition}: {total} filas difieren de la vista ({differences['missing']} sin "
            f"predicción, {differences['foreign']} ajenas, {differences['repeated']} repetidas "
            f"y {differences['target']} con otro objetivo)"
        )
    return table.num_rows


def write_receipts(output, contracts, parent, tables, labels_used_until):
    """Escribir y validar el recibo walk-forward de cada mercado de la ventana.

    `labels_used_until` es la maduración medida de `label_maturity.window_labels_until`,
    la misma que publica la campaña. `read_window_receipt` rechaza el recibo si alcanza la
    evaluación.
    """
    records = {}
    for market, view in sorted(contracts.items()):
        predictions = {}
        for partition, table in tables.items():
            rows = np.asarray(table["market"].to_pylist()) == market
            if rows.any():
                count, digest = prediction_fingerprint(
                    table["prediction_at"].cast(pa.int64()).to_numpy()[rows],
                    np.asarray(table["asset_id"].to_pylist())[rows],
                    table["prediction"].to_numpy()[rows],
                )
                predictions[partition] = dict(rows=count, sha256=digest)
        record = dict(
            kind=RECEIPT_KIND,
            schema_version=1,
            protocol=view["protocol"],
            fold=view["fold"],
            parent=parent,
            labels_used_until=labels_used_until,
            predictions=predictions,
        )
        read_window_receipt(record)
        path = Path(output) / "receipts" / f"{market}.json"
        atomic_json(path, record)
        records[market] = dict(path=str(path.relative_to(output)), sha256=sha256(path))
    return records


def _prepare_output(view, output, dataset):
    safe_destination(output)
    for protected in (*dataset.roots.values(), view.parent, Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)


def _dtype(name):
    dtypes = {"float32": torch.float32, "float64": torch.float64}
    if name not in dtypes:
        raise ValueError("El candidato se ajusta en float32 o float64")
    return dtypes[name]


def fit_window(
    view,
    output,
    recipe,
    *,
    seed,
    model,
    parent_id,
    device,
    warmup_months,
    stop=None,
    optimizer_factory=None,
    audit=False,
    joint_epoch=None,
):
    """Ajustar una semilla en una ventana, predecir sus tramos y escribir sus recibos.

    `model` declara `feature_seed`, `key_seed` y `dtype`. `warmup_months` es el
    calentamiento de entradas de la receta. Reanuda la ejecución y los índices confirmados
    si la salida ya existe. Devuelve el informe de la ventana, un estado `paused` si se
    pidió parar en una barrera o `awaiting_joint_stop` si la meseta conjunta espera la
    época común del grupo (`joint_epoch`).
    """
    require_learning_allowed("la ventana walk-forward de la GRU candidata")
    if type(recipe) is not CandidateRecipe:
        raise ValueError("La ventana necesita la receta declarada de la candidata")
    view, output = Path(view), Path(output)
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    contracts = _window(dataset)
    fold = next(iter(contracts.values()))["fold"]
    _prepare_output(view, output, dataset)
    output.mkdir(parents=True, exist_ok=True)
    sources = window_sources(dataset, output / "indices", PARTITIONS, warmup_months)
    adapter = CandidateInputAdapter(
        sources["train"].specification(),
        dtype=_dtype(model["dtype"]),
        device=device,
        parameter_seed=seed,
        feature_seed=model["feature_seed"],
        key_seed=model["key_seed"],
    )
    folder = output / "run"
    trainer = CandidateChronologicalTrainer(
        adapter,
        recipe,
        train=sources["train"],
        validation=sources["validation"],
        heldout={name: sources[name] for name in HELDOUT},
        output=folder,
        world=WORLD,
        fold=fold["id"],
        optimizer_factory=optimizer_factory,
        audit=audit,
    )
    report = trainer.run(resume=(folder / "run.json").is_file(), stop=stop, joint_epoch=joint_epoch)
    if report["status"] == AWAIT:
        return dict(
            status=AWAIT,
            individual_stop_epoch=report["individual_stop_epoch"],
            view=dict(path=str(view.resolve()), sha256=dataset.identity),
            final_test_opened=False,
        )
    if report["status"] != "completed":
        return dict(status=report["status"], final_test_opened=False)
    predictions, tables = {}, {}
    for name, record in report["predictions"].items():
        path = folder / record["path"]
        tables[name] = pq.read_table(path)
        predictions[name] = dict(
            path=str(path.relative_to(output)),
            sha256=record["sha256"],
            rows=check_view_rows(tables[name], dataset, name),
            metrics=record["metrics"],
        )
    best = report["best_checkpoint"]
    checkpoint = dict(
        path=str((folder / best["path"]).relative_to(output)),
        sha256=best["sha256"],
        parameters_sha256=best["parameters_sha256"],
    )
    parent = dict(id=parent_id, sha256=checkpoint["sha256"])
    # El ajuste, la selección y la calibración leen las etiquetas de la propia vista.
    until = window_labels_until(view, view)
    receipts = write_receipts(output, contracts, parent, {n: tables[n] for n in PUBLISHED}, until)
    window = dict(
        schema_version=1,
        kind=WINDOW_KIND,
        status="completed",
        model=MODEL,
        view=dict(path=str(view.resolve()), sha256=dataset.identity),
        fold=fold,
        markets=sorted(contracts),
        seed=seed,
        recipe=recipe.identity(),
        run=dict(
            path=str((folder / "run.json").relative_to(output)),
            sha256=sha256(folder / "run.json"),
            run_id=report["run_id"],
        ),
        checkpoint=checkpoint,
        best_epoch=report["best_epoch"],
        best_score=report["best_score"],
        joint_stop_epoch=report.get("joint_stop_epoch"),
        sources={
            name: dict(index_sha256=source.identity, phase=asdict(source.phase))
            for name, source in sources.items()
        },
        bank_policy=bank_policy(warmup_months),
        predictions=predictions,
        receipts=receipts,
        final_test_opened=False,
        finished_at_utc=datetime.now(UTC).isoformat(),
        **policy_identity(HISTORICAL_MASKED),
    )
    atomic_json(output / WINDOW_REPORT, window)
    return window


def _anchor(anchor, anchor_view):
    """Informe de ventana y de ejecución del ancla, con sus huellas comprobadas."""
    window, window_sha256 = read_manifest(anchor / WINDOW_REPORT, 8 * 1024**2)
    if (
        window.get("kind") != WINDOW_KIND
        or window.get("status") != "completed"
        or window.get("final_test_opened") is not False
        or window["view"]["sha256"] != sha256(anchor_view)
    ):
        raise ValueError("El ancla no es una ventana completa de la candidata sobre esa vista")
    path = anchor / window["run"]["path"]
    safe_destination(path)
    report, report_sha256 = read_manifest(path, 16 * 1024**2)
    if (
        report_sha256 != window["run"]["sha256"]
        or report.get("status") != "completed"
        or report["best_checkpoint"]["sha256"] != window["checkpoint"]["sha256"]
    ):
        raise ValueError("La ejecución del ancla no es la que confirmó su ventana")
    return window, window_sha256, report, path.parent


def anchor_adapter(anchor, anchor_view, specification, *, device):
    """Adaptador de otra ventana con el estado elegido en el ancla y su receta.

    Se construye con la configuración del ancla (semillas, precisión, codec) y después se
    cargan sus parámetros elegidos. Devuelve adaptador, receta, informe de ventana y su huella.
    """
    window, window_sha256, report, folder = _anchor(Path(anchor), Path(anchor_view))
    identity = report["identity"]
    declared = identity["recipe"]
    recipe = CandidateRecipe(**{item.name: declared[item.name] for item in fields(CandidateRecipe)})
    if recipe.identity() != declared:
        raise ValueError("La receta del ancla no se puede reconstruir sin cambios")
    configuration = identity["adapter"]["configuration"]
    adapter = CandidateInputAdapter(
        specification,
        dtype=_dtype(identity["dtype"].removeprefix("torch.")),
        device=device,
        parameter_seed=configuration["parameter_seed"],
        feature_seed=configuration["feature_seed"],
        key_seed=configuration["key_seed"],
    )
    parameters = restore_selected(folder, adapter, carried=True)
    if parameters != window["checkpoint"]["parameters_sha256"]:
        raise ValueError("Los parámetros cargados no son los elegidos en el ancla")
    return adapter, recipe, window, window_sha256


def carry_window(
    anchor,
    anchor_view,
    view,
    output,
    *,
    parent_id,
    device,
    stop=None,
    modality_ablation=None,
    regenerate=False,
):
    """Predecir calibración y evaluación de una ventana posterior con el estado del ancla.

    No ajusta nada. `carried_window` exige la misma edición y protocolo, y que la
    información del ancla termine antes de la calibración trasladada. Con
    `modality_ablation` predice solo la evaluación, también en la propia ventana del ancla,
    y no escribe recibos walk-forward. La GRU y el banco empiezan vacíos en el tramo, que
    lee las entradas ablacionadas. Con `regenerate` repite la validación, la calibración y
    la evaluación de la propia ventana del ancla, también sin recibos walk-forward.
    """
    require_learning_allowed("la predicción trasladada de la GRU candidata")
    started = time.perf_counter()
    anchor, anchor_view, view = Path(anchor), Path(anchor_view), Path(view)
    dataset = CorpusDataset(
        view, input_policy=HISTORICAL_MASKED, modality_ablation=modality_ablation
    )
    contracts = _window(dataset)
    anchor_meta, _ = read_manifest(anchor_view, 8 * 1024**2)
    partitions = predicted_partitions(modality_ablation, regenerate)
    same_view(anchor_meta, dataset.manifest, regenerate)
    anchor_fold, fold, months = carried_window(
        anchor_meta,
        dataset.manifest,
        input_policy=HISTORICAL_MASKED,
        same_window=modality_ablation is not None or regenerate,
    )
    output = _destination(output, dataset.roots.values())
    _prepare_output(view, output, dataset)
    output.mkdir(parents=True)
    # Cada tramo trasladado repite el calentamiento declarado en el ancla.
    anchor_window, _ = read_manifest(anchor / WINDOW_REPORT, 8 * 1024**2)
    warmup_months = anchor_warmup(anchor_window)
    sources = window_sources(dataset, output / "indices", partitions, warmup_months)
    adapter, recipe, window, window_sha256 = anchor_adapter(
        anchor, anchor_view, sources[partitions[0]].specification(), device=device
    )
    folder = output / "predictions"
    folder.mkdir()
    predictor = CandidateChronologicalPredictor(
        adapter, recipe, sources=sources, output=folder, world=WORLD, fold=fold["id"]
    )
    predictions, tables = {}, {}
    try:
        for name in partitions:
            path = folder / f"{name}-predictions.parquet"
            metrics = predictor.evaluate(sources[name], stop=stop, destination=path)
            tables[name] = pq.read_table(path)
            predictions[name] = dict(
                path=str(path.relative_to(output)),
                sha256=sha256(path),
                rows=check_view_rows(tables[name], dataset, name),
                metrics=metrics,
            )
    except _Pause:
        return dict(status="paused", final_test_opened=False)
    checkpoint = window["checkpoint"]
    parent = dict(id=parent_id, sha256=checkpoint["sha256"])
    # Una predicción ablacionada o regenerada no es una predicción walk-forward del brazo.
    published = not (modality_ablation or regenerate)
    receipts = {}
    if published:
        # Los parámetros se fijaron en la vista del ancla y la calibración común usa además
        # la calibración de esta ventana.
        until = window_labels_until(anchor_view, view)
        receipts = write_receipts(output, contracts, parent, tables, until)
    return _receipt(
        output,
        dict(
            model=MODEL,
            anchor=dict(
                run_sha256=window_sha256,
                checkpoint_sha256=checkpoint["sha256"],
                parameters_sha256=checkpoint["parameters_sha256"],
                manifest_sha256=window["view"]["sha256"],
                fold=anchor_fold,
            ),
            manifest_sha256=dataset.identity,
            fold=fold,
            months_since_anchor_information=months,
            recipe=recipe.identity(),
            adapter=adapter.identity(),
            sources={
                name: dict(index_sha256=source.identity, phase=asdict(source.phase))
                for name, source in sources.items()
            },
            bank_policy=bank_policy(warmup_months),
            bank_scope=dict(world=predictor.world, fold=predictor.fold),
            predictions=predictions,
            receipts=receipts,
            seconds=time.perf_counter() - started,
            **policy_identity(HISTORICAL_MASKED),
            **ablation_record(modality_ablation),
            **regeneration_record(regenerate),
        ),
    )


# Estos son los campos del caso que la campaña declara para la GRU candidata. La regla solo
# aparece si la campaña declara una parada temprana.
CASE_FIELDS = {"recipe", "recipe_sha256", "variant", "seed", "search_case"}
STOPPING_FIELD = "stopping_rule"


def campaign_case(case):
    """Construye la receta del caso, las opciones de modelo y el calentamiento con su huella.

    Si el caso declara la parada temprana de la campaña, la receta la aplica en lugar de
    la regla del protocolo, con la misma métrica.
    """
    if not isinstance(case, dict) or set(case) - {STOPPING_FIELD} != CASE_FIELDS:
        raise ValueError("El trabajo no declara un caso de la GRU candidata de la campaña")
    path = Path(case["recipe"])
    if sha256(path) != case["recipe_sha256"]:
        raise ValueError("La receta de la candidata cambió desde la planificación")
    recipe, document = load_recipe(path, variant=case["variant"], search_case=case["search_case"])
    model = document["model"]
    if case["seed"] not in model["seeds"]:
        raise ValueError("La semilla del caso no pertenece a la receta")
    rule = dict(recipe.selection, max_epochs=recipe.epochs)
    recipe = with_rule(recipe, campaign_rule(rule, case.get(STOPPING_FIELD)))
    options = {key: model[key] for key in ("feature_seed", "key_seed", "dtype")}
    return recipe, options, document["walk_forward"]["warmup_months"]


def run_job(run, *, device="cuda:0", optimizer_factory=None, regenerate=False):
    """Ejecutar un trabajo de la campaña: ajuste de ventana o predicción trasladada.

    Con `regenerate`, `run.anchor` es el intento del propio ajuste y se repiten sus
    predicciones por inferencia, sin ajustar nada.
    """
    if regenerate:
        return carry_window(
            run.anchor["folder"],
            run.anchor["view"],
            run.view,
            run.folder,
            parent_id=run.job["id"],
            device=device,
            stop=run.stop,
            regenerate=True,
        )
    if run.job["kind"] == "carry":
        report = carry_window(
            run.anchor["folder"],
            run.anchor["view"],
            run.view,
            run.folder,
            parent_id=run.anchor["job"],
            device=device,
            stop=run.stop,
        )
        view = report.get("manifest_sha256")
    else:
        recipe, model, warmup_months = campaign_case(run.case)
        report = fit_window(
            run.view,
            run.folder,
            recipe,
            seed=run.case["seed"],
            model=model,
            parent_id=run.job["id"],
            device=device,
            warmup_months=warmup_months,
            stop=run.stop,
            optimizer_factory=optimizer_factory,
            joint_epoch=run.joint_epoch,
        )
        view = report.get("view", {}).get("sha256")
    if report["status"] in ("completed", AWAIT) and view != run.view_sha256:
        raise ValueError("La candidata no confirma la vista del trabajo")
    return report
