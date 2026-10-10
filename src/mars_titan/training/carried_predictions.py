"""Predecir una ventana posterior con el estado seleccionado en la última ventana reentrenada.

Es la pieza de la variante B del presupuesto walk-forward. El modelo de la ventana
ancla se aplica sin cambios a la calibración y la evaluación de una ventana posterior
de la misma edición y el mismo protocolo. No se ajustan pesos, normalizadores ni
selección. Antes de leer predicciones se comprueba que el ancla dejó de aprender
(fin de su validación) antes de que empiece la calibración de la ventana trasladada.
Cada ventana conserva sus propias filas y su purga por intervalo de etiqueta.

Con `modality_ablation`, el mismo traslado vuelve a predecir solo la evaluación con una
variante de `data.modality_ablation` en la lectura. Es la pieza de la ablación de
modalidades: la ventana puede ser la propia ventana del ancla, cuyo estado se eligió con
su validación, anterior a la calibración y a la evaluación. Sin el parámetro, el traslado
no cambia.

Con `regenerate`, el traslado vuelve a predecir la validación, la calibración y la
evaluación de la propia ventana del ancla con su estado elegido. Es la regeneración de la
retención v2: repite por inferencia las tablas por fila que escribió el ajuste, que se
comparan bit a bit con las huellas registradas antes de liberar nada.
"""

import time
from datetime import UTC, date, datetime
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import masked_inputs, policy_identity
from mars_titan.data.modality_ablation import ablation_identity
from mars_titan.data.storage import atomic_json, outside_source, sha256

from .temporal_contract import temporal_contracts

# Tramos que necesita la comparación walk-forward: calibración común y evaluación.
CARRIED_PARTITIONS = ("calibration", "evaluation")
# Tramo de la ablación de modalidades. La calibración es la de la predicción original.
ABLATED_PARTITIONS = ("evaluation",)


# Tramos que escribe un ajuste de la campaña y que repite su regeneración.
REGENERATED_PARTITIONS = ("validation", "calibration", "evaluation")


def predicted_partitions(modality_ablation, regenerate=False):
    """Tramos que predice un traslado: normal, con ablación o como regeneración del ajuste."""
    if regenerate:
        if modality_ablation is not None:
            raise ValueError("La regeneración repite el ajuste, sin ablación de modalidades")
        return REGENERATED_PARTITIONS
    return CARRIED_PARTITIONS if modality_ablation is None else ABLATED_PARTITIONS


def regeneration_record(regenerate):
    """Campo del informe que marca una regeneración. Un traslado normal no lo declara."""
    return dict(regenerated=True) if regenerate else {}


def same_view(anchor_manifest, manifest, regenerate):
    """Una regeneración solo predice la propia vista del ancla."""
    if regenerate and anchor_manifest != manifest:
        raise ValueError("La regeneración solo predice la propia ventana del ancla")


def ablation_record(modality_ablation):
    """Campo del recibo que identifica la ablación. Un traslado normal no lo declara."""
    if modality_ablation is None:
        return {}
    return dict(modality_ablation=ablation_identity(modality_ablation))


def _months(start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return (last.year - first.year) * 12 + last.month - first.month


def carried_window(anchor_manifest, manifest, *, input_policy, same_window=False):
    """Validar que la ventana trasladada es posterior al ancla en la misma edición.

    Devuelve la ventana del ancla, la trasladada y los meses entre el final de la
    información del ancla y el comienzo de la evaluación trasladada. `same_window` admite
    además la misma vista del ancla, que solo usa la ablación de modalidades.
    """
    anchors = temporal_contracts(anchor_manifest, input_policy=input_policy)
    targets = temporal_contracts(manifest, input_policy=input_policy)
    if not anchors or set(anchors) != set(targets):
        raise ValueError("El ancla y la ventana trasladada no declaran los mismos mercados")
    for market, target in targets.items():
        if any(anchors[market][key] != target[key] for key in ("protocol", "parent_sha256")):
            raise ValueError("El ancla pertenece a otro protocolo o a otra edición")
    # La unión de mercados ya exige la misma ventana en los dos calendarios.
    anchor, target = (next(iter(anchors.values()))["fold"], next(iter(targets.values()))["fold"])
    # El ancla seleccionó su estado con etiquetas maduras antes del final de su validación.
    later = anchor["evaluation"][0] < target["evaluation"][0]
    same = same_window and anchor_manifest == manifest
    if not ((later or same) and anchor["validation"][1] <= target["calibration"][0]):
        raise ValueError("La ventana trasladada no es posterior a la información del ancla")
    return anchor, target, _months(anchor["validation"][1], target["evaluation"][0])


def _destination(output, sources):
    output = Path(output)
    safe_destination(output)
    for protected in (*sources, Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    if output.exists() or output.is_symlink():
        raise ValueError("Las predicciones trasladadas necesitan un directorio nuevo")
    return output


def _receipt(output, record):
    json_record = dict(
        schema_version=1,
        kind="carried_predictions",
        status="completed",
        final_test_opened=False,
        scientific_training_started=False,
        finished_at_utc=datetime.now(UTC).isoformat(),
        **record,
    )
    atomic_json(output / "carry.json", json_record)
    return json_record


def selected_reference(anchor, anchor_manifest, *, input_policy):
    """Lee el informe y el estado elegido de una referencia neuronal confirmada.

    El ancla debe estar completa y seleccionada, con la reserva cerrada, la vista indicada,
    la misma política de entradas y el mismo entorno y código.
    """
    from .reference_run import _confirmed_state, read_json, scientific_identity

    anchor, anchor_manifest = Path(anchor), Path(anchor_manifest)
    report = read_json(anchor / "run.json")
    identity = report["identity"]
    case = identity["case"]
    if (
        report.get("status") != "completed"
        or report.get("final_test_opened") is not False
        or not case.get("selection")
        or sha256(anchor_manifest) != identity["manifest_sha256"]
        or any(identity.get(key) != value for key, value in policy_identity(input_policy).items())
    ):
        raise ValueError("El ancla no es una referencia seleccionada de la misma política")
    current = scientific_identity(
        kind=case["kind"], input_policy=input_policy, head=case.get("head")
    )
    if any(identity.get(key) != value for key, value in current.items()):
        raise ValueError("El entorno o el código no coincide con el ancla")
    state = _confirmed_state(anchor, identity, report["checkpoint"], report.get("selection"))
    return report, state


def reference_model(identity, state, device):
    """Construye la referencia de una identidad confirmada y carga su estado elegido.

    Devuelve el modelo y si emite los cinco cuantiles de la cabeza común.
    """
    from mars_titan.models.baselines.multimodal import STRICT_FUSION, MultimodalReference
    from mars_titan.models.quantile_head import QUANTILE_HEAD

    case = identity["case"]
    quantiles = case.get("head") == QUANTILE_HEAD
    model = MultimodalReference(
        case["kind"],
        identity["dimensions"],
        context=identity["context"],
        mask_fusion=identity.get("mask_fusion", STRICT_FUSION),
        **case["architecture"],
        **({"head": QUANTILE_HEAD} if quantiles else {}),
    ).to(device)
    model.load_state_dict(state["model"])
    return model, quantiles


def carry_reference(
    anchor,
    anchor_manifest,
    manifest,
    output,
    *,
    batch_size,
    input_policy,
    stop=None,
    modality_ablation=None,
    regenerate=False,
):
    """Aplicar el estado seleccionado de una referencia neuronal a otra ventana."""
    import torch

    from mars_titan.data.embeddings import require_cuda

    from .reference_run import _evaluate, configured_corpus

    started = time.perf_counter()
    anchor, anchor_manifest, manifest = Path(anchor), Path(anchor_manifest), Path(manifest)
    report, state = selected_reference(anchor, anchor_manifest, input_policy=input_policy)
    identity = report["identity"]
    anchor_meta, _ = read_manifest(anchor_manifest, 8 * 1024**2)
    partitions = predicted_partitions(modality_ablation, regenerate)
    dataset = configured_corpus(
        manifest, input_policy=input_policy, modality_ablation=modality_ablation
    )
    same_view(anchor_meta, dataset.manifest, regenerate)
    anchor_fold, fold, age = carried_window(
        anchor_meta,
        dataset.manifest,
        input_policy=input_policy,
        same_window=modality_ablation is not None or regenerate,
    )
    if dataset.context != identity["context"] or dataset.manifest["scope"] != report["scope"]:
        raise ValueError("La ventana trasladada no conserva el contexto ni el alcance del ancla")
    output = _destination(output, dataset.roots.values())
    device = require_cuda()
    model, quantiles = reference_model(identity, state, device)
    first = next(dataset.batches(partition="train", batch_size=1, epoch=0, seed=0))
    if {name: value.shape[-1] for name, value in first["inputs"].items()} != identity["dimensions"]:
        raise ValueError("Las dimensiones de la ventana no coinciden con las del ancla")
    output.mkdir(parents=True)
    predictions = {}
    torch.cuda.reset_peak_memory_stats(0)
    for partition in partitions:
        path = output / f"{partition}-predictions.parquet"
        metrics = _evaluate(
            model,
            dataset,
            batch_size,
            device=device,
            partition=partition,
            destination=path,
            stop=stop,
            quantiles=quantiles,
        )
        predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
    return _receipt(
        output,
        dict(
            model="reference",
            anchor=dict(
                run_sha256=sha256(anchor / "run.json"),
                checkpoint_sha256=report["checkpoint"]["sha256"],
                manifest_sha256=identity["manifest_sha256"],
                fold=anchor_fold,
            ),
            manifest_sha256=dataset.identity,
            fold=fold,
            months_since_anchor_information=age,
            batch_size=batch_size,
            predictions=predictions,
            seconds=time.perf_counter() - started,
            peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0),
            **policy_identity(input_policy),
            **ablation_record(modality_ablation),
            **regeneration_record(regenerate),
        ),
    )


def _tabular_model(anchor, report, kind):
    """Cargar el modelo confirmado del ancla después de comprobar su huella."""
    checkpoint = report["checkpoint"]
    path = anchor / checkpoint["path"]
    safe_destination(path)
    if kind == "ridge":
        from mars_titan.models.baselines.ridge import RidgeModel

        if sha256(path) != checkpoint["sha256"]:
            raise ValueError("La huella del modelo Ridge del ancla no coincide")
        return RidgeModel.load(path)
    from mars_titan.models.baselines.external_boosting import ExternalBoostingModel

    return ExternalBoostingModel.load(
        path, checkpoint["sha256"], training_rows=report["samples"]["train"]
    )


TABULAR_REPORTS = {"ridge": "ridge", "xgboost": "xgboost_external_cuda"}


def carry_tabular(
    anchor,
    anchor_manifest,
    manifest,
    output,
    *,
    kind,
    batch_size,
    input_policy,
    modality_ablation=None,
    regenerate=False,
):
    """Aplicar el modelo Ridge o XGBoost seleccionado en el ancla a otra ventana."""
    import numpy as np

    from .corpus_inputs import CorpusDataset
    from .tabular_corpus import _matrix, _predict, feature_order

    started = time.perf_counter()
    if kind not in TABULAR_REPORTS:
        raise ValueError("Solo se trasladan modelos Ridge o XGBoost")
    anchor, anchor_manifest, manifest = Path(anchor), Path(anchor_manifest), Path(manifest)
    report, report_sha256 = read_manifest(anchor / "run.json", 8 * 1024**2)
    declared = report.get("identity", report)
    if (
        report.get("status") != "completed"
        or report.get("model") != TABULAR_REPORTS[kind]
        or report.get("final_test_opened") is not False
        or declared.get("manifest_sha256") != sha256(anchor_manifest)
        or report.get("feature_order") != feature_order(input_policy)
        or any(declared.get(key) != value for key, value in policy_identity(input_policy).items())
    ):
        raise ValueError("El ancla no es un modelo tabular confirmado de la misma política")
    anchor_meta, _ = read_manifest(anchor_manifest, 8 * 1024**2)
    partitions = predicted_partitions(modality_ablation, regenerate)
    dataset = CorpusDataset(
        manifest, input_policy=input_policy, modality_ablation=modality_ablation
    )
    same_view(anchor_meta, dataset.manifest, regenerate)
    anchor_fold, fold, age = carried_window(
        anchor_meta,
        dataset.manifest,
        input_policy=input_policy,
        same_window=modality_ablation is not None or regenerate,
    )
    output = _destination(output, dataset.roots.values())
    masked = masked_inputs(input_policy)
    first = next(dataset.batches(partition="train", batch_size=1, epoch=0, seed=0))
    if "features" in report and _matrix(first, presence=masked).shape[1] != report["features"]:
        raise ValueError("Las columnas de la ventana no coinciden con las del ancla")
    model = _tabular_model(anchor, report, kind)
    dtype = np.float64 if kind == "ridge" else np.float32
    output.mkdir(parents=True)
    predictions = {}
    for partition in partitions:
        path = output / f"{partition}-predictions.parquet"
        metrics = _predict(
            model, None, dataset, partition, batch_size, path, dtype=dtype, presence=masked
        )
        predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
    return _receipt(
        output,
        dict(
            model=TABULAR_REPORTS[kind],
            anchor=dict(
                run_sha256=report_sha256,
                checkpoint_sha256=report["checkpoint"]["sha256"],
                manifest_sha256=declared["manifest_sha256"],
                fold=anchor_fold,
            ),
            manifest_sha256=dataset.identity,
            fold=fold,
            months_since_anchor_information=age,
            batch_size=batch_size,
            feature_order=feature_order(input_policy),
            predictions=predictions,
            seconds=time.perf_counter() - started,
            **policy_identity(input_policy),
            **ablation_record(modality_ablation),
            **regeneration_record(regenerate),
        ),
    )
