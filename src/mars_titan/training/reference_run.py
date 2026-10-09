"""Épocas completas de referencias multimodales con cursor y estado recuperables."""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from mars_titan.budget_training import seed_run, validate_loss
from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.input_policy import STRICT_INPUTS, masked_inputs, policy_identity
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.models.baselines.multimodal import (
    PRESENCE_FUSION,
    STRICT_FUSION,
    MultimodalReference,
    transformer_batch_options,
    validate_architecture,
)
from mars_titan.models.baselines.transformer import (
    transformer_options,
    validate_attention_budget,
)
from mars_titan.models.quantile_head import (
    CONTRACT,
    LEVELS,
    PINBALL,
    QUANTILE_COLUMNS,
    QUANTILE_HEAD,
    median,
    pinball_loss,
)
from mars_titan.profiling import CostProbe

from .checkpoints import (
    StopRequest,
    capture_rng,
    load_training_state,
    restore_rng,
    save_training_state,
)
from .corpus_inputs import CorpusDataset
from .learning_hold import require_learning_allowed
from .selection import (
    AWAIT,
    CONTINUE,
    FINISH,
    FIXED_BUDGET,
    JOINT_PLATEAU,
    advance_selection,
    awaiting,
    bind_joint_epoch,
    epoch_decision,
    initial_selection,
    validate_selection,
)

KINDS = ("rnn", "lstm", "gru", "dlinear", "transformer")
# Política anterior: filas completas de ajuste y validación.
FULL_TRAIN_VALIDATION = "full_train_validation_v1"
# Filas completas de validación, calibración y evaluación. El ajuste se resume por sesión.
HELDOUT_FULL_TRAIN_SESSIONS = "heldout_full_train_sessions_v1"
PREDICTION_RETENTIONS = (FULL_TRAIN_VALIDATION, HELDOUT_FULL_TRAIN_SESSIONS)


class _Pause(Exception):
    """Solicitud de parada entre operaciones, sin declarar un fallo científico."""


def read_json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise ValueError("Falta un recibo regular dentro del presupuesto")
    return json.loads(path.read_text())


def configured_corpus(manifest, *, input_policy=STRICT_INPUTS, modality_ablation=None):
    """Activar las tablas solo con un presupuesto explícito para esta ejecución."""
    budget = os.environ.get("MARS_TITAN_INPUT_CACHE_MIB")
    if budget is None:
        return CorpusDataset(
            manifest, input_policy=input_policy, modality_ablation=modality_ablation
        )
    if (
        not 1 <= len(budget) <= 4
        or not budget.isascii()
        or not budget.isdecimal()
        or not 1 <= int(budget) <= 4096
    ):
        raise ValueError("MARS_TITAN_INPUT_CACHE_MIB debe ser un entero entre 1 y 4096")
    return CorpusDataset(
        manifest,
        cache_bytes=int(budget) * 1024**2,
        cache_sample_tables=True,
        input_policy=input_policy,
        modality_ablation=modality_ablation,
    )


_SOURCES = (
    "training/reference_run.py",
    "training/reference_campaign.py",
    "training/selection.py",
    "training/checkpoints.py",
    "training/corpus_inputs.py",
    "training/temporal_corpus.py",
    "training/temporal_contract.py",
    "evaluation/splits.py",
    "evaluation/split_readiness.py",
    "data/cohort_contexts.py",
    "data/samples.py",
    "data/temporal.py",
    "profiling.py",
    "models/baselines/dlinear.py",
    "models/baselines/multimodal.py",
    "training/cohort_contract.py",
    "evaluation/session_metrics.py",
    "models/baselines/campaign.py",
    "budget_training.py",
    "data/streaming.py",
    "data/batches.py",
    "data/cohort_files.py",
    "data/storage.py",
    "data/embeddings.py",
)


def scientific_identity(*, kind=None, input_policy=STRICT_INPUTS, head=None):
    """Compartir versiones, política numérica y transformaciones entre todos los casos.

    El Transformer, la lectura con máscaras y la cabeza de cuantiles añaden sus
    huellas solo en esos casos. Así, la identidad de las referencias estrictas y
    escalares anteriores conserva sus campos.
    """
    root = Path(__file__).parents[1]
    sources = _SOURCES
    if kind == "transformer":
        sources += ("models/baselines/transformer.py",)
    if masked_inputs(input_policy):
        sources += ("data/input_policy.py",)
    if head == QUANTILE_HEAD:
        sources += ("models/quantile_head.py",)
    return dict(
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        numpy=np.__version__,
        pyarrow=pa.__version__,
        exchange_calendars=importlib.metadata.version("exchange-calendars"),
        python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0),
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        numerics={
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        code={name: sha256(root / name) for name in sources},
    )


def _options(
    case,
    batch_size,
    checkpoint_seconds,
    checkpoint_steps,
    *,
    input_policy=STRICT_INPUTS,
    prediction_retention=FULL_TRAIN_VALIDATION,
):
    required = {"kind", "loss", "learning_rate", "seed", "epochs", "huber_delta"}
    if (
        not required <= set(case) <= required | {"architecture", "selection", "head"}
        or case["kind"] not in KINDS
        or type(case["epochs"]) is not int
        or not 1 <= case["epochs"] <= 1000
        or type(case["seed"]) is not int
        or not 0 <= case["seed"] < 2**32
        or type(batch_size) is not int
        or not 1 <= batch_size <= 4096
        or type(checkpoint_steps) is not int
        or checkpoint_steps < 0
        or not math.isfinite(checkpoint_seconds)
        or checkpoint_seconds <= 0
        or not math.isfinite(case["learning_rate"])
        or case["learning_rate"] <= 0
    ):
        raise ValueError("La configuración del entrenamiento no es válida")
    if prediction_retention not in PREDICTION_RETENTIONS:
        raise ValueError("La retención de predicciones no pertenece al contrato")
    transformer = case["kind"] == "transformer"
    if (transformer or masked_inputs(input_policy)) and "architecture" not in case:
        raise ValueError(
            "El Transformer y la política con máscaras requieren una arquitectura científica"
        )
    if "architecture" in case:
        architecture = case["architecture"]
        expected = {"hidden_size", "layers", "dropout"} | (
            {"transformer"} if transformer else set()
        )
        if not isinstance(architecture, dict) or set(architecture) != expected:
            raise ValueError("La arquitectura necesita anchura, profundidad y regularización")
        validate_architecture(**{k: v for k, v in architecture.items() if k != "transformer"})
        if transformer and not isinstance(architecture["transformer"], dict):
            raise ValueError("El Transformer necesita cabezas y anchura FFN explícitas")
        if transformer:
            transformer_options(architecture["transformer"])
    if "selection" in case:
        validate_selection(case["selection"], epochs=case["epochs"])
    # La ruta escalar se identifica por la ausencia del campo. La cabeza de cuantiles
    # solo admite su pinball y conserva huber_delta para compartir el esquema del caso.
    if "head" in case:
        if case["head"] != QUANTILE_HEAD or "architecture" not in case:
            raise ValueError("La cabeza de cuantiles necesita su nombre y una arquitectura")
        if case["loss"] != PINBALL:
            raise ValueError("La cabeza de cuantiles se ajusta con su pérdida pinball")
        validate_loss("mae", case["huber_delta"])
    else:
        validate_loss(case["loss"], case["huber_delta"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("Configura CUBLAS_WORKSPACE_CONFIG antes de iniciar PyTorch")


def _inputs(batch, device):
    return {k: torch.from_numpy(v).to(device) for k, v in batch["inputs"].items()}


def _forward(model, batch, device):
    """Entregar la presencia solo cuando el lector aplica la política con máscaras."""
    if "presence" not in batch:
        return model(_inputs(batch, device))
    return model(_inputs(batch, device), torch.from_numpy(batch["presence"]).to(device))


def _statistics():
    return dict(samples=0, squared_error=0.0, absolute_error=0.0, elapsed_seconds=0.0)


def _metrics(statistics):
    count = statistics["samples"]
    if not count:
        raise ValueError("No se puede evaluar una partición vacía")
    return {
        **statistics,
        "mse": statistics["squared_error"] / count,
        "mae": statistics["absolute_error"] / count,
        "samples_per_second": count / max(statistics["elapsed_seconds"], 1e-9),
    }


def _update(statistics, prediction, target):
    error = prediction.detach().to(torch.float64) - target.to(torch.float64)
    sums = torch.stack([error.square().sum(), error.abs().sum()]).tolist()
    if not all(math.isfinite(v) for v in sums):
        raise ValueError("El error contiene valores no finitos")
    statistics["samples"] += len(target)
    statistics["squared_error"] += sums[0]
    statistics["absolute_error"] += sums[1]


def row_loss(case, emitted, prediction, target, quantiles):
    """Calcula la pérdida por fila, pinball con la cabeza de cuantiles o la escalar del caso."""
    if quantiles:
        return pinball_loss(emitted, target, reduction="none")
    if case["loss"] == "huber":
        return torch.nn.functional.huber_loss(
            prediction, target, delta=case["huber_delta"], reduction="none"
        )
    if case["loss"] == "mae":
        return torch.nn.functional.l1_loss(prediction, target, reduction="none")
    return torch.nn.functional.mse_loss(prediction, target, reduction="none")


def _session_table(errors, zero):
    """Resumir errores por mercado e instante, junto al control de predicción nula."""
    keys = sorted(errors.sessions, key=lambda key: (key[1], key[0]))
    if keys != sorted(zero.sessions, key=lambda key: (key[1], key[0])):
        raise ValueError("El resumen por sesión no concilia el modelo y el control nulo")
    model_rows = [errors.sessions[key] for key in keys]
    zero_rows = [zero.sessions[key] for key in keys]
    return pa.table(
        {
            "market": [market for market, _ in keys],
            "prediction_at": pa.array(
                [moment for _, moment in keys], type=pa.timestamp("us", tz="UTC")
            ),
            "samples": pa.array([row[0] for row in model_rows], type=pa.int64()),
            "absolute_error": [row[1] for row in model_rows],
            "squared_error": [row[2] for row in model_rows],
            "zero_absolute_error": [row[1] for row in zero_rows],
            "zero_squared_error": [row[2] for row in zero_rows],
        }
    )


def _point(emitted, target, quantiles):
    """Separar la predicción puntual y, con la cabeza de cuantiles, sus cinco niveles."""
    if quantiles and emitted.shape != (len(target), len(LEVELS)):
        raise ValueError("La cabeza de cuantiles debe devolver cinco niveles por fila")
    prediction = median(emitted) if quantiles else emitted
    if prediction.shape != target.shape:
        raise ValueError("Predicción y etiqueta no tienen la misma forma")
    return prediction


def _evaluate(
    model,
    dataset,
    batch_size,
    *,
    device="cuda:0",
    partition="validation",
    destination=None,
    sessions=None,
    stop=None,
    quantiles=False,
):
    """Evaluar una partición completa. `sessions` resume filas sin guardarlas una a una.

    Con `quantiles`, la predicción puntual es la mediana y las tablas por fila añaden
    los cinco niveles en las columnas del contrato de la cabeza.
    """
    if destination is not None and sessions is not None:
        raise ValueError("Una evaluación guarda filas completas o un resumen por sesión")
    model.eval()
    accumulator = SessionErrors()
    zero = SessionErrors() if sessions is not None else None
    # Dos flujos separados no dependen de las fronteras entre lotes.
    digests = dict(sample_ids=hashlib.sha256(), predictions=hashlib.sha256())
    start = time.perf_counter()

    def tables():
        with torch.inference_mode():
            for batch in dataset.batches(
                partition=partition, batch_size=batch_size, epoch=0, seed=0
            ):
                if stop is not None and stop.requested:
                    raise _Pause
                emitted = _forward(model, batch, device)
                predicted = _point(emitted, batch["target"], quantiles)
                predictions = predicted.cpu().numpy()
                accumulator.update(
                    batch["market"],
                    batch["prediction_at"],
                    predictions.astype(np.float64) - batch["target"],
                )
                if zero is not None:
                    zero.update(batch["market"], batch["prediction_at"], -batch["target"])
                    digests["sample_ids"].update(
                        "".join(f"{key}\n" for key in batch["sample_ids"]).encode()
                    )
                    digests["predictions"].update(predictions.astype("<f4").tobytes())
                if destination is None:
                    continue
                columns = {
                    "sample_id": batch["sample_ids"],
                    "asset_id": ["/".join(key.split("/")[:2]) for key in batch["sample_ids"]],
                    "market": batch["market"],
                    "prediction_at": pa.array(
                        batch["prediction_at"], type=pa.timestamp("us", tz="UTC")
                    ),
                    "target": batch["target"],
                    "prediction": predictions,
                    "zero": np.zeros(len(predictions), dtype=np.float64),
                }
                if quantiles:
                    # La mediana se guarda dos veces con los mismos bits: como predicción
                    # puntual y como su nivel, igual que exige ForecastPanel.
                    levels = emitted.cpu().numpy()
                    columns.update(zip(QUANTILE_COLUMNS, levels.T, strict=True))
                yield pa.table(columns)

    if destination is None:
        for _ in tables():
            pass
    else:
        atomic_parquet_batches(destination, tables())
    torch.cuda.synchronize(0)
    statistics = accumulator.summary()
    statistics["elapsed_seconds"] = time.perf_counter() - start
    if statistics["samples"] != dataset.manifest["counts"][partition]:
        raise ValueError("La evaluación no reconcilia toda la población")
    metrics = _metrics(statistics)
    if sessions is not None:
        atomic_parquet_batches(sessions, [_session_table(accumulator, zero)])
        metrics.update(
            zero=_metrics(zero.summary() | dict(elapsed_seconds=statistics["elapsed_seconds"])),
            sample_ids_sha256=digests["sample_ids"].hexdigest(),
            predictions_float32_sha256=digests["predictions"].hexdigest(),
        )
    return metrics


def retained_artifacts(report):
    """Enumerar los archivos que acredita un recibo, incluido el resumen del ajuste."""
    return [*report["predictions"].values(), *filter(None, [report.get("train_summary")])]


def training_weights(assets, weighting):
    """Ponderar por mercado con recuentos de ajuste, sin retirar filas de ninguno."""
    if weighting not in {"natural", "balanced_markets"}:
        raise ValueError("La ponderación debe ser natural o equilibrada entre mercados")
    counts = {}
    for asset in assets:
        market = asset["market"]
        counts[market] = counts.get(market, 0) + asset["counts"]["train"]
    if not counts or min(counts.values()) < 1:
        raise ValueError("Cada mercado necesita una población de ajuste no vacía")
    total = sum(counts.values())
    return {
        m: total / (len(counts) * n) if weighting == "balanced_markets" else 1.0
        for m, n in counts.items()
    }


def _confirmed_state(directory, identity, checkpoint, selection_state=None):
    choice = "best" if identity["case"].get("selection") else "latest"
    index = read_json(directory / "checkpoints/latest.json")
    record = index["best"] if choice == "best" else next(iter(index["latest"]), None)
    if record is None or checkpoint != dict(
        path=f"checkpoints/{record['name']}", sha256=record["sha256"]
    ):
        raise ValueError("El punto de control confirmado ha cambiado")
    state = load_training_state(
        directory / "checkpoints",
        expected_identity=identity,
        selection=choice,
        expected_sha256=checkpoint["sha256"],
    )
    if choice == "best" and (
        selection_state is None
        or state.get("epoch") != selection_state["best_epoch"]
        or state.get("confirmed_cursor") is not None
        or state.get("statistics", {}).get("samples") != 0
        or state.get("selection", {}).get("best_score") != selection_state["best_score"]
    ):
        raise ValueError("El estado no corresponde a la época y evaluación seleccionadas")
    return state


def _parent(parent, dataset, case, model, batch_size, weighting, input_policy=STRICT_INPUTS):
    if parent is None:
        return None
    report = read_json(parent / "run.json")
    identity = report["identity"]
    inputs = policy_identity(input_policy)
    if (
        report.get("status") != "completed"
        or identity["manifest_sha256"] != dataset.identity
        or identity["case"]["kind"] != case["kind"]
        or identity["case"]["seed"] != case["seed"]
        or identity["case"].get("architecture") != case.get("architecture")
        or identity["case"].get("head") != case.get("head")
        or identity["batch_size"] != batch_size
        or identity["weighting"] != weighting
        or any(identity.get(key) != inputs.get(key) for key in ("input_policy", "mask_contract"))
        or identity.get("mask_fusion") != (PRESENCE_FUSION if inputs else None)
    ):
        raise ValueError("El origen no corresponde a la población, arquitectura y semilla")
    checkpoint = report["checkpoint"]
    current = scientific_identity(
        kind=case["kind"], input_policy=input_policy, head=case.get("head")
    )
    if any(identity.get(k) != v for k, v in current.items()):
        raise ValueError("El entorno o el código no coincide con el origen de la continuación")
    state = _confirmed_state(parent, identity, checkpoint, report.get("selection"))
    model.load_state_dict(state["model"])
    return dict(parent_checkpoint_sha256=checkpoint["sha256"], optimizer_policy="new_adamw")


def run_reference_case(
    manifest: Path,
    output: Path,
    case: dict,
    *,
    batch_size: int = 16,
    resume: bool = False,
    checkpoint_seconds: float = 900,
    checkpoint_steps: int = 0,
    stop: StopRequest | None = None,
    initialize_from: Path | None = None,
    weighting: str = "natural",
    input_policy: str = STRICT_INPUTS,
    prediction_retention: str = FULL_TRAIN_VALIDATION,
    joint_epoch: int | None = None,
) -> dict:
    """Ajustar una referencia sobre toda la edición, sin abrir el test final.

    Sin argumentos nuevos se conserva la ruta estricta: lectura, identidad y archivos.
    La política con máscaras activa la fusión con presencia y la registra en la identidad.
    Con la meseta conjunta, el ajuste espera en su primera meseta (`awaiting_joint_stop`)
    hasta que se reanuda con la época común del grupo (`joint_epoch`).
    """
    require_learning_allowed("el ajuste de la referencia neuronal")
    _options(
        case,
        batch_size,
        checkpoint_seconds,
        checkpoint_steps,
        input_policy=input_policy,
        prediction_retention=prediction_retention,
    )
    masked = masked_inputs(input_policy)
    quantiles = case.get("head") == QUANTILE_HEAD
    start = time.perf_counter()
    device = require_cuda()
    dataset = configured_corpus(manifest, input_policy=input_policy)
    if min(dataset.manifest["counts"].values()) < 1:
        raise ValueError("Se necesitan datos de ajuste y validación en este brazo")
    if case["kind"] == "transformer":
        architecture = case["architecture"]
        validate_attention_budget(
            batch_size,
            context=dataset.context,
            heads=architecture["transformer"]["heads"],
            layers=architecture["layers"],
            **transformer_batch_options(case["kind"], batch_size),
        )
    for protected in (*dataset.roots.values(), Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    if output.is_symlink() or (output.exists() and not resume) or (resume and not output.is_dir()):
        raise ValueError("Usa una ejecución nueva o solicita continuar una existente")
    seed_run(case["seed"])
    first = next(
        dataset.batches(partition="train", batch_size=batch_size, epoch=0, seed=case["seed"])
    )
    dimensions = {name: value.shape[-1] for name, value in first["inputs"].items()}
    model = (
        MultimodalReference(
            case["kind"],
            dimensions,
            context=dataset.context,
            mask_fusion=PRESENCE_FUSION if masked else STRICT_FUSION,
            **case["architecture"],
            **({"head": QUANTILE_HEAD} if quantiles else {}),
            **transformer_batch_options(case["kind"], batch_size),
        )
        if "architecture" in case
        else CostProbe(case["kind"], dimensions, context=dataset.context)
    ).to(device)
    weights = training_weights(dataset.assets, weighting)
    initialization = _parent(
        initialize_from, dataset, case, model, batch_size, weighting, input_policy
    )
    identity = dict(
        **scientific_identity(kind=case["kind"], input_policy=input_policy, head=case.get("head")),
        manifest_sha256=dataset.identity,
        case=case,
        model_family="scientific_multimodal_reference"
        if "architecture" in case
        else "legacy_cost_probe",
        batch_size=batch_size,
        dimensions=dimensions,
        context=dataset.context,
        initialization=initialization,
        weighting=weighting,
        market_weights=weights,
        optimizer="AdamW",
        precision="float32",
        device="cuda:0",
        input_cache=dict(
            budget_bytes=dataset.cache_limit,
            sample_tables=dataset.cache_sample_tables,
            entry_limit=dataset.cache_entry_limit,
        ),
    )
    # Los campos nuevos solo aparecen fuera de la ruta estricta anterior.
    identity.update(policy_identity(input_policy))
    if masked:
        identity["mask_fusion"] = PRESENCE_FUSION
    if prediction_retention != FULL_TRAIN_VALIDATION:
        identity["prediction_retention"] = prediction_retention
    if quantiles:
        identity["output_head"] = dict(CONTRACT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=case["learning_rate"])
    report_path = output / "run.json"
    if resume and report_path.exists():
        report = read_json(report_path)
        if report["identity"] != identity:
            raise ValueError("La identidad o configuración de la ejecución ha cambiado")
        # Una ejecución ya conjunta solo continúa o se confirma con su misma época común.
        bind_joint_epoch(report, joint_epoch, case.get("selection"), case["epochs"])
        if report["status"] == "completed":
            _confirmed_state(output, identity, report["checkpoint"], report.get("selection"))
            for item in retained_artifacts(report):
                if sha256(output / item["path"]) != item["sha256"]:
                    raise ValueError("Han cambiado las predicciones confirmadas")
            return report
    else:
        if resume and any(
            not (p.name.startswith(".run.json.") and p.is_file() and not p.is_symlink())
            for p in output.iterdir()
        ):
            raise ValueError("El inicio interrumpido contiene artefactos desconocidos")
        output.mkdir(parents=True, exist_ok=resume)
        report = dict(
            schema_version=1,
            status="running",
            identity=identity,
            scope=dataset.manifest["scope"],
            cohort_complete=dataset.manifest["cohort_complete"],
            samples=dataset.manifest["counts"],
            epochs=[],
            global_step=0,
            parameters=sum(p.numel() for p in model.parameters()),
            device="cuda:0",
            final_test_opened=False,
            initialization=initialization,
            started_at_utc=datetime.now(UTC).isoformat(),
            attempts=[],
        )
        atomic_json(report_path, report)
    epoch, cursor, step, statistics, history = 0, None, 0, _statistics(), []
    selection, selection_options = None, case.get("selection")
    evaluating_selected = False
    if resume and (output / "checkpoints/latest.json").exists():
        state = load_training_state(output / "checkpoints", expected_identity=identity)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], "cuda:0")
        epoch, cursor, step = state["epoch"], state["confirmed_cursor"], state["global_step"]
        statistics, history = state["statistics"], state["history"]
        selection = state.get("selection")
        if state.get("initial_validation") is not None:
            report["initial_validation"] = state["initial_validation"]
    report["selection"] = selection
    bind_joint_epoch(report, joint_epoch, case.get("selection"), case["epochs"])
    stop = stop or StopRequest()
    last_saved = time.perf_counter()

    def decision():
        if not selection_options:
            return FINISH if epoch >= case["epochs"] else CONTINUE
        return epoch_decision(selection, selection_options, case["epochs"], joint_epoch)

    def save(*, pin=False, best=False):
        nonlocal last_saved
        if any(
            sha256(Path(__file__).parents[1] / name) != expected
            for name, expected in identity["code"].items()
        ):
            raise ValueError("El código ha cambiado durante la ejecución")
        state = dict(
            global_step=step,
            epoch=epoch,
            confirmed_cursor=cursor,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=capture_rng("cuda:0"),
            statistics=statistics,
            history=history,
            selection=selection,
            initial_validation=report.get("initial_validation"),
        )
        path = save_training_state(
            output / "checkpoints", state, identity=identity, pin=pin, best=best
        )
        report["checkpoint"] = dict(path=str(path.relative_to(output)), sha256=sha256(path))
        report["recovery_checkpoint"] = dict(report["checkpoint"])
        report["selection"] = selection
        last_saved = time.perf_counter()

    torch.cuda.reset_peak_memory_stats(0)
    try:
        save()
        if initialization is not None and selection_options and selection is None:
            baseline = _evaluate(
                model, dataset, batch_size, device=device, stop=stop, quantiles=quantiles
            )
            report["initial_validation"] = baseline
            selection = initial_selection(baseline["session_mae"], selection_options)
            save(best=True)
            atomic_json(report_path, report)
        while (next_step := decision()) != FINISH:
            if next_step == AWAIT:
                # El estado tras la validación ya está confirmado y la época común la fija el grupo.
                report.update(awaiting(selection, case["epochs"]), global_step=step, epochs=history)
                return report
            if stop.requested:
                raise _Pause
            model.train()
            iterator = dataset.batches(
                partition="train",
                batch_size=batch_size,
                epoch=epoch,
                seed=case["seed"],
                cursor=cursor,
            )
            segment = time.perf_counter()
            for batch in iterator:
                optimizer.zero_grad(set_to_none=True)
                target = torch.from_numpy(batch["target"]).to(device, dtype=torch.float32)
                emitted = _forward(model, batch, device)
                prediction = _point(emitted, target, quantiles)
                loss = row_loss(case, emitted, prediction, target, quantiles)
                if weighting != "natural":
                    loss = loss * torch.tensor([weights[m] for m in batch["market"]], device=device)
                loss = loss.mean()
                if not torch.isfinite(loss).item():
                    raise ValueError("La pérdida no es finita")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), math.inf, error_if_nonfinite=True
                )
                optimizer.step()
                _update(statistics, prediction, torch.from_numpy(batch["target"]).to(device))
                cursor, step = batch["confirmed_cursor"], step + 1
                if (
                    stop.requested
                    or time.perf_counter() - last_saved >= checkpoint_seconds
                    or (checkpoint_steps and step % checkpoint_steps == 0)
                ):
                    torch.cuda.synchronize(0)
                    statistics["elapsed_seconds"] += time.perf_counter() - segment
                    save()
                    segment = time.perf_counter()
                if stop.requested:
                    report.update(status="paused", global_step=step, epochs=history)
                    atomic_json(report_path, report)
                    return report
            torch.cuda.synchronize(0)
            statistics["elapsed_seconds"] += time.perf_counter() - segment
            if statistics["samples"] != dataset.manifest["counts"]["train"]:
                raise ValueError("La época no recorrió exactamente toda la población")
            validation = _evaluate(
                model, dataset, batch_size, device=device, stop=stop, quantiles=quantiles
            )
            history.append(dict(epoch=epoch + 1, train=_metrics(statistics), validation=validation))
            epoch, cursor, statistics = epoch + 1, None, _statistics()
            if selection_options:
                selection = advance_selection(
                    selection, validation["session_mae"], epoch, selection_options
                )
            save(pin=epoch == case["epochs"], best=bool(selection and selection["last_improved"]))
            report.update(global_step=step, epochs=history, status="running")
            atomic_json(report_path, report)
        report["stopped_early"] = epoch < case["epochs"]
        if selection_options:
            record = read_json(output / "checkpoints/latest.json")["best"]
            if record is None:
                raise ValueError("La selección no tiene una época confirmada")
            checkpoint = dict(path=f"checkpoints/{record['name']}", sha256=record["sha256"])
            selected = _confirmed_state(output, identity, checkpoint, selection)
            model.load_state_dict(selected["model"])
            report["checkpoint"] = checkpoint
            evaluating_selected = True
        legacy = prediction_retention == FULL_TRAIN_VALIDATION
        # Calibración y evaluación se predicen después de fijar el estado seleccionado.
        partitions = (
            ("train", "validation")
            if legacy
            else tuple(name for name in dataset.partitions if name != "train")
        )
        predictions = {}
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
            if not legacy:
                predictions[partition]["bytes"] = path.stat().st_size
        if not legacy:
            path = output / "train-sessions.parquet"
            metrics = _evaluate(
                model,
                dataset,
                batch_size,
                device=device,
                partition="train",
                sessions=path,
                stop=stop,
                quantiles=quantiles,
            )
            report["train_summary"] = dict(
                path=path.name, sha256=sha256(path), bytes=path.stat().st_size, metrics=metrics
            )
        report.update(
            status="completed",
            predictions=predictions,
            global_step=step,
            finished_at_utc=datetime.now(UTC).isoformat(),
        )
        if selection_options and {"minimum_epochs", "stopping"} & set(selection_options):
            joint = selection_options.get("stopping") == JOINT_PLATEAU
            report.update(
                stop_reason=(
                    "validation_plateau"
                    if selection["should_stop"]
                    else JOINT_PLATEAU
                    if joint and epoch < case["epochs"]
                    else "budget_exhausted"
                ),
                last_epoch_improved=selection["last_improved"],
            )
            if selection_options.get("stopping") in (FIXED_BUDGET, JOINT_PLATEAU):
                report["plateau_epoch"] = selection["plateau_epoch"]
        return report
    except _Pause:
        if not evaluating_selected:
            save()
        report.update(status="paused", global_step=step, epochs=history)
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["attempts"].append(
            dict(
                seconds=time.perf_counter() - start,
                peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(0),
                input_cache_bytes=dataset.cached_bytes,
            )
        )
        atomic_json(report_path, report)
