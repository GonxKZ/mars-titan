"""Control en línea de la referencia Transformer con las etiquetas maduras del banco.

`transformer_compact_online` parte del estado elegido de `transformer_compact` en la misma
ventana y semilla. Recorre calibración y evaluación en orden temporal con el índice de
observaciones de MARS-TITAN: en cada instante emite primero sus predicciones y después
aprende con las etiquetas que maduran en ese instante, las mismas que admite el banco
episódico y en el mismo momento. Cada tramo empieza desde el estado elegido, igual que el
banco empieza vacío en cada tramo. El control sirve para descartar que una mejora de
MARS-TITAN venga solo de seguir aprendiendo.

La regla se declara antes de ejecutar: optimizador, tasa de aprendizaje, etiquetas por
paso, instantes de maduración entre actualizaciones, recorte de gradiente y tope. El tope
cuenta etiquetas. En cada tramo no se usan más etiquetas que escrituras hizo el banco de
`mars_titan_m1` en el mismo ámbito, ventana y semilla. Las actualizaciones se hacen con el
modelo en modo de evaluación, sin dropout, para que el recorrido sea determinista.
"""

import itertools
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.models.quantile_head import median

from .carried_predictions import _destination, reference_model, selected_reference
from .learning_hold import require_learning_allowed
from .reference_run import _forward, _point, row_loss
from .titans_walk_forward import PredictionRows, _sources, checked_tables

KIND = "reference_online_predictions"
PARTITIONS = ("calibration", "evaluation")
CAP_RULE = "episodic_bank_writes"
OPTIMIZERS = ("sgd",)
RULE_FIELDS = (
    "optimizer",
    "learning_rate",
    "block_rows",
    "update_every",
    "max_grad_norm",
    "update_cap",
)
# El tope lo fija el banco de MARS-TITAN M1, que admite cada etiqueta madura de una
# predicción emitida.
BANK_KIND = "mars_titan_walk_forward_window"
BANK_COMPONENTS = {"episodic_bank": "m1"}
STRICT_FP32 = dict(
    float32_matmul_precision="highest", cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False
)


class Paused(Exception):
    """Señala una parada solicitada entre instantes. El intento no se reanuda después."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_rule(rule):
    """Comprueba que la regla se declaró completa antes de ejecutar, sin valores pendientes."""
    _require(
        isinstance(rule, dict)
        and set(rule) == set(RULE_FIELDS)
        and rule["optimizer"] in OPTIMIZERS
        and type(rule["learning_rate"]) is float
        and 0 < rule["learning_rate"] <= 1
        and type(rule["block_rows"]) is int
        and 1 <= rule["block_rows"] <= 4096
        and type(rule["update_every"]) is int
        and 1 <= rule["update_every"] <= 10_000
        and type(rule["max_grad_norm"]) in (int, float)
        and math.isfinite(rule["max_grad_norm"])
        and rule["max_grad_norm"] > 0
        and rule["update_cap"] == CAP_RULE,
        "La regla en línea declara optimizador sgd, tasa, etiquetas por paso, instantes entre "
        f"actualizaciones, recorte y el tope {CAP_RULE}, sin valores pendientes",
    )
    return dict(rule)


def strict_fp32():
    """Lee los indicadores de precisión del proceso y exige FP32 estricto sin TF32 ni autocast.

    PyTorch 2.14 tiene dos interfaces. Las antiguas dan los tres indicadores del informe y
    fallan si se mezclan con las nuevas, así que un fallo de lectura también se rechaza.
    Las nuevas (`fp32_precision`) no pueden pedir TF32 en matmul, convolución ni RNN.
    Una petición genérica de TF32 se rechaza aunque cada submódulo la anule, porque el
    control prefiere detenerse a depender de esa herencia.
    """
    backends = torch.backends
    try:
        numerics = dict(
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=backends.cudnn.allow_tf32,
        )
        modern = [
            getattr(backend, "fp32_precision", "none")
            for backend in (
                backends,
                backends.cuda.matmul,
                backends.cudnn,
                backends.cudnn.conv,
                backends.cudnn.rnn,
            )
        ]
    except RuntimeError:
        numerics, modern = None, []
    _require(
        numerics == STRICT_FP32
        and all(value in ("none", "ieee") for value in modern)
        and not any(torch.is_autocast_enabled(device) for device in ("cpu", "cuda")),
        "El control en línea exige FP32 estricto: precisión highest, sin TF32 ni autocast",
    )
    return numerics


def _sgd(parameters, rule):
    return torch.optim.SGD(parameters, lr=rule["learning_rate"], momentum=0.0, weight_decay=0.0)


def _gather(blocks, refs):
    """Monta el lote de actualización con las entradas guardadas de cada decisión madura."""
    first = blocks[refs[0][0]][0]
    batch = dict(
        inputs={
            name: np.stack([blocks[ref[0]][0]["inputs"][name][ref[1]] for ref in refs])
            for name in first["inputs"]
        }
    )
    if "presence" in first:
        batch["presence"] = np.stack([blocks[ref[0]][0]["presence"][ref[1]] for ref in refs])
    return batch


def online_pass(
    model,
    source,
    *,
    rule,
    cap,
    case,
    quantiles,
    optimizer,
    device,
    reader_rows=256,
    stop=None,
    audit=None,
):
    """Recorre un tramo, predice cada instante y aprende después con lo que madura.

    `source` entrega los eventos del índice de observaciones de MARS-TITAN. Se omiten los
    grupos del calentamiento, que solo tienen entradas. Una etiqueta solo se acepta si su
    predicción ya se emitió en un instante anterior. Devuelve las filas resueltas y las
    métricas. `audit` recibe la emisión y cada paso con sus decisiones.
    """
    _require(type(cap) is int and cap >= 0, "El tope de etiquetas debe ser un entero no negativo")
    phase = source.phase
    labelled = source.label_decisions()
    start = sum(1 for at, _ in source.metadata["groups"] if at < phase.decision_start)
    rows, errors = PredictionRows(quantiles), SessionErrors()
    counters = dict(
        predictions=0,
        labels=0,
        labels_used=0,
        labels_beyond_cap=0,
        labels_after_last_update=0,
        updates=0,
        maturity_instants=0,
        update_instants=0,
    )
    blocks, pending, ready, resolved = {}, {}, [], []
    identifiers = itertools.count()
    model.eval()

    def release(refs):
        for block, *_ in refs:
            blocks[block][1] -= 1
            if not blocks[block][1]:
                del blocks[block]

    def flush():
        for first in range(0, len(resolved), 4096):
            chunk = resolved[first : first + 4096]
            errors.update(*(list(values) for values in zip(*chunk, strict=True)))
        resolved.clear()

    def step(refs):
        target = torch.tensor([ref[2] for ref in refs], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        emitted = _forward(model, _gather(blocks, refs), device)
        prediction = _point(emitted, target, quantiles)
        loss = row_loss(case, emitted, prediction, target, quantiles).mean()
        _require(torch.isfinite(loss).item(), "La pérdida en línea no es finita")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), rule["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        counters["updates"] += 1

    for event in source.batched_events(start_cursor=start, block_rows=reader_rows):
        if stop is not None and stop.requested:
            raise Paused
        matured = []
        for key, decision_at, value in event.labels:
            entry = pending.pop((key, decision_at), None)
            # Las etiquetas de un instante se leen antes de sus predicciones, así que una
            # entrada pendiente siempre procede de un instante anterior.
            _require(
                entry is not None,
                "La etiqueta no tiene una predicción emitida antes de su maduración",
            )
            block, index, issued, levels = entry
            rows.append((key, decision_at, issued, value, levels))
            resolved.append((key.split("/", 1)[0], decision_at, issued - value))
            matured.append((block, index, value, key, decision_at))
            if audit is not None:
                audit.append(("label", event.at, key, decision_at))
        counters["labels"] += len(matured)
        if len(resolved) >= 4096:
            flush()
        for batch in event.inputs:
            moments = batch["prediction_at"].astype("datetime64[us]").astype(np.int64)
            _require(
                bool(np.all(moments == event.at)) and event.at >= phase.decision_start,
                "Las entradas no pertenecen al instante del evento ni al tramo",
            )
            with torch.no_grad():
                emitted = _forward(model, batch, device)
            points = (median(emitted) if quantiles else emitted).cpu().numpy().astype(np.float64)
            levels = emitted.cpu().numpy().astype(np.float64) if quantiles else None
            identifier, kept = next(identifiers), 0
            keys = [sample.rsplit("/", 1)[0] for sample in batch["sample_ids"]]
            for index, key in enumerate(keys):
                # Solo se guardan las entradas de las decisiones que maduran en el tramo.
                decisions = labelled.get(key)
                position = -1 if decisions is None else np.searchsorted(decisions, event.at)
                if position < 0 or position == len(decisions) or decisions[position] != event.at:
                    continue
                pending[(key, event.at)] = (
                    identifier,
                    index,
                    float(points[index]),
                    None if levels is None else levels[index].tolist(),
                )
                kept += 1
            if kept:
                blocks[identifier] = [batch, kept]
            counters["predictions"] += len(keys)
            if audit is not None:
                audit.append(("predict", event.at, tuple(keys)))
        if matured:
            ready += matured
            counters["maturity_instants"] += 1
            if counters["maturity_instants"] % rule["update_every"] == 0:
                counters["update_instants"] += 1
                allowed = max(cap - counters["labels_used"], 0)
                used, beyond = ready[:allowed], ready[allowed:]
                for first in range(0, len(used), rule["block_rows"]):
                    refs = used[first : first + rule["block_rows"]]
                    if audit is not None:
                        audit.append(("step", event.at, tuple(ref[3:] for ref in refs)))
                    step(refs)
                counters["labels_used"] += len(used)
                counters["labels_beyond_cap"] += len(beyond)
                release(ready)
                ready = []
        if event.close_phase:
            counters["labels_after_last_update"] += len(ready)
            release(ready)
            ready = []
    flush()
    _require(not pending and not blocks, "Una decisión con etiqueta no recibió su maduración")
    _require(counters["labels"], "El tramo no contiene etiquetas maduras")
    summary = errors.summary()
    metrics = dict(
        samples=summary["samples"],
        session_count=summary["session_count"],
        session_mae=summary["session_mae"],
        session_mse=summary["session_mse"],
        cap=cap,
        **counters,
    )
    return rows, metrics


def _bank_run(folder, view, seed):
    """Lee el informe confirmado de `mars_titan_m1` y exige la misma vista y semilla."""
    report, _ = read_manifest(Path(folder) / "run.json", 16 * 1024**2)
    identity = report.get("identity", {})
    _require(
        report.get("kind") == BANK_KIND
        and report.get("status") == "completed"
        and report.get("final_test_opened") is False
        and identity.get("variant", {}).get("components") == BANK_COMPONENTS
        and report["request"]["view_sha256"] == sha256(view)
        and report["request"]["seed"] == seed
        and all(name in report["predictions"] for name in PARTITIONS),
        "El tope necesita la ventana confirmada de mars_titan_m1 con la misma vista y semilla",
    )
    return report


def run_online_reference(
    anchor,
    view,
    bank,
    output,
    *,
    rule,
    input_policy,
    reader_rows=256,
    stop=None,
    optimizer_factory=None,
):
    """Ejecuta el control en línea de una ventana y escribe sus predicciones y su informe.

    `anchor` es la carpeta del estado elegido de `transformer_compact` y `bank` la de
    `mars_titan_m1`, ambas en la misma vista y semilla. La protección del aprendizaje y la
    precisión se comprueban antes de leer ninguna fuente. `optimizer_factory` solo existe
    para comprobar el recorrido con un optimizador que no modifica pesos.
    """
    from mars_titan.data.embeddings import require_cuda

    from .corpus_inputs import CorpusDataset

    require_learning_allowed("el control en línea transformer_compact_online")
    rule = checked_rule(rule)
    numerics = strict_fp32()
    _require(
        input_policy == HISTORICAL_MASKED,
        "El control en línea usa el índice de observaciones de la política con máscaras",
    )
    started = time.perf_counter()
    anchor, view, bank, output = Path(anchor), Path(view), Path(bank), Path(output)
    anchor_report, state = selected_reference(anchor, view, input_policy=input_policy)
    identity = anchor_report["identity"]
    case = identity["case"]
    _require(case["kind"] == "transformer", "El control en línea parte de transformer_compact")
    bank_report = _bank_run(bank, view, case["seed"])
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    _require(dataset.context == identity["context"], "La vista no conserva el contexto del ancla")
    output = _destination(output, dataset.roots.values())
    phases = {
        name: FinancialPhase(**bank_report["identity"]["phases"][name]) for name in PARTITIONS
    }
    sources = _sources(dataset, phases, output / "indices")
    for name, source in sources.items():
        _require(
            source.identity == bank_report["identity"]["indices"][name],
            f"El índice de {name} no es el del banco de mars_titan_m1",
        )
        _require(
            source.specification().dimensions == identity["dimensions"],
            "Las dimensiones de la ventana no coinciden con las del ancla",
        )
    device = require_cuda()
    predictions, caps = {}, {}
    for name in PARTITIONS:
        bank_metrics = bank_report["predictions"][name]["metrics"]
        caps[name] = bank_metrics["admitted"]
        # Cada tramo parte del estado elegido, como el banco parte vacío.
        model, quantiles = reference_model(identity, state, device)
        optimizer = (optimizer_factory or _sgd)(model.parameters(), rule)
        rows, metrics = online_pass(
            model,
            sources[name],
            rule=rule,
            cap=caps[name],
            case=case,
            quantiles=quantiles,
            optimizer=optimizer,
            device=device,
            reader_rows=reader_rows,
            stop=stop,
        )
        _require(
            metrics["labels"] == bank_metrics["labels"],
            f"El tramo {name} no recibe las mismas etiquetas maduras que el banco",
        )
        tables = checked_tables(rows, metrics, dataset, name)
        path = output / f"{name}-predictions.parquet"
        written = atomic_parquet_batches(path, tables)
        _require(written == rows.count, f"El Parquet de {name} no conserva sus filas")
        predictions[name] = dict(path=path.name, sha256=sha256(path), rows=written, metrics=metrics)
    report = dict(
        schema_version=1,
        kind=KIND,
        status="completed",
        final_test_opened=False,
        anchor=dict(
            run_sha256=sha256(anchor / "run.json"),
            checkpoint_sha256=anchor_report["checkpoint"]["sha256"],
            manifest_sha256=identity["manifest_sha256"],
            seed=case["seed"],
        ),
        bank=dict(
            run_sha256=sha256(bank / "run.json"), run_id=bank_report["run_id"], admitted=caps
        ),
        rule=rule,
        update_mode="eval_without_dropout",
        phases={name: asdict(phase) for name, phase in phases.items()},
        indices={name: source.identity for name, source in sources.items()},
        numerics=numerics,
        manifest_sha256=dataset.identity,
        reader_rows=reader_rows,
        predictions=predictions,
        seconds=time.perf_counter() - started,
        finished_at_utc=datetime.now(UTC).isoformat(),
        **policy_identity(input_policy),
    )
    atomic_json(output / "online.json", report)
    return report
