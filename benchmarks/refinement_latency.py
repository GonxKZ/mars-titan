"""Medir el recorrido congelado y separar latencia de decisión y coste amortizado."""

import hashlib
import resource
import time
from contextlib import contextmanager

import numpy as np
import pyarrow as pa
import torch

from mars_titan import nvtx_ranges
from mars_titan.training.refinement_inference import _model_digest
from mars_titan.training.titans_walk_forward import PredictionRows


def _synchronize(device):
    if str(device) == "cuda:0":
        torch.cuda.synchronize(device)
    elif str(device) != "cpu":
        raise ValueError("La medida requiere cpu o cuda:0 explícitos")


@contextmanager
def _instrument(engine, records, maximum):
    saved, stack = {}, []
    missing = object()

    def wrap(method, stage, counter):
        def measured(*args, **kwargs):
            if len(records) + len(stack) >= maximum:
                raise ValueError("La medida supera el límite de registros")
            run, phase, event = args[:3]
            at = event if stage == "admit" else event.at
            count = run.counters.get(counter, 0)
            frame = dict(children=0.0)
            stack.append(frame)
            started = time.perf_counter()
            try:
                _synchronize(engine.device)
                with nvtx_ranges.phase(f"latency.{stage}"):
                    result = method(*args, **kwargs)
                _synchronize(engine.device)
            finally:
                elapsed = time.perf_counter() - started
                stack.pop()
                if stack:
                    stack[-1]["children"] += elapsed
            records.append(
                dict(
                    stage=stage,
                    at=int(at),
                    rows=run.counters.get(counter, 0) - count,
                    historical_warmup=at < phase.decision_start,
                    seconds=elapsed,
                    exclusive_seconds=elapsed - frame["children"],
                    depth=len(stack),
                )
            )
            return result

        return measured

    try:
        for name, counter in (
            ("observe", "observations"),
            ("labels", "labels"),
            ("admit", "admitted"),
        ):
            attribute = "_" + name
            saved[attribute] = engine.__dict__.get(attribute, missing)
            setattr(engine, attribute, wrap(getattr(engine, attribute), name, counter))
        yield
    finally:
        for attribute, original in saved.items():
            if original is missing:
                engine.__dict__.pop(attribute, None)
            else:
                setattr(engine, attribute, original)


def measure_pass(engine, source, *, instrument, max_records=32768, stop=None):
    """Una fase completa. Los estados temporales se reinician dentro del ejecutor."""
    if type(instrument) is not bool:
        raise ValueError("La instrumentación necesita un booleano explícito")
    if type(max_records) is not int or not 1 <= max_records <= 131072:
        raise ValueError("El límite de registros debe ser un entero entre 1 y 131072")
    _synchronize(engine.device)
    models = (engine.predictor, engine.readout, engine.policy)
    if any(
        model is not None
        and (
            any(module.training for module in model.modules())
            or any(p.requires_grad or p.grad is not None for p in model.parameters())
        )
        for model in models
    ):
        raise ValueError("La medida requiere parámetros congelados y sin gradientes")
    audit_start = time.perf_counter()
    before = tuple(_model_digest(model) for model in models)
    audit_seconds = time.perf_counter() - audit_start
    cpu_rng = torch.random.get_rng_state()
    cuda = str(engine.device) == "cuda:0"
    cuda_rng = torch.cuda.get_rng_state(engine.device) if cuda else None
    if cuda:
        torch.cuda.reset_peak_memory_stats(engine.device)
    rows, records = PredictionRows(engine.quantiles), []
    started = time.perf_counter()
    if instrument:
        with _instrument(engine, records, max_records):
            metrics = engine.evaluate(source, stop=stop, rows=rows)
    else:
        metrics = engine.evaluate(source, stop=stop, rows=rows)
    _synchronize(engine.device)
    elapsed = time.perf_counter() - started
    audit_start = time.perf_counter()
    if before != tuple(_model_digest(model) for model in models) or metrics["updates"] != 0:
        raise ValueError("La medida cambió los parámetros o ejecutó actualizaciones")
    if not torch.equal(cpu_rng, torch.random.get_rng_state()) or (
        cuda and not torch.equal(cuda_rng, torch.cuda.get_rng_state(engine.device))
    ):
        raise ValueError("La medida cambió el estado RNG compartido")
    audit_seconds += time.perf_counter() - audit_start
    result = dict(
        metrics=metrics,
        instrumented=instrument,
        device=str(engine.device),
        phase_seconds=elapsed,
        audit_seconds=audit_seconds,
        shared_parameters_unchanged=True,
        shared_rng_unchanged=True,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.device) if cuda else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(engine.device) if cuda else None,
        events=records,
        timing_scope="phase_including_input_warmup_prediction_and_maturity",
        cuda_timing="synchronized_method_boundaries" if cuda and instrument else None,
        memory_scope="process_lifetime_rss_and_phase_cuda_allocator_including_resident_models",
    )
    return result, rows


def prediction_digest(rows):
    """Comparar identidad, orden, valores y bits tras unificar los bloques Arrow."""
    return table_digest(pa.concat_tables(rows.finish()))


def table_digest(table):
    """La fragmentación de Arrow no cambia la huella de los mismos valores y tipos."""
    table = table.combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def _quantiles(records):
    times = np.asarray([row["seconds"] for row in records], dtype=np.float64)
    if len(times) and (not np.isfinite(times).all() or np.any(times < 0)):
        raise ValueError("Las latencias deben ser finitas y no negativas")
    return dict(
        count=len(records),
        rows=sum(row["rows"] for row in records),
        **dict(
            zip(
                ("p50_seconds", "p95_seconds", "p99_seconds"),
                np.quantile(times, [0.5, 0.95, 0.99]).tolist() if len(times) else [None] * 3,
                strict=True,
            )
        ),
        total_exclusive_seconds=sum(row["exclusive_seconds"] for row in records),
    )


def summarize(records, *, warmup_decisions=5):
    """Los percentiles son por evento. El cociente por filas es solo coste amortizado."""
    if type(warmup_decisions) is not int or warmup_decisions < 0:
        raise ValueError("El calentamiento técnico debe ser un número entero no negativo")
    _quantiles(records)
    observed = [row for row in records if row["stage"] == "observe"]
    historical = [row for row in observed if row["historical_warmup"]]
    eligible = [row for row in observed if not row["historical_warmup"]]
    steady = eligible[warmup_decisions:]
    amortized = [dict(row, seconds=row["seconds"] / row["rows"]) for row in steady if row["rows"]]
    amortized_summary = _quantiles(amortized)
    amortized_summary.pop("total_exclusive_seconds")
    return dict(
        historical_warmup_decisions=len(historical),
        technical_warmup_decisions=min(warmup_decisions, len(eligible)),
        steady_decisions=_quantiles(steady),
        amortized_seconds_per_asset=amortized_summary,
        labels=_quantiles([row for row in records if row["stage"] == "labels"]),
        admission=_quantiles([row for row in records if row["stage"] == "admit"]),
        additive_exclusive_seconds=sum(row["exclusive_seconds"] for row in records),
    )
