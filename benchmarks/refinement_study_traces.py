"""Reconstrucción congelada, trazas maduras y recibos recuperables por fase."""

import fcntl
import math
import os
import platform
import resource
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.training.learning_hold import require_learning_allowed

MAX_ROWS = 131072
FINAL_TEST_US = 1704067200000000


class TraceSink:
    """Escribir por bloques y publicar solo después de comprobar la fase completa."""

    def __init__(self, path, *, partition, block_rows=1024):
        if partition not in {"train", "validation", "calibration", "evaluation"}:
            raise ValueError("La partición de las trazas no está admitida")
        if type(block_rows) is not int or not 1 <= block_rows <= 4096:
            raise ValueError("El bloque de escritura debe estar entre 1 y 4096 filas")
        self.path, self.partition, self.block_rows = Path(path), partition, block_rows
        safe_destination(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        os.close(fd)
        self.temporary = Path(name)
        self.buffer, self.seen, self.writer = [], set(), None
        self.count, self.confirmed = 0, False
        self.schema = pa.schema(
            [
                ("flow_id", pa.string()),
                ("decision_at", pa.int64()),
                ("input_available_at", pa.int64()),
                ("partition", pa.string()),
                ("snapshot_context", pa.string()),
                ("features", pa.list_(pa.float64(), 9)),
                ("predictions", pa.list_(pa.float64(), 6)),
                # Arrow no conserva listas fijas exteriores nulas al pasar por Parquet.
                # Las dimensiones de los valores presentes se comprueban al admitirlos.
                ("quantiles", pa.list_(pa.list_(pa.float64()))),
                ("maturity_at", pa.int64()),
                ("target", pa.float64()),
            ]
        )

    def append(self, record):
        if self.confirmed or self.count >= MAX_ROWS:
            raise ValueError("Las trazas están cerradas o superan el límite de filas")
        if set(record) != set(self.schema.names):
            raise ValueError("La traza no conserva los campos declarados")
        times = [record[name] for name in ("input_available_at", "decision_at", "maturity_at")]
        if (
            any(type(value) is not int for value in times)
            or not 0 <= times[0] <= times[1] < times[2] < FINAL_TEST_US
            or record["partition"] != self.partition
            or not isinstance(record["flow_id"], str)
            or not record["flow_id"].startswith("US/")
            or not isinstance(record["snapshot_context"], str)
            or not record["snapshot_context"]
        ):
            raise ValueError("La traza no conserva disponibilidad, maduración o identidad")
        features, predictions = np.asarray(record["features"]), np.asarray(record["predictions"])
        if features.shape != (9,) or predictions.shape != (6,):
            raise ValueError("Las trazas necesitan nueve rasgos y seis predicciones")
        arrays = [features, predictions, np.asarray([record["target"]])]
        if record["quantiles"] is not None:
            quantiles = np.asarray(record["quantiles"])
            if quantiles.shape != (6, 5) or np.any(np.diff(quantiles, axis=1) < 0):
                raise ValueError("Los cuantiles de las seis rutas no conservan su orden")
            arrays.append(quantiles)
        if any(a.dtype.kind not in "fi" or not np.isfinite(a).all() for a in arrays):
            raise ValueError("La traza contiene valores no numéricos, NaN o infinitos")
        key = record["flow_id"], record["decision_at"]
        if key in self.seen:
            raise ValueError("La traza de una decisión está repetida")
        self.seen.add(key)
        self.buffer.append(record)
        self.count += 1
        if len(self.buffer) >= self.block_rows:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.temporary, self.schema, compression="zstd")
        self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
        self.buffer.clear()

    def table(self):
        self.flush()
        if self.writer is None:
            raise ValueError("La fase no ha producido trazas maduras")
        self.writer.close()
        self.writer = None
        return pq.read_table(self.temporary)

    def confirm(self):
        if self.confirmed:
            raise ValueError("La fase ya está confirmada")
        if self.buffer or self.writer is not None:
            self.table()
        if not self.count:
            raise ValueError("No se puede confirmar una fase vacía")
        with self.temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(self.temporary, self.path)
        parent = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        self.confirmed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.writer is not None:
            self.writer.close()
        self.temporary.unlink(missing_ok=True)


def completed_phase(folder, identity):
    path = Path(folder) / "phase.json"
    if not path.exists():
        return None
    receipt = read_manifest(path)[0]
    if receipt.get("identity") != identity:
        raise ValueError("La identidad de la fase cambió después de iniciarla")
    if receipt.get("status") != "completed":
        return None
    for name, digest in receipt["artifacts"].items():
        artifact = Path(folder) / name
        safe_destination(artifact)
        if (
            not artifact.resolve().is_relative_to(Path(folder).resolve())
            or sha256(artifact) != digest
        ):
            raise ValueError("Un artefacto confirmado de la fase ha cambiado")
    return receipt


def expected_rows(dataset, phase):
    """Las mismas etiquetas aceptadas de la vista, restringidas al tramo de la fase."""
    result = dict(market=[], asset_id=[], prediction_at=[], target=[])
    for asset in dataset.assets:
        with pq.ParquetFile(dataset._file(asset, "samples")) as file:
            count = file.metadata.num_rows
        _, times, targets, maturity = dataset._labels(asset, phase.partition, count)
        keep = (
            (times >= phase.decision_start)
            & (times < phase.decision_end)
            & (maturity < phase.close_at)
        )
        size = int(keep.sum())
        result["market"].extend([asset["market"]] * size)
        result["asset_id"].extend([f"{asset['market']}/{asset['symbol']}"] * size)
        result["prediction_at"].extend(times[keep].tolist())
        result["target"].extend(targets[keep].tolist())
    if not 1 <= len(result["target"]) <= MAX_ROWS:
        raise ValueError("El tramo de trazas está vacío o supera el presupuesto de filas")
    result["prediction_at"] = np.asarray(result["prediction_at"], dtype=np.int64)
    result["target"] = np.asarray(result["target"], dtype=np.float64)
    return result


def validate_trace_rows(table, expected, phase):
    from mars_titan.training.candidate_walk_forward import row_differences

    decisions = table["decision_at"].to_numpy()
    maturity = table["maturity_at"].to_numpy()
    if (
        np.any(decisions < phase.decision_start)
        or np.any(decisions >= phase.decision_end)
        or np.any(maturity >= phase.close_at)
        or np.any(maturity <= decisions)
        or set(table["partition"].to_pylist()) != {phase.partition}
    ):
        raise ValueError("Una traza cruza las fronteras temporales de la fase")
    flows = table["flow_id"].to_pylist()
    observed = dict(
        market=[flow.split("/", 1)[0] for flow in flows],
        asset_id=flows,
        prediction_at=decisions,
        target=table["target"].to_numpy(),
    )
    differences = row_differences(observed, expected)
    if any(differences.values()):
        raise ValueError(
            f"Las trazas no conservan las filas y objetivos de la vista: {differences}"
        )
    return differences


def reconstruct(work, job, source, *, mode, trace=None, route=None, policy=None, device="cuda:0"):
    """Cargar el mejor padre y el mejor lector, sin ajustar pesos al trasladar el codec."""
    from mars_titan.posttraining.chronological_windows import _arm_parts, _best, _readout_models
    from mars_titan.training.mars_titan_walk_forward import _carried_readout, _native
    from mars_titan.training.refinement_inference import RefinementInference

    arm = work / "runs" / job["parent"]
    parent_view = work / "fold-017/view.json"
    report, digest, family, parent, parent_report = _arm_parts(arm, parent_view)
    if report["request"]["seed"] != job["seed"]:
        raise ValueError("El lector no conserva la semilla emparejada")
    frozen, variant, codec, readout, recipe = _readout_models(
        family,
        report,
        parent,
        parent_report,
        source.specification(),
        device,
        carried=True,
    )
    if variant.admission != "m1":
        raise ValueError("Las seis alternativas necesitan el mismo banco M1")
    _, state = _best(arm, report)
    _carried_readout(state["model"], readout)
    readout.eval().requires_grad_(False)
    engine = RefinementInference(
        frozen,
        readout,
        recipe,
        admission=variant.admission,
        retention=family.retention(recipe, variant),
        native=_native(variant.admission),
        codec=codec,
        world=family.world,
        fold="fold-018",
        mode=mode,
        trace=trace,
        route=route,
        policy=policy,
    )
    return engine, dict(
        reader_run_sha256=digest,
        reader_checkpoint=report["fit"]["best_checkpoint"],
        parent_run_sha256=sha256(parent / "run.json"),
        parent_checkpoint=parent_report["checkpoint"],
        parent_view_sha256=sha256(parent_view),
        inference=engine.identity(),
    )


def collect(work, job, *, stop=None, device="cuda:0"):
    """Recoger ajuste y validación de la puerta, con reinicio y calentamiento propios."""
    from mars_titan.training.corpus_inputs import CorpusDataset
    from mars_titan.training.titans_walk_forward import _sources

    require_learning_allowed("la colección de alternativas de refinamiento")
    staged = read_manifest(work / "staged.json")[0]
    dataset = CorpusDataset(work / "fold-018/view.json", input_policy=HISTORICAL_MASKED)
    phases = {name: FinancialPhase(**staged["phases"][name]) for name in ("train", "validation")}
    sources = _sources(dataset, phases, work / "indices/policy")
    result = dict(schema_version=1, status="completed", job=job, final_test_opened=False, phases={})
    for name in ("train", "validation"):
        output = work / "runs" / job["id"] / name
        output.mkdir(parents=True, exist_ok=True)
        with TraceSink(output / "traces.parquet", partition=name) as sink:
            engine, origin = reconstruct(
                work, job, sources[name], mode="collect", trace=sink.append, device=device
            )
            identity = dict(
                prepared_sha256=sha256(work / "prepared.json"),
                source=sources[name].identity,
                origin=origin,
                job=job,
                partition=name,
            )
            previous = completed_phase(output, identity)
            if previous is not None:
                result["phases"][name] = previous
                continue
            atomic_json(output / "phase.json", dict(status="running", identity=identity))
            started = time.perf_counter()
            metrics = engine.evaluate(sources[name], stop=stop)
            elapsed = time.perf_counter() - started
            table = sink.table()
            exact = validate_trace_rows(table, expected_rows(dataset, phases[name]), phases[name])
            if table.num_rows != metrics["labels"] or metrics["updates"] != 0:
                raise ValueError("Las trazas no concilian con la inferencia sin ajuste")
            sink.confirm()
            receipt = dict(
                status="completed",
                identity=identity,
                rows=table.num_rows,
                exact_rows=exact,
                metrics=metrics,
                process_seconds=elapsed,
                throughput_scored_rows_per_second=table.num_rows / elapsed,
                timing_scope="full_phase_including_input_warmup_collection_and_maturity",
                artifacts={"traces.parquet": sha256(sink.path)},
                final_test_opened=False,
                shared_parameters_unchanged=True,
                completed_at_utc=datetime.now(UTC).isoformat(),
            )
            atomic_json(output / "phase.json", receipt)
            result["phases"][name] = receipt
    atomic_json(work / "runs" / job["id"] / "run.json", result)
    return result


def reader_components(depth):
    if type(depth) is int and depth == 1:
        return dict(episodic_bank="m1")
    if type(depth) is int and depth == 4:
        return dict(episodic_bank="m1", refinements=4, refinement_episodes="first_read")
    raise ValueError("El estudio solo ajusta lectores K1 y K4")


def dispatch(work, job, budget):
    from mars_titan.training import mars_titan_walk_forward as mars
    from mars_titan.training import titans_walk_forward as titans

    folder, output = work / job["fold"], work / "runs" / job["id"]
    common = dict(
        seed=job["seed"],
        output=output,
        device="cuda:0",
        stop=budget,
        search_case="lr1e-3",
        indices=work / "indices" / job["fold"],
    )
    with titans.unfused_attention():
        if job["kind"] == "core":
            return titans.run_titans_window(
                folder / "view.json",
                folder / "protocol.json",
                job["fold"],
                work / "recipes" / f"{job['projection']}.json",
                variant="mac_online",
                **common,
            )
        if job["kind"] == "reader":
            return mars.run_mars_titan_window(
                folder / "view.json",
                work / "runs" / job["parent"],
                work / "recipes/reader.json",
                components=reader_components(job["depth"]),
                **common,
            )
        if job["kind"] == "collect":
            return collect(work, job, stop=budget)
        if job["kind"] in ("select", "evaluate"):
            from .refinement_study_policy import evaluate_selected, select_policy

            function = select_policy if job["kind"] == "select" else evaluate_selected
            return function(work, job, stop=budget)
    raise ValueError("El trabajo no pertenece a una fase ejecutable")


def remaining_seconds(work):
    from .refinement_study import configuration

    costs = [
        read_manifest(path)[0]["process_seconds"]
        for path in sorted((work / "measurements").glob("*.json"))
    ]
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in costs
    ):
        raise ValueError("El coste de un intento debe ser finito y no negativo")
    return max(0, configuration()["max_total_seconds"] - sum(costs))


def execute_job(work, job_id):
    """Una sola carga CUDA con límite global que cuenta también los intentos fallidos."""
    from .refinement_study import verify_prepared

    require_learning_allowed("el ejecutor del estudio de refinamientos")
    receipt = verify_prepared(work)
    job = next((value for value in receipt["jobs"] if value["id"] == job_id), None)
    if job is None:
        raise ValueError("El trabajo no pertenece al catálogo preparado")
    safe_destination(work / "execution.lock")
    with (work / "execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Ya hay una ejecución activa de este estudio") from error
        return _measured(work, job, receipt)


def _measured(work, job, receipt):
    import torch
    from torch.optim.optimizer import (
        register_optimizer_step_post_hook,
        register_optimizer_step_pre_hook,
    )

    from .memory_microstudy import Budget

    remaining = remaining_seconds(work)
    if remaining <= 0:
        raise RuntimeError("Se agotó el presupuesto local de seis horas")
    if not torch.cuda.is_available():
        raise RuntimeError("El estudio requiere cuda:0 y no admite fallback silencioso a CPU")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(0)
    config = dict(receipt["configuration"])
    config["max_job_seconds"] = min(remaining, config["max_job_seconds"])
    budget = Budget(config)
    before = register_optimizer_step_pre_hook(budget.before)
    after = register_optimizer_step_post_hook(budget.after)
    report = dict(
        job=job,
        status="failed",
        prepared_sha256=sha256(work / "prepared.json"),
        started_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
        runtime=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
            device="cuda:0",
            precision="fp32_strict",
            cpu_threads=2,
        ),
    )
    try:
        result = dispatch(work, job, budget)
        report["status"] = result["status"]
        return result
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        before.remove()
        after.remove()
        report.update(
            process_seconds=time.monotonic() - budget.started,
            optimizer_steps=budget.steps,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0),
        )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        atomic_json(work / "measurements" / f"{job['id'].replace('/', '_')}-{stamp}.json", report)
