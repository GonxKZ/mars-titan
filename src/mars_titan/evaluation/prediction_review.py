"""Recalcular errores de predicciones confirmadas sin cargar modelos ni el test final."""

import argparse
import hashlib
import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation import session_metrics, splits

METRICS = ("samples", "session_count", "session_mae", "session_mse", "mae", "mse")
PARTITIONS = ("validation", "train")
MAX_JOBS = 8192
ABS_TOL, REL_TOL = 1e-12, 1e-10


def _json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(32 * 1024**2 + 1)
    if len(raw) > 32 * 1024**2:
        raise ValueError("El JSON supera el presupuesto de lectura")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("El JSON debe ser un objeto")
    return value, hashlib.sha256(raw).hexdigest()


def _inside(root, relative):
    part = Path(relative)
    result = (root / part).resolve()
    if part.is_absolute() or not result.is_relative_to(root.resolve()):
        raise ValueError("La ruta sale del directorio de la campaña")
    return result


def _signature(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _sources(sources):
    if not isinstance(sources, list) or not 1 <= len(sources) <= 32:
        raise ValueError("Se necesitan entre 1 y 32 fuentes explícitas")
    seen = set()
    for source in sources:
        name = source["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", name) or name in seen:
            raise ValueError("Los identificadores de fuente deben ser válidos y únicos")
        seen.add(name)
        if not isinstance(source["summary"], str):
            raise ValueError("La fuente necesita una ruta de resumen")
        if "manifest" in source and (
            not isinstance(source["manifest"], str) or not source["manifest"]
        ):
            raise ValueError("El manifiesto temporal necesita una ruta explícita")


def _temporal_contract(source, summary):
    if "manifest" not in source:
        return None
    path = Path(source["manifest"])
    if path.is_symlink():
        raise ValueError("El manifiesto temporal no puede ser un enlace simbólico")
    manifest, digest = _json(path)
    if (
        not isinstance(summary.get("identity"), dict)
        or summary["identity"].get("manifest_sha256") != digest
        or manifest.get("kind") != "corpus_supervision"
        or manifest.get("cohort_complete") is not True
        or manifest.get("final_test_opened") is not False
    ):
        raise ValueError("El resumen y el manifiesto temporal no conservan su contrato y huella")
    view = manifest["temporal_view"]
    protocol, fold = view["protocol"], view["fold"]
    if view["schema_version"] != 1 or fold not in splits.build_folds(protocol):
        raise ValueError("La ventana temporal no pertenece al protocolo confirmado")
    # Las ventanas no pueden ampliar la reserva final vigente del proyecto.
    if protocol["final_test_start"] > "2024-01-01":
        raise ValueError("El protocolo temporal invade el test reservado")
    return dict(
        manifest_sha256=digest, fold=fold["id"], windows={p: fold[p] for p in PARTITIONS}, view=view
    )


def _run_manifest(folder, run, report, temporal, views):
    if not isinstance(report.get("identity"), dict):
        raise ValueError("La ejecución no contiene una identidad temporal válida")
    expected = report["identity"].get("manifest_sha256")
    if expected == temporal["manifest_sha256"]:
        return expected
    arm = run.get("arm")
    if arm not in {"US", "CN", "US+CN"}:
        raise ValueError("La ejecución no conserva el manifiesto temporal del resumen")
    if arm not in views:
        path = folder / "views" / f"{arm}.json"
        if path.is_symlink():
            raise ValueError("La vista del mercado no puede ser un enlace simbólico")
        view, digest = _json(_inside(folder, path.relative_to(folder)))
        if (
            view.get("source_manifest_sha256") != temporal["manifest_sha256"]
            or view.get("temporal_view") != temporal["view"]
            or view.get("selected_arm") != arm
            or view.get("kind") != "corpus_supervision"
            or view.get("final_test_opened") is not False
            or view.get("cohort_complete") is not True
        ):
            raise ValueError("La vista del mercado no conserva el origen y contrato temporal")
        views[arm] = digest
    if expected != views[arm]:
        raise ValueError("La ejecución no conserva la huella de su vista temporal")
    return expected


def _confirmed(source, partitions):
    summary_path = Path(source["summary"])
    summary, _ = _json(summary_path)
    if summary.get("final_test_opened") is not False:
        raise ValueError("La fuente no acredita que el test permanezca cerrado")
    temporal = _temporal_contract(source, summary)
    runs = summary["runs"]
    if not isinstance(runs, list) or len(runs) > MAX_JOBS:
        raise ValueError("El resumen supera el presupuesto de ejecuciones")
    seen, views = set(), {}
    for run in runs:
        run_id = run["id"]
        if not isinstance(run_id, str) or len(run_id) > 256 or run_id in seen:
            raise ValueError("El resumen contiene identificadores inválidos o duplicados")
        seen.add(run_id)
        if run["status"] != "completed":
            continue
        relative = run.get("path")
        if relative is None:
            relative = run["attempts"][-1]["path"]
        folder = _inside(summary_path.parent, relative)
        report, digest = _json(folder / "run.json")
        if digest != run["report_sha256"]:
            raise ValueError(f"La huella del informe confirmado no coincide: {run_id}")
        manifest = (
            _run_manifest(summary_path.parent, run, report, temporal, views) if temporal else None
        )
        if (
            report.get("status") != "completed"
            or report.get("final_test_opened") is not False
            or report.get("cohort_complete") is not True
        ):
            raise ValueError(f"El informe no acredita una evaluación completa: {run_id}")
        for partition in partitions:
            artifact = report["predictions"][partition]
            if artifact["path"] != f"{partition}-predictions.parquet":
                raise ValueError("La partición no apunta a su archivo de predicciones")
            if (folder / artifact["path"]).is_symlink():
                raise ValueError("El archivo de predicciones no puede ser un enlace simbólico")
            metrics = artifact["metrics"]
            count = metrics.get("median", metrics)["samples"]
            for population in (summary.get("counts", {}), report.get("samples", {})):
                if partition in population and population[partition] != count:
                    raise ValueError("Las predicciones no concilian la población declarada")
            path = _inside(folder, artifact["path"])
            yield dict(
                key=f"{source['id']}/{run_id}/{partition}",
                path=path,
                report_sha256=digest,
                artifact=artifact,
                partition=partition,
                temporal_contract=(
                    dict(
                        manifest_sha256=temporal["manifest_sha256"],
                        run_manifest_sha256=manifest,
                        fold=temporal["fold"],
                        bounds=temporal["windows"][partition],
                    )
                    if temporal
                    else None
                ),
            )


def _metrics(path, partition, declared, temporal_contract=None):
    nested = "median" in declared
    columns = {"median" if nested else "prediction": "prediction"}
    if nested:
        columns.update(
            {key: key for key in ("center", "nearest", "parent", "zero") if key in declared}
        )
    columns.setdefault("zero", "zero")
    accumulators = {key: session_metrics.SessionErrors() for key in columns}
    minimum, maximum = None, None
    bounds = (
        temporal_contract["bounds"]
        if temporal_contract
        else ([None, "2023-01-01"] if partition == "train" else ["2023-01-01", "2024-01-01"])
    )
    start = np.datetime64(bounds[0], "us") if bounds[0] is not None else None
    end = np.datetime64(bounds[1], "us")
    with pq.ParquetFile(path, pre_buffer=False) as parquet:
        names = parquet.schema_arrow.names
        required = ["market", "prediction_at", "target", *columns.values()]
        if any(name not in names for name in required):
            raise ValueError("Faltan columnas de predicción o de sus controles")
        stamp = parquet.schema_arrow.field("prediction_at").type
        if stamp != pa.timestamp("us", tz="UTC"):
            raise ValueError("Las fechas deben expresarse en microsegundos UTC")
        for group in range(parquet.num_row_groups):
            if parquet.metadata.row_group(group).total_byte_size > 512 * 1024**2:
                raise ValueError("El bloque Parquet supera el presupuesto descomprimido")
            # Un iterador por bloque evita retener buffers de los bloques anteriores.
            for batch in parquet.iter_batches(
                batch_size=4096, row_groups=[group], columns=required, use_threads=False
            ):
                if any(column.null_count for column in batch.columns):
                    raise ValueError("Las predicciones contienen valores ausentes")
                times = batch.column("prediction_at").to_numpy(zero_copy_only=False)
                lower, upper = times.min(), times.max()
                if np.isnat(times).any() or upper >= end or (start is not None and lower < start):
                    raise ValueError("Las predicciones incumplen la partición temporal")
                minimum = lower if minimum is None else min(minimum, lower)
                maximum = upper if maximum is None else max(maximum, upper)
                markets = batch.column("market").to_numpy(zero_copy_only=False)
                target = batch.column("target").to_numpy().astype(np.float64)
                for key, column in columns.items():
                    prediction = batch.column(column).to_numpy().astype(np.float64)
                    if key == "zero" and np.any(prediction != 0):
                        raise ValueError("El control cero contiene predicciones no nulas")
                    accumulators[key].update(markets, times, prediction - target)
    result = {}
    for key, accumulator in accumulators.items():
        values = accumulator.summary()
        if not values["samples"]:
            raise ValueError("El archivo de predicciones está vacío")
        values.update(
            mae=values["absolute_error"] / values["samples"],
            mse=values["squared_error"] / values["samples"],
        )
        result[key] = {name: values[name] for name in METRICS}
        expected = declared.get(key, {}) if nested else (declared if key == "prediction" else {})
        for name in METRICS:
            if name not in expected:
                if key != "zero" and name in METRICS[:4]:
                    raise ValueError(f"Falta una métrica declarada: {key}.{name}")
                continue
            observed = values[name]
            valid = (
                observed == expected[name]
                if name in METRICS[:2]
                else math.isclose(observed, expected[name], rel_tol=REL_TOL, abs_tol=ABS_TOL)
            )
            if not valid:
                raise ValueError(
                    f"Métrica discrepante: {key}.{name} ({observed}, {expected[name]})"
                )
    primary = result["median" if nested else "prediction"]["session_mae"]
    return dict(
        metrics=result,
        delta_session_mae={
            key: primary - result[key]["session_mae"] for key in ("parent", "zero") if key in result
        },
        first_prediction_at=str(minimum),
        last_prediction_at=str(maximum),
    )


def _evaluate(job, max_file_bytes):
    path = job["path"]
    if not 0 < path.stat().st_size <= max_file_bytes:
        raise ValueError("El archivo supera el presupuesto de bytes o está vacío")
    digest = sha256(path)
    if digest != job["artifact"]["sha256"]:
        raise ValueError("La huella de las predicciones no coincide con el informe")
    result = _metrics(path, job["partition"], job["artifact"]["metrics"], job["temporal_contract"])
    if job["temporal_contract"]:
        result["temporal_contract"] = job["temporal_contract"]
    return result


def review_campaigns(
    sources, state_path, *, partitions=PARTITIONS, max_jobs=8, max_file_bytes=512 * 1024**2
):
    """Revisar solo novedades y confirmar una instantánea acotada al terminar el ciclo."""
    if (
        not partitions
        or len(set(partitions)) != len(partitions)
        or set(partitions) - set(PARTITIONS)
    ):
        raise ValueError("Solo se admiten las particiones train y validation")
    _sources(sources)
    if type(max_jobs) is not int or not 1 <= max_jobs <= 1024 or not 1 <= max_file_bytes <= 2**31:
        raise ValueError("El presupuesto de trabajo debe ser positivo y acotado")
    start = time.perf_counter()
    state_path = Path(state_path)
    for source in sources:
        outside_source(Path(source["summary"]).parent, state_path)
        if "manifest" in source:
            outside_source(Path(source["manifest"]).parent, state_path)
    previous = _json(state_path)[0] if state_path.exists() else {}
    version = [sha256(Path(module.__file__)) for module in (session_metrics, splits)]
    version.insert(0, sha256(Path(__file__)))
    entries = previous.get("entries", {}) if previous.get("reviewer_sha256") == version else {}
    state = dict(
        schema_version=1,
        reviewer_sha256=version,
        entries=entries,
        checked_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
        processed_jobs=0,
        hashed_bytes=0,
        source_errors={},
        unavailable_sources=[],
    )
    jobs = []
    for source in sources:
        try:
            if not Path(source["summary"]).exists():
                state["unavailable_sources"].append(source["id"])
                continue
            current = list(_confirmed(source, partitions))
            if len(jobs) + len(current) > MAX_JOBS:
                raise ValueError("Las fuentes superan el presupuesto total de evaluaciones")
            jobs.extend(current)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            state["source_errors"][source["id"]] = str(error)[:1000]
    counts = dict(confirmed=len(jobs), verified=0, failed=0, pending=0)
    for job in sorted(jobs, key=lambda item: partitions.index(item["partition"])):
        key, path = job["key"], job["path"]
        try:
            signature = _signature(path)
        except OSError:
            signature = None
        fingerprint = [
            job["report_sha256"],
            signature,
            job["artifact"]["sha256"],
            max_file_bytes,
            job["temporal_contract"],
        ]
        cached = entries.get(key)
        if cached is None or cached.get("fingerprint") != fingerprint:
            if state["processed_jobs"] >= max_jobs:
                counts["pending"] += 1
                continue
            state["processed_jobs"] += 1
            began = time.perf_counter()
            cached = dict(fingerprint=fingerprint, checked_at_utc=datetime.now(UTC).isoformat())
            try:
                if signature and 0 < signature[2] <= max_file_bytes:
                    state["hashed_bytes"] += signature[2]
                cached.update(_evaluate(job, max_file_bytes))
                if _signature(path) != signature:
                    raise ValueError("Las predicciones cambiaron durante la revisión")
                cached["status"] = "verified"
            except (OSError, ValueError, TypeError, KeyError, pa.ArrowException) as error:
                cached = dict(
                    fingerprint=fingerprint,
                    status="failed",
                    error=str(error)[:1000],
                    checked_at_utc=datetime.now(UTC).isoformat(),
                )
            cached["elapsed_seconds"] = time.perf_counter() - began
            entries[key] = cached
        counts["verified" if cached["status"] == "verified" else "failed"] += 1
    # Los resultados antiguos permanecen como evidencia, pero no cuentan como vigentes.
    if len(entries) > MAX_JOBS:
        raise ValueError("El historial supera el presupuesto de evaluaciones")
    state.update(
        counts=counts,
        elapsed_seconds=time.perf_counter() - start,
        tolerances=dict(absolute=ABS_TOL, relative=REL_TOL),
    )
    atomic_json(state_path, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument("--partition", choices=PARTITIONS, action="append")
    args = parser.parse_args(argv)
    config, _ = _json(args.config)
    state = review_campaigns(
        config["sources"],
        args.state,
        max_jobs=args.max_jobs,
        partitions=tuple(args.partition or PARTITIONS),
    )
    print(
        json.dumps(
            {key: value for key, value in state.items() if key != "entries"}, ensure_ascii=False
        )
    )
    return 1 if state["source_errors"] or state["counts"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
