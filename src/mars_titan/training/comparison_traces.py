"""Política de observación de la cola y cursor confirmado en el checkpoint de ajuste."""

import math
import re

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json

POLICY = "learning-traces.json"
QUALIFICATION = "trace-qualification.json"
MAX_TRACE_BYTES = 64 * 1024**2
_DIGEST = re.compile(r"[a-f0-9]{64}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _checked_measurements(policy, evidence):
    measurements = evidence.get("measurements")
    threshold = evidence.get("max_relative_overhead")
    overrides = policy.get("every_by_model", {})
    _require(
        isinstance(measurements, dict)
        and set(measurements) == set(policy["models"])
        and isinstance(overrides, dict)
        and overrides.keys() <= measurements.keys()
        and all(type(value) is int and value > 0 for value in overrides.values())
        and type(threshold) in (int, float)
        and math.isfinite(threshold)
        and threshold >= 0,
        "Falta la medida CUDA por modelo o su umbral",
    )
    for name, item in measurements.items():
        expected = overrides.get(name, policy["every"])
        overhead = item.get("relative_overhead") if isinstance(item, dict) else None
        _require(
            isinstance(item, dict)
            and item.get("device") == "cuda:0"
            and item.get("exact") is True
            and type(item.get("every")) is int
            and item["every"] == expected
            and isinstance(item.get("report_sha256"), str)
            and _DIGEST.fullmatch(item["report_sha256"])
            and all(
                type(item.get(k)) is int and item[k] > 0
                for k in ("prediction_rows", "gradient_sets", "trace_rows", "timing_pairs")
            )
            and type(overhead) in (int, float)
            and math.isfinite(overhead)
            and -1 < overhead <= threshold,
            f"La medida CUDA de {name} no acredita la cadencia, la paridad o el coste",
        )


def read_policy(state, commit, declaration):
    path = state / POLICY
    if not path.exists() and not path.is_symlink():
        return None, None
    policy, digest = read_manifest(path, 64 * 1024)
    evidence, measured = read_manifest(state / QUALIFICATION, 2 * 1024**2)
    _require(
        isinstance(policy, dict)
        and type(policy.get("schema_version")) is int
        and policy["schema_version"] == 1
        and policy.get("runtime_commit") == commit
        and policy.get("declaration_sha256") == declaration
        and policy.get("qualification_sha256") == measured,
        "La política de trazas no corresponde a esta versión, declaración o comprobación",
    )
    models, every, limit = policy.get("models"), policy.get("every"), policy.get("max_bytes")
    _require(
        isinstance(models, list)
        and 0 < len(models) <= 128
        and all(isinstance(m, str) and re.fullmatch(r"[a-z0-9_]{1,80}", m) for m in models)
        and len(set(models)) == len(models)
        and type(every) is int
        and every > 0
        and type(limit) is int
        and 1024 <= limit <= MAX_TRACE_BYTES,
        "Los modelos, la cadencia o el presupuesto de trazas no son válidos",
    )
    _require(
        isinstance(evidence, dict)
        and evidence.get("passed") is True
        and all(evidence.get(k) == policy[k] for k in ("every", "max_bytes", "models")),
        "La cadencia, los modelos y el presupuesto deben coincidir con la medida aceptada",
    )
    _checked_measurements(policy, evidence)
    return policy, digest


def _verify_existing_manifest(folder, job, every, limit):
    path = folder / "traces/manifest.json"
    if not path.exists():
        _require(
            not any((folder / "traces/parts").glob("part-*.parquet")),
            "Hay partes sin manifiesto de trazas",
        )
        return
    manifest, _ = read_manifest(path, 64 * 1024)
    family = job.model if job.runner == "fit" else "mars_titan_readout"
    _require(
        isinstance(manifest, dict)
        and manifest.get("kind") == "learning_traces"
        and manifest.get("step_unit") == "optimizer_step"
        and manifest.get("config") == dict(every=every, max_bytes=limit, enabled=True)
        and isinstance(manifest.get("identity"), dict)
        and manifest["identity"].get("family") == family,
        "El manifiesto de trazas no corresponde al trabajo y su cadencia",
    )
    fit_path = folder / "fit/run.json"
    if fit_path.exists():
        from mars_titan.learning_traces.hooks import trace_identity

        fit, _ = read_manifest(fit_path)
        _require(
            manifest["identity"] == trace_identity(family, fit["identity"]),
            "El manifiesto de trazas corresponde a otro ajuste",
        )


def prepare(layout, job, models, record, permission):
    """Ligar el intento a la política y al paso real desde el que podrá observarse."""
    policy, digest = read_policy(
        layout.state, permission.get("code_commit"), permission.get("declaration_sha256")
    )
    _require(
        digest == permission.get("learning_trace_policy_sha256"),
        "La política de trazas difiere de la autorizada",
    )
    past = [a["learning_traces"] for a in record["attempts"] if "learning_traces" in a]
    folder = layout.output(job.key)
    if policy is not None:
        _require(
            all(m in models and models[m]["runner"] in ("fit", "reader") for m in policy["models"]),
            "La política de trazas contiene un modelo ajeno a los ajustes de esta campaña",
        )
    if policy is None or job.model not in policy["models"]:
        _require(not past and not (folder / "traces").exists(), "Hay trazas sin política activa")
        return None
    _require(
        all(a.get("policy_sha256") == digest for a in past)
        and (past or not (folder / "traces").exists()),
        "Las trazas existentes no pertenecen a esta política",
    )
    safe_destination(folder / "traces")
    every = policy.get("every_by_model", {}).get(job.model, policy["every"])
    if past:
        _verify_existing_manifest(folder, job, every, policy["max_bytes"])
    index = folder / "fit/checkpoints/latest.json"
    resume = 0
    if index.exists():
        document, _ = read_manifest(index, 512 * 1024)
        resume = document["latest"][0]["global_step"]
    start = past[0]["start_step"] if past else resume
    _require(type(resume) is int and type(start) is int and 0 <= start <= resume, "Cursor inválido")
    return dict(
        every=every,
        max_bytes=policy["max_bytes"],
        policy_sha256=digest,
        qualification_sha256=policy["qualification_sha256"],
        start_step=start,
        resume_step=resume,
    )


def confirm(folder, entry, *, paused):
    """Leer solo un checkpoint confirmado y publicar su cursor de trazas para la exportación."""
    from .checkpoints import load_training_state

    path = folder / "fit/run.json"
    if paused and not path.exists() and not (folder / "traces").exists():
        return False
    fit, _ = read_manifest(path)
    digest = fit["recovery_checkpoint"]["sha256"]
    _require(
        isinstance(digest, str)
        and _DIGEST.fullmatch(digest)
        and fit["recovery_checkpoint"]["path"] == f"checkpoints/state-{digest}.pt",
        "La referencia de recuperación de las trazas es incompatible",
    )
    state = load_training_state(
        folder / "fit/checkpoints", expected_identity=fit["identity"], expected_sha256=digest
    )
    cursor = state.get("traces")
    if cursor is None and state["global_step"] == fit["global_step"] == entry["start_step"]:
        # Un ajuste terminado puede conservar su checkpoint y tener aún emisiones pendientes.
        return False
    _require(
        state["global_step"] == fit["global_step"]
        and entry["start_step"] <= state["global_step"]
        and isinstance(cursor, dict)
        and set(cursor) == {"sequence", "bytes", "exhausted_at"}
        and type(cursor["sequence"]) is int
        and cursor["sequence"] >= 0
        and type(cursor["bytes"]) is int
        and 0 <= cursor["bytes"] <= entry["max_bytes"]
        and (
            cursor["exhausted_at"] is None
            or type(cursor["exhausted_at"]) is int
            and entry["start_step"] <= cursor["exhausted_at"] <= state["global_step"]
        ),
        "El checkpoint no confirma el cursor de trazas del ajuste",
    )
    atomic_json(
        folder / "trace-receipt.json",
        dict(
            schema_version=1,
            kind="comparison_learning_trace_checkpoint",
            fit_run_id=fit["run_id"],
            checkpoint_sha256=digest,
            start_step=entry["start_step"],
            global_step=state["global_step"],
            cursor=cursor,
            policy_sha256=entry["policy_sha256"],
            qualification_sha256=entry["qualification_sha256"],
        ),
    )
    return True
