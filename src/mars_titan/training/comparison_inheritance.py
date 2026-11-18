"""Heredar trabajos compatibles en un estado nuevo sin sobrescribir sus resultados de origen."""

import copy
import fcntl
import importlib.metadata
import json
import os
import re
import shutil
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import atomic_json, sha256

from . import comparison_input_migration as packed
from .comparison_campaign import (
    DONE,
    MACHINES,
    PAUSED,
    RUNNING,
    checked_declaration,
    cohort_identity,
    plan,
    read_record,
)

_MODULE = re.compile(r"mars_titan(?:\.[A-Za-z_][A-Za-z_0-9]*)+")
GPU_RELEASE = packed.GPU_RELEASE
WARMUP_RELEASE = packed.WARMUP_RELEASE
GPU_DECLARATION = "585594e40886506fe2a4cf011d1e1cc350cc49a8c801d7c049eda5e8685c27d6"
READOUT_MODULE = "mars_titan.training.mars_titan_run"
PREVIOUS_READOUT = "fd4ac414e2798161035e1d6aabf785f436cc7e185895bfd341bad980f1ace218"
RESIDENT_READOUT = "94008c121f34cf897f389f5389a59e5dcce1073bc0988d44d2a38fea955e6457"
RESIDENT_WARMUP = "70997d22c1e0d228cd066fbfc69fc97f3bb476d26540ea6cc5507418291e10a7"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _document(path):
    _require(path.is_file() and path.stat().st_size <= 16 * 1024**2, "Falta un informe acotado")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_file(folder, item):
    name = item["path"]
    _require(isinstance(name, str), "La ruta del resultado no es válida")
    path = folder / name
    _require(
        not Path(name).is_absolute()
        and ".." not in Path(name).parts
        and path.is_file()
        and path.resolve().is_relative_to(folder.resolve()),
        "El resultado debe permanecer dentro de su trabajo",
    )
    _require(sha256(path) == item["sha256"], "Ha cambiado la huella de un resultado")


def _compatible_report(
    folder,
    report,
    job,
    model,
    declaration,
    root,
    cohort_sha,
    *,
    warmup_release=False,
    packed_release=False,
):
    _require(report.get("final_test_opened") is False, "El test final debe seguir cerrado")
    request = report.get("request", {})
    _require(request.get("view_sha256") == cohort_sha, "El resultado pertenece a otra cohorte")
    if job.seed is not None:
        _require(request.get("seed") == job.seed, "El resultado pertenece a otra semilla")
    if job.runner in ("fit", "reader"):
        recipe = Path(root) / declaration["recipes"][model["recipe"]]
        _require(
            request.get("recipe_sha256") == sha256(recipe), "La receta del resultado ha cambiado"
        )
        code = request.get("code")
        _require(isinstance(code, dict) and code, "Falta la identidad del código numérico")
        for module, digest in code.items():
            _require(_MODULE.fullmatch(module), "El módulo del código no es válido")
            path = Path(root) / "src" / (module.replace(".", "/") + ".py")
            if not path.is_file():
                path = Path(root) / "src" / module.replace(".", "/") / "__init__.py"
            current = sha256(path) if path.is_file() else None
            if current == digest and current is not None:
                continue
            if (
                packed_release
                and packed.equivalent(root, module, digest, current)
                and (
                    report.get("status") == DONE
                    or (job.runner == "fit" and report.get("status") == PAUSED)
                )
            ):
                continue
            # Un ajuste interrumpido conserva su identidad exacta. La excepción solo enlaza
            # predicciones ya emitidas por un lector terminado de la versión comprobada.
            helper = path.with_name("validation_warmup.py")
            _require(
                warmup_release
                and job.runner == "reader"
                and report.get("status") == DONE
                and report.get("kind") == "mars_titan_walk_forward_window"
                and module == READOUT_MODULE
                and digest == PREVIOUS_READOUT
                and current == RESIDENT_READOUT
                and helper.is_file()
                and sha256(helper) == RESIDENT_WARMUP,
                f"Ha cambiado el código de {module}",
            )
    for item in report.get("predictions", {}).values():
        _verify_file(folder, item)
    for field in ("fit", "checkpoint"):
        if isinstance(report.get(field), dict):
            _verify_file(folder, report[field])
    if isinstance((report.get("fit") or {}).get("best_checkpoint"), dict):
        _verify_file(folder, report["fit"]["best_checkpoint"])
    for item in report.get("fits", {}).values():
        if "run_sha256" in item:
            _verify_file(folder, dict(path=item["path"] + "/run.json", sha256=item["run_sha256"]))
        else:
            _verify_file(folder, item)


def _check_versions(record):
    for attempt in record.get("attempts", []):
        versions = (attempt.get("resources") or {}).get("versions") or {}
        for package in ("torch", "numpy", "pyarrow", "lightgbm"):
            if versions.get(package) is not None:
                _require(
                    versions[package] == importlib.metadata.version(package),
                    f"Ha cambiado la versión numérica de {package}",
                )


def _copy_completed_parent(source, destination):
    # El lector exige rutas regulares hasta el checkpoint. Una copia independiente
    # permite validar esos archivos y conserva el padre histórico en su directorio.
    shutil.copytree(source, destination, symlinks=True)
    for path in destination.rglob("*"):
        _require(not path.is_symlink(), "El padre contiene un enlace intermedio")
        if path.is_file():
            _require(
                sha256(path) == sha256(source / path.relative_to(destination)),
                "La copia del padre ha cambiado de contenido",
            )


def _hold_jobs(stack, source, jobs, *, finish_active):
    waiting = []
    for job in jobs:
        path = source.lock(job.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = stack.enter_context(path.open("a"))
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not finish_active:
                raise ValueError("La cola de origen sigue activa. Debe pausarse antes") from None
            waiting.append((job.key, handle))
    # Los trabajos disponibles ya están reservados. Cada proceso activo acaba con su versión.
    for key, handle in waiting:
        print(f"Esperando a que termine {key}", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)


def inherit(old, new, old_digest, new_digest, source, target, *, commit, root, finish_active=False):
    """Copiar los estados pausados y enlazar los terminados, con el origen cerrado para aprender.

    El destino conserva una referencia local a los resultados completados y copia los índices.
    El directorio de origen debe mantenerse. No se transportan modelos entre ordenadores.
    La autorización del destino se registra después con la orden habitual `authorize`.
    """
    checked_declaration(old, root=root)
    checked_declaration(new, root=root)
    _require(not target.state.exists(), "El destino ya existe")
    _require(
        not target.state.resolve().is_relative_to(source.state.resolve()),
        "El destino no puede estar dentro del origen",
    )
    for field in ("origins", "seeds", "cohort", "stopping", "analysis"):
        _require(
            old.get(field) == new.get(field), f"Ha cambiado el contrato o presupuesto: {field}"
        )
    _require(
        old["execution"].get("neural_blocks") == new["execution"].get("neural_blocks"),
        "Ha cambiado el reparto de los bloques neuronales",
    )
    permission = _document(source.authorization)
    permission_sha = sha256(source.authorization)
    _require(
        permission.get("declaration_sha256") == old_digest, "La declaración de origen no coincide"
    )
    _require(permission.get("code_changes_sha256") is None, "El origen necesita una versión fija")
    warmup_release = (
        old_digest == new_digest == GPU_DECLARATION
        and permission.get("code_commit") == GPU_RELEASE
        and commit == WARMUP_RELEASE
    )
    packed_release = (
        old_digest == new_digest == GPU_DECLARATION
        and permission.get("code_commit") in (GPU_RELEASE, WARMUP_RELEASE)
        and commit == packed.PACKED_RELEASE
    )
    if packed_release:
        _require(
            permission.get("status") != "continued_in_another_state",
            "El origen ya tiene una continuación registrada",
        )
    if commit == packed.PACKED_RELEASE and permission.get("code_commit") != commit:
        _require(
            packed_release, "El origen no corresponde a una versión comprobada para estas entradas"
        )
    jobs = {job.key: job for machine in MACHINES for job in plan(old, machine)}
    candidates = {job.key: job for machine in MACHINES for job in plan(new, machine)}
    _require(len(jobs) <= 4096, "La herencia supera el límite de trabajos")
    with ExitStack() as stack:
        _hold_jobs(stack, source, jobs.values(), finish_active=finish_active)
        selected, omitted = {}, {}
        for key, job in jobs.items():
            record = read_record(source, key)
            _require(record["status"] != RUNNING, "Hay un trabajo activo. Pausa antes la cola")
            if not record.get("attempts"):
                continue
            if packed_release:
                _require(
                    record["status"] in (DONE, PAUSED),
                    "Hay un intento sin resolver en el origen. Debe conservarse su resultado",
                )
            if key not in candidates or old["models"][job.model] != new["models"][job.model]:
                omitted[key] = "El modelo no forma parte del mismo contrato en el destino"
                continue
            if record["status"] not in (DONE, PAUSED):
                omitted[key] = (
                    "Solo se heredan resultados completos y puntos de recuperación pausados"
                )
                continue
            _check_versions(record)
            report = _document(source.output(key) / "run.json")
            _require(
                report.get("status") == record["status"], "El registro no coincide con su informe"
            )
            if packed_release and record["status"] == PAUSED:
                _require(
                    job.runner == "fit"
                    and _document(source.output(key) / "fit/run.json").get("global_step") == 0,
                    "Solo se traslada el paso cero. El ajuste iniciado termina con su versión",
                )
            _compatible_report(
                source.output(key),
                report,
                job,
                new["models"][job.model],
                new,
                root,
                sha256(source.cohort(job.year) / "manifest.json"),
                warmup_release=warmup_release,
                packed_release=packed_release,
            )
            selected[key] = record
        for key in selected:
            parent = candidates[key].needs
            if parent:
                _require(
                    parent in selected and selected[parent]["status"] == DONE,
                    "Un lector necesita heredar su padre terminado",
                )
        bundle = _document(source.bundle) if source.bundle.exists() else None
        if bundle is not None:
            _require(
                bundle.get("cohorts") == cohort_identity(old, source),
                "Las cohortes no coinciden con el paquete importado",
            )
        target.state.mkdir(parents=True, exist_ok=False)
        status_path = target.state / "inheritance-status.json"
        atomic_json(status_path, dict(status="running"))
        receipt = dict(
            kind="comparison_campaign_inheritance_v1",
            schema_version=1,
            source_state=str(source.state.resolve()),
            source_authorization_sha256=permission_sha,
            source_declaration_sha256=old_digest,
            declaration_sha256=new_digest,
            code_commit=commit,
            inherited={},
            omitted=omitted,
            created_at_utc=datetime.now(UTC).isoformat(),
            final_test_opened=False,
        )
        try:
            shutil.copytree(source.state / "cohorts", target.state / "cohorts")
            if bundle is not None:
                atomic_json(target.bundle, bundle)
            if source.indices.is_dir():
                shutil.copytree(source.indices, target.indices)
            parents = {job.needs for job in candidates.values() if job.needs}
            for key, record in selected.items():
                destination = target.output(key)
                destination.parent.mkdir(parents=True, exist_ok=True)
                regular_parent = (
                    (warmup_release or packed_release)
                    and key in parents
                    and record["status"] == DONE
                )
                initial = None
                if regular_parent:
                    _copy_completed_parent(source.output(key).resolve(strict=True), destination)
                elif record["status"] == DONE:
                    destination.symlink_to(source.output(key).resolve(), target_is_directory=True)
                elif packed_release:
                    initial = packed.transfer_initial(
                        source.output(key),
                        destination,
                        root,
                        host_flags=job.model == "titans_mac"
                        and os.environ.get("MARS_TITAN_CUDA_HOST_FINITE", "1") == "1",
                        archived=os.environ.get("MARS_TITAN_TRAIN_ARCHIVE", "default") != "",
                    )
                else:
                    shutil.copytree(source.output(key), destination)
                copied = copy.deepcopy(record)
                copied["source_execution"] = dict(
                    code_commit=permission["code_commit"],
                    declaration_sha256=old_digest,
                    source_state=str(source.state.resolve()),
                    source_record_sha256=sha256(source.record(key)),
                )
                if record.get("source_execution"):
                    copied["source_execution"]["previous"] = record["source_execution"]
                target.record(key).parent.mkdir(parents=True, exist_ok=True)
                atomic_json(target.record(key), copied)
                if source.log(key).is_file():
                    target.log(key).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source.log(key), target.log(key))
                receipt["inherited"][key] = dict(
                    status=record["status"], **copied["source_execution"]
                )
                if regular_parent:
                    receipt["inherited"][key]["storage"] = "verified_parent_copy"
                if initial:
                    receipt["inherited"][key]["initial_checkpoint"] = initial
            _require(
                sha256(source.authorization) == permission_sha, "El permiso de origen ha cambiado"
            )
            atomic_json(target.state / "inheritance.json", receipt)
            atomic_json(
                source.authorization,
                dict(
                    permission,
                    training_allowed=False,
                    status="continued_in_another_state",
                    continuation_state=str(target.state.resolve()),
                    closed_at_utc=datetime.now(UTC).isoformat(),
                ),
            )
            atomic_json(status_path, dict(status="completed"))
        except BaseException as error:
            atomic_json(
                status_path, dict(status="failed", error=f"{type(error).__name__}: {error}")
            )
            raise
    return receipt
