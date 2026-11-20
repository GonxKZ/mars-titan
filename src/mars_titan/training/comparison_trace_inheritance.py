"""Verificar los resultados y el checkpoint que PC1 conserva al activar las trazas."""

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, sha256

from . import comparison_input_migration as packed

BATCHES_MODULE = "mars_titan.data.batches"
TRACE_RELEASE = "d5f27f50073d12706c08130a155ea958d822b28b"
PACKED_BATCHES_SHA256 = "ef261c2e67e7296008c4a49632c63befd845ec7514c00ac8c3c39bc1c63c1b13"
QUALIFIED_WRITER_SHA256 = "d8edc73125159da9e968b9c32364ccfa7c2eae6a71f56e5b6686677f9135a28a"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return read_manifest(path, 16 * 1024**2)[0]


def verify_tree(source, target):
    """Comparar los nombres y todos los bytes de una copia independiente de directorio."""
    _require(target.is_dir() and not target.is_symlink(), "La copia debe ser un directorio regular")
    original = {p.relative_to(source) for p in source.rglob("*")}
    copied = {p.relative_to(target) for p in target.rglob("*")}
    _require(original == copied, "La copia no contiene los mismos archivos que el origen")
    for name in original:
        before, after = source / name, target / name
        _require(
            not before.is_symlink() and not after.is_symlink(),
            "La copia contiene un enlace intermedio",
        )
        if before.is_dir():
            _require(after.is_dir(), "Ha cambiado un directorio de la copia")
        else:
            _require(
                before.is_file()
                and after.is_file()
                and not before.samefile(after)
                and sha256(before) == sha256(after),
                "La copia ha cambiado de contenido o comparte un archivo con el origen",
            )


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def staging(source, target, commit):
    """Recuperar la carpeta privada de trabajo tras una caída anterior a la publicación."""
    parent = target.state.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging_root = parent / f".{target.state.name}.inheritance"
    owner = dict(
        source=str(source.state.resolve()), target=str(target.state.resolve()), commit=commit
    )
    if not staging_root.exists():
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            reserved = Path(tmp) / "reserved"
            reserved.mkdir(mode=0o700)
            atomic_json(reserved / "owner.json", owner)
            reserved.rename(staging_root)
            _sync(parent)
    _require(
        staging_root.is_dir()
        and not staging_root.is_symlink()
        and staging_root.stat().st_uid == os.getuid()
        and not staging_root.stat().st_mode & 0o077
        and _read(staging_root / "owner.json") == owner
        and {path.name for path in staging_root.iterdir()} <= {"owner.json", "state"},
        "La carpeta de preparación ya existe y pertenece a otro traslado",
    )
    work = staging_root / "state"
    if work.exists() or work.is_symlink():
        _require(work.is_dir() and not work.is_symlink(), "La copia incompleta no es un directorio")
        shutil.rmtree(work)
    try:
        yield work
    finally:
        shutil.rmtree(staging_root)


def publish(stage, target):
    for directory, _, files in os.walk(stage, topdown=False, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                _sync(path)
        _sync(directory)
    stage.rename(target)
    _sync(target.parent)


def recheck_copy(source, target, jobs, receipt):
    entries = receipt["inherited"]
    keys = {
        key
        for key in jobs
        if source.record(key).is_file() and _read(source.record(key)).get("attempts")
    }
    _require(
        set(entries) == keys and not receipt["omitted"],
        "Cambió la cola después de preparar su copia",
    )
    verify_tree(source.state / "cohorts", target.state / "cohorts")
    if source.indices.is_dir():
        verify_tree(source.indices, target.indices)
    if source.bundle.exists():
        _require(
            target.bundle.is_file() and sha256(source.bundle) == sha256(target.bundle),
            "Cambió el paquete de datos de la copia",
        )
    for key, entry in entries.items():
        before, after = source.output(key), target.output(key)
        _require(
            sha256(source.record(key)) == entry["source_record_sha256"],
            "Cambió el registro de origen",
        )
        previous = _read(source.record(key))
        copied = _read(target.record(key))
        origin = copied.pop("source_execution", {})
        fields = ("code_commit", "declaration_sha256", "source_state", "source_record_sha256")
        expected = {name: entry[name] for name in fields}
        if previous.get("source_execution"):
            expected["previous"] = previous.pop("source_execution")
        _require(copied == previous and origin == expected, "La copia del registro ya no coincide")
        if source.log(key).is_file():
            _require(
                target.log(key).is_file() and sha256(source.log(key)) == sha256(target.log(key)),
                "Cambió el registro de salida copiado",
            )
        if entry["status"] == "paused" or entry.get("storage") == "verified_parent_copy":
            verify_tree(before.resolve(strict=True), after)
        else:
            _require(
                after.is_symlink() and after.resolve(strict=True) == before.resolve(strict=True),
                "Ha cambiado la referencia al trabajo terminado",
            )


def packed_history(source, digest):
    path = source.state / "inheritance.json"
    if not path.exists() and not (source.state / "inheritance-status.json").exists():
        return {}
    record = _read(path)
    _require(
        _read(source.state / "inheritance-status.json").get("status") == "completed"
        and record.get("kind") == "comparison_campaign_inheritance_v1"
        and record.get("code_commit") == packed.PACKED_RELEASE
        and record.get("source_declaration_sha256") == record.get("declaration_sha256") == digest
        and record.get("final_test_opened") is False
        and isinstance(record.get("inherited"), dict),
        "Falta el recibo confirmado de la herencia de entradas preparadas",
    )
    return record


def completed_before_packing(source, key, record, history, digest):
    """Solo una terminación ya confirmada en el recibo admite equivalencias de código."""
    entry = history.get("inherited", {}).get(key)
    previous = record.get("source_execution")
    if entry is not None:
        fields = ("code_commit", "declaration_sha256", "source_state", "source_record_sha256")
        origin = {field: entry[field] for field in fields}
        if "previous" in entry:
            origin["previous"] = entry["previous"]
        _require(
            origin == previous
            and previous["code_commit"] in {packed.GPU_RELEASE, packed.WARMUP_RELEASE}
            and previous["declaration_sha256"] == digest
            and previous["source_state"] == history["source_state"],
            "El resultado no conserva la procedencia del recibo anterior",
        )
        prior = Path(previous["source_state"])
        prior_record = prior / "queue" / source.record(key).name
        _require(
            sha256(prior_record) == previous["source_record_sha256"],
            "Ha cambiado el registro histórico",
        )
        if entry["status"] == "completed":
            _require(
                record["status"] == "completed"
                and record["attempts"] == _read(prior_record)["attempts"]
                and sha256(source.output(key) / "run.json")
                == sha256(prior / "runs" / key / "run.json"),
                "El trabajo terminado difiere de su resultado histórico",
            )
            return True
        _require(entry["status"] == "paused", "La procedencia anterior tiene otro estado")
    else:
        _require(previous is None, "El resultado histórico no consta en la herencia")
    execution = record["attempts"][-1].get("execution", {})
    _require(
        execution.get("code_commit") == packed.PACKED_RELEASE
        and execution.get("declaration_sha256") == digest
        and "code_changes_sha256" in execution
        and execution["code_changes_sha256"] is None,
        "El último intento carece de una ejecución limpia con las entradas preparadas",
    )
    return False


def completed_writer(root, job, report, release):
    """Aceptar el cambio de almacenamiento únicamente en predicciones de lectores ya emitidas."""
    code = report.get("request", {}).get("code", {})
    if BATCHES_MODULE not in code:
        return None
    previous = code[BATCHES_MODULE]
    _require(
        previous
        == PACKED_BATCHES_SHA256
        == packed.git_digest(root, packed.PACKED_RELEASE, BATCHES_MODULE),
        "El código del escritor de origen no coincide con las entradas preparadas",
    )
    current = sha256(packed._path(root, BATCHES_MODULE))
    if current == previous:
        return None
    _require(
        job.runner == "reader"
        and report.get("status") == "completed"
        and report.get("kind") == "mars_titan_walk_forward_window"
        and release == TRACE_RELEASE
        and current == QUALIFIED_WRITER_SHA256 == packed.git_digest(root, release, BATCHES_MODULE),
        "El escritor solo está cualificado para un lector terminado con la revisión verificada",
    )
    return previous, current


def same_implementation(root, modules, *, writer=None):
    for module, digest in modules.items():
        path = packed._path(root, module)
        current = sha256(path) if path.is_file() else None
        if module == BATCHES_MODULE and writer == (digest, current):
            continue
        _require(
            current is not None
            and current == digest == packed.git_digest(root, packed.PACKED_RELEASE, module),
            f"El código numérico de {module} cambió desde la versión de origen",
        )


def _verify_indices(indices_root, outer):
    dataset = outer["request"]["view_sha256"]
    identities = outer["identity"]["indices"]
    expected = {"train", "validation", "calibration", "evaluation"}
    _require(
        set(identities) == expected and len(set(identities.values())) == len(expected),
        "El ajuste necesita los cuatro índices de la ventana",
    )
    matched = set()
    for path in (indices_root / dataset[:16]).glob("*/manifest.json"):
        digest = sha256(path)
        if digest not in identities.values():
            continue
        index = _read(path)
        phase = index["identity"]["phase"]["partition"]
        events_sha = index["events_sha256"]
        events = path.parent / f"events-{events_sha}.parquet"
        _require(
            index.get("kind") == "financial_observation_index"
            and index.get("final_test_opened") is False
            and index["identity"]["source_sha256"] == dataset
            and identities.get(phase) == digest
            and phase not in matched
            and index["events_path"] == events.name
            and events.is_file()
            and events.stat().st_size == index["bytes"]
            and sha256(events) == events_sha,
            "El índice o sus eventos no corresponden al ajuste pausado",
        )
        matched.add(phase)
    _require(matched == expected, "Falta un índice confirmado del ajuste pausado")


def recovery_checkpoint(folder, indices_root, root):
    """Abrir la copia con su identidad original y contrastar el cursor con el informe."""
    from .checkpoints import _index, _intact, load_training_state

    outer = _read(folder / "run.json")
    fit = _read(folder / "fit/run.json")
    identity, window = fit["identity"], outer["identity"]
    reference = fit["recovery_checkpoint"]
    digest = reference["sha256"]
    _require(
        fit.get("status") == outer.get("status") == "paused"
        and fit.get("final_test_opened") is False
        and outer.get("final_test_opened") is False
        and fit["run_id"] == packed.run_id(identity) == window["fit_run_id"]
        and outer["run_id"] == packed.run_id(window)
        and window["request"] == outer["request"]
        and outer["request"]["view_sha256"] == identity["dataset_sha256"]
        and set(identity["sources"]) == {"train", "validation"}
        and all(s["index_sha256"] == window["indices"][p] for p, s in identity["sources"].items())
        and reference["path"] == f"checkpoints/state-{digest}.pt",
        "La ventana, el ajuste y la referencia de recuperación no comparten identidad",
    )
    same_implementation(root, identity["implementation"])
    _verify_indices(indices_root, outer)
    directory = folder / "fit/checkpoints"
    index = _index(directory)
    best = index["best"]
    best_ref = (
        None if best is None else dict(path="checkpoints/" + best["name"], sha256=best["sha256"])
    )
    _require(
        fit.get("best_checkpoint") == best_ref,
        "El mejor estado del ajuste no coincide con su índice",
    )
    retained = [*index["latest"], *index["pinned"], *([best] if best else [])]
    _require(
        all(_intact(directory, item) for item in retained),
        "Ha cambiado un checkpoint retenido por el ajuste",
    )
    state = load_training_state(directory, expected_identity=identity, expected_sha256=digest)
    _require(
        type(state["global_step"]) is int
        and state["global_step"] == fit["global_step"]
        and state["cursor"] == fit["cursor"]
        and state.get("history") == fit["history"]
        and state.get("selection") == fit["selection"]
        and isinstance(state.get("rng"), dict)
        and state["rng"]
        and isinstance(state.get("optimizer"), dict)
        and state["optimizer"]
        and isinstance(state.get("model"), dict)
        and state["model"],
        "El cursor, los parámetros o el estado del optimizador y RNG no concuerdan con el ajuste",
    )
    return dict(
        reference,
        global_step=fit["global_step"],
        cursor=fit["cursor"],
        fit_run_id=fit["run_id"],
        retained_checkpoints=len({item["sha256"] for item in retained}),
    )
