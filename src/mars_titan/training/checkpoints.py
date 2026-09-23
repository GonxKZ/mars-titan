"""Puntos de control locales con publicación atómica y continuación verificable."""

import fcntl
import json
import os
import pickle
import random
import re
import shutil
import signal
import subprocess
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.storage import atomic_json, sha256

MAX_STATE_BYTES = 512 * 1024**2


def _json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ValueError("El manifiesto del punto de control no es regular o excede su límite")
    try:
        return json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise ValueError("El manifiesto del punto de control está dañado") from error


@contextmanager
def _locked(directory, identity, *, create):
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("El directorio de puntos de control no puede ser un enlace")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("El punto de control necesita una identidad explícita")
    encoded = json.dumps(identity, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if len(encoded.encode()) > 256 * 1024:
        raise ValueError("La identidad excede el presupuesto de 256 KiB")
    canonical = json.loads(encoded)
    marker = directory / "identity.json"
    if not create and not marker.is_file():
        raise ValueError("El directorio no tiene una identidad confirmada")
    if directory.exists() and not marker.exists():
        if any(item.name != ".lock" for item in directory.iterdir()):
            raise ValueError("El directorio contiene datos de otra ejecución")
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if marker.exists():
            if _json(marker) != canonical:
                raise ValueError("La identidad no coincide con los datos y la configuración")
        else:
            atomic_json(marker, canonical)
        yield directory, canonical
    finally:
        os.close(descriptor)


def _record_valid(record):
    return (
        isinstance(record, dict)
        and set(record) == {"name", "sha256", "size", "global_step"}
        and isinstance(record["sha256"], str)
        and re.fullmatch(r"[a-f0-9]{64}", record["sha256"]) is not None
        and record["name"] == f"state-{record['sha256']}.pt"
        and type(record["size"]) is int
        and 0 < record["size"] <= MAX_STATE_BYTES
        and type(record["global_step"]) is int
        and record["global_step"] >= 0
    )


def _index(directory):
    path = directory / "latest.json"
    if not path.exists():
        return {"schema_version": 1, "latest": [], "best": None, "pinned": []}
    index = _json(path)
    if (
        not isinstance(index, dict)
        or set(index) != {"schema_version", "latest", "best", "pinned"}
        or index["schema_version"] != 1
        or not isinstance(index["latest"], list)
        or not 1 <= len(index["latest"]) <= 2
        or not isinstance(index["pinned"], list)
        or len(index["pinned"]) > 64
        or not all(_record_valid(r) for r in index["latest"] + index["pinned"])
        or (index["best"] is not None and not _record_valid(index["best"]))
    ):
        raise ValueError("El manifiesto del punto de control tiene un esquema inválido")
    return index


def _intact(directory, record):
    path = directory / record["name"]
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == record["size"]
        and sha256(path) == record["sha256"]
    )


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            torch.cuda.synchronize(value.device)
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(k) + _tensor_bytes(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    if value is None or type(value) in (bool, int, float, str):
        return len(value.encode()) if isinstance(value, str) else 16
    raise ValueError("El estado solo admite tensores y tipos simples compatibles con weights_only")


def _read_payload(path, identity, step):
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > 100_000 or sum(e.file_size for e in entries) > MAX_STATE_BYTES:
                raise ValueError("El estado descomprimido excede el presupuesto")
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError("El estado no admite una carga restringida íntegra") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("identity") != identity
        or not isinstance(payload.get("state"), dict)
        or payload["state"].get("global_step") != step
    ):
        raise ValueError("El estado cargado no concuerda con su identidad y manifiesto")
    return payload["state"]


def save_training_state(
    directory: Path, state: dict, *, identity: dict, best: bool = False, pin: bool = False
) -> Path:
    """Confirmar un estado tras la actualización y conservar sus referencias necesarias."""
    if (
        not isinstance(state, dict)
        or type(state.get("global_step")) is not int
        or state["global_step"] < 0
    ):
        raise ValueError("El estado necesita un contador de pasos confirmado")
    estimated = _tensor_bytes(state) + 1024**2
    if estimated > MAX_STATE_BYTES:
        raise ValueError("El estado excede el presupuesto de 512 MiB")
    with _locked(directory, identity, create=True) as (directory, canonical):
        index = _index(directory)
        previous = [r for r in index["latest"] if _intact(directory, r)]
        if previous and state["global_step"] < previous[0]["global_step"]:
            raise ValueError("El estado retrocede respecto al último paso confirmado")
        if shutil.disk_usage(directory).free < estimated + 16 * 1024**2:
            raise OSError(28, "No hay espacio para confirmar el siguiente punto de control")
        pending = directory / ".pending.pt"
        descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save({"schema_version": 1, "identity": canonical, "state": state}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            size, digest = pending.stat().st_size, sha256(pending)
            if size > MAX_STATE_BYTES:
                raise ValueError("El archivo serializado excede el presupuesto")
            _read_payload(pending, canonical, state["global_step"])
            record = dict(
                name=f"state-{digest}.pt",
                sha256=digest,
                size=size,
                global_step=state["global_step"],
            )
            destination = directory / record["name"]
            os.replace(pending, destination)
            index["latest"] = [record] + [r for r in previous if r["name"] != record["name"]][:1]
            if best:
                index["best"] = record
            if pin and record not in index["pinned"]:
                if len(index["pinned"]) >= 64:
                    raise ValueError("Se ha alcanzado el límite de estados referenciados")
                index["pinned"].append(record)
            atomic_json(directory / "latest.json", index)
            keep = {r["name"] for r in index["latest"] + index["pinned"]}
            if index["best"]:
                keep.add(index["best"]["name"])
            listing = subprocess.run(
                ["rg", "--files", "--hidden", "--no-ignore", str(directory), "-g", "state-*.pt"],
                capture_output=True,
                text=True,
                check=True,
            )
            for filename in listing.stdout.splitlines():
                path = Path(filename)
                if (
                    path.parent == directory
                    and path.name not in keep
                    and re.fullmatch(r"state-[a-f0-9]{64}\.pt", path.name)
                    and not path.is_symlink()
                ):
                    path.unlink()
            return destination
        finally:
            if pending.exists() and not pending.is_symlink():
                pending.unlink()


def load_training_state(
    directory: Path, *, expected_identity: dict, selection="latest", expected_sha256=None
) -> dict:
    """Restaurar solo estados confirmados. Avisar si se descarta el más reciente."""
    if selection not in {"latest", "best"}:
        raise ValueError("La selección del punto de control debe ser latest o best")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
    ):
        raise ValueError("La identidad explícita del punto de control no es válida")
    with _locked(directory, expected_identity, create=False) as (directory, canonical):
        index = _index(directory)
        selected = index["best"] if selection == "best" else next(iter(index["latest"]), None)
        if expected_sha256 is not None and (
            selected is None
            or selected["sha256"] != expected_sha256
            or not _intact(directory, selected)
        ):
            raise ValueError(
                "El punto de control seleccionado ha cambiado o no conserva su identidad íntegra"
            )
        if selection == "best":
            record = index["best"]
            if record is None or not _intact(directory, record):
                raise ValueError("No existe un punto de control seleccionado íntegro")
            return _read_payload(directory / record["name"], canonical, record["global_step"])
        candidates = index["latest"][:1] if expected_sha256 is not None else index["latest"]
        for position, record in enumerate(candidates):
            if not _intact(directory, record):
                continue
            path = directory / record["name"]
            state = _read_payload(path, canonical, record["global_step"])
            if position:
                warnings.warn(
                    "Se recupera un punto de control anterior porque el último está dañado",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return state
        raise ValueError("No existe un punto de control confirmado e íntegro")


def capture_rng(device: str) -> dict:
    """Conservar los generadores usados por las referencias y su dispositivo explícito."""
    if device not in {"cpu", "cuda:0"}:
        raise ValueError("El dispositivo debe ser cpu o cuda:0")
    if device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("cuda:0 no está disponible para guardar el estado")
    numpy_state = np.random.get_state()
    return dict(
        device=device,
        python=random.getstate(),
        torch=torch.get_rng_state(),
        numpy=[numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)), *numpy_state[2:]],
        cuda=torch.cuda.get_rng_state(0) if device == "cuda:0" else None,
    )


def restore_rng(state: dict, device: str) -> None:
    """Restaurar generadores sin cambiar de dispositivo de forma implícita."""
    if state["device"] != device:
        raise ValueError("El dispositivo no coincide con el estado aleatorio guardado")
    if device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("cuda:0 no está disponible para continuar")
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], value[1].numpy().astype(np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if device == "cuda:0":
        torch.cuda.set_rng_state(state["cuda"], 0)


class StopRequest:
    """Convertir SIGINT y SIGTERM en una solicitud para la siguiente barrera segura."""

    def __init__(self):
        self.requested = False
        self.previous = {}

    def request_stop(self, *_):
        self.requested = True

    def __enter__(self):
        for name in (signal.SIGINT, signal.SIGTERM):
            self.previous[name] = signal.signal(name, self.request_stop)
        return self

    def __exit__(self, *_):
        for name, previous in self.previous.items():
            signal.signal(name, previous)
