"""Artefactos sellados de sesión, sin índice ni publicación independientes."""

import hashlib
import io
import json
import math
import os
import pickle
import re
import zipfile
from pathlib import Path

import torch


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload(value, budget, depth=0):
    budget[1] -= 1
    if depth > 24 or budget[1] < 0:
        raise ValueError("El artefacto excede su profundidad o número de elementos")
    if isinstance(value, torch.Tensor):
        size = value.numel() * value.element_size()
        budget[0] -= size + 128
        if (
            value.device.type != "cpu"
            or value.layout != torch.strided
            or value.requires_grad
            or value.grad_fn is not None
            or not value.is_contiguous()
            or value.storage_offset() != 0
            or value.untyped_storage().nbytes() != size
            or value.dtype
            not in (torch.float32, torch.float64, torch.int64, torch.uint8, torch.bool)
            or budget[0] < 0
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                "El tensor necesita almacenamiento CPU propio, finito, acotado y sin grafo"
            )
        return value
    budget[0] -= len(value.encode()) + 32 if type(value) is str else 32
    if isinstance(value, bytes):
        budget[0] -= len(value)
    if budget[0] < 0:
        raise ValueError("El contenido del artefacto supera el presupuesto")
    if value is None or type(value) in (bool, int, str, bytes):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise ValueError("Las claves del estado necesitan cadenas o enteros")
        return {
            _payload(key, budget, depth + 1): _payload(value[key], budget, depth + 1)
            for key in sorted(value, key=lambda key: (type(key).__name__, str(key)))
        }
    if isinstance(value, (list, tuple)):
        items = [_payload(item, budget, depth + 1) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    raise ValueError("El artefacto contiene un tipo o número no admitido")


class _Buffer(io.BytesIO):
    def __init__(self, maximum):
        super().__init__()
        self.maximum = maximum

    def write(self, value):
        if self.tell() + len(value) > self.maximum:
            raise ValueError("La serialización excede el presupuesto de archivo")
        return super().write(value)


class SessionArtifacts:
    """Guardar propuestas y retirar solo archivos que el coordinador declara huérfanos."""

    def __init__(
        self,
        native,
        directory,
        *,
        max_bytes=64 * 1024**2,
        max_files=128,
        max_total_bytes=512 * 1024**2,
    ):
        if (
            type(max_bytes) is not int
            or not 65536 <= max_bytes <= 64 * 1024**2
            or type(max_files) is not int
            or not 1 <= max_files <= 4096
            or type(max_total_bytes) is not int
            or not max_bytes <= max_total_bytes <= 2 * 1024**3
        ):
            raise ValueError("El presupuesto de artefactos de sesión no es válido")
        self.native, self.directory = native, Path(directory)
        self.max_bytes, self.max_files, self.max_total_bytes = max_bytes, max_files, max_total_bytes
        native.require_safe_path(str(self.directory))

    @staticmethod
    def _header(identity, kind):
        if (
            not isinstance(identity, str)
            or not re.fullmatch(r"[0-9a-f]{64}", identity)
            or not isinstance(kind, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", kind)
        ):
            raise ValueError("Falta la identidad o el tipo de artefacto")
        return dict(
            schema_version=1,
            identity=identity,
            kind=kind,
            format="torch_weights_only_ordered_maps_v1",
            torch_version=str(torch.__version__),
        )

    def _inventory(self):
        if not self.directory.exists():
            return []
        result, total = [], 0
        for path in self.directory.iterdir():
            self.native.require_safe_path(str(path))
            if not path.is_file() or not re.fullmatch(
                r"(?:[0-9a-f]{64}\.bin|\.[0-9a-f]{64}\.bin\.pending-[A-Za-z0-9]+)", path.name
            ):
                raise ValueError("El almacén de artefactos contiene una entrada ajena")
            total += path.stat().st_size
            result.append(path)
            if len(result) > self.max_files or total > self.max_total_bytes:
                raise ValueError("El almacén de artefactos excede su presupuesto")
        return result

    def stage(self, data, *, identity, kind):
        header = self._header(identity, kind)
        value = _payload(data, [self.max_bytes - 65536, 1_000_000])
        buffer = _Buffer(self.max_bytes)
        torch.save(dict(header=header, data=value), buffer)
        content = buffer.getvalue()
        name = hashlib.sha256(content).hexdigest() + ".bin"
        files = self._inventory()
        extra = 0 if any(path.name == name for path in files) else len(content)
        if len(files) + int(extra > 0) > self.max_files or (
            sum(path.stat().st_size for path in files) + extra > self.max_total_bytes
        ):
            raise ValueError("La propuesta agota el presupuesto de artefactos")
        result = self.native.seal_blob(str(self.directory), self.max_bytes, content)
        return dict(result, **header)

    def _bytes(self, reference, *, identity, kind):
        header = self._header(identity, kind)
        if (
            not isinstance(reference, dict)
            or set(reference) != set(header) | {"name", "sha256", "bytes"}
            or _canonical({key: reference[key] for key in header}) != _canonical(header)
            or type(reference["bytes"]) is not int
            or not 0 < reference["bytes"] <= self.max_bytes
            or not isinstance(reference["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])
            or reference["name"] != reference["sha256"] + ".bin"
        ):
            raise ValueError("La referencia del artefacto no conserva su contrato")
        try:
            value = self.native.read_blob(str(self.directory / reference["name"]), self.max_bytes)
        except (RuntimeError, OSError) as error:
            raise ValueError("No se puede leer el artefacto confirmado") from error
        if (
            len(value) != reference["bytes"]
            or hashlib.sha256(value).hexdigest() != reference["sha256"]
        ):
            raise ValueError("El artefacto confirmado ha cambiado")
        return value

    def read(self, reference, *, identity, kind):
        content = self._bytes(reference, identity=identity, kind=kind)
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                entries = archive.infolist()
                if len(entries) > 16384 or sum(item.file_size for item in entries) > self.max_bytes:
                    raise ValueError("El artefacto descomprimido excede el presupuesto")
            value = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        except (zipfile.BadZipFile, pickle.UnpicklingError, RuntimeError) as error:
            raise ValueError("El artefacto no admite una carga restringida") from error
        if (
            not isinstance(value, dict)
            or set(value) != {"header", "data"}
            or _canonical(value["header"]) != _canonical(self._header(identity, kind))
        ):
            raise ValueError("El contenido del artefacto pertenece a otro contrato")
        return _payload(value["data"], [self.max_bytes - 65536, 1_000_000])

    def prune_unreferenced(self, live_references):
        if not isinstance(live_references, (tuple, list)) or len(live_references) > self.max_files:
            raise ValueError("El conjunto de artefactos vivos supera el presupuesto")
        names = set()
        for reference in live_references:
            self._bytes(reference, identity=reference["identity"], kind=reference["kind"])
            names.add(reference["name"])
        for path in self._inventory():
            if path.name not in names:
                path.unlink()
        if self.directory.exists():
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
