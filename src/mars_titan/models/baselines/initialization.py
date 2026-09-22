"""Reutilizar pesos compatibles en otra ejecución, sin confundirlo con reanudación."""

from pathlib import Path
from zipfile import ZIP_STORED, BadZipFile, ZipFile

import torch

from mars_titan.data.storage import sha256


def _canonical_hashes(hashes):
    result = {}
    for name, digest in hashes.items():
        key = "targets/" + name.rsplit("/targets/", 1)[1] if "/targets/" in name else name
        if key in result:
            raise ValueError("Las huellas contienen identidades duplicadas")
        result[key] = digest
    return result


def initialize_weights(model, path, *, config, hashes, max_bytes=64 * 1024**2):
    path = Path(path)
    if not 0 < path.stat().st_size <= max_bytes:
        raise ValueError("El checkpoint supera el presupuesto de lectura")
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if (
                len(entries) > 4096
                or sum(entry.file_size for entry in entries) > max_bytes
                or any(entry.compress_type != ZIP_STORED for entry in entries)
            ):
                raise ValueError("El checkpoint contiene compresión o supera el presupuesto")
    except BadZipFile as error:
        raise ValueError(
            "La inicialización requiere el formato ZIP sin compresión del proyecto"
        ) from error
    digest = sha256(path)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if sha256(path) != digest:
        raise ValueError("El checkpoint ha cambiado durante la lectura")
    fields = (
        "kind",
        "dimensions",
        "context",
        "hidden_size",
        "moving_average_kernel",
        "price_output_horizon",
    )
    if any(state["config"].get(key) != config.get(key) for key in fields):
        raise ValueError("La arquitectura de origen no coincide con la referencia")
    if _canonical_hashes(state["input_hashes"]) != _canonical_hashes(hashes):
        raise ValueError("Las huellas de datos, etiquetas o código no coinciden")
    if type(state["next_epoch"]) is not int or state["next_epoch"] < 1:
        raise ValueError("El origen no acredita una época completada")
    expected, weights = model.state_dict(), state["model"]
    if expected.keys() != weights.keys() or any(
        not isinstance(weights[name], torch.Tensor)
        or weights[name].shape != value.shape
        or weights[name].dtype != value.dtype
        or weights[name].layout != value.layout
        or not torch.isfinite(weights[name]).all()
        for name, value in expected.items()
    ):
        raise ValueError("Los pesos de origen no tienen formas, tipos o valores válidos")
    model.load_state_dict(weights)
    return {
        "sha256": digest,
        "source_completed_epochs": state["next_epoch"],
        "source_config": state["config"],
        "policy": "weights_only_new_optimizer_and_rng",
    }
