"""Instantánea pura de episodios maduros, sin admitir ni publicar registros."""

import hashlib
import re
from dataclasses import dataclass

import torch

from .config import bounded_integer, canonical, require_identity
from .state import check_finite, require_payload

FIELDS = ("keys", "values", "labels", "ids", "decision_at", "available_at", "maturity_at")


def _digest_id(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} necesita una huella SHA256")


def _device(value):
    device = torch.device(value)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and device.index is None):
        raise ValueError("El dispositivo debe ser CPU o CUDA con índice explícito")
    return device


def _signature(tensors):
    return tuple(
        (
            id(t),
            t._version,
            t.shape,
            t.dtype,
            t.device,
            t.stride(),
            t.storage_offset(),
            t.untyped_storage().data_ptr(),
            t.untyped_storage().nbytes(),
            t.requires_grad,
        )
        for t in tensors
    )


def _fingerprint(metadata, tensors):
    result = hashlib.sha256(canonical(metadata).encode())
    for name, tensor in zip(FIELDS, tensors, strict=True):
        value = tensor.detach().cpu().contiguous().numpy()
        result.update(canonical([name, str(value.dtype), list(value.shape)]).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def _validate(tensors, *, dtype, cutoff, max_bytes, source):
    if type(tensors) is not dict or set(tensors) != set(FIELDS):
        raise ValueError("La instantánea no conserva sus campos tensoriales")
    ids = tensors["ids"]
    if not isinstance(ids, torch.Tensor) or ids.ndim != 1:
        raise ValueError("Los IDs deben ser un vector")
    count = len(ids)
    bounded_integer(count, "episodios", 0, 1024)
    expected_dtypes = (
        torch.float32 if source else torch.float64,
        torch.float32 if source else dtype,
        torch.float64,
        torch.int64,
        torch.int64,
        torch.int64,
        torch.int64,
    )
    total = 0
    for index, (name, expected_dtype) in enumerate(zip(FIELDS, expected_dtypes, strict=True)):
        value = tensors[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or value.dtype != expected_dtype
            or value.shape != ((count, 64) if index < 2 else (count,))
            or value.requires_grad
            or value.grad_fn is not None
            or not value.is_contiguous()
        ):
            raise ValueError("El episodio necesita formas y tipos CPU compatibles, sin grafo")
        total += value.untyped_storage().nbytes()
        if total + 1024 > max_bytes:
            raise ValueError("Las fuentes de la instantánea superan el presupuesto")
        check_finite(value, "El episodio")
    if (ids <= 0).any() or (ids[1:] <= ids[:-1]).any():
        raise ValueError("Los IDs deben ser positivos, únicos y crecientes")
    decision, available, maturity = (tensors[name] for name in FIELDS[4:])
    if (
        (available < 0) | (available > decision) | (decision >= maturity) | (maturity > cutoff)
    ).any():
        raise ValueError("El episodio no conserva disponibilidad y maduración al corte")
    norms = tensors["keys"].double().square().sum(-1)
    if ((norms - 1).abs() > 1e-6).any():
        raise ValueError("Las claves deben estar normalizadas por el banco")
    return count


@dataclass(frozen=True, init=False)
class EpisodeSnapshot:
    """Copias por dispositivo. El consumidor acredita la procedencia de las fechas."""

    codec_id: str
    context_id: str
    cutoff: int
    dtype: torch.dtype
    device: torch.device
    max_bytes: int
    _values: tuple
    _versions: tuple
    _digest: str

    @classmethod
    def create(
        cls,
        *,
        keys,
        values,
        labels,
        ids,
        decision_at,
        available_at,
        maturity_at,
        cutoff,
        codec_id,
        context_id,
        device="cpu",
        dtype=torch.float32,
        max_bytes=2 * 1024**2,
    ):
        if torch.is_inference_mode_enabled():
            raise ValueError(
                "La instantánea necesita contadores de versión fuera de inference_mode"
            )
        cls._options(codec_id, context_id, cutoff, dtype, max_bytes)
        device = _device(device)
        tensors = dict(
            zip(
                FIELDS,
                (keys, values, labels, ids, decision_at, available_at, maturity_at),
                strict=True,
            )
        )
        count = _validate(tensors, dtype=dtype, cutoff=cutoff, max_bytes=max_bytes, source=True)
        # La cota de destino se comprueba antes de crear copias o convertir precisión.
        cls._result_budget(count, dtype, max_bytes)
        converted = dict(tensors, keys=keys.double(), values=values.to(dtype=dtype))
        return cls._build(converted, codec_id, context_id, cutoff, dtype, device, max_bytes)

    @staticmethod
    def _options(codec_id, context_id, cutoff, dtype, max_bytes):
        _digest_id(codec_id, "El codec")
        _digest_id(context_id, "El contexto")
        bounded_integer(cutoff, "corte", 0, 2**63 - 1)
        bounded_integer(max_bytes, "presupuesto de instantánea", 1024, 8 * 1024**2)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("La instantánea admite FP32 o FP64")

    @staticmethod
    def _result_budget(count, dtype, max_bytes):
        size = 4 if dtype == torch.float32 else 8
        if 1024 + count * (64 * (8 + size) + 5 * 8) > max_bytes:
            raise ValueError("La instantánea de destino supera el presupuesto")

    @classmethod
    def _build(cls, tensors, codec_id, context_id, cutoff, dtype, device, max_bytes):
        instance = object.__new__(cls)
        for name, value in dict(
            codec_id=codec_id,
            context_id=context_id,
            cutoff=cutoff,
            dtype=dtype,
            device=device,
            max_bytes=max_bytes,
        ).items():
            object.__setattr__(instance, name, value)
        copied = tuple(
            tensors[name].detach().to(device=device if index < 4 else "cpu", copy=True).contiguous()
            for index, name in enumerate(FIELDS)
        )
        object.__setattr__(instance, "_values", copied)
        object.__setattr__(instance, "_versions", _signature(copied))
        object.__setattr__(
            instance,
            "_digest",
            _fingerprint(instance.identity(), tuple(tensors[name] for name in FIELDS)),
        )
        return instance

    def identity(self):
        return dict(
            schema_version=1,
            recipe="mature_episode_snapshot_64_v1",
            codec_id=self.codec_id,
            context_id=self.context_id,
            cutoff=self.cutoff,
            dtype=str(self.dtype),
            max_bytes=self.max_bytes,
            key_dtype="torch.float64",
            labels_dtype="torch.float64",
        )

    @property
    def count(self):
        return self._values[3].numel()

    def check(self, *, codec_id, context_id, cutoff, dtype, device):
        if (
            codec_id != self.codec_id
            or context_id != self.context_id
            or type(cutoff) is not int
            or cutoff != self.cutoff
            or dtype != self.dtype
            or torch.device(device) != self.device
            or _signature(self._values) != self._versions
        ):
            raise ValueError("La instantánea ha cambiado o no corresponde al contexto del lector")

    def verify(self):
        """Contrastar bytes en una frontera de recuperación, nunca por consulta de K."""
        if (
            _signature(self._values) != self._versions
            or _fingerprint(self.identity(), self._values) != self._digest
        ):
            raise ValueError("Los bytes de la instantánea no conservan su huella")

    def usage(self):
        return dict(zip(FIELDS, (t.numel() * t.element_size() for t in self._values), strict=True))

    def export_cpu(self):
        self.verify()
        return dict(
            schema_version=1,
            identity=self.identity(),
            sha256=self._digest,
            tensors={
                name: value.detach().to(device="cpu", copy=True).contiguous()
                for name, value in zip(FIELDS, self._values, strict=True)
            },
        )

    @classmethod
    def restore(cls, payload, *, codec_id, context_id, cutoff, device, dtype):
        if torch.is_inference_mode_enabled():
            raise ValueError(
                "La recuperación necesita contadores de versión fuera de inference_mode"
            )
        value = require_payload(payload, {"schema_version", "identity", "sha256", "tensors"})
        identity = value["identity"]
        if not isinstance(identity, dict):
            raise ValueError("Falta la identidad de la instantánea")
        maximum = identity.get("max_bytes")
        cls._options(codec_id, context_id, cutoff, dtype, maximum)
        expected = dict(
            schema_version=1,
            recipe="mature_episode_snapshot_64_v1",
            codec_id=codec_id,
            context_id=context_id,
            cutoff=cutoff,
            dtype=str(dtype),
            max_bytes=maximum,
            key_dtype="torch.float64",
            labels_dtype="torch.float64",
        )
        require_identity(identity, expected)
        count = _validate(
            value["tensors"], dtype=dtype, cutoff=cutoff, max_bytes=maximum, source=False
        )
        cls._result_budget(count, dtype, maximum)
        tensors = tuple(value["tensors"][name] for name in FIELDS)
        if _fingerprint(expected, tensors) != value["sha256"]:
            raise ValueError("El archivo no conserva los bytes de la instantánea")
        return cls._build(
            value["tensors"], codec_id, context_id, cutoff, dtype, _device(device), maximum
        )
