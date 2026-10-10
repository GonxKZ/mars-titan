"""Frontera CPU entre un lote del corpus y las entradas visibles del predictor."""

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch

from mars_titan.data.input_policy import (
    MODALITIES,
    STRICT_INPUTS,
    masked_inputs,
    validate_historical_vectors,
)
from mars_titan.data.price_windows import PRICE_WINDOW_CHANNELS, validate_price_window
from mars_titan.training.cohort_contract import representation_identity

from .config import bounded_integer, canonical

FINAL_TEST_US = 1_704_067_200_000_000
HISTORICAL_START_US = 946_684_800_000_000
MAX_INPUT_BYTES = 64 * 1024**2


@dataclass(frozen=True, init=False)
class FinancialInputSpec:
    source_sha256: str
    view_sha256: str
    input_policy: str
    context: int
    widths: tuple[int, ...]
    _representation_json: str

    def __init__(
        self, *, source_sha256, view_sha256, representation, dimensions, input_policy=STRICT_INPUTS
    ):
        for digest in (source_sha256, view_sha256):
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("La entrada necesita huellas SHA-256 explícitas")
        if not isinstance(dimensions, dict) or set(dimensions) != set(MODALITIES):
            raise ValueError("La entrada necesita las cinco dimensiones canónicas")
        for width in dimensions.values():
            bounded_integer(width, "dimensión", 1, 2048)
        resolved = representation_identity(representation, input_policy=input_policy)
        context = resolved["context_sessions"]
        bounded_integer(context, "contexto", 2, 512)
        # Desde la v3.1 la representación declara el canal de presencia de cada sesión.
        channels = len(PRICE_WINDOW_CHANNELS) if "price_window" in resolved else 5
        if dimensions["prices"] != channels or (masked_inputs(input_policy) and context != 64):
            raise ValueError("El contexto histórico requiere 64 sesiones y sus canales de precio")
        if masked_inputs(input_policy) and any(
            dimensions[name] != 3 * len(resolved[catalog])
            for name, catalog in (
                ("fundamentals", "fundamental_concepts"),
                ("macro", "macro_indicators"),
            )
        ):
            raise ValueError("Las dimensiones no corresponden a los catálogos históricos")
        representation_json = canonical(resolved)
        if len(representation_json.encode()) > 64 * 1024:
            raise ValueError("La identidad de representación supera 64 KiB")
        for key, value in (
            ("source_sha256", source_sha256),
            ("view_sha256", view_sha256),
            ("input_policy", input_policy),
            ("context", context),
            ("widths", tuple(dimensions[name] for name in MODALITIES)),
            ("_representation_json", representation_json),
        ):
            object.__setattr__(self, key, value)

    @property
    def dimensions(self):
        return dict(zip(MODALITIES, self.widths, strict=True))

    @property
    def representation(self):
        return json.loads(self._representation_json)

    def identity(self):
        return dict(
            schema_version=1,
            source_sha256=self.source_sha256,
            view_sha256=self.view_sha256,
            input_policy=self.input_policy,
            representation=self.representation,
            dimensions=self.dimensions,
            context=self.context,
        )

    def fingerprint(self):
        return hashlib.sha256(canonical(self.identity()).encode()).hexdigest()


def _timestamps(values, size, name):
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.dtype("datetime64[us]")
        or values.shape != (size,)
        or np.isnat(values).any()
    ):
        raise ValueError(f"{name} necesita fechas conocidas en microsegundos UTC")
    return tuple(int(value) for value in values.astype(np.int64))


def _cpu_inputs(inputs, specification):
    if not isinstance(inputs, dict) or set(inputs) != set(MODALITIES):
        raise ValueError("Faltan modalidades en el lote")
    prices = inputs["prices"]
    if not isinstance(prices, np.ndarray) or prices.ndim != 3:
        raise ValueError("Los precios necesitan un lote NumPy de ventanas")
    size = len(prices)
    bounded_integer(size, "lote", 1, 256)
    total, copied = 0, {}
    for name, width in specification.dimensions.items():
        value = inputs[name]
        shape = (size, specification.context, width) if name == "prices" else (size, width)
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape:
            raise ValueError("Las entradas originales necesitan forma declarada y tipo float32")
        total += value.nbytes
        if total > MAX_INPUT_BYTES:
            raise ValueError("Las entradas superan 64 MiB")
        copied[name] = value.copy()
        if not np.isfinite(copied[name]).all():
            raise ValueError("Las entradas no son finitas")
        if name == "prices":
            validate_price_window(copied[name])
    return copied, size


def _cpu_presence(inputs, presence, size, specification):
    if masked_inputs(specification.input_policy):
        if not isinstance(presence, np.ndarray) or presence.shape != (size, 5):
            raise ValueError("La presencia no corresponde al lote")
        validate_historical_vectors(
            {name: inputs[name] for name in MODALITIES[1:]},
            presence,
            specification.representation,
        )
        return presence
    if presence is not None:
        raise ValueError("La ruta estricta no admite una política de máscaras implícita")
    return np.ones((size, 5), dtype=np.bool_)


def _decision_identity(batch, size, specification):
    moments = _timestamps(batch.get("prediction_at"), size, "prediction_at")
    available = _timestamps(batch.get("input_available_at"), size, "input_available_at")
    lower = HISTORICAL_START_US if masked_inputs(specification.input_policy) else 0
    if len(set(moments)) != 1 or any(not lower <= at < FINAL_TEST_US for at in moments):
        raise ValueError("El lote necesita un único corte admitido anterior a 2024")
    if any(not 0 <= known <= at for known, at in zip(available, moments, strict=True)):
        raise ValueError("La entrada procede del futuro o no tiene disponibilidad válida")
    ids = batch.get("sample_ids")
    if not isinstance(ids, (tuple, list)) or len(ids) != size:
        raise ValueError("Faltan IDs de muestra")
    flows = []
    for identifier, at in zip(ids, moments, strict=True):
        if not isinstance(identifier, str) or len(identifier) > 128:
            raise ValueError("El ID de muestra no está admitido")
        split = identifier.rsplit("/", 1)
        if (
            len(split) != 2
            or split[1] != str(at)
            or re.fullmatch(r"(?:US|CN)/[A-Z0-9.^_=\-]{1,64}", split[0]) is None
        ):
            raise ValueError("La muestra no identifica flujo y corte")
        flows.append(split[0])
    if len(set(flows)) != size:
        raise ValueError("El lote repite un flujo")
    return tuple(flows), tuple(ids), moments, available


@dataclass(frozen=True, init=False)
class CPUDecisionBatch:
    """Copia CPU validada compartida por el predictor y un codec separado."""

    inputs: MappingProxyType
    presence: np.ndarray
    flow_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    prediction_at: tuple[int, ...]
    input_available_at: tuple[int, ...]
    input_contract_id: str
    input_digest: str
    _array_contract: tuple

    def _signature(self):
        return tuple(
            (name, value.shape, value.dtype.str, value.strides)
            for name, value in (*self.inputs.items(), ("presence", self.presence))
        )

    def _digest(self):
        identity = dict(
            flow_ids=self.flow_ids,
            sample_ids=self.sample_ids,
            prediction_at=self.prediction_at,
            input_available_at=self.input_available_at,
            input_contract_id=self.input_contract_id,
        )
        digest = hashlib.sha256(canonical(identity).encode())
        for name in MODALITIES:
            digest.update(self.inputs[name].tobytes())
        digest.update(self.presence.tobytes())
        return digest.hexdigest()

    def verify(self):
        """Comprobar la vista antes de compartirla con un consumidor CPU."""
        if self._signature() != self._array_contract or any(
            value.flags.writeable for value in (*self.inputs.values(), self.presence)
        ):
            raise ValueError("El contrato de la vista CPU ha cambiado")
        if self._digest() != self.input_digest:
            raise ValueError("La huella de la vista CPU no coincide con sus datos")


def validated_cpu_batch(batch, specification):
    """Validar y copiar una observación una sola vez, sin campos de supervisión."""
    if not isinstance(batch, dict) or not isinstance(specification, FinancialInputSpec):
        raise ValueError("Falta la especificación de entradas")
    inputs, size = _cpu_inputs(batch.get("inputs"), specification)
    presence = _cpu_presence(inputs, batch.get("presence"), size, specification).copy()
    flows, ids, moments, available = _decision_identity(batch, size, specification)
    identity = dict(
        flow_ids=flows,
        sample_ids=ids,
        prediction_at=moments,
        input_available_at=available,
        input_contract_id=specification.fingerprint(),
    )
    for name in MODALITIES:
        value = inputs[name]
        inputs[name] = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    presence = np.frombuffer(presence.tobytes(), dtype=presence.dtype).reshape(presence.shape)
    result = object.__new__(CPUDecisionBatch)
    for name, value in dict(
        inputs=MappingProxyType(inputs),
        presence=presence,
        **identity,
    ).items():
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_array_contract", result._signature())
    object.__setattr__(result, "input_digest", result._digest())
    return result


@dataclass(frozen=True, init=False)
class DecisionBatch:
    inputs: dict[str, torch.Tensor]
    presence: torch.Tensor
    flow_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    prediction_at: tuple[int, ...]
    input_available_at: tuple[int, ...]
    input_contract_id: str
    input_digest: str
    _versions: tuple

    @classmethod
    def _create(cls, inputs, presence, flows, ids, moments, available, identity, digest):
        result = object.__new__(cls)
        for name, value in zip(
            (
                "inputs",
                "presence",
                "flow_ids",
                "sample_ids",
                "prediction_at",
                "input_available_at",
                "input_contract_id",
                "input_digest",
            ),
            (inputs, presence, flows, ids, moments, available, identity, digest),
            strict=True,
        ):
            object.__setattr__(result, name, value)
        object.__setattr__(result, "_versions", result._signature())
        return result

    def _signature(self):
        return tuple(
            (name, id(value), value._version, value.shape, value.dtype, value.device)
            for name, value in (*self.inputs.items(), ("presence", self.presence))
        )

    def verify(self):
        if self._signature() != self._versions:
            raise ValueError("Las entradas cambiaron después de su validación CPU")
        for value in self.inputs.values():
            if not torch.isfinite(value).all():
                raise ValueError("Una entrada contiene NaN o infinito")

    @classmethod
    def from_corpus(cls, batch, specification, *, device="cpu", dtype=torch.float32):
        """Copiar entradas verificadas en CPU, sin leer objetivos o maduración."""
        return cls.from_validated(
            validated_cpu_batch(batch, specification), device=device, dtype=dtype
        )

    @classmethod
    def from_validated(cls, batch, *, device="cpu", dtype=torch.float32):
        if not isinstance(batch, CPUDecisionBatch):
            raise ValueError("La conversión necesita un lote CPU verificado")
        batch.verify()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("El cálculo requiere float32 o float64")
        tensors = {
            name: torch.tensor(value, device=device, dtype=dtype)
            for name, value in batch.inputs.items()
        }
        return cls._create(
            tensors,
            torch.tensor(batch.presence, device=device, dtype=torch.bool),
            batch.flow_ids,
            batch.sample_ids,
            batch.prediction_at,
            batch.input_available_at,
            batch.input_contract_id,
            batch.input_digest,
        )

    def select(self, indices):
        self.verify()
        if (
            not isinstance(indices, (tuple, list))
            or not indices
            or any(type(i) is not int or not 0 <= i < len(self.flow_ids) for i in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("La selección de flujos necesita índices únicos válidos")
        indices = list(indices)
        return DecisionBatch._create(
            {name: values[indices] for name, values in self.inputs.items()},
            self.presence[indices],
            tuple(self.flow_ids[i] for i in indices),
            tuple(self.sample_ids[i] for i in indices),
            tuple(self.prediction_at[i] for i in indices),
            tuple(self.input_available_at[i] for i in indices),
            self.input_contract_id,
            hashlib.sha256(
                canonical(["selection_v1", self.input_digest, indices]).encode()
            ).hexdigest(),
        )
