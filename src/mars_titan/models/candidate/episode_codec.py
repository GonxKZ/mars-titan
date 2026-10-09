"""Proyecciones fijas CPU del candidato, con identidad y almacenamiento propios."""

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.input_policy import MODALITIES
from mars_titan.models.titans.config import bounded_integer, canonical
from mars_titan.models.titans.financial_inputs import CPUDecisionBatch, DecisionBatch

from .input_adapter import CandidateInputAdapter

MAX_CODEC_BYTES = 64 * 1024**2
_CODE_MODULES = (
    __name__,
    "mars_titan.models.titans.financial_inputs",
    "mars_titan.data.input_policy",
    "mars_titan.training.cohort_contract",
)


def _numerics():
    return dict(
        torch_version=str(torch.__version__),
        cpu_capability=torch.backends.cpu.get_cpu_capability(),
        cpu_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        mkldnn_enabled=torch.backends.mkldnn.enabled,
        mkldnn_matmul_precision=torch.backends.mkldnn.matmul.fp32_precision,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
    )


def _immutable(value):
    array = value.detach().numpy()
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, init=False)
class CandidateEncoding:
    keys: np.ndarray
    values: np.ndarray
    flow_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    prediction_at: tuple[int, ...]
    input_available_at: tuple[int, ...]
    input_digest: str
    codec_id: str
    estimated_peak_bytes: int
    _array_contract: tuple
    _digest: str

    def _signature(self):
        return tuple((a.shape, a.dtype.str, a.strides) for a in (self.keys, self.values))

    def _content_digest(self):
        digest = hashlib.sha256(
            canonical(
                [
                    self.flow_ids,
                    self.sample_ids,
                    self.prediction_at,
                    self.input_available_at,
                    self.input_digest,
                    self.codec_id,
                ]
            ).encode()
        )
        digest.update(self.keys.tobytes())
        digest.update(self.values.tobytes())
        return digest.hexdigest()

    def verify(self):
        if self._signature() != self._array_contract or any(
            value.flags.writeable for value in (self.keys, self.values)
        ):
            raise ValueError("La codificación GRU cambió su contrato de arrays")
        if self._content_digest() != self._digest:
            raise ValueError("La codificación GRU no conserva sus bytes e identidades")


class FrozenCandidateCodec:
    """El codec no conoce pesos aprendidos y no reconstruye la GRU al codificar."""

    def __init__(self, adapter, *, max_working_bytes=MAX_CODEC_BYTES):
        if type(adapter) is not CandidateInputAdapter:
            raise ValueError("El codec necesita el adaptador identificado del candidato")
        bounded_integer(max_working_bytes, "presupuesto del codec", 1, MAX_CODEC_BYTES)
        if not hasattr(adapter.model, "cpu_episode_codec"):
            raise ValueError("El enlace no incluye las proyecciones CPU del candidato")
        self._codec = adapter.model.cpu_episode_codec(max_working_bytes)
        self._native = adapter.native
        self._input_contract = adapter.specification.fingerprint()
        self._max_working_bytes = max_working_bytes
        self._numerics = _numerics()
        identity = dict(
            schema_version=1,
            recipe="candidate_cpu_fixed_rows_v1",
            input_specification=adapter.specification.identity(),
            projection_id=self._codec.projection_id,
            dtype=self._codec.dtype,
            key_width=128,
            value_width=256,
            device="cpu",
            row_unit=1,
            max_batch=adapter.model.config.max_batch,
            max_working_bytes=max_working_bytes,
            normalization=dict(
                epsilon=1e-12,
                unit_tolerance=1e-5,
                exact_zero_allowed=True,
                subunit_nonzero="reject",
            ),
            numerics=self._numerics,
            native_binary_sha256=adapter.native.binary_sha256,
            implementation={
                name: hashlib.sha256(
                    Path(importlib.import_module(name).__file__).read_bytes()
                ).hexdigest()
                for name in _CODE_MODULES
            },
        )
        self._identity_json = canonical(identity)
        self._codec_id = hashlib.sha256(self._identity_json.encode()).hexdigest()

    def identity(self):
        return json.loads(self._identity_json)

    def fingerprint(self):
        return self._codec_id

    def encode(self, batch):
        if (
            not isinstance(batch, CPUDecisionBatch)
            or batch.input_contract_id != self._input_contract
        ):
            raise ValueError("El lote no corresponde a los inputs del codec GRU")
        batch.verify()
        if _numerics() != self._numerics:
            raise ValueError("La ejecución CPU del codec cambió respecto de su identidad")
        estimated = self._codec.estimated_bytes(len(batch.flow_ids))
        if estimated > self._max_working_bytes:
            raise ValueError("La codificación excede el presupuesto del codec")
        dtype = torch.float64 if self._codec.dtype == "float64" else torch.float32
        with torch.inference_mode(False), torch.no_grad():
            tensors = DecisionBatch.from_validated(batch, device="cpu", dtype=dtype)
            encoded = self._codec.encode(
                self._native.CandidateInputs(
                    *(tensors.inputs[name] for name in MODALITIES), tensors.presence
                )
            )
        result = object.__new__(CandidateEncoding)
        for name, value in dict(
            keys=_immutable(encoded.keys),
            values=_immutable(encoded.values),
            flow_ids=batch.flow_ids,
            sample_ids=batch.sample_ids,
            prediction_at=batch.prediction_at,
            input_available_at=batch.input_available_at,
            input_digest=batch.input_digest,
            codec_id=self._codec_id,
            estimated_peak_bytes=estimated,
        ).items():
            object.__setattr__(result, name, value)
        object.__setattr__(result, "_array_contract", result._signature())
        object.__setattr__(result, "_digest", result._content_digest())
        return result
