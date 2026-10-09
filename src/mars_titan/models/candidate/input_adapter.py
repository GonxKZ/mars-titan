"""Usar el candidato C++ con lotes verificados y memoria vacía, sin ciclo temporal."""

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path

import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES, STRICT_INPUTS
from mars_titan.memory.native_backend import load_native
from mars_titan.models.titans.config import bounded_integer, canonical
from mars_titan.models.titans.financial_inputs import (
    CPUDecisionBatch,
    DecisionBatch,
    FinancialInputSpec,
    validated_cpu_batch,
)

_CONFIG_FIELDS = (
    "dimensions",
    "max_batch",
    "max_episodes",
    "neighbors",
    "parameter_seed",
    "feature_seed",
    "key_seed",
    "temperature",
    "normalization_id",
    "input_policy",
)
_SOURCE_MODULES = (
    __name__,
    "mars_titan.models.titans.financial_inputs",
    "mars_titan.models.titans.config",
    "mars_titan.data.input_policy",
    "mars_titan.training.cohort_contract",
    "mars_titan.memory.native_backend",
)
_MAX_ARCHIVE_BYTES = 128 * 1024**2


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _normalization(specification):
    representation = specification.representation
    for key in ("input_policy", "mask_contract"):
        representation.pop(key, None)
    return _digest(dict(representation=representation, dimensions=specification.dimensions))


def _specification(value):
    if not isinstance(value, FinancialInputSpec) or value.context != 64:
        raise ValueError("El candidato necesita FinancialInputSpec con ventanas de 64 sesiones")


def _native(path):
    module = load_native(path)
    if getattr(module, "candidate_abi_version", None) != 1:
        raise ValueError("El enlace no incluye la versión tensorial del candidato")
    return module


@dataclass(frozen=True)
class CandidatePrediction:
    quantiles: torch.Tensor
    median: torch.Tensor
    native: object


class CandidateInputAdapter:
    """La GRU y su refinador originales, con identidad y política explícitas.

    No conserva estado entre ventanas ni admite episodios. El resultado nativo
    conserva sus rasgos y claves fijos 256/128 para una integración posterior.
    """

    def __init__(
        self,
        specification,
        *,
        native_path=None,
        dtype=torch.float32,
        device="cpu",
        parameter_seed=42,
        feature_seed=43,
        key_seed=44,
    ):
        _specification(specification)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("El candidato necesita float32 o float64")
        for seed in (parameter_seed, feature_seed, key_seed):
            bounded_integer(seed, "semilla", 0, 2**63 - 1)
        native = _native(native_path)
        config = native.CandidateConfig()
        config.dimensions = specification.widths
        config.input_policy = specification.input_policy
        config.normalization_id = _normalization(specification)
        config.parameter_seed, config.feature_seed, config.key_seed = (
            parameter_seed,
            feature_seed,
            key_seed,
        )
        model = native.Candidate(config, str(dtype).removeprefix("torch."), str(device))
        self._bind(specification, native, model)

    def _bind(self, specification, native, model):
        config = model.config
        if (
            tuple(config.dimensions) != specification.widths
            or config.input_policy != specification.input_policy
            or config.normalization_id != _normalization(specification)
        ):
            raise ValueError("El candidato y la identidad semántica de entradas no coinciden")
        self.specification = specification
        self.native = native
        self.model = model
        self._input_contract = specification.fingerprint()

    def from_corpus(self, raw_batch, *, refinements=1):
        """Validar el lote NumPy antes de convertirlo, sin consultar campos de objetivo."""
        return self.forward(
            validated_cpu_batch(raw_batch, self.specification), refinements=refinements
        )

    def forward(self, batch, *, refinements=1):
        if (
            not isinstance(batch, CPUDecisionBatch)
            or batch.input_contract_id != self._input_contract
        ):
            raise ValueError("El lote CPU pertenece a otro contrato de entrada")
        if type(refinements) is not int or refinements not in (1, 2, 4):
            raise ValueError("El refinamiento candidato admite K = 1, 2 o 4")
        parameter = self.model.named_parameters()["head_weight"]
        # El validador usa contadores de versión. El núcleo conserva el modo del llamante.
        with torch.inference_mode(False), torch.no_grad():
            tensors = DecisionBatch.from_validated(
                batch, device=parameter.device, dtype=parameter.dtype
            )
        inputs = self.native.CandidateInputs(
            *(tensors.inputs[name] for name in MODALITIES), tensors.presence
        )
        result = self.model.forward(inputs, self.model.empty_memory(), refinements)
        return CandidatePrediction(result.quantiles, result.quantiles[:, 2], result)

    def identity(self):
        """Huella fuerte en las fronteras de guardado/carga, no en cada forward."""
        config = self.model.config
        parameter = self.model.named_parameters()["head_weight"]
        return dict(
            schema_version=1,
            kind="native_gru_input_adapter",
            input_specification=self.specification.identity(),
            configuration={name: getattr(config, name) for name in _CONFIG_FIELDS},
            parameters_sha256=self.model.parameter_fingerprint(),
            representation_id=self.model.representation_id(),
            dtype=str(parameter.dtype),
            training=self.model.training,
            torch_version=str(torch.__version__),
            native_binary_sha256=self.native.binary_sha256,
            implementation={
                name: hashlib.sha256(
                    Path(importlib.import_module(name).__file__).read_bytes()
                ).hexdigest()
                for name in _SOURCE_MODULES
            },
        )

    def transfer_strict_parameters(self, source):
        """Copiar parámetros compatibles y conservar el codec histórico del destino."""
        if (
            type(source) is not CandidateInputAdapter
            or source.specification.input_policy != STRICT_INPUTS
            or self.specification.input_policy != HISTORICAL_MASKED
            or _normalization(source.specification) != _normalization(self.specification)
            or source.native is not self.native
        ):
            raise ValueError("El traslado necesita representaciones semánticas compatibles")
        source_identity = source.identity()
        destination_before = self.model.parameter_fingerprint()
        names = self.model.transfer_strict_parameters(source.model)
        return dict(
            schema_version=1,
            source_identity=source_identity,
            source_parameters=source_identity["parameters_sha256"],
            destination_before=destination_before,
            destination_parameters=self.model.parameter_fingerprint(),
            destination_codec=self.model.representation_id(),
            copied=names,
            initialized={"fusion_weight[:,640:645]": "zero"},
            retained=["feature_projection", "key_projection"],
        )

    def export_state(self):
        identity = self.identity()
        archive = self.model.save_state()
        return dict(
            schema_version=1,
            identity=identity,
            archive=archive,
            archive_sha256=hashlib.sha256(archive).hexdigest(),
        )

    @classmethod
    def restore(cls, payload, specification, *, device="cpu", native_path=None):
        _specification(specification)
        if (
            type(payload) is not dict
            or set(payload) != {"schema_version", "identity", "archive", "archive_sha256"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or type(payload["archive"]) is not bytes
            or not 0 < len(payload["archive"]) <= _MAX_ARCHIVE_BYTES
            or hashlib.sha256(payload["archive"]).hexdigest() != payload["archive_sha256"]
        ):
            raise ValueError("El archivo del adaptador no conserva su contrato o huella")
        native = _native(native_path)
        model = native.Candidate.load_state(
            payload["archive"], str(device), specification.input_policy
        )
        result = object.__new__(cls)
        result._bind(specification, native, model)
        if canonical(result.identity()) != canonical(payload["identity"]):
            raise ValueError("La identidad recuperada del candidato es incompatible")
        return result
