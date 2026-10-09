"""Consumir la GRU nativa con parámetros congelados y una instantánea episódica por evento.

La GRU reinicia su estado oculto en cada ventana. K repite la consulta y el
refinamiento sobre la misma instantánea, nunca la GRU, el codec ni la admisión.
El estado de trabajo de cada predicción se descarta al terminar el bloque.
"""

import hashlib
import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from mars_titan.data.input_policy import MODALITIES
from mars_titan.models.titans import frozen_financial
from mars_titan.models.titans.config import canonical

from . import episode_codec
from .input_adapter import CandidateInputAdapter

_SOURCE_MODULES = (
    __name__,
    "mars_titan.models.candidate.input_adapter",
    "mars_titan.models.candidate.episode_codec",
)


def _implementation():
    return {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        for name in _SOURCE_MODULES
    }


def _numerics():
    # Mismos flags CUDA/CPU que el consumidor Titans y los del codec CPU.
    return {**frozen_financial._numerics(), **episode_codec._numerics()}


@dataclass(frozen=True)
class CandidatePreparation:
    point_predictions: torch.Tensor
    quantiles: torch.Tensor
    read_ids: torch.Tensor


class FrozenCandidateConsumer:
    """Referencia GRU con identidad propia. No sustituye a Titans ni cambia B."""

    def __init__(self, adapter, *, refinements=1):
        if type(adapter) is not CandidateInputAdapter:
            raise ValueError("El consumidor GRU necesita el adaptador identificado")
        if type(refinements) is not int or refinements not in (1, 2, 4):
            raise ValueError("El consumidor GRU admite K = 1, 2 o 4")
        if torch.is_inference_mode_enabled():
            raise ValueError("El consumidor GRU necesita no_grad explícito, no inference_mode")
        self.adapter, self.model, self.refinements = adapter, adapter.model, refinements
        self._check_frozen()
        parameter = self.model.named_parameters()["head_weight"]
        self.device, self.dtype = parameter.device, parameter.dtype
        self._flags = _numerics()
        self._code = _implementation()
        self._parameters = self.model.parameter_fingerprint()
        self._representation = self.model.representation_id()
        self._identity = dict(
            schema_version=1,
            recipe="frozen_candidate_gru_episodic_v1",
            adapter=adapter.identity(),
            refinements=refinements,
            numerics=self._flags,
            implementation=self._code,
            dtype=str(self.dtype),
            device=str(self.device),
            shared_parameters="frozen_no_optimizer",
            working_state="hidden_per_window_and_refinement_discarded",
            fast_weights="none",
            read="median_quantile_after_k_refinements",
        )
        self.model_id = hashlib.sha256(canonical(self._identity).encode()).hexdigest()

    @property
    def input_spec(self):
        return self.adapter.specification

    @property
    def max_batch(self):
        return self.model.config.max_batch

    def _check_frozen(self):
        if any(
            name in vars(self) for name, _ in inspect.getmembers(type(self), inspect.isfunction)
        ):
            raise ValueError("El consumidor GRU no admite métodos sustituidos en la instancia")
        if self.model.training or any(
            value.requires_grad or value.grad is not None
            for value in self.model.named_parameters().values()
        ):
            raise ValueError("La GRU compartida debe estar en eval y sin gradientes")

    def identity(self):
        return json.loads(canonical(self._identity))

    def verify(self, *, strong=True):
        if type(strong) is not bool:
            raise ValueError("La verificación necesita un modo explícito")
        self._check_frozen()
        if _numerics() != self._flags or self.model.representation_id() != self._representation:
            raise ValueError("El cálculo de la GRU cambió respecto de su identidad")
        if strong and (
            self.model.parameter_fingerprint() != self._parameters
            or _implementation() != self._code
        ):
            raise ValueError("Los pesos o el código de la GRU cambiaron durante la sesión")

    def memory(self, view=None):
        """Instantánea inmutable común a todos los bloques de un evento."""
        if view is None:
            return self.model.empty_memory()
        return self.model.snapshot(
            view["keys"].to(self.device),
            view["values"].to(self.device),
            view["returns"].to(self.device),
            view["ids"],
            self._representation,
        )

    def prepare(self, batch, memory):
        self.verify(strong=False)
        if len(set(batch.prediction_at)) != 1:
            raise ValueError("La preparación GRU necesita un único corte")
        inputs = self.adapter.native.CandidateInputs(
            *(batch.inputs[name] for name in MODALITIES), batch.presence
        )
        with torch.no_grad():
            result = self.model.forward(inputs, memory, self.refinements)
        quantiles = result.quantiles
        return CandidatePreparation(quantiles[:, 2], quantiles, result.read.ids)
