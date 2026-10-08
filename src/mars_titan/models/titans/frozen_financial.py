"""Consumir el predictor y su lector con parámetros compartidos congelados."""

import hashlib
import importlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import canonical
from .episodic_readout import EpisodicReadout, apply_episodic_readout
from .financial import FinancialPredictor, _tensor_digest
from .local_control import MACProjectionControl

_TITANS_MODULES = (
    "config",
    "state",
    "neural_memory",
    "mac",
    "financial",
    "financial_inputs",
    "financial_blocks",
    "local_control",
    "episodic_readout",
    "episodic_snapshot",
    "frozen_financial",
)
_SOURCE_MODULES = tuple(f"mars_titan.models.titans.{name}" for name in _TITANS_MODULES) + (
    "mars_titan.models.baselines.transformer",
    "mars_titan.models.baselines.multimodal",
    "mars_titan.cm.numerical_radius",
    "mars_titan.data.input_policy",
    "mars_titan.training.cohort_contract",
    "mars_titan.training.corpus_inputs",
    "mars_titan.data.audited_prices",
    "mars_titan.data.temporal",
)


def _implementation():
    return {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        for name in _SOURCE_MODULES
    }


def _numerics():
    return dict(
        torch_version=str(torch.__version__),
        matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        sdpa_math=torch.backends.cuda.math_sdp_enabled(),
        sdpa_flash=torch.backends.cuda.flash_sdp_enabled(),
        sdpa_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
        sdpa_cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
        mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
        cpu_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        default_dtype=str(torch.get_default_dtype()),
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def _execution_signature(models):
    from torch.nn.modules import module as torch_module

    if torch.backends.mha.get_fastpath_enabled():
        raise ValueError("El consumidor exige fastpath=False declarado antes de construirlo")
    if torch.is_inference_mode_enabled() or any(
        torch.is_autocast_enabled(device) for device in ("cpu", "cuda")
    ):
        raise ValueError("El consumidor congelado necesita precisión explícita y no_grad")
    if any(
        value
        for name, value in vars(torch_module).items()
        if name.startswith("_global_") and name.endswith("_hooks")
    ):
        raise ValueError("El consumidor no admite hooks globales")
    signatures, modes = [], {}
    for prefix, model in models:
        for name, module in model.named_modules():
            kind = type(module)
            origin = inspect.getmodule(kind)
            if (
                not (kind.__module__.startswith("torch.nn.") or kind.__module__ in _SOURCE_MODULES)
                or origin is None
                or getattr(origin, kind.__name__, None) is not kind
                or module.training
            ):
                raise ValueError("Todos los submódulos deben ser conocidos y permanecer en eval")
            allowed = (
                FinancialPredictor._after_load
                if type(module) is FinancialPredictor
                else MACProjectionControl._after_load
                if type(module) is MACProjectionControl
                else None
            )
            for attribute, hooks in vars(module).items():
                if attribute.endswith("_hooks") and hooks:
                    if attribute != "_load_state_dict_post_hooks" or list(hooks.values()) != [
                        allowed
                    ]:
                        raise ValueError("El consumidor no admite hooks adicionales")
                if callable(hooks) and hasattr(kind, attribute):
                    raise ValueError("El consumidor no admite métodos sustituidos en una instancia")
            methods = tuple(
                (key, id(value), id(getattr(value, "__code__", None)))
                for key, value in inspect.getmembers(kind, inspect.isfunction)
            )
            qualified = prefix + "/" + name
            modes[qualified] = dict(type=kind.__module__ + "." + kind.__qualname__, training=False)
            signatures.append((qualified, id(module), methods))
        for name, value in (*model.named_parameters(), *model.named_buffers()):
            if value.requires_grad or value.grad_fn is not None or value.grad is not None:
                raise ValueError("Los parámetros y buffers compartidos deben estar congelados")
            signatures.append(
                (
                    prefix,
                    name,
                    id(value),
                    value._version,
                    value.dtype,
                    value.device,
                    value.shape,
                    value.stride(),
                    value.storage_offset(),
                    value.untyped_storage().data_ptr(),
                    value.untyped_storage().nbytes(),
                )
            )
    return tuple(signatures), modes


@dataclass(frozen=True)
class FrozenPreparation:
    point_predictions: torch.Tensor | None
    next_state: object
    readout: object
    local_control: object


class FrozenFinancialConsumer:
    """Una preparación MAC por bloque y K lecturas posteriores con la misma cabeza.

    verify comprueba bytes en fronteras de sesión y recuperación. prepare revisa
    versiones, dispositivos, modos y métodos sin copiar los parámetros a CPU.
    Un cambio mediante .data puede eludir el contador, por eso no se sustituye
    la comprobación fuerte por el control barato durante una sesión.
    """

    def __init__(self, predictor, *, readout=None):
        if type(predictor) is not FinancialPredictor or (
            readout is not None and type(readout) is not EpisodicReadout
        ):
            raise ValueError("El consumidor necesita el predictor y lector identificados")
        if readout is not None and (
            readout.config.hidden_size != predictor.config.hidden_size
            or readout.step_logit.device != predictor.head.weight.device
            or readout.step_logit.dtype != predictor.head.weight.dtype
        ):
            raise ValueError(
                "El lector y predictor no comparten dimensión, precisión y dispositivo"
            )
        self.predictor, self.readout = predictor, readout
        self._models = (("predictor", predictor),) + ((("readout", readout),) if readout else ())
        self._signature, modes = _execution_signature(self._models)
        self._flags = _numerics()
        predictor.verify_parameter_identity()
        self._readout_id = self._readout_digest()
        self._code = _implementation()
        self._identity = dict(
            schema_version=1,
            recipe="frozen_financial_mac_then_episodic_v1",
            predictor=predictor.get_extra_state(),
            predictor_parameters_sha256=predictor._parameter_id,
            readout=readout.get_extra_state() if readout is not None else None,
            readout_parameters_sha256=self._readout_id,
            implementation=self._code,
            numerics=self._flags,
            modules=modes,
            dtype=str(predictor.head.weight.dtype),
            device=str(predictor.head.weight.device),
            shared_parameters="frozen_no_optimizer",
            memory_updates="once_per_observation",
            local_control_scope="mac_transition_before_episodic_readout",
        )
        self.model_id = hashlib.sha256(canonical(self._identity).encode()).hexdigest()

    def _readout_digest(self):
        if self.readout is None:
            return None
        values = {
            name: _tensor_digest(tensor)
            for name, tensor in (*self.readout.named_parameters(), *self.readout.named_buffers())
        }
        return hashlib.sha256(canonical(values).encode()).hexdigest()

    def identity(self):
        import json

        return json.loads(canonical(self._identity))

    def verify(self, *, strong=True):
        if type(strong) is not bool:
            raise ValueError("La verificación necesita un modo explícito")
        signature, _ = _execution_signature(self._models)
        if (
            signature != self._signature
            or _numerics() != self._flags
            or self.predictor.get_extra_state() != self._identity["predictor"]
            or (self.readout.get_extra_state() if self.readout else None)
            != self._identity["readout"]
        ):
            raise ValueError("El cálculo del consumidor cambió respecto de su identidad")
        if strong:
            self.predictor.verify_parameter_identity()
            if self._readout_digest() != self._readout_id or _implementation() != self._code:
                raise ValueError("El código o los pesos del lector cambiaron durante la sesión")

    def prepare(self, batch, state, *, context_id, snapshot=None, warmup=False, selection=None):
        self.verify(strong=False)
        if type(warmup) is not bool or len(set(batch.prediction_at)) != 1:
            raise ValueError("La preparación necesita un evento y corte únicos")
        if warmup and snapshot is not None:
            raise ValueError("El calentamiento no consulta un banco episódico")
        options = {}
        if self.predictor.local_control is not None:
            options.update(control_selection=selection, control_context_id=context_id)
        elif selection is not None:
            raise ValueError("El predictor sin C no admite un plan de medición")
        prepared = self.predictor.prepare(batch, state, differentiable=False, **options)
        if warmup:
            return FrozenPreparation(None, prepared.next_state, None, prepared.local_control)
        result = apply_episodic_readout(
            prepared,
            self.predictor.head,
            self.readout,
            snapshot=snapshot,
            context_id=context_id,
            cutoff=batch.prediction_at[0],
            differentiable=False,
        )
        return FrozenPreparation(
            result.point_predictions, prepared.next_state, result.readout, prepared.local_control
        )
