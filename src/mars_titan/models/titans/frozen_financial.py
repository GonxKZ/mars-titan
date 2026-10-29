"""Consumir el predictor y su lector con parámetros compartidos congelados."""

import hashlib
import importlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from mars_titan.models.quantile_head import QuantileHead
from mars_titan.models.quantile_routing import QuantileRouting

from .bank_readout import BankReadout
from .config import canonical
from .episodic_readout import EpisodicReadout, apply_episodic_readout
from .episodic_snapshot import EpisodeSnapshot
from .evidence_bridge import SharedEvidenceBridge
from .financial import FinancialPredictor, _tensor_digest
from .laplace_scale import LaplaceScale
from .local_control import MACProjectionControl
from .opponent_heads import OpponentSpread
from .output_objectives import reads_scalar_point
from .regional_execution import RegionalExecution
from .regional_fusion import RegionalFusion
from .regional_hierarchy import RegionalHierarchy
from .session_ic_objective import SessionScale

_TRANSFORMER_ACTIVATION = torch.nn.functional.gelu

_TITANS_MODULES = (
    "config",
    "state",
    "causal_convolution",
    "neural_memory",
    "mac",
    "financial",
    "financial_inputs",
    "financial_blocks",
    "presence_modulation",
    "regional_fusion",
    "regional_execution",
    "auxiliary_horizons",
    "bank_readout",
    "laplace_scale",
    "opponent_heads",
    "reliability_gates",
    "huber_memory",
    "scale_invariant_time",
    "differential_attention",
    "session_context",
    "local_control",
    "episodic_readout",
    "episodic_snapshot",
    "evidence_bridge",
    "bridge_intervention",
    "bridge_hierarchy",
    "bridge_recurrence",
    "bridge_stopping_record",
    "regional_mechanism_trace",
    "regional_episode_readout",
    "session_ranks",
    "session_ic_objective",
    "output_objectives",
    "frozen_financial",
)
_SOURCE_MODULES = tuple(f"mars_titan.models.titans.{name}" for name in _TITANS_MODULES) + (
    "mars_titan.models.baselines.transformer",
    "mars_titan.models.baselines.multimodal",
    "mars_titan.models.quantile_head",
    "mars_titan.models.quantile_routing",
    "mars_titan.cm.numerical_radius",
    "mars_titan.data.input_policy",
    "mars_titan.training.cohort_contract",
    "mars_titan.training.corpus_inputs",
    "mars_titan.data.audited_prices",
    "mars_titan.data.temporal",
)
_KRON_MODULES = (
    "mars_titan.models.kronlstm",
    "mars_titan.models.kronlstm.rule",
    "mars_titan.models.kronlstm.price_encoder",
    "mars_titan.models.kronlstm.precomputed",
)


def _implementation(*, hierarchy=False, kronlstm=False):
    sources = _SOURCE_MODULES + ((RegionalHierarchy.__module__,) if hierarchy else ())
    if kronlstm:
        sources += _KRON_MODULES
    return {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        for name in sources
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


def _module_settings(module):
    """Atributos del cálculo usado, sin serializar objetos ni recorrer __dict__."""
    nn = torch.nn
    fields, settings = (), {}
    if isinstance(module, nn.Linear):
        fields = ("in_features", "out_features")
        settings["bias"] = module.bias is not None
        if isinstance(module, QuantileHead) and module.routing is not None:
            if not isinstance(module.routing, QuantileRouting):
                raise ValueError("La cabeza congelada no conserva una ruta identificada")
            settings["quantile_routing"] = module.routing.identity()
    elif isinstance(module, nn.LayerNorm):
        fields = ("normalized_shape", "eps", "elementwise_affine")
        settings["bias"] = module.bias is not None
    elif isinstance(module, nn.RMSNorm):
        fields = ("normalized_shape", "eps", "elementwise_affine")
    elif isinstance(module, nn.Conv1d):
        fields = (
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "padding_mode",
        )
        settings["bias"] = module.bias is not None
    elif isinstance(module, nn.MultiheadAttention):
        fields = (
            "embed_dim",
            "num_heads",
            "head_dim",
            "kdim",
            "vdim",
            "dropout",
            "batch_first",
            "add_zero_attn",
        )
        settings.update(
            qkv_same_embed_dim=module._qkv_same_embed_dim,
            in_proj_bias=module.in_proj_bias is not None,
            bias_k=module.bias_k is not None,
            bias_v=module.bias_v is not None,
        )
    elif isinstance(module, nn.TransformerEncoderLayer):
        if module.activation is not _TRANSFORMER_ACTIVATION:
            raise ValueError("La activación del Transformer fue sustituida fuera de su contrato")
        fields = ("norm_first", "activation_relu_or_gelu")
        settings["activation"] = "torch.nn.functional.gelu"
    elif isinstance(module, nn.Dropout):
        fields = ("p", "inplace")
    elif isinstance(module, nn.GELU):
        fields = ("approximate",)
    elif isinstance(module, (nn.ReLU, nn.SiLU)):
        fields = ("inplace",)
    elif type(module) is FinancialPredictor:
        fields = ("masked",)
    elif type(module) in (SessionScale, LaplaceScale, OpponentSpread):
        settings["output_objective"] = module.config.identity()
    elif type(module) is RegionalFusion:
        fields = ("hidden_size", "max_batch")
        settings["regional_fusion"] = module.config.identity()
    elif type(module) in (RegionalExecution, RegionalHierarchy):
        fields = ("hidden_size", "max_batch")
        settings["regional_execution"] = module.config.identity()
    elif type(module) is BankReadout:
        fields = ("dim",)
        settings["bank_readout"] = module.config.identity()
    elif type(module) is SharedEvidenceBridge:
        settings["evidence_bridge"] = module.config.identity()
    elif type(module).__module__ in _KRON_MODULES:
        settings["kronlstm"] = module.configuration
    settings.update((name, getattr(module, name)) for name in fields)
    return settings


def _validate_intervention(readout):
    """Conciliar la geometría fija una vez, antes de sellar los buffers del lector."""
    if readout is None:
        return
    from .evidence_bridge import SharedEvidenceBridge

    config = readout.config.evidence_bridge
    bridge = readout.evidence_bridge
    physical = bridge.config if type(bridge) is SharedEvidenceBridge else None
    if not any(
        item is not None and (item.intervention is not None or item.hierarchy is not None)
        for item in (config, physical)
    ):
        return
    if (
        type(bridge) is not SharedEvidenceBridge
        or config is None
        or canonical(bridge.config.identity()) != canonical(config.identity())
    ):
        raise ValueError("El módulo no conserva la intervención de regiones y puentes declarada")
    names = (
        ("edges", "hierarchy_edges")
        if config.hierarchy is not None
        else ("edges", "active_regions")
    )
    for name in names:
        value = getattr(bridge, name, None)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bool
            or value.layout != torch.strided
            or value.device != readout.step_logit.device
        ):
            raise ValueError("Los puentes y regiones necesitan buffers booleanos compatibles")
    if config.hierarchy is not None and (
        bridge.edges.shape != (4, 4) or bridge.hierarchy_edges.shape != (7, 7)
    ):
        raise ValueError("La jerarquía no conserva cuatro vistas y siete nodos")
    bridge.validate_payload(bridge.state_dict())


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
    sources = _SOURCE_MODULES + (
        _KRON_MODULES
        if any(
            type(model) is FinancialPredictor and model.config.temporal_encoder is not None
            for _, model in models
        )
        else ()
    )
    # Las instancias de una clase comparten sus métodos. La caché vive solo en esta llamada.
    class_methods = {}
    for prefix, model in models:
        for name, module in model.named_modules():
            kind = type(module)
            origin = inspect.getmodule(kind)
            if (
                not (
                    kind.__module__.startswith("torch.nn.")
                    or kind.__module__ in sources
                    or kind is RegionalHierarchy
                )
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
            if kind not in class_methods:
                class_methods[kind] = tuple(
                    (key, id(value), id(getattr(value, "__code__", None)))
                    for key, value in inspect.getmembers(kind, inspect.isfunction)
                )
            methods = class_methods[kind]
            qualified = prefix + "/" + name
            settings = _module_settings(module)
            try:
                encoded_settings = canonical(settings)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Los atributos de la capa no tienen tipos finitos admitidos"
                ) from error
            modes[qualified] = dict(
                type=kind.__module__ + "." + kind.__qualname__, training=False, settings=settings
            )
            signatures.append((qualified, id(module), methods, encoded_settings))
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
    quantiles: torch.Tensor | None = None


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
        if predictor.config.session_context is not None:
            raise ValueError("Este consumidor no gestiona el ciclo de sesión del contexto PT4")
        if predictor.regional_execution is not None:
            raise ValueError(
                "El programa regional requiere el consumidor de mecanismos y su archivo"
            )
        if readout is not None and not reads_scalar_point(predictor.output_objective):
            raise ValueError("El lector necesita una integración propia del objetivo de salida")
        # Este consumidor aún no registra la fuente ni la configuración de C5 (#751).
        if predictor.cross_asset_attention is not None:
            raise ValueError("El consumidor congelado aún no registra la atención entre activos")
        # Tampoco registra la red holográfica de #783.
        if predictor.holographic_network is not None:
            raise ValueError("El consumidor congelado aún no registra la red holográfica")
        # Ni recibe la señal de fiabilidad madura de #791, que sus puertas necesitan.
        if predictor.reliability_gates is not None:
            raise ValueError("El consumidor congelado aún no recibe la fiabilidad madura")
        if readout is not None and (
            readout.config.hidden_size != predictor.config.hidden_size
            or readout.step_logit.device != predictor.head.weight.device
            or readout.step_logit.dtype != predictor.head.weight.dtype
        ):
            raise ValueError(
                "El lector y predictor no comparten dimensión, precisión y dispositivo"
            )
        self.predictor, self.readout = predictor, readout
        _validate_intervention(readout)
        self._models = (("predictor", predictor),) + ((("readout", readout),) if readout else ())
        self._signature, modes = _execution_signature(self._models)
        self._flags = _numerics()
        predictor.verify_parameter_identity()
        self._readout_id = self._readout_digest()
        self._code = _implementation(kronlstm=predictor.config.temporal_encoder is not None)
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
            if (
                self._readout_digest() != self._readout_id
                or _implementation(kronlstm=self.predictor.config.temporal_encoder is not None)
                != self._code
            ):
                raise ValueError("El código o los pesos del lector cambiaron durante la sesión")

    def prepare(
        self,
        batch,
        state,
        *,
        context_id,
        snapshot=None,
        warmup=False,
        selection=None,
        session_complete=False,
    ):
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
        if not warmup and type(self.readout) is EpisodicReadout:
            self.readout.preflight_node_updates(
                self.readout.config.refinements,
                snapshot.count if isinstance(snapshot, EpisodeSnapshot) else 0,
            )
        prepared = self.predictor.prepare(
            batch, state, differentiable=False, session_complete=session_complete, **options
        )
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
            result.point_predictions,
            prepared.next_state,
            result.readout,
            prepared.local_control,
            result.quantiles,
        )
