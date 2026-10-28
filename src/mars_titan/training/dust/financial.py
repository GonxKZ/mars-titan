"""Sitios afines del Transformer financiero escalar, sin pesos propios."""

import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.data.input_policy import MODALITIES
from mars_titan.data.price_windows import gate_price_window
from mars_titan.models.baselines.multimodal import SCALAR_HEAD, fuse_modalities
from mars_titan.models.baselines.transformer import (
    CompactPriceTransformer,
    validate_attention_budget,
)
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.frozen_financial import _module_settings


def finite(*values):
    """Rechazar también resultados intermedios que una máscara pudiera ocultar."""
    if values and not torch.stack([torch.isfinite(value).all() for value in values]).all().item():
        raise ValueError("El cálculo de Dust contiene NaN o infinito")


def linear_pullback(values, error):
    """Producto exterior y sesgo, sumados sobre filas y posiciones."""
    x, e = values.reshape(-1, values.shape[-1]), error.reshape(-1, error.shape[-1])
    if x.shape[0] != e.shape[0] or x.dtype != e.dtype or x.device != e.device:
        raise ValueError("Las entradas y errores del sitio afín no son compatibles")
    return e.T @ x, e.sum(dim=0)


def layer_norm_pullback(normalized, error):
    """Pullback de escala y sesgo de LayerNorm sobre su entrada ya normalizada."""
    if normalized.shape != error.shape:
        raise ValueError("La normalización y su error no tienen la misma forma")
    axes = tuple(range(error.ndim - 1))
    return (normalized * error).sum(dim=axes), error.sum(dim=axes)


@dataclass(frozen=True)
class _Site:
    kind: str
    inputs: torch.Tensor
    output: torch.Tensor
    weight: str
    bias: str
    part: int | None = None


@dataclass(frozen=True)
class _Attention:
    prefix: str
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    output: torch.Tensor
    last_only: bool


class FinancialDustAdapter:
    """Reutilizar el predictor admitido y exponer su composición para los probes.

    No es un nn.Module contenedor. Los parámetros conservan sus propietarios, nombres,
    almacenamiento y banderas. La ruta limpia llama a las funciones de producción.
    """

    def __init__(self, predictor):
        if type(predictor) is not FinancialPredictor:
            raise ValueError("Dust necesita el predictor financiero escalar explícito")
        config = predictor.config
        optional = (
            "quantile_routing",
            "gate_bias",
            "memory_stability",
            "presence_modulation",
            "memory_multiscale",
            "memory_plasticity",
            "memory_key_competition",
            "session_context",
            "sector_map",
            "regional_fusion",
            "regional_execution",
            "memory_bank_readout",
            "regional_variant",
            "laplace_scale_head",
            "auxiliary_horizons",
            "session_rank_inputs",
            "session_ic_objective",
            "opponent_heads",
        )
        if (
            config.variant != "transformer_direct"
            or config.head != SCALAR_HEAD
            or config.bank_policy != "disabled"
            or config.refinements != 1
            or config.memory_residual_layer_norm
            or config.memory_projections != "linear_v1"
            or predictor.local_control is not None
            or any(getattr(config, name) is not None for name in optional)
            or any(
                getattr(predictor, name) is not None
                for name in (
                    "mac",
                    "session_context",
                    "regional_fusion",
                    "regional_execution",
                    "regional_variant",
                    "presence_modulation",
                    "fusion_branch",
                    "output_objective",
                    "auxiliary_heads",
                )
            )
            or type(predictor.price_encoder) is not CompactPriceTransformer
            or type(predictor.head) is not nn.Linear
        ):
            raise ValueError(
                "La variante Dust inicial admite TransformerDirect escalar sin ampliaciones"
            )
        self.predictor = predictor
        self.device, self.dtype = predictor.head.weight.device, predictor.head.weight.dtype
        if str(self.device) not in {"cpu", "cuda:0"} or self.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("Dust requiere cpu o cuda:0 y precisión FP32 o FP64")
        self.hidden_size, self.layers = config.hidden_size, config.layers
        self.max_batch, self.context = config.max_batch, config.inputs.context
        self._configuration = canonical(config.identity())
        self._backbone = canonical(predictor.price_encoder.configuration)
        self._parameter_names = self._expected_parameters()
        if set(self._parameter_names) != set(dict(self.named_parameters())):
            raise ValueError("Los sitios Dust no cubren exactamente todos los parámetros")
        if any(not value.requires_grad for value in self.parameters()):
            raise ValueError(
                "Dust requiere el predictor completo entrenable, sin parámetros congelados"
            )
        self._structure = self._structure_signature()
        self.validate_model()

    def named_parameters(self):
        return self.predictor.named_parameters()

    def parameters(self):
        return self.predictor.parameters()

    def _expected_parameters(self):
        linear = ["price_encoder.projection", "head"]
        norms = ["price_encoder.norm"]
        packed = []
        for index in range(self.layers):
            prefix = f"price_encoder.blocks.{index}"
            linear.extend(
                (prefix + ".self_attn.out_proj", prefix + ".linear1", prefix + ".linear2")
            )
            norms.extend((prefix + ".norm1", prefix + ".norm2"))
            packed.extend(
                (prefix + ".self_attn.in_proj_weight", prefix + ".self_attn.in_proj_bias")
            )
        linear.extend(f"encoders.{name}.0" for name in MODALITIES[1:])
        linear.extend(f"fusion.{3 * index}" for index in range(self.layers))
        return [
            *(name + suffix for name in (*linear, *norms) for suffix in (".weight", ".bias")),
            *packed,
        ]

    def _structure_signature(self):
        from torch.nn.modules import module as torch_module

        if any(
            value
            for name, value in vars(torch_module).items()
            if name.startswith("_global_") and name.endswith("_hooks")
        ):
            raise ValueError("Dust no admite hooks globales")
        records = []
        for name, module in self.predictor.named_modules():
            if isinstance(module, nn.Dropout) and (
                type(module.p) not in (int, float) or module.p != 0 or module.inplace
            ):
                raise ValueError("El contrato Dust requiere Dropout con p=0 y sin inplace")
            for attribute, value in vars(module).items():
                if attribute.endswith("_hooks") and value:
                    allowed = (
                        attribute == "_load_state_dict_post_hooks"
                        and type(module) is FinancialPredictor
                        and list(value.values()) == [FinancialPredictor._after_load]
                    )
                    if not allowed:
                        raise ValueError("Dust no admite hooks adicionales en el predictor")
                if callable(value) and hasattr(type(module), attribute):
                    raise ValueError("Dust no admite métodos sustituidos en una instancia")
            methods = tuple(
                (key, id(value), id(getattr(value, "__code__", None)))
                for key, value in inspect.getmembers(type(module), inspect.isfunction)
            )
            records.append(
                (name, id(module), type(module), methods, canonical(_module_settings(module)))
            )
        storages = set()
        for name, value in (*self.named_parameters(), *self.predictor.named_buffers()):
            storage = (value.device, value.untyped_storage().data_ptr())
            if storage in storages:
                raise ValueError("Dust no admite alias entre parámetros o buffers")
            storages.add(storage)
            records.append(
                (
                    name,
                    id(value),
                    value.shape,
                    value.stride(),
                    value.storage_offset(),
                    value.dtype,
                    value.device,
                    value.requires_grad,
                    storage,
                )
            )
        return tuple(records)

    def validate_model(self):
        if (
            canonical(self.predictor.config.identity()) != self._configuration
            or canonical(self.predictor.price_encoder.configuration) != self._backbone
            or self._structure_signature() != self._structure
        ):
            raise ValueError("La arquitectura o los sitios Dust cambiaron después de su admisión")
        self.predictor.verify_parameter_identity()
        finite(*self.parameters(), *self.predictor.buffers())

    def identity(self):
        return dict(
            schema_version=1,
            kind="dust_financial_transformer_direct_scalar_v1",
            experimental=True,
            configuration=self.predictor.config.identity(),
            dtype=str(self.dtype),
            device=str(self.device),
            parameter_names=[name for name, _ in self.named_parameters()],
            parameter_shapes={name: list(value.shape) for name, value in self.named_parameters()},
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            states="independent_windows_without_publishing_prediction_state",
            trainable_scope="full_trainable_predictor_only",
            own_parameters=False,
        )

    def validate_inputs(self, inputs, presence):
        if not isinstance(inputs, Mapping) or set(inputs) != set(MODALITIES):
            raise ValueError("Dust requiere las cinco modalidades del predictor")
        prices = inputs["prices"]
        if not isinstance(prices, torch.Tensor) or prices.ndim != 3:
            raise ValueError("Los precios necesitan ventanas tensoriales por fila")
        rows = len(prices)
        if not 1 <= rows <= self.max_batch:
            raise ValueError("El presupuesto de filas Dust supera el lote físico del predictor")
        for name, value in inputs.items():
            width = self.predictor.config.inputs.dimensions[name]
            shape = (rows, self.context, width) if name == "prices" else (rows, width)
            if (
                not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or value.shape != shape
                or value.dtype != self.dtype
                or value.device != self.device
            ):
                raise ValueError("Las modalidades Dust no conservan forma, precisión y dispositivo")
        if presence is not None and (
            not isinstance(presence, torch.Tensor)
            or presence.dtype != torch.bool
            or presence.device != self.device
            or presence.shape != (rows, 5)
        ):
            raise ValueError("La presencia Dust necesita cinco booleanos por fila")
        if self.predictor.masked and presence is None:
            raise ValueError("La entrada histórica de Dust requiere presencia explícita")
        if any(torch.is_autocast_enabled(device) for device in ("cpu", "cuda")):
            raise ValueError("Dust no admite autocast ni precisión implícita")
        validate_attention_budget(
            rows,
            context=self.context,
            heads=self.predictor.price_encoder._settings["heads"],
            layers=self.layers,
            max_batch=self.max_batch,
        )
        finite(*inputs.values())
        return rows

    def forward(self, inputs, presence):
        """Ruta limpia común a BP y Dust con las funciones y pesos de producción."""
        self.validate_inputs(inputs, presence)
        self.predictor._check_parameters()
        model = self.predictor
        price, checks = model.price_encoder.encode_last(gate_price_window(inputs["prices"]))
        finite(*(value for value, _ in checks))
        token = fuse_modalities(
            price, inputs, model.encoders, model.fusion, presence if model.masked else None
        )
        result = model.head(token).squeeze(-1)
        finite(token, result)
        return result

    def _attention_output(self, q, k, v, *, last_only):
        heads = self.predictor.price_encoder._settings["heads"]
        batch, width = q.shape[0], self.hidden_size // heads
        projected = [value.reshape(batch, -1, heads, width).transpose(1, 2) for value in (q, k, v)]
        output = F.scaled_dot_product_attention(*projected, is_causal=not last_only)
        return output.transpose(1, 2).reshape(batch, -1, self.hidden_size)

    def _instrumented(self, inputs, presence, jitter, *, capture=False):
        self.validate_inputs(inputs, presence)
        self.predictor._check_parameters()
        jitter = {} if jitter is None else jitter
        if not isinstance(jitter, dict) or set(jitter) - self.site_names():
            raise ValueError("La perturbación no identifica un sitio Dust conocido")
        sites, attentions, checks = {}, [], []

        def output(name, value):
            checks.append(value)
            if name not in jitter:
                return value
            added = jitter[name]
            if (
                not isinstance(added, torch.Tensor)
                or added.shape != value.shape
                or added.dtype != value.dtype
                or added.device != value.device
            ):
                raise ValueError("La perturbación no conserva la forma del sitio Dust")
            result = value + added
            checks.append(result)
            return result

        def affine(name, module, values):
            result = module(values)
            if capture:
                sites[name] = _Site("linear", values, result, name + ".weight", name + ".bias")
            return output(name, result)

        def norm(name, module, values):
            result = module(values)
            if capture:
                normalized = F.layer_norm(values, module.normalized_shape, eps=module.eps)
                sites[name] = _Site("norm", normalized, result, name + ".weight", name + ".bias")
                checks.append(normalized)
            return output(name, result)

        encoder = self.predictor.price_encoder
        prices = gate_price_window(inputs["prices"])
        encoded = affine("price_encoder.projection", encoder.projection, prices)
        encoded = encoded + encoder.positions.to(dtype=prices.dtype)
        hidden = self.hidden_size
        for index, block in enumerate(encoder.blocks):
            prefix = f"price_encoder.blocks.{index}"
            last = index == self.layers - 1
            normalized = norm(prefix + ".norm1", block.norm1, encoded)
            weight, bias = block.self_attn.in_proj_weight, block.self_attn.in_proj_bias
            if last:
                query_input = normalized[:, -1:]
                q = F.linear(query_input, weight[:hidden], bias[:hidden])
                k, v = F.linear(normalized, weight[hidden:], bias[hidden:]).split(hidden, dim=-1)
            else:
                query_input = normalized
                q, k, v = F.linear(normalized, weight, bias).split(hidden, dim=-1)
            raw = []
            for part, (kind, value) in enumerate(zip(("q", "k", "v"), (q, k, v), strict=True)):
                name = prefix + "." + kind
                if capture:
                    sites[name] = _Site(
                        "packed",
                        query_input if kind == "q" else normalized,
                        value,
                        prefix + ".self_attn.in_proj_weight",
                        prefix + ".self_attn.in_proj_bias",
                        part,
                    )
                raw.append(output(name, value))
            attended = self._attention_output(*raw, last_only=last)
            if capture:
                attentions.append(_Attention(prefix, q, k, v, attended, last))
            attended = output(prefix + ".attention_output", attended)
            residual = encoded[:, -1:] if last else encoded
            token = residual + affine(
                prefix + ".self_attn.out_proj", block.self_attn.out_proj, attended
            )
            inner = norm(prefix + ".norm2", block.norm2, token)
            inner = block.activation(affine(prefix + ".linear1", block.linear1, inner))
            encoded = token + affine(prefix + ".linear2", block.linear2, inner)
        price = norm("price_encoder.norm", encoder.norm, encoded)[:, 0]
        representations = [price]
        for index, name in enumerate(MODALITIES[1:], 1):
            linear, activation = self.predictor.encoders[name]
            projected = activation(affine(f"encoders.{name}.0", linear, inputs[name]))
            if self.predictor.masked:
                projected = projected * presence[:, index : index + 1]
            representations.append(projected)
        if self.predictor.masked:
            representations.append(presence.to(dtype=self.dtype))
        token = torch.cat(representations, dim=-1)
        for index in range(self.layers):
            offset = 3 * index
            linear = self.predictor.fusion[offset]
            activation, dropout = (
                self.predictor.fusion[offset + 1],
                self.predictor.fusion[offset + 2],
            )
            token = dropout(activation(affine(f"fusion.{offset}", linear, token)))
        prediction = affine("head", self.predictor.head, token).squeeze(-1)
        finite(*checks, prediction)
        return prediction, sites, attentions

    def instrumented_forward(self, inputs, presence, jitter=None):
        """Misma composición con sitios explícitos y sin modificar módulos compartidos."""
        return self._instrumented(inputs, presence, jitter)[0]

    def site_names(self):
        names = {
            name.removesuffix(".weight")
            for name in self._parameter_names
            if name.endswith(".weight")
        }
        for index in range(self.layers):
            prefix = f"price_encoder.blocks.{index}"
            names.update(prefix + "." + kind for kind in ("q", "k", "v", "attention_output"))
        return names
