"""Codificador causal de ventana con bloque KronLSTM completo y caché explícita.

Adaptado de fla/layers/kronlstm.py, fla/models/kronlstm/modeling_kronlstm.py
y fla/modules/mlp.py en NX-AI/KronLSTM 3d800c9a.
Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li.
Licencia MIT conservada en el paquete. Entrada, escala y backend propios.
QKV acumula en FP64 con entrada y pesos FP32 en CPU y devuelve FP32.
CUDA mantiene la proyección nativa. Las puertas de olvido y objetivo usan FP32.
"""

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.models.baselines.transformer import MAX_BATCH_LIMIT, raise_nonfinite
from mars_titan.models.titans.state import check_finite, deferred_checks

from .rule import MAX_RECURRENT_ELEMENTS, kronlstm_scan

_UPSTREAM = "3d800c9a153a88ee9624fe6acf7f39a58017a17e"


class _QKVProjection(nn.Linear):
    """Acumular QKV en FP64 solo para la referencia FP32 de CPU."""

    def forward(self, value):
        if (
            value.device.type == self.weight.device.type == "cpu"
            and value.dtype == self.weight.dtype == torch.float32
        ):
            bias = None if self.bias is None else self.bias.double()
            return F.linear(value.double(), self.weight.double(), bias).to(value.dtype)
        return super().forward(value)


class _RMSNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(size))

    @property
    def configuration(self):
        return dict(kind="kron_rms_norm", size=self.weight.shape[0], eps=self.eps)

    def forward(self, value):
        variance = value.square().mean(-1, keepdim=True)
        check_finite(variance, "La energía para RMSNorm")
        return value * torch.rsqrt(variance + self.eps) * self.weight


class _Block(nn.Module):
    def __init__(self, hidden, heads, intermediate, rule):
        super().__init__()
        self.hidden, self.heads, self.width, self.rule = hidden, heads, hidden // heads, rule
        self.attn_norm = _RMSNorm(hidden)
        self.qkv_proj = _QKVProjection(hidden, 3 * hidden, bias=False)
        self.qkv_conv1d = nn.Conv1d(3 * hidden, 3 * hidden, 4, groups=3 * hidden, bias=False)
        self.b_proj = nn.Linear(hidden, heads, bias=False)
        self.c_proj = nn.Linear(hidden, heads, bias=False)
        self.f_proj = nn.Sequential(
            nn.Linear(hidden, self.width, bias=False), nn.Linear(self.width, hidden, bias=False)
        )
        self.A_log = nn.Parameter(torch.empty(heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(hidden, dtype=torch.float32))
        self.g_proj = nn.Linear(hidden, hidden)
        self.o_norm = _RMSNorm(self.width)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_norm = _RMSNorm(hidden)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    @property
    def configuration(self):
        return dict(
            kind="kron_block",
            hidden=self.hidden,
            heads=self.heads,
            head_width=self.width,
            rule=self.rule,
            conv_size=self.qkv_conv1d.weight.shape[-1],
            normalize_qk=True,
            normalize_v=True,
            decay_gate="vector",
            decouple_erase=True,
            gate_precision="float32",
            cpu_float32_qkv_accumulation="float64",
            activation="swiglu",
        )

    def reset_gates(self):
        with torch.no_grad():
            self.A_log.uniform_(0, 16).log_()
            dt = torch.exp(torch.empty_like(self.dt_bias).uniform_(math.log(0.001), math.log(0.1)))
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            self.g_proj.weight.zero_()
            self.g_proj.bias.fill_(3.0)
        for name, parameter in self.named_parameters():
            if name in {"A_log", "dt_bias"} or name.endswith("bias") or "norm.weight" in name:
                parameter._no_weight_decay = True

    def forward(self, encoded, state):
        batch, steps, _ = encoded.shape
        check_finite(self.A_log, "El parámetro A_log de KronLSTM")
        check_finite(self.dt_bias, "El parámetro dt_bias de KronLSTM")
        normalized = self.attn_norm(encoded)
        qkv = self.qkv_proj(normalized)
        check_finite(qkv, "La proyección QKV de KronLSTM")
        history = qkv.new_zeros(batch, 3 * self.hidden, 4) if state is None else state["conv_state"]
        history = torch.cat((history, qkv.transpose(1, 2)), dim=-1)
        convolved = F.conv1d(history, self.qkv_conv1d.weight, groups=3 * self.hidden)[:, :, 1:]
        check_finite(convolved, "La convolución causal KronLSTM")
        q, k, v = F.silu(convolved.transpose(1, 2)).chunk(3, dim=-1)
        q, k, v = (
            F.normalize(value.reshape(batch, steps, self.heads, self.width), dim=-1)
            for value in (q, k, v)
        )
        beta_projection = self.b_proj(normalized)
        check_finite(beta_projection, "La proyección beta KronLSTM")
        beta = beta_projection.sigmoid()
        decay_projection = self.f_proj(normalized).float()
        target_projection = self.c_proj(normalized).float()
        check_finite(decay_projection, "La proyección de olvido KronLSTM")
        check_finite(target_projection, "La proyección objetivo KronLSTM")
        decay = F.softplus(decay_projection + self.dt_bias.float())
        decay = -self.A_log.float().exp()[None, None, :, None] * decay.reshape(
            batch, steps, self.heads, self.width
        )
        target = target_projection.sigmoid()
        output, recurrent = kronlstm_scan(
            q,
            k,
            v,
            decay,
            beta,
            target_gate=target,
            initial_state=None if state is None else state["recurrent_state"],
            rule=self.rule,
        )
        output_gate = self.g_proj(normalized).reshape(batch, steps, self.heads, self.width)
        check_finite(output_gate, "La puerta de salida KronLSTM")
        mixed = self.o_norm(output) * output_gate.sigmoid()
        attended = self.o_proj(mixed.reshape(batch, steps, self.hidden))
        residual = encoded + attended
        check_finite(residual, "El residual de atención KronLSTM")
        mlp_input = self.mlp_norm(residual)
        gate, up = self.gate_proj(mlp_input), self.up_proj(mlp_input)
        activated = F.silu(gate) * up
        check_finite(activated, "La activación SwiGLU KronLSTM")
        result = residual + self.down_proj(activated)
        check_finite(result, "El residual FFN KronLSTM")
        return result, dict(recurrent_state=recurrent, conv_state=history[:, :, -4:].clone())


class CompactPriceKronLSTM(nn.Module):
    """Variante no canónica con estado nuevo en cada ventana completa.

    encode_steps/encode_step reciben y devuelven caché sin mutarla. La caché solo
    corresponde a pesos fijos: su huella se comprueba al usarla, exportarla o
    restaurarla. Esa comprobación copia pesos al host únicamente en la API con
    caché. encode_last/encode_sequence no calculan hashes ni conservan estado.
    """

    def __init__(
        self,
        input_size,
        *,
        context=64,
        hidden_size=64,
        layers=1,
        heads=4,
        max_batch=256,
        rule="scalar",
    ):
        super().__init__()
        if (
            type(input_size) is not int
            or not 1 <= input_size <= 2048
            or type(context) is not int
            or not 2 <= context <= 512
            or type(hidden_size) is not int
            or hidden_size not in {32, 64, 128}
            or type(layers) is not int
            or layers not in {1, 2}
            or type(heads) is not int
            or heads not in {1, 2, 4, 8}
            or hidden_size % heads
            or type(max_batch) is not int
            or not 1 <= max_batch <= MAX_BATCH_LIMIT
            or type(rule) is not str
            or rule not in {"scalar", "vector"}
        ):
            raise ValueError("La arquitectura o regla del codificador KronLSTM no es válida")
        intermediate = 256 * ((int(hidden_size * 8 / 3) + 255) // 256)
        self._settings = dict(
            input_size=input_size,
            context=context,
            hidden_size=hidden_size,
            layers=layers,
            heads=heads,
            rule=rule,
            intermediate_size=intermediate,
        )
        self.max_batch = max_batch
        self.projection = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            [_Block(hidden_size, heads, intermediate, rule) for _ in range(layers)]
        )
        self.norm = _RMSNorm(hidden_size)
        self._initialize(layers)

    def _initialize(self, layers):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.normal_(block.o_proj.weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.normal_(block.down_proj.weight, std=0.02 / math.sqrt(2 * layers))
            block.reset_gates()

    @property
    def configuration(self):
        """Devolver una identidad independiente para la escala y semántica completas."""
        return dict(
            schema_version=1,
            kind="compact_price_kronlstm",
            **self._settings,
            upstream_commit=_UPSTREAM,
            causal=True,
            state_policy="reset_per_window",
            aggregation="last_token",
            backend="pytorch_reference_cpu_qkv64_v1",
            cpu_float32_qkv_accumulation="float64",
            conv_size=4,
            normalize_v=True,
            normalize_qk=True,
            decouple_erase=True,
            allow_neg_eigval=False,
            decay_gate="vector",
            activation="swiglu",
            normalization="rms_norm",
            norm_eps=1e-5,
            gate_precision="float32",
            v_norm_floor=1e-18,
            output_gate_bias=3.0,
            max_batch=self.max_batch,
            max_recurrent_elements=MAX_RECURRENT_ELEMENTS,
            dtypes=["float32", "float64"],
        )

    def get_extra_state(self):
        return self.configuration

    def set_extra_state(self, state):
        try:
            matches = json.dumps(state, sort_keys=True, allow_nan=False) == json.dumps(
                self.configuration, sort_keys=True, allow_nan=False
            )
        except (TypeError, ValueError) as error:
            raise ValueError("El contrato KronLSTM no es válido") from error
        if not matches:
            raise ValueError("El contrato KronLSTM no coincide con la configuración")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        self.validate_payload(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def validate_payload(self, state_dict, prefix=""):
        """Comprobar el subárbol completo antes de copiar el primer parámetro.

        El prefijo permite al predictor padre aplicar el mismo preflight antes
        de cargar sus otros componentes. Las ubicaciones pueden diferir entre
        checkpoint y destino, pero no se convierte la precisión de sus pesos.
        """
        if not isinstance(state_dict, Mapping) or type(prefix) is not str:
            raise ValueError("El checkpoint KronLSTM no contiene un mapa de pesos válido")
        self.set_extra_state(state_dict.get(prefix + "_extra_state"))
        expected = self.state_dict()
        actual = {name.removeprefix(prefix) for name in state_dict if name.startswith(prefix)}
        if actual != set(expected):
            raise ValueError("El checkpoint no contiene exactamente los pesos de KronLSTM")
        checks = {}
        for name, installed in expected.items():
            if not isinstance(installed, torch.Tensor):
                continue
            incoming = state_dict[prefix + name]
            if (
                not isinstance(incoming, torch.Tensor)
                or incoming.layout != torch.strided
                or incoming.is_nested
                or incoming.shape != installed.shape
                or incoming.dtype != installed.dtype
                or incoming.device.type not in {"cpu", "cuda"}
            ):
                raise ValueError("El peso KronLSTM no conserva tipo, forma o precisión")
            checks.setdefault(incoming.device, []).append(
                (incoming, "El checkpoint KronLSTM contiene NaN o infinito")
            )
        for group in checks.values():
            raise_nonfinite(group)

    def _prices(self, prices, *, complete):
        settings = self._settings
        if (
            not isinstance(prices, torch.Tensor)
            or prices.layout != torch.strided
            or prices.is_nested
            or prices.ndim != 3
            or prices.device.type not in {"cpu", "cuda"}
            or prices.shape[2] != settings["input_size"]
            or not 1 <= prices.shape[0] <= self.max_batch
            or not 1 <= prices.shape[1] <= settings["context"]
            or (complete and prices.shape[1] != settings["context"])
        ):
            raise ValueError("La ventana, forma o lote de precios KronLSTM no es válido")
        if (
            prices.dtype not in (torch.float32, torch.float64)
            or prices.dtype != self.projection.weight.dtype
        ):
            raise ValueError("La precisión de precios y pesos debe coincidir en float32 o float64")
        if prices.device != self.projection.weight.device:
            raise ValueError("Los precios y pesos deben compartir dispositivo")
        if torch.is_autocast_enabled(prices.device.type):
            raise ValueError("La referencia KronLSTM no admite autocast")
        elements = (
            prices.shape[0]
            * prices.shape[1]
            * settings["layers"]
            * settings["hidden_size"] ** 2
            // settings["heads"]
        )
        if elements > MAX_RECURRENT_ELEMENTS:
            raise ValueError("La ventana supera el presupuesto lógico de estados KronLSTM")
        raise_nonfinite([(prices, "Los precios contienen NaN o infinito")])

    def _run(self, prices, state=None):
        with deferred_checks():
            encoded = self.projection(prices)
            check_finite(encoded, "La proyección de precios KronLSTM")
            layers = []
            for index, block in enumerate(self.blocks):
                encoded, updated = block(encoded, None if state is None else state["layers"][index])
                layers.append(updated)
            encoded = self.norm(encoded)
            check_finite(encoded, "La representación KronLSTM")
        return encoded, tuple(layers)

    def encode_sequence(self, prices):
        """Codificar una ventana completa sin estado persistente entre llamadas."""
        self._prices(prices, complete=True)
        return self._run(prices)[0]

    def encode_last(self, prices):
        """Entregar la representación final y comprobaciones compatibles con la fusión."""
        sequence = self.encode_sequence(prices)
        result = sequence[:, -1]
        return result, [
            (prices, "Los precios contienen NaN o infinito"),
            (result, "La representación KronLSTM contiene NaN o infinito"),
        ]

    def forward(self, prices):
        return self.encode_last(prices)[0]

    def _installed_settings(self):
        records = {}
        for name, module in self.named_modules():
            if module is self:
                continue
            if type(module) in {_Block, _RMSNorm}:
                settings = module.configuration
            elif isinstance(module, nn.Conv1d):
                settings = {
                    field: getattr(module, field)
                    for field in (
                        "in_channels",
                        "out_channels",
                        "kernel_size",
                        "stride",
                        "padding",
                        "dilation",
                        "groups",
                        "padding_mode",
                    )
                }
                settings["bias"] = module.bias is not None
            elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
                settings = dict(
                    normalized_shape=module.normalized_shape,
                    eps=module.eps,
                    elementwise_affine=module.elementwise_affine,
                )
            elif isinstance(module, nn.Linear):
                settings = dict(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                )
            else:
                settings = {}
            records[name] = dict(
                kind=type(module).__module__ + "." + type(module).__qualname__, settings=settings
            )
        return records

    def _weights_sha(self):
        binding = dict(configuration=self.configuration, modules=self._installed_settings())
        digest = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode())
        for name, value in self.named_parameters():
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _cache_capacity(self, batch):
        settings = self._settings
        elements = (
            batch
            * settings["layers"]
            * (settings["hidden_size"] ** 2 // settings["heads"] + 12 * settings["hidden_size"])
        )
        if not 1 <= batch <= self.max_batch or elements > MAX_RECURRENT_ELEMENTS:
            raise ValueError("La caché KronLSTM supera el presupuesto lógico de elementos")

    def _cache(self, state, batch):
        if not isinstance(state, dict) or set(state) != {
            "configuration",
            "weights_sha256",
            "cursor",
            "layers",
        }:
            raise ValueError("El estado de caché KronLSTM no tiene los campos válidos")
        self.set_extra_state(state["configuration"])
        cursor = state["cursor"]
        if type(cursor) is not int or not 0 <= cursor <= self._settings["context"]:
            raise ValueError("El cursor de caché KronLSTM no es válido")
        layers = state["layers"]
        if not isinstance(layers, (tuple, list)) or len(layers) != len(self.blocks):
            raise ValueError("Las capas de caché KronLSTM no coinciden")
        width = self._settings["hidden_size"] // self._settings["heads"]
        shapes = dict(
            recurrent_state=(batch, self._settings["heads"], width, width),
            conv_state=(batch, 3 * self._settings["hidden_size"], 4),
        )
        checks = []
        for layer in layers:
            if not isinstance(layer, dict) or set(layer) != set(shapes):
                raise ValueError("La capa de caché KronLSTM no tiene los campos válidos")
            for name, shape in shapes.items():
                value = layer[name]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.layout != torch.strided
                    or value.is_nested
                    or tuple(value.shape) != shape
                    or value.dtype != self.projection.weight.dtype
                    or value.device != self.projection.weight.device
                ):
                    raise ValueError("La forma, precisión o dispositivo de caché no coincide")
                checks.append((value, "El estado de caché contiene NaN o infinito"))
        raise_nonfinite(checks)
        if state["weights_sha256"] != self._weights_sha():
            raise ValueError("La caché corresponde a otros pesos o parámetros")

    def encode_steps(self, prices, state=None):
        """Continuar un prefijo de la ventana, únicamente con los pesos que lo crearon."""
        self._prices(prices, complete=False)
        self._cache_capacity(prices.shape[0])
        if state is not None:
            self._cache(state, prices.shape[0])
        cursor = 0 if state is None else state["cursor"]
        if cursor + prices.shape[1] > self._settings["context"]:
            raise ValueError("El cursor y el bloque superan la ventana declarada")
        weights = self._weights_sha() if state is None else state["weights_sha256"]
        output, layers = self._run(prices, state)
        return output, dict(
            configuration=self.configuration,
            weights_sha256=weights,
            cursor=cursor + prices.shape[1],
            layers=layers,
        )

    def encode_step(self, prices, state=None):
        """Codificar una observación [B,input_size] sin reconstruir un estado avanzado."""
        if not isinstance(prices, torch.Tensor) or prices.ndim != 2:
            raise ValueError("El paso de precios debe ser [B,input_size]")
        output, updated = self.encode_steps(prices[:, None], state)
        return output[:, 0], updated

    def export_state(self, state):
        """Exportar copias desligadas del grafo y vinculadas a la huella de pesos."""
        return self.restore_state(state)

    def restore_state(self, state):
        """Validar antes de copiar, sin mover ni convertir silenciosamente los tensores."""
        try:
            batch = state["layers"][0]["recurrent_state"].shape[0]
        except (TypeError, KeyError, IndexError, AttributeError) as error:
            raise ValueError("El estado de caché KronLSTM no es válido") from error
        self._cache_capacity(batch)
        self._cache(state, batch)
        return dict(
            configuration=deepcopy(state["configuration"]),
            weights_sha256=state["weights_sha256"],
            cursor=state["cursor"],
            layers=tuple(
                {name: value.detach().clone() for name, value in layer.items()}
                for layer in state["layers"]
            ),
        )
