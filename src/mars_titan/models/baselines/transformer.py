"""Codificador de precios con atención causal y posiciones explícitas por ventana."""

import json
import math

import torch
from torch import nn
from torch.nn import functional as F

# Lote por defecto y presupuesto de atención por defecto. Con estos valores el contrato
# del codificador queda igual que antes de admitir lotes mayores.
_MAX_BATCH = 256
_MAX_ATTENTION_ELEMENTS = 2**24
# Lote máximo admitido de forma explícita. El presupuesto de atención crece con el lote y
# conserva los mismos elementos por ventana que el valor por defecto.
MAX_BATCH_LIMIT = 8192
_ELEMENTS_PER_WINDOW = _MAX_ATTENTION_ELEMENTS // _MAX_BATCH


def attention_budget(max_batch=_MAX_BATCH):
    """Elementos de atención admitidos para un lote máximo, nunca menos que el presupuesto base."""
    if type(max_batch) is not int or not 1 <= max_batch <= MAX_BATCH_LIMIT:
        raise ValueError(f"El lote máximo debe ser un entero entre 1 y {MAX_BATCH_LIMIT}")
    return max(_MAX_ATTENTION_ELEMENTS, _ELEMENTS_PER_WINDOW * max_batch)


def first_nonfinite(checks):
    """Mensaje de la primera comprobación con NaN o infinito, con una sola sincronización.

    `checks` contiene pares (tensor, mensaje) en el orden en que el cálculo los comprobaba
    uno a uno. Cada `isfinite(...).all()` se queda en el dispositivo y solo la lista final
    se copia al host, en lugar de esperar a la GPU en cada comprobación.
    """
    if not checks:
        return None
    flags = torch.stack([torch.isfinite(value.detach()).all() for value, _ in checks])
    for finite, (_, message) in zip(flags.tolist(), checks, strict=True):
        if not finite:
            return message
    return None


def raise_nonfinite(checks):
    message = first_nonfinite(checks)
    if message is not None:
        raise ValueError(message)


def transformer_options(options):
    """Resolver las dos opciones acotadas sin inicializar parámetros."""
    if options is None:
        return dict(heads=4, feedforward_multiplier=2)
    if (
        not isinstance(options, dict)
        or set(options) != {"heads", "feedforward_multiplier"}
        or type(options["heads"]) is not int
        or options["heads"] not in {1, 2, 4, 8}
        or type(options["feedforward_multiplier"]) is not int
        or options["feedforward_multiplier"] not in {2, 4}
    ):
        raise ValueError("Las opciones del Transformer no pertenecen al contrato compacto")
    return dict(options)


def validate_attention_budget(batch, *, context, heads, layers, max_batch=_MAX_BATCH):
    """Comprobar lote y posiciones de atención sin construir el modelo ni leer datos."""
    elements = attention_budget(max_batch)
    if type(batch) is not int or not 1 <= batch <= max_batch:
        raise ValueError("El lote de precios supera el presupuesto o está vacío")
    if batch * context**2 * heads * layers > elements:
        raise ValueError("La atención supera el presupuesto conjunto de lote y contexto")


class CompactPriceTransformer(nn.Module):
    """Proyectar una ventana completa sin memoria ni estado entre llamadas.

    La atención y la FFN no usan dropout. La referencia multimodal conserva
    su regularización en la fusión. Los límites de lote no seleccionan datos.
    `max_batch` amplía el lote admitido y su presupuesto de atención. Con el valor por
    defecto el contrato y los checkpoints anteriores no cambian.

    `forward` solo necesita la representación del último token. Todas las capas salvo la
    última se calculan completas y la última solo proyecta claves y valores de toda la
    ventana: la consulta, la atención, la FFN y las normalizaciones posteriores se calculan
    para el último token. Con la máscara causal el último token atiende a toda la ventana,
    así que el resultado es el mismo que `encode_sequence(...)[:, -1]` en aritmética exacta.

    Las capas se evalúan con sus propios pesos y `scaled_dot_product_attention`, sin pasar
    por `TransformerEncoderLayer.forward`. En evaluación sin gradiente esa llamada puede
    tomar la ruta nativa de PyTorch, que en CUDA aproxima GELU con tanh. Así el resultado
    no depende del modo ni del indicador global de fastpath.
    """

    max_batch = _MAX_BATCH

    def __init__(
        self,
        input_size,
        *,
        context=64,
        hidden_size=64,
        layers=1,
        heads=4,
        feedforward_multiplier=2,
        max_batch=_MAX_BATCH,
    ):
        super().__init__()
        options = transformer_options(
            dict(heads=heads, feedforward_multiplier=feedforward_multiplier)
        )
        elements = attention_budget(max_batch)
        if (
            type(input_size) is not int
            or not 1 <= input_size <= 2048
            or type(context) is not int
            or not 2 <= context <= 512
            or type(hidden_size) is not int
            or hidden_size not in {32, 64, 128}
            or type(layers) is not int
            or layers not in {1, 2}
            or hidden_size % heads
        ):
            raise ValueError("La forma o arquitectura del Transformer no es válida")
        self.max_batch, self._max_attention_elements = max_batch, elements
        self._settings = dict(
            input_size=input_size,
            context=context,
            hidden_size=hidden_size,
            layers=layers,
            **options,
        )
        self.projection = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_size,
                    heads,
                    dim_feedforward=hidden_size * feedforward_multiplier,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        # Una base numérica fija permite restaurar después de cambiar el dtype global.
        dtype, device = torch.float32, self.projection.weight.device
        positions = torch.arange(context, dtype=dtype, device=device).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=dtype, device=device)
            * (-math.log(10000.0) / hidden_size)
        )
        phases = positions * frequencies
        encoded = torch.empty(context, hidden_size, dtype=dtype, device=device)
        encoded[:, 0::2], encoded[:, 1::2] = phases.sin(), phases.cos()
        # Se reconstruyen desde el contrato y no se aceptan como pesos aprendidos.
        self.register_buffer("positions", encoded, persistent=False)
        # Las capas aplican la causalidad con `is_causal`. La máscara se conserva porque las
        # huellas de parámetros de Titans-MAC también recorren los buffers no persistentes.
        self.register_buffer(
            "causal_mask",
            torch.ones(context, context, dtype=torch.bool, device=device).triu(1),
            persistent=False,
        )

    @property
    def configuration(self):
        """Identificar también decisiones que no cambian la forma de los pesos."""
        return dict(
            schema_version=1,
            kind="compact_price_transformer",
            **self._settings,
            causal=True,
            position_encoding="sinusoidal_v1",
            position_basis_dtype="float32",
            normalization="pre_layer_norm",
            final_normalization="layer_norm",
            layer_norm_eps=1e-5,
            activation="gelu",
            dropout=0.0,
            aggregation="last_token",
            state_policy="reset_per_window",
            max_batch=self.max_batch,
            max_attention_elements=self._max_attention_elements,
            dtypes=("float32", "float64"),
        )

    def get_extra_state(self):
        """Guardar el contrato junto a los pesos, también dentro de otro módulo."""
        return self.configuration

    def set_extra_state(self, state):
        """Validar el contrato sin cambiar la arquitectura construida."""
        try:
            matches = json.dumps(state, sort_keys=True, allow_nan=False) == json.dumps(
                self.configuration, sort_keys=True, allow_nan=False
            )
        except (TypeError, ValueError) as error:
            raise ValueError("El contrato del codificador Transformer no es válido") from error
        if not matches:
            raise ValueError("El contrato del codificador Transformer no coincide")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # El hook se ejecuta antes de copiar pesos de sus hijos. No depende de strict.
        self.set_extra_state(state_dict.get(prefix + "_extra_state"))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _validated(self, prices):
        """Comprobar forma, presupuesto, precisión y dispositivo sin esperar a la GPU."""
        if (
            not isinstance(prices, torch.Tensor)
            or prices.layout != torch.strided
            or prices.is_nested
        ):
            raise ValueError("La ventana de precios debe ser un tensor denso")
        settings = self._settings
        if prices.ndim != 3 or prices.shape[1:] != (settings["context"], settings["input_size"]):
            raise ValueError("La ventana de precios no tiene la forma esperada")
        validate_attention_budget(
            prices.shape[0],
            context=settings["context"],
            heads=settings["heads"],
            layers=settings["layers"],
            max_batch=self.max_batch,
        )
        if (
            prices.dtype not in (torch.float32, torch.float64)
            or prices.dtype != self.projection.weight.dtype
        ):
            raise ValueError("La precisión debe ser float32 o float64 y coincidir con los pesos")
        if prices.device != self.projection.weight.device:
            raise ValueError("Los precios y los pesos deben compartir dispositivo")
        if torch.is_autocast_enabled(prices.device.type):
            raise ValueError("Esta referencia no admite autocast")
        return self.projection(prices) + self.positions.to(dtype=prices.dtype)

    def _layer(self, block, encoded, *, last_only):
        """Capa con normalización previa: x + SA(LN1(x)) y después y + FFN(LN2(y)).

        Con `last_only` las claves y valores cubren toda la ventana y el resto se calcula
        solo para el último token, al que la máscara causal no oculta ninguna clave.
        """
        attention, settings = block.self_attn, self._settings
        hidden, heads = settings["hidden_size"], settings["heads"]
        batch, width = encoded.shape[0], hidden // heads
        normalized = block.norm1(encoded)
        weight, bias = attention.in_proj_weight, attention.in_proj_bias
        if last_only:
            residual = encoded[:, -1:]
            query = F.linear(normalized[:, -1:], weight[:hidden], bias[:hidden])
            keys, values = F.linear(normalized, weight[hidden:], bias[hidden:]).split(
                hidden, dim=-1
            )
        else:
            residual = encoded
            query, keys, values = F.linear(normalized, weight, bias).split(hidden, dim=-1)
        query, keys, values = (
            value.view(batch, -1, heads, width).transpose(1, 2) for value in (query, keys, values)
        )
        attended = F.scaled_dot_product_attention(query, keys, values, is_causal=not last_only)
        attended = attended.transpose(1, 2).reshape(batch, -1, hidden)
        token = residual + attention.out_proj(attended)
        return token + block.linear2(block.activation(block.linear1(block.norm2(token))))

    def encode_sequence(self, prices):
        """Devolver todos los tokens para contrastar el orden de información."""
        encoded = self._validated(prices)
        raise_nonfinite([(prices, "Los precios contienen valores no finitos")])
        for block in self.blocks:
            encoded = self._layer(block, encoded, last_only=False)
        representation = self.norm(encoded)
        raise_nonfinite(
            [(representation, "La representación Transformer contiene valores no finitos")]
        )
        return representation

    def encode_last(self, prices):
        """Representación del último token y sus comprobaciones de finitud pendientes.

        El llamante decide cuándo comprobarlas con `raise_nonfinite`, de modo que una
        referencia multimodal reúne todas las de su forward en una sola sincronización.
        """
        encoded = self._validated(prices)
        *complete, last = self.blocks
        for block in complete:
            encoded = self._layer(block, encoded, last_only=False)
        representation = self.norm(self._layer(last, encoded, last_only=True))[:, 0]
        checks = [
            (prices, "Los precios contienen valores no finitos"),
            (representation, "La representación Transformer contiene valores no finitos"),
        ]
        return representation, checks

    def forward(self, prices):
        representation, checks = self.encode_last(prices)
        raise_nonfinite(checks)
        return representation
