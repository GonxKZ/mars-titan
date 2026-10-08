"""Codificador de precios con atención causal y posiciones explícitas por ventana."""

import math

import torch
from torch import nn

_MAX_BATCH = 256
_MAX_ATTENTION_ELEMENTS = 2**24


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


class CompactPriceTransformer(nn.Module):
    """Proyectar una ventana completa sin memoria ni estado entre llamadas.

    La atención y la FFN no usan dropout. La referencia multimodal conserva
    su regularización en la fusión. Los límites de lote no seleccionan datos.
    """

    max_batch = _MAX_BATCH

    def __init__(
        self, input_size, *, context=64, hidden_size=64, layers=1, heads=4, feedforward_multiplier=2
    ):
        super().__init__()
        options = transformer_options(
            dict(heads=heads, feedforward_multiplier=feedforward_multiplier)
        )
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
            max_batch=_MAX_BATCH,
            max_attention_elements=_MAX_ATTENTION_ELEMENTS,
            dtypes=("float32", "float64"),
        )

    def encode_sequence(self, prices):
        """Devolver todos los tokens para contrastar el orden de información."""
        if (
            not isinstance(prices, torch.Tensor)
            or prices.layout != torch.strided
            or prices.is_nested
        ):
            raise ValueError("La ventana de precios debe ser un tensor denso")
        settings = self._settings
        if prices.ndim != 3 or prices.shape[1:] != (settings["context"], settings["input_size"]):
            raise ValueError("La ventana de precios no tiene la forma esperada")
        batch = prices.shape[0]
        if not 1 <= batch <= _MAX_BATCH:
            raise ValueError("El lote de precios supera el presupuesto o está vacío")
        if (
            batch * settings["context"] ** 2 * settings["heads"] * settings["layers"]
            > _MAX_ATTENTION_ELEMENTS
        ):
            raise ValueError("La atención supera el presupuesto conjunto de lote y contexto")
        if (
            prices.dtype not in (torch.float32, torch.float64)
            or prices.dtype != self.projection.weight.dtype
        ):
            raise ValueError("La precisión debe ser float32 o float64 y coincidir con los pesos")
        if prices.device != self.projection.weight.device:
            raise ValueError("Los precios y los pesos deben compartir dispositivo")
        if torch.is_autocast_enabled(prices.device.type):
            raise ValueError("Esta referencia no admite autocast")
        if not torch.isfinite(prices).all():
            raise ValueError("Los precios contienen valores no finitos")
        encoded = self.projection(prices) + self.positions.to(dtype=prices.dtype)
        for block in self.blocks:
            encoded = block(encoded, src_mask=self.causal_mask, is_causal=True)
        return self.norm(encoded)

    def forward(self, prices):
        return self.encode_sequence(prices)[:, -1]
