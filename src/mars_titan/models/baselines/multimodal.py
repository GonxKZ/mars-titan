"""Referencias científicas configurables, separadas de las sondas de coste."""

import math

import torch
from torch import nn

from .dlinear import DLinear

MODALITIES = ("prices", "news", "charts", "fundamentals", "macro")
_RECURRENT = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


def validate_architecture(hidden_size, layers, dropout):
    if (
        type(hidden_size) is not int
        or hidden_size not in {32, 64, 128}
        or type(layers) is not int
        or layers not in {1, 2}
        or type(dropout) not in (int, float)
        or not math.isfinite(dropout)
        or dropout not in {0, 0.1, 0.2}
    ):
        raise ValueError("La arquitectura o regularización no pertenece al diseño experimental")
    return dict(hidden_size=hidden_size, layers=layers, dropout=float(dropout))


class MultimodalReference(nn.Module):
    """Fusión común con un codificador de precios y estado reiniciado por ventana.

    La profundidad controla la pila recurrente y las capas de fusión. DLinear
    conserva su descomposición temporal y solo amplía las capas de fusión.
    El dropout se aplica en la fusión, con el generador que guarda el checkpoint.
    La representación compartida permite añadir después una cabeza separada.
    """

    def __init__(self, kind, dimensions, *, context=64, hidden_size=64, layers=1, dropout=0.0):
        super().__init__()
        self.architecture = validate_architecture(hidden_size, layers, dropout)
        if (
            kind not in {*_RECURRENT, "dlinear"}
            or set(dimensions) != set(MODALITIES)
            or any(type(d) is not int or not 1 <= d <= 2048 for d in dimensions.values())
            or type(context) is not int
            or not 2 <= context <= 512
        ):
            raise ValueError("La arquitectura, las dimensiones o el contexto no son válidos")
        self.kind, self.dimensions, self.context = kind, dict(dimensions), context
        if kind in _RECURRENT:
            self.price_encoder = _RECURRENT[kind](
                dimensions["prices"],
                hidden_size,
                num_layers=layers,
                dropout=0.0,
                batch_first=True,
            )
        else:
            kernel = min(25, context if context % 2 else context - 1)
            self.price_encoder = nn.Sequential(
                DLinear(context, kernel), nn.Linear(dimensions["prices"], hidden_size), nn.SiLU()
            )
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(dimensions[name], hidden_size), nn.SiLU())
                for name in MODALITIES
                if name != "prices"
            }
        )
        blocks = []
        for layer in range(layers):
            width = len(MODALITIES) * hidden_size if layer == 0 else hidden_size
            blocks.extend((nn.Linear(width, hidden_size), nn.SiLU(), nn.Dropout(dropout)))
        self.fusion, self.head = nn.Sequential(*blocks), nn.Linear(hidden_size, 1)

    def encode(self, inputs):
        """Devolver la representación sin conservar estado entre llamadas."""
        if set(inputs) != set(MODALITIES):
            raise ValueError("Se requieren las cuatro modalidades y el contexto macro")
        prices = inputs["prices"]
        if prices.ndim != 3 or prices.shape[1:] != (self.context, self.dimensions["prices"]):
            raise ValueError("La ventana de precios no tiene la forma y contexto esperados")
        batch = prices.shape[0]
        if batch < 1 or any(
            inputs[name].shape != (batch, self.dimensions[name]) for name in self.encoders
        ):
            raise ValueError("Las modalidades no comparten el lote y las dimensiones esperadas")
        if self.kind in _RECURRENT:
            _, hidden = self.price_encoder(prices)
            price = (hidden[0] if self.kind == "lstm" else hidden)[-1]
        else:
            price = self.price_encoder(prices)
        representations = [price] + [self.encoders[name](inputs[name]) for name in self.encoders]
        return self.fusion(torch.cat(representations, dim=-1))

    def forward(self, inputs):
        return self.head(self.encode(inputs)).squeeze(-1)
