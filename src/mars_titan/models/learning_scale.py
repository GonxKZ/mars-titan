"""Escalas de aprendizaje estimadas solo con el tramo de ajuste de cada ventana (#836).

La edición entrega las entradas sin normalizar: los fundamentales y los indicadores macro
llegan como niveles en logaritmo con signo, la ventana de precios está anclada a su primer
cierre y la etiqueta residual vale del orden de una centésima. El diagnóstico de #836 midió
que, con esas escalas, un predictor recién construido emite con una amplitud varias veces
mayor que la etiqueta y que una parte de las columnas macro del año evaluado queda fuera del
rango visto al ajustar.

Este módulo declara tres piezas que comparten todas las familias neuronales:

- la ventana de precios reescrita como rendimientos por sesión (`price_returns`),
- un centro y una escala robustos por columna, estimados con filas del tramo de ajuste, y
  un recorte simétrico que acota cualquier deriva posterior (`LearningScaler`),
- la escala de la etiqueta de ese mismo tramo, que multiplica la salida de la cabeza.

Ninguna estadística usa filas de validación, calibración ni evaluación. Sin declarar
`LearningScale`, el predictor, su cálculo y su identidad no cambian.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from mars_titan.data.price_windows import PRICE_FEATURES, PRICE_WINDOW_CHANNELS

TRANSFORM = "learning_scale_v1"
# Modalidades con el diseño valores | observado | log1p de la antigüedad en días.
NUMERIC = ("fundamentals", "macro")
DENSE = ("news", "charts")
MODALITIES = ("prices", *DENSE, *NUMERIC)
# Bits de presencia de cada modalidad densa, en el orden de `data.input_policy.MODALITIES`.
PRESENCE_BIT = {"news": 1, "charts": 2}
# Cuantiles de la escala robusta. En una normal, su semidistancia es la desviación típica.
LOWER, UPPER = 0.16, 0.84
# Una columna con muchos ceros o casi constante tiene una semidistancia casi nula y dividir
# por ella amplificaría su ruido. La escala nunca baja de esta fracción de la desviación
# típica de la columna ni de esta fracción de la escala mediana de su bloque.
STD_FRACTION = 0.5
RELATIVE_FLOOR = 0.1
# El intervalo 10-90 de la cabeza de cuantiles recién construida mide 2·softplus(0) veces su
# escala, así que esta constante hace que coincida con el de las etiquetas del ajuste.
INITIAL_INTERVAL = 2 * math.log(2)
MIN_ROWS = 64


@dataclass(frozen=True)
class LearningScale:
    """Declaración de las escalas. `clip` es el recorte en unidades de escala robusta."""

    clip: float = 5.0
    price_returns: bool = True
    zero_median: bool = True

    def __post_init__(self):
        if (
            type(self.clip) not in (int, float)
            or not math.isfinite(self.clip)
            or not 1 <= self.clip <= 100
            or type(self.price_returns) is not bool
            or type(self.zero_median) is not bool
        ):
            raise ValueError(
                "learning_scale necesita clip entre 1 y 100 y dos opciones booleanas explícitas"
            )

    def identity(self):
        return dict(
            name=TRANSFORM,
            clip=float(self.clip),
            price_window="session_returns_v1" if self.price_returns else "anchored_logs",
            initial_median="zero" if self.zero_median else "sampled",
            statistics="median_and_floored_half_distance_of_quantiles_of_the_fit_rows_v1",
            scale_rule=dict(
                quantiles=[LOWER, UPPER], std_fraction=STD_FRACTION, relative_floor=RELATIVE_FLOOR
            ),
            absent_values="zero_after_scaling",
            output="head_output_times_the_10_90_interval_of_the_fit_labels_over_2_ln_2",
        )


def parse_learning_scale(value):
    """Aceptar la declaración JSON de una receta, con sus tres campos explícitos."""
    if value is None or isinstance(value, LearningScale):
        return value
    if not isinstance(value, dict) or set(value) != set(asdict(LearningScale())):
        raise ValueError("learning_scale debe declarar clip, price_returns y zero_median")
    return LearningScale(**value)


def price_returns(window):
    """Reescribir la ventana anclada como rendimientos por sesión, con la misma forma.

    La entrada trae, por sesión, el logaritmo de apertura, máximo, mínimo y cierre frente al
    cierre ancla, el volumen relativo y, desde la v3.1, el bit de presencia. La salida deja
    en esos cuatro canales el rendimiento nocturno frente al último cierre presente, el
    rendimiento entre apertura y cierre, la mecha superior y la mecha inferior. El ancla es
    el primer cierre presente, así que la ventana original se reconstruye sumando. Una sesión
    ausente queda en cero y la primera sesión no tiene rendimiento nocturno.
    """
    if (
        not isinstance(window, torch.Tensor)
        or window.ndim != 3
        or window.shape[-1] not in (len(PRICE_FEATURES), len(PRICE_WINDOW_CHANNELS))
    ):
        raise ValueError("La ventana de precios necesita [filas, sesiones, 5 o 6 canales]")
    opening, high, low, close = window[..., :4].unbind(-1)
    if window.shape[-1] == len(PRICE_WINDOW_CHANNELS):
        present = window[..., -1] > 0
    else:
        present = torch.ones_like(close, dtype=torch.bool)
    index = torch.arange(window.shape[1], device=window.device).expand_as(close)
    last = torch.where(present, index, torch.full_like(index, -1)).cummax(dim=-1).values
    previous = torch.cat([torch.full_like(last[:, :1], -1), last[:, :-1]], dim=-1)
    known = present & (previous >= 0)
    before = close.gather(-1, previous.clamp(min=0))
    zero = torch.zeros((), dtype=window.dtype, device=window.device)
    returns = torch.stack(
        (
            torch.where(known, opening - before, zero),
            torch.where(present, close - opening, zero),
            torch.where(present, high - torch.maximum(opening, close), zero),
            torch.where(present, torch.minimum(opening, close) - low, zero),
        ),
        dim=-1,
    )
    return torch.cat([returns, window[..., 4:]], dim=-1)


def _robust(values):
    """Centro y escala sin suelo por columna de `[filas, columnas]`. Sin filas, ceros."""
    width = values.shape[1]
    if not len(values):
        return np.zeros(width), np.zeros(width)
    values = values.astype(np.float64)
    lower, center, upper = np.quantile(values, (LOWER, 0.5, UPPER), axis=0)
    return center, np.maximum((upper - lower) / 2, STD_FRACTION * values.std(axis=0))


def _floored(scale):
    """Aplicar el suelo relativo del bloque. Sin ninguna columna con dispersión, la unidad."""
    positive = scale[scale > 0]
    if not len(positive):
        return np.ones_like(scale)
    return np.maximum(scale, RELATIVE_FLOOR * np.median(positive))


def _masked_columns(values, observed):
    """Centro y escala sin suelo de cada columna con solo sus filas observadas."""
    centers, scales = np.zeros(values.shape[1]), np.zeros(values.shape[1])
    for column in range(values.shape[1]):
        chosen = values[observed[:, column], column]
        if len(chosen):
            center, scale = _robust(chosen[:, None])
            centers[column], scales[column] = center[0], scale[0]
    return centers, scales


def fit_statistics(config, inputs, presence, targets):
    """Estadísticas de las filas de ajuste recibidas, listas para guardar en JSON.

    `inputs` son los arreglos de cada modalidad y `presence` sus cinco bits por fila.
    `targets` son etiquetas del mismo tramo y no tienen por qué ser las de esas filas, porque
    solo se usa su distribución. Quien llama elige las filas y responde de que todas
    pertenecen al tramo de ajuste.
    """
    if not isinstance(config, LearningScale):
        raise ValueError("Las estadísticas necesitan la declaración LearningScale")
    targets = np.asarray(targets, dtype=np.float64)
    presence = np.asarray(presence)
    rows = len(presence)
    if (
        set(inputs) != set(MODALITIES)
        or targets.ndim != 1
        or min(len(targets), rows) < MIN_ROWS
        or not np.isfinite(targets).all()
        or presence.dtype != np.bool_
        or presence.shape != (rows, len(MODALITIES))
        or any(len(value) != rows for value in inputs.values())
        or any(not np.isfinite(value).all() for value in inputs.values())
    ):
        raise ValueError(
            f"Las estadísticas necesitan al menos {MIN_ROWS} filas y etiquetas finitas"
        )
    result = {}
    window = torch.from_numpy(np.ascontiguousarray(inputs["prices"]))
    if config.price_returns:
        window = price_returns(window)
    window = window.numpy()
    channels = len(PRICE_FEATURES)
    present = (
        window[..., -1] > 0
        if window.shape[-1] == len(PRICE_WINDOW_CHANNELS)
        else np.ones(window.shape[:2], dtype=bool)
    )
    # Los canales de precios tienen unidades distintas, así que cada uno es su propio bloque.
    center, scale = _robust(window[..., :channels][present])
    result["prices"] = center, np.where(scale > 0, scale, 1.0)
    for name in DENSE:
        center, scale = _robust(inputs[name][presence[:, PRESENCE_BIT[name]]])
        result[name] = center, _floored(scale)
    for name in NUMERIC:
        values = inputs[name]
        third, remainder = divmod(values.shape[1], 3)
        if remainder:
            raise ValueError("Una modalidad numérica necesita valores, presencia y antigüedad")
        observed = values[:, third : 2 * third] > 0
        level, age = (
            _masked_columns(values[:, part], observed)
            for part in (slice(0, third), slice(2 * third, None))
        )
        result[name] = (
            np.concatenate([level[0], np.zeros(third), age[0]]),
            np.concatenate([_floored(level[1]), np.ones(third), _floored(age[1])]),
        )
    lower, upper = np.quantile(targets, (0.1, 0.9))
    label = float((upper - lower) / INITIAL_INTERVAL)
    if not label > 0:
        raise ValueError("Las etiquetas del ajuste no tienen dispersión")
    return dict(
        schema_version=1,
        transform=config.identity(),
        rows=rows,
        labels=len(targets),
        label_scale=label,
        modalities={
            name: dict(
                center=np.asarray(center, dtype=np.float32).tolist(),
                scale=np.asarray(scale, dtype=np.float32).tolist(),
            )
            for name, (center, scale) in result.items()
        },
    )


def statistics_digest(statistics):
    text = json.dumps(statistics, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


class LearningScaler(nn.Module):
    """Aplica a un lote el centro, la escala y el recorte guardados en sus buffers.

    Se construye con la identidad (centro cero y escala uno) y sin estadísticas. Hasta
    recibirlas con `load` o con un estado guardado, `forward` se niega a transformar, para
    que ningún ajuste empiece con las entradas sin normalizar por un descuido.
    """

    def __init__(self, config, dimensions, *, dtype=torch.float32):
        super().__init__()
        if not isinstance(config, LearningScale) or set(dimensions) != set(MODALITIES):
            raise ValueError("El escalador necesita su declaración y las cinco modalidades")
        self.config = config
        self.widths = {
            name: len(PRICE_FEATURES) if name == "prices" else int(dimensions[name])
            for name in MODALITIES
        }
        if any(self.widths[name] % 3 for name in NUMERIC):
            raise ValueError("Una modalidad numérica necesita valores, presencia y antigüedad")
        for name, width in self.widths.items():
            self.register_buffer(f"{name}_center", torch.zeros(width, dtype=dtype))
            self.register_buffer(f"{name}_inverse", torch.ones(width, dtype=dtype))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    def load(self, statistics):
        """Copiar las estadísticas de `fit_statistics` tras comprobar su forma y su rango."""
        if (
            not isinstance(statistics, dict)
            or statistics.get("schema_version") != 1
            or statistics.get("transform") != self.config.identity()
            or set(statistics.get("modalities", ())) != set(MODALITIES)
        ):
            raise ValueError("Las estadísticas no corresponden a esta declaración de escalas")
        values = {}
        for name, width in self.widths.items():
            center = torch.tensor(statistics["modalities"][name]["center"], dtype=torch.float64)
            scale = torch.tensor(statistics["modalities"][name]["scale"], dtype=torch.float64)
            if (
                center.shape != (width,)
                or scale.shape != (width,)
                or not torch.isfinite(center).all()
                or not torch.isfinite(scale).all()
                or not (scale > 0).all()
            ):
                raise ValueError(f"Las estadísticas de {name} no tienen forma o rango válidos")
            values[name] = center, scale.reciprocal()
        with torch.no_grad():
            for name, (center, inverse) in values.items():
                getattr(self, f"{name}_center").copy_(center)
                getattr(self, f"{name}_inverse").copy_(inverse)
            self.fitted.fill_(True)

    def _scale(self, name, values):
        center, inverse = getattr(self, f"{name}_center"), getattr(self, f"{name}_inverse")
        return ((values - center) * inverse).clamp(-self.config.clip, self.config.clip)

    def forward(self, inputs, presence):
        """Devolver las mismas modalidades, con la misma forma, tipo y dispositivo.

        `presence` son los cinco bits por fila del lote. Lo ausente sigue valiendo cero
        después de escalar: una sesión sin mercado, un bloque denso sin dato y un valor
        numérico no observado.
        """
        if not bool(self.fitted):
            raise ValueError("Faltan las estadísticas del tramo de ajuste de learning_scale")
        result = {}
        window = inputs["prices"]
        if self.config.price_returns:
            window = price_returns(window)
        channels = len(PRICE_FEATURES)
        scaled = self._scale("prices", window[..., :channels])
        zero = torch.zeros((), dtype=scaled.dtype, device=scaled.device)
        if window.shape[-1] > channels:
            scaled = torch.where(window[..., -1:] > 0, scaled, zero)
        result["prices"] = torch.cat([scaled, window[..., channels:]], dim=-1)
        for name, bit in PRESENCE_BIT.items():
            result[name] = torch.where(
                presence[:, bit : bit + 1], self._scale(name, inputs[name]), zero
            )
        for name in NUMERIC:
            values = inputs[name]
            third = self.widths[name] // 3
            observed = values[..., third : 2 * third]
            scaled = self._scale(name, values)
            # Un valor no observado se queda en cero, igual que en la edición.
            result[name] = torch.cat(
                [scaled[..., :third] * observed, observed, scaled[..., 2 * third :] * observed],
                dim=-1,
            )
        return result
