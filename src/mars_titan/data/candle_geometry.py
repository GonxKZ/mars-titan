"""Geometría de 64 velas pasadas sin rasterización ni parámetros aprendidos."""

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np

from .prices import check_ordering_rtol, ordering_excess
from .storage import sha256

MODEL = "candle_geometry_linear64_v1"
SESSIONS, WIDTH = 64, 512
CHANNELS = ("open", "high", "low", "close", "body", "upper_wick", "lower_wick", "present")


def _window(prices, end_index, present, ordering_rtol):
    check_ordering_rtol(ordering_rtol)
    if (
        not isinstance(prices, np.ndarray)
        or prices.ndim != 2
        or prices.shape[1] != 4
        or prices.dtype.kind not in "fiu"
    ):
        raise ValueError("La geometría necesita una matriz numérica OHLC")
    slots = np.ones(SESSIONS, dtype=bool) if present is None else np.asarray(present)
    if slots.shape != (SESSIONS,) or slots.dtype != np.bool_ or not slots[-1]:
        raise ValueError("La presencia necesita 64 sesiones y una decisión observada")
    count = int(slots.sum())
    if (
        count < 2
        or isinstance(end_index, (bool, np.bool_))
        or not isinstance(end_index, (int, np.integer))
        or not count - 1 <= end_index < len(prices)
    ):
        raise ValueError("El índice no identifica una ventana pasada suficiente")
    # Solo se convierte la ventana seleccionada. El resto de la historia no se inspecciona.
    window = np.array(prices[end_index - count + 1 : end_index + 1], dtype=np.float64, copy=True)
    if not np.isfinite(window).all() or (window <= 0).any():
        raise ValueError("La geometría requiere precios positivos y finitos")
    if (ordering_excess(*window.T) > ordering_rtol).any():
        raise ValueError("Los precios OHLC exceden la tolerancia de orden")
    return window, slots


def _geometry(window, slots):
    # Dividir antes de restar conserva el rango relativo incluso con precios subnormales.
    scale = window[:, 1].max()
    normalized = window / scale
    low = normalized[:, 2].min()
    normalized = (normalized - low) / max(1.0 - low, 1e-9)
    opening, upper, lower, close = normalized.T
    result = np.zeros((SESSIONS, len(CHANNELS)), dtype=np.float64)
    result[slots, :4] = normalized
    result[slots, 4] = close - opening
    result[slots, 5] = upper - np.maximum(opening, close)
    result[slots, 6] = np.minimum(opening, close) - lower
    result[slots, 7] = 1.0
    return result.reshape(WIDTH).astype(np.float32)


def geometry_vector(prices, *, end_index, present=None, ordering_rtol=0.0):
    """Ocho valores por sesión, en orden temporal, con huecos a cero y máscara explícita.

    `prices` conserva solo filas observadas. `present` sitúa las últimas filas admitidas en
    sus 64 posiciones del calendario. La selección temporal y la procedencia se acreditan
    antes, en la materialización. El resultado no modifica precios ni repara el redondeo OHLC.
    """
    window, slots = _window(prices, end_index, present, ordering_rtol)
    return _geometry(window, slots)


def geometry_chart(prices, *, end_index, present=None, ordering_rtol=0.0):
    """Huella de las entradas observadas y vector. No utiliza ni identifica bytes de PNG."""
    window, slots = _window(prices, end_index, present, ordering_rtol)
    digest = hashlib.sha256(MODEL.encode() + b"\0")
    digest.update(slots.tobytes())
    digest.update(window.astype("<f8", copy=False).tobytes())
    return digest.hexdigest(), _geometry(window, slots)


def geometry_spec():
    """Contrato versionado, calculable en CPU y sin datos de ajuste."""
    return dict(
        version=1,
        context_sessions=SESSIONS,
        width=WIDTH,
        channels=list(CHANNELS),
        layout="time_major",
        normalization="(price/max_high-min_low/max_high)/max(1-min_low/max_high,1e-9)",
        absent_fill="positive_zero_all_channels",
        ohlc_ordering="audit_tolerance_without_repair",
        content_hash="sha256_model_nul_presence_bool64_observed_ohlc_little_endian_float64",
        computation="numpy_float64_rounded_once_to_float32",
        device="cpu",
        learned_parameters=0,
        pretraining="none",
        numpy=np.__version__,
        code_sha256=sha256(Path(__file__)),
        price_validation_sha256=sha256(Path(__file__).with_name("prices.py")),
    )


def geometry_encoder_spec(base_spec):
    """Conservar la procedencia del texto y declarar la nueva representación visual."""
    result = {**deepcopy(base_spec), "image_model": MODEL, "image_policy": geometry_spec()}
    result.pop("image_url", None)
    result.pop("image_weights_sha256", None)
    return result


class GeometryEncoders:
    """Delegar el texto en el codificador elegido y sustituir solo la representación de velas.

    La materialización de cohortes proporciona OHLC y su máscara a `chart`. Las rutas que
    únicamente conservan PNG deben rechazar esta representación, pues han perdido precisión.
    """

    def __init__(self, base):
        from .vector_carry import CollectingEncoders, ReuseOnlyEncoders

        if isinstance(base, (CollectingEncoders, ReuseOnlyEncoders)):
            raise ValueError("La geometría requiere un codificador de texto de una sola pasada")
        if not isinstance(getattr(base, "spec", None), dict) or not callable(
            getattr(base, "text", None)
        ):
            raise ValueError("La composición requiere un codificador de texto identificado")
        self.base, self._base_spec = base, deepcopy(base.spec)
        self.spec = geometry_encoder_spec(base.spec)
        self._expected_spec = deepcopy(self.spec)

    def _check(self):
        if self.base.spec != self._base_spec or self.spec != self._expected_spec:
            raise ValueError("La identidad de la composición de codificadores ha cambiado")

    def text(self, text):
        self._check()
        return self.base.text(text)

    def chart(self, prices, *, end_index, context=64, ordering_rtol=0.0, present=None):
        self._check()
        if type(context) is not int or context != SESSIONS:
            raise ValueError("La representación geométrica requiere 64 sesiones")
        return geometry_chart(
            prices, end_index=end_index, ordering_rtol=ordering_rtol, present=present
        )

    def images(self, pngs):
        raise ValueError("La representación geométrica necesita los OHLC, no un PNG")
