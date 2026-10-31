"""Contexto inmutable de una emisión de S2, sin etiquetas ni actualización del predictor.

La actividad es la máscara de participación de áreas y ranuras que utiliza H1.
Los valores, máscaras y log-edades macro se conservan tal como llegaron. Una edad
redondeada no se convierte en una fecha de publicación. Las fechas individuales
solo se registran cuando las aporta la fuente y, en otro caso, quedan desconocidas.
"""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from mars_titan.data.input_policy import MODALITIES, validate_historical_vectors
from mars_titan.models.titans.config import FingerprintOnce, bounded_integer, canonical
from mars_titan.models.titans.episodic_snapshot import _digest_id
from mars_titan.models.titans.financial_inputs import (
    FINAL_TEST_US,
    FLOW_ID,
    HISTORICAL_START_US,
)

UNKNOWN_PUBLICATION = np.iinfo(np.int64).min
MAX_MACRO_INDICATORS = 2048 // 3
MAX_AREAS = 128
MAX_KEY_WIDTH = 2048
_METADATA = (
    "schema_version",
    "layout_id",
    "flow_id",
    "sample_id",
    "prediction_at",
    "input_available_at",
    "input_sha256",
    "action",
)
_BUFFERS = ("key_inputs", "macro", "presence", "macro_available_at", "area_activity")


def _names(values, name, maximum, minimum=0):
    if (
        not isinstance(values, (list, tuple))
        or not minimum <= len(values) <= maximum
        or any(type(v) is not str or not 1 <= len(v) <= 128 or not v.isprintable() for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} necesita nombres únicos, legibles y acotados")
    return tuple(values)


def _vector(value, shape, dtype, name):
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
        raise ValueError(f"{name} no conserva su forma y tipo NumPy")
    return value


def _publication_times(values, macro, available_at, count):
    if values is None:
        return np.full(count, UNKNOWN_PUBLICATION, dtype=np.int64)
    dates = _vector(values, (count,), np.dtype(np.int64), "Las fechas macro")
    known = dates != UNKNOWN_PUBLICATION
    if np.any(dates[known] > available_at) or np.any(known & (macro[count : 2 * count] == 0)):
        raise ValueError(
            "Una publicación macro supera la disponibilidad de la entrada o acompaña "
            "un concepto ausente"
        )
    return dates


def normalized_key(values, *, allow_zero=False):
    """Norma L2 con escalado previo, sin elevar a cuadrado entradas de tamaño extremo."""
    scale = np.max(np.abs(values))
    if allow_zero and scale == 0:
        return values
    if not np.isfinite(values).all() or not scale > 0:
        raise ValueError("La clave de contexto debe ser finita y distinta de cero")
    scaled = values / scale
    return scaled / np.linalg.norm(scaled)


@dataclass(frozen=True)
class FailureContextLayout(FingerprintOnce):
    input_id: str
    codec_id: str
    macro_catalog_id: str
    macro_indicators: tuple[str, ...]
    macro_families: tuple[str, ...]
    family_names: tuple[str, ...]
    area_names: tuple[str, ...] = ()
    area_modalities: tuple[str | None, ...] = ()
    key_width: int = 64
    allow_zero_key: bool = False

    def __post_init__(self):
        for name in ("input_id", "codec_id", "macro_catalog_id"):
            _digest_id(getattr(self, name), name)
        for name, maximum, minimum in (
            ("macro_indicators", MAX_MACRO_INDICATORS, 1),
            ("family_names", 12, 12),
            ("area_names", MAX_AREAS, 0),
        ):
            object.__setattr__(self, name, _names(getattr(self, name), name, maximum, minimum))
        categories = self.macro_families
        if (
            not isinstance(categories, (list, tuple))
            or len(categories) != len(self.macro_indicators)
            or any(type(v) is not str or v not in self.family_names for v in categories)
        ):
            raise ValueError("Cada indicador de contexto necesita su familia macro declarada")
        owners = self.area_modalities
        if (
            not isinstance(owners, (list, tuple))
            or len(owners) != len(self.area_names)
            or any(v is not None and (type(v) is not str or v not in MODALITIES) for v in owners)
        ):
            raise ValueError("Cada área necesita su modalidad o una ranura compartida")
        object.__setattr__(self, "macro_families", tuple(categories))
        object.__setattr__(self, "area_modalities", tuple(owners))
        bounded_integer(self.key_width, "dimensión de contexto", 1, MAX_KEY_WIDTH)
        if type(self.allow_zero_key) is not bool:
            raise ValueError("La admisión de claves nulas necesita una opción booleana explícita")

    def identity(self):
        values = asdict(self)
        allow_zero = values.pop("allow_zero_key")
        if allow_zero:
            values["zero_key_policy"] = "keep_context_exclude_similarity_v1"
        return dict(
            schema_version=1,
            kind="mature_failure_context_v1",
            **values,
            modalities=list(MODALITIES),
            macro_layout="values_observed_log_age",
            publication_dates="provided_or_unknown_never_inferred_from_rounded_age",
            area_activity="h1_alive_mask_from_block_presence_and_shared_slots",
            key_geometry=(
                "scaled_l2_zero_ineligible_float64_v2" if allow_zero else "scaled_l2_float64_v1"
            ),
        )

    def activity(self, presence):
        return np.asarray(
            [
                presence.any() if owner is None else presence[MODALITIES.index(owner)]
                for owner in self.area_modalities
            ],
            dtype=np.bool_,
        )


@dataclass(frozen=True, init=False)
class FailureContext:
    layout_id: str
    flow_id: str
    sample_id: str
    prediction_at: int
    input_available_at: int
    input_sha256: str
    action: str
    context_id: str
    _key: bytes
    _macro: bytes
    _presence: bytes
    _dates: bytes
    _areas: bytes

    def __init__(
        self,
        layout,
        *,
        flow_id,
        sample_id,
        prediction_at,
        input_available_at,
        input_sha256,
        key_inputs,
        macro,
        presence,
        action,
        macro_available_at=None,
    ):
        if type(layout) is not FailureContextLayout:
            raise ValueError("El contexto necesita su diseño identificado")
        if type(flow_id) is not str or FLOW_ID.fullmatch(flow_id) is None:
            raise ValueError("El contexto necesita una identidad de flujo válida")
        for value, name, maximum in ((sample_id, "muestra", 1024), (action, "mecanismo", 256)):
            if type(value) is not str or not 1 <= len(value) <= maximum or not value.isprintable():
                raise ValueError(f"El identificador de {name} no es legible y acotado")
        if (
            type(prediction_at) is not int
            or not HISTORICAL_START_US <= prediction_at < FINAL_TEST_US
            or type(input_available_at) is not int
            or not 0 < input_available_at <= prediction_at
        ):
            raise ValueError("El contexto necesita instantes válidos y no puede leer el futuro")
        _digest_id(input_sha256, "La entrada del contexto")
        if (
            not isinstance(key_inputs, np.ndarray)
            or key_inputs.shape != (layout.key_width,)
            or key_inputs.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        ):
            raise ValueError("La clave de contexto no conserva su dimensión y tipo real")
        key = key_inputs.astype(np.float64, copy=False)
        normalized_key(key, allow_zero=layout.allow_zero_key)
        count = len(layout.macro_indicators)
        macro = _vector(macro, (3 * count,), np.dtype(np.float32), "El contexto macro")
        presence = _vector(presence, (len(MODALITIES),), np.dtype(np.bool_), "La presencia")
        validate_historical_vectors(
            {"macro": macro.reshape(1, -1)},
            presence.reshape(1, -1),
            {"macro_indicators": layout.macro_indicators},
        )
        dates = _publication_times(macro_available_at, macro, input_available_at, count)
        for name, value in (
            ("layout_id", layout.fingerprint()),
            ("flow_id", flow_id),
            ("sample_id", sample_id),
            ("prediction_at", prediction_at),
            ("input_available_at", input_available_at),
            ("input_sha256", input_sha256),
            ("action", action),
            ("_key", key.tobytes()),
            ("_macro", macro.tobytes()),
            ("_presence", presence.tobytes()),
            ("_dates", dates.tobytes()),
            ("_areas", layout.activity(presence).tobytes()),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "context_id", self._digest())

    def _metadata(self):
        return dict(
            schema_version=1,
            **{name: getattr(self, name) for name in _METADATA if name != "schema_version"},
        )

    def _digest(self):
        value = hashlib.sha256(canonical(self._metadata()).encode())
        for raw in (self._key, self._macro, self._presence, self._dates, self._areas):
            value.update(len(raw).to_bytes(8, "little"))
            value.update(raw)
        return value.hexdigest()

    @property
    def key_inputs(self):
        return np.frombuffer(self._key, dtype=np.float64)

    def normalized_key(self):
        self.verify()
        return normalized_key(self.key_inputs, allow_zero=True)

    @property
    def macro(self):
        return np.frombuffer(self._macro, dtype=np.float32)

    @property
    def presence(self):
        return np.frombuffer(self._presence, dtype=np.bool_)

    @property
    def macro_available_at(self):
        return np.frombuffer(self._dates, dtype=np.int64)

    @property
    def area_activity(self):
        return np.frombuffer(self._areas, dtype=np.bool_)

    def export(self):
        self.verify()
        return dict(
            self._metadata(),
            context_id=self.context_id,
            **dict(
                zip(
                    _BUFFERS,
                    (self._key, self._macro, self._presence, self._dates, self._areas),
                    strict=True,
                )
            ),
        )

    def verify(self):
        """Contrastar el sello antes de admitir o serializar el contexto."""
        if (
            any(
                type(raw) is not bytes
                for raw in (self._key, self._macro, self._presence, self._dates, self._areas)
            )
            or self._digest() != self.context_id
        ):
            raise ValueError("La huella del contexto no conserva los datos capturados")

    @classmethod
    def restore(cls, layout, payload):
        if (
            type(layout) is not FailureContextLayout
            or type(payload) is not dict
            or set(payload) != set(_METADATA) | set(_BUFFERS) | {"context_id"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["layout_id"] != layout.fingerprint()
        ):
            raise ValueError("El diseño del contexto recuperado no conserva su contrato")
        _digest_id(payload["context_id"], "El contexto recuperado")
        sizes = (
            layout.key_width * 8,
            len(layout.macro_indicators) * 12,
            len(MODALITIES),
            len(layout.macro_indicators) * 8,
            len(layout.area_names),
        )
        if any(
            type(payload[name]) is not bytes or len(payload[name]) != size
            for name, size in zip(_BUFFERS, sizes, strict=True)
        ):
            raise ValueError("Los buffers del contexto recuperado no conservan su tamaño")
        values = {
            name: payload[name] for name in _METADATA if name not in ("schema_version", "layout_id")
        }
        result = cls(
            layout,
            **values,
            key_inputs=np.frombuffer(payload["key_inputs"], dtype=np.float64),
            macro=np.frombuffer(payload["macro"], dtype=np.float32),
            presence=np.frombuffer(payload["presence"], dtype=np.bool_),
            macro_available_at=np.frombuffer(payload["macro_available_at"], dtype=np.int64),
        )
        if result.export() != payload:
            raise ValueError("La huella o la actividad del contexto recuperado no coinciden")
        return result
