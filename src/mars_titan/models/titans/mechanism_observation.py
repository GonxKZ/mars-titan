"""Vector de cohorte sellado con su procedencia, sin retener arrays del corpus."""

import hashlib
import json
import math
from dataclasses import dataclass

import torch

from mars_titan.calibration.online_conformal import _instant
from mars_titan.data.input_policy import masked_inputs

from .config import bounded_integer, canonical
from .episodic_snapshot import _device, _digest_id
from .financial_inputs import HISTORICAL_START_US
from .mechanism_policy import FEATURES

KIND = "pre_mac_cohort_observation_v1"
MAX_BYTES = 64 * 1024
_FIXED = dict(
    schema_version=1,
    kind=KIND,
    feature_names=list(FEATURES),
    feature_dtype="float64",
    normalization="none_raw_features",
    stage="before_mac_before_current_feedback",
    aggregation="canonical_flow_order_mean_over_all_rows_absent_zero",
)
_IDENTITIES = (
    "consumer_id",
    "parameter_id",
    "input_contract_id",
    "context_id",
    "producer_sha256",
    "flow_ids_sha256",
    "rows_sha256",
)
_GROUP_FIELDS = {
    "market",
    "rows",
    "oldest_input_available_at",
    "newest_input_available_at",
    "context_committed_at",
    "values",
}


def _close(value, expected):
    return math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)


def _validate_group(group, cutoff, previous, bank_entries):
    if (
        type(group) is not dict
        or set(group) != _GROUP_FIELDS
        or group["market"] not in ("CN", "US")
    ):
        raise ValueError("La observación no conserva su cohorte de mercado")
    count = group["rows"]
    bounded_integer(count, "filas de observación", 1, 8192)
    oldest = _instant(group["oldest_input_available_at"], "La primera disponibilidad")
    newest = _instant(group["newest_input_available_at"], "La última disponibilidad")
    if not oldest <= newest <= cutoff:
        raise ValueError("La observación contiene disponibilidad futura o invertida")
    values = group["values"]
    if (
        type(values) is not list
        or len(values) != len(FEATURES)
        or any(type(v) is not float or not math.isfinite(v) for v in values)
    ):
        raise ValueError("El vector no conserva los rasgos finitos en FP64")
    if any(v < 0 for v in values[:17]):
        raise ValueError("Una fracción, edad, magnitud o recuento es negativo")
    if (
        not _close(values[0], math.log1p(count))
        or not _close(values[11], math.log1p((cutoff - oldest) / 1_000_000))
        or not _close(values[12], math.log1p(bank_entries))
    ):
        raise ValueError("El vector contradice sus recuentos o edades")
    for fraction, magnitude in zip(values[1:6], values[6:11], strict=True):
        if fraction > 1 or not _close(fraction * count, round(fraction * count)):
            raise ValueError("La fracción de presencia no corresponde a la cohorte")
        if fraction == 0 and magnitude != 0:
            raise ValueError("Una fuente ausente aporta una magnitud")
    committed = group["context_committed_at"]
    if committed is None:
        if values[13:15] != [0.0, 0.0]:
            raise ValueError("Falta el contexto anterior declarado por el vector")
    else:
        committed = _instant(committed, "El contexto anterior")
        if previous is None or committed > previous or committed >= cutoff:
            raise ValueError("El contexto de la observación procede del futuro")
        if values[13] != 1 or not _close(
            values[14], math.log1p((cutoff - committed) / 86_400_000_000)
        ):
            raise ValueError("La edad del contexto no corresponde a su origen")
    if values[15] not in (0, 1) or (values[15] == 0 and any(values[16:])):
        raise ValueError("Los rasgos no conservan la ausencia de calibración")
    for mean, maximum in ((values[17], values[18]), (values[19], values[20])):
        if maximum < 0 or (abs(mean) > maximum and not _close(abs(mean), maximum)):
            raise ValueError("Los resúmenes de corrección son incompatibles")


@dataclass(frozen=True, init=False)
class MechanismObservation:
    """JSON inmutable con valores calculados y huellas, no autenticación del host."""

    _json: str
    sha256: str

    def __init__(self, body):
        fields = {
            *_FIXED,
            *_IDENTITIES,
            "cutoff",
            "previous_cutoff",
            "snapshot_sha256",
            "bank_entries",
            "input_policy",
            "groups",
        }
        if (
            type(body) is not dict
            or set(body) != fields
            or type(body["schema_version"]) is not int
            or any(body[k] != v for k, v in _FIXED.items())
        ):
            raise ValueError("La observación no corresponde al esquema de rasgos")
        for key in _IDENTITIES:
            _digest_id(body[key], key)
        cutoff = _instant(body["cutoff"], "La observación")
        if masked_inputs(body["input_policy"]) and cutoff < HISTORICAL_START_US:
            raise ValueError("La observación histórica no puede preceder al año 2000")
        previous = body["previous_cutoff"]
        if previous is not None and _instant(previous, "El snapshot anterior") >= cutoff:
            raise ValueError("La observación no es posterior al snapshot confirmado")
        count = body["bank_entries"]
        bounded_integer(count, "episodios elegibles", 0, 1024)
        if body["snapshot_sha256"] is None:
            if count:
                raise ValueError("Falta el snapshot del banco no vacío")
        else:
            _digest_id(body["snapshot_sha256"], "El snapshot del banco")
        groups = body["groups"]
        if type(groups) is not list or not 1 <= len(groups) <= 2:
            raise ValueError("La observación necesita una o dos cohortes")
        for group in groups:
            _validate_group(group, cutoff, previous, count)
        markets = [g["market"] for g in groups]
        if markets != sorted(set(markets)) or sum(g["rows"] for g in groups) > 8192:
            raise ValueError("Las cohortes se repiten, desordenan o exceden su presupuesto")
        encoded = canonical(body)
        if len(encoded.encode()) > MAX_BYTES:
            raise ValueError("La observación supera 64 KiB")
        object.__setattr__(self, "_json", encoded)
        object.__setattr__(self, "sha256", hashlib.sha256(encoded.encode()).hexdigest())

    def record(self):
        return dict(observation=json.loads(self._json), sha256=self.sha256)

    @classmethod
    def restore(cls, record):
        if type(record) is not dict or set(record) != {"observation", "sha256"}:
            raise ValueError("El registro de observación no conserva todos sus campos")
        value = cls(record["observation"])
        if record["sha256"] != value.sha256:
            raise ValueError("La huella de la observación no corresponde a sus valores")
        return value

    def tensor(self, *, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("La observación requiere FP32 o FP64 explícitos")
        groups = json.loads(self._json)["groups"]
        limit = torch.finfo(dtype).max
        if any(abs(v) > limit for group in groups for v in group["values"]):
            raise ValueError("Los rasgos no tienen representación finita en la precisión elegida")
        return torch.tensor(
            [g["values"] for g in groups],
            device=_device(device),
            dtype=dtype,
        )
