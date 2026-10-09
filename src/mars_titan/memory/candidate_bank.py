"""Banco episódico 128×256 de la GRU, con tensores CPU y el reservorio causal v2 nativo.

El banco solo guarda episodios maduros. No consulta vecinos, no normaliza claves y
no conserva estados de trabajo. La lectura ordena por ID una vista que recibe
`Candidate.snapshot`. Las plazas internas conservan el orden del reservorio.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

KEY_WIDTH, VALUE_WIDTH = 128, 256
MAX_CAPACITY = 8192
MAX_INCOMING = 8192
MAX_WORKING_BYTES = 256 * 1024**2
MAX_SEEN = 2**32
UNIT_TOLERANCE = 1e-5
_METADATA_BYTES = 64 * 1024
_INT64_MAX = 2**63 - 1
_PARTITIONS = {"train", "validation", "calibration", "evaluation"}
_FIELDS = ("ids", "keys", "values", "times", "labels")
_STATE = {
    "identity",
    *_FIELDS,
    "seen",
    "last_id",
    "confirmed_at",
    "reservoir_rng",
}
_CODE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class CandidateBankConfig:
    """Capacidad física y semilla explícitas. El presupuesto cubre la propuesta completa."""

    capacity: int = 1024
    seed: int = 73
    max_working_bytes: int = MAX_WORKING_BYTES

    def __post_init__(self):
        bounds = (
            (self.capacity, 1, MAX_CAPACITY),
            (self.seed, 0, 2**64 - 1),
            (self.max_working_bytes, 1, MAX_WORKING_BYTES),
        )
        if any(type(value) is not int or not low <= value <= high for value, low, high in bounds):
            raise ValueError("La configuración del banco GRU excede sus límites")


@dataclass(frozen=True)
class CandidateRecord:
    """Metadatos de un episodio retenido para conciliarlo con su procedencia."""

    id: int
    decision_at: int
    available_at: int
    maturity_at: int
    label: float
    label_valid: bool = True
    reward_valid: bool = False


@dataclass(frozen=True)
class CandidateEpisodes:
    """Lote maduro en orden de admisión. Las columnas de tiempo son decisión,
    disponibilidad de inputs y maduración. Las etiquetas originales son FP64."""

    ids: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    times: torch.Tensor
    labels: torch.Tensor


def _tensor(value, dtype, shape, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or value.dtype != dtype
        or tuple(value.shape) != shape
        or value.requires_grad
        or not value.is_contiguous()
    ):
        raise ValueError(f"El tensor {name} del banco GRU no conserva tipo, forma o CPU")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"El tensor {name} del banco GRU contiene NaN o infinito")


def _check_rows(ids, keys, values, times, labels, dtype):
    if not isinstance(ids, torch.Tensor) or ids.dim() != 1:
        raise ValueError("Los IDs del banco GRU necesitan un vector")
    rows = ids.shape[0]
    _tensor(ids, torch.int64, (rows,), "ids")
    _tensor(keys, dtype, (rows, KEY_WIDTH), "keys")
    _tensor(values, dtype, (rows, VALUE_WIDTH), "values")
    _tensor(times, torch.int64, (rows, 3), "times")
    _tensor(labels, torch.float64, (rows,), "labels")
    if rows == 0:
        return
    decision, available, maturity = times.unbind(1)
    if (
        bool((ids <= 0).any())
        or bool((available < 0).any())
        or bool((available > decision).any())
        or bool((maturity <= decision).any())
    ):
        raise ValueError("El episodio GRU no conserva ID, disponibilidad y maduración")
    norms = keys.norm(dim=1)
    unit = ((norms - 1).abs() <= UNIT_TOLERANCE) | (keys == 0).all(dim=1)
    if not bool(unit.all()):
        raise ValueError("Las claves GRU deben tener norma uno o ser exactamente nulas")


class CandidateEpisodeBank:
    """Proponer copias del banco. El coordinador publica la generación completa."""

    def __init__(
        self, native, config, *, codec_id, representation_id, dtype, world, partition, fold
    ):
        if (
            type(config) is not CandidateBankConfig
            or not isinstance(codec_id, str)
            or len(codec_id) != 64
            or any(value not in "0123456789abcdef" for value in codec_id)
            or not isinstance(representation_id, str)
            or not representation_id
            or dtype not in (torch.float32, torch.float64)
            or partition not in _PARTITIONS
            or any(not isinstance(text, str) or not text for text in (world, fold))
        ):
            raise ValueError("Falta la identidad del banco GRU, de su codec o de su ámbito")
        self.config, self.dtype, self._native = config, dtype, native
        self._scope = dict(world=world, partition=partition, fold=fold)
        identity = dict(
            schema_version=1,
            recipe="candidate_tensor_reservoir_v1",
            implementation=_CODE,
            codec_id=codec_id,
            representation_id=representation_id,
            dtype=str(dtype).removeprefix("torch."),
            key_width=KEY_WIDTH,
            value_width=VALUE_WIDTH,
            config=asdict(config),
            scope=self._scope,
            labels="float64_original",
            null_keys="exact_zero_admitted",
            unit_tolerance=UNIT_TOLERANCE,
            reservoir_rng="mt19937_64_seed_seq_uint64_low_high_only",
            scope_in_rng=False,
            read_order="ascending_id",
            native_sha256=native.binary_sha256,
        )
        self._identity_json = _canonical(identity)
        empty = torch.empty((0,), dtype=torch.int64)
        self._state = dict(
            ids=empty,
            keys=torch.empty((0, KEY_WIDTH), dtype=dtype),
            values=torch.empty((0, VALUE_WIDTH), dtype=dtype),
            times=torch.empty((0, 3), dtype=torch.int64),
            labels=torch.empty((0,), dtype=torch.float64),
        )
        self.seen = self.last_id = self.confirmed_at = 0
        self._rng = native.causal_reservoir_state(config.seed)

    def identity(self):
        return json.loads(self._identity_json)

    def fingerprint(self):
        return hashlib.sha256(self._identity_json.encode()).hexdigest()

    @property
    def size(self):
        return self._state["ids"].shape[0]

    def _slot_bytes(self):
        return (KEY_WIDTH + VALUE_WIDTH) * self.dtype.itemsize + 5 * 8

    def estimated_bytes(self, incoming):
        """Dos generaciones protegidas, entradas, propuesta y su copia de publicación."""
        after = min(self.seen + incoming, self.config.capacity)
        return self._slot_bytes() * (2 * self.size + incoming + 2 * after) + _METADATA_BYTES

    def _new(self, state, seen, last_id, confirmed_at, rng):
        result = object.__new__(type(self))
        result.config, result.dtype, result._native = self.config, self.dtype, self._native
        result._scope = self._scope
        result._identity_json = self._identity_json
        result._state, result._rng = state, rng
        result.seen, result.last_id, result.confirmed_at = seen, last_id, confirmed_at
        return result

    def records(self):
        state = self._state
        return tuple(
            CandidateRecord(identifier, decision, available, maturity, label)
            for identifier, (decision, available, maturity), label in zip(
                state["ids"].tolist(),
                state["times"].tolist(),
                state["labels"].tolist(),
                strict=True,
            )
        )

    def read_view(self):
        """Episodios ordenados por ID sin reordenar las plazas.

        `returns` es la conversión explícita de las etiquetas FP64 a la precisión de
        lectura. Las etiquetas guardadas no cambian.
        """
        order = torch.argsort(self._state["ids"], stable=True)
        view = {name: self._state[name].index_select(0, order) for name in _FIELDS}
        view["returns"] = view["labels"].to(self.dtype)
        if not bool(torch.isfinite(view["returns"]).all()):
            raise ValueError("Una etiqueta GRU no cabe en la precisión de lectura")
        return view

    def propose(self, episodes, *, confirmed_at):
        if type(episodes) is not CandidateEpisodes or type(confirmed_at) is not int:
            raise ValueError("La propuesta GRU necesita episodios identificados y un corte entero")
        fields = [getattr(episodes, name) for name in _FIELDS]
        _check_rows(*fields, self.dtype)
        ids, times = episodes.ids, episodes.times
        count = ids.shape[0]
        if (
            not 1 <= count <= MAX_INCOMING
            or count > MAX_SEEN - self.seen
            or confirmed_at < self.confirmed_at
            or int(ids[0]) <= self.last_id
            or (count > 1 and not bool((ids[1:] > ids[:-1]).all()))
            or bool((times[:, 2] > confirmed_at).any())
        ):
            raise ValueError("El lote GRU retrocede, repite IDs, madura tarde o excede su cupo")
        if self.estimated_bytes(count) > self.config.max_working_bytes:
            raise ValueError("La propuesta GRU supera el presupuesto de trabajo del banco")
        slots, rng = self._native.causal_reservoir_draws(
            self._rng, self.seen, self.config.capacity, count
        )
        size = self.size
        after = min(self.seen + count, self.config.capacity)
        writes = {slot: index for index, slot in enumerate(slots) if slot >= 0}
        if (
            len(slots) != count
            or not set(range(size, after)) <= set(writes)
            or max(writes, default=-1) >= after
        ):
            raise ValueError("El reservorio nativo devolvió plazas incompatibles")
        targets = torch.tensor(sorted(writes), dtype=torch.int64)
        sources = torch.tensor([writes[slot] for slot in sorted(writes)], dtype=torch.int64)
        state = {}
        for name, incoming in zip(_FIELDS, fields, strict=True):
            current = self._state[name]
            value = torch.empty((after, *current.shape[1:]), dtype=current.dtype)
            value[:size] = current
            value.index_copy_(0, targets, incoming.index_select(0, sources))
            state[name] = value
        return self._new(state, self.seen + count, int(ids[-1]), confirmed_at, rng)

    def snapshot(self):
        return dict(
            identity=self.identity(),
            **{name: self._state[name].clone() for name in _FIELDS},
            seen=self.seen,
            last_id=self.last_id,
            confirmed_at=self.confirmed_at,
            reservoir_rng=self._rng,
        )

    def restore(self, payload):
        if (
            not isinstance(payload, dict)
            or set(payload) != _STATE
            or _canonical(payload["identity"]) != self._identity_json
            or any(type(payload[key]) is not int for key in ("seen", "last_id", "confirmed_at"))
            or not isinstance(payload["reservoir_rng"], str)
        ):
            raise ValueError("El snapshot pertenece a otro contrato del banco GRU")
        seen, last_id, confirmed_at = (payload[key] for key in ("seen", "last_id", "confirmed_at"))
        state = {name: payload[name] for name in _FIELDS}
        _check_rows(*(state[name] for name in _FIELDS), self.dtype)
        ids, times = state["ids"], state["times"]
        size = ids.shape[0]
        if (
            not 0 <= seen <= MAX_SEEN
            or size != min(seen, self.config.capacity)
            or not seen <= last_id <= _INT64_MAX
            or confirmed_at < 0
            or (seen == 0 and (last_id != 0 or confirmed_at != 0))
            or (size and (int(ids.max()) > last_id or bool((times[:, 2] > confirmed_at).any())))
            or len(set(ids.tolist())) != size
            or (seen <= self.config.capacity and size > 1 and not bool((ids[1:] > ids[:-1]).all()))
        ):
            raise ValueError("El snapshot GRU no conserva capacidad, contadores, IDs o tiempos")
        rng = payload["reservoir_rng"]
        _, canonical = self._native.causal_reservoir_draws(rng, seen, self.config.capacity, 0)
        initial = self._native.causal_reservoir_state(self.config.seed)
        if canonical != rng or (seen <= self.config.capacity and rng != initial):
            raise ValueError("El estado del reservorio GRU no es canónico o no corresponde")
        owned = {name: value.clone() for name, value in state.items()}
        return self._new(owned, seen, last_id, confirmed_at, rng)
