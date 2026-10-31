"""Fuente madura N1, funcional y acotada, todavía independiente del ejecutor nativo."""

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from .salience_budget import _FLOW, MAX_COUNT, MAX_TIME, MatureWrite, _finite, _integer

KIND = "mature_dual_neuromodulation_source_v1"
_MARKETS = {"US", "CN"}
_RECORD_FIELDS = {"events", "window", "history", "cusum", "signal"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _rms(values):
    scale = max(map(abs, values), default=0.0)
    return (
        0.0
        if scale == 0
        else scale * math.sqrt(math.fsum((x / scale) ** 2 for x in values) / len(values))
    )


def _mean_nonnegative(values):
    scale = max(values, default=0.0)
    return 0.0 if scale == 0 else scale * (math.fsum(x / scale for x in values) / len(values))


def _unit(value, scale):
    if value == 0:
        return 0.0
    if value >= scale:
        return 1 / (1 + scale / value)
    ratio = value / scale
    return ratio / (1 + ratio)


def _clipped_ratio(value, scale, cap):
    if value <= scale:
        return min(cap, value / scale)
    inverse = scale / value
    return cap if inverse <= 1 / cap else 1 / inverse


@dataclass(frozen=True)
class NeuromodulationSourceConfig:
    window: int = 32
    min_events: int = 8
    expected_scale: float = 0.01
    scale_floor: float = 1e-6
    cusum_reference: float = 1.0
    cusum_slack: float = 0.05
    cusum_limit: float = 5.0
    max_rows: int = 40960
    source: str = "observed"
    seed: int = 0

    def __post_init__(self):
        _integer(self.window, 2, 256, "La ventana N1")
        _integer(self.min_events, 2, self.window, "El inicio N1")
        _integer(self.max_rows, 1, 40960, "Las filas maduras N1 por instante")
        _integer(self.seed, 0, 2**32 - 1, "La semilla N1")
        for name, low, high in (
            ("expected_scale", 1e-12, 1e6),
            ("scale_floor", 1e-12, 1e6),
            ("cusum_reference", 0.0, 8.0),
            ("cusum_slack", 0.0, 8.0),
            ("cusum_limit", 0.01, 64.0),
        ):
            value = _finite(getattr(self, name), name)
            if not low <= value <= high:
                raise ValueError(f"{name} no pertenece al intervalo N1 permitido")
            object.__setattr__(self, name, value)
        if self.source not in ("observed", "shuffled_past_prefix"):
            raise ValueError("La fuente N1 no está declarada")

    def identity(self):
        return dict(
            kind=KIND,
            configuration=asdict(self),
            unit="one_market_event_per_publication_instant",
            expected="window_rms_of_mature_target_rms_over_fixed_scale_plus_rms",
            unexpected="clipped_standardized_absolute_error_cusum",
            same_instant="predictions_use_state_before_all_publications_at_that_instant",
            eligibility="native_registry_must_prove_unique_issued_mature_outcomes",
        )


@dataclass(frozen=True)
class NeuromodulationSnapshot:
    configuration_id: str
    state_id: str
    generation: int
    cutoff: int | None
    prediction_at: int
    selection: str
    flow_ids: tuple[str, ...]
    signals: tuple[tuple[float, float], ...]

    def array(self):
        return np.array(self.signals, dtype=np.float64)

    def record(self):
        return asdict(self)


class NeuromodulationSource:
    """Estadísticas maduras, sin consultar pendientes ni reajustar el predictor."""

    def __init__(self, config):
        if type(config) is not NeuromodulationSourceConfig:
            raise ValueError("La fuente N1 necesita su configuración identificada")
        self._config = config
        self.generation, self.cutoff, self.applied_total = 0, None, 0
        self._base, self._current, self._rows = {}, {}, ()

    @property
    def config(self):
        return self._config

    def _validated(self, rows, cutoff):
        if type(rows) not in (tuple, list) or len(rows) > self.config.max_rows:
            raise ValueError("La publicación N1 supera su secuencia de filas permitida")
        keys = set()
        for row in rows:
            if (
                type(row) is not MatureWrite
                or type(row.flow_id) is not str
                or _FLOW.fullmatch(row.flow_id) is None
            ):
                raise ValueError("N1 necesita identidades maduras US/CN válidas")
            _integer(row.prediction_at, 0, MAX_TIME, "La decisión N1")
            _integer(row.available_at, row.prediction_at + 1, cutoff, "La disponibilidad N1")
            if row.key in keys:
                raise ValueError("N1 repite una predicción en el mismo instante")
            keys.add(row.key)
        result = []
        for row in rows:
            issued, target = (
                _finite(row.issued, "El punto emitido"),
                _finite(row.target, "El residuo maduro"),
            )
            _finite(target - issued, "El error financiero maduro")
            result.append(
                MatureWrite(row.flow_id, row.prediction_at, row.available_at, issued, target)
            )
        return tuple(sorted(result, key=lambda r: (r.available_at, r.prediction_at, r.flow_id)))

    def _prior_pair(self, history, market, events):
        if not history:
            return [0.0, 0.0]
        key = int.from_bytes(hashlib.sha256(market.encode()).digest()[:4], "little")
        seed = np.random.SeedSequence([self.config.seed, key, events])
        return list(history[int(np.random.default_rng(seed).integers(len(history)))])

    def _following(self, base, rows):
        following = copy.deepcopy(base)
        groups = {}
        for row in rows:
            groups.setdefault(row.flow_id.split("/", 1)[0], []).append(row)
        c = self.config
        for market, group in sorted(groups.items()):
            old = base.get(
                market, dict(events=0, window=[], history=[], cusum=0.0, signal=[0.0, 0.0])
            )
            if old["events"] == MAX_COUNT:
                raise ValueError("El contador N1 no admite otro evento")
            errors = [abs(r.target - r.issued) for r in group]
            cusum = 0.0
            if old["events"] >= c.min_events:
                scale = max(c.scale_floor, _rms([r[1] for r in old["window"]]))
                k = c.cusum_reference + c.cusum_slack
                score = _clipped_ratio(_mean_nonnegative(errors), scale, k + c.cusum_limit)
                cusum = min(c.cusum_limit, max(0.0, old["cusum"] + score - k))
            window = (old["window"] + [[_rms([r.target for r in group]), _rms(errors)]])[
                -c.window :
            ]
            events = old["events"] + 1
            expected = (
                _unit(_rms([r[0] for r in window]), c.expected_scale)
                if events >= c.min_events
                else 0.0
            )
            actual = [expected, cusum / c.cusum_limit]
            selected = (
                actual
                if c.source == "observed"
                else self._prior_pair(old["history"][-c.window :], market, old["events"])
            )
            following[market] = dict(
                events=events,
                window=window,
                history=(old["history"] + [actual])[-(c.window + 1) :],
                cusum=cusum,
                signal=selected,
            )
        return following

    def advance(self, rows, *, cutoff, generation, applied_total):
        _integer(cutoff, 0, MAX_TIME, "El corte N1")
        _integer(generation, 1, MAX_COUNT, "La generación N1")
        _integer(applied_total, 0, MAX_COUNT, "El total aplicado N1")
        if (
            type(rows) not in (tuple, list)
            or generation != self.generation + 1
            or applied_total != self.applied_total + len(rows)
            or (self.cutoff is not None and cutoff < self.cutoff)
        ):
            raise ValueError("La publicación N1 no concilia corte, generación o contador")
        same = cutoff == self.cutoff
        base = copy.deepcopy(self._base if same else self._current)
        all_rows = self._validated((self._rows if same else ()) + tuple(rows), cutoff)
        result = NeuromodulationSource(self.config)
        result.generation, result.cutoff, result.applied_total = generation, cutoff, applied_total
        result._base, result._rows = base, all_rows
        result._current = self._following(base, all_rows)
        return result

    def snapshot(self, flows, *, prediction_at):
        _integer(prediction_at, 0, MAX_TIME, "La decisión de consulta N1")
        if (
            type(flows) not in (tuple, list)
            or not 1 <= len(flows) <= 8192
            or any(type(f) is not str or _FLOW.fullmatch(f) is None for f in flows)
            or len(set(flows)) != len(flows)
        ):
            raise ValueError("El snapshot N1 necesita flujos válidos y únicos")
        if self.cutoff is not None and prediction_at < self.cutoff:
            raise ValueError("El snapshot N1 no puede consultar un estado del futuro")
        same = prediction_at == self.cutoff
        source = self._base if same else self._current
        values = tuple(
            tuple(source.get(f.split("/", 1)[0], {}).get("signal", [0.0, 0.0])) for f in flows
        )
        return NeuromodulationSnapshot(
            _digest(self.config.identity()),
            self.export()["sha256"],
            self.generation,
            self.cutoff,
            prediction_at,
            "before_instant" if same else "after_publication",
            tuple(flows),
            values,
        )

    def export(self):
        value = dict(
            schema_version=1,
            kind=KIND,
            configuration=self.config.identity(),
            generation=self.generation,
            cutoff=self.cutoff,
            applied_total=self.applied_total,
            base=copy.deepcopy(self._base),
            rows=[asdict(r) for r in self._rows],
        )
        return dict(value, sha256=_digest(value))

    def _checked_base(self, base):
        if type(base) is not dict or not set(base) <= _MARKETS:
            raise ValueError("El estado N1 no conserva sus mercados")
        result = copy.deepcopy(base)
        c = self.config
        for market, row in result.items():
            if type(row) is not dict or set(row) != _RECORD_FIELDS:
                raise ValueError("El estado N1 no conserva sus campos")
            _integer(row["events"], 1, MAX_COUNT, "Los eventos N1")
            for key, length, high in (
                ("window", min(row["events"], c.window), None),
                ("history", min(row["events"], c.window + 1), 1.0),
            ):
                values = row[key]
                if (
                    type(values) is not list
                    or len(values) != length
                    or any(
                        type(v) is not list
                        or len(v) != 2
                        or any(
                            type(x) is not float
                            or not math.isfinite(x)
                            or x < 0
                            or (high is not None and x > high)
                            for x in v
                        )
                        for v in values
                    )
                ):
                    raise ValueError("La ventana o historial N1 no es válido")
            cusum = row["cusum"]
            if (
                type(cusum) is not float
                or not math.isfinite(cusum)
                or not 0 <= cusum <= c.cusum_limit
                or (row["events"] <= c.min_events and cusum != 0)
            ):
                raise ValueError("El CUSUM N1 no es coherente")
            expected = (
                _unit(_rms([r[0] for r in row["window"]]), c.expected_scale)
                if row["events"] >= c.min_events
                else 0.0
            )
            actual = [expected, cusum / c.cusum_limit]
            selected = (
                actual
                if c.source == "observed"
                else self._prior_pair(row["history"][:-1], market, row["events"] - 1)
            )
            if row["history"][-1] != actual or row["signal"] != selected:
                raise ValueError("Las señales N1 no concilian con sus estadísticas")
        return result

    @classmethod
    def restore(cls, config, payload):
        result = cls(config)
        fields = {
            "schema_version",
            "kind",
            "configuration",
            "generation",
            "cutoff",
            "applied_total",
            "base",
            "rows",
            "sha256",
        }
        if type(payload) is not dict or set(payload) != fields:
            raise ValueError("La fuente N1 no conserva su esquema")
        unsigned = {k: v for k, v in payload.items() if k != "sha256"}
        if (
            payload["schema_version"] != 1
            or type(payload["schema_version"]) is not int
            or payload["kind"] != KIND
            or payload["configuration"] != config.identity()
            or payload["sha256"] != _digest(unsigned)
        ):
            raise ValueError("El estado N1 no corresponde a su configuración o huella")
        _integer(payload["generation"], 0, MAX_COUNT, "La generación N1")
        _integer(payload["applied_total"], 0, MAX_COUNT, "El total N1")
        if payload["generation"] == 0:
            if payload != result.export():
                raise ValueError("El estado inicial N1 no está vacío")
            return result
        _integer(payload["cutoff"], 0, MAX_TIME, "El corte N1")
        if type(payload["rows"]) is not list or any(
            type(r) is not dict or set(r) != set(MatureWrite.__dataclass_fields__)
            for r in payload["rows"]
        ):
            raise ValueError("Las filas guardadas N1 no conservan el contrato")
        rows = result._validated(
            tuple(MatureWrite(**r) for r in payload["rows"]), payload["cutoff"]
        )
        base = result._checked_base(payload["base"])
        if sum(r["events"] for r in base.values()) > payload["applied_total"] - len(rows):
            raise ValueError("Las estadísticas N1 exceden los resultados aplicados")
        result.generation, result.cutoff, result.applied_total = (
            payload["generation"],
            payload["cutoff"],
            payload["applied_total"],
        )
        result._base, result._rows, result._current = base, rows, result._following(base, rows)
        return result
