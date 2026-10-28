"""Selección E3 de escrituras maduras con cupos iguales y escalas anteriores por activo.

El selector no consulta datos, publica bancos ni verifica por sí solo que una predicción
fue emitida. Su consumidor debe enlazar las filas con su registro de emisiones y entregar
la cohorte completa, después de emitir las decisiones del evento. La selección devuelve
otro estado sin modificar el anterior.
"""

import hashlib
import heapq
import json
import math
from dataclasses import asdict, dataclass
from types import MappingProxyType

from .flow_cursors import _FLOW, MAX_FLOWS

KIND = "salience_write_budget_v1"
MAX_ROWS = 8192
MAX_COUNT = 2**53 - 1
MAX_TIME = 2**63 - 1
RULES = ("standardized_error", "absolute_error", "uniform")


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} debe ser un entero entre {low} y {high}")


def _finite(value, name):
    if type(value) not in (float, int):
        raise ValueError(f"{name} debe ser un número finito")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} no cabe en FP64") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(payload):
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


@dataclass(frozen=True)
class SalienceBudgetConfig:
    """Regla y cupo de cada mercado en una publicación completa, fijados antes de evaluar."""

    rule: str = "standardized_error"
    writes_per_market: int = 64
    min_history: int = 4
    initial_scale: float = 0.01
    scale_floor: float = 1e-8
    seed: int = 0

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError("E3 necesita error estandarizado, absoluto o selección uniforme")
        _integer(self.writes_per_market, 0, MAX_ROWS, "El cupo")
        _integer(self.min_history, 1, MAX_COUNT, "El historial mínimo")
        _integer(self.seed, 0, 2**32 - 1, "La semilla")
        initial = _finite(self.initial_scale, "La escala inicial")
        floor = _finite(self.scale_floor, "El suelo de escala")
        if not 0 < floor <= initial:
            raise ValueError("Las escalas deben cumplir 0 < suelo <= inicial")
        object.__setattr__(self, "initial_scale", initial)
        object.__setattr__(self, "scale_floor", floor)

    def identity(self):
        return dict(
            asdict(self),
            kind=KIND,
            cohort="all_mature_candidates_of_each_market_in_one_publication",
            scale="prior_asset_root_mean_square_of_issued_prediction_errors",
            order="flow_id_then_prediction_at",
            statistics="all_eligible_rows_after_selection",
            uniform="sha256_domain_seed_cutoff_and_real_decision_identity",
        )


@dataclass(frozen=True)
class MatureWrite:
    """Identidad, disponibilidad, punto emitido y etiqueta que entrega el consumidor."""

    flow_id: str
    prediction_at: int
    available_at: int
    issued: float
    target: float

    @property
    def key(self):
        return self.flow_id, self.prediction_at


@dataclass(frozen=True)
class WriteSelection:
    """Identidades elegidas y diagnóstico acotado de todos los candidatos elegibles."""

    selected: tuple
    scores: tuple
    scales: tuple


class MatureWriteBudget:
    """Estado agregado E3 de hasta 8.192 activos, sin episodios ni objetivos históricos."""

    def __init__(self, config, *, records=None, cursor=None, cohorts=0, selected=0):
        if type(config) is not SalienceBudgetConfig:
            raise ValueError("E3 necesita una configuración identificada")
        if records is None:
            records = {}
        if type(records) is not dict or len(records) > MAX_FLOWS:
            raise ValueError("El estado supera el límite de activos o no es un diccionario")
        _integer(cohorts, 0, MAX_COUNT, "Las publicaciones")
        _integer(selected, 0, MAX_COUNT, "Las escrituras seleccionadas")
        if cursor is not None:
            _integer(cursor, 0, MAX_TIME, "El corte confirmado")
        checked = {}
        for flow, values in records.items():
            if not isinstance(flow, str) or _FLOW.fullmatch(flow) is None:
                raise ValueError("El estado no conserva un activo US o CN válido")
            if not isinstance(values, (tuple, list)) or len(values) != 4:
                raise ValueError("Cada activo necesita cuenta, RMS y dos cursores")
            count, rms, decision, available = values
            _integer(count, 1, MAX_COUNT, "La cuenta por activo")
            _integer(decision, 0, MAX_TIME, "La decisión anterior")
            _integer(available, decision + 1, MAX_TIME, "La disponibilidad anterior")
            rms = _finite(rms, "El RMS anterior")
            if rms < 0 or cursor is None or available > cursor:
                raise ValueError("La escala o la disponibilidad contradice al estado confirmado")
            checked[flow] = (count, rms, decision, available)
        observed = sum(value[0] for value in checked.values())
        if (
            observed > MAX_COUNT
            or selected > observed
            or cohorts > observed
            or (cursor is None) != (observed == 0)
            or (cohorts == 0) != (observed == 0)
        ):
            raise ValueError("Las cuentas y el corte E3 no son coherentes")
        self.config = config
        self._records = MappingProxyType(checked)
        self.cursor, self.cohorts, self.selected, self.observed = (
            cursor,
            cohorts,
            selected,
            observed,
        )

    def _cohort(self, cohort, cutoff, *, ordered=True):
        """Validar toda la disponibilidad antes de acceder al primer valor financiero."""
        _integer(cutoff, 0, MAX_TIME, "El corte de publicación")
        if not isinstance(cohort, (list, tuple)) or len(cohort) > MAX_ROWS:
            raise ValueError("E3 necesita una secuencia de hasta 8.192 candidatos")
        if (
            ordered
            and self.cursor is not None
            and (cutoff < self.cursor or (cohort and cutoff == self.cursor))
        ):
            raise ValueError("Una publicación E3 no puede repetirse ni retroceder")
        seen = set()
        for item in cohort:
            if type(item) is not MatureWrite or not isinstance(item.flow_id, str):
                raise ValueError("Cada candidato necesita una fila madura tipada")
            if _FLOW.fullmatch(item.flow_id) is None:
                raise ValueError("El candidato no identifica un activo US o CN válido")
            _integer(item.prediction_at, 0, MAX_TIME, "La decisión")
            _integer(item.available_at, item.prediction_at + 1, cutoff, "La disponibilidad madura")
            previous = self._records.get(item.flow_id)
            if item.key in seen or (
                ordered and previous is not None and item.prediction_at <= previous[2]
            ):
                raise ValueError("E3 repite o retrocede una decisión ya procesada")
            seen.add(item.key)
        if len(set(self._records) | {item.flow_id for item in cohort}) > MAX_FLOWS:
            raise ValueError("La publicación supera los 8.192 activos, sin expulsión automática")
        return tuple(sorted(cohort, key=lambda item: item.key))

    def _score(self, item, error, cutoff):
        previous = self._records.get(item.flow_id)
        scale = (
            max(previous[1], self.config.scale_floor)
            if previous is not None and previous[0] >= self.config.min_history
            else self.config.initial_scale
        )
        score = abs(error)
        if self.config.rule == "standardized_error":
            score /= scale
        elif self.config.rule == "uniform":
            identity = [KIND, self.config.seed, cutoff, *item.key]
            # Se ordenan los bytes completos. No se reduce una prioridad de 256 bits a FP64.
            score = hashlib.sha256(_canonical(identity).encode()).digest()
        if not isinstance(score, bytes) and not math.isfinite(score):
            raise ValueError("La puntuación E3 ha desbordado su precisión")
        return score, scale

    def select(self, cohort, *, cutoff):
        """Elegir IDs reales y proponer el estado posterior, sin modificar este estado.

        Cada mercado conserva min(cupo, candidatos). La cohorte completa se puntúa con el
        prefijo anterior. Después, todos sus errores avanzan sus escalas, incluidos los que
        no se escriben. Dos llamadas con el mismo corte no pueden repartir otro cupo.
        """
        return self._select(self._cohort(cohort, cutoff), cutoff)

    def _select(self, cohort, cutoff):
        """Puntuación común tras validar el consumidor su orden y sus candidatos."""
        if not cohort:
            return WriteSelection((), (), ()), self
        grouped, scores, scales = {}, [], []
        records = dict(self._records)
        for item in cohort:
            issued = _finite(item.issued, "La predicción emitida")
            target = _finite(item.target, "La etiqueta madura")
            error = target - issued
            if not math.isfinite(error):
                raise ValueError("El error emitido ha desbordado su precisión")
            score, scale = self._score(item, error, cutoff)
            # Para uniform, la mayor prioridad hash gana y los empates usan la identidad.
            priority = -int.from_bytes(score) if isinstance(score, bytes) else -score
            grouped.setdefault(item.flow_id.split("/", 1)[0], []).append((priority, item.key))
            scores.append((item.key, score.hex() if isinstance(score, bytes) else score))
            scales.append((item.key, scale))
            count, rms, _, _ = records.get(item.flow_id, (0, 0.0, 0, 0))
            count += 1
            rms = math.hypot(rms * math.sqrt((count - 1) / count), error / math.sqrt(count))
            records[item.flow_id] = (count, rms, item.prediction_at, item.available_at)
        selected = tuple(
            sorted(
                key
                for group in grouped.values()
                for _, key in heapq.nsmallest(self.config.writes_per_market, group)
            )
        )
        updated = MatureWriteBudget(
            self.config,
            records=records,
            cursor=cutoff,
            cohorts=self.cohorts + 1,
            selected=self.selected + len(selected),
        )
        return WriteSelection(selected, tuple(scores), tuple(scales)), updated

    def export(self):
        """Exportar JSON independiente. Su huella detecta cambios, no autentica emisiones."""
        payload = dict(
            schema_version=1,
            kind=KIND,
            configuration=self.config.identity(),
            cursor=self.cursor,
            cohorts=self.cohorts,
            selected=self.selected,
            records={key: list(value) for key, value in sorted(self._records.items())},
        )
        return dict(payload, sha256=_digest(payload))

    @classmethod
    def restore(cls, config, payload):
        fields = {
            "schema_version",
            "kind",
            "configuration",
            "cursor",
            "cohorts",
            "selected",
            "records",
            "sha256",
        }
        if (
            type(config) is not SalienceBudgetConfig
            or type(payload) is not dict
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["kind"] != KIND
            or _canonical(payload["configuration"]) != _canonical(config.identity())
        ):
            raise ValueError("El estado no conserva el esquema y la configuración E3")
        restored = cls(
            config,
            records=payload["records"],
            cursor=payload["cursor"],
            cohorts=payload["cohorts"],
            selected=payload["selected"],
        )
        if _canonical(restored.export()) != _canonical(payload):
            raise ValueError("Las huellas o los valores exportados E3 han cambiado")
        return restored


class CommittedWriteBudget:
    """Cupos por generación nativa, con estadísticas compartidas con el selector E3.

    El ejecutor debe acreditar que cada resultado pertenece a una predicción pendiente
    emitida y se aplica una sola vez. Esta estadística no conserva todos sus IDs ni puede
    sustituir ese registro. Admite etiquetas tardías y dos publicaciones con el mismo
    corte. El cupo es por publicación, no por sesión del calendario.
    """

    KIND = "salience_committed_publication_v1"

    def __init__(
        self,
        config,
        *,
        statistics=None,
        generation=0,
        cutoff=None,
        last_observed=0,
        last_selected=(),
    ):
        if type(config) is not SalienceBudgetConfig:
            raise ValueError("La publicación E3 necesita una configuración identificada")
        statistics = MatureWriteBudget(config) if statistics is None else statistics
        if type(statistics) is not MatureWriteBudget or statistics.config != config:
            raise ValueError("Las estadísticas E3 pertenecen a otra configuración")
        _integer(generation, 0, MAX_COUNT, "La generación E3")
        _integer(last_observed, 0, MAX_ROWS, "Los candidatos de la publicación")
        if cutoff is not None:
            _integer(cutoff, 0, MAX_TIME, "El corte E3")
        if (
            (generation == 0) != (cutoff is None)
            or statistics.cohorts > generation
            or last_observed > statistics.observed
            or (statistics.cursor is not None and statistics.cursor > cutoff)
            or not isinstance(last_selected, (list, tuple))
            or len(last_selected) > min(last_observed, statistics.selected)
        ):
            raise ValueError("La publicación y sus cuentas E3 no concilian")
        keys, counts = [], {}
        for key in last_selected:
            if not isinstance(key, (list, tuple)) or len(key) != 2:
                raise ValueError("La selección E3 necesita identidades reales")
            flow, decision = key
            if not isinstance(flow, str) or _FLOW.fullmatch(flow) is None:
                raise ValueError("La selección E3 no identifica un activo válido")
            _integer(decision, 0, cutoff - 1, "La decisión elegida")
            keys.append((flow, decision))
            market = flow.split("/", 1)[0]
            counts[market] = counts.get(market, 0) + 1
        if keys != sorted(set(keys)) or any(
            count > config.writes_per_market for count in counts.values()
        ):
            raise ValueError("La selección E3 repite identidades o supera el cupo")
        self.config, self.statistics = config, statistics
        self.generation, self.cutoff = generation, cutoff
        self.last_observed, self.last_selected = last_observed, tuple(keys)

    def select(self, cohort, *, cutoff, generation):
        """Proponer una publicación siguiente sin modificar el estado confirmado.

        Los candidatos son exactamente los resultados únicos admitidos por el ejecutor.
        Todos avanzan el RMS. Solo los elegidos escriben la corrección. La generación
        avanza también sin resultados y evita que un reintento renueve el cupo.
        """
        _integer(generation, 1, MAX_COUNT, "La generación E3")
        _integer(cutoff, 0, MAX_TIME, "El corte E3")
        if generation != self.generation + 1 or (self.cutoff is not None and cutoff < self.cutoff):
            raise ValueError("La publicación E3 repite, salta o retrocede una generación")
        rows = self.statistics._cohort(cohort, cutoff, ordered=False)
        selection, statistics = self.statistics._select(rows, cutoff)
        return selection, CommittedWriteBudget(
            self.config,
            statistics=statistics,
            generation=generation,
            cutoff=cutoff,
            last_observed=len(rows),
            last_selected=selection.selected,
        )

    def export(self):
        payload = dict(
            schema_version=1,
            kind=self.KIND,
            configuration=self.config.identity(),
            generation=self.generation,
            cutoff=self.cutoff,
            statistics=self.statistics.export(),
            last=dict(
                observed=self.last_observed, selected=[list(key) for key in self.last_selected]
            ),
        )
        return dict(payload, sha256=_digest(payload))

    @classmethod
    def restore(cls, config, payload):
        fields = {
            "schema_version",
            "kind",
            "configuration",
            "generation",
            "cutoff",
            "statistics",
            "last",
            "sha256",
        }
        if (
            type(config) is not SalienceBudgetConfig
            or type(payload) is not dict
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["kind"] != cls.KIND
            or payload["configuration"] != config.identity()
            or type(payload["last"]) is not dict
            or set(payload["last"]) != {"observed", "selected"}
            or payload["sha256"] != _digest({k: v for k, v in payload.items() if k != "sha256"})
        ):
            raise ValueError("El estado de la publicación E3 no conserva su contrato o huella")
        return cls(
            config,
            statistics=MatureWriteBudget.restore(config, payload["statistics"]),
            generation=payload["generation"],
            cutoff=payload["cutoff"],
            last_observed=payload["last"]["observed"],
            last_selected=payload["last"]["selected"],
        )
