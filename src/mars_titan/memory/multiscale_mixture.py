"""Mezcla de correctores con decaimiento de calendario y errores emitidos maduros (E1).

Los expertos son copias del mismo backend. Sus coeficientes decaen con semividas distintas.
Hedge actualiza la mezcla con pérdidas acotadas de las predicciones guardadas al decidir.
El retraso financiero y el recorte impiden atribuir directamente la garantía del artículo.

R2 de #786 (`mature_cms_v1`) adapta el sistema de memoria continua de Nested Learning
(Behrouz y otros, NeurIPS 2025), cuyos bloques se actualizan con frecuencias distintas. Aquí
cada experto escribe con su periodo de calendario UTC: la sesión, la semana ISO o el mes. Un
experto lento guarda en orden canónico los resultados maduros de su periodo abierto y, en la
primera escritura de un periodo posterior, aplica un único paso proximal sobre la pérdida
media de los eventos del periodo cerrado. Las lecturas solo usan los pasos aplicados, así que
ninguna velocidad lee un error que no haya madurado antes de la escritura que lo aplica.
"""

import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

import torch

from .associative_memory import (
    MAX_COHORT,
    AssociativeMemoryConfig,
    MatureCorrection,
    MatureFeedback,
    _canonical,
    _check_keys,
    _check_matrix,
    _check_vector,
    _finite,
    _matrix_digest,
    canonical_order,
)

KIND = "multiscale_mature_mixture_v1"
DAY_US = 86_400_000_000
MAX_EXPERTS = 4
WEIGHTING = ("hedge", "uniform")
# R2 de #786. Sin periodos declarados todos los expertos escriben en cada evento, como en E1.
WRITE_PERIODS = ("session", "week", "month")
CONTINUUM = "mature_cms_v1"
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def declaration():
    return dict(
        kind=KIND,
        experts=[1, MAX_EXPERTS],
        half_lives_days=[0.01, 1e6],
        weighting=list(WEIGHTING),
        weight_rate=[0.0, 100.0],
        loss_scale=[1e-6, 10.0],
        clock="publication_cutoff_in_utc_microseconds",
        discrete_forgetting=0.0,
        write_periods=list(WRITE_PERIODS),
    )


@dataclass(frozen=True)
class MultiscaleMixtureConfig:
    """Semividas de la parte homogénea y regla de mezcla fijadas antes de evaluar."""

    half_lives_days: tuple[float, ...] = (7.0, 91.0, 365.0)
    weighting: str = "hedge"
    weight_rate: float = 1.0
    loss_scale: float = 0.02
    # R2 de #786: el periodo de escritura de cada experto. Sin declararlo la identidad de E1
    # no cambia y todos escriben en cada evento.
    write_periods: tuple[str, ...] | None = None

    def __post_init__(self):
        if (
            not isinstance(self.half_lives_days, (list, tuple))
            or not 1 <= len(self.half_lives_days) <= MAX_EXPERTS
        ):
            raise ValueError("E1 necesita de una a cuatro semividas declaradas")
        half_lives = tuple(
            _finite(value, "La semivida en días", 0.01, 1e6) for value in self.half_lives_days
        )
        if tuple(sorted(half_lives)) != half_lives or self.weighting not in WEIGHTING:
            raise ValueError("Las semividas deben estar ordenadas y la mezcla ser hedge o uniform")
        rate = _finite(self.weight_rate, "La tasa de la mezcla", 0.0, 100.0)
        if self.weighting == "uniform" and rate != 0.0:
            raise ValueError("La mezcla uniforme declara tasa cero")
        object.__setattr__(self, "half_lives_days", half_lives)
        object.__setattr__(self, "weight_rate", rate)
        object.__setattr__(
            self, "loss_scale", _finite(self.loss_scale, "La escala de pérdida", 1e-6, 10.0)
        )
        periods = self.write_periods
        if periods is not None:
            # El primer experto escribe en cada sesión y los demás van de rápido a lento. Con
            # todos en la sesión es E1 y se declara sin periodos, así cada cálculo tiene una
            # sola identidad.
            if (
                not isinstance(periods, (list, tuple))
                or len(periods) != len(half_lives)
                or any(type(period) is not str or period not in WRITE_PERIODS for period in periods)
                or periods[0] != "session"
                or list(periods) != sorted(periods, key=WRITE_PERIODS.index)
                or set(periods) == {"session"}
            ):
                raise ValueError(
                    "R2 declara un periodo session, week o month por experto, ordenados, con el "
                    "primero en session y alguno más lento"
                )
            object.__setattr__(self, "write_periods", tuple(periods))

    def identity(self):
        identity = dict(
            asdict(self),
            kind=KIND,
            clock="utc_microseconds_at_mature_publication",
            decay="all_backend_coefficients_times_exp2_minus_elapsed_over_half_life",
            loss="canonical_cohort_mean_of_min_abs_issued_expert_minus_mature_core_residual_over_scale_1",
            read="immutable_calendar_decay_then_convex_mixture",
            initial_weights="uniform",
            bound="no_regret_guarantee_claimed_for_delayed_financial_predictions",
        )
        if self.write_periods is None:
            identity.pop("write_periods")
        else:
            identity.update(
                write_periods=list(self.write_periods),
                continuum=CONTINUUM,
                period_clock="utc_calendar_of_each_write_cutoff_iso_weeks_from_monday",
                closed_period="one_proximal_step_on_the_mean_loss_of_its_events_at_the_first_later_write",
                open_period="kept_in_canonical_order_and_never_read",
            )
        return identity


def _clock(value):
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError("El reloj necesita microsegundos UTC int64")


def _expert_bytes(correction, mixture):
    """Bytes de coeficientes y logpesos, sin tablas derivadas ni cola pendiente."""
    size = correction.memory.key_size
    if correction.areas is None:
        coefficients = size * correction.memory.value_size
    else:
        width = correction.areas.feature_size(size)
        coefficients = correction.areas.areas * width * correction.memory.value_size
        if correction.query_bridge is not None:
            coefficients += correction.areas.areas * width * width
    experts = len(mixture.half_lives_days)
    return (experts * coefficients + experts) * 8


def state_tensor_bytes(correction, mixture):
    """Bytes de coeficientes y logpesos y, con R2, la cota de sus periodos abiertos."""
    total = _expert_bytes(correction, mixture)
    if mixture.write_periods is not None:
        # Cada periodo abierto guarda como mucho MAX_COHORT filas con tres enteros, la clave,
        # el valor y el peso.
        slow = sum(period != "session" for period in mixture.write_periods)
        total += slow * MAX_COHORT * (3 + correction.memory.key_size + 2) * 8
    return total


def period_index(period, at):
    """Periodo de calendario UTC de un corte: la semana ISO, que empieza el lunes, o el mes."""
    _clock(at)
    if period == "week":
        # El 1 de enero de 1970 fue jueves. Sumar tres días alinea el corte de semana al lunes.
        return (at // DAY_US + 3) // 7
    if period != "month":
        raise ValueError("Solo week y month tienen periodo de calendario")
    try:
        moment = EPOCH + timedelta(microseconds=at)
    except OverflowError as error:
        raise ValueError("El corte queda fuera del calendario") from error
    return moment.year * 12 + moment.month - 1


_FEEDBACK = ("ids", "decision_at", "available_at", "keys", "values", "weights")


def _tensor_digest(value):
    kind = "<i8" if value.dtype == torch.int64 else "<f8"
    digest = hashlib.sha256(str((kind, tuple(value.shape))).encode())
    digest.update(value.detach().cpu().contiguous().numpy().astype(kind, copy=False).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PendingPeriod:
    """Resultados maduros del periodo abierto de un experto lento, en orden canónico.

    Los pesos de cada evento suman uno, así que en total suman `events`.
    """

    period: int
    events: int
    feedback: MatureFeedback

    @property
    def rows(self):
        return self.feedback.ids.shape[0]

    def key(self, index):
        feedback = self.feedback
        return (
            int(feedback.available_at[index]),
            int(feedback.decision_at[index]),
            int(feedback.ids[index]),
        )

    def closed(self):
        """El periodo como una escritura: cada evento pesa 1/events y los pesos suman uno."""
        values = {name: getattr(self.feedback, name) for name in _FEEDBACK}
        values["weights"] = values["weights"] / self.events
        return MatureFeedback(**values)

    @staticmethod
    def extend(waiting, feedback, order, period):
        """Añadir un evento en orden canónico al periodo abierto, o empezarlo si es None."""
        index = torch.tensor(order, dtype=torch.int64)
        parts = {name: getattr(feedback, name).index_select(0, index) for name in _FEEDBACK}
        if waiting is not None:
            parts = {
                name: torch.cat([getattr(waiting.feedback, name), value])
                for name, value in parts.items()
            }
        if parts["ids"].shape[0] > MAX_COHORT:
            raise ValueError(f"Un periodo de R2 acumula como máximo {MAX_COHORT} resultados")
        events = 1 if waiting is None else waiting.events + 1
        return PendingPeriod(period, events, MatureFeedback(**parts))

    def export(self):
        return dict(
            period=self.period,
            events=self.events,
            **{name: getattr(self.feedback, name).clone() for name in _FEEDBACK},
        )

    def digest(self):
        return dict(
            period=self.period,
            events=self.events,
            **{name: _tensor_digest(getattr(self.feedback, name)) for name in _FEEDBACK},
        )

    @staticmethod
    def restore(payload, key_size):
        if not isinstance(payload, dict) or set(payload) != {"period", "events", *_FEEDBACK}:
            raise ValueError("El periodo abierto de R2 no conserva todos sus campos")
        rows = payload["ids"].shape[0] if isinstance(payload["ids"], torch.Tensor) else -1
        if not 1 <= rows <= MAX_COHORT:
            raise ValueError(f"El periodo abierto de R2 necesita entre 1 y {MAX_COHORT} resultados")
        for name in ("ids", "decision_at", "available_at"):
            _check_vector(payload[name], f"El campo {name} del periodo", rows, torch.int64)
        _check_matrix(payload["keys"], "Las claves del periodo", rows, key_size)
        _check_matrix(payload["values"], "Los valores del periodo", rows, 1)
        _check_vector(payload["weights"], "Los pesos del periodo", rows, torch.float64)
        if type(payload["period"]) is not int or type(payload["events"]) is not int:
            raise ValueError("El periodo y sus eventos deben ser enteros")
        return PendingPeriod(
            payload["period"],
            payload["events"],
            MatureFeedback(**{name: payload[name].clone() for name in _FEEDBACK}),
        )


def _backend_matches(memory, correction):
    from .area_memory import AreaMemory
    from .associative_memory import AssociativeMemory
    from .role_binding import RoleBoundMemory

    if correction.role_binding is not None:
        return (
            type(memory) is RoleBoundMemory
            and memory.correction == correction
            and memory.correction.role_binding is correction.role_binding
        )
    if correction.areas is not None:
        return (
            type(memory) is AreaMemory
            and memory.config == correction.memory
            and memory.layout == correction.areas
            and memory.query_bridge == correction.query_bridge
        )
    return type(memory) is AssociativeMemory and memory.config == correction.memory


class MultiscaleMemory:
    """Expertos, pesos de mezcla y reloj confirmados. Las lecturas no publican estado."""

    def __init__(
        self,
        correction,
        mixture,
        *,
        experts=None,
        log_weights=None,
        clock=None,
        cohorts=0,
        pending=None,
    ):
        if (
            type(correction) is not MatureCorrection
            or type(mixture) is not MultiscaleMixtureConfig
            or type(correction.memory) is not AssociativeMemoryConfig
            or correction.memory.rule not in ("delta", "proximal")
            or correction.memory.forgetting != 0.0
            or getattr(correction, "multiscale", None) is not None
        ):
            raise ValueError(
                "E1 necesita un corrector delta/proximal sin mezcla ni olvido discreto"
            )
        if mixture.write_periods is not None and (
            correction.memory.rule != "proximal"
            or correction.areas is not None
            or correction.role_binding is not None
        ):
            raise ValueError("R2 escribe sus periodos con la regla proximal, sin áreas ni roles")
        count = len(mixture.half_lives_days)
        if experts is not None and (
            not isinstance(experts, (list, tuple)) or len(experts) != count
        ):
            raise ValueError(
                "Los expertos deben ser una secuencia acotada de la dimensión declarada"
            )
        experts = (
            tuple(correction.empty() for _ in range(count)) if experts is None else tuple(experts)
        )
        if len(experts) != count or any(not _backend_matches(item, correction) for item in experts):
            raise ValueError("Los expertos no comparten backend, contador y cursor")
        if mixture.write_periods is None:
            if pending is not None or len({(item.writes, item.cursor) for item in experts}) != 1:
                raise ValueError("Los expertos no comparten backend, contador y cursor")
        if clock is not None:
            _clock(clock)
        if (
            type(cohorts) is not int
            or not 0 <= cohorts <= experts[0].writes
            or (clock is None) != (cohorts == 0)
            or (cohorts == 0) != (experts[0].writes == 0)
            or (clock is not None and clock < experts[0].cursor[0])
        ):
            raise ValueError("Reloj, cohortes y escrituras de E1 no son coherentes")
        if log_weights is None:
            log_weights = torch.zeros(count, dtype=torch.float64)
        _check_vector(log_weights, "Los logpesos", count, torch.float64)
        if (
            not torch.isfinite(log_weights).all()
            or log_weights.max() != 0
            or (
                (cohorts == 0 or mixture.weighting == "uniform")
                and torch.count_nonzero(log_weights)
            )
        ):
            raise ValueError("Los logpesos deben ser finitos, centrados y coherentes con la mezcla")
        if mixture.write_periods is not None:
            pending = self._check_pending(correction, mixture, experts, pending, clock, cohorts)
        self.correction, self.mixture, self._experts = correction, mixture, experts
        self._log_weights = log_weights.detach().clone()
        self.clock, self.cohorts = clock, cohorts
        self._pending = pending

    @staticmethod
    def _check_pending(correction, mixture, experts, pending, clock, cohorts):
        """Cada experto lento guarda su periodo abierto y le faltan justo esas filas."""
        count = len(experts)
        if pending is None:
            pending = (None,) * count
        if not isinstance(pending, (list, tuple)) or len(pending) != count:
            raise ValueError("R2 necesita un periodo abierto por experto")
        pending = tuple(pending)
        first = experts[0]
        common = (first.writes, first.cursor)
        for item, period, waiting in zip(experts, mixture.write_periods, pending, strict=True):
            if period == "session" or cohorts == 0:
                # Antes de la primera escritura nadie espera, y la sesión nunca espera.
                if waiting is not None or (item.writes, item.cursor) != common:
                    raise ValueError("Una sesión o una memoria sin escribir no esperan resultados")
                continue
            if type(waiting) is not PendingPeriod:
                raise ValueError("Cada experto lento escrito guarda su periodo abierto")
            rows, order = canonical_order(
                waiting.closed(),
                clock,
                key_size=correction.memory.key_size,
                value_size=1,
                weighted=True,
                cursor=item.cursor,
            )
            if (
                order != list(range(rows))
                or not 1 <= waiting.events <= min(cohorts, rows)
                or waiting.period != period_index(period, clock)
                or item.writes + rows != first.writes
                or waiting.key(rows - 1) != first.cursor
            ):
                raise ValueError("El periodo abierto de R2 no es coherente con sus escrituras")
        return pending

    @property
    def config(self):
        return self.correction.memory

    @property
    def writes(self):
        return self._experts[0].writes

    @property
    def cursor(self):
        return self._experts[0].cursor

    @property
    def weights(self):
        return self._log_weights.softmax(dim=0)

    @property
    def matrix(self):
        return torch.cat(
            [item.matrix.reshape(-1, 1) for item in self._experts]
            + [self._log_weights.reshape(-1, 1)]
        )

    @property
    def numeric_state_bytes(self):
        total = _expert_bytes(self.correction, self.mixture)
        if self.correction.areas is not None:
            total += len(self._experts) * self.correction.areas.areas * self.config.key_size * 8
        # Los expertos comparten el objeto inmutable de roles y sus tablas fijas.
        if self.correction.role_binding is not None:
            total += self.correction.role_binding.numeric_state_bytes
        if self.mixture.write_periods is not None:
            per_row = (3 + self.config.key_size + 2) * 8
            total += sum(item.rows * per_row for item in self._pending if item is not None)
        return total

    @property
    def applied_writes(self):
        """Filas que ya ha aplicado cada experto."""
        return [item.writes for item in self._experts]

    @property
    def pending_rows(self):
        """Filas que espera cada experto, cero en E1 y en los expertos de sesión."""
        if self.mixture.write_periods is None:
            return [0] * len(self._experts)
        return [0 if item is None else item.rows for item in self._pending]

    def _decayed(self, at):
        _clock(at)
        if self.clock is not None and at < self.clock:
            raise ValueError("E1 no permite consultar ni escribir antes de su estado confirmado")
        elapsed = 0 if self.clock is None else at - self.clock
        return tuple(
            item.decay_coefficients(math.exp2(-elapsed / (DAY_US * half_life)))
            for item, half_life in zip(self._experts, self.mixture.half_lives_days, strict=True)
        )

    def predict(self, keys, *, at):
        rows = keys.shape[0] if isinstance(keys, torch.Tensor) and keys.ndim == 2 else -1
        if not 1 <= rows <= MAX_COHORT:
            raise ValueError("La consulta multiescala necesita una cohorte no vacía y acotada")
        _check_keys(keys, rows, self.config.key_size)
        parts = torch.cat([item.read(keys) for item in self._decayed(at)], dim=1)
        result = (parts * self.weights).sum(dim=1, keepdim=True)
        if not torch.isfinite(result).all() or not torch.isfinite(parts).all():
            raise ValueError("La mezcla o un experto produjo valores no finitos")
        return result, parts

    def read(self, keys, *, at):
        return self.predict(keys, at=at)[0]

    def read_diagnostic(self, keys, *, at, **options):
        """Lesionar cada experto sin publicar componentes ni cambiar los pesos de la mezcla."""
        rows = keys.shape[0] if isinstance(keys, torch.Tensor) and keys.ndim == 2 else -1
        if not 1 <= rows <= MAX_COHORT or self.correction.areas is None:
            raise ValueError("El diagnóstico E1 necesita áreas y una cohorte no vacía y acotada")
        _check_keys(keys, rows, self.config.key_size)
        values = torch.cat(
            [item.read_diagnostic(keys, **options) for item in self._decayed(at)], dim=1
        )
        result = (values * self.weights).sum(dim=1, keepdim=True)
        if not torch.isfinite(values).all() or not torch.isfinite(result).all():
            raise ValueError("La lectura diagnóstica de E1 produjo valores no finitos")
        return result

    def write(self, feedback, *, cutoff, issued_predictions):
        rows, order = canonical_order(
            feedback,
            cutoff,
            key_size=self.config.key_size,
            value_size=1,
            weighted=self.config.rule == "proximal",
            cursor=self.cursor,
        )
        _check_matrix(
            issued_predictions, "Las predicciones de expertos emitidas", rows, len(self._experts)
        )
        if rows == 0:
            return self
        experts = self._decayed(cutoff)
        selection = torch.tensor(order, dtype=torch.int64)
        error = issued_predictions[selection] - feedback.values[selection]
        if not torch.isfinite(error).all():
            raise ValueError("Los errores de expertos no son finitos")
        loss = (error.abs().clamp(max=self.mixture.loss_scale) / self.mixture.loss_scale).mean(
            dim=0
        )
        scores = self._log_weights
        if self.mixture.weighting == "hedge":
            scores = scores - self.mixture.weight_rate * loss
            scores = scores - scores.max()
        if self.mixture.write_periods is None:
            updated = tuple(item.write(feedback, cutoff=cutoff) for item in experts)
            pending = None
        else:
            updated, pending = self._continuum_write(experts, feedback, order, cutoff)
        return MultiscaleMemory(
            self.correction,
            self.mixture,
            experts=updated,
            log_weights=scores,
            clock=cutoff,
            cohorts=self.cohorts + 1,
            pending=pending,
        )

    def _continuum_write(self, experts, feedback, order, cutoff):
        """R2: la sesión escribe ya y cada experto lento cierra su periodo al cambiar de periodo.

        El periodo cerrado se aplica después del decaimiento hasta este corte y antes de añadir
        el evento, que abre o amplía el periodo actual.
        """
        updated, pending = [], []
        for item, period, waiting in zip(
            experts, self.mixture.write_periods, self._pending, strict=True
        ):
            if period == "session":
                updated.append(item.write(feedback, cutoff=cutoff))
                pending.append(None)
                continue
            current = period_index(period, cutoff)
            if waiting is not None and waiting.period != current:
                item = item.write(waiting.closed(), cutoff=cutoff)
                waiting = None
            updated.append(item)
            pending.append(PendingPeriod.extend(waiting, feedback, order, current))
        return tuple(updated), tuple(pending)

    def identity(self):
        return dict(base=self.correction.identity(), mixture=self.mixture.identity())

    def export(self):
        payload = dict(
            schema_version=1,
            configuration=self.identity(),
            experts=[item.export() for item in self._experts],
            log_weights=self._log_weights.clone(),
            clock=self.clock,
            cohorts=self.cohorts,
            writes=self.writes,
            cursor=list(self.cursor) if self.cursor is not None else None,
            matrix_sha256=_matrix_digest(self.matrix),
        )
        # Sin periodos el estado conserva los campos y la huella de E1.
        if self.mixture.write_periods is not None:
            payload["pending"] = [None if item is None else item.export() for item in self._pending]
            digests = [None if item is None else item.digest() for item in self._pending]
            payload["pending_sha256"] = hashlib.sha256(_canonical(digests).encode()).hexdigest()
        payload["state_sha256"] = self._state_digest(payload)
        return payload

    @staticmethod
    def _state_digest(payload):
        fields = {
            name: payload[name]
            for name in (
                "schema_version",
                "configuration",
                "clock",
                "cohorts",
                "writes",
                "cursor",
                "matrix_sha256",
            )
        }
        if "pending_sha256" in payload:
            fields["pending_sha256"] = payload["pending_sha256"]
        return hashlib.sha256(_canonical(fields).encode()).hexdigest()

    @classmethod
    def restore(cls, correction, mixture, payload):
        fields = {
            "schema_version",
            "configuration",
            "experts",
            "log_weights",
            "clock",
            "cohorts",
            "writes",
            "cursor",
            "matrix_sha256",
            "state_sha256",
        }
        if mixture.write_periods is not None:
            fields |= {"pending", "pending_sha256"}
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or not isinstance(payload["experts"], list)
            or len(payload["experts"]) != len(mixture.half_lives_days)
            or _canonical(payload["configuration"])
            != _canonical(dict(base=correction.identity(), mixture=mixture.identity()))
        ):
            raise ValueError("El estado no conserva la configuración multiescala")
        pending = None
        if mixture.write_periods is not None:
            # El constructor comprueba que haya uno por experto.
            if not isinstance(payload["pending"], list):
                raise ValueError("R2 necesita un periodo abierto por experto")
            pending = [
                None if item is None else PendingPeriod.restore(item, correction.memory.key_size)
                for item in payload["pending"]
            ]
        memory = cls(
            correction,
            mixture,
            experts=[correction.restore(p) for p in payload["experts"]],
            log_weights=payload["log_weights"],
            clock=payload["clock"],
            cohorts=payload["cohorts"],
            pending=pending,
        )
        expected = memory.export()
        checked = ("writes", "cursor", "matrix_sha256", "state_sha256")
        if mixture.write_periods is not None:
            checked += ("pending_sha256",)
        if type(payload["writes"]) is not int or any(
            _canonical(payload[field]) != _canonical(expected[field]) for field in checked
        ):
            raise ValueError("El estado multiescala no coincide con sus huellas o metadatos")
        return memory
