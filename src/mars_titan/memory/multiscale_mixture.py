"""Mezcla de correctores con decaimiento de calendario y errores emitidos maduros (E1).

Los expertos son copias del mismo backend. Sus coeficientes decaen con semividas distintas.
Hedge actualiza la mezcla con pérdidas acotadas de las predicciones guardadas al decidir.
El retraso financiero y el recorte impiden atribuir directamente la garantía del artículo.
"""

import hashlib
import math
from dataclasses import asdict, dataclass

import torch

from .associative_memory import (
    MAX_COHORT,
    AssociativeMemoryConfig,
    MatureCorrection,
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
    )


@dataclass(frozen=True)
class MultiscaleMixtureConfig:
    """Semividas de la parte homogénea y regla de mezcla fijadas antes de evaluar."""

    half_lives_days: tuple[float, ...] = (7.0, 91.0, 365.0)
    weighting: str = "hedge"
    weight_rate: float = 1.0
    loss_scale: float = 0.02

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

    def identity(self):
        return dict(
            asdict(self),
            kind=KIND,
            clock="utc_microseconds_at_mature_publication",
            decay="all_backend_coefficients_times_exp2_minus_elapsed_over_half_life",
            loss="canonical_cohort_mean_of_min_abs_issued_expert_minus_mature_core_residual_over_scale_1",
            read="immutable_calendar_decay_then_convex_mixture",
            initial_weights="uniform",
            bound="no_regret_guarantee_claimed_for_delayed_financial_predictions",
        )


def _clock(value):
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError("El reloj necesita microsegundos UTC int64")


def state_tensor_bytes(correction, mixture):
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
        self, correction, mixture, *, experts=None, log_weights=None, clock=None, cohorts=0
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
        if (
            len(experts) != count
            or any(not _backend_matches(item, correction) for item in experts)
            or len({(item.writes, item.cursor) for item in experts}) != 1
        ):
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
        self.correction, self.mixture, self._experts = correction, mixture, experts
        self._log_weights = log_weights.detach().clone()
        self.clock, self.cohorts = clock, cohorts

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
        total = state_tensor_bytes(self.correction, self.mixture)
        if self.correction.areas is not None:
            total += len(self._experts) * self.correction.areas.areas * self.config.key_size * 8
        # Los expertos comparten el objeto inmutable de roles y sus tablas fijas.
        if self.correction.role_binding is not None:
            total += self.correction.role_binding.numeric_state_bytes
        return total

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
        updated = tuple(item.write(feedback, cutoff=cutoff) for item in experts)
        return MultiscaleMemory(
            self.correction,
            self.mixture,
            experts=updated,
            log_weights=scores,
            clock=cutoff,
            cohorts=self.cohorts + 1,
        )

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
        memory = cls(
            correction,
            mixture,
            experts=[correction.restore(p) for p in payload["experts"]],
            log_weights=payload["log_weights"],
            clock=payload["clock"],
            cohorts=payload["cohorts"],
        )
        expected = memory.export()
        if type(payload["writes"]) is not int or any(
            _canonical(payload[field]) != _canonical(expected[field])
            for field in ("writes", "cursor", "matrix_sha256", "state_sha256")
        ):
            raise ValueError("El estado multiescala no coincide con sus huellas o metadatos")
        return memory
