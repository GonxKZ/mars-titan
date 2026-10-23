"""Control finito de propuestas a partir del error emitido del padre frente a cero."""

import json
import math
from dataclasses import dataclass, field, fields
from fractions import Fraction

from mars_titan.models.titans.config import bounded_integer
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US, FLOW_ID
from mars_titan.proposals.contracts import ProposalRegistry
from mars_titan.proposals.mature_review import (
    MAX_REVIEW_BYTES,
    MatureReview,
    encoded,
    fingerprint,
    identifier,
    require,
)

HISTORY_START_US = 946_684_800_000_000
MAX_WINDOW_US = 366 * 86_400 * 1_000_000
CONTROLS = ("none", "fixed", "enumerate", "mature_parent_vs_zero")


def _finite(value, name, *, positive=False):
    require(type(value) in (int, float), f"{name} necesita un número finito")
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    require(valid, f"{name} está fuera del dominio permitido")


@dataclass(frozen=True)
class ErrorProposalPlan:
    """Cortes y reglas declarados antes de revisar una fase cerrada del padre.

    La declaración es un contrato local, no una prueba independiente de su fecha.
    La identidad incluye el universo, los pesos y el orden finito de candidatas.
    """

    source_contract_id: str
    parent_consumer_id: str
    source_partition: str
    declared_before: int
    parent_selected_before: int
    generation_start: int
    generation_end: int
    selection_start: int
    selection_end: int
    evaluation_start: int
    evaluation_end: int
    flow_ids: tuple[str, ...]
    market_weights: tuple[tuple[str, float], ...]
    min_sessions_per_market: int
    min_excess_mae: float
    candidate_ids: tuple[str, ...]
    control: str
    max_source_records: int = 256
    max_source_bytes: int = 64 * 1024**2
    _sealed: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, "_sealed", encoded(self._current_identity(), MAX_REVIEW_BYTES))

    def _validate(self):
        identifier(self.source_contract_id, "El contrato de fuente")
        identifier(self.parent_consumer_id, "El consumidor padre")
        require(
            type(self.source_partition) is str and self.source_partition in ("train", "validation"),
            "Partición de generación prohibida",
        )
        require(type(self.control) is str and self.control in CONTROLS, "Control desconocido")
        self._validate_dates()
        self._validate_universe()
        self._validate_limits()

    def _validate_dates(self):
        names = (
            "declared_before",
            "parent_selected_before",
            "generation_start",
            "generation_end",
            "selection_start",
            "selection_end",
            "evaluation_start",
            "evaluation_end",
        )
        for name in names:
            bounded_integer(getattr(self, name), name, HISTORY_START_US, FINAL_TEST_US)
        require(
            self.declared_before < self.generation_start
            and self.parent_selected_before < self.generation_start
            and self.generation_start
            < self.generation_end
            <= self.selection_start
            < self.selection_end
            <= self.evaluation_start
            < self.evaluation_end
            and self.evaluation_end - self.generation_start <= MAX_WINDOW_US,
            "Los cortes deben ser sucesivos, anteriores al test final y de hasta un año",
        )

    def _validate_universe(self):
        require(
            type(self.flow_ids) is tuple
            and 1 <= len(self.flow_ids) <= 8192
            and all(type(flow) is str and FLOW_ID.fullmatch(flow) for flow in self.flow_ids)
            and tuple(sorted(set(self.flow_ids))) == self.flow_ids,
            "El universo necesita flujos únicos y acotados",
        )
        require(
            type(self.market_weights) is tuple and 1 <= len(self.market_weights) <= 2,
            "Faltan los pesos explícitos de mercado",
        )
        markets = []
        for item in self.market_weights:
            require(type(item) is tuple and len(item) == 2, "Peso de mercado inválido")
            market, weight = item
            require(type(market) is str and market in ("US", "CN"), "Mercado desconocido")
            _finite(weight, "El peso del mercado", positive=True)
            markets.append(market)
        require(
            len(set(markets)) == len(markets)
            and set(markets) == {flow.split("/", 1)[0] for flow in self.flow_ids},
            "Los pesos deben cubrir exactamente los mercados del universo",
        )

    def _validate_limits(self):
        bounded_integer(self.min_sessions_per_market, "mínimo de sesiones", 1, 8192)
        _finite(self.min_excess_mae, "El margen mínimo")
        bounded_integer(self.max_source_records, "registros de fuente", 1, 256)
        bounded_integer(self.max_source_bytes, "bytes de fuente", 1, 64 * 1024**2)
        require(
            type(self.candidate_ids) is tuple and 1 <= len(self.candidate_ids) <= 7,
            "Se necesitan entre una y siete candidatas",
        )
        for candidate in self.candidate_ids:
            identifier(candidate, "La candidata")
        require(len(set(self.candidate_ids)) == len(self.candidate_ids), "Candidatas repetidas")

    def _current_identity(self):
        return dict(
            schema_version=1,
            kind="finite_error_proposal_plan_v1",
            **{item.name: getattr(self, item.name) for item in fields(self) if item.init},
        )

    def identity(self):
        self._validate()
        require(
            encoded(self._current_identity(), MAX_REVIEW_BYTES) == self._sealed,
            "El plan no conserva su sello",
        )
        return json.loads(self._sealed)

    @property
    def plan_id(self):
        return fingerprint(self.identity())

    def check_registry(self, registry):
        self.identity()
        require(type(registry) is ProposalRegistry, "Falta el registro tipado")
        registry.identity()
        require(
            registry.baseline_id not in self.candidate_ids, "El padre no es una candidata nueva"
        )
        for candidate in self.candidate_ids:
            registry.resolve(candidate)


def _mean(values):
    """Promediar pérdidas finitas sin desbordar primero su suma."""
    require(bool(values), "No se puede promediar un conjunto vacío")
    try:
        result = math.fsum(value / len(values) for value in values)
    except OverflowError as error:
        raise ValueError("La media excede la precisión disponible") from error
    require(math.isfinite(result), "La media no es finita")
    return result


def _weighted(values, weights):
    # Solo hay uno o dos mercados. Esta reducción evita perder un peso antes
    # de multiplicarlo por una pérdida cuyo producto todavía es representable.
    exact_weights = {market: Fraction(weight) for market, weight in weights.items()}
    total = sum(exact_weights.values())
    result = float(
        sum(Fraction(values[market]) * exact_weights[market] for market in sorted(weights)) / total
    )
    require(math.isfinite(result), "La media ponderada no es finita")
    return result


def parent_zero_statistics(review, *, plan):
    """Promediar activos, sesiones y mercados sin descartar resultados pendientes."""
    require(
        type(plan) is ErrorProposalPlan and type(review) is MatureReview,
        "La estadística necesita un plan y una revisión tipados",
    )
    plan.identity()
    body = review.record()
    require(
        body["parent_consumer_id"] == plan.parent_consumer_id
        and body["contract_id"] == plan.source_contract_id
        and body["review_at"] == plan.generation_end,
        "La revisión no corresponde al padre, contrato o corte del plan",
    )
    return _point_statistics(
        body["rows"],
        flow_ids=plan.flow_ids,
        generation_start=plan.generation_start,
        generation_end=plan.generation_end,
        market_weights=plan.market_weights,
        min_sessions_per_market=plan.min_sessions_per_market,
        min_excess_mae=plan.min_excess_mae,
    )


def _point_statistics(
    values,
    *,
    flow_ids,
    generation_start,
    generation_end,
    market_weights,
    min_sessions_per_market,
    min_excess_mae,
):
    """Aritmética común sobre filas ya verificadas por su lector de procedencia."""
    states = dict.fromkeys(("label", "pending", "excluded", "censored"), 0)
    flows = set(flow_ids)
    groups = {}
    for row in sorted(values, key=lambda row: (row["flow_id"], row["prediction_at"])):
        require(
            row["flow_id"] in flows and generation_start <= row["prediction_at"] < generation_end,
            "Una fila está fuera del universo o tramo de generación",
        )
        states[row["status"]] += 1
        key = (row["flow_id"].split("/", 1)[0], row["prediction_at"])
        groups.setdefault(key, []).append(row)
    weights = dict(market_weights)
    sessions = {market: sum(key[0] == market for key in groups) for market in sorted(weights)}
    reason = (
        "incomplete_outcomes"
        if states["label"] != len(values)
        else "missing_market"
        if any(count == 0 for count in sessions.values())
        else "insufficient_sessions"
        if any(count < min_sessions_per_market for count in sessions.values())
        else "complete"
    )
    complete = reason == "complete"
    cohorts = []
    for (market, prediction_at), rows in sorted(groups.items()):
        parent = _mean([row["emitted_point_loss"] for row in rows]) if complete else None
        zero = _mean([abs(row["target"]) for row in rows]) if complete else None
        cohorts.append(
            dict(
                market=market,
                prediction_at=prediction_at,
                samples=len(rows),
                parent_mae=parent,
                zero_mae=zero,
                excess_mae=None if not complete else parent - zero,
            )
        )
    parent = zero = excess = None
    if complete:
        parent = _weighted(
            {
                market: _mean([item["parent_mae"] for item in cohorts if item["market"] == market])
                for market in weights
            },
            weights,
        )
        zero = _weighted(
            {
                market: _mean([item["zero_mae"] for item in cohorts if item["market"] == market])
                for market in weights
            },
            weights,
        )
        excess = parent - zero
        require(math.isfinite(excess), "La diferencia de MAE no es finita")
    return dict(
        status="complete" if complete else "incomplete",
        reason=reason,
        samples=len(values),
        state_counts=states,
        sessions_by_market=sessions,
        parent_mae=parent,
        zero_mae=zero,
        excess_mae=excess,
        triggered=complete and excess > min_excess_mae,
        cohorts=cohorts,
    )
