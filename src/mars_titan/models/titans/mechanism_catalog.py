"""Catálogo cerrado de propuestas, con trabajo lógico y momentos de decisión explícitos."""

from dataclasses import asdict, dataclass, fields
from itertools import product

from .config import FingerprintOnce, bounded_integer, canonical
from .refinement_routes import ROUTES

CONTEXT_ACTIONS = ("off", "parent")
CALIBRATION_ACTIONS = ("none", "static", "online")
EVENT_ORDER = "predict_all_then_feedback_v1"


@dataclass(frozen=True)
class MechanismAction:
    context: str = "off"
    route: str = "k1"
    calibration: str = "none"

    def __post_init__(self):
        for name, allowed in (
            ("context", CONTEXT_ACTIONS),
            ("route", ROUTES),
            ("calibration", CALIBRATION_ACTIONS),
        ):
            if type(getattr(self, name)) is not str or getattr(self, name) not in allowed:
                raise ValueError(f"La acción de {name} no pertenece al catálogo")

    @property
    def key(self):
        return f"{self.context}/{self.route}/{self.calibration}"


@dataclass(frozen=True)
class MechanismBudget:
    max_reader_steps: int = 4
    max_reader_searches: int = 2
    max_context_markets: int = 2
    max_online_rows: int = 8192

    def __post_init__(self):
        if type(self.max_reader_steps) is not int or self.max_reader_steps not in (1, 2, 4):
            raise ValueError("El presupuesto de refinamiento debe ser K1, K2 o K4")
        bounded_integer(self.max_reader_searches, "búsquedas por fila", 0, 2)
        bounded_integer(self.max_context_markets, "mercados con contexto", 0, 2)
        bounded_integer(self.max_online_rows, "filas de la cohorte PT2", 1, 8192)


@dataclass(frozen=True)
class MechanismAvailability:
    context: bool = False
    reader: bool = False
    quantiles: bool = False
    static: bool = False
    online: bool = False
    online_capacity: bool = True
    online_maturity: bool = True
    context_capacity: bool = True
    nonempty_bank: bool = False
    reader_limit: int = 4

    def __post_init__(self):
        for field in fields(self):
            if field.name != "reader_limit" and type(getattr(self, field.name)) is not bool:
                raise ValueError("La disponibilidad de mecanismos necesita booleanos explícitos")
        if type(self.reader_limit) is not int or self.reader_limit not in (0, 1, 2, 4):
            raise ValueError("El límite admisible del lector no es válido")


@dataclass(frozen=True)
class ResolvedMechanisms:
    requested: MechanismAction
    applied: MechanismAction
    reasons: tuple[str, ...]
    reader_steps: int
    reader_searches: int

    @property
    def status(self):
        return "fallback" if self.reasons else "applied"


def reader_work(route, availability):
    if not availability.reader:
        return 0, 0
    steps = int(route[1])
    searches = (1 + int("r" in route)) if availability.nonempty_bank else 0
    return steps, searches


@dataclass(frozen=True)
class MechanismCatalog(FingerprintOnce):
    budget: MechanismBudget = MechanismBudget()
    reader_contract: str = "legacy"

    def __post_init__(self):
        if type(self.budget) is not MechanismBudget:
            raise ValueError("El catálogo necesita su presupuesto identificado")
        if type(self.reader_contract) is not str or self.reader_contract not in (
            "legacy",
            "shared_evidence_bridge_v1",
            "shared_evidence_bridge_nonlinear_v1",
            "shared_evidence_bridge_intervention_v1",
            "shared_evidence_bridge_hierarchy_v1",
        ):
            raise ValueError("El catálogo necesita un contrato de lector conocido")

    def actions(self):
        return tuple(
            MechanismAction(*values)
            for values in product(CONTEXT_ACTIONS, ROUTES, CALIBRATION_ACTIONS)
        )

    def identity(self):
        result = dict(
            schema_version=1,
            kind="financial_mechanism_catalog_v1",
            actions=[asdict(action) for action in self.actions()],
            budget=asdict(self.budget),
            stages=dict(context="before_mac", route="after_k1", calibration="cohort_emission"),
            event_order=EVENT_ORDER,
            arbitrary_code_allowed=False,
            baseline=asdict(MechanismAction()),
            fallback="raw_k1_on_the_selected_pre_mac_context_without_recomputation",
            work_units="reader_steps_and_searches_per_row_context_writes_per_market",
            utility_conversion="none_no_implicit_conversion_of_time_to_prediction_error",
        )
        if self.reader_contract != "legacy":
            result.update(
                schema_version=2,
                reader_contract=self.reader_contract,
                reader_route_semantics="c_reuses_positions_r_reselects_values_reprojected_each_step",
                regional_trace="common_ids_separate_regional_weights_and_derived_views",
            )
            if self.reader_contract == "shared_evidence_bridge_hierarchy_v1":
                result["regional_trace"] = (
                    "common_ids_four_views_two_groups_one_root_effective_graph"
                )
        return result

    @classmethod
    def restore(cls, identity):
        """Restaurar un catálogo completo sin reinterpretar contratos ni campos omitidos."""
        if type(identity) is not dict:
            raise ValueError("Falta la identidad completa del catálogo")
        try:
            catalog = cls(
                MechanismBudget(**identity["budget"]),
                reader_contract=identity.get("reader_contract", "legacy"),
            )
            matches = canonical(identity) == canonical(catalog.identity())
        except (TypeError, KeyError) as error:
            raise ValueError("La identidad del catálogo está incompleta o no es válida") from error
        if not matches:
            raise ValueError("El catálogo no conserva sus campos y semántica registrados")
        return catalog

    def resolve(self, action, availability, *, rows):
        """Resolver disponibilidad antes de ejecutar. Un contrato inválido no es un retorno."""
        if type(action) is not MechanismAction or type(availability) is not MechanismAvailability:
            raise ValueError("La resolución necesita acción y disponibilidad tipadas")
        bounded_integer(rows, "filas de la cohorte", 1, 8192)
        budget = self.budget
        baseline_steps, baseline_searches = reader_work("k1", availability)
        if availability.reader and (
            availability.reader_limit < 1 or baseline_searches > budget.max_reader_searches
        ):
            raise ValueError("El presupuesto no permite ejecutar el K1 de referencia")
        reasons = []
        if action.context == "parent":
            if not availability.context:
                reasons.append("unavailable_context")
            if not availability.context_capacity or budget.max_context_markets == 0:
                reasons.append("context_budget")
        selected_context = "off" if reasons else action.context
        if action.route != "k1" and not availability.reader:
            reasons.append("unavailable_readout")
        steps, searches = reader_work(action.route, availability)
        if steps > min(budget.max_reader_steps, availability.reader_limit):
            reasons.append("reader_budget")
        if searches > budget.max_reader_searches:
            reasons.append("search_budget")
        if action.calibration != "none":
            if not availability.quantiles:
                reasons.append("unavailable_quantiles")
            if not availability.static or (
                action.calibration == "online" and not availability.online
            ):
                reasons.append("unavailable_calibration")
            if action.calibration == "online" and (
                not availability.online_capacity or rows > budget.max_online_rows
            ):
                reasons.append("pending_calibration_budget")
            if action.calibration == "online" and not availability.online_maturity:
                reasons.append("unavailable_label_calendar")
        return ResolvedMechanisms(
            action,
            MechanismAction(selected_context, "k1", "none") if reasons else action,
            tuple(reasons),
            baseline_steps if reasons else steps,
            baseline_searches if reasons else searches,
        )
