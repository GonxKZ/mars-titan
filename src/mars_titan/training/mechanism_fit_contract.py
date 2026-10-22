"""Protocolo identificado del ajuste de ganancias, previo a cualquier lectura de fuentes."""

import math
from dataclasses import asdict, dataclass

from mars_titan.models.titans.config import FingerprintOnce, bounded_integer
from mars_titan.models.titans.episodic_snapshot import _digest_id
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US, HISTORICAL_START_US
from mars_titan.models.titans.mechanism_catalog import MechanismCatalog


def finite(value, name, minimum=0.0, maximum=1e12):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} debe ser un número finito entre {minimum} y {maximum}")


@dataclass(frozen=True)
class MechanismFitConfig(FingerprintOnce):
    """Un tramo de ajuste y costes declarados, sin selección retrospectiva de ventanas."""

    parent_consumer_id: str
    parent_selected_before: int
    fit_start: int
    fit_end: int
    ridge: float
    interval_weight: float
    min_gain: float
    market_weights: tuple
    costs: tuple
    min_cohorts: int = 32
    max_cohorts: int = 4096
    max_sources: int = 8
    max_source_bytes: int = 512 * 1024**2
    max_condition: float = 1e12
    source_mode: str = "closed_phases"
    catalog_id: str | None = None

    def __post_init__(self):
        if type(self.source_mode) is not str or self.source_mode not in (
            "closed_phases",
            "native_windows",
        ):
            raise ValueError("Modalidad de fuentes de ajuste desconocida")
        _digest_id(self.parent_consumer_id, "El consumidor padre")
        if self.catalog_id is not None:
            _digest_id(self.catalog_id, "El catálogo declarado del ajuste")
        if any(
            type(v) is not int for v in (self.parent_selected_before, self.fit_start, self.fit_end)
        ) or not (
            0 <= self.parent_selected_before <= self.fit_start < self.fit_end <= FINAL_TEST_US
            and self.fit_start >= HISTORICAL_START_US
        ):
            raise ValueError("El padre y el tramo de ajuste deben preceder al test final")
        finite(self.ridge, "Ridge", 1e-12, 1e6)
        finite(self.interval_weight, "El peso de intervalos")
        finite(self.min_gain, "La ganancia mínima")
        finite(self.max_condition, "El límite de condicionamiento", 1.0)
        bounded_integer(self.min_cohorts, "cohortes mínimas", 2, 8192)
        bounded_integer(self.max_cohorts, "cohortes máximas", self.min_cohorts, 8192)
        bounded_integer(self.max_sources, "sesiones fuente", 1, 16)
        bounded_integer(self.max_source_bytes, "bytes fuente", 1024**2, 1024**3)
        if type(self.market_weights) is not tuple or not 1 <= len(self.market_weights) <= 2:
            raise ValueError("Los pesos de mercados necesitan una tupla explícita")
        markets = []
        for item in self.market_weights:
            if type(item) is not tuple or len(item) != 2 or item[0] not in ("CN", "US"):
                raise ValueError("Cada peso necesita un mercado admitido")
            finite(item[1], "El peso de mercado", 1e-12, 1.0)
            markets.append(item[0])
        if markets != sorted(set(markets)):
            raise ValueError("Los mercados de ajuste se repiten o desordenan")
        if type(self.costs) is not tuple or len(self.costs) != 36:
            raise ValueError("El ajuste necesita los 36 costes declarados")
        for cost in self.costs:
            finite(cost, "El coste incremental")
        if self.costs[0] != 0:
            raise ValueError("El coste incremental de la referencia debe ser cero")

    def check_catalog(self, catalog):
        """Exigir un vínculo explícito para lectores con semántica regional."""
        if type(catalog) is not MechanismCatalog:
            raise ValueError("El ajuste necesita un catálogo identificado")
        if self.catalog_id is None:
            if catalog.reader_contract != "legacy":
                raise ValueError("El ajuste regional debe declarar la huella del catálogo")
        elif catalog.fingerprint() != self.catalog_id:
            raise ValueError("La fuente pertenece a otro catálogo de mecanismos")

    def identity(self):
        fields = asdict(self)
        fields.pop("source_mode")
        if self.catalog_id is None:
            fields.pop("catalog_id")
        record = dict(
            schema_version=1,
            kind="closed_native_cohort_mechanism_fit_v1",
            **fields,
            point_objective="mae_baseline_minus_mae_candidate",
            interval_objective="mean_interval_score_80_95_baseline_minus_candidate",
            normalization="fit_cohorts_only_weighted_population",
            missing_targets="masked_not_observed_zero",
            intercept="ridge_penalized_towards_zero_gain",
            solver="augmented_weighted_least_squares_gels_fp64",
            weighting="declared_market_weight_divided_by_observed_cohorts_in_market",
            support="marginal_ranges_and_declared_objective_count_mask",
            parent_time="declared_protocol_cut_not_independent_timestamp_attestation",
        )
        if self.source_mode == "native_windows":
            record.update(
                schema_version=2,
                kind="windowed_native_cohort_mechanism_fit_v1",
                source_mode=self.source_mode,
                source_time="decision_in_fit_range_and_native_reception_before_fit_end",
                source_budget="requested_windows_with_accumulated_packet_and_native_record_bytes",
            )
        return record
