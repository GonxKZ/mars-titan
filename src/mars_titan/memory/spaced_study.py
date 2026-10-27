"""Declaración cerrada del diagnóstico E2 sobre unos parámetros ya seleccionados."""

import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest

from .spaced_retention import POLICIES, SpacedRetentionConfig

ARMS = ("baseline", *POLICIES)
KIND = "spaced_utility_frozen_study_v1"
REGIONAL_KIND = "spaced_regional_utility_frozen_study_v1"


@dataclass(frozen=True)
class SpacedAnalysis:
    """Contrastes y remuestreo fijados antes de ejecutar los seis controles E2."""

    block_length: int = 16
    replicates: int = 2000
    seed: int = 20261027
    confidence: float = 0.95
    market_weighting: str = "session"
    sensitivity_block_lengths: tuple[int, ...] = (8, 32)
    rank_ic_min_assets: int = 3
    primary_contrast: str = "utility_minus_baseline"
    min_relevant_mae: float = 0.00001

    def __post_init__(self):
        for name, lower, upper in (
            ("block_length", 1, 100_000),
            ("replicates", 10, 100_000),
            ("seed", 0, 2**63 - 1),
            ("rank_ic_min_assets", 3, 8192),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"El parámetro estadístico E2 {name} no es válido")
        lengths = self.sensitivity_block_lengths
        if (
            type(self.confidence) is not float
            or not 0.5 <= self.confidence < 1
            or self.market_weighting not in ("session", "market")
            or type(lengths) is not tuple
            or len(lengths) > 16
            or any(type(length) is not int or not 1 <= length <= 100_000 for length in lengths)
            or len(set(lengths)) != len(lengths)
            or self.block_length in lengths
            or self.primary_contrast != "utility_minus_baseline"
            or type(self.min_relevant_mae) is not float
            or not math.isfinite(self.min_relevant_mae)
            or self.min_relevant_mae < 0
        ):
            raise ValueError("El análisis E2 no conserva ponderación, bloques, margen y contraste")

    def identity(self):
        return dict(asdict(self), sensitivity_block_lengths=list(self.sensitivity_block_lengths))

    def bootstrap_options(self):
        excluded = {"rank_ic_min_assets", "primary_contrast", "min_relevant_mae"}
        return {name: value for name, value in self.identity().items() if name not in excluded}

    @classmethod
    def from_identity(cls, value):
        if (
            type(value) is not dict
            or set(value) != {field.name for field in fields(cls)}
            or type(value["sensitivity_block_lengths"]) is not list
        ):
            raise ValueError("El análisis E2 necesita sus nueve opciones explícitas")
        return cls(
            **dict(value, sensitivity_block_lengths=tuple(value["sensitivity_block_lengths"]))
        )


@dataclass(frozen=True)
class SpacedStudy:
    retention: SpacedRetentionConfig
    arms: tuple[str, ...]
    max_seconds: float
    max_phase_rows: int
    reader_contract: str = "standard"
    analysis: SpacedAnalysis | None = None

    def identity(self):
        value = dict(
            kind=KIND,
            retention=asdict(self.retention),
            arms=list(self.arms),
            compute_budget=dict(max_seconds=self.max_seconds, max_phase_rows=self.max_phase_rows),
            state_reset="each_phase_empty_fast_memory_bank_and_evidence",
            interrupted_phase="repeat_from_start",
            final_test_access=False,
        )
        if self.reader_contract == "regional":
            value.update(kind=REGIONAL_KIND, reader_contract="regional")
        if self.analysis is not None:
            value.update(analysis=self.analysis.identity(), reader_contract=self.reader_contract)
        return value

    def retention_for(self, arm):
        if type(arm) is not str or arm not in self.arms:
            raise ValueError("La configuración E2 no declara esa variante")
        return None if arm == "baseline" else replace(self.retention, policy=arm)


def load_study(path):
    document, digest = read_manifest(Path(path), maximum=64 * 1024)
    regional = type(document) is dict and document.get("kind") == REGIONAL_KIND
    comparison = type(document) is dict and document.get("schema_version") == 3
    if (
        type(document) is not dict
        or set(document)
        != {
            "schema_version",
            "kind",
            "variant_id",
            "status",
            "default_enabled",
            "final_test_access",
            "retention",
            "arms",
            "compute_budget",
        }
        | ({"reader_contract"} if regional or comparison else set())
        | ({"analysis"} if comparison else set())
        or type(document["schema_version"]) is not int
        or document["schema_version"] != (3 if comparison else 2 if regional else 1)
        or document["kind"] != (REGIONAL_KIND if regional else KIND)
        or document["variant_id"]
        != (
            "spaced_regional_utility_consolidation_v1"
            if regional
            else "spaced_utility_consolidation_v1"
        )
        or (
            (regional or comparison)
            and document["reader_contract"] != ("regional" if regional else "standard")
        )
        or document["status"] != "declared_not_executed"
        or document["default_enabled"] is not False
        or document["final_test_access"] is not False
    ):
        raise ValueError("El estudio E2 debe conservar su esquema y seguir desactivado")
    arms, retention, budget = document["arms"], document["retention"], document["compute_budget"]
    if (
        type(arms) is not list
        or not arms
        or len(arms) > len(ARMS)
        or any(type(arm) is not str or arm not in ARMS for arm in arms)
        or len(set(arms)) != len(arms)
        or "baseline" not in arms
        or (comparison and set(arms) != set(ARMS))
        or type(retention) is not dict
        or set(retention) != {"capacity", "frontier", "seed", "max_working_bytes"}
        or type(budget) is not dict
        or set(budget) != {"max_seconds", "max_phase_rows"}
    ):
        raise ValueError("El estudio necesita controles, retención y presupuesto explícitos")
    seconds, rows = budget["max_seconds"], budget["max_phase_rows"]
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or not 0 < seconds <= 14400
        or type(rows) is not int
        or not 1 <= rows <= 5_000_000
    ):
        raise ValueError("El tiempo o las filas por fase exceden el presupuesto")
    return SpacedStudy(
        SpacedRetentionConfig(**retention),
        tuple(arms),
        float(seconds),
        rows,
        "regional" if regional else "standard",
        SpacedAnalysis.from_identity(document["analysis"]) if comparison else None,
    ), digest
