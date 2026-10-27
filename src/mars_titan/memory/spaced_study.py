"""Declaración cerrada del diagnóstico E2 sobre unos parámetros ya seleccionados."""

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest

from .spaced_retention import POLICIES, SpacedRetentionConfig

ARMS = ("baseline", *POLICIES)
KIND = "spaced_utility_frozen_study_v1"


@dataclass(frozen=True)
class SpacedStudy:
    retention: SpacedRetentionConfig
    arms: tuple[str, ...]
    max_seconds: float
    max_phase_rows: int

    def identity(self):
        return dict(
            kind=KIND,
            retention=asdict(self.retention),
            arms=list(self.arms),
            compute_budget=dict(max_seconds=self.max_seconds, max_phase_rows=self.max_phase_rows),
            state_reset="each_phase_empty_fast_memory_bank_and_evidence",
            interrupted_phase="repeat_from_start",
            final_test_access=False,
        )

    def retention_for(self, arm):
        if type(arm) is not str or arm not in self.arms:
            raise ValueError("La configuración E2 no declara esa variante")
        return None if arm == "baseline" else replace(self.retention, policy=arm)


def load_study(path):
    document, digest = read_manifest(Path(path), maximum=64 * 1024)
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
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["kind"] != KIND
        or document["variant_id"] != "spaced_utility_consolidation_v1"
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
        SpacedRetentionConfig(**retention), tuple(arms), float(seconds), rows
    ), digest
