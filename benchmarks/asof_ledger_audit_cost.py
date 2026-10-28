"""Medir en CPU la auditoría as-of del registro de etiquetas (G3, #748).

La auditoría repite la emisión de la etapa dos veces y otras dos por cada corte, así que su
coste es el de la etapa multiplicado por 2 + 2C. Se mide sobre PT2 en línea, la etapa más
cara de las que ya existen, con datos sintéticos de semilla fija que solo fijan las formas:
256 activos de EE. UU., una calibración de 63 sesiones y un tramo auditado de 21 o 63
sesiones con un corte por sesión. Nada se ajusta, selecciona ni evalúa con estos valores.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from benchmarks.output_stage_cost import synthetic, timed
from mars_titan.calibration import online_conformal
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.integrity import asof_ledger_audit as audit
from mars_titan.models.quantile_head import LEVELS

ASSETS = 256
CALIBRATION_SESSIONS = 63
NOMINALS = (0.8, 0.95)
# Sesiones auditadas y repeticiones de cada caso.
CASES = {"month_256": (21, 3), "quarter_256": (63, 1)}


def ledger(panel, assets, matures):
    return dict(
        asset_id=assets,
        group=np.asarray(panel.markets)[panel.market],
        issued_at=panel.prediction_at,
        available_at=matures,
        target=panel.target.copy(),
        quantiles=panel.quantiles,
    )


def stage(calibration):
    def emit(records):
        calibrator = online_conformal.OnlineConformal.start(
            calibration.target,
            calibration.quantiles,
            np.asarray(calibration.markets)[calibration.market],
            levels=LEVELS,
            nominals=NOMINALS,
            min_rows=1,
            rate_fraction=0.25,
        )
        emitted, _, _ = online_conformal.replay_online_conformal(
            calibrator,
            records["quantiles"],
            records["group"],
            records["issued_at"],
            records["target"],
            records["available_at"],
        )
        return emitted, records["issued_at"]

    return emit


def measure_case(sessions, repeats):
    calibration, _, _ = synthetic(ASSETS, CALIBRATION_SESSIONS, 0, seed=1)
    panel, assets, matures = synthetic(ASSETS, sessions, CALIBRATION_SESSIONS + 2, seed=2)
    records = ledger(panel, assets, matures)
    emit = stage(calibration)
    cutoffs = np.unique(records["issued_at"])
    _, emission = timed(lambda: emit(records), repeats)
    receipt, cost = timed(
        lambda: audit.audit(emit, records, columns=("target",), cutoffs=cutoffs), repeats
    )
    started = time.perf_counter()
    accepted = audit.rejection(emit, records)
    rejection_s = time.perf_counter() - started
    return dict(
        assets=ASSETS,
        audited_rows=panel.rows,
        cutoffs=len(cutoffs),
        emissions_per_audit=2 + 2 * len(cutoffs),
        stage_emission=emission,
        audit=cost,
        rejection_s=rejection_s,
        passed=receipt["passed"],
        rejection_accepted=accepted,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    load = os.getloadavg()
    cases = {name: measure_case(*shape) for name, shape in CASES.items()}
    sources = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (
            "src/mars_titan/integrity/asof_ledger_audit.py",
            "src/mars_titan/calibration/online_conformal.py",
            "benchmarks/asof_ledger_audit_cost.py",
        )
    }
    receipt = dict(
        schema_version=1,
        kind="asof_ledger_audit_cost",
        issue=748,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=commit,
        sources_sha256=sources,
        data="sintéticos con semilla fija, solo para fijar formas. Nada se ajusta ni evalúa",
        environment=dict(
            cpu=cpu_name(),
            cpu_count=os.cpu_count(),
            python=platform.python_version(),
            numpy=np.__version__,
            load_average_at_start=load,
            load_average_at_end=os.getloadavg(),
            concurrent_load="otras sesiones comparten la máquina, ver las cargas medias",
        ),
        not_measured=[
            "GPU y modelos, porque la auditoría solo repite etapas de salida",
            "lectura de Parquet, que hará la criba",
            "energía, sin instrumento adecuado",
        ],
        cases=cases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({name: case["audit"]["median_s"] for name, case in cases.items()}))


if __name__ == "__main__":
    main()
