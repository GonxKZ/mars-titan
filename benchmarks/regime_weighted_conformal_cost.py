"""Medir en CPU el conformal ponderado por contextos parecidos y sus controles (A3, #748).

La etapa no tiene tensores ni parámetros entrenables, así que no hay forward ni backward. Los
paneles son sintéticos con semilla fija (`output_stage_cost.synthetic`) y solo fijan formas:
30 sesiones previas para definir el régimen, 63 de calibración (G) y un tramo evaluado de
tres o seis meses con 256 activos de EE. UU. Las sesiones sintéticas son días consecutivos,
así que la ventana de 120 días reúne más filas pasadas que con el calendario real (unas 83
sesiones). Se añade una sola sesión con 4.200 activos para mostrar cómo crece el coste, que es
proporcional a filas evaluadas por filas pasadas. Nada se ajusta, selecciona ni evalúa con
estos valores.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from benchmarks.output_stage_cost import peak_bytes, synthetic, timed
from mars_titan.calibration import regime_weighted_conformal as rwc
from mars_titan.hardware.platform_identity import cpu_name

WARMUP_SESSIONS = 30
CALIBRATION_SESSIONS = 63
# Activos, sesiones evaluadas y repeticiones de cada caso.
CASES = {
    "quarter_256": (256, 63, 3),
    "half_year_256": (256, 126, 3),
    "one_session_4200": (4200, 1, 1),
}


def parts(assets, sessions):
    """Calibración, evaluación e historial de etiquetas con su madurez."""
    warmup, _, warmup_matures = synthetic(assets, WARMUP_SESSIONS, 0, seed=1)
    calibration, _, calibration_matures = synthetic(
        assets, CALIBRATION_SESSIONS, WARMUP_SESSIONS, seed=2
    )
    evaluation, _, evaluation_matures = synthetic(
        assets, sessions, WARMUP_SESSIONS + CALIBRATION_SESSIONS, seed=3
    )
    panels = (warmup, calibration, evaluation)
    history = dict(
        market=np.full(sum(panel.rows for panel in panels), "US"),
        target=np.concatenate([panel.target for panel in panels]),
        matures_at=np.concatenate([warmup_matures, calibration_matures, evaluation_matures]),
    )
    return (calibration, calibration_matures), (evaluation, evaluation_matures), history


def measure_case(assets, sessions, repeats):
    calibration, evaluation, history = parts(assets, sessions)
    record, fit_cost = timed(lambda: rwc.fit(*calibration, history), repeats)
    result = dict(
        assets=assets,
        evaluated_sessions=sessions,
        evaluated_rows=evaluation[0].rows,
        config=record["config"],
        fit=fit_cost,
    )
    for weighting in rwc.WEIGHTINGS:

        def replay(weighting=weighting):
            return rwc.replay(record, calibration, evaluation, history, weighting=weighting)

        (_, summary), cost = timed(replay, repeats)
        result[weighting] = dict(
            replay=cost,
            numpy_peak_bytes=peak_bytes(replay),
            past_rows=summary["past_rows"],
            effective_rows=summary["effective_rows"],
        )
    return result


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
            "src/mars_titan/calibration/regime_weighted_conformal.py",
            "src/mars_titan/calibration/conformal_quantiles.py",
            "benchmarks/regime_weighted_conformal_cost.py",
        )
    }
    receipt = dict(
        schema_version=1,
        kind="regime_weighted_conformal_cost",
        issue=748,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=commit,
        sources_sha256=sources,
        data="sintéticos con semilla fija, solo para fijar formas. Nada se ajusta ni evalúa",
        environment=dict(
            cpu=cpu_name(),
            cpu_count=os.cpu_count(),
            threads=dict(
                omp=os.environ.get("OMP_NUM_THREADS"),
                openblas=os.environ.get("OPENBLAS_NUM_THREADS"),
            ),
            python=platform.python_version(),
            numpy=np.__version__,
            load_average_at_start=load,
            load_average_at_end=os.getloadavg(),
            concurrent_load="otras sesiones comparten la máquina, ver las cargas medias",
        ),
        not_measured=[
            "forward y backward, porque la etapa no tiene tensores ni parámetros entrenables",
            "GPU, por el mismo motivo",
            "lectura de Parquet y alineación con la vista, que hará la criba",
            "energía, sin instrumento adecuado",
        ],
        cases=cases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + "\n")
    print(
        json.dumps(
            {
                name: {w: case[w]["replay"]["median_s"] for w in rwc.WEIGHTINGS}
                for name, case in cases.items()
            }
        )
    )


if __name__ == "__main__":
    main()
