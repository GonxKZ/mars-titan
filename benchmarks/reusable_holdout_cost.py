"""Medir en CPU la escalera que protege S en la segunda ola (#794).

Las series son normales con semilla fija y solo fijan las formas. S tiene unas 1.000 sesiones,
así que cada caso usa 1.008 días UTC, con una decisión por día o con dos, y la declaración de
`configs/experiments/reusable-holdout-v1.json`: bloques de 16 días y 2.000 réplicas. La
escalera recibe 10 o 40 envíos, que cubren de sobra los modelos de una ola. Cada caso calienta
una vez y repite la escalera completa. Nada se ajusta ni se selecciona.
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

from benchmarks.output_stage_cost import peak_bytes, timed
from mars_titan.data.storage import atomic_json
from mars_titan.evaluation import reusable_holdout as rh
from mars_titan.hardware.platform_identity import cpu_name

DECLARATION = Path("configs/experiments/reusable-holdout-v1.json")
DAYS = 1008
# Envíos, decisiones por día y repeticiones de cada caso.
CASES = {
    "submissions_10": (10, 1, 5),
    "submissions_40": (40, 1, 3),
    "submissions_40_two_per_day": (40, 2, 3),
}


def submissions(count, per_day):
    """Envíos con IC por sesión de ruido y un 5 % de sesiones sin definir."""
    rng = np.random.default_rng(count * 10 + per_day)
    sessions = DAYS * per_day
    return [
        rh.Submission(
            f"model_{index:02d}", rng.normal(0.01, 0.1, sessions), rng.random(sessions) > 0.05
        )
        for index in range(count)
    ]


def measure(declaration, count, per_day, repeats):
    hour = 3_600_000_000
    day = np.repeat(np.arange(DAYS), per_day)
    times = day * rh.DAY_MICROSECONDS + np.tile(np.arange(per_day) * 6 * hour, DAYS)
    entries = submissions(count, per_day)
    rh.run_ladder(declaration, times, entries)
    result, cost = timed(lambda: rh.run_ladder(declaration, times, entries), repeats)
    return dict(
        days=DAYS,
        decisions_per_day=per_day,
        submissions=count,
        improvements=sum(item["improved"] for item in result["public"]["board"]),
        ladder=cost,
        per_submission_median_s=cost["median_s"] / count,
        traced_peak_bytes=peak_bytes(lambda: rh.run_ladder(declaration, times, entries)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("La salida debe ser nueva")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    declaration, _ = rh.load_declaration(DECLARATION)
    load = os.getloadavg()
    cases = {name: measure(declaration, *shape) for name, shape in CASES.items()}
    sources = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (
            "src/mars_titan/evaluation/reusable_holdout.py",
            "src/mars_titan/evaluation/paired_comparisons.py",
            "benchmarks/reusable_holdout_cost.py",
            str(DECLARATION),
        )
    }
    receipt = dict(
        schema_version=1,
        kind="reusable_holdout_cost",
        issue=794,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=commit,
        sources_sha256=sources,
        data="IC por sesión de ruido con semilla fija, solo para fijar formas. Nada se ajusta",
        declaration=dict(
            block_days=declaration["block_days"],
            replicates=declaration["replicates"],
            seed=declaration["seed"],
            resolution=declaration["resolution"],
        ),
        environment=dict(
            cpu=cpu_name(),
            cpu_count=os.cpu_count(),
            python=platform.python_version(),
            numpy=np.__version__,
            threads={
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
            load_average_at_start=load,
            load_average_at_end=os.getloadavg(),
            concurrent_load="otras sesiones comparten la máquina, ver las cargas medias",
        ),
        not_measured=[
            "GPU, porque la escalera es de CPU y no tiene tensores",
            "series reales de S, cerradas hasta la segunda ola",
            "energía, sin instrumento adecuado",
        ],
        cases=cases,
    )
    atomic_json(args.output, receipt)
    print(json.dumps({name: case["ladder"]["median_s"] for name, case in cases.items()}))


if __name__ == "__main__":
    main()
