"""Coste de la auditoría del código del preprocesado que repite cada publicación (#442).

`past_only_preprocessing.audit_code` recalcula las dos implementaciones del residualizador y
las rutas de régimen con precios deterministas del propio módulo, una vez sin alterar y una
vez por corte. No lee la edición, no ajusta nada y no abre la reserva de 2024. Mide cada
parte por separado y la auditoría completa, con repeticiones alternas.

    PYTHONPATH=src uv run --no-sync python \\
        reports/engineering/publication-preprocessing-20261013/measure_audit_cost.py \\
        --repetitions 5 --output <medida.json>
"""

import argparse
import json
import os
import platform
import resource
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mars_titan.data.budget_targets import residual_targets
from mars_titan.data.residual_arrays import residual_targets_array
from mars_titan.data.storage import atomic_json
from mars_titan.evaluation import regime_calendar
from mars_titan.integrity import past_only_preprocessing as past


def _summary(values):
    return dict(
        seconds=[round(value, 4) for value in values],
        median=round(statistics.median(values), 4),
        minimum=round(min(values), 4),
        maximum=round(max(values), 4),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parts = dict(
        residual_reference=lambda: past.audit_residualizer(residual_targets),
        residual_arrays=lambda: past.audit_residualizer(residual_targets_array),
        regimes=lambda: past.audit_regimes(regime_calendar.market_routes),
        audit_code=past.audit_code,
    )
    results = {name: run() for name, run in parts.items()}
    assert results["audit_code"]["passed"]
    timings = {name: [] for name in parts}
    for _ in range(args.repetitions):
        for name, run in parts.items():
            began = time.perf_counter()
            run()
            timings[name].append(time.perf_counter() - began)
    clock, _, _ = past.residual_history()
    days, _, closes = past.regime_history()
    document = dict(
        schema_version=1,
        kind="past_only_preprocessing_audit_cost",
        history=dict(
            residual_sessions=len(clock.days),
            regime_sessions=len(days),
            regime_assets=int(closes.shape[0]),
            cuts=list(past.AUDIT_CUTS),
        ),
        compared={
            name: dict(compared=entry["compared"], changed=len(entry["changed"]))
            for name, entry in results["audit_code"].items()
            if isinstance(entry, dict)
        },
        repetitions=args.repetitions,
        timings={name: _summary(values) for name, values in timings.items()},
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        host=dict(
            platform=platform.platform(),
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            load_average=os.getloadavg(),
        ),
        measured_at_utc=datetime.now(UTC).isoformat(),
        training_executed=False,
        final_test_opened=False,
    )
    atomic_json(args.output, document)
    print(json.dumps({k: v for k, v in document.items() if k != "host"}, indent=1))


if __name__ == "__main__":
    main()
