"""Coste en CPU de los árbitros en línea y sus controles (#789), sin ajustar nada.

Sin --run imprime el protocolo. Con --run construye paneles ficticios con dos mercados,
128 activos por sesión y la madurez de cada etiqueta cinco sesiones después de decidir, y
mide `online_mixture` y `uniform_mixture` en FP64. Con cinco opciones (la época común y
cuatro versiones) mide el árbitro de versiones, su control desordenado y la media uniforme.
Con 32 y 80 opciones mide la combinación de modelos y su media simple. La criba compara 78
modelos con predicciones propias de calibración y S, que son 67 opciones sin los trabajos
repetidos.

Las dos pasadas invierten el orden de los modos y comparten equipo con las cargas que ya
existan, así que los percentiles son descriptivos. La memoria es el pico de tracemalloc en
una llamada aparte, fuera de la medida de tiempo. La salida debe ser nueva.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

DAY_US = 86_400_000_000
MARKETS, ASSETS, LAG_SESSIONS, WINDOW, MIN_ASSETS, SEED = ("US", "CN"), 128, 5, 21, 3, 20261028
# (sesiones por mercado, opciones). 320 sesiones son la calibración y un año de S.
SHAPES = ((64, 5), (320, 5), (320, 32), (320, 80))
MODES = {
    5: ("version_arbiter", "shuffled_control", "uniform_mixture"),
    32: ("model_combination", "simple_mean"),
    80: ("model_combination", "simple_mean"),
}
WARMUP, REPETITIONS = 1, 5


def protocol():
    return dict(
        kind="online_arbiters_cost_v1",
        device="cpu",
        dtype="float64",
        markets=list(MARKETS),
        assets_per_session=ASSETS,
        maturity="each label matures five sessions after its decision",
        shapes=[
            dict(sessions_per_market=sessions, options=options, modes=list(MODES[options]))
            for sessions, options in SHAPES
        ],
        reliability_window_sessions=WINDOW,
        min_assets=MIN_ASSETS,
        shuffled_seed=SEED,
        warmup=WARMUP,
        repetitions=REPETITIONS,
        orders="each shape runs its modes in the declared order and then reversed",
        memory="tracemalloc peak of one separate call",
    )


def inputs(sessions, options):
    """Panel ficticio, predicciones con algo de señal y madurez de cada fila."""
    import numpy as np
    import pyarrow as pa

    from mars_titan.evaluation.forecast_panel import ForecastPanel

    rng = np.random.default_rng(sessions * 100 + options)
    rows = len(MARKETS) * sessions * ASSETS
    market = np.repeat(np.array(MARKETS), sessions * ASSETS)
    day = np.tile(np.repeat(np.arange(sessions), ASSETS), len(MARKETS))
    asset = np.tile(np.arange(ASSETS), len(MARKETS) * sessions)
    offset = np.repeat(np.arange(len(MARKETS)), sessions * ASSETS) * 3_600_000_000
    at = 1_546_380_000_000_000 + day * DAY_US + offset
    table = pa.table(
        dict(
            sample_id=[f"{m}/A{a:04d}/{d:05d}" for m, a, d in zip(market, asset, day, strict=True)],
            market=market.tolist(),
            prediction_at=pa.array(at, type=pa.timestamp("us", tz="UTC")),
            target=rng.standard_normal(rows),
            prediction=np.zeros(rows),
        )
    )
    panel = ForecastPanel.from_arrow(table)
    signal = rng.uniform(0.0, 0.2, options)
    predictions = signal * panel.target[:, None] + rng.standard_normal((panel.rows, options))
    matures = np.asarray(panel.prediction_at, dtype=np.int64) + LAG_SESSIONS * DAY_US
    return panel, predictions, matures


def call(mode, panel, predictions, matures):
    from mars_titan.evaluation import online_arbiters as arbiters

    settings = dict(window=WINDOW, min_assets=MIN_ASSETS)
    if mode == "version_arbiter":
        return arbiters.online_mixture(panel, predictions, matures, margin=0.0, **settings)
    if mode == "shuffled_control":
        return arbiters.online_mixture(
            panel, predictions, matures, margin=0.0, seed=SEED, **settings
        )
    if mode == "model_combination":
        return arbiters.online_mixture(panel, predictions, matures, margin=None, **settings)
    return arbiters.uniform_mixture(panel, predictions)


def measure(shapes=SHAPES, *, warmup=WARMUP, repetitions=REPETITIONS):
    records, memory = [], []
    for sessions, options in shapes:
        data = inputs(sessions, options)
        modes = MODES[options]
        for order in (modes, tuple(reversed(modes))):
            for mode in order:
                for _ in range(warmup):
                    call(mode, *data)
                samples = []
                for _ in range(repetitions):
                    start = time.perf_counter_ns()
                    call(mode, *data)
                    samples.append((time.perf_counter_ns() - start) / 1e9)
                records.append(dict(mode=mode, sessions=sessions, options=options, s=samples))
        for mode in modes:
            tracemalloc.start()
            call(mode, *data)
            memory.append(
                dict(
                    mode=mode,
                    sessions=sessions,
                    options=options,
                    peak_bytes=tracemalloc.get_traced_memory()[1],
                )
            )
            tracemalloc.stop()
    summary = []
    for sessions, options in shapes:
        for mode in MODES[options]:
            pooled = sorted(
                value
                for record in records
                if (record["mode"], record["sessions"], record["options"])
                == (mode, sessions, options)
                for value in record["s"]
            )
            summary.append(
                dict(
                    mode=mode,
                    sessions=sessions,
                    options=options,
                    p50_s=statistics.median(pooled),
                    p95_s=pooled[min(len(pooled) - 1, round(0.95 * (len(pooled) - 1)))],
                )
            )
    return dict(summary=summary, memory=memory, records=records)


def run():
    import numpy as np

    load = os.getloadavg()
    result = measure()
    return dict(
        protocol=protocol(),
        finished_at=datetime.now(UTC).isoformat(),
        environment=dict(
            python=sys.version.split()[0],
            numpy=np.__version__,
            machine=platform.machine(),
            processor=platform.processor(),
            threads={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            load_average_1min=dict(start=load[0], end=os.getloadavg()[0]),
        ),
        **result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not arguments.run:
        print(json.dumps(protocol(), indent=2))
        return
    if arguments.output is None or arguments.output.exists():
        raise SystemExit("La medida necesita --output con un destino nuevo")
    result = run()
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
