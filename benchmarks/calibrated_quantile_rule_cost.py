"""Medir en CPU la regla fija sobre cuantiles calibrados y no operar (G2, #748).

La regla no aprende, así que su coste es el de recorrer el entorno: una decisión por sesión
con el motor contable en Python. Las cintas son sintéticas con semilla fija y solo fijan las
formas: 64 y 256 activos durante 252 sesiones, con puntuaciones que afirman una subida en
una parte de las filas. Se mide `compare`, que evalúa la regla y no operar sobre el mismo
entorno, y por separado el tiempo de decidir de la regla. Nada se ajusta ni se selecciona.
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

from benchmarks.output_stage_cost import timed
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.simulation import calibrated_quantile_rule as rule
from mars_titan.simulation.environment import FinancialEnv
from mars_titan.simulation.market import MarketTape

SESSIONS = 252
# Activos y repeticiones de cada caso.
CASES = {"assets_64": (64, 3), "assets_256": (256, 3)}


def tape(assets, seed):
    rng = np.random.default_rng(seed)
    close = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, (SESSIONS, assets)), axis=0))
    prices = np.empty((SESSIONS, assets, 5))
    prices[:, :, :4] = close[:, :, None]
    prices[:, :, 4] = 1e7
    # Una afirmación alcista en un 10 % de las filas y cero en el resto, como la salida de A2.
    claims = rng.random((SESSIONS, assets)) < 0.1
    scores = np.where(claims, rng.uniform(0.001, 0.02, (SESSIONS, assets)), 0.0)
    times = 100 * (np.arange(SESSIONS) + 1)
    return MarketTape(
        prices,
        times,
        [f"A{index:04d}" for index in range(assets)],
        scores,
        domain="synthetic",
        currency="USD",
        partition="validation",
        open_times=times - 50,
    )


def measure_case(assets, repeats):
    source = tape(assets, seed=assets)
    reports, cost = timed(lambda: rule.compare(FinancialEnv(source, cost_bps=10)), repeats)
    env = FinancialEnv(source)
    observation, _ = env.reset(seed=42)
    started = time.perf_counter()
    for _ in range(1000):
        rule.rule_policy(observation, 0)
    decision_s = (time.perf_counter() - started) / 1000
    return dict(
        assets=assets,
        sessions=SESSIONS,
        compare=cost,
        rule_decision_s=decision_s,
        steps={name: report["financial_validation"]["steps"] for name, report in reports.items()},
    )


def run(args):
    """Mide los casos con los argumentos de la orden y devuelve el recibo."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    load = os.getloadavg()
    cases = {name: measure_case(*shape) for name, shape in CASES.items()}
    sources = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (
            "src/mars_titan/simulation/calibrated_quantile_rule.py",
            "benchmarks/calibrated_quantile_rule_cost.py",
        )
    }
    receipt = dict(
        schema_version=1,
        kind="calibrated_quantile_rule_cost",
        issue=748,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=commit,
        sources_sha256=sources,
        data="cintas sintéticas con semilla fija, solo para fijar formas. Nada se ajusta",
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
            "motor contable nativo, sin binarios compilados vigentes",
            "GPU, porque la regla no tiene tensores",
            "cintas reconstruidas con datos reales, bloqueadas con la evaluación científica",
            "energía, sin instrumento adecuado",
        ],
        cases=cases,
    )
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("La medida necesita un destino nuevo")
    receipt = run(args)
    # atomic_json crea la carpeta, escribe en un temporal y lo confirma con os.replace.
    atomic_json(args.output, receipt)
    print(
        json.dumps({name: case["compare"]["median_s"] for name, case in receipt["cases"].items()})
    )


if __name__ == "__main__":
    main()
