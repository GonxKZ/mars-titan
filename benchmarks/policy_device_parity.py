"""Paridad CPU y CUDA de la evaluación congelada de las políticas iniciales, sin aprender.

Para cada mercado monta con `rl_stage_tapes` la cinta de evaluación de 2023 de la última
ventana de política, con precios reales de la edición sin ajustar y puntuaciones sintéticas
que no proceden de ningún modelo. Cada política se evalúa con `--initial-evaluation
--decisions`, el modo sin aprendizaje que la protección admite sin rebajarse (#546): los
binarios activan su cerrojo de solo evaluación y no crean entrenador, controlador ni Adam.
Los pesos son los iniciales de la semilla de la configuración y salen del generador de CPU,
así que la CPU y `cuda:0` evalúan los mismos parámetros. `device_parity.compare_runs`
compara acciones, salidas y patrimonio con `cuda:0` como referencia (#439).

Los binarios se lanzan con el lanzador de la etapa, que en `cuda:0` toma el bloqueo GPU
exclusivo y el presupuesto de VRAM. La protección de aprendizaje no se toca. Se ejecuta con
`memslot gpu` y la GPU libre.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from benchmarks import rl_stage_tapes
from mars_titan.data.storage import atomic_json
from mars_titan.hardware import platform_identity
from mars_titan.simulation import device_parity, native_policy_runs
from mars_titan.simulation.listing_status import read_listing_status
from mars_titan.simulation.storage import read_tape

POLICIES = (
    Path(__file__).resolve().parents[1] / "configs/simulation/historical-masked-rl-policies.json"
)
# Una configuración por red: las tres variantes PPO comparten actor y crítico iniciales.
ARMS = {
    "ppo": ("ppo_clip_full_kl", "native_ppo"),
    "double_dqn": ("double_dqn", "native_ppo"),
    "klpo": ("klpo_terminal", "native_klpo"),
}
EVALUATION_ONLY = "native_policy_evaluation_only"


def config(engine, market, folder):
    """Configuración de la etapa declarada para la variante, con la semilla 42."""
    arm, binary = ARMS[engine]
    stage = dict(policies=json.loads(POLICIES.read_text()))
    job = dict(
        id=f"{market}/{market}/parity/{arm}/fit-s42",
        kind="fit",
        arm=arm,
        engine=binary,
        seed=42,
        market=market,
    )
    build = (
        native_policy_runs.ppo_config if binary == "native_ppo" else native_policy_runs.klpo_config
    )
    path = folder / "config.json"
    folder.mkdir(parents=True)
    path.write_text(json.dumps(build(stage, job)))
    return path


def launch(binary, arguments, *, cuda):
    """Lanzar un binario con el lanzador de la etapa, en el diagnóstico CPU o en cuda:0."""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0" if cuda else "-1")
    command = [sys.executable, str(native_policy_runs.LAUNCHER), "--binary", str(binary)]
    command += [*map(str, arguments)] + ([] if cuda else ["--diagnostic"])
    started = time.perf_counter()
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=3600, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return round(time.perf_counter() - started, 2)


def evaluate(engine, market, tape, root, binary):
    """Evaluación inicial en CPU y en cuda:0 con su registro de decisiones."""
    path = config(engine, market, root)
    seconds, runs = {}, {}
    for device, cuda in (("cpu", False), ("cuda", True)):
        output = root / device
        arguments = ["--config", path, "--output", output, "--audit-tape", tape]
        seconds[device] = launch(
            binary["path"], [*arguments, "--initial-evaluation", "--decisions"], cuda=cuda
        )
        runs[device] = device_parity.read_run(output)
        identity = runs[device][0]["identity"]
        if identity["optimizer_steps"] != 0 or identity["transitions"] != 0:
            raise RuntimeError("La evaluación inicial declara pasos o transiciones de ajuste")
    return runs, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edition", required=True, type=Path)
    parser.add_argument("--listing-status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--markets", nargs="+", default=["US", "CN"])
    parser.add_argument("--engines", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--candidates", type=int, default=400)
    parser.add_argument("--max-assets", type=int, default=128)
    args = parser.parse_args()
    binaries = {
        name: native_policy_runs.probe_binary(name, EVALUATION_ONLY)
        for name in {ARMS[engine][1] for engine in args.engines}
    }
    for name in binaries:
        native_policy_runs.probe_binary(name, device_parity.DECISION_LOGITS)
    status = read_listing_status(args.listing_status)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for market in args.markets:
        prepared = rl_stage_tapes.prepare(
            args.edition,
            args.output / "tapes",
            market,
            candidates=args.candidates,
            max_assets=args.max_assets,
            status=status,
        )
        tape = Path(next(t["folder"] for t in prepared["tapes"] if t["role"] == "evaluation"))
        for engine in args.engines:
            binary = binaries[ARMS[engine][1]]
            runs, seconds = evaluate(engine, market, tape, args.output / market / engine, binary)
            # La referencia es cuda:0, donde la etapa selecciona y audita cada política.
            summary = device_parity.compare_runs(runs["cuda"], runs["cpu"])
            results.append(
                dict(
                    market=market,
                    engine=engine,
                    arm=ARMS[engine][0],
                    binary_sha256=binary["binary_sha256"],
                    native_source_sha256=binary["native_source_sha256"],
                    assets=len(read_tape(tape).assets),
                    evaluation_sessions=len(read_tape(tape)),
                    evaluation_tapes=runs["cuda"][0]["identity"]["tapes"],
                    seconds=seconds,
                    **summary,
                )
            )
            print(json.dumps(results[-1])[:600], flush=True)
    report = dict(
        schema_version=1,
        kind="policy_device_parity",
        recorded_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mode="initial_evaluation_without_learning",
        edition=str(args.edition),
        listing_status_sha256=status[1],
        scores="synthetic_not_from_any_model",
        weights="initial_from_configuration_seed",
        optimizer_steps=0,
        tolerance=device_parity.TOLERANCE,
        margin_limits=list(device_parity.MARGIN_LIMITS),
        cpu=platform_identity.cpu_name(),
        platform=platform_identity.platform_identity(),
        results=results,
    )
    atomic_json(args.report, report)


if __name__ == "__main__":
    main()
