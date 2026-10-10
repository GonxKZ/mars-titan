"""Paridad CPU y CUDA de la evaluación congelada sobre cintas reales, sin aprender.

Para cada mercado monta con `rl_stage_tapes` la cinta de evaluación de 2023 de la última
ventana de política y, con el mismo diseño de activos, dos cintas mensuales reales de
noviembre (ajuste) y diciembre de 2022 (validación). Las puntuaciones son sintéticas y no
proceden de ningún modelo. Double DQN recorre 32 transiciones sin actualizar y KLPO recoge
una oleada sin llegar a `update_ready`, con una selección cerrada fabricada con su actor
inicial. Las dos políticas se evalúan con `--decisions` en CPU y en `cuda:0` con los mismos
pesos, y `device_parity.compare_runs` compara acciones, logits y patrimonio.

Los binarios exigen la protección de aprendizaje para leer cintas reconstruidas. La medida
la levanta solo para sus procesos con un archivo temporal, porque no ajusta ni evalúa
ninguna política aprendida. Los binarios se lanzan con el lanzador de la etapa, que en
`cuda:0` toma el bloqueo GPU exclusivo y el presupuesto de VRAM. Se ejecuta con `memslot gpu`
y la GPU libre.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from benchmarks import rl_stage_tapes
from mars_titan.data.storage import atomic_json
from mars_titan.evaluation.splits import build_folds
from mars_titan.simulation import device_parity, native_policy_runs, window_tapes
from mars_titan.simulation.market_rules import china_a_share_instrument
from mars_titan.simulation.storage import read_tape, write_tape
from mars_titan.training.learning_hold import HOLD_ENV
from tests.simulation.policy_tape_fixture import monthly_protocol, monthly_window

POLICIES = (
    Path(__file__).resolve().parents[1] / "configs/simulation/historical-masked-rl-policies.json"
)
MONTHS = (("train", "2022-11-01"), ("validation", "2022-12-01"))


def stage(environments):
    """Etapa declarada con el presupuesto de diagnóstico de 32 transiciones.

    Double DQN necesita 16 entornos para su evaluación adaptativa. KLPO usa uno, para que su
    oleada recorra una sola cinta mensual y quepa en esas 32 transiciones.
    """
    policies = json.loads(POLICIES.read_text())
    policies["budget"] = dict(
        transitions=32,
        environments=environments,
        rollout_transitions=16,
        evaluation_transitions=32,
    )
    policies["hyperparameters"].update(minibatch_size=16, epochs=2)
    return dict(policies=policies)


def job(arm, engine, market):
    return dict(
        id=f"{market}/{market}/parity/{arm}/fit-s42",
        kind="fit",
        arm=arm,
        engine=engine,
        seed=42,
        market=market,
    )


def prepare(edition, root, market, *, candidates, max_assets):
    """Cintas de ajuste, validación y evaluación reales con un mismo diseño de activos."""
    annual = rl_stage_tapes.prepare(
        edition, root / "annual", market, candidates=candidates, max_assets=max_assets
    )
    evaluation = Path(next(t["folder"] for t in annual["tapes"] if t["role"] == "evaluation"))
    symbols = [asset.split("/", 1)[1] for asset in read_tape(evaluation).assets]
    folds = build_folds(monthly_protocol(market))
    starts = [fold["evaluation"][0] for fold in folds]
    paths = {"evaluation": evaluation}
    for role, start in MONTHS:
        receipt, values = monthly_window(market, starts.index(start), symbols)
        tape, _ = window_tapes.build_segment_tape(
            edition, receipt, values, market=market, role=role, lag=0, symbols=symbols
        )
        if [asset.split("/", 1)[1] for asset in tape.assets] != symbols:
            raise ValueError(f"El diseño de {market} pierde activos en {start}")
        rules = {a: china_a_share_instrument(a) for a in tape.assets} if market == "CN" else None
        write_tape(tape, root / market / role, instruments=rules)
        paths[role] = root / market / role
    return paths, len(symbols)


def launch(binary, hold, arguments, cuda=False):
    """Lanzar un binario con el lanzador de la etapa, en el diagnóstico CPU o en cuda:0."""
    environment = dict(os.environ, **{HOLD_ENV: str(hold)})
    environment["CUDA_VISIBLE_DEVICES"] = "0" if cuda else "-1"
    command = [sys.executable, str(native_policy_runs.LAUNCHER), "--binary", str(binary)]
    command += [*map(str, arguments)] + ([] if cuda else ["--diagnostic"])
    started = time.perf_counter()
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=3600, check=False
    )
    return result, time.perf_counter() - started


def _seal(path, changes):
    envelope = json.loads(path.read_text())
    payload = dict(envelope["payload"], **changes)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    path.write_text(
        json.dumps(dict(payload=payload, sha256=hashlib.sha256(encoded.encode()).hexdigest()))
    )


def evaluate(engine, market, paths, root, hold, cuda=False):
    """Ajuste sin actualizaciones y evaluación con registro en un dispositivo."""
    root.mkdir(parents=True)
    name = "native_ppo" if engine == "double_dqn" else "native_klpo"
    binary = native_policy_runs.binary_path(name)
    if engine == "double_dqn":
        document = native_policy_runs.ppo_config(stage(16), job("double_dqn", name, market))
    else:
        document = native_policy_runs.klpo_config(stage(1), job("klpo_terminal", name, market))
    config = root / "config.json"
    config.write_text(json.dumps(document))
    fit = root / "fit"
    arguments = ["--config", config, "--output", fit, "--train-tape", paths["train"]]
    arguments += ["--validation-tape", paths["validation"]]
    if engine == "klpo":
        arguments += ["--stop-after", len(read_tape(paths["train"])) - 1]
    result, fit_seconds = launch(binary, hold, arguments, cuda)
    if result.returncode not in (0, 2):
        raise RuntimeError(result.stderr)
    run = json.loads((fit / "run.json").read_text())
    if run["optimizer_steps"] != 0:
        raise RuntimeError("El ajuste ejecutó pasos de optimizador")
    selection = fit
    if engine == "klpo":
        # Selección cerrada con el actor inicial, solo para recorrer la evaluación congelada.
        selection = root / "closed"
        shutil.copytree(fit, selection)
        planned = json.loads((fit / "experiment.json").read_text())["payload"]["planned_waves"]
        _seal(selection / "selection.json", dict(status="completed", evaluated_waves=planned))
    output = root / "evaluation"
    audit = ["--config", config, "--audit-run", selection, "--audit-tape", paths["evaluation"]]
    result, audit_seconds = launch(binary, hold, [*audit, "--output", output, "--decisions"], cuda)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return device_parity.read_run(output), dict(
        fit_seconds=round(fit_seconds, 2), audit_seconds=round(audit_seconds, 2)
    )


def cpu_identity():
    """Modelo de CPU de /proc/cpuinfo, o la arquitectura si el sistema no lo publica."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.machine()


def gpu_identity():
    import torch

    query = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    return dict(
        nvidia_smi=subprocess.run(query, capture_output=True, text=True, check=True).stdout.strip(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edition", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--markets", nargs="+", default=["US", "CN"])
    parser.add_argument("--engines", nargs="+", default=["double_dqn", "klpo"])
    parser.add_argument("--candidates", type=int, default=400)
    parser.add_argument("--max-assets", type=int, default=128)
    args = parser.parse_args()
    names = {"double_dqn": "native_ppo", "klpo": "native_klpo"}
    binaries = {
        engine: native_policy_runs.probe_binary(names[engine], device_parity.DECISION_LOGITS)
        for engine in args.engines
    }
    args.output.mkdir(parents=True, exist_ok=False)
    hold = args.output / "training-hold.json"
    hold.write_text(json.dumps({"training_allowed": True}))
    results = []
    for market in args.markets:
        paths, layout = prepare(
            args.edition,
            args.output / "tapes",
            market,
            candidates=args.candidates,
            max_assets=args.max_assets,
        )
        for engine in args.engines:
            root = args.output / market / engine
            cpu, cpu_times = evaluate(engine, market, paths, root / "cpu", hold)
            cuda, cuda_times = evaluate(engine, market, paths, root / "cuda", hold, cuda=True)
            # La referencia es cuda:0, donde la etapa selecciona y audita cada política.
            summary = device_parity.compare_runs(cuda, cpu)
            results.append(
                dict(
                    market=market,
                    engine=engine,
                    binary_sha256=binaries[engine]["binary_sha256"],
                    native_source_sha256=binaries[engine]["native_source_sha256"],
                    layout_assets=layout,
                    evaluation_sessions=len(read_tape(paths["evaluation"])),
                    evaluation_tapes=cuda[0]["identity"]["tapes"],
                    cpu=cpu_times,
                    cuda=cuda_times,
                    **summary,
                )
            )
            print(json.dumps(results[-1])[:600], flush=True)
    report = dict(
        schema_version=1,
        kind="policy_device_parity",
        recorded_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        edition=str(args.edition),
        scores="synthetic_not_from_any_model",
        weights="initial_without_optimizer_steps",
        tolerance=device_parity.TOLERANCE,
        margin_limits=list(device_parity.MARGIN_LIMITS),
        cpu=cpu_identity(),
        gpu=gpu_identity(),
        results=results,
    )
    atomic_json(args.report, report)


if __name__ == "__main__":
    main()
