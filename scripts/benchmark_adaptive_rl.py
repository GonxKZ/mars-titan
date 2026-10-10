"""Medir RL nativo por pares con fuentes fijas, calentamiento y una sola carga GPU."""

import argparse
import copy
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from benchmark_native_ppo import checkpoint_bytes, concurrent_snapshot, require

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.simulation.adaptive_campaign import (
    AUXILIARY,
    FAMILIES,
    VARIANTS,
    _confirm_bytes,
    _digest,
    _file_hash,
    _json,
    _relative,
)
from mars_titan.training.learning_hold import require_learning_allowed

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_native_ppo.py"
DEFAULT_VARIANTS = ["ppo", "double_dqn", "ppo_gru", "ppo_episodic_hmm"]
STOP_SECONDS = 75
TIMINGS = ("training_seconds", "evaluation_seconds", "checkpoint_seconds", "setup_seconds")


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


def load_sources(path):
    index, index_hash = _json(path)
    rows = index["records"]
    require(
        index.get("status") == "completed"
        and index.get("domain") == "synthetic"
        and index.get("final_test_opened") is False
        and len(rows) == 896,
        "Se necesita el catálogo sintético completo de 896 escenarios",
    )
    ordered = {}
    for split, count in (("train", 32), ("validation", 16), ("audit", 64)):
        groups = [
            sorted(
                (row for row in rows if row["split"] == split and row["family"] == family),
                key=lambda row: row["seed"],
            )
            for family in FAMILIES
        ]
        require(all(len(group) == count for group in groups), "El catálogo no está equilibrado")
        if split != "audit":
            ordered[split] = [group[at] for at in range(count) for group in groups]
    sealed, sources, arguments = {path: index_hash}, [], []
    for split, partition in ordered.items():
        for row in partition:
            directory = _relative(path.parent, row["path"])
            market, market_hash = _json(directory / "manifest.json")
            context, context_hash = _json(directory / "context.json")
            require(
                market_hash == row["manifest_sha256"]
                and context_hash == row["context_sha256"]
                and context["market_manifest_sha256"] == market_hash,
                "Cambió el manifiesto o contexto de una fuente",
            )
            sealed.update(
                {directory / "manifest.json": market_hash, directory / "context.json": context_hash}
            )
            data = []
            for file, expected in (
                (
                    directory / "market.parquet",
                    dict(bytes=market["file_bytes"], sha256=market["file_sha256"]),
                ),
                (_relative(directory, context["file"]["path"]), context["file"]),
            ):
                require(
                    _file_hash(file) == expected["sha256"]
                    and file.stat().st_size == expected["bytes"],
                    "Una fuente no conserva sus datos",
                )
                sealed[file] = expected["sha256"]
                data.append(dict(bytes=expected["bytes"], sha256=expected["sha256"]))
            sources.append(
                dict(
                    split=split,
                    family=row["family"],
                    seed=row["seed"],
                    manifest_sha256=market_hash,
                    context_sha256=context_hash,
                    files=data,
                    assets=market["assets"],
                    sessions=market["sessions"],
                    context_fields=len(context["fields"]),
                )
            )
            arguments.extend((f"--{split}-tape", str(directory)))
    hmm_path = _relative(path.parent, index["hmm"]["path"])
    hmm, hmm_hash = _json(hmm_path, 4 * 1024**2)
    require(
        hmm_hash == index["hmm"]["sha256"]
        and hmm["fit_split"] == "train"
        and set(hmm["train_manifest_sha256"])
        == {row["manifest_sha256"] for row in ordered["train"]},
        "El HMM no corresponde exclusivamente a entrenamiento",
    )
    sealed[hmm_path] = hmm_hash
    return arguments, sources, sealed, hmm_path


def run_process(command, log, timeout, environment):
    with log.open("wb") as stream:
        started = time.perf_counter()
        child = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            return child.wait(timeout=timeout), time.perf_counter() - started
        except BaseException:
            descriptors = []
            with ExitStack() as stack:
                children = Path(f"/proc/{child.pid}/task/{child.pid}/children")
                try:
                    descendants = children.read_text().split()
                except (FileNotFoundError, ProcessLookupError):
                    descendants = []
                for pid in descendants:
                    try:
                        descriptor = os.pidfd_open(int(pid))
                        stack.callback(os.close, descriptor)
                        parent = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1]
                        if int(parent) == child.pid:
                            descriptors.append(descriptor)
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                if child.poll() is not None:
                    raise
                child.terminate()
                try:
                    child.wait(timeout=STOP_SECONDS)
                except subprocess.TimeoutExpired:
                    for descriptor in descriptors:
                        try:
                            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    child.kill()
                    child.wait(timeout=5)
            raise


def measure(output, config):
    receipt, _ = _json(output / "run.json")
    identity = _json(output / "identity.json")[0]
    require(
        receipt["status"] == "completed"
        and receipt["transitions"] == config["training"]["total_transitions"]
        and receipt["device"] == "cuda:0"
        and receipt["diagnostic"] is False
        and receipt["seed"] == config["training"]["seed"]
        and receipt["agent_variant"] == config["agent"]["variant"]
        and identity["identity"]["configuration"] == config
        and receipt["identity_sha256"] == identity["sha256"] == _digest(identity["identity"]),
        "El recibo no corresponde a un caso completo con su configuración",
    )
    require(
        set(receipt["timings"]) == set(TIMINGS)
        and all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0
            for value in [receipt["invocation_seconds"], *receipt["timings"].values()]
        ),
        "Los tiempos internos no son válidos",
    )
    require(
        type(receipt["resources"]["ram_peak_bytes"]) is int
        and receipt["resources"]["ram_peak_bytes"] > 0,
        "El pico RSS no es válido",
    )
    index = _json(output / "ppo-index.json")[0]
    require(
        index["sha256"] == _digest(index["payload"]), "El índice del checkpoint perdió su sello"
    )
    return dict(
        **{
            key: receipt[key]
            for key in (
                "transitions",
                "observed_transitions",
                "optimizer_steps",
                "parameters",
                "invocation_seconds",
                "timings",
            )
        },
        resources={
            key: receipt["resources"].get(key)
            for key in (
                "ram_peak_bytes",
                "ram_peak_method",
                "rollout_budget_bytes",
                "vram_budget_bytes",
                "vram_total_bytes",
                "vram_peak_bytes",
            )
        },
        torch_version=identity["identity"]["torch_version"],
        native_source_sha256=identity["identity"].get("native_source_sha256"),
        native_build_sha256=identity["identity"]["native_build_sha256"],
        disk=checkpoint_bytes(output),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scenarios", "binary", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/simulation/adaptive-ppo.json"
    )
    parser.add_argument("--reference-binary", type=Path)
    parser.add_argument("--reference-launcher", type=Path)
    parser.add_argument(
        "--variants", nargs="+", choices=(*VARIANTS, *AUXILIARY), default=DEFAULT_VARIANTS
    )
    parser.add_argument("--workers", nargs="+", type=int, choices=(1, 2, 4, 8), default=[1])
    parser.add_argument("--repetitions", type=int, choices=range(1, 11), default=5)
    parser.add_argument("--transitions", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args(argv)
    if not (
        16 <= args.transitions <= 1 << 20
        and args.transitions % 16 == 0
        and math.isfinite(args.timeout)
        and 1 <= args.timeout <= 7200
        and len(set(args.workers)) == len(args.workers)
        and len(set(args.variants)) == len(args.variants)
        and (args.reference_binary or not args.reference_launcher)
    ):
        parser.error("Los límites, combinaciones o referencia no son válidos")
    require_learning_allowed("la medición de RL nativo")
    report = None
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        safe_destination(args.output)
        output = args.output.resolve()
        require(not output.exists(), "La salida ya existe")
        arguments, sources, sealed, hmm_path = load_sources(args.scenarios.resolve())
        base, base_hash = _json(args.config)
        require(
            base["schema_version"] == 2
            and base["environments"] == 16
            and base["training"]["rollout_transitions"] == 1024
            and base["hyperparameters"]["minibatch_size"] == 64
            and base["selection"]["early_stopping"] is False,
            "La configuración no conserva el protocolo",
        )
        methods = {"candidate": (args.binary.resolve(), LAUNCHER)}
        if args.reference_binary:
            methods = {
                "reference": (
                    args.reference_binary.resolve(),
                    (args.reference_launcher or LAUNCHER).resolve(),
                ),
                **methods,
            }
        sealed.update(
            {args.config.resolve(): base_hash, Path(__file__): _file_hash(Path(__file__))}
        )
        identities = {}
        for method, (binary, launcher) in methods.items():
            require(os.access(binary, os.X_OK), "El binario no es ejecutable")
            files = dict(binary=binary, launcher=launcher)
            backend = binary.with_name("libmars_titan_simulation.so")
            if backend.exists():
                files["backend"] = backend
            identities[method] = {
                kind: dict(sha256=_file_hash(file), bytes=file.stat().st_size)
                for kind, file in files.items()
            }
            sealed.update(
                {file: identities[method][kind]["sha256"] for kind, file in files.items()}
            )
        require(
            all(not path.resolve().is_relative_to(output) for path in sealed)
            and not output.is_relative_to(args.scenarios.resolve().parent),
            "La salida debe quedar separada de entradas y binarios",
        )
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        ).strip()
        output.mkdir(parents=True)
        _confirm_bytes(output / "hmm.json", hmm_path.read_bytes())
        hmm_hash = _file_hash(output / "hmm.json")
        report = dict(
            schema_version=1,
            kind="adaptive_rl_benchmark",
            status="running",
            created_at=datetime.now(UTC).isoformat(),
            python=platform.python_version(),
            platform=platform.platform(),
            machine=platform.machine(),
            cpu_model=cpu_name(),
            logical_cpus=os.cpu_count(),
            gpu=gpu,
            sources=sources,
            inputs_bytes=sum(item["bytes"] for row in sources for item in row["files"]),
            index_sha256=sealed[args.scenarios.resolve()],
            binaries=identities,
            base_configuration_sha256=base_hash,
            harness_sha256=sealed[Path(__file__)],
            hmm_sha256=hmm_hash,
            environments=base["environments"],
            rollout_transitions=base["training"]["rollout_transitions"],
            minibatch_size=base["hyperparameters"]["minibatch_size"],
            warmup_runs=1,
            repetitions=args.repetitions,
            timeout_seconds=args.timeout,
            cache_reset=False,
            process_start="new_process",
            energy=None,
            vram_peak_bytes=None,
            records=[],
            summaries=[],
        )
        environment = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        for variant in args.variants:
            for workers in args.workers:
                config = copy.deepcopy(base)
                config["training"].update(total_transitions=args.transitions, workers=workers)
                config["agent"].update(variant=variant, markov_fields=[], hmm_file=None)
                if variant in {"ppo_hmm", "ppo_episodic_hmm", *AUXILIARY}:
                    config["agent"].update(
                        markov_fields=[4, 5, 6], hmm_file=dict(path="hmm.json", sha256=hmm_hash)
                    )
                config_path = output / f"{variant}-w{workers}.json"
                atomic_json(config_path, config)
                config_hash = _file_hash(config_path)
                for repetition in range(args.repetitions + 1):
                    order = (
                        list(methods)
                        if repetition == 0 or repetition % 2
                        else list(reversed(methods))
                    )
                    for method in order:
                        require(
                            all(_file_hash(path) == value for path, value in sealed.items()),
                            "Cambió una entrada o ejecutable del benchmark",
                        )
                        folder = output / f"{variant}-w{workers}/{repetition}-{method}"
                        folder.mkdir(parents=True)
                        binary, launcher = methods[method]
                        command = [
                            sys.executable,
                            str(launcher),
                            "--binary",
                            str(binary),
                            "--config",
                            str(config_path),
                            "--output",
                            str(folder / "run"),
                            "--watch-state",
                            str(folder / "watch.json"),
                            *arguments,
                        ]
                        environment["LD_LIBRARY_PATH"] = (
                            str(binary.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
                        )
                        concurrent = concurrent_snapshot()
                        report["active_case"] = str(folder.relative_to(output))
                        atomic_json(output / "benchmark.json", report)
                        code, elapsed = run_process(
                            command, folder / "process.log", args.timeout, environment
                        )
                        require(code == 0, f"El lanzador terminó con código {code}")
                        require(
                            all(_file_hash(path) == value for path, value in sealed.items())
                            and _file_hash(config_path) == config_hash,
                            "Cambió un archivo durante la medida",
                        )
                        report["records"].append(
                            dict(
                                method=method,
                                variant=variant,
                                workers=workers,
                                seed=config["training"]["seed"],
                                repetition=repetition,
                                warmup=repetition == 0,
                                path=str(folder.relative_to(output)),
                                configuration=config_path.name,
                                configuration_sha256=config_hash,
                                process_seconds=elapsed,
                                transitions_per_second=args.transitions / elapsed,
                                concurrent=concurrent,
                                **measure(folder / "run", config),
                            )
                        )
                        atomic_json(output / "benchmark.json", report)
                for method in methods:
                    values = [
                        row["process_seconds"]
                        for row in report["records"]
                        if row["method"] == method
                        and row["variant"] == variant
                        and row["workers"] == workers
                        and not row["warmup"]
                    ]
                    report["summaries"].append(
                        dict(
                            method=method,
                            variant=variant,
                            workers=workers,
                            count=len(values),
                            median=statistics.median(values),
                            minimum=min(values),
                            maximum=max(values),
                            stdev_sample=statistics.stdev(values) if len(values) > 1 else None,
                        )
                    )
        report["status"] = "completed"
        report["active_case"] = None
        atomic_json(output / "benchmark.json", report)
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
        KeyboardInterrupt,
    ) as error:
        if report is not None:
            report.update(status="failed", reason=type(error).__name__)
            atomic_json(output / "benchmark.json", report)
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
