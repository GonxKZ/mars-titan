"""Generar escenarios separados, con semillas de evaluación independientes."""

import argparse
import json
import re
import resource
import signal
import threading
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.episodes.encoding import EncodedWorld
from mars_titan.episodes.storage import write_world
from mars_titan.episodes.worlds import WorldConfig, generate_world, recipe_fingerprints
from mars_titan.training.experiment_resources import GpuLease, check_host_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/episodes/scenarios.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-encoding", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (
        config.get("schema_version") != 1
        or config.get("final_test_opened") is not False
        or set(config["train_seeds"]) & set(config["validation_seeds"])
    ):
        raise ValueError("El diseño requiere test cerrado y semillas separadas")
    if (
        not isinstance(config["scenarios"], dict)
        or not 1 <= len(config["scenarios"]) <= 8
        or any(not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", name) for name in config["scenarios"])
    ):
        raise ValueError("Los nombres de escenario no cumplen el contrato")
    for partition in ("train", "validation"):
        seeds = config[partition + "_seeds"]
        if (
            not isinstance(seeds, list)
            or not 1 <= len(seeds) <= 8
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        ):
            raise ValueError("Las semillas deben ser únicas y estar acotadas")
    outside_source(Path("dataset"), args.output)
    if args.output.exists() and not args.resume:
        raise ValueError("La generación requiere una salida nueva o recuperación explícita")
    identity = {
        "config_sha256": sha256(args.config),
        "script_sha256": sha256(Path(__file__)),
        "transformations": recipe_fingerprints(),
        "modules": {
            name: sha256(Path(__file__).parents[1] / "src/mars_titan/episodes" / name)
            for name in ("worlds.py", "encoding.py", "storage.py")
        },
    }
    existing = args.output / "summary.json"
    if args.resume and (
        not existing.is_file() or json.loads(existing.read_text()).get("identity") != identity
    ):
        raise ValueError("La campaña no tiene la identidad de configuración esperada")
    started = time.perf_counter()
    summary = dict(
        schema_version=1,
        activity="synthetic_generation",
        model="factor_world",
        domain="synthetic",
        status="running",
        runs=[],
        config_sha256=sha256(args.config),
        identity=identity,
        final_test_opened=False,
    )
    atomic_json(existing, summary)
    phase = "generation"
    try:
        with ExitStack() as cleanup:
            stop = threading.Event()
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous = signal.signal(signum, lambda *_args: stop.set())
                cleanup.callback(signal.signal, signum, previous)
            encoders = None
            check_resources = check_host_memory
            if args.reference_encoding:
                phase = "admission"
                lease = cleanup.enter_context(GpuLease())
                phase = "encoding"
                check_resources = lease.check
                from mars_titan.data.embeddings import FrozenEncoders

                encoders = FrozenEncoders()
                check_resources()
                reference = json.loads(args.reference_encoding.read_text())
            for partition in ("train", "validation"):
                for scenario, signal_strength in config["scenarios"].items():
                    for seed in config[partition + "_seeds"]:
                        case_started = time.perf_counter()
                        started_at = datetime.now(UTC).isoformat()
                        world = generate_world(
                            WorldConfig(
                                **config["generator"],
                                seed=seed,
                                signal=signal_strength,
                                partition=partition,
                            )
                        )
                        if encoders:
                            world = EncodedWorld(
                                world,
                                encoders,
                                expected_spec=reference["configuration"]["encoders"],
                            )
                        folder = args.output / f"{partition}-{scenario}-s{seed}"
                        result = write_world(
                            world,
                            folder,
                            resume=args.resume and folder.exists(),
                            stop=stop.is_set,
                            check_resources=check_resources,
                        )
                        atomic_json(
                            folder / "run.json",
                            dict(
                                schema_version=1,
                                activity="synthetic_generation",
                                model="factor_world",
                                domain="synthetic",
                                status=result["status"],
                                identity={
                                    "case": {"kind": "factor_world", "seed": seed},
                                    "manifest_sha256": world.source_sha256,
                                },
                                samples={partition: sum(r[1] for r in world.index)},
                                started_at_utc=started_at,
                                finished_at_utc=datetime.now(UTC).isoformat(),
                                total_seconds=time.perf_counter() - case_started,
                                final_test_opened=False,
                            ),
                        )
                        summary["runs"].append(
                            dict(
                                id=folder.name,
                                path=folder.name,
                                status=result["status"],
                                rows=sum(r[1] for r in world.index),
                                partition=partition,
                                encoding=result["encoding"],
                            )
                        )
                        summary["status"] = "paused" if stop.is_set() else "running"
                        atomic_json(args.output / "summary.json", summary)
                        if stop.is_set():
                            return 0
            summary.update(
                status="completed",
                elapsed_seconds=time.perf_counter() - started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            )
            atomic_json(args.output / "summary.json", summary)
    except Exception as error:
        summary.update(
            status="blocked" if phase == "admission" else "failed", error_type=type(error).__name__
        )
        atomic_json(existing, summary)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
