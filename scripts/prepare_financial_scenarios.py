"""Preparar escenarios contables con la referencia analítica de eventos ficticios."""

import argparse
import json
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json
from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.storage import read_tape, write_tape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/episodes/scenarios.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings, _ = read_manifest(args.config, 1024**2)
    if settings.get("schema_version") != 1 or settings.get("final_test_opened") is not False:
        raise ValueError("La configuración debe mantener el test cerrado")
    if set(settings["train_seeds"]) & set(settings["validation_seeds"]):
        raise ValueError("Las semillas de ajuste y evaluación deben ser distintas")
    safe_destination(args.output)
    if args.output.exists() and not args.resume:
        raise ValueError("Usa una salida nueva o recuperación explícita")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    records = []
    for scenario, signal in settings["scenarios"].items():
        if scenario not in {"no_signal", "known_signal"}:
            raise ValueError("El escenario no está definido")
        for partition in ("train", "validation"):
            for seed in settings[f"{partition}_seeds"]:
                world = generate_world(
                    WorldConfig(
                        **settings["generator"], seed=seed, partition=partition, signal=signal
                    )
                )
                tape = MarketTape.from_world(
                    world, lambda x: 0.002 * x["news"][:, 0], parent_id="fixed_event_reference-v1"
                )
                destination = args.output / f"{scenario}-{partition}-{seed}"
                if destination.exists():
                    if read_tape(destination).sha256 != tape.sha256:
                        raise ValueError("El escenario previo pertenece a otra identidad")
                else:
                    write_tape(tape, destination)
                records.append(
                    dict(name=destination.name, sha256=tape.sha256, partition=partition, seed=seed)
                )
    atomic_json(
        args.output / "index.json",
        dict(
            schema_version=1,
            domain="synthetic",
            parent="fixed_event_reference-v1",
            records=records,
            final_test_opened=False,
        ),
    )
    print(json.dumps(dict(scenarios=len(records), final_test_opened=False)))


if __name__ == "__main__":
    main()
