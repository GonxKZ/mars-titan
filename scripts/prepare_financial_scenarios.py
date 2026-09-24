"""Preparar escenarios sintéticos con una referencia analítica o un padre verificado."""

import argparse
import fcntl
import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.storage import read_tape, write_tape

ANALYTIC_PARENT = "fixed_event_reference-v1"


def _cases(settings):
    if settings.get("schema_version") != 1 or settings.get("final_test_opened") is not False:
        raise ValueError("La configuración debe mantener el test cerrado")
    for partition in ("train", "validation"):
        seeds = settings.get(f"{partition}_seeds")
        if (
            not isinstance(seeds, list)
            or not 1 <= len(seeds) <= 32
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("Las semillas deben ser únicas, enteras y acotadas")
    if set(settings["train_seeds"]) & set(settings["validation_seeds"]):
        raise ValueError("Las semillas de ajuste y evaluación deben ser distintas")
    scenarios = settings.get("scenarios")
    if (
        not isinstance(scenarios, dict)
        or not scenarios
        or not set(scenarios) <= {"no_signal", "known_signal"}
    ):
        raise ValueError("El escenario no está definido")
    cases = []
    for scenario, signal in scenarios.items():
        for partition in ("train", "validation"):
            for seed in settings[f"{partition}_seeds"]:
                world = WorldConfig(
                    **settings["generator"], seed=seed, partition=partition, signal=signal
                )
                world.validate()
                cases.append((f"{scenario}-{partition}-{seed}", world))
    return cases


@contextmanager
def _trained_prediction(paths, context):
    from mars_titan.training.experiment_resources import GpuLease

    with GpuLease() as lease:
        from mars_titan.data.embeddings import FrozenEncoders
        from mars_titan.episodes.encoding import EncodedWorld
        from mars_titan.posttraining.parents import load_parent
        from mars_titan.posttraining.preparation import encoder_contract

        binding = encoder_contract(paths["supervision"], paths["encoded"])
        ordered, ordered_hash = read_manifest(paths["ordered"], 8 * 1024**2)
        supervision, supervision_hash = read_manifest(paths["supervision"], 8 * 1024**2)
        shapes = dict(prices=[context, 5], news=[384], charts=[512], fundamentals=[45], macro=[420])
        if (
            ordered.get("kind") != "causal_prediction_corpus"
            or ordered.get("status") != "completed"
            or ordered.get("final_test_opened") is not False
            or ordered.get("source_sha256") != binding["supervision_sha256"]
            or ordered.get("identity", {}).get("source_sha256") != binding["supervision_sha256"]
            or supervision_hash != binding["supervision_sha256"]
            or ordered.get("counts") != supervision.get("counts")
            or ordered.get("shapes") != shapes
            or binding["context"] != context
            or any(
                ordered.get(key) != supervision.get(key)
                for key in ("cohort_id", "news_content_policy")
            )
        ):
            raise ValueError(
                "El corpus ordenado, la supervisión y la codificación no comparten contrato"
            )
        parent = load_parent(paths["ordered"], paths["parent"], lease=lease)
        lease.check()
        if (
            parent.shapes != {key: tuple(shape) for key, shape in shapes.items()}
            or parent.identity["ordered_manifest_sha256"] != ordered_hash
        ):
            raise ValueError("El padre no conserva las dimensiones y el contexto del escenario")
        encoders = FrozenEncoders()
        if encoders.spec != binding["encoders"]:
            raise ValueError("Los codificadores no coinciden con los utilizados por el padre")
        lease.check()
        proof = dict(mode="trained", binding=binding, parent=parent.identity)
        yield (
            lambda world: EncodedWorld(world, encoders, expected_spec=binding["encoders"]),
            parent.predict,
            parent.identity["checkpoint_sha256"],
            proof,
            lease.check,
        )


def _code(trained):
    root = Path(__file__).parents[1]
    files = [
        "scripts/prepare_financial_scenarios.py",
        "src/mars_titan/simulation/market.py",
        "src/mars_titan/simulation/storage.py",
        "src/mars_titan/episodes/worlds.py",
    ]
    if trained:
        files += ["src/mars_titan/episodes/encoding.py", "src/mars_titan/posttraining/parents.py"]
    return {name: sha256(root / name) for name in files}


def _prepare(output, cases, identity, prediction, *, resume):
    encode, predict, parent_id, _proof, check_resources = prediction
    if resume:
        index, _ = read_manifest(output / "index.json", 8 * 1024**2)
        if (
            set(index)
            != {"schema_version", "domain", "parent", "identity", "records", "final_test_opened"}
            or type(index.get("schema_version")) is not int
            or index["schema_version"] != 1
            or index.get("domain") != "synthetic"
            or index.get("identity") != identity
            or index.get("parent") != parent_id
            or index.get("final_test_opened") is not False
        ):
            raise ValueError("La recuperación pertenece a otra identidad de preparación")
        expected_names = [name for name, _ in cases]
        records = index.get("records")
        if (
            not isinstance(records, list)
            or [r.get("name") for r in records] != expected_names[: len(records)]
            or len(records) > len(cases)
        ):
            raise ValueError("El índice no conserva los escenarios confirmados")
    else:
        index = dict(
            schema_version=1,
            domain="synthetic",
            parent=parent_id,
            identity=identity,
            records=[],
            final_test_opened=False,
        )
        output.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(output / ".prepare.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(output / "index.json", index)
        for position, (name, config) in enumerate(cases):
            destination = output / name
            previous = read_tape(destination) if destination.exists() else None
            if check_resources is not None:
                check_resources()
            world = encode(generate_world(config))
            tape = MarketTape.from_world(
                world, predict, parent_id=parent_id, check_resources=check_resources
            )
            if check_resources is not None:
                check_resources()
            if _code(identity["prediction"]["mode"] == "trained") != identity["code"]:
                raise ValueError("El código ha cambiado durante la preparación")
            if previous is not None:
                if previous.sha256 != tape.sha256:
                    raise ValueError("El escenario previo pertenece a otra identidad")
            elif position < len(index["records"]):
                raise ValueError("Falta un escenario previamente confirmado")
            else:
                write_tape(tape, destination)
            record = dict(
                name=name, sha256=tape.sha256, partition=config.partition, seed=config.seed
            )
            if position < len(index["records"]):
                if index["records"][position] != record:
                    raise ValueError("El escenario ya no coincide con su registro confirmado")
            else:
                index["records"].append(record)
                atomic_json(output / "index.json", index)
        return index
    finally:
        os.close(descriptor)


def prepare_scenarios(
    config, output, *, resume=False, parent=None, ordered=None, supervision=None, encoded=None
):
    paths = dict(parent=parent, ordered=ordered, supervision=supervision, encoded=encoded)
    supplied = [value is not None for value in paths.values()]
    if any(supplied) and not all(supplied):
        raise ValueError("parent, ordered, supervision y encoded forman un conjunto obligatorio")
    config, output = Path(config), Path(output)
    trained = all(supplied)
    paths = {key: Path(value) for key, value in paths.items()} if trained else {}
    safe_destination(output)
    if output.exists() and not resume or resume and not (output / "index.json").is_file():
        raise ValueError("Usa una salida nueva o un índice confirmado para recuperar")
    for protected in [config, *(p.parent for p in paths.values())]:
        outside_source(protected, output)
        outside_source(output, protected)
    settings, config_hash = read_manifest(config, 1024**2)
    cases = _cases(settings)
    prediction = (
        _trained_prediction(paths, cases[0][1].context)
        if trained
        else nullcontext(
            (
                lambda world: world,
                lambda x: 0.002 * x["news"][:, 0],
                ANALYTIC_PARENT,
                dict(mode="analytic"),
                None,
            )
        )
    )
    with prediction as selected:
        identity = dict(config_sha256=config_hash, prediction=selected[3], code=_code(trained))
        return _prepare(output, cases, identity, selected, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/episodes/scenarios.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    for name in ("parent", "ordered", "supervision", "encoded"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    values = [
        getattr(args, name) is not None for name in ("parent", "ordered", "supervision", "encoded")
    ]
    if any(values) and not all(values):
        parser.error(
            "--parent, --ordered, --supervision y --encoded forman un conjunto obligatorio"
        )
    result = prepare_scenarios(**vars(args))
    print(json.dumps(dict(scenarios=len(result["records"]), final_test_opened=False)))


if __name__ == "__main__":
    main()
