"""Comparar seis padres completos sobre datos reales y dos aumentos emparejados."""

import argparse
import fcntl
import gc
import os
from contextlib import nullcontext
from pathlib import Path

import torch

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.embeddings import FrozenEncoders
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments.actions import ActionGrid
from mars_titan.environments.corpus_source import ParquetCohortSource, prepare_causal_corpus
from mars_titan.episodes.augmentation import fit_volatility
from mars_titan.episodes.parents import ParentCache
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.experiment_resources import GpuLease
from mars_titan.training.klpo_queue import selected_parents
from mars_titan.training.predictive_parents import _verified_file
from mars_titan.training.run_receipts import initialize_receipt

from .inputs import CONDITIONS, PairedInputs, fingerprint, fit_normalization
from .parents import NEURAL, load_parent
from .preparation import EpisodeFactory, encoder_contract, prepare_augmentation
from .run import MODES, code_identity, run_case, validate_case
from .selection import selection_policy


def read_design(path):
    plan, digest = read_manifest(Path(path))
    real_only = plan.get("conditions") == ["real"] and "selection" in plan
    options = {
        "epochs",
        "learning_rate",
        "weight_decay",
        "clip_norm",
        "beta",
        "behavior_epsilon",
        "auxiliary_samples",
    }
    if "selection" in plan:
        options.add("selection")
    keys = {
        "schema_version",
        "conditions",
        "modes",
        "neural_controls",
        "seeds",
        "fraction",
        "decisions",
        "warmup",
        "batch_size",
        "checkpoint_seconds",
        "final_test_opened",
    } | options
    if real_only:
        keys -= {"fraction", "decisions", "warmup"}
    if (
        set(plan) != keys
        or plan["schema_version"] != 1
        or (not real_only and plan["conditions"] != list(CONDITIONS))
        or plan["modes"] != list(MODES[:6])
        or plan["neural_controls"] != list(MODES[6:])
        or (not real_only and plan["fraction"] != 0.25)
        or plan["final_test_opened"] is not False
        or not isinstance(plan["seeds"], list)
        or not 1 <= len(plan["seeds"]) <= 10
        or any(type(s) is not int or not 0 <= s < 2**32 for s in plan["seeds"])
        or len(set(plan["seeds"])) != len(plan["seeds"])
        or (
            not real_only
            and any(
                type(plan[k]) is not int or not 1 <= plan[k] <= 128 for k in ("decisions", "warmup")
            )
        )
        or type(plan["batch_size"]) is not int
        or not 1 <= plan["batch_size"] <= 4096
        or type(plan["checkpoint_seconds"]) not in (int, float)
        or not 0 < plan["checkpoint_seconds"] <= 900
    ):
        raise ValueError("El diseño no conserva los controles, condiciones y presupuestos fijados")

    def cases(kind):
        if kind not in {*NEURAL, "ridge", "xgboost"}:
            raise ValueError("La familia no pertenece a los seis padres de la comparación")
        modes = plan["modes"] + (plan["neural_controls"] if kind in NEURAL else [])
        result = []
        for seed in plan["seeds"]:
            for condition in plan["conditions"]:
                for mode in modes:
                    case = {key: plan[key] for key in options} | dict(
                        mode=mode, condition=condition, seed=seed
                    )
                    validate_case(case)
                    result.append(dict(id=f"seed-{seed}/{condition}/{mode}", case=case))
        return result

    cases("gru")
    return plan, cases, digest


def _queue_code():
    return code_identity() | {
        f"posttraining/{name}": sha256(Path(__file__).with_name(name))
        for name in ("queue.py", "preparation.py")
    }


def _completed(output, record):
    path, _ = _verified_file(output, record, maximum_bytes=8 * 1024**2)
    result = read_manifest(path, 8 * 1024**2)[0]
    if result["status"] != "completed" or result["final_test_opened"] is not False:
        raise ValueError("Un resultado confirmado ya no está completo")
    for item in (result["checkpoint"], result["predictions"]["validation"]):
        _verified_file(path.parent, item)
    return result


def _confirm_artifact(summary, root, path):
    name, digest = str(path.relative_to(root)), sha256(path)
    artifacts = summary.setdefault("artifacts", {})
    if name in artifacts and artifacts[name] != digest:
        raise ValueError("Un artefacto confirmado de la cola ha cambiado")
    artifacts[name] = digest
    atomic_json(root / "summary.json", summary)


def _prepare_augmentations(plan, train, output, binding, summary, stop, lease):
    if plan["conditions"] == ["real"]:
        return {}
    calibration_path = output / "calibration.json"
    if calibration_path.exists():
        calibration = read_manifest(calibration_path)[0]
    else:
        calibration = fit_volatility(train)
        atomic_json(calibration_path, calibration)
    _confirm_artifact(summary, output, calibration_path)
    if (
        calibration.get("source_sha256") != train.manifest_sha256
        or calibration.get("fit_partition") != "train"
    ):
        raise ValueError("La calibración no corresponde a las etiquetas de entrenamiento")
    encoders, augmentations = FrozenEncoders(), {}
    for seed in plan["seeds"]:
        folder = output / "augmentation" / f"seed-{seed}"
        augmentations[seed] = prepare_augmentation(
            train,
            folder,
            encoders,
            expected_spec=binding["encoders"],
            seed=seed,
            decisions=plan["decisions"],
            warmup=plan["warmup"],
            volatility=calibration["volatility"],
            stop=stop,
            check_resources=lease.check,
        )
        _confirm_artifact(summary, output, folder / "augmentation.json")
    del encoders
    gc.collect()
    torch.cuda.empty_cache()
    return augmentations


def run_queue(config, reference, tabular, encoded, output, *, arm="US", stop=None):
    """Ejecutar secuencialmente la cola o recuperarla, con una única concesión de GPU."""
    config, reference, tabular, encoded, output = map(
        Path, (config, reference, tabular, encoded, output)
    )
    plan, cases, config_hash = read_design(config)
    policy = selection_policy(cases("gru")[0]["case"])
    stop = stop or StopRequest()
    # La admisión precede a pesos, fuentes grandes y preparación de codificadores.
    with GpuLease() as lease:
        torch.set_num_threads(4)
        proof = selected_parents(reference, tabular, arm)
        binding = encoder_contract(Path(proof["manifest"]), encoded)
        if binding["supervision_sha256"] != proof["manifest_sha256"]:
            raise ValueError("La supervisión ha cambiado tras verificar los padres")
        safe_destination(output)
        for protected in (
            reference.parent,
            tabular.parent,
            encoded.parent,
            Path(proof["manifest"]).parent,
            config,
        ):
            outside_source(protected, output)
            outside_source(output, protected)
        identity = dict(
            proof=proof,
            config_sha256=config_hash,
            binding=binding,
            code=_queue_code(),
            selection_policy=policy,
        )
        output.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(output / ".queue.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            confirmed = initialize_receipt(
                output, identity, record="summary.json", lock=".queue.lock"
            )
            summary = (
                read_manifest(output / "summary.json", 8 * 1024**2)[0]
                if confirmed
                else dict(
                    schema_version=1,
                    kind="real_continuations_queue"
                    if plan["conditions"] == ["real"]
                    else "paired_posttraining_queue",
                    selection_policy=policy,
                    identity=identity,
                    status="running",
                    runs={},
                    planned_runs=sum(len(cases(k)) for k in proof["parents"]),
                    completed_runs=0,
                    final_test_opened=False,
                )
            )
            atomic_json(output / "summary.json", summary)
            try:
                ordered = output / "ordered"
                prepared = prepare_causal_corpus(
                    Path(proof["manifest"]),
                    ordered,
                    batch_size=plan["batch_size"],
                    resume=ordered.exists(),
                    stop=stop,
                )
                if prepared["status"] != "completed":
                    raise InterruptedError
                ordered_path = ordered / "manifest.json"
                _confirm_artifact(summary, output, ordered_path)
                if prepared["source_sha256"] != binding["supervision_sha256"]:
                    raise ValueError("Los datos ordenados no corresponden a la supervisión")
                with (
                    ParquetCohortSource(ordered_path, partition="train") as train,
                    ParquetCohortSource(ordered_path, partition="validation") as validation,
                ):
                    grid = ActionGrid.from_dict(prepared["grid"])
                    augmentations = _prepare_augmentations(
                        plan, train, output, binding, summary, stop, lease
                    )
                    for kind, parent_record in proof["parents"].items():
                        if stop.requested:
                            raise InterruptedError
                        path = Path(parent_record["report"])
                        if sha256(path) != parent_record["sha256"]:
                            raise ValueError("El informe del padre ha cambiado")
                        parent = load_parent(ordered_path, path, lease=lease)
                        folder = output / "parents" / kind
                        folder.mkdir(parents=True, exist_ok=True)
                        with ParentCache(
                            folder / "predictions.sqlite",
                            parent.identity["checkpoint_sha256"],
                            fingerprint(binding),
                            parent.predict,
                        ) as cache:
                            norm_path = folder / "normalization.json"
                            if norm_path.exists():
                                normalization = read_manifest(norm_path, 4 * 1024**2)[0]
                            else:
                                data = PairedInputs(train, validation, cache)
                                normalization = fit_normalization(
                                    data, batch_size=plan["batch_size"], stop=stop
                                )
                                atomic_json(norm_path, normalization)
                            _confirm_artifact(summary, output, norm_path)
                            for seed in plan["seeds"]:
                                extra_folder = output / "augmentation" / f"seed-{seed}"
                                context = (
                                    EpisodeFactory(extra_folder, augmentations[seed])
                                    if augmentations
                                    else nullcontext()
                                )
                                with context as extras:
                                    data = PairedInputs(
                                        train,
                                        validation,
                                        cache,
                                        windows=extras.windows if extras else (),
                                        synthetic=extras,
                                        synthetic_identity=extras.identity if extras else None,
                                    )
                                    for item in (
                                        r for r in cases(kind) if r["case"]["seed"] == seed
                                    ):
                                        if stop.requested:
                                            raise InterruptedError
                                        if (
                                            sha256(config) != config_hash
                                            or _queue_code() != identity["code"]
                                        ):
                                            raise ValueError(
                                                "El diseño o el código de la cola ha cambiado"
                                            )
                                        identifier = f"{kind}/{item['id']}"
                                        destination = folder / "runs" / item["id"]
                                        previous = summary["runs"].get(identifier)
                                        if (
                                            previous is not None
                                            and previous["status"] == "completed"
                                        ):
                                            result = _completed(output, previous)
                                            expected = dict(
                                                case=item["case"],
                                                dataset=data.identity,
                                                parent=parent.identity,
                                                normalization=normalization,
                                                batch_size=plan["batch_size"],
                                                code=code_identity(),
                                            )
                                            if any(
                                                fingerprint(result["identity"].get(k))
                                                != fingerprint(v)
                                                for k, v in expected.items()
                                            ):
                                                raise ValueError(
                                                    "El caso confirmado pertenece a otra identidad"
                                                )
                                            continue
                                        result = run_case(
                                            data,
                                            destination,
                                            item["case"],
                                            grid,
                                            normalization,
                                            parent=parent,
                                            lease=lease,
                                            batch_size=plan["batch_size"],
                                            checkpoint_seconds=plan["checkpoint_seconds"],
                                            resume=destination.exists(),
                                            stop=stop,
                                        )
                                        summary["runs"][identifier] = dict(
                                            status=result["status"],
                                            path=str(
                                                (destination / "run.json").relative_to(output)
                                            ),
                                            sha256=sha256(destination / "run.json"),
                                        )
                                        summary["completed_runs"] = sum(
                                            r["status"] == "completed"
                                            for r in summary["runs"].values()
                                        )
                                        atomic_json(output / "summary.json", summary)
                                        if result["status"] != "completed":
                                            raise InterruptedError
                        del parent
                        gc.collect()
                        torch.cuda.empty_cache()
                if (
                    summary["completed_runs"] != summary["planned_runs"]
                    or _queue_code() != identity["code"]
                ):
                    raise ValueError("La cola no conserva todos los casos y su identidad")
                summary["status"] = "completed"
            except InterruptedError:
                summary["status"] = "paused"
            except BaseException as error:
                summary.update(
                    status="failed",
                    last_failure=dict(type=type(error).__name__, message=str(error)),
                )
                raise
            finally:
                atomic_json(output / "summary.json", summary)
            return summary
        finally:
            os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "reference", "tabular", "encoded", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arm", choices=("US", "CN", "US+CN"), default="US")
    args = vars(parser.parse_args())
    with StopRequest() as stop:
        result = run_queue(**args, stop=stop)
    print(
        f"Estado: {result['status']}. "
        f"Ajustes terminados: {result['completed_runs']}/{result['planned_runs']}"
    )


if __name__ == "__main__":
    main()
