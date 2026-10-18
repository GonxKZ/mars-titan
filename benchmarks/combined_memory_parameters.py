"""Inventariar parámetros y bytes guardados sin ejecutar los predictores."""

import argparse
import json
from pathlib import Path

import torch

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.checkpoints import load_training_state

from .combined_memory_study import verify_prepared


def parameter_counts(state, identity):
    """Contar los papeles del optimizador, congelados e inertes del estado seleccionado.

    Los tensores no clasificados se informan aparte. No se convierten en parámetros
    por aparecer en state_dict. Los pesos rápidos por flujo y el banco no viven aquí.
    """
    roles = identity["parameter_roles"]
    trained = [name for names in roles.values() for name in names]
    frozen = identity.get("frozen_parameters", [])
    inert = identity.get("inert_parameters", [])
    names = trained + frozen + inert
    if len(names) != len(set(names)) or any(
        not isinstance(state.get(name), torch.Tensor) for name in names
    ):
        raise ValueError("Los parámetros declarados faltan, se solapan o no son tensores")

    def count(selected):
        return dict(
            elements=sum(state[name].numel() for name in selected),
            bytes=sum(state[name].numel() * state[name].element_size() for name in selected),
        )

    totals, optimized = count(names), count(trained)
    remaining = [
        name
        for name, value in state.items()
        if isinstance(value, torch.Tensor) and name not in names
    ]
    return dict(
        registered_parameter_elements=totals["elements"],
        registered_parameter_bytes=totals["bytes"],
        stage_trainable_elements=optimized["elements"],
        stage_trainable_bytes=optimized["bytes"],
        stage_trainable_scope="parameters_listed_for_the_stage_optimizer",
        roles={name: count(values) for name, values in roles.items()},
        frozen=count(frozen),
        inert=count(inert),
        unclassified_tensor_elements=count(remaining)["elements"],
        unclassified_tensor_bytes=count(remaining)["bytes"],
        unclassified_tensor_names=remaining,
    )


def collect(work, output):
    work, output = Path(work), Path(output)
    if output.exists():
        raise FileExistsError("El inventario necesita un destino nuevo")
    receipt = verify_prepared(work)
    models, pending = {}, []
    for job in receipt["jobs"]:
        folder = work / "runs" / job["id"]
        path = folder / ("window.json" if job["kind"] == "candidate" else "run.json")
        run = json.loads(path.read_text()) if path.exists() else {}
        if run.get("status") != "completed":
            pending.append(job["id"])
            continue
        if run.get("final_test_opened") is not False:
            raise ValueError("La ejecución no conserva la reserva final")
        if job["kind"] == "reference":
            parent = models[job["parent"]]
            models[job["id"]] = dict(
                parent=job["parent"],
                stage_trainable_elements=0,
                inference_registered_parameter_elements=parent[
                    "inference_registered_parameter_elements"
                ],
                inference_registered_parameter_bytes=parent["inference_registered_parameter_bytes"],
            )
            continue
        fit = json.loads((folder / "fit/run.json").read_text())
        checkpoint = fit["best_checkpoint"]["sha256"]
        state = load_training_state(
            folder / "fit/checkpoints",
            expected_identity=fit["identity"],
            selection="best",
            expected_sha256=checkpoint,
        )
        record = parameter_counts(state["model"], fit["identity"])
        record["checkpoint_sha256"] = checkpoint
        for unit in ("elements", "bytes"):
            count = record[f"registered_parameter_{unit}"]
            if job["kind"] in ("readout", "cm_readout"):
                count += models[job["parent"]][f"inference_registered_parameter_{unit}"]
            record[f"inference_registered_parameter_{unit}"] = count
        models[job["id"]] = record
        del state
    result = dict(
        kind="selected_parameter_inventory",
        prepared_sha256=sha256(work / "prepared.json"),
        source_sha256=sha256(Path(__file__)),
        models=models,
        pending=pending,
        scope="shared_model_parameters_and_stage_optimizer_roles",
        fast_states_and_episodic_banks_included=False,
    )
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.work, args.output)
    print(json.dumps(dict(models=len(result["models"]), pending=len(result["pending"]))))


if __name__ == "__main__":
    main()
