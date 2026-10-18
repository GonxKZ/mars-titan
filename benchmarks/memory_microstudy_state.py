"""Reproducir las predicciones elegidas y observar el estado rápido sin reajustar pesos."""

import argparse
import json
import math
import resource
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.prediction_files import canonical_order, same_table
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.learning_traces.memory import MemoryObserver
from mars_titan.training import titans_walk_forward as window
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import ChronologicalInference
from mars_titan.training.learning_hold import require_learning_allowed
from mars_titan.training.walk_forward_phases import window_phases

from .memory_microstudy import CONFIG, jobs


def same_predictions(expected, actual):
    return same_table(
        expected.take(canonical_order(expected)), actual.take(canonical_order(actual))
    )


def forbid_optimizer_steps(optimizer, args, kwargs):
    raise RuntimeError("La reproducción de estado no admite pasos de optimizador")


class Scalars:
    def __init__(self):
        self.values = {}

    def scalar(self, metric, value, *, group="all"):
        number = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        if not math.isfinite(number):
            raise ValueError("El diagnóstico de memoria contiene un valor no finito")
        key = f"{group}/{metric}"
        if key in self.values:
            raise ValueError("La métrica de estado está duplicada")
        self.values[key] = number


def replay(work, job_id, output, *, frozen_parent=False):
    """Diagnóstico sobre el checkpoint seleccionado y el mismo calentamiento de evaluación."""
    require_learning_allowed("la reproducción observada del microensayo")
    job = next((job for job in jobs() if job["id"] == job_id), None)
    if job is None or job["kind"] not in ("titans", "cm_core"):
        raise ValueError("La observación de pesos rápidos necesita un núcleo declarado")
    if type(frozen_parent) is not bool or (frozen_parent and job_id != "mac_online-42"):
        raise ValueError("El control congelado pertenece al padre declarado del postentrenamiento")
    work, output = Path(work), Path(output)
    if output.exists():
        raise FileExistsError("El diagnóstico necesita un destino nuevo")
    prepared = json.loads((work / "prepared.json").read_text())
    if prepared["configuration_sha256"] != sha256(CONFIG):
        raise ValueError("La configuración difiere de la preparada")
    folder = work / "runs" / job_id
    report = json.loads((folder / "run.json").read_text())
    view = work / "input/view.json"
    if (
        report["status"] != "completed"
        or report["final_test_opened"] is not False
        or report["request"]["view_sha256"] != sha256(view)
    ):
        raise ValueError("El núcleo no procede de esta ventana completada")
    if not torch.cuda.is_available():
        raise RuntimeError("El diagnóstico necesita cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    with window.unfused_attention():
        identity, request = report["identity"], report["request"]
        recipe_path = (
            work / "recipes" / ("paper.json" if job.get("name") == "mac_paper" else "core.json")
        )
        recipe, document, options = window._recipe(
            recipe_path, identity["stopping_rule"], request["search_case"]
        )
        dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
        phases = window_phases(identity["window"], options["warmup_months"])
        sources = window._sources(dataset, phases, folder / "indices")
        model, _ = window._predictor(
            document,
            sources["train"].specification(),
            request["variant"],
            job["seed"],
            "cuda:0",
            request.get("local_control"),
            request.get("predictor_changes"),
        )
        fit = json.loads((folder / "fit/run.json").read_text())
        state = load_training_state(
            folder / "fit/checkpoints",
            expected_identity=fit["identity"],
            selection="best",
            expected_sha256=report["checkpoint"]["sha256"],
        )
        model.load_state_dict(state["model"])
        if frozen_parent:
            model.requires_grad_(False)
        before = model._parameter_digest()
        observer = MemoryObserver(dict(core=model))
        destination = window.PredictionRows(True)
        inference = ChronologicalInference(model, recipe)
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        observed_start = time.monotonic()
        rng = torch.cuda.get_rng_state(0).clone()
        cpu_rng = torch.get_rng_state().clone()
        from torch.optim.optimizer import register_optimizer_step_pre_hook

        guard = register_optimizer_step_pre_hook(forbid_optimizer_steps)
        observer.start()
        try:
            metrics = inference.predict(sources["evaluation"], destination)
        finally:
            observer.stop()
            guard.remove()
        torch.cuda.synchronize(0)
        observed_seconds = time.monotonic() - observed_start
        collected = Scalars()
        observer.record(collected)
        prediction = report["predictions"]["evaluation"]
        expected_path = folder / prediction["path"]
        if sha256(expected_path) != prediction["sha256"]:
            raise ValueError("Las predicciones originales han cambiado")
        expected = pq.read_table(expected_path)
        actual = pa.concat_tables(destination.finish())
        exact = same_predictions(expected, actual)
        if not frozen_parent and not exact:
            raise ValueError("La reproducción con observador cambia las predicciones originales")
        if (
            before != model._parameter_digest()
            or not torch.equal(rng, torch.cuda.get_rng_state(0))
            or not torch.equal(cpu_rng, torch.get_rng_state())
        ):
            raise ValueError("La reproducción ha cambiado pesos compartidos o RNG")
        result = dict(
            schema_version=1,
            job=job_id,
            optimizer_steps=0,
            exact_predictions=exact,
            shared_parameters_unchanged=True,
            cuda_rng_unchanged=True,
            cpu_rng_unchanged=True,
            checkpoint_sha256=report["checkpoint"]["sha256"],
            predictions_sha256=prediction["sha256"],
            measured_scope="evaluation_input_warmup_and_evaluation_with_selected_parameters",
            memory=collected.values,
            metrics=metrics,
            observed_seconds=observed_seconds,
            process_seconds=time.monotonic() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0),
            final_test_opened=False,
            script_sha256=sha256(Path(__file__)),
        )
        if frozen_parent:
            # Este es el control declarado de los adaptadores. Congelar parámetros puede
            # cambiar la ruta numérica de autograd aunque sus valores sean los mismos.
            predictions = output.with_suffix(".parquet")
            if predictions.exists():
                raise FileExistsError("Ya existen las predicciones del control congelado")
            output.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(actual, predictions)
            result.update(
                control="frozen_parent",
                predictions=dict(
                    path=predictions.name, sha256=sha256(predictions), rows=len(actual)
                ),
            )
        atomic_json(output, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-parent", action="store_true")
    args = parser.parse_args()
    result = replay(args.work, args.job, args.output, frozen_parent=args.frozen_parent)
    print(json.dumps(dict(job=args.job, exact_predictions=result["exact_predictions"])))


if __name__ == "__main__":
    main()
