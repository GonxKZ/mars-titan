"""Diagnóstico de trayectorias al congelar parámetros, sin ajustar pesos."""

import argparse
import json
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.titans.config import ATTENTION_LAYOUTS
from mars_titan.models.titans.frozen_financial import _implementation, _numerics
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.training import mars_titan_walk_forward as mars
from mars_titan.training import titans_walk_forward as window
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import ChronologicalInference
from mars_titan.training.learning_hold import require_learning_allowed
from mars_titan.training.walk_forward_phases import window_phases

from .combined_memory_study import verify_prepared
from .memory_microstudy_state import forbid_optimizer_steps

ROUTES = (
    "fp32_tracked",
    "fp32_frozen",
    "fp32_local",
    "fp32_flat_tracked",
    "fp32_flat_frozen",
    "fp64_tracked",
    "fp64_frozen",
)


@contextmanager
def projection_local_graph(model):
    """Referencia secuencial: proyectar sin grafo dentro del gradiente interno de evaluación.

    No es una política de entrenamiento. La derivada interna sigue calculándose respecto
    a las copias de los pesos rápidos. No se cambian valores ni flags de los parámetros.
    """
    if torch.is_grad_enabled() or model.training:
        raise ValueError("El diagnóstico requiere un ámbito exterior no_grad")
    handles, stack = [], []

    def before(module, inputs):
        context = torch.no_grad()
        context.__enter__()
        stack.append(context)

    def after(module, inputs, output):
        stack.pop().__exit__(None, None, None)
        return output

    try:
        for memory in model.modules():
            if not isinstance(memory, NeuralMemory):
                continue
            for name in (
                "key_projection",
                "value_projection",
                "alpha_projection",
                "eta_projection",
                "theta_projection",
            ):
                projection = getattr(memory, name)
                handles.append(projection.register_forward_pre_hook(before))
                handles.append(projection.register_forward_hook(after, always_call=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        while stack:
            stack.pop().__exit__(None, None, None)


def replay(work, parent_id, route, output, *, attention_layout=None):
    require_learning_allowed("el diagnóstico numérico acotado")
    if route not in ROUTES or parent_id not in ("fold-018/cell0-42", "fold-018/cell7-42"):
        raise ValueError("El diagnóstico solo admite las referencias declaradas")
    if attention_layout is not None and attention_layout not in ATTENTION_LAYOUTS:
        raise ValueError("attention_layout no pertenece a las disposiciones admitidas")
    linear_setting = os.environ.get("TORCH_LINEAR_FLATTEN_3D")
    if ("_flat_" in route and linear_setting != "1") or (
        "_flat_" not in route and linear_setting not in (None, "0")
    ):
        raise ValueError("TORCH_LINEAR_FLATTEN_3D debe fijarse antes de iniciar el proceso")
    work, output = Path(work), Path(output)
    verify_prepared(work)
    if output.exists():
        raise FileExistsError("El diagnóstico necesita un destino nuevo")
    if not torch.cuda.is_available():
        raise RuntimeError("El diagnóstico necesita cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    parent, view = work / "runs" / parent_id, work / "fold-018/view.json"
    started = time.monotonic()
    with window.unfused_attention():
        report, digest = mars._parent(parent, view, 42)
        document = report["identity"]["recipe"]
        dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
        phases = window_phases(
            report["identity"]["window"], window.walk_forward_options(document)["warmup_months"]
        )
        sources = window._sources(dataset, phases, work / "indices/fold-018")
        model = mars._frozen_parent(
            document, sources["train"].specification(), 42, "cuda:0", parent, report
        )
        if attention_layout is not None:
            # Intervención explícita sobre un checkpoint ya cargado. Se conserva su valor,
            # se identifica la disposición nueva y el recorrido crea estados rápidos nuevos.
            model.config = replace(model.config, attention_layout=attention_layout)
            model.mac.config = replace(model.mac.config, attention_layout=attention_layout)
            model._seal_parameters()
        model.requires_grad_(not route.endswith("_frozen"))
        recipe = window.case_recipe(document, report["request"].get("search_case"))
        if route.startswith("fp64"):
            # Los valores FP32 del checkpoint se representan exactamente al ampliar el tipo.
            model.double()
            recipe = replace(recipe, precision=None)
        before = model._parameter_digest()
        rows = window.PredictionRows(True)
        inference = ChronologicalInference(model, recipe)
        from torch.optim.optimizer import register_optimizer_step_pre_hook

        guard = register_optimizer_step_pre_hook(forbid_optimizer_steps)
        torch.cuda.reset_peak_memory_stats(0)
        try:
            with torch.no_grad():
                with projection_local_graph(model) if route == "fp32_local" else nullcontext():
                    # Se usa el mismo recorrido. Los objetivos solo se puntúan en el motor,
                    # no entran en la memoria, ni se publican ni seleccionan estas rutas.
                    metrics = inference.predict(sources["evaluation"], rows)
        finally:
            guard.remove()
        tables = window.checked_tables(rows, metrics, dataset, "evaluation")
        import pyarrow as pa

        table = pa.concat_tables(tables).select(["sample_id", "prediction"]).sort_by("sample_id")
        if before != model._parameter_digest() or any(
            p.grad is not None for p in model.parameters()
        ):
            raise ValueError("El diagnóstico cambió pesos o acumuló gradientes compartidos")
        output.mkdir(parents=True)
        path = output / "predictions.parquet"
        pq.write_table(table, path)
        expected = parent / report["predictions"]["evaluation"]["path"]
        original = pq.read_table(expected, columns=["sample_id", "prediction"]).sort_by("sample_id")
        if not table["sample_id"].equals(original["sample_id"]):
            raise ValueError("El diagnóstico no conserva las filas originales")
        delta = np.abs(table["prediction"].to_numpy() - original["prediction"].to_numpy())
        record = dict(
            parent=parent_id,
            route=route,
            status="completed",
            optimizer_steps=0,
            shared_parameters_unchanged=True,
            shared_gradients_none=True,
            parent_sha256=digest,
            checkpoint_sha256=report["checkpoint"]["sha256"],
            predictions=dict(path=path.name, sha256=sha256(path), rows=len(table)),
            maximum_difference_to_original=float(delta.max()),
            mean_difference_to_original=float(delta.mean()),
            elapsed_seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            numerics=_numerics(),
            compute_dtype=str(model.head.weight.dtype),
            linear_flatten_3d=linear_setting,
            implementation=_implementation(),
            script_sha256=sha256(Path(__file__)),
            target_metrics_included=False,
            final_test_opened=False,
        )
        if attention_layout is not None:
            record["diagnostic_attention_layout"] = attention_layout
        atomic_json(output / "run.json", record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-layout", choices=ATTENTION_LAYOUTS)
    args = parser.parse_args()
    options = {} if args.attention_layout is None else dict(attention_layout=args.attention_layout)
    print(
        json.dumps(
            replay(args.work, args.parent, args.route, args.output, **options), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
