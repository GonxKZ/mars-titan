"""Referencia del padre con la misma ruta congelada utilizada por sus lectores."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training import mars_titan_walk_forward as mars
from mars_titan.training import titans_walk_forward as window
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import ChronologicalInference
from mars_titan.training.learning_hold import require_learning_allowed
from mars_titan.training.walk_forward_phases import window_phases

from .memory_microstudy_state import forbid_optimizer_steps


def frozen_reference(view, parent, output, *, seed, device="cuda:0", indices=None):
    require_learning_allowed("la referencia congelada del estudio combinado")
    if str(device).startswith("cuda") and (
        str(device) != "cuda:0" or not torch.cuda.is_available()
    ):
        raise RuntimeError("La referencia requiere cuda:0 disponible")
    view, parent, output = Path(view), Path(parent), Path(output)
    if output.exists():
        raise FileExistsError("La referencia necesita un destino nuevo")
    report, parent_sha = mars._parent(parent, view, seed)
    document = report["identity"]["recipe"]
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    phases = window_phases(
        report["identity"]["window"], window.walk_forward_options(document)["warmup_months"]
    )
    sources = window._sources(
        dataset, phases, Path(indices) if indices else output.parent / "indices"
    )
    model = mars._frozen_parent(
        document, sources["train"].specification(), seed, device, parent, report, allow_context=True
    )
    recipe = window.case_recipe(document, report["request"].get("search_case"))
    before = model._parameter_digest()
    rows = window.PredictionRows(True)
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    guard = register_optimizer_step_pre_hook(forbid_optimizer_steps)
    try:
        metrics = ChronologicalInference(model, recipe).predict(sources["evaluation"], rows)
    finally:
        guard.remove()
    table = pa.concat_tables(window.checked_tables(rows, metrics, dataset, "evaluation"))
    if model._parameter_digest() != before:
        raise ValueError("La referencia cambió parámetros compartidos")
    output.mkdir(parents=True)
    prediction = output / "evaluation-predictions.parquet"
    pq.write_table(table, prediction)
    result = dict(
        status="completed",
        kind="frozen_runtime_parent",
        optimizer_steps=0,
        checkpoint_sha256=report["checkpoint"]["sha256"],
        parent_sha256=parent_sha,
        shared_parameters_unchanged=True,
        final_test_opened=False,
        predictions=dict(
            evaluation=dict(path=prediction.name, sha256=sha256(prediction), rows=len(table))
        ),
        metrics=metrics,
    )
    atomic_json(output / "run.json", result)
    return result
