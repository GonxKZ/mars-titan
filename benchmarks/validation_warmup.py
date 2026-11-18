"""Repetir la validación de un lector terminado con sus pesos y filas reales."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_observations import FinancialObservationSource
from mars_titan.training import mars_titan_walk_forward as window
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.kernel_policy import apply_kernel_policy
from mars_titan.training.mars_titan_run import ReadoutTrainer
from mars_titan.training.titans_walk_forward import PredictionRows, unfused_attention
from mars_titan.training.validation_warmup import SWITCH


def measure(state, model, year, seed, output):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    window._cuda("cuda:0")
    torch.cuda.set_device("cuda:0")
    apply_kernel_policy("fp32_strict")
    if not 2000 <= year <= 2023:
        raise ValueError("El ensayo requiere una ventana histórica de desarrollo")
    view = state / "cohorts" / str(year) / "manifest.json"
    anchor = state / "runs" / model / str(year) / f"seed-{seed}"
    report = json.loads((anchor / "run.json").read_text())
    if report["status"] != "completed" or report["request"]["view_sha256"] != sha256(view):
        raise ValueError("El lector no ha terminado en la vista solicitada")
    req, ident = report["request"], report["identity"]
    parent = Path(req["parent"]["path"])
    parent_report, parent_sha = window._parent(parent, view, seed)
    if parent_sha != req["parent"]["run_sha256"]:
        raise ValueError("El informe del padre ha cambiado")
    d = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    root = state / "indices" / d.identity[:16]

    def source(partition):
        matches = [
            p
            for p in root.glob(f"{partition}-*/manifest.json")
            if sha256(p) == ident["indices"][partition]
        ]
        if len(matches) != 1:
            raise ValueError(f"Falta el índice verificado de {partition}")
        return FinancialObservationSource(d, matches[0])

    train, validation = source("train"), source("validation")
    spec = train.specification()
    predictor = window._frozen_parent(
        parent_report["identity"]["recipe"],
        spec,
        seed,
        "cuda:0",
        parent,
        parent_report,
        allow_context=True,
    )
    fam = window._mars_family(req["components"])
    variant = fam.variant(predictor, parent_report)
    codec = FrozenEpisodeCodec(spec)
    recipe = window.case_recipe(ident["recipe"], req["search_case"])
    fit_path = anchor / report["fit"]["path"]
    if sha256(fit_path) != report["fit"]["sha256"]:
        raise ValueError("Ha cambiado el informe de ajuste del lector")
    fit = json.loads(fit_path.read_text())
    saved = load_training_state(
        anchor / "fit/checkpoints",
        expected_identity=fit["identity"],
        selection="best",
        expected_sha256=fit["best_checkpoint"]["sha256"],
    )
    extra = {"scalers": window._anchor_scalers(fit)} if variant.admission == "m3" else {}
    engines = {}
    for mode in ("0", "1"):
        os.environ[SWITCH] = mode
        readout = window._readout(variant, codec, predictor, recipe, seed)
        engine = ReadoutTrainer(
            predictor,
            readout,
            recipe,
            admission=variant.admission,
            retention=fam.retention(recipe, variant, **extra),
            native=window._native(variant.admission),
            codec=codec,
            train=train,
            validation=validation,
            output=output.parent / "unused-fit",
            world=fam.world,
            fold=ident["window"]["id"],
        )
        engine._load_readout(saved)
        engines[mode] = engine
    ref_path = anchor / report["predictions"]["validation"]["path"]
    if sha256(ref_path) != report["predictions"]["validation"]["sha256"]:
        raise ValueError("Han cambiado las predicciones de referencia")
    expected = pq.read_table(ref_path)
    result = dict(
        model=model,
        year=year,
        seed=seed,
        gpu=torch.cuda.get_device_name(0),
        torch=str(torch.__version__),
        validation_rows=expected.num_rows,
        validation_sha256=sha256(ref_path),
        optimizer_steps=0,
        blocks=recipe.block_rows,
        preparation_seconds=time.perf_counter() - started,
        passes=[],
    )
    reference_metrics = None
    for n, mode in enumerate(("0", "1", "1", "0", "1", "0")):
        engine = engines[mode]
        rows = PredictionRows(engine.quantiles)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        metrics = engine.evaluate(validation, rows=rows)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        table = pa.concat_tables(rows.finish())
        if not table.equals(expected) or (
            reference_metrics is not None and metrics != reference_metrics
        ):
            raise ValueError("La validación no reproduce todas las filas, cuantiles y métricas")
        reference_metrics = metrics
        record = dict(
            pass_index=n,
            resident=bool(int(mode)),
            seconds=seconds,
            scored_rows_per_second=expected.num_rows / seconds,
            allocated_bytes=torch.cuda.max_memory_allocated(),
            reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cache=engine.validation_warmup.receipt() if mode == "1" else None,
        )
        result["passes"].append(record)
        result["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(output, result)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["mars_titan_m0", "mars_titan_m1", "mars_titan_m2", "mars_titan_m3"],
        required=True,
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with unfused_attention():
        measure(args.state, args.model, args.year, args.seed, args.output)
