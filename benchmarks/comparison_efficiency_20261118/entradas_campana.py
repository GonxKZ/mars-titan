"""Comprobar en CUDA el lector instalado, con los pesos guardados en el estado local."""

import argparse
import gc
import os
import time
from contextlib import nullcontext

import pyarrow as pa
import pyarrow.parquet as pq
import recorrido
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory.financial_observations import FinancialObservationSource
from mars_titan.training.corpus_inputs import CorpusDataset


def sources(year, model, seed):
    state = recorrido.STATE
    view = state / "cohorts" / str(year) / "manifest.json"
    saved = recorrido.read(state / "runs" / model / str(year) / f"seed-{seed}" / "run.json")
    if saved["status"] != "completed" or saved["request"]["view_sha256"] != sha256(view):
        raise ValueError("El ensayo necesita un trabajo completado en esta cohorte")
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    folder = state / "indices" / dataset.identity[:16]
    by_sha = {sha256(path): path for path in folder.glob("*/manifest.json")}
    return {
        phase: FinancialObservationSource(dataset, by_sha[saved["identity"]["indices"][phase]])
        for phase in ("train", "validation")
    }


def check_model(model, year, seed, *, long_slice=False):
    from mars_titan.memory import training_event_cache as cache
    from mars_titan.models.titans import host_finite

    raw = sources(year, model, seed)
    train = raw["train"]
    packed = dict(
        raw, train=cache.PackedTrainingSource(train.dataset, train.path, os.environ[cache.ENV])
    )
    spec = train.specification()
    interval = int(0.9 * len(train.metadata["groups"])), 144
    trials = []
    cases = [(False, True), (True, True)]
    segments = 16
    if long_slice:
        segments = 68
        interval = int(0.8 * len(train.metadata["groups"])), 576
        order = (True, False) if model == "titans_mac" else (False, True)
        cases = [(option, False) for option in order]
    elif model in ("titans_mac", "mars_titan_m2"):
        cases += [(True, False), (False, False), (False, False), (True, False)]
    for archive, audit in cases:
        selected = packed if archive else raw
        case_start = time.perf_counter()
        with host_finite.on_host() if archive and model == "titans_mac" else nullcontext():
            engine, report, folder = recorrido.build(model, selected, spec, seed, year=year)
            before = recorrido.parameter_digest(engine)
            cache.reset_usage()
            case = recorrido.profile_slice(
                engine,
                spec,
                "original",
                audit,
                source_range=interval,
                rng_seed=seed,
                until=segments,
            )
            case.update(
                packed=archive,
                usage=cache.receipt(),
                weights=before,
                build_and_pass_seconds=time.perf_counter() - case_start,
            )
            if recorrido.parameter_digest(engine) != before or (
                trials and trials[0]["weights"] != before
            ):
                raise ValueError("Los pesos del ensayo han cambiado")
            if archive and not case["usage"].get("packed_passes"):
                raise ValueError("Falta la copia verificada del tamaño que utiliza el trabajo")
            if archive and audit:
                if case["parity"] != trials[0]["parity"]:
                    raise ValueError("Han cambiado las emisiones, los contadores o los gradientes")
                if model in ("mars_titan_m2", "titans_mac"):
                    quantiles = engine.emits_levels if model == "titans_mac" else engine.quantiles
                    rows = recorrido.tw.PredictionRows(quantiles)
                    info = report["predictions"]["validation"]
                    file = folder / info["path"]
                    if sha256(file) != info["sha256"]:
                        raise ValueError("El Parquet de validación ha cambiado")
                    expected = pq.read_table(file)
                    start = time.perf_counter()
                    engine.evaluate(selected["validation"], rows=rows)
                    got = pa.concat_tables(rows.finish())
                    if not got.equals(expected):
                        raise ValueError("La validación difiere de las predicciones guardadas")
                    case["saved_validation"] = dict(
                        rows=got.num_rows,
                        seconds=time.perf_counter() - start,
                        sha256=info["sha256"],
                    )
            trials.append(case)
            print(
                model,
                "preparada" if archive else "origen",
                case["seconds"],
                case["samples"],
                flush=True,
            )
            del engine
        gc.collect()
        torch.cuda.empty_cache()
    if any(
        case[field] != trials[0][field]
        for case in trials
        for field in ("samples", "events", "labels")
    ):
        raise ValueError("Los dos lectores midieron distintos eventos, observaciones o etiquetas")
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year", type=int, default=2023, help="Año del origen con los cuatro trabajos completos"
    )
    parser.add_argument("--seed", type=int, default=43, help="Semilla de los trabajos guardados")
    parser.add_argument(
        "--long-slice",
        action="store_true",
        help="Medir también la apertura en un tramo de 68 segmentos de Titans-MAC y M2",
    )
    args = parser.parse_args()
    output = recorrido.OUT / ("long-slices.json" if args.long_slice else "qualification.json")
    if output.exists():
        raise ValueError("El directorio ya contiene una medición")
    recorrido.rw._cuda("cuda:0")
    torch.cuda.set_device("cuda:0")
    recorrido.apply_kernel_policy("fp32_strict")
    guard = recorrido.perf._forbid_steps()
    try:
        with recorrido.tw.unfused_attention():
            result = dict(
                year=args.year,
                seed=args.seed,
                torch=str(torch.__version__),
                gpu=torch.cuda.get_device_name(0),
                real_optimizer_steps=0,
                cases={},
            )
            models = ("mars_titan_m2", "titans_mac")
            if not args.long_slice:
                models += ("mars_titan_m1", "mars_titan_m3")
            for model in models:
                result["cases"][model] = check_model(
                    model, args.year, args.seed, long_slice=args.long_slice
                )
                atomic_json(output, result)
    finally:
        guard.remove()


if __name__ == "__main__":
    main()
