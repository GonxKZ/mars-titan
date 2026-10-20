"""Comparar la frontera validada con su repetición por token, sobre el predictor completo."""

import argparse
import json
import time
from pathlib import Path

import torch

from benchmarks.neural_plasticity_cost import (
    DECLARATION,
    ROOT,
    load_inputs,
    passage,
    subset,
    summary,
)
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.titans import neural_memory, plasticity
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.training.refinement_inference import _model_digest
from mars_titan.training.titans_walk_forward import unfused_attention


def measure(data, output, *, repetitions=30):
    if output.exists():
        raise FileExistsError("La comparación necesita un destino nuevo")
    if type(repetitions) is not int or not 2 <= repetitions <= 1000:
        raise ValueError("Las repeticiones deben estar entre 2 y 1000")
    if not torch.cuda.is_available():
        raise RuntimeError("La comparación requiere cuda:0")
    torch.cuda.set_device("cuda:0")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, spec, raws = load_inputs(data)
    declaration = json.loads(DECLARATION.read_text())
    recipe = json.loads((ROOT / declaration["base_recipe"]).read_text())
    base = {key: value for key, value in recipe["predictor"].items() if key != "dtype"}
    original = neural_memory.advance_plasticity
    variants = dict(repeated_validation=plasticity.advance, checked_boundary=original)
    cases = {}
    started = time.perf_counter()
    try:
        with unfused_attention():
            for size in (1, 128):
                batches = [
                    DecisionBatch.from_corpus(subset(raw, size), spec, device="cuda:0")
                    for raw in raws
                ]
                model = FinancialPredictor(
                    FinancialConfig(
                        spec,
                        variant="mac_online",
                        seed=42,
                        **(base | declaration["arms"]["cascade_combined"]["predictor_changes"]),
                    ),
                    device="cuda:0",
                ).train()
                digest = _model_digest(model)
                cpu_rng = torch.random.get_rng_state()
                gpu_rng = torch.cuda.get_rng_state()
                for graph in (False, True):
                    records = {name: [] for name in variants}
                    for repeat in range(repetitions + 3):
                        order = list(variants)
                        if repeat % 2:
                            order.reverse()
                        outputs = {}
                        for name in order:
                            neural_memory.advance_plasticity = variants[name]
                            pass_started = time.perf_counter()
                            record, outputs[name] = passage(model, batches, graph=graph)
                            record["complete_wall_seconds"] = time.perf_counter() - pass_started
                            if repeat >= 3:
                                records[name].append(record)
                        torch.testing.assert_close(*outputs.values(), rtol=0, atol=0)
                    cases[f"{size}/{'graph' if graph else 'inference'}"] = {
                        name: dict(
                            decision=summary(
                                [s for r in runs for s in r["decision_seconds"]], rows=size
                            ),
                            total_pass=summary(
                                [r["complete_wall_seconds"] for r in runs],
                                rows=size * len(batches),
                            ),
                            passes=runs,
                        )
                        for name, runs in records.items()
                    }
                if _model_digest(model) != digest:
                    raise RuntimeError("La comparación modificó parámetros compartidos")
                if not torch.equal(cpu_rng, torch.random.get_rng_state()) or not torch.equal(
                    gpu_rng, torch.cuda.get_rng_state()
                ):
                    raise RuntimeError("La comparación consumió aleatoriedad global")
                del model, batches
                torch.cuda.empty_cache()
    finally:
        neural_memory.advance_plasticity = original
    receipt = dict(
        schema_version=1,
        kind="neural_plasticity_boundary_cost",
        cases=cases,
        code_sha256={
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                Path(__file__),
                Path(plasticity.__file__),
                Path(neural_memory.__file__),
                ROOT / "benchmarks/neural_plasticity_cost.py",
            )
        },
        data_sha256=sha256(data),
        declaration_sha256=sha256(DECLARATION),
        repetitions=repetitions,
        warmup=3,
        device="cuda:0",
        dtype="float32",
        tf32=False,
        process_seconds=time.perf_counter() - started,
        order="Alternancia por pareja, invertida en cada repetición",
        parameters_unchanged=True,
        rng_unchanged=True,
        bitwise_predictions=True,
        scope=(
            "Recorrido completo de passage con inicialización, validación, gradientes, "
            "copia de salidas a CPU y limpieza. Entradas precodificadas residentes. "
            "Sin optimizador."
        ),
        scientific_evaluation=False,
    )
    atomic_json(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    measure(args.data, args.output, repetitions=args.repetitions)


if __name__ == "__main__":
    main()
