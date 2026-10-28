"""Comparar la conversión anterior con la acotada, manteniendo el mismo productor E2."""

import hashlib
import importlib.util
import json
import runpy
import statistics
import subprocess
import sys
import tempfile
import tracemalloc
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent
SOURCE = Path("src/mars_titan/memory/spaced_utility_readout.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    driver = runpy.run_path(str(BASE / "cost_probe.py"))
    current = driver["sample"].__globals__["apply_spaced_utility_readout"]
    patch = BASE / "rows-optimization.patch"
    with tempfile.TemporaryDirectory(prefix="mars-titan-e2-rows-") as temporary:
        target = Path(temporary) / SOURCE
        target.parent.mkdir(parents=True)
        target.write_bytes(SOURCE.read_bytes())
        subprocess.run(
            ["git", "apply", "--unidiff-zero", "--reverse", str(patch)],
            cwd=temporary,
            check=True,
        )
        reference_sha = sha(target)
        name = "mars_titan.memory._e2_rows_cost_reference"
        spec = importlib.util.spec_from_file_location(name, target)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[name] = reference
        spec.loader.exec_module(reference)
    functions = dict(reference=reference.apply_spaced_utility_readout, bounded=current)
    results = []
    for pass_id in (1, 2):
        for count in (64, 256, 1024):
            args = driver["fixture"](count)
            samples = dict(reference=[], bounded=[])
            for repetition in range(12):
                outputs = []
                order = (
                    ("reference", "bounded")
                    if (repetition + pass_id) % 2
                    else ("bounded", "reference")
                )
                for mode in order:
                    driver["sample"].__globals__["apply_spaced_utility_readout"] = functions[mode]
                    elapsed, rows, size = driver["sample"](*args, True)
                    outputs.append((rows, size))
                    if repetition >= 2:
                        samples[mode].append(elapsed)
                assert outputs[0] == outputs[1]
            peaks = {}
            for mode in functions:
                driver["sample"].__globals__["apply_spaced_utility_readout"] = functions[mode]
                tracemalloc.start()
                driver["sample"](*args, True)
                _, peaks[mode] = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            left, right = (statistics.median(samples[m]) for m in ("reference", "bounded"))
            results.append(
                dict(
                    pass_id=pass_id,
                    rows=count,
                    samples_ms=samples,
                    reference_median_ms=left,
                    bounded_median_ms=right,
                    reduction_pct=100 * (left - right) / left,
                    python_allocation_peak_bytes=peaks,
                    exact_rows_and_json_bytes=True,
                )
            )
    accepted = all(r["reduction_pct"] >= (5 if r["rows"] == 1024 else -5) for r in results)
    report = dict(
        kind="paired_fixture_rows_comparison",
        warmup_per_mode=2,
        repetitions_per_mode=10,
        device="cpu",
        cpu_threads=2,
        reference_source_sha256=reference_sha,
        current_source_sha256=sha(SOURCE),
        patch_sha256=sha(patch),
        probe_sha256=sha(Path(__file__)),
        driver_sha256=sha(BASE / "cost_probe.py"),
        accepted=accepted,
        criterion="Al menos 5% de reducción mediana con 1024 filas en ambas pasadas y ninguna "
        "regresión mediana mayor de 5% con 64 o 256 filas.",
        results=results,
        limits=[
            "Mismo proceso, referencias intercaladas y mismas filas completas.",
            "La referencia reconstruye solo la conversión anterior. "
            "El resto del productor coincide.",
            "Tracemalloc se mide aparte y excluye las asignaciones nativas.",
            "Otras aplicaciones y perfil de energía existentes, sin modificaciones.",
            "El criterio es de coste técnico, no de precisión financiera.",
        ],
    )
    Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(accepted=accepted, reductions=[r["reduction_pct"] for r in results])))


if __name__ == "__main__":
    main()
