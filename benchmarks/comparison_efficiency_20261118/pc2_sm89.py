"""Contrastar en la RTX 4070 las rutas de PC2 con los checkpoints de la misma cohorte."""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from collections import Counter
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory import training_event_cache
from mars_titan.models.titans import financial, host_finite

REFERENCE = "f52cc3c332ba358e7afc1772e2038f05505c2285"
PC2 = "559bfa0f849b6f07241b5296e4c44d1f47c4b2c4"


def source_tree(runtime, mode, *, original_preparation=False):
    """Cotejar los módulos Python del ensayo con los blobs de la referencia o su combinación."""
    repo = Path(__file__).resolve().parents[2]
    git = ["git", "-C", str(repo)]
    tree = (
        subprocess.check_output([*git, "rev-parse", REFERENCE + "^{tree}"], text=True).strip()
        if mode == "reference"
        else subprocess.check_output(
            [*git, "merge-tree", "--write-tree", REFERENCE, PC2], text=True
        ).strip()
    )
    inventory = subprocess.check_output([*git, "ls-tree", "-rz", tree, "--", "src/mars_titan"])
    expected = {}
    for row in inventory.split(b"\0"):
        if row.endswith(b".py"):
            header, name = row.split(b"\t", 1)
            expected[os.fsdecode(name)] = header.split()[2].decode()
    restored = []
    if original_preparation:
        if mode in ("reference", "portable"):
            raise ValueError("El control con preparación original necesita un kernel en prueba")
        restored = [
            "src/mars_titan/models/learning_scale.py",
            "src/mars_titan/models/titans/financial.py",
        ]
        for name in restored:
            expected[name] = subprocess.check_output(
                [*git, "rev-parse", f"{REFERENCE}:{name}"], text=True
            ).strip()
    for name, blob in expected.items():
        content = (runtime / name).read_bytes()
        # Git identifica cada blob por su cabecera, su tamaño decimal y sus bytes.
        actual = hashlib.sha1(
            f"blob {len(content)}\0".encode() + content, usedforsecurity=False
        ).hexdigest()
        if blob != actual:
            raise ValueError(f"El contenido de {name} no coincide con el árbol del ensayo")
    kernel = "src/mars_titan/models/titans/cuda_kernels.py"
    if not expected or (runtime / kernel).is_file() != (kernel in expected):
        raise ValueError("La copia numérica no corresponde al modo solicitado")
    return dict(
        tree=tree,
        python_modules=len(expected),
        reference=REFERENCE,
        pc2=PC2,
        restored_from_reference=restored,
    )


class LaunchCounts:
    """Contar lanzamientos efectivos solo en la pasada de paridad."""

    def __init__(self, program, calls):
        self.program, self.calls = program, calls

    def __getitem__(self, grid):
        launch = self.program[grid]

        def run(*args, **kwargs):
            self.calls[tuple(args[0].shape)] += 1
            return launch(*args, **kwargs)

        return run


def configure(stack, mode, audit):
    counts = {"memory_write": Counter(), "neighbors": Counter()}
    if mode == "reference":
        return counts
    from mars_titan.models.titans import cuda_kernels as kernels

    device = torch.device("cuda:0")
    if kernels._supported_device(device):
        raise ValueError("El código de PC2 ya selecciona esta arquitectura. Revisa la referencia")
    if mode != "portable":
        # Esta sustitución solo existe en este proceso de ensayo.
        stack.enter_context(patch.object(kernels, "_supported_device", lambda d: d == device))
        if mode == "write":
            from mars_titan.models.titans import episodic_readout, evidence_bridge

            def sorted_ids(scores, count):
                return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :count]

            for module in (episodic_readout, evidence_bridge):
                stack.enter_context(patch.object(module, "stable_neighbors", sorted_ids))
        if mode == "neighbors":
            from mars_titan.models.titans import neural_memory

            stack.enter_context(
                patch.object(neural_memory, "try_memory_write", lambda *a, **kw: None)
            )
        if audit:
            triton, write, neighbors = kernels._programs()
            programs = (
                triton,
                LaunchCounts(write, counts["memory_write"]),
                LaunchCounts(neighbors, counts["neighbors"]),
            )
            stack.enter_context(patch.object(kernels, "_programs", lambda: programs))
    return counts


def parameter_check(predictor):
    """Medir el sellado sobre los parámetros y los buffers del checkpoint restaurado."""
    named = (*predictor.named_parameters(), *predictor.named_buffers())

    def individual():
        fingerprints = {name: financial._tensor_digest(tensor) for name, tensor in named}
        return hashlib.sha256(financial.canonical(fingerprints).encode()).hexdigest()

    expected = individual()
    timings = {"original": [], "installed": []}
    for label in ("original", "installed", "installed", "original") * 3:
        compute = individual if label == "original" else predictor._parameter_digest
        torch.cuda.synchronize(0)
        before = time.perf_counter()
        digest = compute()
        torch.cuda.synchronize(0)
        timings[label].append(time.perf_counter() - before)
        if digest != expected:
            raise ValueError("La huella de los parámetros o los buffers ha cambiado")
    return dict(
        tensors=len(named),
        tensor_bytes=sum(t.numel() * t.element_size() for _, t in named),
        sha256=expected,
        seconds=timings,
    )


def compare(args):
    import recorrido
    from entradas_campana import sources

    if not Path(financial.__file__).resolve().is_relative_to(args.runtime.resolve() / "src"):
        raise ValueError("El módulo financiero no procede del directorio solicitado")
    checked = source_tree(args.runtime, args.mode, original_preparation=args.original_preparation)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError("Esta medición necesita cuda:0 con SM 8.9")
    if args.output.exists():
        raise ValueError("La salida del ensayo ya existe")
    torch.cuda.set_device("cuda:0")
    recorrido.rw._cuda("cuda:0")
    recorrido.apply_kernel_policy("fp32_strict")
    audit = args.kind == "audit"
    started = time.perf_counter()
    with ExitStack() as stack:
        guard = recorrido.perf._forbid_steps()
        stack.callback(guard.remove)
        stack.enter_context(recorrido.tw.unfused_attention())
        if args.model == "titans_mac":
            stack.enter_context(host_finite.on_host())
        counts = configure(stack, args.mode, audit)
        src = sources(2023, args.model, 43)
        train = src["train"]
        spec = train.specification()
        src["train"] = training_event_cache.PackedTrainingSource(
            train.dataset, train.path, os.environ[training_event_cache.ENV]
        )
        engine, outer, folder = recorrido.build(args.model, src, spec, 43, year=2023)
        before = recorrido.parameter_digest(engine)
        fraction, groups, segments = (0.9, 144, 16) if audit else (0.8, 576, 68)
        training_event_cache.reset_usage()
        case = recorrido.profile_slice(
            engine,
            spec,
            "original",
            audit,
            source_range=(int(fraction * len(train.metadata["groups"])), groups),
            rng_seed=43,
            until=segments,
        )
        built_and_run = time.perf_counter() - started
        receipt = training_event_cache.receipt()
        if receipt.get("packed_passes") != 1 or before != recorrido.parameter_digest(engine):
            raise ValueError("El ensayo requiere el archivo verificado y pesos fijos")
        calls = {
            name: dict(
                total=sum(items.values()),
                most_common=[[list(shape), n] for shape, n in items.most_common(4)],
            )
            for name, items in counts.items()
        }
        result = dict(
            model=args.model,
            mode=args.mode,
            source=checked,
            kind=args.kind,
            torch=str(torch.__version__),
            triton=version("triton"),
            device=torch.cuda.get_device_name(0),
            capability=list(torch.cuda.get_device_capability(0)),
            year=2023,
            seed=43,
            weights_sha256=before,
            archive_sha256=receipt["archive_sha256"],
            sample_input_sha256=receipt["input_sha256"],
            build_and_pass_seconds=built_and_run,
            real_optimizer_steps=0,
            case=case,
            launches=calls if audit else None,
        )
        if audit:
            result["parameter_check"] = parameter_check(engine.predictor)
        if args.validation:
            quantiles = (
                engine.quantiles if args.model.startswith("mars_titan_") else engine.emits_levels
            )
            rows = recorrido.tw.PredictionRows(quantiles)
            prediction = outer["predictions"]["validation"]
            file = folder / prediction["path"]
            if sha256(file) != prediction["sha256"]:
                raise ValueError("Ha cambiado el Parquet guardado")
            expected = pq.read_table(file)
            engine.evaluate(src["validation"], rows=rows)
            got = pa.concat_tables(rows.finish())
            result["validation"] = dict(
                rows=got.num_rows, exact=got.equals(expected), sha256=prediction["sha256"]
            )
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["wall_seconds"] = time.perf_counter() - started
        atomic_json(args.output, result)
        print(
            json.dumps(
                dict(
                    model=args.model,
                    mode=args.mode,
                    kind=args.kind,
                    samples=case["samples"],
                    pass_seconds=case["pass_seconds"],
                    steady_seconds=case["seconds"],
                    launches=result["launches"],
                    validation=result.get("validation"),
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("reference", "portable", "fused", "write", "neighbors"), required=True
    )
    parser.add_argument("--kind", choices=("audit", "timing"), required=True)
    parser.add_argument(
        "--model",
        choices=(
            "titans_mac",
            "mars_titan_m1",
            "mars_titan_m2",
            "mars_titan_m3",
            "gru",
            "lstm",
            "transformer",
        ),
        required=True,
    )
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Cotejar también la validación guardada de desarrollo",
    )
    parser.add_argument(
        "--original-preparation",
        action="store_true",
        help="Usar los kernels con el escalador y el predictor originales",
    )
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
