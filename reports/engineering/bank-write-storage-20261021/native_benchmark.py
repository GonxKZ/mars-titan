"""Sesiones protegidas sobre dos árboles de fuentes, sin sustituir métodos."""

import argparse
import hashlib
import json
import os
import resource
import runpy
import shutil
import subprocess
import tempfile
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = ROOT / "reports/engineering/bank-write-20261021/benchmark.py"
previous = runpy.run_path(str(PREVIOUS))
helper = previous["helper"]

from mars_titan.models.titans import neural_memory  # noqa: E402
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer  # noqa: E402
from mars_titan.training.titans_walk_forward import _predictor  # noqa: E402


def worker(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita cuda:0, sin alternativa CPU implícita")
    source_path = Path(neural_memory.__file__)
    memory_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if memory_sha != args.expected_memory_sha:
        raise RuntimeError("El proceso ha importado otra fuente de memoria")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    native = previous["load_native"](os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    native.require_safe_path(str(args.storage_root))
    declaration = json.loads((ROOT / "configs/experiments/bank-write-v1.json").read_text())
    recipe = deepcopy(json.loads((ROOT / declaration["base_recipe"]).read_text()))
    recipe["predictor"]["max_batch"] = 256
    source = helper["flow_source"](
        args.fixture_root, flows=tuple(f"US/T{i:04d}" for i in range(args.rows))
    )
    model, _ = _predictor(
        recipe,
        source["spec"],
        "mac_online",
        42,
        "cuda:0",
        changes=declaration["arms"]["ema_query_" + args.rule]["predictor_changes"],
    )
    engine = FrozenMechanismConsumer(model.eval().requires_grad_(False))
    samples, peaks, sizes, hashes = [], [], [], []
    for index in range(5):
        torch.cuda.reset_peak_memory_stats(0)
        phases, size, output_sha = previous["trajectory"](
            native, source, engine, args.storage_root / f"session-{index}"
        )
        if index:
            samples.append(phases)
            peaks.append(torch.cuda.max_memory_allocated(0))
            sizes.append(size)
            hashes.append(output_sha)
    if len(set(hashes)) != 1 or len(set(sizes)) != 1:
        raise RuntimeError("Las repeticiones cambian las predicciones o los bytes retenidos")
    report = dict(
        status="complete",
        rule=args.rule,
        rows=args.rows,
        configuration=model.config.identity(),
        parameter_id=model._parameter_id,
        consumer_id=engine.model_id,
        input_spec_id=source["spec"].fingerprint(),
        native_sha256=native.binary_sha256,
        memory_source_sha256=memory_sha,
        memory_source=str(source_path),
        torch=str(torch.__version__),
        torch_git=torch.version.git_version,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        cpu_threads=torch.get_num_threads(),
        device="cuda:0",
        dtype="torch.float32",
        tf32=False,
        autocast=False,
        deterministic_algorithms=True,
        outer_optimizer_steps=0,
        real_corpus=False,
        warmup=1,
        repetitions=4,
        resident_models=1,
        model_max_batch=256,
        block_rows=64,
        **helper["summarize"]([s["total"] for s in samples]),
        phases_ms=samples,
        allocated_peak_samples_bytes=peaks,
        retained_bytes=sizes,
        output_sha256=hashes[0],
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rows", type=int, choices=(4, 256))
    parser.add_argument("--rule", choices=("mean_partial", "per_bank"))
    parser.add_argument("--expected-memory-sha")
    parser.add_argument("--fixture-root", type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir() or not args.storage_root.is_dir():
        raise ValueError("Se necesitan una salida nueva y directorios existentes")
    if args.worker:
        if (
            args.rows is None
            or args.rule is None
            or args.expected_memory_sha is None
            or args.fixture_root is None
        ):
            raise ValueError("El trabajador necesita filas, regla y huella de fuente")
        worker(args)
        return
    if args.reference_source is None or not args.reference_source.is_dir():
        raise ValueError("Falta el árbol de fuentes de referencia")
    if args.candidate_source is None or not args.candidate_source.is_dir():
        raise ValueError("Falta el árbol de fuentes del candidato descartado")
    versions = {
        "copied": args.reference_source,
        "borrowed": args.candidate_source,
    }
    report = dict(
        status="running_not_complete",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        order="versiones_agrupadas_orden_invertido_entre_condiciones",
        cases=[],
        children=[],
        outer_optimizer_steps=0,
        real_corpus=False,
        energy_measured=False,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    started = time.perf_counter()
    for case, (rows, rule) in enumerate(
        ((4, "mean_partial"), (4, "per_bank"), (256, "mean_partial"), (256, "per_bank"))
    ):
        measurements = {}
        order = ("copied", "borrowed") if case % 2 == 0 else ("borrowed", "copied")
        # Ruta estable dentro de un temporal propio, recreada secuencialmente.
        with tempfile.TemporaryDirectory(prefix="shared-fixture-", dir=args.storage_root) as tmp:
            shared_fixture = Path(tmp) / "source"
            for version in order:
                if shared_fixture.exists():
                    shutil.rmtree(shared_fixture)
                folder = args.storage_root / f"{rows}-{rule}-{version}"
                folder.mkdir(exist_ok=False)
                output = folder / "result.json"
                memory_sha = hashlib.sha256(
                    (versions[version] / "mars_titan/models/titans/neural_memory.py").read_bytes()
                ).hexdigest()
                env = dict(os.environ, PYTHONPATH=f"{versions[version]}:{ROOT}")
                command = [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--output",
                    str(output),
                    "--storage-root",
                    str(folder),
                    "--rows",
                    str(rows),
                    "--rule",
                    rule,
                    "--expected-memory-sha",
                    memory_sha,
                    "--fixture-root",
                    str(shared_fixture),
                ]
                tick = time.perf_counter()
                with (folder / "process.log").open("w") as stream:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=1200,
                    )
                report["children"].append(
                    dict(
                        rows=rows,
                        rule=rule,
                        version=version,
                        exit_code=result.returncode,
                        wall_seconds=time.perf_counter() - tick,
                    )
                )
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                if result.returncode:
                    raise RuntimeError("Ha fallado el proceso, consulte su registro")
                measurements[version] = json.loads(output.read_text())
        baseline, candidate = (measurements[name] for name in ("copied", "borrowed"))
        for field in (
            "output_sha256",
            "parameter_id",
            "input_spec_id",
            "configuration",
            "native_sha256",
            "retained_bytes",
        ):
            if baseline[field] != candidate[field]:
                raise RuntimeError("El par difiere en " + field)
        report["cases"].append(
            dict(
                scope="native_trajectory",
                rows=rows,
                rule=rule,
                order=order,
                equal_bits=True,
                versions=measurements,
            )
        )
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report.update(
        status="complete",
        process_seconds=time.perf_counter() - started,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                PREVIOUS,
                previous["HELPER"],
                previous["TRAJECTORY"],
            )
        },
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
