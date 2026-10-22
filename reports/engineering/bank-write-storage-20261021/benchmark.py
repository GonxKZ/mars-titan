"""Comparación alternada de copias y préstamo de pesos sobre fixtures idénticos."""

import argparse
import hashlib
import json
import os
import platform
import resource
import runpy
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = ROOT / "reports/engineering/bank-write-20261021/benchmark.py"
REFERENCE = Path(__file__).with_name("copied_reference.py")
CANDIDATE = Path(__file__).with_name("borrowed_candidate.py")
previous = runpy.run_path(str(PREVIOUS))
reference = runpy.run_path(str(REFERENCE))["_associative_problem"]
candidate = runpy.run_path(str(CANDIDATE))["_associative_problem"]
helper = previous["helper"]

from mars_titan.memory.native_backend import load_native  # noqa: E402
from mars_titan.models.titans.bank_readout import BankReadoutConfig  # noqa: E402
from mars_titan.models.titans.config import MemoryConfig, MemoryMultiscale  # noqa: E402
from mars_titan.models.titans.neural_memory import NeuralMemory  # noqa: E402
from mars_titan.training.titans_walk_forward import _predictor  # noqa: E402

RULES = ("mean_partial", "per_bank")
VERSIONS = ("copied", "borrowed")


def install(model, implementation):
    changed = 0
    for module in model.modules():
        if isinstance(module, NeuralMemory) and module.config.momentum_copies == 3:
            module._associative_problem = MethodType(implementation, module)
            changed += 1
    if changed != 1:
        raise ValueError("Se esperaba una única memoria neuronal BANK-WRITE")
    return model


def digest(value):
    """Huella recursiva de tensores y metadatos, fuera de la región temporizada."""
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(str((item.dtype, tuple(item.shape))).encode())
            h.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                h.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for part in item:
                visit(part)
        else:
            h.update(repr(item).encode())

    visit(value)
    return h.hexdigest()


def paired(calls, *, warmup=3, repetitions=20):
    samples, peaks, residents, reserved = ({name: [] for name in VERSIONS} for _ in range(4))
    for index in range(warmup + repetitions):
        order = VERSIONS if index % 2 == 0 else VERSIONS[::-1]
        for name in order:
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
            resident = torch.cuda.memory_allocated(0)
            started = time.perf_counter_ns()
            value = calls[name]()
            torch.cuda.synchronize(0)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if index >= warmup:
                samples[name].append(elapsed)
                peaks[name].append(torch.cuda.max_memory_allocated(0))
                residents[name].append(resident)
                reserved[name].append(torch.cuda.max_memory_reserved(0))
            del value
    return {
        name: dict(
            **helper["summarize"](samples[name]),
            allocated_peak_samples_bytes=peaks[name],
            allocated_resident_samples_bytes=residents[name],
            reserved_peak_samples_bytes=reserved[name],
        )
        for name in VERSIONS
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("La salida debe ser una ruta nueva en un directorio existente")
    if not args.storage_root.is_dir():
        raise ValueError("No existe el directorio de almacenamiento temporal")
    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita cuda:0, sin alternativa CPU implícita")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    native.require_safe_path(str(args.storage_root))
    declaration = json.loads((ROOT / "configs/experiments/bank-write-v1.json").read_text())
    recipe = json.loads((ROOT / declaration["base_recipe"]).read_text())
    report = dict(
        status="running_not_complete",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        torch_git=torch.version.git_version,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        cpu_threads=torch.get_num_threads(),
        device="cuda:0",
        tf32=False,
        autocast=False,
        deterministic_algorithms=True,
        real_corpus=False,
        outer_optimizer_steps=0,
        native_sha256=native.binary_sha256,
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        cases=[],
        profiles=[],
        energy_measured=False,
    )

    def save():
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    save()
    started = time.perf_counter()
    for dtype in (torch.float32, torch.float64):
        for rows in (1, 64, 256):
            for rule in RULES:
                config = MemoryConfig(
                    64,
                    depth=2,
                    max_batch=256,
                    max_tokens=1,
                    max_state_bytes=128 * 1024**2,
                    multiscale=MemoryMultiscale(form="ema_bank"),
                    bank_readout=BankReadoutConfig(write_gradient=rule),
                )
                models = {
                    name: NeuralMemory(config, device="cuda:0", dtype=dtype)
                    .eval()
                    .requires_grad_(False)
                    for name in VERSIONS
                }
                install(models["copied"], reference)
                install(models["borrowed"], candidate)
                if digest(models["copied"].state_dict()) != digest(models["borrowed"].state_dict()):
                    raise RuntimeError("Los parámetros iniciales no coinciden")
                observed = torch.linspace(-0.2, 0.3, rows * 64, device="cuda:0", dtype=dtype)
                observed = observed.reshape(rows, 1, 64)
                state = models["borrowed"].initial_state(rows)
                before = digest(models["borrowed"].export_state(state))
                following = {name: model.update(observed, state) for name, model in models.items()}
                fingerprints = {
                    name: digest(model.export_state(following[name]))
                    for name, model in models.items()
                }
                if len(set(fingerprints.values())) != 1:
                    raise RuntimeError("La actualización cambió sus bits")
                del following
                measured = paired(
                    {name: partial(model.update, observed, state) for name, model in models.items()}
                )
                if before != digest(models["borrowed"].export_state(state)):
                    raise RuntimeError("Se modificó el estado de entrada")
                report["cases"].append(
                    dict(
                        scope="memory_update",
                        rule=rule,
                        rows=rows,
                        dtype=str(dtype),
                        configuration=config.identity(),
                        output_sha256=fingerprints["copied"],
                        equal_bits=True,
                        warmup=3,
                        repetitions=20,
                        versions=measured,
                    )
                )
                if rows == 256 and dtype == torch.float32:
                    for name in VERSIONS:
                        report["profiles"].append(
                            dict(
                                rule=rule,
                                version=name,
                                events=previous["profile_update"](models[name], observed, state),
                            )
                        )
                save()
                del models, state, observed, measured
                torch.cuda.empty_cache()

    arms = tuple("ema_query_" + infix + rule for infix in ("", "graph_") for rule in RULES)
    with tempfile.TemporaryDirectory(prefix="mars-bank-storage-", dir=args.storage_root) as tmp:
        folder = Path(tmp)
        for rows in (1, 64, 256):
            source = helper["flow_source"](
                folder / f"source-{rows}", flows=tuple(f"US/T{i:04d}" for i in range(rows))
            )
            inputs = {
                day: [helper["raw"](source["batches"][day][i : i + 64]) for i in range(0, rows, 64)]
                for day in (125, 126, 127, 128)
            }
            for arm in arms:
                models = {}
                for name in VERSIONS:
                    model, _ = _predictor(
                        recipe,
                        source["spec"],
                        "mac_online",
                        42,
                        "cuda:0",
                        changes=declaration["arms"][arm]["predictor_changes"],
                    )
                    models[name] = model.eval().requires_grad_(False)
                install(models["copied"], reference)
                install(models["borrowed"], candidate)
                if models["copied"]._parameter_id != models["borrowed"]._parameter_id:
                    raise RuntimeError("Los parámetros financieros no coinciden")
                states = []
                for block in range(len(inputs[128])):
                    state = None
                    for day in (125, 126, 127):
                        batch = helper["DecisionBatch"].from_corpus(
                            inputs[day][block],
                            model.config.inputs,
                            device="cuda:0",
                            dtype=torch.float32,
                        )
                        state = model.initial_state(batch.flow_ids) if state is None else state
                        state = model.prepare(batch, state).next_state
                    states.append(state)
                before = digest([model.export_state(s) for s in states])
                outputs = {
                    name: helper["prepare_financial"](model, states, inputs[128])
                    for name, model in models.items()
                }
                fingerprints = {
                    name: digest(
                        (value[0], [models[name].export_state(p.next_state) for p in value[1]])
                    )
                    for name, value in outputs.items()
                }
                if len(set(fingerprints.values())) != 1:
                    raise RuntimeError("La preparación financiera cambió sus bits")
                del outputs
                measured = paired(
                    {
                        name: partial(helper["prepare_financial"], model, states, inputs[128])
                        for name, model in models.items()
                    }
                )
                if before != digest([model.export_state(s) for s in states]):
                    raise RuntimeError("Se modificó el estado financiero de entrada")
                report["cases"].append(
                    dict(
                        scope="financial_prepare",
                        arm=arm,
                        rows=rows,
                        dtype="torch.float32",
                        block_rows=64,
                        model_max_batch=128,
                        configuration=model.config.identity(),
                        parameter_id=model._parameter_id,
                        output_sha256=fingerprints["copied"],
                        equal_bits=True,
                        warmup=3,
                        repetitions=20,
                        versions=measured,
                    )
                )
                save()
                del models, model, state, states, batch, measured
                torch.cuda.empty_cache()

    paths = [
        Path(__file__),
        REFERENCE,
        CANDIDATE,
        PREVIOUS,
        previous["HELPER"],
        previous["TRAJECTORY"],
        ROOT / "uv.lock",
        ROOT / "configs/experiments/bank-write-v1.json",
        ROOT / declaration["base_recipe"],
    ]
    paths.extend(
        ROOT / "src/mars_titan/models/titans" / (name + ".py")
        for name in (
            "neural_memory",
            "config",
            "bank_readout",
            "financial",
            "financial_blocks",
            "state",
            "transition_jacobian",
        )
    )
    report.update(
        status="complete",
        process_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    )
    save()


if __name__ == "__main__":
    main()
