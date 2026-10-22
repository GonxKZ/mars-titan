"""Coste de escritura por banco, preparación y sesión recuperable sobre fixtures."""

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
from copy import deepcopy
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "reports/engineering/bank-readout-20261021/benchmark.py"
TRAJECTORY = ROOT / "reports/engineering/mechanism-session-20261021/benchmark.py"
helper = runpy.run_path(str(HELPER))
trajectory = runpy.run_path(str(TRAJECTORY))["trajectory"]

from mars_titan.memory.native_backend import load_native  # noqa: E402
from mars_titan.models.titans.bank_readout import BankReadoutConfig  # noqa: E402
from mars_titan.models.titans.config import MemoryConfig, MemoryMultiscale  # noqa: E402
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer  # noqa: E402
from mars_titan.models.titans.neural_memory import NeuralMemory  # noqa: E402
from mars_titan.training.titans_walk_forward import _predictor  # noqa: E402

RULES = ("common_translation", "mean_partial", "per_bank")


def timed(call, *, warmup=3, repetitions=20):
    for _ in range(warmup):
        value = call()
        del value
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    resident = torch.cuda.memory_allocated(0)
    values = []
    for _ in range(repetitions):
        torch.cuda.synchronize(0)
        tick = time.perf_counter_ns()
        value = call()
        torch.cuda.synchronize(0)
        values.append((time.perf_counter_ns() - tick) / 1e6)
        del value
    return dict(
        **helper["summarize"](values),
        allocated_resident_bytes=resident,
        allocated_peak_bytes=torch.cuda.max_memory_allocated(0),
        reserved_peak_bytes=torch.cuda.max_memory_reserved(0),
    )


def profile_update(model, observed, state):
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            for _ in range(3):
                model.update(observed, state)
            torch.cuda.synchronize(0)
        return [
            dict(
                name=e.key,
                calls=e.count,
                self_cpu_us=e.self_cpu_time_total,
                self_device_us=getattr(e, "self_device_time_total", None),
            )
            for e in profile.key_averages()
        ]
    except RuntimeError as error:
        return dict(unavailable=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
        or not args.storage_root.is_dir()
    ):
        raise ValueError("Se necesita una salida nueva y un directorio de trabajo existente")
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
    old = json.loads((ROOT / declaration["legacy_reference"]).read_text())
    recipe = json.loads((ROOT / declaration["base_recipe"]).read_text())
    report = dict(
        status="running_not_complete",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        torch_git=torch.version.git_version,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        device="cuda:0",
        cpu_threads=torch.get_num_threads(),
        real_corpus=False,
        optimizer_steps=0,
        cases=[],
        profiles=[],
        native_sha256=native.binary_sha256,
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        energy_measured=False,
        component_max_batch=recipe["predictor"]["max_batch"],
        native_max_batch=256,
    )

    def save():
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    save()
    started = time.perf_counter()
    for dtype in (torch.float32, torch.float64):
        for count in (1, 64, 256):
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
                model = (
                    NeuralMemory(config, device="cuda:0", dtype=dtype).eval().requires_grad_(False)
                )
                observed = torch.linspace(
                    -0.2, 0.3, count * 64, dtype=dtype, device="cuda:0"
                ).reshape(count, 1, 64)
                state = model.initial_state(count)
                result = timed(partial(model.update, observed, state))
                following = model.update(observed, state)
                read = model.read(observed, following)
                tensors = (*state.weights, *state.momentum, *state.levels, state.steps)
                report["cases"].append(
                    dict(
                        scope="memory_update",
                        rows=count,
                        dtype=str(dtype),
                        rule=rule,
                        **result,
                        configuration=config.identity(),
                        state_tensor_bytes=sum(t.numel() * t.element_size() for t in tensors),
                        post_write_read_sha256=hashlib.sha256(
                            read.detach().cpu().numpy().tobytes()
                        ).hexdigest(),
                    )
                )
                if dtype == torch.float32 and count == 256:
                    report["profiles"].append(
                        dict(
                            scope="memory_update",
                            rule=rule,
                            rows=count,
                            events=profile_update(model, observed, state),
                        )
                    )
                save()
                del model, observed, state, following, read, tensors
                torch.cuda.empty_cache()
    financial = {"ema_query_common_translation": old["arms"]["ema_query"]}
    for name in (
        "ema_query_mean_partial",
        "ema_query_per_bank",
        "replica_uniform_mean_partial",
        "replica_uniform_per_bank",
        "ema_query_graph_mean_partial",
        "ema_query_graph_per_bank",
    ):
        financial[name] = declaration["arms"][name]
    with tempfile.TemporaryDirectory(prefix="mars-bank-write-cost-", dir=args.storage_root) as tmp:
        folder = Path(tmp)
        for count in (1, 64, 256):
            source = helper["flow_source"](
                folder / f"source-{count}", flows=tuple(f"US/T{i:04d}" for i in range(count))
            )
            inputs = {
                day: [
                    helper["raw"](source["batches"][day][i : i + 64]) for i in range(0, count, 64)
                ]
                for day in (125, 126, 127, 128)
            }
            for name, arm in financial.items():
                model, _ = _predictor(
                    recipe,
                    source["spec"],
                    "mac_online",
                    42,
                    "cuda:0",
                    changes=arm["predictor_changes"],
                )
                model.eval().requires_grad_(False)
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
                measured = timed(partial(helper["prepare_financial"], model, states, inputs[128]))
                quantiles, proposals = helper["prepare_financial"](model, states, inputs[128])
                report["cases"].append(
                    dict(
                        scope="financial_prepare",
                        rows=count,
                        arm=name,
                        dtype="torch.float32",
                        **measured,
                        configuration=model.config.identity(),
                        parameter_id=model._parameter_id,
                        input_spec_id=source["spec"].fingerprint(),
                        tensor_state_bytes=sum(
                            model.state_usage(s)["tensor_bytes"] for s in states
                        ),
                        counts=model.operation_counts(count),
                        output_sha256=hashlib.sha256(
                            quantiles.detach().cpu().numpy().tobytes()
                        ).hexdigest(),
                        blocks=len(states),
                        block_rows=64,
                        supervision_origin=source["label_origin"],
                    )
                )
                save()
                del model, state, states, batch, quantiles, proposals
                torch.cuda.empty_cache()
        for count in (4, 256):
            source = helper["flow_source"](
                folder / f"native-source-{count}", flows=tuple(f"US/T{i:04d}" for i in range(count))
            )
            engines = {}
            native_recipe = deepcopy(recipe)
            native_recipe["predictor"]["max_batch"] = 256
            for rule in ("mean_partial", "per_bank"):
                model, _ = _predictor(
                    native_recipe,
                    source["spec"],
                    "mac_online",
                    42,
                    "cuda:0",
                    changes=declaration["arms"]["ema_query_" + rule]["predictor_changes"],
                )
                engines[rule] = FrozenMechanismConsumer(model.eval().requires_grad_(False))
            samples, peaks, digests, sizes = ({rule: [] for rule in engines} for _ in range(4))
            for repetition in range(5):
                order = tuple(engines) if repetition % 2 == 0 else tuple(reversed(engines))
                for rule in order:
                    torch.cuda.reset_peak_memory_stats(0)
                    values, size, output_sha = trajectory(
                        native,
                        source,
                        engines[rule],
                        folder / f"native-{count}-{rule}-{repetition}",
                    )
                    if repetition:
                        samples[rule].append(values)
                        sizes[rule].append(size)
                        digests[rule].append(output_sha)
                        peaks[rule].append(torch.cuda.max_memory_allocated(0))
            for rule in engines:
                if len(set(digests[rule])) != 1:
                    raise RuntimeError("Las repeticiones de una regla cambian sus predicciones")
                totals = [s["total"] for s in samples[rule]]
                report["cases"].append(
                    dict(
                        scope="native_trajectory",
                        rows=count,
                        rule=rule,
                        dtype="torch.float32",
                        **helper["summarize"](totals),
                        phase_samples_ms=samples[rule],
                        retained_bytes=sizes[rule],
                        allocated_peak_bytes=max(peaks[rule]),
                        output_sha256=digests[rule][0],
                        configuration=engines[rule].predictor.config.identity(),
                        parameter_id=engines[rule].predictor._parameter_id,
                        input_spec_id=source["spec"].fingerprint(),
                        consumer_id=engines[rule].model_id,
                        supervision_origin=source["label_origin"],
                        warmup=1,
                        repetitions=4,
                        resident_models=2,
                        prediction_rows_per_second=3 * count * 1000 / float(np.mean(totals)),
                    )
                )
            save()
            del model, engines
            torch.cuda.empty_cache()
    paths = [
        Path(__file__),
        HELPER,
        TRAJECTORY,
        ROOT / "configs/experiments/bank-write-v1.json",
        ROOT / declaration["base_recipe"],
        ROOT / declaration["legacy_reference"],
        ROOT / "src/mars_titan/training/titans_walk_forward.py",
        ROOT / "tests/memory/test_financial_session_controls.py",
        ROOT / "uv.lock",
    ]
    paths.extend(
        ROOT / ("src/mars_titan/models/titans/" + name + ".py")
        for name in (
            "bank_readout",
            "config",
            "neural_memory",
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
