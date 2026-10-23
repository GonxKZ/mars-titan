"""Coste CUDA de la máscara de acciones con coeficientes manuales, sin ajuste."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mars_titan.models.titans.mechanism_policy import FEATURES, MechanismPolicy

MODES = ("legacy", "mask_all", "mask_subset")


def model(dtype, mode):
    width = len(FEATURES)
    point = torch.zeros((width + 1, 12), dtype=dtype, device="cuda:0")
    interval = torch.zeros((width + 1, 36), dtype=dtype, device="cuda:0")
    point[-1, 1], interval[-1, 4] = 2, 3
    mask = None if mode == "legacy" else torch.ones(36, dtype=torch.bool, device="cuda:0")
    if mode == "mask_subset":
        mask[4] = False
    return MechanismPolicy(
        mean=point.new_zeros(width),
        scale=point.new_ones(width),
        point_coefficients=point,
        interval_coefficients=interval,
        support_min=point.new_full((width,), -10),
        support_max=point.new_full((width,), 10),
        costs=point.new_zeros(36),
        interval_weight=0.5,
        min_gain=0.0,
        action_mask=mask,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("La salida necesita una ruta nueva con directorio existente")
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere cuda:0 y no se pasa a CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    report = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        device="cuda:0",
        gpu=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        torch=str(torch.__version__),
        torch_commit=torch.version.git_version,
        cuda=torch.version.cuda,
        python=platform.python_version(),
        numpy=np.__version__,
        cpu_threads=torch.get_num_threads(),
        warmup=10,
        repetitions=100,
        fitted=False,
        optimizer_steps=0,
        real_data=False,
        scope="manual_policy_forward_with_synchronization_not_financial_predictor_or_fit",
        cases=[],
    )
    started = time.perf_counter()
    for dtype in (torch.float32, torch.float64):
        for size in (1, 2):
            models = {mode: model(dtype, mode) for mode in MODES}
            x = torch.zeros((size, len(FEATURES)), dtype=dtype, device="cuda:0")
            available = torch.ones((size, 36), dtype=torch.bool, device="cuda:0")
            cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state(0)
            samples = {mode: [] for mode in MODES}
            peaks = {mode: 0 for mode in MODES}
            score_hashes = {mode: set() for mode in MODES}
            reference = None
            begin, finish = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for repetition in range(110):
                order = MODES[repetition % 3 :] + MODES[: repetition % 3]
                for mode in order:
                    torch.cuda.synchronize(0)
                    torch.cuda.reset_peak_memory_stats(0)
                    start = time.perf_counter_ns()
                    begin.record()
                    result = models[mode](x, available)
                    finish.record()
                    torch.cuda.synchronize(0)
                    wall = (time.perf_counter_ns() - start) / 1e6
                    gpu = begin.elapsed_time(finish)
                    scores = result.scores.cpu().numpy()
                    if reference is None:
                        reference = scores.copy()
                    if not np.array_equal(scores, reference):
                        raise RuntimeError(
                            "La máscara alteró las utilidades sin aplicar admisibilidad"
                        )
                    expected = 3 if mode == "mask_subset" else 4
                    if result.action_ids.tolist() != [expected] * size:
                        raise RuntimeError("La máscara no conserva la elección manual esperada")
                    if repetition >= 10:
                        samples[mode].append(dict(wall_ms=wall, cuda_event_ms=gpu))
                        peaks[mode] = max(peaks[mode], torch.cuda.max_memory_allocated(0))
                        score_hashes[mode].add(hashlib.sha256(scores.tobytes()).hexdigest())
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
                gpu_rng, torch.cuda.get_rng_state(0)
            ):
                raise RuntimeError("La decisión consumió aleatoriedad")
            for mode in MODES:
                values = np.array([s["wall_ms"] for s in samples[mode]])
                kernels = np.array([s["cuda_event_ms"] for s in samples[mode]])
                report["cases"].append(
                    dict(
                        dtype=str(dtype),
                        cohorts=size,
                        mode=mode,
                        samples=samples[mode],
                        p50_wall_ms=float(np.percentile(values, 50)),
                        p95_wall_ms=float(np.percentile(values, 95)),
                        p99_wall_ms=float(np.percentile(values, 99)),
                        p50_cuda_event_ms=float(np.median(kernels)),
                        p50_cohorts_per_second=size * 1000 / float(np.median(values)),
                        allocated_peak_bytes=peaks[mode],
                        raw_scores_sha256=sorted(score_hashes[mode]),
                        chosen_action=3 if mode == "mask_subset" else 4,
                    )
                )
    root = Path(__file__).resolve().parents[3]
    report.update(
        process_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        sources={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), root / "src/mars_titan/models/titans/mechanism_policy.py")
        },
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps([{k: v for k, v in case.items() if k != "samples"} for case in report["cases"]])
    )


if __name__ == "__main__":
    main()
