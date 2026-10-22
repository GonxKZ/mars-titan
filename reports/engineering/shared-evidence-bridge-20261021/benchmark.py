"""Coste del componente de vistas compartidas, sin ejecutar el predictor ni ajustarlo."""

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

from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig, SharedEvidenceBridge

ROOT = Path(__file__).resolve().parents[3]


def evidence(count, dtype, device, generator):
    keys = torch.randn((count, 64), generator=generator, dtype=torch.float32)
    keys /= keys.norm(dim=1, keepdim=True)
    return EpisodeSnapshot.create(
        keys=keys,
        values=torch.randn((count, 64), generator=generator, dtype=torch.float32),
        labels=torch.randn((count,), generator=generator, dtype=torch.float64),
        ids=torch.arange(1, count + 1, dtype=torch.int64),
        decision_at=torch.full((count,), 2, dtype=torch.int64),
        available_at=torch.full((count,), 1, dtype=torch.int64),
        maturity_at=torch.full((count,), 3, dtype=torch.int64),
        cutoff=4,
        codec_id="a" * 64,
        context_id="b" * 64,
        dtype=dtype,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("Se necesita una salida nueva en una carpeta existente")
    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita cuda:0 explícito, sin alternativa CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    report = dict(
        status="running_not_complete",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        code_parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        scope="standalone_shared_evidence_bridge_not_full_prediction",
        real_corpus=False,
        shared_optimizer_steps=0,
        predictive_improvement_measured=False,
        device="cuda:0",
        gpu=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch=str(torch.__version__),
        torch_git=torch.version.git_version,
        python=platform.python_version(),
        cuda=torch.version.cuda,
        cpu_threads=torch.get_num_threads(),
        tf32=False,
        autocast=False,
        warmup=3,
        repetitions=20,
        energy_measured=False,
        cases=[],
    )

    def save():
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    save()
    started = time.perf_counter()
    for dtype in (torch.float64, torch.float32):
        for rows in (4, 64, 256):
            for episodes in (8, 128, 1024):
                for regions in (3, 8):
                    generator = torch.Generator().manual_seed(42)
                    cpu_source = evidence(episodes, dtype, "cpu", generator)
                    source = EpisodeSnapshot.restore(
                        cpu_source.export_cpu(),
                        codec_id="a" * 64,
                        context_id="b" * 64,
                        cutoff=4,
                        dtype=dtype,
                        device="cuda:0",
                    )
                    state = torch.randn((rows, 32), dtype=dtype, generator=generator)
                    config = EvidenceBridgeConfig(
                        codec_id="a" * 64,
                        hidden_size=32,
                        regions=tuple(f"view_{i}" for i in range(regions)),
                        cycles=2,
                    )
                    cpu = SharedEvidenceBridge(config, dtype=dtype)
                    model = SharedEvidenceBridge(config, dtype=dtype, device="cuda:0")
                    model.load_state_dict(cpu.state_dict())
                    state_gpu = state.to(device="cuda:0")
                    with torch.no_grad():
                        expected = cpu(state, cpu_source, context_id="b" * 64, cutoff=4)
                        actual = model(state_gpu, source, context_id="b" * 64, cutoff=4)
                        tolerance = (
                            dict(rtol=1e-11, atol=1e-12)
                            if dtype == torch.float64
                            else dict(rtol=1e-5, atol=2e-6)
                        )
                        assert torch.equal(expected.evidence_ids, actual.evidence_ids.cpu())
                        for name in (
                            "values",
                            "region_values",
                            "evidence_weights",
                            "bridge_weights",
                        ):
                            torch.testing.assert_close(
                                getattr(actual, name).cpu(), getattr(expected, name), **tolerance
                            )
                        del actual, expected, cpu
                        source.verify()
                        torch.cuda.synchronize(0)
                        before = torch.cuda.memory_allocated(0)
                        torch.cuda.reset_peak_memory_stats(0)
                        samples = []
                        for repetition in range(23):
                            torch.cuda.synchronize(0)
                            tick = time.perf_counter_ns()
                            result = model(state_gpu, source, context_id="b" * 64, cutoff=4)
                            torch.cuda.synchronize(0)
                            elapsed = (time.perf_counter_ns() - tick) / 1e6
                            if repetition >= 3:
                                samples.append(elapsed)
                            del result
                        report["cases"].append(
                            dict(
                                rows=rows,
                                episodes=episodes,
                                regions=regions,
                                hidden_size=32,
                                dtype=str(dtype),
                                cycles=2,
                                tolerance=tolerance,
                                snapshot_sha256=source._digest,
                                configuration=config.identity(),
                                samples_ms=samples,
                                p50_ms=float(np.percentile(samples, 50)),
                                p95_ms=float(np.percentile(samples, 95)),
                                p99_ms=float(np.percentile(samples, 99)),
                                rows_per_second=rows * 1000 / float(np.mean(samples)),
                                allocated_before_bytes=before,
                                allocated_peak_bytes=torch.cuda.max_memory_allocated(0),
                                reserved_peak_bytes=torch.cuda.max_memory_reserved(0),
                                estimated_working_bytes=model.estimated_bytes(rows, episodes),
                            )
                        )
                    source.verify()
                    del model, state_gpu, state, source, cpu_source
                    torch.cuda.empty_cache()
                    save()
    paths = [Path(__file__), ROOT / "uv.lock"]
    paths.extend(
        ROOT / ("src/mars_titan/models/titans/" + name + ".py")
        for name in ("evidence_bridge", "episodic_snapshot", "episodic_readout", "state", "config")
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
