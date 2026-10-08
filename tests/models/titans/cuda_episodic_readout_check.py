"""Comprobación CUDA manual del lector puro, sin ajuste ni optimizadores."""

import json
import subprocess
import time

import torch
from test_episodic_readout import CODEC, CONTEXT, source

from mars_titan.models.titans.episodic_readout import (
    EpisodeSnapshot,
    EpisodicReadout,
    EpisodicReadoutConfig,
)


def test_cuda_readout_parity_gradients_and_recovery(tmp_path):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        text=True,
    )
    assert torch.cuda.is_available(), "La prueba requiere CUDA y no permite fallback"
    device = torch.device("cuda:0")
    cap = 128 * 1024**2
    torch.cuda.set_per_process_memory_fraction(
        cap / torch.cuda.get_device_properties(device).total_memory, device
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    records = []
    for dtype in (torch.float32, torch.float64):
        for steps in (1, 2, 4):
            config = EpisodicReadoutConfig(CODEC, hidden_size=32, refinements=steps)
            cpu = EpisodicReadout(config, dtype=dtype)
            gpu = EpisodicReadout(config, dtype=dtype, device=device)
            gpu.load_state_dict(cpu.state_dict())
            memory = EpisodeSnapshot.create(
                **source(8), cutoff=10, codec_id=CODEC, context_id=CONTEXT, dtype=dtype
            )
            gpu_memory = EpisodeSnapshot.restore(
                memory.export_cpu(),
                codec_id=CODEC,
                context_id=CONTEXT,
                cutoff=10,
                device=device,
                dtype=dtype,
            )
            z = (torch.arange(64, dtype=dtype).reshape(2, 32) / 64).requires_grad_()
            gpu_z = z.detach().to(device).requires_grad_()
            expected = cpu(z, memory, context_id=CONTEXT, cutoff=10, differentiable=True)
            actual = gpu(gpu_z, gpu_memory, context_id=CONTEXT, cutoff=10, differentiable=True)
            rtol, atol = (5e-5, 3e-6) if dtype == torch.float32 else (2e-9, 2e-10)
            torch.testing.assert_close(actual.state.cpu(), expected.state, rtol=rtol, atol=atol)
            for left, right in zip(expected.reads, actual.reads, strict=True):
                assert torch.equal(left.ids, right.ids.cpu())
                torch.testing.assert_close(right.weights.cpu(), left.weights, rtol=rtol, atol=atol)
            gradients = []
            for model, value, output in ((cpu, z, expected), (gpu, gpu_z, actual)):
                gradients.append(
                    torch.autograd.grad(
                        output.state.square().sum(),
                        (
                            value,
                            model.query_projection.weight,
                            model.value_projection.weight,
                            model.refinement.weight,
                            model.step_logit,
                        ),
                    )
                )
            maximum = 0.0
            for left, right in zip(*gradients, strict=True):
                torch.testing.assert_close(right.cpu(), left, rtol=rtol, atol=atol)
                maximum = max(maximum, float((right.cpu() - left).abs().max()))
            recovered = EpisodeSnapshot.restore(
                gpu_memory.export_cpu(),
                codec_id=CODEC,
                context_id=CONTEXT,
                cutoff=10,
                device=device,
                dtype=dtype,
            )
            repeated = gpu(gpu_z, recovered, context_id=CONTEXT, cutoff=10)
            assert torch.equal(repeated.state, actual.state.detach())
            assert all(p.grad is None for p in gpu.parameters())
            records.append(
                dict(
                    dtype=str(dtype),
                    K=steps,
                    exact_ids=True,
                    exact_recovery=True,
                    max_gradient_error=maximum,
                    rtol=rtol,
                    atol=atol,
                )
            )
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    torch.cuda.synchronize(device)
    assert torch.cuda.max_memory_allocated(device) <= cap
    (tmp_path / "cuda-episodic-readout.json").write_text(
        json.dumps(
            dict(
                records=records,
                hardware=hardware.strip(),
                torch=torch.__version__,
                tf32=False,
                cpu_rng_unchanged=True,
                cuda_rng_unchanged=True,
                cap_torch_bytes=cap,
                peak_torch_bytes=torch.cuda.max_memory_allocated(device),
                peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(device),
                process_seconds=time.perf_counter() - started,
                scope="Fixtures del lector puro, sin MAC, corpus real ni optimizador.",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
