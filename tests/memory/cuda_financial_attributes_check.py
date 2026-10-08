"""Sonda CUDA focal de la firma numérica, posterior a la matriz de integración."""

import json
import subprocess

import numpy as np
import pytest
import test_native_episode_backend as backend_fixtures
import torch
from cuda_financial_session_check import consumer, state_tensors, walk
from test_financial_session import inputs, resources, session

native = backend_fixtures.native


def test_cuda_numeric_attributes_preserve_recovery_and_reject_changes(native, tmp_path):
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        text=True,
    )
    assert torch.cuda.is_available(), "La prueba requiere CUDA sin fallback"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    cap = 128 * 1024**2
    torch.cuda.set_per_process_memory_fraction(
        cap / torch.cuda.get_device_properties(device).total_memory, device
    )
    torch.cuda.reset_peak_memory_stats(device)
    cpu_rng, cuda_rng = (
        torch.random.get_rng_state().clone(),
        torch.cuda.get_rng_state(device).clone(),
    )
    data = resources(tmp_path / "source")
    cpu = consumer(data, "mac_online", torch.float64, "cpu", "bank", 2, False)
    gpu = consumer(data, "mac_online", torch.float64, device, "bank", 2, False, parent=cpu[3])
    expected = walk(native, tmp_path / "cpu", cpu, recover=False)
    actual = walk(native, tmp_path / "gpu", gpu, recover=False)
    recovered = walk(native, tmp_path / "recovered", gpu, recover=True)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-9, atol=2e-10)
    assert actual[0] == recovered[0] and actual[2:] == recovered[2:]
    for first, second in zip(state_tensors(actual[1]), state_tensors(recovered[1]), strict=True):
        assert torch.equal(first, second)
    with session(native, tmp_path / "guard", gpu) as run:
        run.step(inputs(gpu, 125), [])
        before = (tmp_path / "guard/latest.json").read_bytes()
        gpu[3].predictor.price_encoder.norm.eps = 0.5
        with pytest.raises(ValueError):
            run.step(inputs(gpu, 126), [])
        assert (tmp_path / "guard/latest.json").read_bytes() == before
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    assert peak <= cap
    (tmp_path / "cuda-financial-attributes.json").write_text(
        json.dumps(
            dict(
                hardware=hardware.strip(),
                torch=torch.__version__,
                dtype="float64",
                variant="mac_online",
                K=2,
                mha_fastpath=False,
                tf32=False,
                exact_recovery=True,
                mutation_rejected_without_publication=True,
                cpu_rng_unchanged=True,
                cuda_rng_unchanged=True,
                cap_torch_bytes=cap,
                peak_torch_bytes=peak,
                max_prediction_error=float(np.max(np.abs(np.asarray(actual[0]) - expected[0]))),
                rtol=2e-9,
                atol=2e-10,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
