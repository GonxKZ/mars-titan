"""Recorrido CUDA manual del consumidor, con fixtures y sin ajuste externo."""

import json
import resource
import subprocess
import time

import numpy as np
import pytest
import test_native_episode_backend as backend_fixtures
import torch
from test_financial_session import inputs, moment, resources, session

from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.local_control import MACProjectionConfig

native = backend_fixtures.native


def state_tensors(state):
    return (state.observed_steps,) + (
        (*state.mac.memory.weights, *state.mac.memory.momentum, state.mac.memory.steps)
        if state.mac
        else ()
    )


def consumer(data, variant, dtype, device, mode, steps, control, parent=None):
    model = FinancialPredictor(
        FinancialConfig(data[1], variant=variant, hidden_size=32),
        local_control=(
            MACProjectionConfig(mode="diagnostic", rank=1, frequency=1, grid_size=4)
            if control
            else None
        ),
        device=device,
        dtype=dtype,
    )
    model.eval().requires_grad_(False)
    readout = (
        EpisodicReadout(
            EpisodicReadoutConfig(
                data[2].fingerprint(), hidden_size=32, refinements=steps, neighbors=2, mode=mode
            ),
            device=device,
            dtype=dtype,
        )
        if mode is not None
        else None
    )
    if readout is not None:
        readout.eval().requires_grad_(False)
    if parent is not None:
        model.load_state_dict(parent.predictor.state_dict())
        if readout is not None:
            readout.load_state_dict(parent.readout.state_dict())
    return (*data[:3], FrozenFinancialConsumer(model, readout=readout), data[4])


def walk(native, directory, data, *, recover):
    run = session(native, directory, data)
    values, previous = [], None
    for index in (63, 64, 125, 126, 127, 128, 129):
        feedback = (
            [native.Feedback(previous, 0, moment(index), 0.25)] if previous is not None else []
        )
        result = run.step(
            inputs(data, index), feedback, kind="warmup" if index == 63 else "decision"
        )
        values.extend(p.value for p in result.predictions)
        previous = result.predictions[0].id if index >= 125 else None
        if recover and index == 127:
            expected = run.snapshot()
            run.close()
            run = session(native, directory, data, resume=True)
            assert run.snapshot() == expected
    manifest = run.fast_state()
    state = run._fast_store.gather(manifest, ("US/AAA",))
    assert int(state.observed_steps[0]) == 7
    assert state.observed_steps.device.type == "cpu"
    assert all(not t.requires_grad and t.grad_fn is None for t in state_tensors(state))
    closed = run.step([], [], kind="settlement", cutoff=moment(201), close_phase=True)
    assert len(closed.finalized) == 1 and not closed.applied
    records = run.retained_episodes()
    snapshot = run.snapshot()
    run.close()
    with session(native, directory, data, resume=True) as restored:
        assert restored.snapshot() == snapshot
    return values, state, records, snapshot


def test_cuda_frozen_financial_session_parity_and_recovery(native, tmp_path):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        text=True,
    )
    assert torch.cuda.is_available(), "La comprobación requiere CUDA sin fallback"
    device = torch.device("cuda:0")
    cap = 128 * 1024**2
    torch.cuda.set_per_process_memory_fraction(
        cap / torch.cuda.get_device_properties(device).total_memory, device
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    cpu_rng, cuda_rng = (
        torch.random.get_rng_state().clone(),
        torch.cuda.get_rng_state(device).clone(),
    )
    data = resources(tmp_path / "source")
    cases = [
        ("transformer_direct", None, 1, False),
        ("mac_disabled", None, 1, False),
        ("mac_frozen", None, 1, False),
        ("mac_online", None, 1, False),
        ("mac_online", "no_bank", 1, False),
        *(("mac_online", "bank", k, False) for k in (1, 2, 4)),
        ("mac_online", "bank", 1, True),
    ]
    records = []
    for dtype in (torch.float32, torch.float64):
        for index, (variant, mode, steps, control) in enumerate(cases):
            name = f"{dtype}-{index}"
            cpu = consumer(data, variant, dtype, "cpu", mode, steps, control)
            gpu = consumer(data, variant, dtype, device, mode, steps, control, parent=cpu[3])
            expected = walk(native, tmp_path / f"cpu-{name}", cpu, recover=False)
            actual = walk(native, tmp_path / f"gpu-{name}", gpu, recover=False)
            repeated = walk(native, tmp_path / f"resumed-{name}", gpu, recover=True)
            rtol, atol = (5e-5, 3e-6) if dtype == torch.float32 else (2e-9, 2e-10)
            np.testing.assert_allclose(actual[0], expected[0], rtol=rtol, atol=atol)
            assert actual[0] == repeated[0] and actual[2:] == repeated[2:]
            for left, right, recovered in zip(
                state_tensors(expected[1]),
                state_tensors(actual[1]),
                state_tensors(repeated[1]),
                strict=True,
            ):
                torch.testing.assert_close(right.cpu(), left, rtol=rtol, atol=atol)
                assert torch.equal(right, recovered)
            assert all(p.grad is None for p in gpu[3].predictor.parameters())
            with pytest.raises((ValueError, RuntimeError)):
                session(native, tmp_path / f"gpu-{name}", cpu, resume=True)
            records.append(
                dict(
                    dtype=str(dtype),
                    variant=variant,
                    readout=mode,
                    K=steps,
                    local_control=control,
                    exact_recovery=True,
                    no_graph=True,
                    rtol=rtol,
                    atol=atol,
                    max_prediction_error=float(np.max(np.abs(np.asarray(actual[0]) - expected[0]))),
                )
            )
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    assert peak <= cap
    (tmp_path / "cuda-financial-session.json").write_text(
        json.dumps(
            dict(
                records=records,
                torch=torch.__version__,
                hardware=hardware.strip(),
                tf32=False,
                mha_fastpath=False,
                cap_torch_bytes=cap,
                peak_torch_bytes=peak,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                elapsed_seconds=time.perf_counter() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                cpu_rng_unchanged=True,
                cuda_rng_unchanged=True,
                scope="Fixtures del consumidor congelado, sin corpus real ni ajuste externo.",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
