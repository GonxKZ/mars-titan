"""Comprobación técnica CUDA explícita, sin optimizadores."""

import io
import json
import resource
import subprocess
import time
from dataclasses import replace

import torch
from test_financial_adapter import raw_batch, specification

from mars_titan.models.titans import financial
from mars_titan.models.titans.local_control import MACProjectionConfig


def test_cuda_local_control_parity_gradients_and_recovery(tmp_path):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        text=True,
    )
    assert torch.cuda.is_available(), "Esta comprobación requiere CUDA y no permite fallback"
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    cap = 256 * 1024**2
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    cpu_rng, cuda_rng = (
        torch.random.get_rng_state().clone(),
        torch.cuda.get_rng_state(device).clone(),
    )

    spec = specification()
    records = []
    for dtype in (torch.float32, torch.float64):
        for mode in ("disabled", "diagnostic", "penalty"):
            config = financial.FinancialConfig(spec, variant="mac_online", hidden_size=32)
            options = MACProjectionConfig(
                mode=mode,
                rank=2,
                frequency=1,
                grid_size=13,
                threshold=0.0,
                weight=0.2 if mode == "penalty" else 0.0,
            )
            cpu = financial.FinancialPredictor(config, local_control=options, dtype=dtype)
            gpu = financial.FinancialPredictor(
                config, local_control=options, dtype=dtype, device=device
            )
            gpu.load_state_dict(cpu.state_dict())
            cpu_batch = financial.DecisionBatch.from_corpus(raw_batch(), spec, dtype=dtype).select(
                [0]
            )
            gpu_batch = financial.DecisionBatch.from_corpus(
                raw_batch(), spec, dtype=dtype, device=device
            ).select([0])

            def prepare(model, batch, state, context, differentiable):
                plan = model.local_control.select_flows(
                    batch.flow_ids, state.observed_steps, context_id=context
                )
                return model.prepare(
                    batch,
                    state,
                    control_selection=plan,
                    control_context_id=context,
                    differentiable=differentiable,
                )

            cpu_output = prepare(
                cpu, cpu_batch, cpu.initial_state(cpu_batch.flow_ids), "a" * 64, True
            )
            gpu_output = prepare(
                gpu, gpu_batch, gpu.initial_state(gpu_batch.flow_ids), "a" * 64, True
            )
            rtol, atol = (5e-4, 3e-6) if dtype == torch.float32 else (2e-8, 2e-10)
            torch.testing.assert_close(
                gpu_output.point_predictions.cpu(),
                cpu_output.point_predictions,
                rtol=rtol,
                atol=atol,
            )
            gradients = []
            for model, output in ((cpu, cpu_output), (gpu, gpu_output)):
                loss = output.point_predictions.square().sum()
                if mode == "penalty":
                    loss = loss + output.local_control.penalty
                gradients.append(
                    torch.autograd.grad(
                        loss,
                        (
                            model.price_encoder.projection.weight,
                            model.fusion[0].weight,
                            model.mac.memory.eta_projection.weight,
                        ),
                    )
                )
            maximum_gradient_error = 0.0
            for left, right in zip(*gradients, strict=True):
                torch.testing.assert_close(right.cpu(), left, rtol=rtol, atol=atol)
                maximum_gradient_error = max(
                    maximum_gradient_error, float((right.cpu() - left).abs().max())
                )
            if mode != "disabled":
                torch.testing.assert_close(
                    gpu_output.local_control.operators.cpu(),
                    cpu_output.local_control.operators,
                    rtol=rtol,
                    atol=atol,
                )
            payload = io.BytesIO()
            torch.save(gpu.export_state(gpu_output.next_state), payload)
            payload.seek(0)
            # El contrato conserva cursores en CPU y pesos rápidos en su dispositivo.
            recovered = gpu.restore_state(torch.load(payload, weights_only=True))
            following = financial.DecisionBatch.from_corpus(
                raw_batch(at=gpu_batch.prediction_at[0] + 10), spec, dtype=dtype, device=device
            ).select([0])
            expected = prepare(gpu, following, gpu_output.next_state, "b" * 64, False)
            actual = prepare(gpu, following, recovered, "b" * 64, False)
            torch.testing.assert_close(
                actual.point_predictions, expected.point_predictions, rtol=0, atol=0
            )
            for name in ("weights", "momentum"):
                for left, right in zip(
                    getattr(expected.next_state.mac.memory, name),
                    getattr(actual.next_state.mac.memory, name),
                    strict=True,
                ):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            torch.testing.assert_close(
                actual.next_state.mac.memory.steps,
                expected.next_state.mac.memory.steps,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual.next_state.observed_steps, expected.next_state.observed_steps, rtol=0, atol=0
            )
            assert actual.next_state.last_prediction_at == expected.next_state.last_prediction_at
            assert actual.next_state.last_sample_ids == expected.next_state.last_sample_ids
            incompatible = financial.FinancialPredictor(
                config, local_control=replace(options, frequency=2), dtype=dtype, device=device
            )
            try:
                incompatible.load_state_dict(gpu.state_dict(), strict=False)
            except ValueError:
                pass
            else:
                raise AssertionError("Se aceptó otro contrato C")
            records.append(
                dict(
                    mode=mode,
                    dtype=str(dtype),
                    gradient_max_absolute_error=maximum_gradient_error,
                    next_prediction_exact=True,
                    next_fast_update_exact=True,
                    incompatible_contract_rejected=True,
                )
            )
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    assert peak <= cap
    (tmp_path / "cuda-local-control.json").write_text(
        json.dumps(
            dict(
                torch=torch.__version__,
                hardware=hardware.strip(),
                records=records,
                peak_torch_bytes=peak,
                cap_torch_bytes=cap,
                cpu_rng_unchanged=True,
                cuda_rng_unchanged=True,
                peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(device),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                elapsed_seconds=time.perf_counter() - started,
                tf32=False,
                scope="Fixtures técnicos sin optimizador, datos reales ni evaluación científica.",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
