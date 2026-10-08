"""Transporte CPU/CUDA de estados ficticios. Ejecución manual y sin optimizadores."""

import io
import json
import resource
import subprocess
import time

import pytest
import torch
from test_financial_adapter import raw_batch, specification

from mars_titan.models.titans.financial import DecisionBatch, FinancialConfig, FinancialPredictor


def state_tensors(state):
    return (state.observed_steps,) + (
        (*state.mac.memory.weights, *state.mac.memory.momentum, state.mac.memory.steps)
        if state.mac
        else ()
    )


def same_state(left, right):
    for key in ("config_id", "parameter_id", "flow_ids", "last_sample_ids", "last_prediction_at"):
        assert getattr(left, key) == getattr(right, key)
    for first, second in zip(state_tensors(left), state_tensors(right), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_cuda_state_blocks_transport_and_recovery(tmp_path, monkeypatch):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        text=True,
    )
    assert torch.cuda.is_available(), "Esta comprobación requiere CUDA sin fallback"
    device = torch.device("cuda:0")
    cap = 128 * 1024**2
    torch.cuda.set_per_process_memory_fraction(
        cap / torch.cuda.get_device_properties(device).total_memory, device
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cpu_rng, cuda_rng = (
        torch.random.get_rng_state().clone(),
        torch.cuda.get_rng_state(device).clone(),
    )
    records = []
    spec = specification()
    for dtype in (torch.float32, torch.float64):
        for variant in ("transformer_direct", "mac_disabled", "mac_frozen", "mac_online"):
            config = FinancialConfig(spec, variant=variant, hidden_size=32)
            model = FinancialPredictor(config, device=device, dtype=dtype)
            batch = DecisionBatch.from_corpus(raw_batch(), spec, device=device, dtype=dtype)
            prepared = model.prepare(
                batch, model.initial_state(batch.flow_ids), differentiable=True
            )
            buffer = io.BytesIO()
            torch.save(model.export_state_cpu(prepared.next_state), buffer)
            buffer.seek(0)
            payload = torch.load(buffer, map_location="cpu", weights_only=True)
            restored = model.restore_state(payload, device=device)
            same_state(prepared.next_state, restored)
            assert restored.observed_steps.device.type == "cpu"
            assert all(
                value.grad_fn is None and not value.requires_grad
                for value in state_tensors(restored)
            )
            if variant != "transformer_direct":
                assert all(value.device == device for value in state_tensors(restored)[1:])
                with pytest.raises(ValueError, match="dispositivo"):
                    model.restore_state(payload)

            transfers = []
            original_to = torch.Tensor.to

            def trace_to(value, *args, transfers=transfers, original_to=original_to, **kwargs):
                target = kwargs.get("device")
                if (
                    value.device.type == "cpu"
                    and target is not None
                    and torch.device(target).type == "cuda"
                ):
                    transfers.append(tuple(value.shape))
                return original_to(value, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(torch.Tensor, "to", trace_to)
                gathered = model.gather_state({"US/BBB": (payload, 1)}, ("US/BBB",))
            same_state(gathered, model.select_state(restored, ("US/BBB",)))
            assert all(shape[0] == 1 for shape in transfers)
            following = DecisionBatch.from_corpus(
                raw_batch(at=batch.prediction_at[0] + 10), spec, device=device, dtype=dtype
            ).select([1])
            expected = model.prepare(
                following, model.select_state(prepared.next_state, ("US/BBB",)), differentiable=True
            )
            actual = model.prepare(following, gathered, differentiable=True)
            torch.testing.assert_close(
                expected.point_predictions, actual.point_predictions, rtol=0, atol=0
            )
            same_state(expected.next_state, actual.next_state)
            expected_grad = torch.autograd.grad(
                expected.point_predictions.sum(), model.head.weight
            )[0]
            actual_grad = torch.autograd.grad(actual.point_predictions.sum(), model.head.weight)[0]
            torch.testing.assert_close(expected_grad, actual_grad, rtol=0, atol=0)
            records.append(
                dict(
                    variant=variant,
                    dtype=str(dtype),
                    cpu_to_cuda_shapes=transfers,
                    recovery_exact=True,
                    next_prediction_exact=True,
                    head_gradient_exact=True,
                )
            )
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    assert peak <= cap
    (tmp_path / "cuda-financial-blocks.json").write_text(
        json.dumps(
            dict(
                torch=torch.__version__,
                hardware=hardware.strip(),
                records=records,
                cap_torch_bytes=cap,
                peak_torch_bytes=peak,
                peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(device),
                elapsed_seconds=time.perf_counter() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                cpu_rng_unchanged=True,
                cuda_rng_unchanged=True,
                tf32=False,
                scope="Transporte y recuperación técnicos, sin datos reales ni optimizador.",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
