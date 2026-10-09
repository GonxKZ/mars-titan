"""Paridad técnica CPU/CUDA de la GRU histórica, sin optimizador ni corpus real."""

import argparse
import gc
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
from test_historical_inputs import raw_batch, specification

from mars_titan.memory.native_backend import load_native
from mars_titan.models.candidate.input_adapter import CandidateInputAdapter
from mars_titan.models.titans.financial_inputs import validated_cpu_batch

LIMIT = 128 * 1024**2


def check_case(dtype, refinements):
    spec = specification()
    before_cpu = torch.random.get_rng_state().clone()
    before_gpu = torch.cuda.get_rng_state(0).clone()
    cpu = CandidateInputAdapter(spec, dtype=dtype, device="cpu")
    gpu = CandidateInputAdapter.restore(cpu.export_state(), spec, device="cuda:0")
    batch = validated_cpu_batch(raw_batch(), spec)
    expected = cpu.forward(batch, refinements=refinements)
    actual = gpu.forward(batch, refinements=refinements)
    rtol, atol = (1e-8, 1e-10) if dtype == torch.float64 else (2e-4, 2e-6)
    comparisons = [
        (actual.quantiles, expected.quantiles),
        (actual.native.state, expected.native.state),
        (actual.native.encoded.fused, expected.native.encoded.fused),
        (actual.native.encoded.episode_keys, expected.native.encoded.episode_keys),
        (actual.native.encoded.episode_features, expected.native.encoded.episode_features),
    ]
    largest = 0.0
    for observed, reference in comparisons:
        observed = observed.cpu()
        torch.testing.assert_close(observed, reference, rtol=rtol, atol=atol)
        largest = max(largest, float((observed - reference).abs().max().detach()))
    expected.median.sum().backward()
    actual.median.sum().backward()
    gradients = 0
    for name, parameter in cpu.model.named_parameters().items():
        observed = gpu.model.named_parameters()[name].grad
        if parameter.grad is None:
            assert observed is None
        else:
            torch.testing.assert_close(observed.cpu(), parameter.grad, rtol=rtol, atol=atol)
            gradients += 1
    saved = gpu.export_state()
    restored = CandidateInputAdapter.restore(saved, spec, device="cuda:0")
    recovered = restored.forward(batch, refinements=refinements)
    assert torch.equal(recovered.quantiles, actual.quantiles)
    assert torch.equal(before_cpu, torch.random.get_rng_state())
    assert torch.equal(before_gpu, torch.cuda.get_rng_state(0))
    with torch.inference_mode():
        inferred = gpu.forward(batch, refinements=refinements)
    torch.testing.assert_close(inferred.quantiles, actual.quantiles, rtol=rtol, atol=atol)
    try:
        CandidateInputAdapter.restore(saved, specification(catalog="cash"), device="cuda:0")
    except ValueError:
        pass
    else:
        raise AssertionError("Se recuperó otro catálogo")
    return dict(
        dtype=str(dtype),
        refinements=refinements,
        rows=2,
        bank="empty",
        rtol=rtol,
        atol=atol,
        max_absolute_error=largest,
        gradients_compared=gradients,
        recovery_exact=True,
        rng_unchanged=True,
        incompatible_catalog_rejected=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("El recibo de destino ya existe")
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible, no se ejecuta una alternativa CPU")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free < 256 * 1024**2:
        raise RuntimeError("La sonda necesita al menos 256 MiB libres tras iniciar el contexto")
    torch.cuda.set_per_process_memory_fraction(LIMIT / total, device)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats(device)
    results = []
    for dtype in (torch.float32, torch.float64):
        for count in (1, 2, 4):
            results.append(check_case(dtype, count))
            gc.collect()
            torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_reserved(device)
    assert peak <= LIMIT
    report = dict(
        schema_version=1,
        created_at=datetime.now(UTC).isoformat(),
        torch=str(torch.__version__),
        device=str(device),
        gpu_inventory=inventory,
        initial_free_bytes=free,
        allocator_limit_bytes=LIMIT,
        peak_reserved_bytes=peak,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_binary_sha256=load_native().binary_sha256,
        optimizer_steps=0,
        cases=results,
    )
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(cases=len(results), peak_reserved_bytes=peak, optimizer_steps=0)))


if __name__ == "__main__":
    main()
