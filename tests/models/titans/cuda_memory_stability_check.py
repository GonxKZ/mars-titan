"""Comprobación manual de PT1 en cuda:0, sin optimizadores ni datos del corpus.

Se ejecuta con pytest indicando la ruta del archivo dentro de una plaza `memslot gpu`. Fija
las huellas FP32 del núcleo sin PT1 capturadas en cuda:0 antes de introducir el componente,
contrasta PT1 activa entre CPU y CUDA y repite las propiedades de la caja y de la cota del
estado en la GPU. Las tolerancias entre dispositivos son las de FP32 en un recorrido corto,
porque CPU y CUDA redondean de forma distinta la LayerNorm y las multiplicaciones.
"""

import json
import subprocess

import torch
from memory_stability_traces import financial_run, financial_trace, memory_run, memory_trace

from mars_titan.models.titans import MemoryConfig, MemoryStability, NeuralMemory

# Capturadas en 42e7dbca con memory_stability_traces.py en cuda:0, FP32 y sin TF32.
CORE_FP32_CUDA = dict(
    memory=dict(
        fingerprint="0b0555f9f566d5436b08c121a24ab1ec1757ef8fa14aadebef62313795836987",
        read="5d736403a4657d197fb0067b35c627f51abfc99f25f61dbbbbee61f6e8586c29",
        state="7edf31c34fb3201de6c6eb6f66a2be4d6bef6b079ed620ff91cc59dfb47c480d",
        gradients="bb1a24ff1164238fb87e3abf03bbf47e4f340164e730cc0ca43bdbc4678f2819",
    ),
    financial=dict(
        quantiles="1bf52fee41a66ea9c392289c8954db194c494eb12c0477476103caf2958cd603",
        state="998aa6df5c6a6d042f0313e21c0dc093b2e15387a7edc321ffb65c65967d248f",
        gradients="41e5578b95b16d47da495e96c252c653b118a62e7f4718457265e0961242bed6",
        unused=[],
    ),
)
PT1_GATE_BIAS = dict(alpha_half_life=256.0, eta=0.15, theta=0.05)
PT1 = dict(alpha_floor=0.002, eta_ceiling=0.3, gradient_clip=4.0, gate_box=True)


def _flatten(values):
    if isinstance(values, torch.Tensor):
        return [values]
    return [item for value in values for item in _flatten(value)]


def _max_difference(left, right):
    return max(
        float((a.detach().cpu() - b.detach().cpu()).abs().max())
        for a, b in zip(_flatten(left), _flatten(right), strict=True)
    )


def test_cuda_disabled_parity_and_pt1_agreement_with_cpu():
    assert torch.cuda.is_available(), "Esta comprobación requiere CUDA sin fallback"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
        text=True,
    ).strip()
    # El asignador solo acepta reiniciar sus picos después de reservar algo en el dispositivo.
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    # Sin PT1, la salida, el estado y los gradientes coinciden bit a bit con el núcleo previo.
    assert memory_trace("cuda:0") == CORE_FP32_CUDA["memory"]
    assert financial_trace("cuda:0") == CORE_FP32_CUDA["financial"]
    stability = MemoryStability(**PT1)
    records = {}
    for name, run in (
        ("memory", lambda where: memory_run(where, PT1_GATE_BIAS, stability=stability)),
        ("financial", lambda where: financial_run(where, PT1_GATE_BIAS, memory_stability=PT1)),
    ):
        cpu, first, second = run("cpu"), run("cuda:0"), run("cuda:0")
        keys = [key for key in cpu if key not in ("fingerprint", "unused")]
        # Dos ejecuciones CUDA dan los mismos bits y CPU y CUDA coinciden con tolerancia FP32.
        assert all(_max_difference(first[key], second[key]) == 0 for key in keys)
        differences = {key: _max_difference(cpu[key], first[key]) for key in keys}
        for key in keys:
            for left, right in zip(_flatten(cpu[key]), _flatten(first[key]), strict=True):
                torch.testing.assert_close(right.cpu(), left, rtol=2e-4, atol=2e-5)
        records[name] = differences
    # La caja y la cota de la Proposición 5 también se cumplen en la GPU.
    config = MemoryConfig(
        dim=8, depth=2, max_batch=4, max_tokens=8, residual_layer_norm=True, stability=stability
    )
    memory = NeuralMemory(config, device=device, dtype=torch.float32)
    generator = torch.Generator().manual_seed(29)
    tokens = (60.0 * torch.randn(4, 240, 8, generator=generator)).to(device)
    state = memory.initial_state(4)
    bound = config.theta_max * PT1["gradient_clip"] / (1 - PT1["eta_ceiling"])
    with torch.no_grad():
        token = tokens[:, 0]
        alpha, eta, _ = memory._gates(
            memory.alpha_projection(token),
            memory.eta_projection(token),
            memory.theta_projection(token),
        )
    assert (alpha >= PT1["alpha_floor"]).all() and (eta <= PT1["eta_ceiling"]).all()
    for start in range(0, 240, 8):
        state = memory.update(tokens[:, start : start + 8], state)
        for momentum in state.momentum:
            assert (momentum.norm(dim=-1) <= bound * (1 + 1e-5)).all()
    summary = dict(
        hardware=hardware,
        torch=torch.__version__,
        cpu_cuda_max_abs_difference=records,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
    )
    print(json.dumps(summary, indent=1))
