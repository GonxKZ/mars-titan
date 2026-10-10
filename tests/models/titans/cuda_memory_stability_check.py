"""Comprobación manual de PT1 en cuda:0, sin optimizadores ni datos del corpus.

Se ejecuta con pytest indicando la ruta del archivo dentro de una plaza `memslot gpu`. Fija
las huellas FP32 del núcleo sin PT1 capturadas en cuda:0 antes de introducir el componente,
con las proyecciones lineales y con las de la sección 4.4, contrasta PT1 activa entre CPU y
CUDA en los dos núcleos y repite las propiedades de la caja y de la cota del estado en la GPU.
Las tolerancias entre dispositivos son las de FP32 en un recorrido corto, porque CPU y CUDA
redondean de forma distinta la LayerNorm y las multiplicaciones.
"""

import json
import subprocess

import torch
from memory_stability_traces import financial_run, financial_trace, memory_run, memory_trace

from mars_titan.models.titans import MemoryConfig, MemoryStability, NeuralMemory
from mars_titan.models.titans.config import PAPER_PROJECTIONS

# Capturadas en 42e7dbca con memory_stability_traces.py en cuda:0, FP32 y sin TF32.
# Las huellas financieras se recapturaron cuando el codificador de precios pasó a calcular la
# última capa solo para el último token (perf/campaign-kernels). Ese cambio altera el redondeo
# FP32 de los cuantiles, el estado y los gradientes del predictor, no el de la memoria sola.
# La diferencia con las anteriores es del orden de 1e-7 a 6e-6 en relativo.
CORE_FP32_CUDA = dict(
    memory=dict(
        fingerprint="0b0555f9f566d5436b08c121a24ab1ec1757ef8fa14aadebef62313795836987",
        read="5d736403a4657d197fb0067b35c627f51abfc99f25f61dbbbbee61f6e8586c29",
        state="7edf31c34fb3201de6c6eb6f66a2be4d6bef6b079ed620ff91cc59dfb47c480d",
        gradients="bb1a24ff1164238fb87e3abf03bbf47e4f340164e730cc0ca43bdbc4678f2819",
    ),
    financial=dict(
        quantiles="563c10d95410b66531e4bb7fc5d1292daa025dd0c221dcbdeee69dc55e2b008a",
        state="dff88139643708a64751f9e9ea78d930f3aecaf4eb3cc949fd5321701856001d",
        gradients="0fe78edadc91dbb25b2aa6012faad749008a0cbbd4c6ea1191215f8f29a6ec89",
        unused=[],
    ),
)
# Núcleo de #474 sin PT1, con SiLU, convolución causal de núcleo 4 y L2 de q y k, capturado
# en cuda:0 sobre el código de esa rama con las mismas trazas.
PROJECTIONS_FP32_CUDA = dict(
    memory=dict(
        fingerprint="210cdd4321cc59c88c26579c4213d433e40c5c5414370262da13b60bd1bfe5e0",
        read="f09d7b128c7f36aaf1126d44b0a36e61330d904a5d30a5ce61b7370626e71a87",
        state="1881c59fedff794c7685b66813897673a9cf377c703f083261eb3cd8b02f9841",
        gradients="d3ea938babd2f6840823b86f32beb893cf5d0d4d48b703fb327218245ed293f4",
    ),
    financial=dict(
        quantiles="2550d5bc32eca1969ba4164310ed4150e694338fcb78a2c2d8b3b59c79e3aeb8",
        state="44a1a40cb5248a69bf6498665f16ade167332efb989e044c85f95ac8184ef390",
        gradients="91885d5a2e91ef3b7579a4f4b7e494b169fecbbf9a3bb66b4b2272ac6df8d1eb",
        unused=[],
    ),
)
PROJECTIONS = dict(qkv_silu=True, qkv_convolution=4)
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
    # Lo mismo con las proyecciones de la sección 4.4 frente al núcleo de #474.
    assert memory_trace("cuda:0", **PROJECTIONS) == PROJECTIONS_FP32_CUDA["memory"]
    projected = financial_trace("cuda:0", memory_projections=PAPER_PROJECTIONS)
    assert projected == PROJECTIONS_FP32_CUDA["financial"]
    stability = MemoryStability(**PT1)
    records = {}
    for name, run in (
        ("memory", lambda where: memory_run(where, PT1_GATE_BIAS, stability=stability)),
        ("financial", lambda where: financial_run(where, PT1_GATE_BIAS, memory_stability=PT1)),
        (
            "memory_with_projections",
            lambda where: memory_run(where, PT1_GATE_BIAS, stability=stability, **PROJECTIONS),
        ),
        (
            "financial_with_projections",
            lambda where: financial_run(
                where, PT1_GATE_BIAS, memory_stability=PT1, memory_projections=PAPER_PROJECTIONS
            ),
        ),
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
