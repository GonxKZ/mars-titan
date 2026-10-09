"""Paridad numérica del núcleo Titans con implementaciones públicas de referencia.

El código de terceros no forma parte del repositorio. Se descarga aparte, en directorios
fijados por commit, y se importa desde un entorno uv aislado con `python -I`:

    <entorno>/bin/python -I benchmarks/titans_reference_parity.py \
        --lucidrains-root <titans-pytorch@commit> --fla-root <flash-linear-attention@commit> \
        --output <recibo.json>

Solo se ejecutan pasos hacia delante, las escrituras internas de la memoria y
`autograd.grad` de funcionales fijos. No se crean optimizadores ni se modifican parámetros
salvo para copiar los mismos pesos en ambas implementaciones antes de cada caso.
"""

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
import types
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from mars_titan.models.titans import GateBias, MemoryConfig, NeuralMemory  # noqa: E402
from mars_titan.models.titans.config import LAYER_NORM_EPS  # noqa: E402

LUCIDRAINS_COMMIT = "1d40c445fa1794fb28582c721786482b5b683e47"
FLA_COMMIT = "855e502613d47d8cc139153f0d87ecae5831ac3c"
THETA_MAX = 0.1
INPUT_SCALE = 0.3

# Tolerancias declaradas antes de ejecutar: |a − b| ≤ atol + rtol · |b| elemento a elemento.
TOLERANCES = {
    "float32": {"forward": (1e-5, 1e-6), "gradient": (1e-4, 1e-6)},
    "float64": {"forward": (1e-10, 1e-12), "gradient": (1e-9, 1e-12)},
}
PACKAGES = (
    "torch",
    "einops",
    "einx",
    "tensordict",
    "assoc-scan",
    "x-transformers",
    "hyper-connections",
    "rotary-embedding-torch",
    "axial-positional-embedding",
    "numpy",
)


@dataclass(frozen=True)
class Case:
    name: str
    dim: int
    depth: int
    flows: int
    tokens: int
    gate_bias: bool
    residual: bool
    seed: int


EXACT_CASES = (
    Case("v1_depth2_d16", 16, 2, 3, 24, False, False, 101),
    Case("v1_depth1_d16", 16, 1, 3, 24, False, False, 102),
    Case("gate_bias_depth2_d16", 16, 2, 3, 24, True, False, 103),
    Case("gate_bias_residual_ln_depth2_d16", 16, 2, 3, 24, True, True, 104),
    Case("gate_bias_residual_ln_depth2_d64", 64, 2, 2, 64, True, True, 105),
)
DECISION_CASE = Case("decisions_gate_bias_residual_ln_d32", 32, 2, 2, 64, True, True, 106)


# Procedencia


def git_head(root: Path) -> str:
    head = (root / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref:"):
        head = (root / ".git" / head.split(" ", 1)[1]).read_text().strip()
    return head


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_commit(root: Path, expected: str) -> str:
    actual = git_head(root)
    if actual != expected:
        raise RuntimeError(f"{root} está en {actual} y el arnés exige {expected}")
    return actual


def load_lucidrains(root: Path):
    require_commit(root, LUCIDRAINS_COMMIT)
    sys.path.insert(0, str(root))
    neural_memory = importlib.import_module("titans_pytorch.neural_memory")
    memory_models = importlib.import_module("titans_pytorch.memory_models")
    files = ("titans_pytorch/neural_memory.py", "titans_pytorch/memory_models.py")
    return neural_memory, memory_models, {name: sha256(root / name) for name in files}


def load_fla(root: Path):
    """Carga solo `fla/ops/titans` sin ejecutar el `__init__` completo de la biblioteca."""
    require_commit(root, FLA_COMMIT)
    for name in ("fla", "fla.ops", "fla.ops.titans"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    loaded = {}
    for name in ("log_impl", "naive"):
        path = root / "fla" / "ops" / "titans" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"fla.ops.titans.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    files = ("fla/ops/titans/naive.py", "fla/ops/titans/log_impl.py")
    return loaded["naive"], {name: sha256(root / name) for name in files}


# Comparación


def compare(actual, expected, dtype, kind):
    rtol, atol = TOLERANCES[str(dtype).removeprefix("torch.")][kind]
    actual, expected = actual.detach().double(), expected.detach().double()
    if actual.shape != expected.shape:
        raise ValueError(f"Formas distintas: {tuple(actual.shape)} y {tuple(expected.shape)}")
    difference = (actual - expected).abs()
    scale = expected.abs()
    ratio = (difference / (atol + rtol * scale)).max().item()
    return {
        "max_abs": difference.max().item(),
        "max_rel": (difference / scale.clamp_min(atol)).max().item(),
        "max_expected": scale.max().item(),
        "worst_tolerance_ratio": ratio,
        "rtol": rtol,
        "atol": atol,
        "passed": bool(ratio <= 1.0 and torch.isfinite(actual).all()),
    }


def relative_gap(actual, expected):
    """Distancia relativa en norma de Frobenius, para decisiones que no son equivalentes."""
    actual, expected = actual.detach().double(), expected.detach().double()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-300)).item()


# Construcción de pares con los mismos pesos


def sum_squared_loss(prediction, target):
    """ℓ = ||M(k) − v||², sumada por componentes como en la ecuación (2) del artículo."""
    return (prediction - target).pow(2).sum(dim=-1)


class ResidualLayerNorm(nn.Module):
    """M(x) = x + LN(MLP(x)) sin afinidad, la memoria `residual_layer_norm` del proyecto."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, x):
        output = self.mlp(x)
        return x + F.layer_norm(output, (output.shape[-1],), eps=LAYER_NORM_EPS)


def our_memory(case: Case, dtype, device):
    config = MemoryConfig(
        dim=case.dim,
        depth=case.depth,
        normalize_qk=False,
        theta_max=THETA_MAX,
        max_batch=max(case.flows, 1),
        max_tokens=max(case.tokens, 1),
        parameter_seed=case.seed,
        gate_bias=GateBias() if case.gate_bias else None,
        residual_layer_norm=case.residual,
    )
    memory = NeuralMemory(config, dtype=dtype, device=device)
    with torch.no_grad():
        # El artículo y la referencia usan un α escalar. Igualar las filas aísla esa forma.
        tied = memory.alpha_projection.weight[:1].clone()
        memory.alpha_projection.weight.copy_(tied.expand_as(memory.alpha_projection.weight))
    return memory


def reference_memory(
    lucidrains,
    case: Case,
    ours: NeuralMemory,
    dtype,
    *,
    batch_size=1,
    chunk_size=1,
    loss="sum",
    norm="project",
):
    neural_memory, memory_models = lucidrains
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(case.seed + 1000)
        mlp = memory_models.MemoryMLP(case.dim, case.depth, expansion_factor=1.0)
        model = ResidualLayerNorm(mlp) if case.residual and norm == "project" else mlp
        options = {}
        if loss == "sum":
            options["store_memory_loss_fn"] = sum_squared_loss
            max_lr = THETA_MAX
        else:
            # La pérdida por defecto promedia sobre D: equivale a multiplicar θ por 1/D.
            max_lr = THETA_MAX * case.dim
        reference = neural_memory.NeuralMemory(
            dim=case.dim,
            chunk_size=chunk_size,
            batch_size=batch_size,
            heads=1,
            model=model,
            default_step_transform_max_lr=max_lr,
            pre_rmsnorm=False,
            qk_rmsnorm=False,
            mem_model_norm_add_residual=case.residual and norm == "reference",
            momentum=True,
            momentum_order=1,
            **options,
        )
    reference = reference.to(device=ours.initial_weights[0].device, dtype=dtype)
    gate_pairs = (
        (ours.eta_projection, reference.to_momentum[0]),
        (ours.theta_projection, reference.to_adaptive_step[0]),
        (ours.alpha_projection, reference.to_decay_factor[0]),
    )
    with torch.no_grad():
        names = reference.memory_model_parameter_names
        for index, weight in enumerate(ours.initial_weights):
            position = names.index(next(n for n in names if n.endswith(f"weights.{index}")))
            reference.memory_model_parameters[position].copy_(weight.T.unsqueeze(0))
        reference.to_keys.weight.copy_(ours.key_projection.weight)
        reference.to_values.weight.copy_(ours.value_projection.weight)
        for mine, theirs in gate_pairs:
            theirs.weight.copy_(mine.weight[:1])
            if mine.bias is None:
                theirs.bias.zero_()
            else:
                theirs.bias.copy_(mine.bias[:1])
    return reference


def inputs(case: Case, dtype, reference):
    generator = torch.Generator().manual_seed(case.seed)
    # Escala de entrada con normas de clave cercanas a 1, donde la escritura queda acotada.
    x = INPUT_SCALE * torch.randn(
        case.flows, case.tokens, case.dim, generator=generator, dtype=torch.float64
    )
    weights = (
        torch.randn(case.flows, case.tokens, case.dim, generator=generator, dtype=torch.float64),
        [
            torch.randn(case.flows, case.dim, case.dim, generator=generator, dtype=torch.float64)
            for _ in range(case.depth)
        ],
    )
    query = reference.to_queries.weight.detach().clone()
    device = query.device
    return (
        x.to(device=device, dtype=dtype),
        query,
        (weights[0].to(device, dtype), [w.to(device, dtype) for w in weights[1]]),
    )


# Recorridos


def run_ours(ours, x, query_weight, *, differentiable):
    queries = F.linear(x, query_weight)
    state = ours.initial_state(x.shape[0], differentiable=differentiable)
    reads, trace = [], []
    for token in range(x.shape[1]):
        state = ours.update(x[:, token : token + 1], state, differentiable=differentiable)
        reads.append(ours.read(queries[:, token : token + 1], state))
        trace.append(state.weights)
    return torch.cat(reads, dim=1), state, trace


def reference_names(reference):
    """Pesos de la MLP en orden de capa. La ganancia de LayerNorm de la referencia se omite."""
    names = [name for name in reference.memory_model_parameter_names if "weights." in name]
    return sorted(names, key=lambda name: int(name.rsplit(".", 1)[1]))


def run_reference(reference, x):
    retrieved, state = reference(x)
    last_weights, last_momentum = state.states
    names = reference_names(reference)
    weights = tuple(last_weights[name].transpose(-1, -2) for name in names)
    momentum = tuple(last_momentum[name][0].transpose(-1, -2) for name in names)
    trace = tuple(state.updates[name].transpose(-1, -2) for name in names)
    return retrieved, weights, momentum, trace


def outer_functional(reads, weights, coefficients):
    read_coefficient, weight_coefficients = coefficients
    total = (reads * read_coefficient).sum()
    for weight, coefficient in zip(weights, weight_coefficients, strict=True):
        total = total + (weight * coefficient).sum()
    return total


def evaluate(ours, reference, x, query_weight, coefficients):
    """Tensores hacia delante y gradientes exteriores de ambas implementaciones."""
    depth = len(ours.initial_weights)
    x_ours = x.clone().requires_grad_(True)
    x_reference = x.clone().requires_grad_(True)
    query_ours = query_weight.clone().requires_grad_(True)
    reads, state, trace = run_ours(ours, x_ours, query_ours, differentiable=True)
    retrieved, ref_weights, ref_momentum, ref_trace = run_reference(reference, x_reference)

    mine = {"forward": {"reads": reads}, "gradient": {}}
    theirs = {"forward": {"reads": retrieved}, "gradient": {}}
    for layer in range(depth):
        mine["forward"][f"final_weights_{layer}"] = state.weights[layer]
        theirs["forward"][f"final_weights_{layer}"] = ref_weights[layer]
        mine["forward"][f"final_momentum_{layer}"] = state.momentum[layer]
        theirs["forward"][f"final_momentum_{layer}"] = ref_momentum[layer]
        mine["forward"][f"weights_every_token_{layer}"] = torch.stack(
            [weights[layer] for weights in trace], dim=1
        )
        theirs["forward"][f"weights_every_token_{layer}"] = ref_trace[layer][:, 1:]

    ours_parameters = {
        "x": x_ours,
        "query_projection": query_ours,
        "key_projection": ours.key_projection.weight,
        "value_projection": ours.value_projection.weight,
        "eta_projection": ours.eta_projection.weight,
        "theta_projection": ours.theta_projection.weight,
        "alpha_projection": ours.alpha_projection.weight,
        **{f"initial_weights_{i}": w for i, w in enumerate(ours.initial_weights)},
    }
    names = reference_names(reference)
    position = {name: index for index, name in enumerate(reference.memory_model_parameter_names)}
    reference_parameters = {
        "x": x_reference,
        "query_projection": reference.to_queries.weight,
        "key_projection": reference.to_keys.weight,
        "value_projection": reference.to_values.weight,
        "eta_projection": reference.to_momentum[0].weight,
        "theta_projection": reference.to_adaptive_step[0].weight,
        "alpha_projection": reference.to_decay_factor[0].weight,
        **{
            f"initial_weights_{i}": reference.memory_model_parameters[position[name]]
            for i, name in enumerate(names)
        },
    }
    if ours.config.gate_bias is not None:
        for name, own, other in (
            ("eta_bias", ours.eta_projection, reference.to_momentum[0]),
            ("theta_bias", ours.theta_projection, reference.to_adaptive_step[0]),
            ("alpha_bias", ours.alpha_projection, reference.to_decay_factor[0]),
        ):
            ours_parameters[name] = own.bias
            reference_parameters[name] = other.bias

    ours_gradients = torch.autograd.grad(
        outer_functional(reads, state.weights, coefficients), list(ours_parameters.values())
    )
    reference_gradients = torch.autograd.grad(
        outer_functional(retrieved, ref_weights, coefficients),
        list(reference_parameters.values()),
    )
    for name, own, other in zip(ours_parameters, ours_gradients, reference_gradients, strict=True):
        if name.startswith("alpha"):
            # Las filas de α comparten el mismo parámetro escalar de la referencia.
            own = own.sum(dim=0, keepdim=True)
        elif name.startswith("initial_weights"):
            other = other[0].T
        mine["gradient"][name] = own
        theirs["gradient"][name] = other
    return mine, theirs


def max_abs(actual, expected):
    return (actual.detach().double() - expected.detach().double()).abs().max().item()


def exact_case(lucidrains, case: Case, dtype, device):
    ours = our_memory(case, dtype, device)
    reference = reference_memory(lucidrains, case, ours, dtype)
    x, query_weight, coefficients = inputs(case, dtype, reference)
    mine, theirs = evaluate(ours, reference, x, query_weight, coefficients)
    result = {"case": asdict(case), "dtype": str(dtype)}
    for group in ("forward", "gradient"):
        result[group] = {
            name: compare(mine[group][name], theirs[group][name], dtype, group)
            for name in mine[group]
        }
    result["passed"] = all(
        entry["passed"] for group in ("forward", "gradient") for entry in result[group].values()
    )
    if dtype == torch.float32:
        # Mismos pesos y entradas FP32 evaluados en FP64: separa redondeo de semántica.
        ours64 = copy.deepcopy(ours).double()
        reference64 = reference_memory(lucidrains, case, ours64, torch.float64)
        with torch.no_grad():
            reference64.to_queries.weight.copy_(reference.to_queries.weight.double())
        coefficients64 = (coefficients[0].double(), [c.double() for c in coefficients[1]])
        mine64, theirs64 = evaluate(
            ours64, reference64, x.double(), query_weight.double(), coefficients64
        )
        result["fp32_rounding"] = {
            group: {
                name: {
                    "between_implementations_fp32": max_abs(mine[group][name], theirs[group][name]),
                    "between_implementations_fp64": max_abs(
                        mine64[group][name], theirs64[group][name]
                    ),
                    "ours_fp32_vs_fp64": max_abs(mine[group][name], mine64[group][name]),
                    "reference_fp32_vs_fp64": max_abs(theirs[group][name], theirs64[group][name]),
                }
                for name in mine[group]
            }
            for group in ("forward", "gradient")
        }
    return result


def paper_minibatch(ours, x, query_weight, block):
    """Sección 3.2: gradientes en los pesos del inicio de cada bloque, tasas por token."""
    with torch.no_grad():
        weights = tuple(w.unsqueeze(0).repeat(x.shape[0], 1, 1) for w in ours.initial_weights)
        momentum = tuple(torch.zeros_like(w) for w in weights)
        anchor = weights
        queries = F.linear(x, query_weight)
    reads = []
    for token in range(x.shape[1]):
        if token % block == 0:
            anchor = weights
        with torch.no_grad():
            observed = x[:, token]
            keys = ours.key_projection(observed)
            values = ours.value_projection(observed)
            alpha = ours.alpha_projection(observed).sigmoid().unsqueeze(-1)
            eta = ours.eta_projection(observed).sigmoid().unsqueeze(-1)
            theta = THETA_MAX * ours.theta_projection(observed).sigmoid().unsqueeze(-1)
        with torch.enable_grad():
            local = tuple(w.clone().requires_grad_(True) for w in anchor)
            residual = ours._apply_memory(keys.unsqueeze(1), local).squeeze(1) - values
            gradients = torch.autograd.grad(residual.square().sum(), local)
        with torch.no_grad():
            momentum = tuple(eta * m - theta * g for m, g in zip(momentum, gradients, strict=True))
            weights = tuple((1 - alpha) * w + s for w, s in zip(weights, momentum, strict=True))
            reads.append(ours._apply_memory(queries[:, token : token + 1], weights))
    return torch.cat(reads, dim=1), weights


def decision_cases(lucidrains, dtype, device):
    """Cuantifica decisiones de la referencia frente al recorrido secuencial del proyecto."""
    case = DECISION_CASE
    results = {}
    with torch.no_grad():
        ours = our_memory(case, dtype, device)
        exact = reference_memory(lucidrains, case, ours, dtype)
        x, query_weight, _ = inputs(case, dtype, exact)
        reads, state, _ = run_ours(ours, x, query_weight, differentiable=False)

    def gap(label, reference, *, kind="decision"):
        with torch.no_grad():
            retrieved, weights, _, _ = run_reference(reference, x)
        entry = {
            "kind": kind,
            "reads_relative_gap": relative_gap(retrieved, reads),
            "final_weights_relative_gap": [
                relative_gap(w, ours_w) for w, ours_w in zip(weights, state.weights, strict=True)
            ],
        }
        if kind == "equivalence":
            entry["reads"] = compare(retrieved, reads, dtype, "forward")
            entry["final_weights"] = [
                compare(w, ours_w, dtype, "forward")
                for w, ours_w in zip(weights, state.weights, strict=True)
            ]
            entry["passed"] = entry["reads"]["passed"] and all(
                item["passed"] for item in entry["final_weights"]
            )
        results[label] = entry
        return entry

    gap("sequential_sum_loss", exact, kind="equivalence")
    gap(
        "mean_loss_with_theta_times_dim",
        reference_memory(lucidrains, case, ours, dtype, loss="mean"),
        kind="equivalence",
    )
    gap(
        "reference_layer_norm_gain_as_fast_weight",
        reference_memory(lucidrains, case, ours, dtype, norm="reference"),
    )
    for block in (4, 16):
        reference = reference_memory(lucidrains, case, ours, dtype, batch_size=block)
        entry = gap(f"minibatch_gradients_block_{block}", reference)
        with torch.no_grad():
            retrieved, weights, _, _ = run_reference(reference, x)
        paper_reads, paper_weights = paper_minibatch(ours, x, query_weight, block)
        entry["paper_section_3_2_transcription"] = {
            "reads": compare(retrieved, paper_reads, dtype, "forward"),
            "final_weights": [
                compare(w, p, dtype, "forward") for w, p in zip(weights, paper_weights, strict=True)
            ],
        }
    gap(
        "chunk_4_shared_rates_and_minibatch_4",
        reference_memory(lucidrains, case, ours, dtype, batch_size=4, chunk_size=4),
    )
    gap(
        "default_whole_call_minibatch",
        reference_memory(lucidrains, case, ours, dtype, batch_size=None),
    )
    return {"case": asdict(case), "dtype": str(dtype), "results": results}


# flash-linear-attention: memoria lineal con LayerNorm y residual estilo TTT


def ttt_layer_norm_loss(memory, key, value, scale, shift):
    """Pérdida de fla: ||LN(k M) w + b − (v − k)||² con M orientada como en fla."""
    projected = key @ memory
    normalized = F.layer_norm(projected, (projected.shape[-1],), eps=1e-6)
    return (normalized * scale + shift - (value - key)).pow(2).sum()


def corrected_layer_norm_gradient(projected, residual_target, scale, shift, eps=1e-6):
    """Derivada de la pérdida respecto a z = kM, con la fórmula entre paréntesis de fla."""
    dim = projected.shape[-1]
    mean = projected.mean(-1, keepdim=True)
    rstd = torch.sqrt(projected.var(-1, unbiased=False, keepdim=True) + eps)
    normalized = (projected - mean) / rstd
    grad = (scale * normalized + shift - residual_target) * scale
    return (
        dim * grad
        - grad.sum(-1, keepdim=True)
        - normalized * (grad * normalized).sum(-1, keepdim=True)
    ) / (rstd * dim)


def fla_cases(naive):
    generator = torch.Generator().manual_seed(207)
    batch, heads, dim = 2, 1, 16
    results = {}

    # Un paso con θ = 1, η = 0 y α = 0 aísla la derivada: M₁ − M₀ = −θ ∇ℓ(M₀).
    memory = torch.randn(batch, heads, dim, dim, generator=generator) * 0.3
    key = F.normalize(torch.randn(batch, 1, heads, dim, generator=generator), dim=-1)
    value = torch.randn(batch, 1, heads, dim, generator=generator)
    scale = 1.0 + 0.1 * torch.randn(heads, dim, generator=generator)
    shift = 0.1 * torch.randn(heads, dim, generator=generator)
    ones = torch.ones(batch, 1, heads, 1)
    zeros = torch.zeros(batch, 1, heads, 1)
    _, updated = naive.chunk_titans_linear_ref(
        key.clone(),
        key.clone(),
        value.clone(),
        scale.clone(),
        shift.clone(),
        ones,
        zeros,
        zeros,
        chunk_size=1,
        initial_state=memory.clone(),
        output_final_state=True,
        use_chunk=False,
    )
    fla_step = updated - memory
    truth64 = []
    corrected = []
    for b in range(batch):
        local = memory[b, 0].double().requires_grad_(True)
        k = key[b, 0].double()
        v = value[b, 0].double()
        (gradient,) = torch.autograd.grad(
            ttt_layer_norm_loss(local, k, v, scale[0].double(), shift[0].double()), local
        )
        truth64.append(-gradient)
        projected = k @ memory[b, 0].double()
        v_new = corrected_layer_norm_gradient(
            projected, v - k, scale[0].double(), shift[0].double()
        )
        corrected.append(-2 * k.transpose(-1, -2) @ v_new)
    truth = torch.stack(truth64).unsqueeze(1)
    corrected = torch.stack(corrected).unsqueeze(1)
    results["one_step_gradient"] = {
        "fla_vs_autograd_relative_gap": relative_gap(fla_step, truth),
        "fla_vs_autograd": compare(fla_step.double(), truth, torch.float32, "forward"),
        "parenthesised_formula_vs_autograd": compare(corrected, truth, torch.float64, "forward"),
        "fla_norm_over_autograd_norm": (fla_step.double().norm() / truth.norm()).item(),
    }

    # Coherencia interna de fla: forma por bloques frente a su recorrido secuencial.
    tokens, block = 64, 16
    q = F.normalize(torch.randn(batch, tokens, heads, dim, generator=generator), dim=-1)
    k = F.normalize(torch.randn(batch, tokens, heads, dim, generator=generator), dim=-1)
    v = torch.randn(batch, tokens, heads, dim, generator=generator)
    theta = 0.05 * torch.rand(batch, tokens, heads, 1, generator=generator) + 0.01
    alpha = 0.05 * torch.rand(batch, tokens, heads, 1, generator=generator) + 0.001
    eta = 0.5 * torch.rand(batch, tokens, heads, 1, generator=generator) + 0.3
    h0 = torch.randn(batch, heads, dim, dim, generator=generator) * 0.1
    common = dict(chunk_size=block, initial_state=h0, output_final_state=True)
    sequential, sequential_state = naive.chunk_titans_linear_ref(
        q, k, v, scale, shift, theta, alpha, eta, use_chunk=False, **common
    )
    chunked, chunked_state = naive.chunk_titans_linear_ref(
        q, k, v, scale, shift, theta, alpha, eta, use_chunk=True, **common
    )
    results["chunked_vs_sequential"] = {
        "tokens": tokens,
        "block": block,
        "outputs_relative_gap": relative_gap(chunked, sequential),
        "final_state_relative_gap": relative_gap(chunked_state, sequential_state),
        "outputs": compare(chunked, sequential, torch.float32, "forward"),
        "final_state": compare(chunked_state, sequential_state, torch.float32, "forward"),
    }
    return results


def summary(exact):
    """Casos superados por precisión y tensores fuera de la tolerancia declarada."""
    result = {}
    for dtype in ("torch.float32", "torch.float64"):
        entries = [entry for entry in exact if entry["dtype"] == dtype]
        result[dtype] = {
            "cases": len(entries),
            "passed": sum(entry["passed"] for entry in entries),
            "outside_tolerance": [
                f"{entry['case']['name']}:{group}:{name}"
                for entry in entries
                for group in ("forward", "gradient")
                for name, value in entry[group].items()
                if not value["passed"]
            ],
        }
    return result


def provenance(args, lucidrains_files, fla_files):
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "command": " ".join(
            ["<entorno>/bin/python -I benchmarks/titans_reference_parity.py"]
            + [
                # Solo el nombre del directorio fijado por commit, sin rutas locales.
                f"--{k.replace('_', '-')} {f'<{v.name}>' if isinstance(v, Path) else v}"
                for k, v in vars(args).items()
                if k != "output"
            ]
            + ["--output <recibo>"]
        ),
        "python": sys.version.split()[0],
        "isolated_flag": bool(sys.flags.isolated),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "threads": torch.get_num_threads(),
        "versions": versions,
        "environment": {
            name: os.environ.get(name, "<sin definir>")
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        },
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "references": {
            "lucidrains/titans-pytorch": {
                "commit": LUCIDRAINS_COMMIT,
                "license": "MIT",
                "files_sha256": lucidrains_files,
            },
            "fla-org/flash-linear-attention": {
                "commit": FLA_COMMIT,
                "license": "MIT",
                "files_sha256": fla_files,
            },
        },
        "project_files_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "src/mars_titan/models/titans/neural_memory.py",
                "src/mars_titan/models/titans/config.py",
                "benchmarks/titans_reference_parity.py",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lucidrains-root", type=Path, required=True)
    parser.add_argument("--fla-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    # cuBLAS solo es determinista con un espacio de trabajo fijo, antes de crear su contexto.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó cuda:0 y no hay dispositivo CUDA visible")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    neural_memory, memory_models, lucidrains_files = load_lucidrains(args.lucidrains_root)
    naive, fla_files = load_fla(args.fla_root)
    lucidrains = (neural_memory, memory_models)
    exact = [
        exact_case(lucidrains, case, dtype, device)
        for dtype in (torch.float32, torch.float64)
        for case in EXACT_CASES
    ]
    decisions = [
        decision_cases(lucidrains, dtype, device) for dtype in (torch.float32, torch.float64)
    ]
    # La operación de fla se comprueba en CPU. Su código no forma parte del núcleo comparado.
    fla = fla_cases(naive) if device.type == "cpu" else None
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": args.device,
        "tolerances": TOLERANCES,
        "theta_max": THETA_MAX,
        "input_scale": INPUT_SCALE,
        "optimizer_steps": 0,
        "provenance": provenance(args, lucidrains_files, fla_files),
        "lucidrains_exact": exact,
        "lucidrains_exact_passed": all(entry["passed"] for entry in exact),
        "summary": summary(exact),
        "lucidrains_decisions": decisions,
        "fla": fla,
        "seconds": round(time.perf_counter() - started, 2),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        receipt["cuda"] = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "torch_cuda": torch.version.cuda,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {"exact_passed": receipt["lucidrains_exact_passed"], "seconds": receipt["seconds"]}
        )
    )


if __name__ == "__main__":
    main()
