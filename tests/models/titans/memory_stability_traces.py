"""Trazas FP32 del núcleo para comprobar que PT1 desactivada no cambia ningún bit.

Las mismas funciones se ejecutaron sobre el código anterior a PT1 para capturar las huellas
fijadas en las pruebas y en la comprobación CUDA. Solo calculan salidas y gradientes, sin
optimizador ni datos del corpus.
"""

import hashlib
import importlib

import numpy as np
import torch

RECIPE_GATE_BIAS = dict(alpha_half_life=256.0, eta=0.5, theta=0.05)


def _digest(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def memory_run(device="cpu", gate_bias=RECIPE_GATE_BIAS, **options):
    """Recorre la memoria de dos capas con residual sobre ocho tokens y deriva en FP32.

    Devuelve los tensores para que la comprobación CUDA pueda comparar dispositivos. Con
    la caja de PT1 hay que pasar un `gate_bias` cuya η inicial quepa en ella.
    """
    package = importlib.import_module("mars_titan.models.titans")
    config = package.MemoryConfig(
        dim=8,
        depth=2,
        max_batch=3,
        max_tokens=8,
        gate_bias=package.GateBias(**gate_bias),
        residual_layer_norm=True,
        **options,
    )
    module = package.NeuralMemory(config, device=device, dtype=torch.float32)
    generator = torch.Generator().manual_seed(11)
    tokens = (3.0 * torch.randn(3, 8, 8, generator=generator)).to(device)
    query = torch.randn(3, 2, 8, generator=generator).to(device)
    state = module.initial_state(3, differentiable=True)
    state = module.update(tokens, state, differentiable=True)
    read = module.read(query, state)
    loss = read.square().sum() + sum(w.square().sum() for w in state.weights)
    gradients = torch.autograd.grad(loss, list(module.parameters()))
    return dict(
        fingerprint=config.fingerprint(),
        read=read,
        state=(*state.weights, *state.momentum),
        gradients=gradients,
    )


def memory_trace(device="cpu", **options):
    """Huellas SHA-256 de la salida, el estado y los gradientes de `memory_run`."""
    run = memory_run(device, **options)
    return dict(
        fingerprint=run["fingerprint"],
        read=_digest([run["read"]]),
        state=_digest(run["state"]),
        gradients=_digest(run["gradients"]),
    )


def financial_run(device="cpu", gate_bias=RECIPE_GATE_BIAS, **options):
    """Recorre cuatro eventos del predictor mac_online con la receta de 2000 y deriva la pinball.

    Igual que `memory_run`, devuelve tensores y admite otro `gate_bias` para la caja de PT1.
    """
    from test_financial_adapter import DIMENSIONS, specification

    from mars_titan.models.quantile_head import pinball_loss
    from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
    from mars_titan.models.titans.financial_inputs import DecisionBatch

    config = FinancialConfig(
        specification(),
        variant="mac_online",
        hidden_size=32,
        head="quantile_head_v1",
        gate_bias=dict(gate_bias),
        memory_residual_layer_norm=True,
        **options,
    )
    model = FinancialPredictor(config, device=device, dtype=torch.float32).train()
    flows = ("US/AAA", "US/BBB", "US/CCC")
    generator = np.random.default_rng(23)
    state = model.initial_state(flows, differentiable=True)
    outputs = []
    for event in range(4):
        at = 1_609_459_200_000_000 + event * 86_400_000_000
        values = {
            name: (3.0 * generator.normal(size=(3, 64, w) if name == "prices" else (3, w))).astype(
                np.float32
            )
            for name, w in DIMENSIONS.items()
        }
        values["fundamentals"][:] = [0, 1, 0]
        values["macro"][:] = [0, 1, 0]
        raw = dict(
            inputs=values,
            presence=np.ones((3, 5), dtype=np.bool_),
            sample_ids=[f"{flow}/{at}" for flow in flows],
            market=["US"] * 3,
            prediction_at=np.full(3, at, dtype="datetime64[us]"),
            input_available_at=np.full(3, at - 1, dtype="datetime64[us]"),
        )
        batch = DecisionBatch.from_corpus(raw, config.inputs, device=device, dtype=torch.float32)
        prepared = model.prepare(batch, state, differentiable=True)
        state = prepared.next_state
        outputs.append(prepared.quantiles)
    quantiles = torch.cat(outputs)
    loss = pinball_loss(quantiles, torch.zeros(len(quantiles), device=device))
    names, parameters = zip(*model.named_parameters(), strict=True)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return dict(
        quantiles=tuple(outputs),
        state=(*state.mac.memory.weights, *state.mac.memory.momentum),
        gradients=tuple(g for g in gradients if g is not None),
        unused=sorted(n for n, g in zip(names, gradients, strict=True) if g is None),
    )


def financial_trace(device="cpu", **options):
    """Huellas SHA-256 de los cuantiles, el estado y los gradientes de `financial_run`."""
    run = financial_run(device, **options)
    return dict(
        quantiles=_digest(run["quantiles"]),
        state=_digest(run["state"]),
        gradients=_digest(run["gradients"]),
        unused=run["unused"],
    )
