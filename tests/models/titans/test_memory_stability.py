"""Pruebas de PT1, la caja de puertas y la escritura interna recortada de la memoria.

Solo calculan salidas, estados y gradientes con lotes técnicos. No usan datos del corpus
ni pasos de optimizador.
"""

import importlib
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from memory_stability_traces import RECIPE_GATE_BIAS, financial_trace, memory_trace
from test_financial_adapter import DIMENSIONS, specification
from torch.func import functional_call
from torch.nn import functional as F

from mars_titan.models.quantile_head import pinball_loss
from mars_titan.models.titans import neural_memory as memory_module
from mars_titan.models.titans.config import CERTIFIED_GATE_BOXES

# Huellas capturadas en 42e7dbca, antes de introducir PT1, con memory_stability_traces.py en
# CPU, FP32, OMP_NUM_THREADS=2 y MKL_NUM_THREADS=2. La comprobación CUDA fija las de cuda:0.
CORE_FP32_CPU = dict(
    memory=dict(
        fingerprint="0b0555f9f566d5436b08c121a24ab1ec1757ef8fa14aadebef62313795836987",
        read="7502caa0e03c7f8b4bce23e89c494a93071f35534f5fd95e963ca64800e6b6e2",
        state="27a44d9f44ac935f1021250a6a3c40fa77fd130daeb850c848c15bb7e2d879e2",
        gradients="6e555c7d0fdf3d18a927702bd92a0a703e3daa589bd17576ca0edb627b0fd6bf",
    ),
    financial=dict(
        quantiles="51076492e02ed62e93d81f63edcf883c9647f8815f7399bc1f07451668ee55c2",
        state="bcb3a940e7f5bbbc27b1491518281e2b3e93c9a02e8d85699e469ecfbf1bb619",
        gradients="f926745c6e3d39a7f5f092dfd6283f1ed8a8d78884390fe43b4089109f4fc2f7",
        unused=[],
    ),
)
GATES = ("alpha_projection", "eta_projection", "theta_projection")
DAY = 86_400_000_000
START = 1_609_459_200_000_000


def api():
    return importlib.import_module("mars_titan.models.titans")


def financial():
    return importlib.import_module("mars_titan.models.titans.financial")


def stability(**options):
    return api().MemoryStability(**options)


def memory(*, dim=4, depth=2, residual=True, dtype=torch.float64, gate_bias=None, **options):
    package = api()
    config = package.MemoryConfig(
        dim=dim,
        depth=depth,
        max_batch=3,
        max_tokens=8,
        residual_layer_norm=residual,
        gate_bias=gate_bias,
        **options,
    )
    return package.NeuralMemory(config, dtype=dtype)


def predictor(*, variant="mac_online", seed=42, dtype=torch.float64, **options):
    module = financial()
    config = module.FinancialConfig(
        specification(),
        variant=variant,
        hidden_size=32,
        seed=seed,
        head="quantile_head_v1",
        gate_bias=dict(alpha_half_life=256.0, eta=0.15, theta=0.05),
        memory_residual_layer_norm=True,
        **options,
    )
    return module.FinancialPredictor(config, dtype=dtype)


def rates(model, token):
    """Claves, valores y tasas de un token como los calcula `update` sin convolución."""
    keys = model.key_projection(token)
    if model.config.normalize_qk:
        keys = F.normalize(keys, dim=-1, eps=1e-12)
    gates = model._gates(
        model.alpha_projection(token), model.eta_projection(token), model.theta_projection(token)
    )
    return keys, model.value_projection(token), *gates


def pt1(**overrides):
    values = dict(alpha_floor=0.002, eta_ceiling=0.3, gradient_clip=0.5, gate_box=True)
    return values | overrides


def decision(config, flows, event, *, scale=3.0, seed=23, perturb=None, dtype=torch.float64):
    """Construye el lote técnico de un evento con entradas aleatorias fijadas por la semilla.

    Con `perturb`, las entradas de ese flujo se desplazan para comprobar la causalidad y el
    aislamiento entre flujos.
    """
    from mars_titan.models.titans.financial_inputs import DecisionBatch

    generator = np.random.default_rng(seed + event)
    size = len(flows)
    values = {
        name: (
            scale * generator.normal(size=(size, 64, w) if name == "prices" else (size, w))
        ).astype(np.float32)
        for name, w in DIMENSIONS.items()
    }
    values["fundamentals"][:] = [0, 1, 0]
    values["macro"][:] = [0, 1, 0]
    if perturb is not None:
        values["prices"][flows.index(perturb)] += 7.0
        values["news"][flows.index(perturb)] -= 5.0
    at = START + event * DAY
    raw = dict(
        inputs=values,
        presence=np.ones((size, 5), dtype=np.bool_),
        sample_ids=[f"{flow}/{at}" for flow in flows],
        market=["US"] * size,
        prediction_at=np.full(size, at, dtype="datetime64[us]"),
        input_available_at=np.full(size, at - 1, dtype="datetime64[us]"),
    )
    return DecisionBatch.from_corpus(raw, config.inputs, dtype=dtype)


def run(model, flows, events, *, perturb_event=None, perturb_flow=None, state=None):
    state = model.initial_state(flows) if state is None else state
    outputs, states = [], []
    for event in events:
        flow = perturb_flow if event == perturb_event else None
        batch = decision(model.config, flows, event, perturb=flow, dtype=model.head.weight.dtype)
        prepared = model.prepare(batch, state)
        state = prepared.next_state
        outputs.append(prepared.quantiles)
        states.append(state)
    return outputs, states


def test_disabled_component_keeps_fp32_outputs_gradients_and_state_bit_for_bit():
    assert memory_trace() == CORE_FP32_CPU["memory"]
    assert financial_trace() == CORE_FP32_CPU["financial"]
    assert "stability" not in api().MemoryConfig(dim=4).identity()
    config = financial().FinancialConfig(specification(), variant="mac_online")
    assert "memory_stability" not in config.identity()


def test_declared_component_is_a_new_identity_with_the_same_parameter_draws():
    rng = torch.get_rng_state().clone()
    base = memory(depth=2)
    declared = memory(depth=2, stability=stability(gradient_clip=0.5))
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert declared.config.fingerprint() != base.config.fingerprint()
    assert declared.config.identity()["stability"] == {
        "alpha_floor": 0.002,
        "eta_ceiling": 0.3,
        "gradient_clip": 0.5,
        "gate_box": True,
        "gate_map": "alpha_floor_plus_scaled_sigmoid_and_eta_scaled_sigmoid_v1",
        "clip_rule": "inner_gradient_row_l2_before_momentum_v1",
        "source": "post_titans_pt1_v1",
    }
    left, right = base.state_dict(), declared.state_dict()
    assert set(left) == set(right)
    for name, value in left.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(right[name], value, rtol=0, atol=0)
    clip_only = stability(gate_box=False, gradient_clip=1.0).identity()
    assert clip_only["gate_map"] == "unchanged"
    box_only = stability().identity()
    assert box_only["clip_rule"] == "none" and box_only["gradient_clip"] is None


@pytest.mark.parametrize(
    "options",
    [
        dict(alpha_floor=0.0),
        dict(alpha_floor=1.0),
        dict(eta_ceiling=0.0),
        dict(eta_ceiling=1.0),
        dict(gradient_clip=0.0),
        dict(gradient_clip=-1.0),
        dict(gradient_clip=math.inf),
        dict(gate_box=False),
        dict(gate_box=1),
        dict(alpha_floor=math.nan),
    ],
)
def test_invalid_declarations_fail_before_building_a_module(options):
    with pytest.raises(ValueError):
        stability(**options)


def test_certified_box_lookup_follows_the_certificate():
    assert CERTIFIED_GATE_BOXES == {"pt1": (0.002, 0.3, 0.1), "b2": (0.01, 0.5, 0.1)}
    assert stability().certified_box(0.1) == "pt1"
    assert stability(alpha_floor=0.01, eta_ceiling=0.5).certified_box(0.1) == "b2"
    assert stability(alpha_floor=0.003, eta_ceiling=0.2).certified_box(0.05) == "pt1"
    # La inicialización de la campaña (α ≈ 0,0027 con η = 0,5) queda fuera de las dos cajas.
    assert stability(alpha_floor=0.0027, eta_ceiling=0.5).certified_box(0.1) is None
    assert stability().certified_box(0.2) is None
    assert stability(gate_box=False, gradient_clip=1.0).certified_box(0.1) is None


@pytest.mark.parametrize("scale", [1.0, 1e3])
def test_gates_stay_inside_the_box_for_any_input(scale):
    box = stability(alpha_floor=0.002, eta_ceiling=0.3)
    model = memory(depth=2, stability=box)
    plain = memory(depth=2)
    generator = torch.Generator().manual_seed(5)
    token = scale * torch.randn(64, 4, generator=generator, dtype=torch.float64)
    with torch.no_grad():
        _, _, alpha, eta, theta = rates(model, token)
        _, _, plain_alpha, plain_eta, _ = rates(plain, token)
    assert alpha.shape == (64, 4, 1) and eta.shape == theta.shape == (64, 1, 1)
    assert alpha.dtype == eta.dtype == theta.dtype == torch.float64
    assert (alpha >= 0.002).all() and (alpha <= 1).all()
    assert (eta >= 0).all() and (eta <= 0.3).all()
    assert (theta >= 0).all() and (theta <= model.config.theta_max).all()
    if scale > 1:
        # Sin caja, las mismas entradas salen de ella, así que la prueba distingue los mapas.
        assert (plain_alpha < 0.002).any() and (plain_eta > 0.3).any()
    # Con solo el recorte declarado las puertas son exactamente las del núcleo.
    clip_only = memory(depth=2, stability=stability(gate_box=False, gradient_clip=1.0))
    with torch.no_grad():
        declared = rates(clip_only, token)
        reference = rates(plain, token)
    assert all(torch.equal(left, right) for left, right in zip(declared, reference, strict=True))


def test_box_bias_reproduces_the_declared_initial_rates_and_rejects_impossible_ones():
    bias = api().GateBias(alpha_half_life=256.0, eta=0.15, theta=0.05)
    model = memory(depth=2, gate_bias=bias, stability=stability())
    with torch.no_grad():
        _, _, alpha, eta, theta = rates(model, torch.zeros(1, 4, dtype=torch.float64))
    expected = -math.expm1(-math.log(2) / 256)
    torch.testing.assert_close(alpha, torch.full_like(alpha, expected), rtol=1e-12, atol=0)
    torch.testing.assert_close(eta, torch.full_like(eta, 0.15), rtol=1e-12, atol=0)
    torch.testing.assert_close(theta, torch.full_like(theta, 0.05), rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="eta_ceiling"):
        memory(gate_bias=api().GateBias(**RECIPE_GATE_BIAS), stability=stability())
    with pytest.raises(ValueError, match="alpha_floor"):
        memory(gate_bias=api().GateBias(alpha_half_life=1000.0, eta=0.15), stability=stability())
    # Con solo el recorte, el bias de la receta no cambia.
    clip_only = stability(gate_box=False, gradient_clip=1.0)
    assert api().GateBias(**RECIPE_GATE_BIAS).logits(0.1, clip_only) == api().GateBias(
        **RECIPE_GATE_BIAS
    ).logits(0.1)


def test_row_clip_scales_long_rows_to_the_limit_and_leaves_the_rest_exactly():
    rows = torch.tensor(
        [[[3.0, 4.0, 0.0], [0.3, 0.4, 0.0], [0.0, 0.0, 0.0], [0.6, 0.8, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    clipped = memory_module.clip_rows(rows, 1.0)
    torch.testing.assert_close(clipped[0, 0], torch.tensor([0.6, 0.8, 0.0], dtype=torch.float64))
    assert torch.equal(clipped[0, 1:], rows[0, 1:])
    clipped.sum().backward()
    assert torch.isfinite(rows.grad).all()
    random = torch.randn(5, 7, 7, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    result = memory_module.clip_rows(random, 0.8)
    norms, original = result.norm(dim=-1), random.norm(dim=-1)
    assert (norms <= 0.8 * (1 + 1e-12)).all()
    torch.testing.assert_close(norms[original > 0.8], torch.full_like(norms[original > 0.8], 0.8))
    cosine = F.cosine_similarity(result, random, dim=-1)
    torch.testing.assert_close(cosine, torch.ones_like(cosine))


def test_depth_one_clip_is_the_gradient_of_a_huber_loss_with_half_the_limit():
    limit = 0.3
    model = memory(dim=4, depth=1, residual=False, stability=stability(gradient_clip=limit))
    generator = torch.Generator().manual_seed(9)
    state = model.initial_state(3)
    state = replace(
        state,
        weights=(2.0 * torch.randn(3, 4, 4, generator=generator, dtype=torch.float64),),
        momentum=(torch.randn(3, 4, 4, generator=generator, dtype=torch.float64),),
    )
    token = torch.randn(3, 1, 4, generator=generator, dtype=torch.float64)
    updated = model.update(token, state)
    with torch.no_grad():
        keys, values, alpha, eta, theta = rates(model, token[:, 0])
    weight = state.weights[0].clone().requires_grad_(True)
    residual = torch.bmm(weight, keys.unsqueeze(-1)).squeeze(-1) - values
    clipped = (residual.abs() > limit / 2).any()
    # ∇_W 2·Huber_{G/2}(Wk − v) = G·sign(r)·k si |r| > G/2 y 2r·k si no, con ‖k‖ = 1.
    huber = 2 * F.huber_loss(residual, torch.zeros_like(residual), delta=limit / 2, reduction="sum")
    (gradient,) = torch.autograd.grad(huber, weight)
    momentum = eta * state.momentum[0] - theta * gradient
    expected = (1 - alpha) * state.weights[0] + momentum
    assert clipped
    torch.testing.assert_close(updated.momentum[0], momentum, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(updated.weights[0], expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("depth", [1, 2])
def test_state_stays_inside_the_bound_of_proposition_5(depth):
    limit, floor, ceiling = 0.5, 0.002, 0.3
    declared = stability(alpha_floor=floor, eta_ceiling=ceiling, gradient_clip=limit)
    model = memory(dim=4, depth=depth, residual=depth == 2, stability=declared)
    plain = memory(dim=4, depth=depth, residual=depth == 2)
    momentum_bound = model.config.theta_max * limit / (1 - ceiling)
    generator = torch.Generator().manual_seed(13)
    tokens = 40.0 * torch.randn(3, 300, 4, generator=generator, dtype=torch.float64)
    state, free = model.initial_state(3), plain.initial_state(3)
    initial = [w.norm(dim=-1) for w in state.weights]
    weight_bound = [
        torch.clamp(norms, min=momentum_bound / floor) * (1 + 1e-12) for norms in initial
    ]
    exceeded = False
    for start in range(0, 300, 6):
        chunk = tokens[:, start : start + 6]
        state, free = model.update(chunk, state), plain.update(chunk, free)
        for layer in range(depth):
            assert (state.momentum[layer].norm(dim=-1) <= momentum_bound * (1 + 1e-12)).all()
            assert (state.weights[layer].norm(dim=-1) <= weight_bound[layer]).all()
            exceeded |= bool((free.momentum[layer].norm(dim=-1) > momentum_bound).any())
    # Sin PT1 las mismas entradas superan la cota, así que la prueba detectaría que se
    # retirara el recorte.
    assert exceeded


@pytest.mark.parametrize("depth", [1, 2])
def test_gradients_match_finite_differences_in_fp64(depth, monkeypatch):
    limit = 1.0
    model = memory(
        dim=3, depth=depth, residual=depth == 2, stability=stability(gradient_clip=limit)
    )
    generator = torch.Generator().manual_seed(17)
    tokens = torch.randn(2, 3, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    query = torch.randn(2, 2, 3, generator=generator, dtype=torch.float64)
    names = [f"{gate}.weight" for gate in GATES]
    parameters = dict(model.named_parameters())
    gates = tuple(parameters[name].detach().clone().requires_grad_(True) for name in names)

    def function(values, *weights):
        overrides = dict(zip(names, weights, strict=True))
        state = model.initial_state(2, differentiable=True)
        state = functional_call(model, overrides, (values, state), dict(differentiable=True))
        return model.read(query, state)

    model.forward = model.update
    squared_norms = []
    clip_rows = memory_module.clip_rows

    def recording(gradient, bound):
        squared_norms.append(gradient.detach().square().sum(dim=-1).flatten())
        return clip_rows(gradient, bound)

    monkeypatch.setattr(memory_module, "clip_rows", recording)
    function(tokens, *gates)
    norms = torch.cat(squared_norms).sqrt()
    # El recorte actúa en unas filas y no en otras, lejos del punto no derivable.
    assert (norms > limit).any() and (norms < limit).any()
    assert ((norms - limit).abs() > 1e-4).all()
    assert torch.autograd.gradcheck(function, (tokens, *gates), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_future_perturbation_does_not_change_past_outputs_or_states():
    model = predictor(memory_stability=pt1())
    flows = ("US/AAA", "US/BBB", "US/CCC")
    outputs, states = run(model, flows, range(4))
    perturbed, perturbed_states = run(
        model, flows, range(4), perturb_event=3, perturb_flow="US/BBB"
    )
    for event in range(3):
        assert torch.equal(outputs[event], perturbed[event])
        for left, right in zip(
            states[event].mac.memory.weights,
            perturbed_states[event].mac.memory.weights,
            strict=True,
        ):
            assert torch.equal(left, right)
    assert not torch.equal(outputs[3], perturbed[3])


def test_flows_do_not_share_state_and_sessions_are_independent():
    model = predictor(memory_stability=pt1())
    flows = ("US/AAA", "US/BBB", "US/CCC")
    outputs, states = run(model, flows, range(4))
    perturbed, perturbed_states = run(
        model, flows, range(4), perturb_event=0, perturb_flow="US/BBB"
    )
    for event in range(4):
        assert torch.equal(outputs[event][[0, 2]], perturbed[event][[0, 2]])
        assert not torch.equal(outputs[event][1], perturbed[event][1])
        for left, right in zip(
            states[event].mac.memory.momentum,
            perturbed_states[event].mac.memory.momentum,
            strict=True,
        ):
            assert torch.equal(left[[0, 2]], right[[0, 2]])
    # Otra sesión intercalada con la primera no cambia ninguna de las dos.
    first = model.initial_state(flows)
    second = model.initial_state(flows)
    interleaved = []
    for event in range(4):
        batch = decision(model.config, flows, event, dtype=torch.float64)
        first = model.prepare(batch, first).next_state
        other = decision(model.config, flows, event, seed=91, dtype=torch.float64)
        second = model.prepare(other, second).next_state
        interleaved.append(first)
    for left, right in zip(
        interleaved[-1].mac.memory.weights, states[-1].mac.memory.weights, strict=True
    ):
        assert torch.equal(left, right)


def test_recovery_from_an_exported_state_continues_bit_for_bit():
    model = predictor(memory_stability=pt1(), dtype=torch.float32)
    flows = ("US/AAA", "US/BBB")
    outputs, states = run(model, flows, range(5))
    payload = model.export_state_cpu(states[1])
    restored = model.restore_state(payload, device="cpu")
    resumed, _ = run(model, flows, range(2, 5), state=restored)
    for left, right in zip(outputs[2:], resumed, strict=True):
        assert torch.equal(left, right)
    plain = predictor(dtype=torch.float32)
    with pytest.raises(ValueError):
        plain.restore_state(payload, device="cpu")
    with pytest.raises(ValueError):
        plain.load_state_dict(model.state_dict())
    other = predictor(memory_stability=pt1(gradient_clip=1.0), dtype=torch.float32)
    with pytest.raises(ValueError):
        other.load_state_dict(model.state_dict())
    memory_payload = model.mac.memory.export_state(states[1].mac.memory)
    assert memory_payload["configuration"]["stability"]["gradient_clip"] == 0.5
    with pytest.raises(ValueError):
        plain.mac.memory.restore_state(memory_payload)


def test_recipe_declaration_pairing_and_quantile_backward_with_the_component():
    module = financial()
    with pytest.raises(ValueError, match="memory_stability"):
        module.FinancialConfig(specification(), memory_stability=dict(alpha_floor=0.002))
    with pytest.raises(ValueError, match="eta_ceiling"):
        module.FinancialConfig(
            specification(), gate_bias=dict(RECIPE_GATE_BIAS), memory_stability=pt1()
        )
    online = predictor(memory_stability=pt1())
    assert online.config.identity()["memory_stability"] == pt1()
    disabled = predictor(variant="mac_disabled", seed=7, memory_stability=pt1())
    receipt = module.copy_paired_parameters(online, disabled)
    assert receipt["runtime_state_transferred"] is False
    with pytest.raises(ValueError):
        module.copy_paired_parameters(online, predictor(variant="mac_disabled"))
    model = predictor(memory_stability=pt1()).train()
    flows = ("US/AAA", "US/BBB")
    state = model.initial_state(flows, differentiable=True)
    quantiles = []
    for event in range(3):
        batch = decision(model.config, flows, event, dtype=torch.float64)
        prepared = model.prepare(batch, state, differentiable=True)
        state = prepared.next_state
        quantiles.append(prepared.quantiles)
    loss = pinball_loss(torch.cat(quantiles), torch.zeros(6, dtype=torch.float64))
    loss.backward()
    gates = [getattr(model.mac.memory, name).weight.grad for name in GATES]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gates)


def test_comparison_arm_is_declared_with_its_control_and_rule_but_not_executed():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    declaration = json.loads(
        (root / "configs/evaluation/titans-memory-stability-comparison.json").read_text()
    )
    recipe = json.loads((root / declaration["base_recipe"]).read_text())
    assert declaration["status"] == "declared_not_executed"
    assert declaration["final_test_opened"] is False and declaration["executions"] == 0
    roles = {name: arm["role"] for name, arm in declaration["arms"].items()}
    assert sorted(roles.values()) == ["control", "innovation", "trivial_alternative"]
    base = {key: value for key, value in recipe["predictor"].items() if key != "dtype"}
    identities = {}
    for arm in declaration["arms"].values():
        config = financial().FinancialConfig(
            specification(), variant=declaration["variant"], **(base | arm["predictor_changes"])
        )
        identities[arm["role"]] = config.identity()
    control, innovation, trivial = (
        identities[role] for role in ("control", "innovation", "trivial_alternative")
    )
    assert "memory_stability" not in control and "memory_stability" not in trivial
    # PT1 solo cambia la inicialización que exige la caja y el propio componente, y la
    # alternativa trivial comparte esa inicialización sin caja ni recorte.
    assert {key for key in innovation if innovation[key] != control.get(key)} == {
        "memory_gate_bias",
        "memory_stability",
    }
    assert {key for key in trivial if trivial[key] != control.get(key)} == {"memory_gate_bias"}
    assert innovation["memory_gate_bias"] == trivial["memory_gate_bias"]
    rule = declaration["decision_rule"]
    assert rule["keep_if"][0].endswith("<= 0.002") and rule["abandon_if"][0].endswith("> 0.005")
    assert declaration["pilot"]["gpu_hours_max"] <= 6
