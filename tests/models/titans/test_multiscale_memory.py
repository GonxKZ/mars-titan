"""PT5: cascada de Benna y Fusi sobre los pesos rápidos y su banco EMA, sin optimizadores.

Las pruebas calculan lecturas, estados, Jacobianos y gradientes con tokens técnicos. No usan
datos del corpus ni pasos de ajuste.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_financial_adapter import specification
from test_memory_stability import decision, financial, memory, pt1

from mars_titan.models.titans.config import (
    MULTISCALE_CAPACITIES,
    MACConfig,
    MemoryConfig,
    MemoryMultiscale,
    MemoryStability,
)
from mars_titan.models.titans.mac import TitansMAC
from mars_titan.models.titans.transition_jacobian import (
    fast_state_dimension,
    fast_state_jacobian,
)

ROOT = Path(__file__).resolve().parents[3]
DECLARATION = ROOT / "configs/evaluation/multiscale-memory-comparison.json"
FORMS = ("cascade", "ema_bank")


def multiscale(form="cascade", coupling=1 / 16):
    return MemoryMultiscale(form=form, coupling=coupling)


def tokens(count, *, batch=2, dim=4, scale=1.0, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return scale * torch.randn(batch, count, dim, generator=generator, dtype=torch.float64)


def scrambled(model, state, seed=9):
    """El mismo estado con niveles distintos entre sí, para que el acoplamiento actúe."""
    generator = torch.Generator().manual_seed(seed)
    levels = tuple(
        level + 0.2 * torch.randn(level.shape, generator=generator, dtype=level.dtype)
        for level in state.levels
    )
    momentum = tuple(
        0.1 * torch.randn(value.shape, generator=generator, dtype=value.dtype)
        for value in state.momentum
    )
    result = replace(state, levels=levels, momentum=momentum)
    model.validate_state(result)
    return result


def captured_alpha(model, monkeypatch):
    seen = []
    original = model._gates

    def gates(*arguments):
        result = original(*arguments)
        seen.append(result[0].detach().clone())
        return result

    monkeypatch.setattr(model, "_gates", gates)
    return seen


def test_disabled_memory_keeps_no_levels_and_the_previous_identity():
    control = memory()
    assert control.config.hidden_levels == 0
    assert "multiscale" not in control.config.identity()
    state = control.initial_state(2)
    assert state.levels == ()
    assert control.export_state(state).keys() == {
        "schema_version",
        "configuration",
        "weights",
        "momentum",
        "steps",
    }


@pytest.mark.parametrize("form", FORMS)
def test_both_forms_start_reading_the_control_weights_with_the_same_draws(form):
    control, model = memory(), memory(multiscale=multiscale(form))
    for name, value in control.state_dict().items():
        if name != "_extra_state":
            assert torch.equal(model.state_dict()[name], value), name
    state = model.initial_state(2)
    assert len(state.levels) == 2 * 2
    for level, weight in zip(
        state.levels, (w for w in state.weights for _ in range(2)), strict=True
    ):
        assert torch.equal(level, weight)
    query = tokens(1)
    torch.testing.assert_close(
        model.read(query, state), control.read(query, control.initial_state(2)), rtol=0, atol=0
    )
    assert model.config.identity()["multiscale"]["form"] == form


def test_cascade_first_write_equals_the_control_because_levels_start_equal():
    control, model = memory(), memory(multiscale=multiscale())
    token = tokens(1)
    expected = control.update(token, control.initial_state(2))
    actual = model.update(token, model.initial_state(2))
    for left, right in zip(actual.weights, expected.weights, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(actual.momentum, expected.momentum, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_the_cascade_reads_only_its_first_level():
    model, control = memory(multiscale=multiscale()), memory()
    state = scrambled(model, model.initial_state(2))
    assert model.readable_weights(state) is state.weights
    query = tokens(1)
    expected = control.read(query, replace(control.initial_state(2), weights=state.weights))
    torch.testing.assert_close(model.read(query, state), expected, rtol=0, atol=0)


def test_cascade_follows_the_euler_equations_with_titans_forgetting(monkeypatch):
    model = memory(multiscale=multiscale())
    state = scrambled(model, model.initial_state(2))
    alphas = captured_alpha(model, monkeypatch)
    result = model.update(tokens(1), state)
    (alpha,) = alphas
    g12, g23, g34 = model.config.multiscale.conductances
    for layer in range(2):
        u1, u2, u3 = state.weights[layer], state.levels[2 * layer], state.levels[2 * layer + 1]
        surprise = result.momentum[layer]
        expected = (
            (1 - alpha) * u1 + surprise + g12 * (u2 - u1),
            u2 + (g12 * (u1 - u2) + g23 * (u3 - u2)) / 2,
            u3 + (g23 * (u2 - u3) - g34 * u3) / 4,
        )
        actual = (result.weights[layer], result.levels[2 * layer], result.levels[2 * layer + 1])
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right, rtol=1e-13, atol=1e-13)


def test_bank_reads_the_mean_takes_the_gradient_there_and_forgets_each_level_alone(monkeypatch):
    model = memory(multiscale=multiscale("ema_bank"))
    state = scrambled(model, model.initial_state(2))
    readable = model.readable_weights(state)
    for layer in range(2):
        weight, second, third = state.weights[layer], *state.levels[2 * layer : 2 * layer + 2]
        torch.testing.assert_close(
            readable[layer], weight + ((second - weight) + (third - weight)) / 3, rtol=0, atol=0
        )
        # Es la media de los tres niveles salvo el redondeo del orden de las sumas.
        torch.testing.assert_close(readable[layer], (weight + second + third) / 3)
    # El control con los pesos de la media y el mismo momentum calcula la misma escritura.
    control = memory()
    reference = replace(control.initial_state(2), weights=readable, momentum=state.momentum)
    token = tokens(1)
    expected = control.update(token, reference)
    alphas = captured_alpha(model, monkeypatch)
    result = model.update(token, state)
    for left, right in zip(result.momentum, expected.momentum, strict=True):
        torch.testing.assert_close(left, right, rtol=1e-13, atol=1e-13)
    (alpha,) = alphas
    second, third = model.config.multiscale.bank_decays
    for layer in range(2):
        surprise = result.momentum[layer]
        torch.testing.assert_close(
            result.weights[layer], (1 - alpha) * state.weights[layer] + surprise, rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.levels[2 * layer],
            (1 - second) * state.levels[2 * layer] + surprise,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.levels[2 * layer + 1],
            (1 - third) * state.levels[2 * layer + 1] + surprise,
            rtol=0,
            atol=0,
        )


def test_scales_of_the_bank_are_the_hidden_scales_of_the_cascade():
    cascade, bank = multiscale(), multiscale("ema_bank")
    assert cascade.timescales == (16.0, 64.0, 256.0)
    assert bank.bank_decays == (1 / 64, 1 / 256)
    assert MULTISCALE_CAPACITIES == (1.0, 2.0, 4.0)
    assert cascade.conductances == (1 / 16, 1 / 32, 1 / 64)


def _weighted_norm(matrix, weights):
    scale = np.diag(weights)
    return np.abs(scale @ matrix @ np.linalg.inv(scale)).sum(axis=1).max()


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("coupling", [1 / 64, 1 / 16, 1 / 4])
def test_transition_contracts_uniformly_inside_the_pt1_box(form, coupling):
    model = multiscale(form, coupling)
    floor = MemoryStability().alpha_floor
    rho, weights = model.contraction(floor)
    assert rho < 1
    generator = np.random.default_rng(17)
    grid = np.concatenate([[floor, 1.0, 1 - coupling], generator.uniform(floor, 1, 200)])
    for alpha in grid:
        matrix = np.array(model.transition(float(alpha)))
        assert _weighted_norm(matrix, weights) <= rho + 1e-15
        assert np.abs(np.linalg.eigvals(matrix)).max() < 1
    # Productos de transiciones con α cambiante, incluidas las alternancias extremas.
    for _ in range(50):
        sequence = generator.choice(grid, size=400)
        product = np.eye(3)
        for alpha in sequence:
            product = np.array(model.transition(float(alpha))) @ product
        assert _weighted_norm(product, weights) <= rho**400 * (1 + 1e-9)
    # La primera fila de la cascada vale 1 − α_lo/2 en α_lo, así que ρ nunca baja de ahí.
    if form == "cascade":
        assert rho >= 1 - floor / 2 - 1e-15


def test_without_a_forgetting_floor_the_cascade_has_no_uniform_contraction():
    model = multiscale()
    rho, _ = model.contraction(1e-12)
    assert rho > 1 - 1e-11
    # Con α = 0 la cadena conserva la suma ponderada por capacidades: un autovalor vale 1.
    matrix = np.array(model.transition(0.0))
    capacities = np.array(MULTISCALE_CAPACITIES)
    leak = np.array([0, 0, model.conductances[2] / capacities[2]])
    assert np.abs(np.linalg.eigvals(matrix + np.outer(leak, [0, 0, 1]))).max() == pytest.approx(1)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("depth", [1, 2])
def test_state_stays_inside_the_bound_with_pt1_gates_and_clip(form, depth):
    gates = pt1()
    model = memory(depth=depth, multiscale=multiscale(form), stability=MemoryStability(**gates))
    rho, weights = model.config.multiscale.contraction(gates["alpha_floor"])
    momentum_bound = model.config.theta_max * gates["gradient_clip"] / (1 - gates["eta_ceiling"])
    state = model.initial_state(2)
    weights_vector = torch.tensor(weights, dtype=torch.float64)

    def norm(value):
        layers = []
        for layer in range(depth):
            stack = torch.stack(
                (value.weights[layer], value.levels[2 * layer], value.levels[2 * layer + 1])
            )
            layers.append((weights_vector[:, None, None, None] * stack.abs()).amax())
        return max(layers)

    bound = max(float(norm(state)), weights[0] * momentum_bound / (1 - rho))
    with torch.no_grad():
        for step in range(40):
            state = model.update(tokens(8, scale=50.0, seed=100 + step), state)
            assert all(m.abs().max() <= momentum_bound * (1 + 1e-12) for m in state.momentum)
            assert float(norm(state)) <= bound * (1 + 1e-12)


def test_full_jacobian_contains_the_exact_hidden_blocks_of_the_cascade():
    config = MACConfig(
        memory=MemoryConfig(dim=4, depth=1, max_batch=1, max_tokens=1, multiscale=multiscale()),
        heads=1,
        persistent_tokens=2,
        max_segment=1,
    )
    mac = TitansMAC(config, dtype=torch.float64)
    state = mac.initial_state(1)
    state = replace(state, memory=scrambled(mac.memory, state.memory))
    order = fast_state_dimension(config)
    assert order == 4 * 16
    jacobian = fast_state_jacobian(mac, tokens(1, batch=1), state)
    assert jacobian.shape == (order, order)
    g12, g23, g34 = config.memory.multiscale.conductances
    eye = torch.eye(16, dtype=torch.float64)
    second, third = slice(32, 48), slice(48, 64)
    # u2 y u3 no intervienen en la pérdida asociativa, así que sus bloques son exactos.
    torch.testing.assert_close(jacobian[second, second], (1 - (g12 + g23) / 2) * eye)
    torch.testing.assert_close(jacobian[second, third], g23 / 2 * eye)
    torch.testing.assert_close(jacobian[third, third], (1 - (g23 + g34) / 4) * eye)
    torch.testing.assert_close(jacobian[third, second], g23 / 4 * eye)
    torch.testing.assert_close(jacobian[third, slice(0, 16)], torch.zeros(16, 16).double())


@pytest.mark.parametrize("form", FORMS)
def test_future_tokens_and_other_flows_do_not_change_a_flow_state(form):
    model = memory(multiscale=multiscale(form))
    sequence = tokens(3)
    changed = sequence.clone()
    changed[0, 2] += 5.0
    changed[1] += 3.0
    with torch.no_grad():
        first = model.update(sequence[:, :2], model.initial_state(2))
        second = model.update(changed[:, :2], model.initial_state(2))
        full = model.update(changed, model.initial_state(2))
    for left, right in zip(
        (*first.weights, *first.levels), (*second.weights, *second.levels), strict=True
    ):
        torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
    # En la cascada el tercer nivel solo recibe la escritura a partir del tercer token.
    step = 1 if form == "ema_bank" else 2
    for left, right in zip(
        (*first.weights, *first.levels[::step]),
        (*second.weights, *second.levels[::step]),
        strict=True,
    ):
        assert not torch.equal(left[1], right[1])
    assert not torch.equal(full.weights[0][0], second.weights[0][0])


@pytest.mark.parametrize("form", FORMS)
def test_recovery_continues_bit_for_bit_and_rejects_another_contract(form):
    model = memory(multiscale=multiscale(form))
    with torch.no_grad():
        state = model.update(tokens(3), model.initial_state(2))
        payload = model.export_state(state)
        assert set(payload) >= {"levels"}
        restored = model.restore_state(payload)
        token = tokens(1, seed=8)
        expected, actual = model.update(token, state), model.update(token, restored)
    for left, right in zip(
        (*expected.weights, *expected.levels), (*actual.weights, *actual.levels), strict=True
    ):
        assert torch.equal(left, right)
    with pytest.raises(ValueError):
        memory().restore_state(payload)
    with pytest.raises(ValueError):
        memory(multiscale=multiscale("ema_bank" if form == "cascade" else "cascade")).restore_state(
            payload
        )


def test_budget_counts_the_hidden_levels():
    single = memory(max_state_bytes=2 * (2 * 2 * 16 * 8 + 8))
    single.initial_state(2)
    with pytest.raises(ValueError, match="presupuesto"):
        memory(multiscale=multiscale(), max_state_bytes=2 * (2 * 2 * 16 * 8 + 8)).initial_state(2)
    # La cuenta previa ya incluye los niveles, antes de reservar ningún tensor.
    with pytest.raises(ValueError, match="presupuesto"):
        memory(multiscale=multiscale(), max_state_bytes=2 * (2 * 2 * 16 * 8 + 8))._check_size(2)


@pytest.mark.parametrize(
    "options",
    [dict(form="power_law"), dict(coupling=0.0), dict(coupling=0.3), dict(coupling=float("nan"))],
)
def test_invalid_declarations_fail_before_building(options):
    with pytest.raises(ValueError):
        MemoryMultiscale(**options)


@pytest.mark.parametrize("form", FORMS)
def test_predictor_carries_the_levels_through_blocks_and_gradients(form):
    model = financial().FinancialPredictor(
        financial().FinancialConfig(
            specification(),
            variant="mac_online",
            hidden_size=32,
            seed=42,
            memory_multiscale=dict(form=form, coupling=1 / 16),
        ),
        dtype=torch.float64,
    )
    assert model.config.identity()["memory_multiscale"] == dict(form=form, coupling=1 / 16)
    flows = ("US/AAA", "US/BBB")
    state = model.initial_state(flows)
    usage = model.state_usage(state)
    assert usage["tensor_bytes_per_flow"] == (8 * 32) * 32 * 8 + 8 + 8
    state = model.prepare(decision(model.config, flows, 0), state).next_state
    payload = model.export_state_cpu(state)
    restored = model.restore_state(payload)
    batch = decision(model.config, flows, 1)
    expected, actual = model.prepare(batch, state), model.prepare(batch, restored)
    assert torch.equal(expected.point_predictions, actual.point_predictions)
    for left, right in zip(
        expected.next_state.mac.memory.levels, actual.next_state.mac.memory.levels, strict=True
    ):
        assert torch.equal(left, right)
    differentiable = model.initial_state(flows, differentiable=True)
    for event in range(2):
        prepared = model.prepare(
            decision(model.config, flows, event), differentiable, differentiable=True
        )
        differentiable = prepared.next_state
    prepared.point_predictions.sum().backward()
    gate = model.mac.memory.alpha_projection.weight.grad
    assert gate is not None and torch.isfinite(gate).all() and gate.abs().sum() > 0


def test_comparison_is_declared_with_its_control_rule_and_budget_but_not_executed():
    declaration = json.loads(DECLARATION.read_text())
    recipe = json.loads((ROOT / declaration["base_recipe"]).read_text())
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
    control = identities["control"]
    for role in ("innovation", "trivial_alternative"):
        changed = {key for key in identities[role] if identities[role][key] != control.get(key)}
        assert changed == {"memory_multiscale"}
    assert identities["innovation"]["memory_multiscale"]["form"] == "cascade"
    assert identities["trivial_alternative"]["memory_multiscale"]["form"] == "ema_bank"
    assert (
        identities["innovation"]["memory_multiscale"]["coupling"]
        == identities["trivial_alternative"]["memory_multiscale"]["coupling"]
    )
    assert {"search_cases", "optimizer_updates", "stopping_rule", "seeds"} <= set(
        declaration["equal"]
    )
    budget = declaration["budget"]
    assert budget["consumed"] == dict(gpu_hours=0, executions=0, optimizer_updates=0)
    assert declaration["pilot"]["gpu_hours_max"] == budget["declared"]["gpu_hours_max"] == 9
    rule = declaration["decision_rule"]
    assert rule["keep_if"][0].endswith("< -minimum_interpretable_difference")
