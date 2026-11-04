"""R1, U1 y U2 de #786 en el predictor financiero: paridad, recetas, rechazos e identidad."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_financial_adapter import raw_batch, specification

from mars_titan.models.titans.config import GateBias, MemoryStability
from mars_titan.models.titans.differential_attention import DIFFERENTIAL_ATTENTION
from mars_titan.models.titans.financial import DecisionBatch, FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import (
    FrozenFinancialConsumer,
    _implementation,
)
from mars_titan.models.titans.huber_memory import HuberMemory
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.models.titans.scale_invariant_time import ScaleInvariantTime
from mars_titan.training.search_cases import checked_predictor_changes
from mars_titan.training.titans_walk_forward import _CODE

ROOT = Path(__file__).resolve().parents[3]
START, DAY = 1_609_459_200_000_000, 86_400_000_000
MODULES = ("huber_memory", "scale_invariant_time", "differential_attention")
DECLARATIONS = {
    "huber_memory": "configs/experiments/huber-memory-v1.json",
    "differential_attention": "configs/experiments/differential-attention-v1.json",
    "scale_invariant_time": "configs/experiments/scale-invariant-time-v1.json",
}
NEUTRAL = {
    "huber": dict(memory_huber=HuberMemory(100.0)),
    "differential": dict(differential_attention=DIFFERENTIAL_ATTENTION),
    "time_scales": dict(memory_time_scales=ScaleInvariantTime(0.0)),
}


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def config(variant="mac_online", **changes):
    # Las opciones de la receta base, gate_bias y la memoria con LayerNorm residual.
    options = dict(gate_bias=GateBias(), memory_residual_layer_norm=True) | changes
    return FinancialConfig(specification(), variant=variant, hidden_size=32, **options)


def steps(spec, count):
    return [
        DecisionBatch.from_corpus(raw_batch(at=START + i * DAY), spec, dtype=torch.float64)
        for i in range(count)
    ]


def walk(model, count=3):
    batches = steps(model.config.inputs, count)
    state = model.initial_state(batches[0].flow_ids, differentiable=True)
    points = []
    for batch in batches:
        prepared = model.prepare(batch, state, differentiable=True)
        points.append(prepared.point_predictions)
        state = prepared.next_state
    total = torch.stack(points)
    total.square().sum().backward()
    return total.detach(), {name: value.grad for name, value in model.named_parameters()}


@pytest.mark.parametrize("changes", NEUTRAL.values(), ids=NEUTRAL)
def test_neutral_settings_reproduce_the_reference_predictor(changes):
    expected, control = walk(FinancialPredictor(config(), dtype=torch.float64))
    observed, gradients = walk(FinancialPredictor(config(**changes), dtype=torch.float64))
    assert torch.equal(observed, expected)
    added = {name: gradients.pop(name) for name in set(gradients) - set(control)}
    for name, value in control.items():
        assert (value is None) == (gradients[name] is None), name
        assert value is None or torch.equal(value, gradients[name]), name
    lambdas = added.pop("mac.differential.weight", None)
    assert lambdas is None or bool((lambdas != 0).all())
    assert all(not torch.count_nonzero(value) for value in added.values())


def test_active_settings_change_the_predictions_with_finite_gradients():
    expected, _ = walk(FinancialPredictor(config(), dtype=torch.float64))
    combined = FinancialPredictor(
        config(
            memory_huber=HuberMemory(0.3),
            memory_time_scales=ScaleInvariantTime(6.0),
            differential_attention=DIFFERENTIAL_ATTENTION,
        ),
        dtype=torch.float64,
    )
    with torch.no_grad():
        combined.mac.differential.weight.copy_(
            torch.tensor([0.5, -0.25, 0.1, 0.3], dtype=torch.float64)
        )
    combined._seal_parameters()
    observed, gradients = walk(combined)
    assert not torch.equal(observed, expected)
    for name in (
        "mac.memory.huber_bias",
        "mac.differential.query",
        "mac.memory.alpha_projection.bias",
    ):
        assert torch.isfinite(gradients[name]).all() and torch.count_nonzero(gradients[name]), name
    identity = combined.config.identity()
    assert identity["memory_huber"] == HuberMemory(0.3).identity()
    assert identity["memory_time_scales"] == ScaleInvariantTime(6.0).identity()
    assert identity["differential_attention"]["name"] == DIFFERENTIAL_ATTENTION
    memory = combined.mac.config.identity()["memory"]
    assert {"associative_loss", "alpha_time_scales"} <= set(memory)


def test_the_default_identity_does_not_name_the_new_changes():
    identity = config().identity()
    assert not {"memory_huber", "memory_time_scales", "differential_attention"} & set(identity)


def test_recipes_declare_the_changes_as_values():
    declared = config(
        memory_huber={"initial_ratio": 2},
        memory_time_scales={"octaves": 4},
        differential_attention=DIFFERENTIAL_ATTENTION,
    )
    assert declared.memory_huber == HuberMemory(2.0)
    assert declared.memory_time_scales == ScaleInvariantTime(4.0)


@pytest.mark.parametrize(
    ("variant", "changes", "message"),
    [
        ("mac_frozen", dict(memory_huber=HuberMemory()), "necesitan mac_online"),
        ("mac_disabled", dict(memory_time_scales=ScaleInvariantTime()), "necesitan mac_online"),
        ("mac_online", dict(memory_huber={"initial_ratio": 1, "x": 2}), "todos sus campos"),
        ("mac_online", dict(memory_huber="huber_memory_v1"), "necesita HuberMemory"),
        ("mac_online", dict(memory_time_scales=6), "necesita ScaleInvariantTime"),
        ("mac_online", dict(differential_attention="differential"), "admite"),
        ("transformer_direct", dict(differential_attention=DIFFERENTIAL_ATTENTION), "admite"),
    ],
)
def test_undeclared_or_misplaced_changes_are_rejected(variant, changes, message):
    with pytest.raises(ValueError, match=message):
        config(variant, **changes)


def test_spread_half_lives_need_gate_bias_and_respect_the_box_of_pt1():
    with pytest.raises(ValueError, match="semivida de gate_bias"):
        replace(config(), gate_bias=None, memory_time_scales=ScaleInvariantTime())
    box = dict(memory_stability=MemoryStability(), gate_bias=GateBias(eta=0.2))
    with pytest.raises(ValueError, match="alpha_floor"):
        config(memory_time_scales=ScaleInvariantTime(6), **box)
    boxed = config(memory_time_scales=ScaleInvariantTime(0.25), **box)
    assert boxed.memory_time_scales.octaves == 0.25


@pytest.mark.parametrize("changes", NEUTRAL.values(), ids=NEUTRAL)
def test_the_local_control_needs_its_own_contrast(changes):
    with pytest.raises(ValueError, match="propio contraste con C"):
        FinancialPredictor(
            config(**changes),
            local_control=MACProjectionConfig(mode="disabled", rank=1, grid_size=4),
            dtype=torch.float64,
        )


@pytest.mark.parametrize("name", DECLARATIONS)
def test_declared_models_build_and_pass_the_screening_check(name):
    declaration = json.loads((ROOT / DECLARATIONS[name]).read_text(encoding="utf-8"))
    assert declaration["issue"] == 786 and declaration["status"] == "declared_not_trained"
    roles = {arm["role"] for arm in declaration["arms"].values()}
    assert roles == {"reference", "variant"}
    reference = FinancialPredictor(config(), dtype=torch.float64).mac.config.identity()
    for arm in declaration["arms"].values():
        changes = checked_predictor_changes(arm["predictor_changes"] or None) or {}
        model = FinancialPredictor(config(**changes), dtype=torch.float64)
        assert (model.mac.config.identity() != reference) == bool(changes)


def test_the_frozen_consumer_and_the_runners_identify_the_new_code():
    model = FinancialPredictor(
        config(
            memory_huber=HuberMemory(),
            memory_time_scales=ScaleInvariantTime(),
            differential_attention=DIFFERENTIAL_ATTENTION,
        ),
        dtype=torch.float64,
    )
    consumer = FrozenFinancialConsumer(model.eval().requires_grad_(False))
    code = _implementation()
    for module in MODULES:
        name = f"mars_titan.models.titans.{module}"
        assert name in code and name in _CODE
    assert consumer.identity()["implementation"] == json.loads(json.dumps(code))
