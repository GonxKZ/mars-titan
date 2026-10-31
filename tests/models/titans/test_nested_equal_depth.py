"""D1 de #751: estados por refinamiento, supervisión declarada e igual profundidad.

Las pruebas calculan pérdidas y gradientes sobre episodios ficticios, sin optimizador.
"""

import json
from pathlib import Path

import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, source
from torch import nn

from mars_titan.memory.mars_titan_variant import check_components, load_declaration
from mars_titan.models.quantile_head import PINBALL, QuantileHead, pinball_loss
from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
from mars_titan.models.titans.episodic_readout import (
    EpisodicPrediction,
    EpisodicReadout,
    EpisodicReadoutConfig,
    head_predictions,
)
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.training.mars_titan_run import ReadoutRecipe, readout_loss

HIDDEN = 32
# Configuraciones con cuatro aplicaciones del puente: anidado, plano exterior y plano interior.
ARMS = ((2, 2), (4, 1), (1, 4))


def readout(k, steps=None, seed=42):
    bridge = (
        None
        if steps is None
        else EvidenceBridgeConfig(
            codec_id=CODEC,
            hidden_size=HIDDEN,
            cycles=0,
            recurrence=BridgeRecurrenceConfig(steps, 1216),
        )
    )
    config = EpisodicReadoutConfig(
        CODEC, hidden_size=HIDDEN, refinements=k, seed=seed, evidence_bridge=bridge
    )
    return EpisodicReadout(config, dtype=torch.float64)


def bank(count=8):
    return EpisodeSnapshot.create(
        **source(count),
        codec_id=CODEC,
        context_id=CONTEXT,
        cutoff=10,
        device="cpu",
        dtype=torch.float64,
    )


def working_state(rows=3):
    return torch.linspace(-0.5, 0.5, rows * HIDDEN, dtype=torch.float64).reshape(rows, HIDDEN)


def heads():
    torch.manual_seed(5)
    return nn.Linear(HIDDEN, 1, dtype=torch.float64), QuantileHead(HIDDEN).double()


def predict(model, head, z=None):
    result = model(
        working_state() if z is None else z,
        bank(),
        context_id=CONTEXT,
        cutoff=10,
        differentiable=True,
    )
    point, quantiles = head_predictions(head, result.state)
    return EpisodicPrediction(point, result, quantiles)


def kind(head):
    return PINBALL if isinstance(head, QuantileHead) else "l1"


def manual(head, state, index, target):
    point, quantiles = head_predictions(head, state)
    if isinstance(head, QuantileHead):
        return pinball_loss(quantiles.index_select(0, index), target)
    return torch.nn.functional.l1_loss(point.index_select(0, index), target)


INDEX = torch.tensor([0, 2])
TARGET = torch.tensor([0.3, -0.2], dtype=torch.float64)


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("steps", [None, 2])
def test_each_refinement_keeps_its_state_and_the_last_one_is_the_result(k, steps):
    model = readout(k, steps)
    z = working_state()
    result = model(z, bank(), context_id=CONTEXT, cutoff=10)
    assert len(result.states) == len(result.reads) == k
    assert result.states[-1] is result.state
    previous = z
    for read, state in zip(result.reads, result.states, strict=True):
        torch.testing.assert_close(model._step(previous, z, read), state, rtol=0, atol=0)
        previous = state


def test_equal_depth_arms_share_parameters_and_bridge_work():
    models = [readout(k, steps) for k, steps in ARMS]
    reference = {name: value for name, value in models[0].named_parameters()}
    for model, (k, steps) in zip(models, ARMS, strict=True):
        assert k * steps == 4
        recurrence = model.config.evidence_bridge.recurrence
        assert recurrence.node_updates(3, k) == 12
        parameters = dict(model.named_parameters())
        assert parameters.keys() == reference.keys()
        for name, value in parameters.items():
            assert torch.equal(value, reference[name])


@pytest.mark.parametrize("which", [0, 1])
def test_without_declared_supervision_the_loss_is_the_emitted_one(which):
    head = heads()[which]
    prediction = predict(readout(2, 2), head)
    loss = readout_loss(prediction, head, INDEX, TARGET, loss=kind(head))
    if which:
        expected = pinball_loss(prediction.quantiles.index_select(0, INDEX), TARGET)
    else:
        expected = torch.nn.functional.l1_loss(
            prediction.point_predictions.index_select(0, INDEX), TARGET
        )
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize(("k", "steps", "supervised"), [(2, 2, (1, 2)), (4, 1, (2, 4))])
def test_declared_supervision_averages_the_loss_after_each_refinement(which, k, steps, supervised):
    head = heads()[which]
    prediction = predict(readout(k, steps), head)
    loss = readout_loss(prediction, head, INDEX, TARGET, loss=kind(head), supervised=supervised)
    states = prediction.readout.states
    expected = torch.stack([manual(head, states[s - 1], INDEX, TARGET) for s in supervised])
    torch.testing.assert_close(loss, expected.mean(), rtol=0, atol=1e-15)


def test_intermediate_supervision_reaches_the_reader_through_the_earlier_state():
    head = heads()[0]
    gradients = []
    for supervised in (None, (1, 2)):
        model = readout(2, 2)
        prediction = predict(model, head)
        readout_loss(prediction, head, INDEX, TARGET, loss="l1", supervised=supervised).backward()
        gradients.append({name: value.grad.clone() for name, value in model.named_parameters()})
    final, mixed = gradients
    assert any(not torch.equal(final[name], mixed[name]) for name in final)
    # Supervisar el primer estado añade su gradiente, así que la media no es la final a medias.
    model = readout(2, 2)
    prediction = predict(model, head)
    (0.5 * manual(head, prediction.readout.states[0], INDEX, TARGET)).backward()
    first = {name: value.grad for name, value in model.named_parameters()}
    for name in final:
        expected = 0.5 * final[name] + (first[name] if first[name] is not None else 0)
        torch.testing.assert_close(mixed[name], expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("supervised", [(1,), (1, 2, 4), (4,)])
def test_supervision_must_end_at_the_reader_k(supervised):
    head = heads()[0]
    prediction = predict(readout(2, 2), head)
    with pytest.raises(ValueError, match="K del lector"):
        readout_loss(prediction, head, INDEX, TARGET, loss="l1", supervised=supervised)


def test_the_recipe_declares_supervision_without_changing_the_previous_identity():
    plain = ReadoutRecipe()
    assert plain.supervised_refinements is None
    assert "supervised_refinements" not in plain.identity()
    declared = ReadoutRecipe(supervised_refinements=[1, 2])
    assert declared.supervised_refinements == (1, 2)
    assert declared.identity()["supervised_refinements"] == (1, 2)
    assert {k: v for k, v in declared.identity().items() if k != "supervised_refinements"} == (
        plain.identity()
    )


@pytest.mark.parametrize("value", [[], [2, 1], [1, 1], [0, 1], [1, 5], ["1"], [True], 2, "1,2"])
def test_the_recipe_rejects_invalid_supervision(value):
    with pytest.raises(ValueError, match="supervisados"):
        ReadoutRecipe(supervised_refinements=value)


def test_declared_arms_are_valid_variants_of_equal_depth_and_supervision():
    document = json.loads(Path("configs/experiments/nested-equal-depth-v1.json").read_text())
    declaration = load_declaration()
    assert document["default_enabled"] is False and document["final_test_access"] is False
    depths = {}
    for name, arm in document["arms"].items():
        components, _ = check_components(declaration, arm["components"])
        k = components.get("refinements", 1)
        steps = components["evidence_bridge"]["recurrence"]["steps"]
        assert k * steps == document["depth"]["bridge_steps_per_prediction"]
        recipe = ReadoutRecipe(**arm["recipe_changes"])
        supervised = recipe.supervised_refinements or (k,)
        assert supervised[-1] == k
        depths[name] = tuple(s * steps for s in supervised)
        changed = {key: value for key, value in components.items() if key != "refinements"}
        changed["evidence_bridge"] = dict(changed["evidence_bridge"], recurrence=None)
        assert changed == dict(
            episodic_bank="m1",
            evidence_bridge=dict(
                regions=("view_1", "view_2", "view_3"),
                topology="ring",
                cycles=0,
                bridge_gain=0.25,
                seed=42,
                recurrence=None,
            ),
        )
    # Los controles de cada contraste se supervisan en las mismas profundidades.
    for contrast in document["contrasts"]:
        if contrast["factor"] != "supervision":
            assert depths[contrast["variant"]] == depths[contrast["control"]]
