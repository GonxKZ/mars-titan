"""C1 y C6 dentro del predictor financiero, con las mismas entradas, estados y cabeza."""

from dataclasses import asdict, replace

import pytest
import torch
from test_financial_adapter import api, raw_batch, setup
from test_frozen_financial import frozen_backend as frozen_backend

from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.models.titans.regional_variants import (
    RegionAxisMixingConfig,
    SharedWorkspaceConfig,
)
from mars_titan.training.search_cases import checked_predictor_changes

OPERATOR = RegionalFusionConfig(topology="ring")
VARIANTS = (
    RegionAxisMixingConfig(OPERATOR),
    RegionAxisMixingConfig(OPERATOR, mixing="none"),
    SharedWorkspaceConfig(OPERATOR),
    SharedWorkspaceConfig(OPERATOR, write="dense", write_top_k=None),
)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_variant_travels_through_the_recipe_and_names_itself(variant):
    base, _ = setup()
    change = dict(regional_variant=asdict(variant))
    assert checked_predictor_changes(change) == change
    config = replace(base.config, **change)
    assert config.regional_variant == variant
    assert "regional_variant" not in base.config.identity()
    assert config.identity()["regional_variant"] == variant.identity()
    assert replace(base.config, regional_variant=None).identity() == base.config.identity()


@pytest.mark.parametrize(
    "other",
    [
        dict(regional_fusion=OPERATOR),
        dict(presence_modulation="film_v1"),
        dict(variant="transformer_direct"),
        dict(regional_variant=dict(operator=asdict(OPERATOR))),
    ],
)
def test_the_variant_replaces_the_operator_and_is_not_combined(other):
    base, _ = setup()
    with pytest.raises(ValueError):
        replace(base.config, **(dict(regional_variant=VARIANTS[0]) | other))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_variant_changes_only_the_fusion_and_advances_mac_once(variant, dtype):
    base, _ = setup()
    before = torch.get_rng_state().clone()
    config = replace(base.config, regional_variant=variant)
    model = api().FinancialPredictor(config, dtype=dtype).eval()
    assert torch.equal(before, torch.get_rng_state())
    reference = api().FinancialPredictor(base.config, dtype=dtype).eval()
    parameters = dict(model.named_parameters())
    for name, value in reference.named_parameters():
        assert torch.equal(value, parameters[name])
    assert {name for name in parameters if name not in dict(reference.named_parameters())} == {
        name for name in parameters if name.startswith("regional_variant.")
    }
    assert model.operation_counts(2)["regional_local_node_evaluations"] == 2 * 5 * 2 * 2
    state = None
    for step in range(2):
        batch = api().DecisionBatch.from_corpus(
            raw_batch(at=1_609_459_200_000_000 + 10 * step),
            config.inputs,
            dtype=dtype,
        )
        if state is None:
            state = model.initial_state(batch.flow_ids, differentiable=True)
        result = model.prepare(batch, state, differentiable=True)
        state = result.next_state
        assert state.mac.memory.steps.tolist() == [step + 1, step + 1]
    result.point_predictions.square().sum().backward()
    gradients = [
        value.grad for name, value in parameters.items() if name.startswith("regional_variant.")
    ]
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert any(torch.count_nonzero(value) for value in gradients)


def test_mixing_predictor_starts_equal_to_its_control():
    base, batch = setup()
    outputs = []
    for variant in VARIANTS[:2]:
        model = (
            api()
            .FinancialPredictor(replace(base.config, regional_variant=variant), dtype=torch.float64)
            .eval()
        )
        state = model.initial_state(batch.flow_ids)
        outputs.append(model.prepare(batch, state).point_predictions)
    assert torch.equal(*outputs)


def test_a_saved_variant_restores_only_into_the_same_declaration():
    base, batch = setup()
    config = replace(base.config, regional_variant=VARIANTS[2])
    reference = api().FinancialPredictor(config, dtype=torch.float64).eval()
    saved = reference.state_dict()
    saved["regional_variant.initial_slots"] = saved["regional_variant.initial_slots"] + 0.5
    outputs = []
    for _ in range(2):
        model = api().FinancialPredictor(config, dtype=torch.float64).eval()
        model.load_state_dict(saved)
        outputs.append(model.prepare(batch, model.initial_state(batch.flow_ids)).point_predictions)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    original = reference.prepare(batch, reference.initial_state(batch.flow_ids)).point_predictions
    assert not torch.equal(outputs[0], original)
    other = api().FinancialPredictor(
        replace(base.config, regional_variant=VARIANTS[3]), dtype=torch.float64
    )
    with pytest.raises(ValueError):
        other.load_state_dict(saved)


def test_the_frozen_consumer_rejects_an_unregistered_variant(frozen_backend):
    base, _ = setup()
    model = (
        api()
        .FinancialPredictor(replace(base.config, regional_variant=VARIANTS[0]), dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    with pytest.raises(ValueError, match="conocidos"):
        FrozenFinancialConsumer(model)
