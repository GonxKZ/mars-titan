"""Inicialización, selección y sellos de la ampliación regional integrada."""

from dataclasses import replace

import pytest
import torch
from test_bridge_episodic_readout import DEVICES, bank, configuration
from test_financial_adapter import raw_batch, setup, specification
from test_frozen_financial import frozen_backend as frozen_backend

from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer


@pytest.mark.parametrize("device", DEVICES)
def test_common_projections_and_refiner_are_paired_without_duplicate_parameters(device):
    before = torch.random.get_rng_state()
    gpu = torch.cuda.get_rng_state(0) if device != "cpu" else None
    config = configuration()
    plain = EpisodicReadout(
        replace(config, evidence_bridge=None), device=device, dtype=torch.float64
    )
    regional = EpisodicReadout(config, device=device, dtype=torch.float64)
    assert not hasattr(regional, "query_projection") and not hasattr(regional, "value_projection")
    for left, right in (
        (plain.query_projection, regional.evidence_bridge.query),
        (plain.value_projection, regional.evidence_bridge.value),
        (plain.refinement, regional.refinement),
    ):
        for name, value in left.named_parameters():
            assert torch.equal(value, dict(right.named_parameters())[name])
    assert torch.equal(plain.step_logit, regional.step_logit)
    assert torch.equal(torch.random.get_rng_state(), before)
    if gpu is not None:
        assert torch.equal(torch.cuda.get_rng_state(0), gpu)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("selection,expected", [("per_step", 4), ("first_read", 1)])
def test_first_read_reuses_the_selection_without_changing_refinement_count(
    device, selection, expected, monkeypatch
):
    reader = EpisodicReadout(
        configuration(refinements=4, episode_selection=selection),
        device=device,
        dtype=torch.float64,
    )
    calls = []
    original = torch.argsort

    def count(*args, **options):
        calls.append(True)
        return original(*args, **options)

    with monkeypatch.context() as guard:
        guard.setattr(torch, "argsort", count)
        result = reader(
            torch.ones((2, 32), dtype=torch.float64, device=device),
            bank(device),
            context_id="b" * 64,
            cutoff=10,
        )
    assert len(calls) == expected and len(result.reads) == 4


@pytest.mark.parametrize("device", DEVICES)
def test_quantile_head_remains_ordered_and_its_median_is_the_emission(device):
    base, _ = setup()
    predictor = (
        FinancialPredictor(
            replace(base.config, head="quantile_head_v1"), device=device, dtype=torch.float64
        )
        .eval()
        .requires_grad_(False)
    )
    reader = (
        EpisodicReadout(configuration(), device=device, dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    consumer = FrozenFinancialConsumer(predictor, readout=reader)
    batch = DecisionBatch.from_validated(
        validated_cpu_batch(raw_batch(), specification()), device=device, dtype=torch.float64
    )
    result = consumer.prepare(
        batch,
        predictor.initial_state(batch.flow_ids),
        context_id="b" * 64,
        snapshot=bank(device, cutoff=batch.prediction_at[0]),
    )
    assert result.quantiles.shape == (2, 5)
    assert (result.quantiles[:, 1:] >= result.quantiles[:, :-1]).all()
    assert torch.equal(result.point_predictions, result.quantiles[:, 2])


@pytest.mark.parametrize("change", ["method", "configuration", "parameter", "edge_data"])
def test_frozen_seals_cover_the_regional_module(change):
    predictor, _ = setup()
    predictor.eval().requires_grad_(False)
    reader = EpisodicReadout(configuration(), dtype=torch.float64).eval().requires_grad_(False)
    consumer = FrozenFinancialConsumer(predictor, readout=reader)
    if change == "method":
        reader.evidence_bridge.forward = lambda *args, **kw: None
    if change == "configuration":
        reader.evidence_bridge.config = replace(reader.evidence_bridge.config, cycles=0)
    if change == "parameter":
        reader.evidence_bridge.region_query.data.add_(1)
    if change == "edge_data":
        reader.evidence_bridge.edges.data[0, 0] = False
    with pytest.raises(ValueError):
        consumer.verify()


def test_mechanism_consumer_requires_a_new_regional_policy_contract():
    predictor, _ = setup()
    predictor.eval().requires_grad_(False)
    reader = EpisodicReadout(configuration(), dtype=torch.float64).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="lector"):
        FrozenMechanismConsumer(predictor, readout=reader)
