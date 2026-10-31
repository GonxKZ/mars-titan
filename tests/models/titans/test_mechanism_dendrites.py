"""Un origen N2 común para todas las propuestas de una cohorte."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_frozen_financial import frozen_backend as frozen_backend

from mars_titan.models.titans.dendritic_context import DendriticPlan
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from tests.models.titans.test_frozen_dendrites import frame, selected


def begin(consumer, cpu, plan):
    return consumer.begin(
        consumer.initial_state(),
        flow_ids=cpu.flow_ids,
        cutoff=cpu.prediction_at[0],
        context_id="a" * 64,
        matures_at=tuple(at + 10 for at in cpu.prediction_at),
        dendritic_plan=plan,
    )


def test_mechanisms_use_one_source_across_partitioned_blocks_and_match_the_ordinary_consumer():
    model, cpu, _, plan = frame()
    model.eval().requires_grad_(False)
    consumer = FrozenMechanismConsumer(model)
    batch = DecisionBatch.from_validated(cpu, dtype=torch.float64)
    expected = FrozenFinancialConsumer(model).prepare(
        batch,
        model.initial_state(batch.flow_ids),
        context_id="a" * 64,
        dendritic_context=plan.view(cpu),
    )
    event = begin(consumer, cpu, plan)
    assert event.dendritic_source_id == plan.source_id
    for i in (1, 0):
        part = selected(model, [i])
        block = DecisionBatch.from_validated(part, dtype=torch.float64)
        event = consumer.prepare(
            block, model.initial_state(block.flow_ids), event, dendritic_context=plan.view(part)
        ).event
    result = consumer.finish(event)
    points = dict(zip(result.flow_ids, result.point_predictions, strict=True))
    np.testing.assert_allclose(
        [points[flow] for flow in batch.flow_ids],
        expected.point_predictions.numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert "mars_titan.models.titans.dendritic_context" in consumer.identity()["implementation"]


@pytest.mark.parametrize(
    "kind", ["missing", "object", "context", "seed", "partial", "contract", "cutoff"]
)
def test_begin_rejects_incomplete_or_foreign_sources(kind):
    model, cpu, rows, plan = frame()
    consumer = FrozenMechanismConsumer(model.eval().requires_grad_(False))
    if kind in ("context", "seed", "partial", "contract", "cutoff"):
        if kind == "cutoff":
            from types import SimpleNamespace

            rows = [SimpleNamespace(**vars(row)) for row in rows]
            for row in rows:
                row.prediction_at += 1
                row.sample_id = f"{row.flow_id}/{row.prediction_at}"
        plan = DendriticPlan.create(
            model.config.active_dendrites,
            rows[:1] if kind == "partial" else rows,
            seed=model.config.seed + int(kind == "seed"),
            input_contract="b" * 64 if kind == "contract" else cpu.input_contract_id,
            context_id=("b" if kind == "context" else "a") * 64,
        )
    elif kind == "missing":
        plan = None
    else:
        plan = object()
    with pytest.raises(ValueError, match="N2"):
        begin(consumer, cpu, plan)


def test_plain_mechanisms_refuse_a_dendritic_plan():
    parent, cpu, _, plan = frame()
    model = FinancialPredictor(replace(parent.config, active_dendrites=None), dtype=torch.float64)
    consumer = FrozenMechanismConsumer(model.eval().requires_grad_(False))
    with pytest.raises(ValueError, match="N2"):
        begin(consumer, cpu, plan)


def test_a_changed_plan_cannot_be_registered_after_its_source_was_sealed():
    model, cpu, _, plan = frame()
    consumer = FrozenMechanismConsumer(model.eval().requires_grad_(False))
    changed = list(plan.entries)
    first = changed[0]
    changed[0] = first[:5] + ((first[5] + 1) % 5,) + first[6:]
    object.__setattr__(plan, "entries", tuple(changed))
    with pytest.raises(ValueError, match="N2"):
        begin(consumer, cpu, plan)


def test_empty_dendritic_records_are_allowed_only_for_n2_without_observations():
    from types import SimpleNamespace

    from mars_titan.memory import dendritic_session

    model, _, _, _ = frame()
    active = FrozenMechanismConsumer(model.eval().requires_grad_(False))
    plain = FinancialPredictor(replace(model.config, active_dendrites=None), dtype=torch.float64)
    inactive = FrozenMechanismConsumer(plain.eval().requires_grad_(False))
    dendritic_session.verify(SimpleNamespace(consumer=active), {"dendritic_context": None}, [], 1)
    for consumer, record in ((inactive, None), (active, {})):
        with pytest.raises(ValueError, match="N2"):
            dendritic_session.verify(
                SimpleNamespace(consumer=consumer), {"dendritic_context": record}, [], 1
            )


def test_a_valid_block_view_from_another_whole_cohort_cannot_replace_the_event_source():
    from types import SimpleNamespace

    model, cpu, rows, plan = frame()
    consumer = FrozenMechanismConsumer(model.eval().requires_grad_(False))
    event = begin(consumer, cpu, plan)
    foreign = [SimpleNamespace(**vars(row)) for row in rows]
    changed = dict(foreign[1].inputs)
    changed["prices"] = changed["prices"].copy()
    changed["prices"][0, 3] += 0.2
    foreign[1].inputs = changed
    other = DendriticPlan.create(
        model.config.active_dendrites,
        foreign,
        seed=model.config.seed,
        input_contract=cpu.input_contract_id,
        context_id="a" * 64,
    )
    part = selected(model, [0])
    batch = DecisionBatch.from_validated(part, dtype=torch.float64)
    state = model.initial_state(batch.flow_ids)
    with pytest.raises(ValueError, match="N2"):
        consumer.prepare(batch, state, event, dendritic_context=other.view(part))
    with pytest.raises(ValueError, match="N2"):
        consumer.prepare(batch, state, event)
    assert consumer.prepare(batch, state, event, dendritic_context=plan.view(part)).event.rows
