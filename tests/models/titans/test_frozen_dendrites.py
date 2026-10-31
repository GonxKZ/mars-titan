"""N2 congelado con el contexto de toda la sesión y lecturas por bloques."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_frozen_financial import frozen_backend as frozen_backend

from mars_titan.models.titans.dendritic_context import DendriticPlan
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from tests.models.titans.test_financial_dendrites import OPTIONS, configured, raw


def frame():
    base, _ = configured()
    config = replace(base.config, active_dendrites={**OPTIONS, "min_assets": 2})
    model = FinancialPredictor(config, dtype=torch.float64)
    with torch.no_grad():
        model.active_dendrites.gain.fill_(1.0)
    model._seal_parameters()
    cpu = validated_cpu_batch(raw(), config.inputs)
    rows = [
        SimpleNamespace(
            flow_id=flow,
            sample_id=cpu.sample_ids[i],
            prediction_at=cpu.prediction_at[i],
            input_available_at=cpu.input_available_at[i],
            inputs={name: value[i] for name, value in cpu.inputs.items()},
        )
        for i, flow in enumerate(cpu.flow_ids)
    ]
    plan = DendriticPlan.create(
        config.active_dendrites,
        rows,
        seed=config.seed,
        input_contract=config.inputs.fingerprint(),
        context_id="a" * 64,
    )
    return model, cpu, rows, plan


def selected(model, indices):
    values = raw()
    values["inputs"] = {name: value[indices] for name, value in values["inputs"].items()}
    for key in ("presence", "prediction_at", "input_available_at"):
        values[key] = values[key][indices]
    values["sample_ids"] = [values["sample_ids"][i] for i in indices]
    return validated_cpu_batch(values, model.config.inputs)


def test_frozen_consumer_accepts_the_registered_dendritic_operator():
    model, batch = configured()
    model.eval().requires_grad_(False)
    consumer = FrozenFinancialConsumer(model)
    result = consumer.prepare(
        batch,
        model.initial_state(batch.flow_ids),
        context_id="a" * 64,
        session_complete=True,
    )
    assert result.point_predictions.shape == (2,)


def test_shared_neural_options_reject_a_view_from_another_snapshot():
    from mars_titan.models.titans.frozen_financial import neural_options

    model, cpu, _, plan = frame()
    batch = DecisionBatch.from_validated(cpu, dtype=torch.float64)
    view = plan.view(cpu)
    assert neural_options(model, batch, context_id="a" * 64, dendritic_context=view) == {
        "dendritic_context": view
    }
    with pytest.raises(ValueError):
        neural_options(model, batch, context_id="b" * 64, dendritic_context=view)


def test_one_event_context_drives_all_blocks_and_matches_the_full_group():
    model, cpu, _, plan = frame()
    consumer = FrozenFinancialConsumer(model.eval().requires_grad_(False))
    batch = DecisionBatch.from_validated(cpu, dtype=torch.float64)
    full = consumer.prepare(
        batch, model.initial_state(batch.flow_ids), context_id="a" * 64, session_complete=True
    )
    points = []
    for i in range(2):
        part = selected(model, [i])
        block = DecisionBatch.from_validated(part, dtype=torch.float64)
        view = plan.view(part)
        assert view.routes[0] != 0
        result = consumer.prepare(
            block, model.initial_state(block.flow_ids), context_id="a" * 64, dendritic_context=view
        )
        points.append(result.point_predictions)
    torch.testing.assert_close(torch.cat(points), full.point_predictions, rtol=1e-12, atol=1e-12)


def test_views_reject_other_inputs_contexts_and_configurations_and_do_not_share_tensors():
    model, cpu, _, plan = frame()
    view = plan.view(cpu)
    batch = DecisionBatch.from_validated(cpu, dtype=torch.float64)
    first = view.tensor(batch, model.config.active_dendrites, model.config.seed)
    first.fill_(8)
    assert view.tensor(batch, model.config.active_dendrites, model.config.seed).max() <= 1
    with pytest.raises(ValueError):
        view.validate(batch, model.config.active_dendrites, model.config.seed, context_id="b" * 64)
    with pytest.raises(ValueError):
        view.validate(
            batch, replace(model.config.active_dendrites, mode="constant"), model.config.seed
        )
    changed = raw()
    changed["inputs"]["prices"][0, 1, 3] += 0.1
    wrong = validated_cpu_batch(changed, model.config.inputs)
    with pytest.raises(ValueError):
        plan.view(wrong)
    changed = raw()
    changed["inputs"]["charts"] += 1
    other = DecisionBatch.from_validated(
        validated_cpu_batch(changed, model.config.inputs), dtype=torch.float64
    )
    with pytest.raises(ValueError):
        view.validate(other, model.config.active_dendrites, model.config.seed)


def test_event_record_is_independent_and_tampering_with_immutable_metadata_is_detected():
    model, cpu, _, plan = frame()
    original = plan.record()
    with pytest.raises(TypeError):
        plan.entries[0] = plan.entries[0]
    copied = plan.record()
    copied["entries"][0][5] = 0
    assert plan.record() == original
    view = plan.view(cpu)
    object.__setattr__(view, "routes", (0, 0))
    batch = DecisionBatch.from_validated(cpu, dtype=torch.float64)
    with pytest.raises(ValueError):
        view.tensor(batch, model.config.active_dendrites, model.config.seed)
    object.__setattr__(plan, "context_id", "b" * 64)
    with pytest.raises(ValueError):
        plan.record()


def test_e2_registers_n2_and_mechanisms_still_reject_unregistered_context():
    from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
    from mars_titan.memory.spaced_retention import SpacedRetentionConfig
    from mars_titan.memory.spaced_session import SpacedTitansBinding
    from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
    from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

    model, _ = configured()
    model.eval().requires_grad_(False)
    base = (
        FinancialPredictor(replace(model.config, active_dendrites=None), dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    FrozenMechanismConsumer(base)
    with pytest.raises(ValueError, match="contexto N2"):
        FrozenMechanismConsumer(model)
    codec = FrozenEpisodeCodec(base.config.inputs)
    readout = (
        EpisodicReadout(
            EpisodicReadoutConfig(codec.fingerprint(), hidden_size=32), dtype=torch.float64
        )
        .eval()
        .requires_grad_(False)
    )
    retention = SpacedRetentionConfig(capacity=8)
    SpacedTitansBinding(FrozenFinancialConsumer(base, readout=readout), codec, retention, "m1")
    spaced = SpacedTitansBinding(
        FrozenFinancialConsumer(model, readout=readout), codec, retention, "m1"
    )
    assert "dendritic_session" in spaced.code()


def test_plan_and_view_factories_reject_invalid_sources_before_use():
    model, cpu, rows, plan = frame()
    parameters = dict(
        seed=model.config.seed,
        input_contract=model.config.inputs.fingerprint(),
        context_id="a" * 64,
    )
    with pytest.raises(ValueError):
        DendriticPlan.create(model.config.active_dendrites, [], **parameters)
    with pytest.raises(ValueError):
        DendriticPlan.create(
            model.config.active_dendrites, rows, **(parameters | {"context_id": "bad"})
        )
    bad = [SimpleNamespace(**vars(row)) for row in rows]
    bad[0].inputs = dict(bad[0].inputs, prices=bad[0].inputs["prices"][:1])
    with pytest.raises(ValueError):
        DendriticPlan.create(model.config.active_dendrites, bad, **parameters)
    with pytest.raises(ValueError):
        plan.view(None)
    wrong = DendriticPlan.create(
        model.config.active_dendrites, rows, **(parameters | {"input_contract": "f" * 64})
    )
    with pytest.raises(ValueError):
        wrong.view(cpu)
    view = plan.view(cpu)
    with pytest.raises(ValueError):
        view.validate(cpu, model.config.active_dendrites, model.config.seed)
    again = DendriticPlan.create(model.config.active_dendrites, rows, **parameters)
    assert plan == again and plan is not again
    assert plan.record()["schema_version"] == 1
    assert len(plan.record()["columns"]) == len(plan.record()["entries"][0]) == 8


def test_bound_context_cannot_be_used_by_a_plain_predictor_or_outside_its_snapshot():
    model, cpu, _, plan = frame()
    batch = DecisionBatch.from_validated(cpu, dtype=torch.float64)
    plain = FinancialPredictor(replace(model.config, active_dendrites=None), dtype=torch.float64)
    with pytest.raises(ValueError):
        plain.prepare(batch, plain.initial_state(batch.flow_ids), dendritic_context=plan.view(cpu))
    consumer = FrozenFinancialConsumer(model.eval().requires_grad_(False))
    with pytest.raises(ValueError):
        consumer.prepare(
            batch,
            model.initial_state(batch.flow_ids),
            context_id="b" * 64,
            dendritic_context=plan.view(cpu),
        )


def test_registered_regime_source_is_part_of_strong_verification(monkeypatch, tmp_path):
    from importlib import import_module
    from pathlib import Path

    model, _ = configured()
    consumer = FrozenFinancialConsumer(model.eval().requires_grad_(False))
    module = import_module("mars_titan.memory.regimes")
    changed = tmp_path / "regimes.py"
    changed.write_bytes(Path(module.__file__).read_bytes() + b"\n")
    monkeypatch.setattr(module, "__file__", str(changed))
    with pytest.raises(ValueError, match="código"):
        consumer.verify(strong=True)
