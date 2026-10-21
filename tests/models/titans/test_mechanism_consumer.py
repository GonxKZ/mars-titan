"""Propuestas de mecanismos sobre fixtures, sin entrenamientos ni publicación durable."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_context_selection import DEVICES, batch_on
from test_episodic_readout import CODEC, CONTEXT, source
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import KINDS, pt4

from mars_titan.calibration.cohort_feedback import CohortFeedback, CohortOutcome
from mars_titan.calibration.online_conformal import OnlineConformal
from mars_titan.models.quantile_head import LEVELS, QUANTILE_HEAD
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer


def calibration(**options):
    target = np.tile(np.arange(-9.0, 10.0), 2)
    model = OnlineConformal.start(
        target,
        np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], (38, 1)),
        np.repeat(np.array(["US", "CN"]), 19),
        levels=LEVELS,
        nominals=(0.8, 0.95),
        min_rows=1,
        rate_fraction=0.25,
    )
    return CohortFeedback.start(model, **options)


def consumer(
    device="cpu", *, kind=None, reader=False, calibrator=None, action=None, populated=False
):
    model = pt4(kind, head=QUANTILE_HEAD).to(device)
    if kind is not None:
        values = model.state_dict()
        values["session_context.output"] = torch.eye(32, dtype=torch.float64, device=device) * 0.2
        model.load_state_dict(values)
    model.eval().requires_grad_(False)
    readout = None
    if reader:
        readout = (
            EpisodicReadout(
                EpisodicReadoutConfig(
                    CODEC, hidden_size=32, refinements=1, mode="bank" if populated else "no_bank"
                ),
                dtype=torch.float64,
                device=device,
            )
            .eval()
            .requires_grad_(False)
        )
    return FrozenMechanismConsumer(model, readout=readout, calibration=calibrator, action=action)


def execute(engine, auxiliary, event=0, *, flows=("US/AAA", "US/BBB"), parts=None, action=None):
    batch = batch_on(engine.predictor, flows, event)
    at = batch.prediction_at[0]
    proposal = engine.begin(
        auxiliary,
        flow_ids=batch.flow_ids,
        cutoff=at,
        context_id=CONTEXT,
        matures_at=(at + 10,) * len(flows),
        action=action,
    )
    next_states = []
    for positions in parts or (tuple(range(len(flows))),):
        block = batch.select(list(positions))
        output = engine.prepare(block, engine.predictor.initial_state(block.flow_ids), proposal)
        proposal = output.event
        next_states.append(output.next_state)
    return engine.finish(proposal), next_states


@pytest.mark.parametrize("device", DEVICES)
def test_disabled_mechanisms_match_the_original_consumer_and_keep_the_source_snapshot(device):
    engine = consumer(device)
    state = engine.initial_state()
    batch = batch_on(engine.predictor, ("US/AAA", "US/BBB"), 0)
    old = FrozenFinancialConsumer(engine.predictor)
    expected = old.prepare(
        batch, engine.predictor.initial_state(batch.flow_ids), context_id=CONTEXT
    )
    result, states = execute(engine, state)
    np.testing.assert_array_equal(
        result.point_predictions, expected.point_predictions.cpu().numpy()
    )
    np.testing.assert_array_equal(result.quantiles, expected.quantiles.cpu().numpy())
    assert states[0].observed_steps.tolist() == [1, 1]
    assert state.cutoff is None and result.state.cutoff == batch.prediction_at[0]
    assert result.state.context is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
def test_context_and_routes_are_provisional_until_the_whole_cohort_and_partition_invariant(
    device, kind
):
    engine = consumer(device, kind=kind, reader=True, action=MechanismAction("parent", "k4cc"))
    before = engine.initial_state()
    complete, states = execute(engine, before)
    shuffled, blocks = execute(engine, before, parts=((1,), (0,)))
    np.testing.assert_allclose(complete.quantiles, shuffled.quantiles, rtol=1e-12, atol=1e-12)
    assert complete.state.context.committed_at[0] == complete.state.cutoff
    assert before.context.committed_at == (-1, -1) and before.context.pending_flows == ((), ())
    assert all(state.observed_steps.tolist() == [1] for state in blocks)
    assert states[0].mac.memory.steps.tolist() == [1, 1]
    assert [row["reader_steps"] for row in complete.rows] == [4, 4]
    if kind == KINDS[0]:
        assert complete.state.context.memory[0].steps.tolist() == [1]
    restored = engine.restore_state(engine.export_state(complete.state))
    assert restored.context.committed_at == complete.state.context.committed_at


@pytest.mark.parametrize("device", DEVICES)
def test_online_feedback_uses_the_emitted_cohort_after_the_current_prediction(device):
    engine = consumer(
        device, calibrator=calibration(), action=MechanismAction(calibration="online")
    )
    first, _ = execute(engine, engine.initial_state())
    second, _ = execute(engine, first.state, 1, action=MechanismAction(calibration="static"))
    assert second.state.calibration.pending_rows == 2
    at = second.state.cutoff
    outcomes = tuple(
        CohortOutcome("US", first.state.cutoff, row["sample_id"], at, 1000.0) for row in first.rows
    )
    after, updates = engine.feedback(second.state, outcomes, at=at)
    assert len(updates) == 1 and after.calibration.pending_rows == 0
    assert after.calibration.corrections("US")["0.8"] > 7
    assert second.state.calibration.corrections("US")["0.8"] == 7
    np.testing.assert_array_equal(second.quantiles[:, 2], second.point_predictions)
    restored = engine.restore_state(engine.export_state(after))
    third, _ = execute(engine, restored, 2)
    assert np.all(third.quantiles[:, -1] - third.raw_quantiles[:, -1] > 7)


def test_a_full_online_queue_uses_k1_before_feedback_frees_space():
    engine = consumer(
        reader=True,
        calibrator=calibration(max_rows=2),
        action=MechanismAction(route="k4cc", calibration="online"),
    )
    first, _ = execute(engine, engine.initial_state())
    second, states = execute(engine, first.state, 1)
    assert all("pending_calibration_budget" in row["reasons"] for row in second.rows)
    assert all(row["emitted_route"] == "k1" and row["reader_steps"] == 1 for row in second.rows)
    assert states[0].observed_steps.tolist() == [1, 1]
    assert second.state.calibration.pending_rows == 2


def test_byte_exhaustion_retains_k1_without_repeating_mac_or_undoing_the_selected_context():
    engine = consumer(
        kind=KINDS[0],
        reader=True,
        calibrator=calibration(max_bytes=4096),
        action=MechanismAction("parent", "k4cc", "online"),
    )
    flows = tuple(f"US/A{i:03d}" for i in range(32))
    result, states = execute(engine, engine.initial_state(), flows=flows)
    assert result.state.calibration.pending_rows == 0
    assert result.state.context.committed_at[0] == result.state.cutoff
    assert states[0].observed_steps.tolist() == [1] * 32
    assert all(
        row["executed_route"] == "k4cc" and row["emitted_route"] == "k1" for row in result.rows
    )
    assert all(
        row["reader_steps"] == 4 and row["discarded_reader_steps"] == 3 for row in result.rows
    )
    assert all("pending_calibration_bytes" in row["reasons"] for row in result.rows)


def test_administrative_closure_censors_future_maturity_without_fabricating_a_mature_update():
    engine = consumer(calibrator=calibration(), action=MechanismAction(calibration="online"))
    result, _ = execute(engine, engine.initial_state())
    at = result.state.cutoff + 1
    closed = engine.close_phase(result.state, at=at, evidence_sha256="d" * 64)
    assert closed.terminal["censored_rows"] == 2
    assert closed.calibration.corrections("US")["0.8"] == 7
    restored = engine.restore_state(engine.export_state(closed))
    with pytest.raises(ValueError, match="cerrad"):
        execute(engine, restored, 1)
    with pytest.raises(ValueError, match="cerrad"):
        engine.feedback(restored, (), at=at)


def test_wrong_identity_and_incomplete_or_repeated_blocks_are_contract_errors():
    engine = consumer()
    auxiliary = engine.initial_state()
    with pytest.raises(ValueError):
        execute(engine, replace(auxiliary, model_id="f" * 64))
    batch = batch_on(engine.predictor, ("US/AAA", "US/BBB"), 0)
    event = engine.begin(
        auxiliary,
        flow_ids=batch.flow_ids,
        cutoff=batch.prediction_at[0],
        context_id=CONTEXT,
        matures_at=(batch.prediction_at[0] + 10,) * 2,
    )
    block = batch.select([0])
    prepared = engine.prepare(block, engine.predictor.initial_state(block.flow_ids), event)
    with pytest.raises(ValueError, match="completa"):
        engine.finish(prepared.event)
    with pytest.raises(ValueError, match="repite"):
        engine.prepare(block, engine.predictor.initial_state(block.flow_ids), prepared.event)


@pytest.mark.parametrize("device", DEVICES)
def test_warmup_needs_no_label_maturity_and_never_reads_or_calibrates(device):
    engine = consumer(
        device,
        kind=KINDS[0],
        reader=True,
        calibrator=calibration(max_rows=1),
        action=MechanismAction("parent", "k4cr", "online"),
    )
    auxiliary = engine.initial_state()
    batch = batch_on(engine.predictor, ("US/AAA", "US/BBB"), 0)
    event = engine.begin(
        auxiliary,
        flow_ids=batch.flow_ids,
        cutoff=batch.prediction_at[0],
        context_id=CONTEXT,
        matures_at=None,
        warmup=True,
    )
    prepared = engine.prepare(batch, engine.predictor.initial_state(batch.flow_ids), event)
    result = engine.finish(prepared.event)
    assert result.point_predictions is result.quantiles is None
    assert result.state.context.committed_at[0] == result.state.cutoff
    assert result.state.calibration.pending_rows == 0
    assert all(row["reader_steps"] == 0 for row in result.rows)


def test_changed_static_calibration_is_rejected_by_the_frozen_identity():
    engine = consumer(calibrator=calibration())
    engine._static["groups"]["US"]["intervals"]["0.8"]["correction"] = 100
    with pytest.raises(ValueError, match="calibr"):
        engine.initial_state()


def test_unknown_calibration_object_is_rejected_before_calling_its_methods():
    class ForgedCalibration:
        def export(self):
            raise AssertionError("No se debe ejecutar un objeto de calibración ajeno")

    engine = consumer(calibrator=calibration())
    state = replace(engine.initial_state(), calibration=ForgedCalibration())
    with pytest.raises(ValueError):
        execute(engine, state)


@pytest.mark.parametrize("version", [True, 1.0, 2])
def test_restore_requires_the_exact_auxiliary_schema_version(version):
    engine = consumer()
    payload = engine.export_state(engine.initial_state())
    payload["schema_version"] = version
    with pytest.raises(ValueError):
        engine.restore_state(payload)


@pytest.mark.parametrize("device", DEVICES)
def test_populated_snapshot_uses_one_route_and_detects_byte_mutation_at_the_boundary(device):
    engine = consumer(device, reader=True, populated=True, action=MechanismAction(route="k4cr"))
    batch = batch_on(engine.predictor, ("US/AAA", "US/BBB"), 0)
    at = batch.prediction_at[0]
    snapshot = EpisodeSnapshot.create(
        **source(16),
        cutoff=at,
        codec_id=CODEC,
        context_id=CONTEXT,
        dtype=torch.float64,
        device=device,
    )
    event = engine.begin(
        engine.initial_state(),
        flow_ids=batch.flow_ids,
        cutoff=at,
        context_id=CONTEXT,
        matures_at=(at + 10,) * 2,
        snapshot=snapshot,
    )
    prepared = engine.prepare(batch, engine.predictor.initial_state(batch.flow_ids), event)
    result = engine.finish(prepared.event)
    assert all(row["reader_steps"] == 4 and row["reader_searches"] == 2 for row in result.rows)
    assert prepared.next_state.mac.memory.steps.tolist() == [1, 1]
    snapshot._values[1].data.add_(0.2)
    with pytest.raises(ValueError, match="instantánea|bytes"):
        engine.finish(prepared.event)


@pytest.mark.parametrize("field", ["max_batch", "predictor", "prepare"])
def test_consumer_configuration_and_dispatch_remain_frozen(field):
    engine = consumer()
    if field == "max_batch":
        engine.max_batch = 1
    elif field == "predictor":
        engine.predictor = consumer().predictor
    else:
        engine.prepare = lambda *_: None
    with pytest.raises(ValueError):
        engine.verify()


def test_a_foreign_snapshot_does_not_get_to_execute_its_verify_method():
    class ForeignSnapshot:
        def verify(self):
            raise AssertionError("No debe llamarse una instantánea ajena")

    engine = consumer()
    batch = batch_on(engine.predictor, ("US/AAA",), 0)
    with pytest.raises(ValueError):
        engine.begin(
            engine.initial_state(),
            flow_ids=batch.flow_ids,
            cutoff=batch.prediction_at[0],
            context_id=CONTEXT,
            matures_at=(batch.prediction_at[0] + 10,),
            snapshot=ForeignSnapshot(),
        )


def test_scalar_head_overflow_after_a_finite_core_is_not_emitted_as_a_prediction():
    model = pt4()
    flows = ("US/AAA",)
    batch = batch_on(model, flows, 0)
    base = model.prepare(batch, model.initial_state(flows)).working_state[0]
    index = int(base.abs().argmax())
    value = float(base[index])
    direction = 1 if value >= 0 else -1
    limit = torch.finfo(torch.float64).max
    weight = direction * limit / (2 * max(abs(value), 1))
    values = model.state_dict()
    values["head.weight"].zero_()
    values["head.weight"][0, index] = weight
    values["head.bias"].fill_(limit * 0.999 - weight * value)
    model.load_state_dict(values)
    model.eval().requires_grad_(False)
    reader = EpisodicReadout(
        EpisodicReadoutConfig(CODEC, hidden_size=32, mode="no_bank"), dtype=torch.float64
    )
    with torch.no_grad():
        reader.refinement.weight.zero_()
        reader.refinement.bias.fill_(direction * 100)
    reader.eval().requires_grad_(False)
    engine = FrozenMechanismConsumer(model, readout=reader)
    with pytest.raises(ValueError, match="finit"):
        execute(engine, engine.initial_state(), flows=flows)


@pytest.mark.cuda
@pytest.mark.parametrize("reader", [False, True])
def test_disabled_mixed_market_path_does_not_repeat_the_head_by_market(reader):
    engine = consumer("cuda:0", reader=reader)
    flows = ("CN/600000.SS", "US/AAA")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    ) as profile:
        result, _ = execute(engine, engine.initial_state(), flows=flows)
    heads = [
        event
        for event in profile.events()
        if event.name == "aten::linear" and event.input_shapes[1] == [5, 32]
    ]
    assert len(heads) == (2 if reader else 1)
    assert result.flow_ids == tuple(sorted(flows))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("action", MechanismCatalog().actions(), ids=lambda action: action.key)
def test_all_registered_combinations_execute_from_the_same_kind_of_snapshot(action, device):
    engine = consumer(
        device, kind=KINDS[0], reader=True, populated=True, calibrator=calibration(), action=action
    )
    auxiliary = engine.initial_state()
    batch = batch_on(engine.predictor, ("US/AAA", "US/BBB"), 0)
    at = batch.prediction_at[0]
    snapshot = EpisodeSnapshot.create(
        **source(16),
        cutoff=at,
        codec_id=CODEC,
        context_id=CONTEXT,
        dtype=torch.float64,
        device=device,
    )
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device != "cpu" else None
    event = engine.begin(
        auxiliary,
        flow_ids=batch.flow_ids,
        cutoff=at,
        context_id=CONTEXT,
        matures_at=(at + 10,) * 2,
        snapshot=snapshot,
    )
    prepared = engine.prepare(batch, engine.predictor.initial_state(batch.flow_ids), event)
    result = engine.finish(prepared.event)
    assert prepared.next_state.observed_steps.tolist() == [1, 1]
    assert prepared.next_state.mac.memory.steps.tolist() == [1, 1]
    assert all(row["emitted_route"] == action.route and not row["reasons"] for row in result.rows)
    assert all(row["reader_steps"] == int(action.route[1]) for row in result.rows)
    assert all(row["reader_searches"] == 1 + int("r" in action.route) for row in result.rows)
    assert result.state.calibration.pending_rows == (2 if action.calibration == "online" else 0)
    assert result.state.context.committed_at[0] == (at if action.context == "parent" else -1)
    assert auxiliary.context.committed_at == (-1, -1) and auxiliary.calibration.pending_rows == 0
    np.testing.assert_array_equal(result.quantiles[:, 2], result.point_predictions)
    assert np.isfinite(result.quantiles).all()
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))


def test_proposal_work_counts_created_and_discarded_calibration_separately():
    engine = consumer(
        kind=KINDS[0],
        reader=True,
        calibrator=calibration(max_bytes=4096),
        action=MechanismAction("parent", "k4cc", "online"),
    )
    result, _ = execute(
        engine, engine.initial_state(), flows=tuple(f"US/A{i:03d}" for i in range(32))
    )
    assert result.publication_status == "proposed_not_durable"
    assert result.work["core_observations"] == 32
    assert result.work["mac_online_observations"] == 32
    assert result.work["reader_refinements"] == 128
    assert result.work["context_updates_proposed"] == 1
    assert result.work["online_attempts"] == result.work["online_failures"] == 1
    assert result.work["online_selected_proposals"] == 0


def test_unknown_flow_iterator_is_rejected_before_iteration():
    class ForeignFlows:
        def __iter__(self):
            raise AssertionError("No se debe recorrer una colección sin límite")

    engine = consumer()
    with pytest.raises(ValueError):
        engine.begin(
            engine.initial_state(),
            flow_ids=ForeignFlows(),
            cutoff=1_609_459_200_000_000,
            context_id=CONTEXT,
            matures_at=(),
        )


@pytest.mark.parametrize("device", DEVICES)
def test_second_market_capacity_failure_discards_the_first_market_proposal(device):
    engine = consumer(
        device,
        reader=True,
        calibrator=calibration(max_bytes=8192),
        action=MechanismAction(route="k4cc", calibration="online"),
    )
    initial = engine.initial_state()
    flows = ("CN/600000.SS",) + tuple(f"US/A{i:03d}" for i in range(64))
    result, _ = execute(engine, initial, flows=flows)
    assert result.work["online_attempts"] == 2
    assert result.work["online_created_proposals"] > 0
    assert result.work["online_discarded_proposals"] == result.work["online_created_proposals"]
    assert result.work["online_selected_proposals"] == 0
    assert result.state.calibration.export() == initial.calibration.export()
    assert all(row["emitted_route"] == "k1" for row in result.rows)


@pytest.mark.parametrize("device", DEVICES)
def test_reader_admission_uses_the_logical_event_instead_of_physical_blocks(device):
    model = pt4(head=QUANTILE_HEAD).to(device).eval().requires_grad_(False)
    readout = (
        EpisodicReadout(
            EpisodicReadoutConfig(CODEC, hidden_size=32, mode="no_bank", max_working_bytes=300000),
            dtype=torch.float64,
            device=device,
        )
        .eval()
        .requires_grad_(False)
    )
    engine = FrozenMechanismConsumer(model, readout=readout, action=MechanismAction(route="k4cc"))
    flows = tuple(f"US/A{i:03d}" for i in range(32))
    complete, _ = execute(engine, engine.initial_state(), flows=flows)
    split, _ = execute(
        engine, engine.initial_state(), flows=flows, parts=tuple((i,) for i in range(32))
    )
    assert readout.estimated_bytes(1, 0, refinements=4) < 300000
    assert readout.estimated_bytes(32, 0, refinements=4) > 300000
    assert all("reader_budget" in row["reasons"] for row in complete.rows)
    np.testing.assert_allclose(complete.quantiles, split.quantiles, rtol=1e-12, atol=1e-12)
    numerical = {"point", "raw_quantiles", "k1_point", "k1_quantiles"}
    assert [
        {key: value for key, value in row.items() if key not in numerical} for row in complete.rows
    ] == [{key: value for key, value in row.items() if key not in numerical} for row in split.rows]


def test_emission_that_fits_json_but_not_feedback_reserve_returns_without_losing_created_work():
    engine = consumer(
        calibrator=calibration(max_bytes=4096), action=MechanismAction(calibration="online")
    )
    result, _ = execute(
        engine, engine.initial_state(), flows=tuple(f"US/A{i:03d}" for i in range(8))
    )
    assert result.work["online_created_proposals"] == 1
    assert result.work["online_discarded_proposals"] == 1
    assert result.work["online_selected_proposals"] == 0
    assert result.state.calibration.pending_rows == 0
    assert all("pending_calibration_bytes" in row["reasons"] for row in result.rows)


@pytest.mark.parametrize(
    "changes",
    [
        dict(warmup=1),
        dict(matures_at=[1_609_459_200_000_010]),
        dict(matures_at=(1_609_459_200_000_000,)),
        dict(matures_at=(1_609_459_199_999_999,)),
        dict(context_id="unknown"),
        dict(cutoff=1_704_067_200_000_000),
        dict(flow_ids=()),
        dict(flow_ids=("US/A",) * 8193),
        dict(warmup=True),
    ],
)
def test_bad_cohort_contracts_do_not_create_auxiliary_state_or_fallbacks(changes):
    engine = consumer(calibrator=calibration())
    initial = engine.initial_state()
    before = initial.calibration.state_sha256()
    arguments = (
        dict(
            flow_ids=("US/A",),
            cutoff=1_609_459_200_000_000,
            context_id=CONTEXT,
            matures_at=(1_609_459_200_000_010,),
        )
        | changes
    )
    with pytest.raises(ValueError):
        engine.begin(initial, **arguments)
    assert initial.cutoff is None and initial.calibration.state_sha256() == before


def test_closed_archive_cannot_forge_its_censored_count_or_advance_the_calibration_clock():
    engine = consumer(calibrator=calibration(), action=MechanismAction(calibration="online"))
    result, _ = execute(engine, engine.initial_state())
    closed = engine.close_phase(result.state, at=result.state.cutoff + 1, evidence_sha256="d" * 64)
    payload = engine.export_state(closed)
    assert payload["calibration"]["clock"] == result.state.cutoff
    payload["terminal"]["censored_rows"] = 0
    with pytest.raises(ValueError, match="censur"):
        engine.restore_state(payload)
