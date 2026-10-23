"""Alternativas previas a etiquetas, desde estados comunes y sin ajuste de parámetros."""

import importlib
import weakref
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, source
from test_financial_adapter import raw_batch
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import KINDS
from test_mechanism_consumer import calibration, consumer, execute
from test_mechanism_observations import AT, DAY, observation_rows
from test_native_episode_backend import native as native

from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.models.titans.mechanism_catalog import MechanismAction

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def api():
    try:
        return importlib.import_module("mars_titan.memory.mechanism_collection")
    except ModuleNotFoundError:
        pytest.fail("Falta el colector de alternativas anteriores al resultado")


def collect(engine, *, parts=False, reverse=False):
    rows = observation_rows(engine)
    if reverse:
        rows = tuple(reversed(rows))
    builder = api().MechanismCollector(
        engine,
        engine.initial_state(),
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * len(rows),
    )
    for block in [rows[:1], rows[1:]] if parts else [rows]:
        builder.add(block, engine.predictor.initial_state(tuple(r.flow_id for r in block)))
    return builder.finish()


@pytest.mark.parametrize("device", DEVICES)
def test_all_admissible_alternatives_are_produced_without_changing_the_source_state(device):
    engine = consumer(device, kind=KINDS[0], reader=True, calibrator=calibration())
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(0) if device != "cpu" else None
    packet = collect(engine)
    body = packet.payload()
    assert body["kind"] == "conditional_mechanism_collection_v1"
    assert body["observation"]["observation"]["cutoff"] == AT
    assert [g["market"] for g in body["groups"]] == ["US"]
    candidates = body["groups"][0]["candidates"]
    assert len(candidates) == 36
    assert all(c["status"] == "observed" and c["requested"] == c["emitted"] for c in candidates)
    assert all(len(c["quantiles"]) == 2 for c in candidates)
    assert body["work"]["core_observations"] == 4
    assert body["work"]["mac_online_observations"] == 4
    assert body["work"]["reader_refinements"] == 36
    assert engine.initial_state().cutoff is None
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(0))
    assert api().CollectedMechanisms.restore(packet.record()).sha256 == packet.sha256


def test_unavailable_actions_remain_missing_instead_of_fake_zero_predictions():
    body = collect(consumer()).payload()
    candidates = body["groups"][0]["candidates"]
    assert candidates[0]["status"] == "observed"
    assert sum(c["status"] == "observed" for c in candidates) == 1
    assert all(c["quantiles"] is None for c in candidates[1:])
    assert all(c["reasons"] for c in candidates[1:])
    assert body["work"]["core_observations"] == 2


@pytest.mark.parametrize("device", DEVICES)
def test_every_collected_action_matches_its_fixed_control_from_a_nonempty_prefix(device):
    engine = consumer(device, kind=KINDS[0], reader=True, calibrator=calibration())
    previous, fast = execute(
        engine, engine.initial_state(), event=-1, action=MechanismAction("parent")
    )
    rows = observation_rows(engine)
    before = api().state_digest(engine.export_state(previous.state))
    fast_before = api().state_digest(engine.predictor.export_state(fast[0]))
    builder = api().MechanismCollector(
        engine, previous.state, rows=rows, cutoff=AT, context_id=CONTEXT, matures_at=(AT + DAY,) * 2
    )
    builder.add(rows, fast[0])
    body = builder.finish().payload()
    batch = DecisionBatch.from_corpus(
        raw_batch(), engine.predictor.config.inputs, device=device, dtype=torch.float64
    )
    for candidate in body["groups"][0]["candidates"]:
        event = engine.begin(
            previous.state,
            flow_ids=batch.flow_ids,
            cutoff=AT,
            context_id=CONTEXT,
            matures_at=(AT + DAY,) * 2,
            action=MechanismAction(*candidate["requested"].split("/")),
        )
        prepared = engine.prepare(batch, fast[0], event)
        emission = engine.finish(prepared.event)
        np.testing.assert_allclose(
            candidate["quantiles"], emission.quantiles, rtol=1e-12, atol=1e-12
        )
    assert api().state_digest(engine.export_state(previous.state)) == before
    assert api().state_digest(engine.predictor.export_state(fast[0])) == fast_before


@pytest.mark.parametrize("device", DEVICES)
def test_blocks_and_permutations_preserve_conditional_predictions(device):
    engine = consumer(device, kind=KINDS[0], reader=True, calibrator=calibration())
    full, split = collect(engine).payload(), collect(engine, parts=True, reverse=True).payload()
    assert full["observation"] == split["observation"]
    for first, second in zip(
        full["groups"][0]["candidates"], split["groups"][0]["candidates"], strict=True
    ):
        np.testing.assert_allclose(first["quantiles"], second["quantiles"], rtol=1e-12, atol=1e-12)
    assert full["work"]["core_observations"] == split["work"]["core_observations"] == 4


def test_incomplete_duplicate_or_closed_collection_cannot_publish_partial_results():
    engine = consumer()
    rows = observation_rows(engine)
    builder = api().MechanismCollector(
        engine,
        engine.initial_state(),
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * 2,
    )
    builder.add(rows[:1], engine.predictor.initial_state((rows[0].flow_id,)))
    with pytest.raises(ValueError):
        builder.finish()
    with pytest.raises(ValueError):
        builder.add(rows[:1], engine.predictor.initial_state((rows[0].flow_id,)))
    builder.add(rows[1:], engine.predictor.initial_state((rows[1].flow_id,)))
    builder.finish()
    with pytest.raises(ValueError):
        builder.finish()


def test_same_input_digest_does_not_allow_substituting_a_rows_arrays():
    engine = consumer()
    rows = observation_rows(engine)
    builder = api().MechanismCollector(
        engine,
        engine.initial_state(),
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * 2,
    )
    changed = np.frombuffer(np.full_like(rows[0].inputs["news"], 0.25).tobytes(), dtype=np.float32)
    poisoned = replace(rows[0], inputs=MappingProxyType({**rows[0].inputs, "news": changed}))
    with pytest.raises(ValueError, match="huella|hash|identidad"):
        builder.add((poisoned,), engine.predictor.initial_state((poisoned.flow_id,)))


@pytest.mark.parametrize(
    "field", ["future", "metadata", "extra", "median", "work", "counter", "blocks", "cross"]
)
def test_resealed_collection_cannot_change_temporal_or_attribution_contracts(field):
    engine = consumer(kind=KINDS[0], reader=True, calibrator=calibration())
    body = collect(engine).payload()
    if field == "future":
        body["rows"][0]["input_available_at"] = AT + 1
    elif field == "metadata":
        body["rows"][0]["input_sha256"] = "d" * 64
    elif field == "extra":
        body["target"] = 0.25
    elif field == "median":
        body["groups"][0]["candidates"][1]["quantiles"][0] = [-10.0, -5.0, 3.0, 5.0, 10.0]
    elif field == "work":
        body["work"]["core_observations"] = True
    elif field == "counter":
        body["work"]["core_observations"] = 0
    elif field == "blocks":
        body["blocks"][0]["flow_ids"] = [body["rows"][0]["flow_id"]]
    else:
        body["groups"][0]["candidates"][0]["cross_returns"] = [dict(market="CN")]
    with pytest.raises(ValueError):
        api().CollectedMechanisms(body)


def test_collection_record_is_immutable_and_rejects_changed_bytes():
    packet = collect(consumer())
    payload = packet.payload()
    payload["rows"][0]["sample_id"] = "changed"
    assert packet.payload()["rows"][0]["sample_id"] != "changed"
    record = packet.record()
    with pytest.raises(ValueError):
        api().CollectedMechanisms.restore(dict(record, sha256="d" * 64))
    record["payload_utf8"] += b" "
    with pytest.raises(ValueError):
        api().CollectedMechanisms.restore(record)


def many_rows(engine, size, *, china=False):
    raw = raw_batch(absent=False)
    raw["inputs"] = {
        name: np.repeat(value[:1], size, axis=0) for name, value in raw["inputs"].items()
    }
    raw["presence"] = np.ones((size, 5), dtype=np.bool_)
    flows = [f"US/T{i:04d}" for i in range(size)]
    if china:
        flows[0] = "CN/600000.SS"
    raw["sample_ids"] = [f"{flow}/{AT}" for flow in flows]
    raw["market"] = [flow.split("/")[0] for flow in flows]
    raw["prediction_at"] = np.full(size, AT, dtype="datetime64[us]")
    raw["input_available_at"] = np.full(size, AT - 1, dtype="datetime64[us]")
    return observation_rows(engine, raw)


def full_collection(engine, rows):
    builder = api().MechanismCollector(
        engine,
        engine.initial_state(),
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * len(rows),
    )
    builder.add(rows, engine.predictor.initial_state(tuple(r.flow_id for r in rows)))
    return builder.finish()


def test_changed_context_cannot_be_used_under_the_old_state_fingerprint():
    engine = consumer(kind=KINDS[0])
    rows, state = observation_rows(engine), engine.initial_state()
    builder = api().MechanismCollector(
        engine, state, rows=rows, cutoff=AT, context_id=CONTEXT, matures_at=(AT + DAY,) * 2
    )
    state.context.memory[0].weights[0].data.add_(0.25)
    with pytest.raises(ValueError, match="estado|contexto|prefijo"):
        builder.add(rows, engine.predictor.initial_state(tuple(r.flow_id for r in rows)))


def test_fallback_retains_the_actual_k1_but_does_not_observe_the_requested_online_action():
    engine = consumer(kind=KINDS[0], reader=True, calibrator=calibration(max_bytes=4096))
    body = full_collection(engine, many_rows(engine, 8)).payload()
    choices = body["groups"][0]["candidates"]
    assert sum(c["status"] == "fallback" for c in choices) == 12
    for item in choices:
        if item["status"] == "fallback":
            context = item["requested"].split("/")[0]
            reference = choices[0 if context == "off" else 18]
            assert item["emitted"] == context + "/k1/none"
            assert item["quantiles"] == reference["quantiles"]
    choices[2]["quantiles"][0] = [value + 0.25 for value in choices[2]["quantiles"][0]]
    with pytest.raises(ValueError):
        api().CollectedMechanisms(body)


def test_coupled_byte_returns_preserve_other_markets_anchor_and_record_its_return():
    engine = consumer(
        kind=KINDS[0],
        reader=True,
        calibrator=calibration(max_bytes=8192),
        action=MechanismAction("parent", "k4cc", "online"),
    )
    body = full_collection(engine, many_rows(engine, 65, china=True)).payload()
    cn, us = body["groups"]
    assert cn["market"] == "CN" and us["market"] == "US"
    candidate = cn["candidates"][0]
    assert candidate["joint_actions"] == {"CN": "off/k1/none", "US": "parent/k4cc/online"}
    assert candidate["status"] == "observed"
    assert candidate["cross_returns"] == [
        dict(
            market="US",
            requested="parent/k4cc/online",
            emitted="parent/k1/none",
            reasons=["pending_calibration_bytes"],
        )
    ]
    assert us["candidates"][2]["status"] == "fallback"


@pytest.mark.parametrize("device", DEVICES)
def test_individual_routes_are_used_if_nine_shared_steps_do_not_fit(device):
    from mars_titan.models.titans.episodic_readout import EpisodicReadout
    from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

    original = consumer(device, kind=KINDS[0], reader=True, calibrator=calibration())
    reader = original.readout
    cap = (
        reader.estimated_bytes(2, 0, refinements=4) + reader.estimated_bytes(2, 0, refinements=9)
    ) // 2
    limited = (
        EpisodicReadout(
            replace(reader.config, max_working_bytes=cap), device=device, dtype=torch.float64
        )
        .eval()
        .requires_grad_(False)
    )
    engine = FrozenMechanismConsumer(original.predictor, readout=limited, calibration=calibration())
    full, small = collect(original).payload(), collect(engine).payload()
    assert full["work"]["shared_route_batches"] == 2
    assert small["work"]["shared_route_batches"] == 0
    assert small["work"]["individual_route_batches"] == 12
    assert small["work"]["reader_refinements"] == 68
    for a, b in zip(full["groups"][0]["candidates"], small["groups"][0]["candidates"], strict=True):
        assert a["status"] == b["status"] == "observed"
        np.testing.assert_allclose(a["quantiles"], b["quantiles"], rtol=1e-12, atol=1e-12)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_external_artifacts_restore_collection_and_pending_supervision(native, tmp_path):
    from mars_titan.memory.mechanism_supervision import MechanismSupervision
    from mars_titan.memory.session_artifacts import SessionArtifacts

    engine = consumer()
    packet = collect(engine)
    pending = MechanismSupervision.start(packet)
    store = SessionArtifacts(native, tmp_path / "artifacts")
    reference = store.stage(packet.record(), identity=engine.model_id, kind="mechanism_collection")
    target_ref = store.stage(pending.record(), identity=engine.model_id, kind="mechanism_targets")
    loaded = api().CollectedMechanisms.restore(
        store.read(reference, identity=engine.model_id, kind="mechanism_collection")
    )
    labels = MechanismSupervision.restore(
        loaded, store.read(target_ref, identity=engine.model_id, kind="mechanism_targets")
    )
    assert loaded.sha256 == packet.sha256
    assert labels.record() == pending.record()
    assert labels.metrics(device="cpu") == pending.metrics(device="cpu")


def test_completed_builder_and_packet_do_not_keep_input_arrays_alive():
    engine = consumer()
    rows = observation_rows(engine)
    references = [weakref.ref(value) for row in rows for value in row.inputs.values()]
    builder = api().MechanismCollector(
        engine,
        engine.initial_state(),
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * 2,
    )
    builder.add(rows, engine.predictor.initial_state(tuple(row.flow_id for row in rows)))
    packet = builder.finish()
    del rows
    assert all(reference() is None for reference in references)
    assert packet.payload()["groups"][0]["candidates"][0]["status"] == "observed"


@pytest.mark.parametrize("device", DEVICES)
def test_float32_collection_records_its_precision_and_matches_the_fixed_emission(device):
    from mars_titan.models.titans.financial import FinancialPredictor
    from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

    parent = consumer(device).predictor
    model = (
        FinancialPredictor(parent.config, device=device, dtype=torch.float32)
        .eval()
        .requires_grad_(False)
    )
    engine = FrozenMechanismConsumer(model)
    body = collect(engine).payload()
    assert body["output_dtype"] == "torch.float32"
    assert body["execution_device"] == device
    batch = DecisionBatch.from_corpus(
        raw_batch(), model.config.inputs, device=device, dtype=torch.float32
    )
    event = engine.begin(
        engine.initial_state(),
        flow_ids=batch.flow_ids,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * 2,
    )
    prepared = engine.prepare(batch, model.initial_state(batch.flow_ids), event)
    emitted = engine.finish(prepared.event)
    np.testing.assert_array_equal(
        body["groups"][0]["candidates"][0]["quantiles"], emitted.quantiles
    )


@pytest.mark.parametrize("device", DEVICES)
def test_populated_snapshot_uses_shared_searches_and_matches_a_refreshed_fixed_route(device):
    from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot

    engine = consumer(device, kind=KINDS[0], reader=True, populated=True, calibrator=calibration())
    snapshot = EpisodeSnapshot.create(
        **source(16),
        cutoff=AT,
        codec_id=CODEC,
        context_id=CONTEXT,
        dtype=torch.float64,
        device=device,
    )
    rows = observation_rows(engine)
    builder = api().MechanismCollector(
        engine,
        engine.initial_state(),
        rows=rows,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * 2,
        snapshot=snapshot,
    )
    fast = engine.predictor.initial_state(tuple(r.flow_id for r in rows))
    builder.add(rows, fast)
    packet = builder.finish()
    body = packet.payload()
    assert body["observation"]["observation"]["bank_entries"] == 16
    assert body["work"]["reader_searches"] == 12
    batch = DecisionBatch.from_corpus(
        raw_batch(), engine.predictor.config.inputs, device=device, dtype=torch.float64
    )
    event = engine.begin(
        engine.initial_state(),
        flow_ids=batch.flow_ids,
        cutoff=AT,
        context_id=CONTEXT,
        matures_at=(AT + DAY,) * 2,
        snapshot=snapshot,
        action=MechanismAction(route="k4cr"),
    )
    emission = engine.finish(engine.prepare(batch, fast, event).event)
    np.testing.assert_allclose(
        body["groups"][0]["candidates"][12]["quantiles"], emission.quantiles, rtol=1e-12, atol=1e-12
    )
