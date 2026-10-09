"""GRU episódica en el coordinador financiero con etiquetas manuales y sin ajustes."""

import copy

import numpy as np
import pytest
import torch
from test_financial_session import moment
from test_financial_session_controls import FLOWS
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_native_episode_backend import native as native
from test_native_episode_backend import record

from mars_titan.data.input_policy import MODALITIES
from mars_titan.memory.candidate_bank import CandidateBankConfig
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.candidate.episode_codec import FrozenCandidateCodec
from mars_titan.models.candidate.frozen_consumer import FrozenCandidateConsumer
from mars_titan.models.candidate.input_adapter import CandidateInputAdapter
from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch

EVENTS = (125, 126, 127, 128)


@pytest.fixture
def gru_native(native):
    """Enlace con el candidato GRU y el reservorio compartido, o una omisión explícita."""
    if not hasattr(native, "causal_reservoir_draws") or not hasattr(native, "Candidate"):
        pytest.skip("El enlace no incluye el candidato GRU ni el reservorio compartido")
    return native


@pytest.fixture
def gru(gru_native, four_flow_source):
    return create(four_flow_source)


def create(source, *, dtype=torch.float64, refinements=1, key_seed=44):
    adapter = CandidateInputAdapter(source["spec"], dtype=dtype, key_seed=key_seed)
    adapter.model.eval()
    for value in adapter.model.named_parameters().values():
        value.requires_grad_(False)
    return FrozenCandidateConsumer(adapter, refinements=refinements), FrozenCandidateCodec(adapter)


def phase(warmup=125, decision=125):
    return FinancialPhase("validation", moment(warmup), moment(decision), moment(200), moment(201))


def open_session(native, source, pair, output, *, admission="m1", capacity=4, **options):
    consumer, codec = pair
    return FinancialSession(
        output,
        native=native,
        consumer=consumer,
        codec=codec,
        prefixes=source["prefixes"],
        retention=options.pop("retention", CandidateBankConfig(capacity=capacity, seed=73)),
        phase=options.pop("phase", phase()),
        world="manual_gru",
        fold="0",
        admission=admission,
        block_rows=options.pop("block_rows", 4),
        **options,
    )


def labels(native, predictions, index, value=None):
    return [
        native.Feedback(
            p.id, 0, moment(index), 0.125 * (int(p.asset[-1]) + 1) if value is None else value
        )
        for p in predictions
    ]


def run_events(native, source, session, events=EVENTS, *, order=None):
    previous, emitted = [], []
    for index in events:
        batches = source["batches"][index]
        if order is not None:
            batches = [batches[position] for position in order]
        result = session.step(batches, labels(native, previous, index))
        emitted.append({p.asset: p.value for p in result.predictions})
        previous = result.predictions
    return emitted


def bank_state(session):
    bundle = session._bundle(session.snapshot()["state"])
    return session._bank(bundle["bank"] if bundle else None)


def bank_payload(session):
    bank, episodes = bank_state(session)
    return bank.snapshot(), episodes


def same_bank(left, right):
    (left_bank, left_episodes), (right_bank, right_episodes) = left, right
    assert left_episodes == right_episodes
    assert set(left_bank) == set(right_bank)
    for name, value in left_bank.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, right_bank[name]), name
        else:
            assert value == right_bank[name], name


def expected_predictions(source, consumer, bank, index):
    """Recalcular cada fila con la GRU nativa, K explícito y la instantánea confirmada
    antes de las etiquetas del evento, sin pasar por el consumidor."""
    memory = consumer.memory(bank.read_view() if bank is not None and bank.seen else None)
    values = {}
    for batch in source["batches"][index]:
        tensors = DecisionBatch.from_validated(batch, dtype=consumer.dtype)
        inputs = consumer.adapter.native.CandidateInputs(
            *(tensors.inputs[name] for name in MODALITIES), tensors.presence
        )
        with torch.no_grad():
            quantiles = consumer.model.forward(inputs, memory, consumer.refinements).quantiles
        values[batch.flow_ids[0]] = float(quantiles[0, 2])
    return values


def earlier_inputs(source, delay_us):
    """Mismas modalidades con disponibilidad anterior al corte de decisión."""
    batches = {}
    for index, group in source["batches"].items():
        batches[index] = [
            validated_cpu_batch(
                dict(
                    inputs={name: np.array(batch.inputs[name]) for name in MODALITIES},
                    presence=np.array(batch.presence),
                    sample_ids=list(batch.sample_ids),
                    prediction_at=np.array(batch.prediction_at, dtype="datetime64[us]"),
                    input_available_at=np.array(
                        [at - delay_us for at in batch.prediction_at], dtype="datetime64[us]"
                    ),
                ),
                source["spec"],
            )
            for batch in group
        ]
    return dict(source, batches=batches)


@pytest.mark.parametrize("refinements", [1, 2, 4])
def test_each_event_reads_one_snapshot_before_admitting_its_mature_labels(
    gru_native, four_flow_source, tmp_path, refinements
):
    native, source = gru_native, four_flow_source
    pair = create(source, refinements=refinements)
    with open_session(native, source, pair, tmp_path / "run", block_rows=1) as session:
        previous, admitted = [], []
        for index in EVENTS:
            before, _ = bank_state(session)
            result = session.step(source["batches"][index], labels(native, previous, index))
            issued = {p.asset: p.value for p in result.predictions}
            assert issued == expected_predictions(source, pair[0], before, index)
            bank, episodes = bank_state(session)
            assert bank.seen == before.seen + len(result.applied) == session.snapshot()["applied"]
            bundle = session._bundle(session.snapshot()["state"])
            control = session._read(bundle["control"], "control")
            assert (control["refinements"], control["observations"]) == (refinements, 4)
            by_decision = {(e["flow_id"], e["prediction_at"]): e for e in episodes.values()}
            for item in result.applied:
                episode = by_decision.get((item.prediction.asset, item.prediction.decision_at))
                if episode is not None:
                    assert episode["issued_prediction"] == item.prediction.value
                    assert episode["error"] == item.label.value - item.prediction.value
            admitted.append(bank.seen)
            previous = result.predictions
        assert admitted == [0, 4, 8, 12]
        assert session.diagnostics() | {"cursor": 0} == dict(
            cursor=0, admitted=12, retained=4, pending=4, capacity=4
        )


def test_bank_keeps_the_codec_bytes_of_each_emitted_row_and_no_pending_label(
    native, four_flow_source, gru, tmp_path
):
    source, (_, codec) = four_flow_source, gru
    rows = {(b.flow_ids[0], b.prediction_at[0]): b for i in EVENTS for b in source["batches"][i]}
    with open_session(native, source, gru, tmp_path / "run", capacity=16) as session:
        run_events(native, source, session, EVENTS[:3])
        bank, episodes = bank_state(session)
        view = bank.read_view()
        assert view["ids"].tolist() == sorted(episodes) and bank.size == 8
        for row, identifier in enumerate(view["ids"].tolist()):
            episode = episodes[identifier]
            encoded = codec.encode(rows[(episode["flow_id"], episode["prediction_at"])])
            assert view["keys"][row].numpy().tobytes() == encoded.keys[0].tobytes()
            assert view["values"][row].numpy().tobytes() == encoded.values[0].tobytes()
            assert view["labels"][row].item() == episode["label"]
            assert view["times"][row].tolist() == [
                episode["prediction_at"],
                episode["input_available_at"],
                episode["maturity_at"],
            ]
        pending = {(p.asset, p.decision_at) for p in session._executor.pending()}
        assert len(pending) == 4
        assert not pending & {(e["flow_id"], e["prediction_at"]) for e in episodes.values()}


def test_asset_order_is_irrelevant_and_physical_blocks_only_change_rounding(
    native, four_flow_source, gru, tmp_path
):
    source = four_flow_source
    results = {}
    for name, order, block_rows in (
        ("canonical", None, 4),
        ("reversed", (3, 2, 1, 0), 4),
        ("single", (2, 0, 3, 1), 1),
    ):
        with open_session(native, source, gru, tmp_path / name, block_rows=block_rows) as session:
            results[name] = run_events(native, source, session, order=order), bank_payload(session)
    assert results["reversed"][0] == results["canonical"][0]
    same_bank(results["reversed"][1], results["canonical"][1])
    for single, canonical in zip(results["single"][0], results["canonical"][0], strict=True):
        assert single.keys() == canonical.keys()
        assert list(single.values()) == pytest.approx(
            list(canonical.values()), rel=1e-12, abs=1e-12
        )
    assert results["single"][1][0]["ids"].tolist() == results["canonical"][1][0]["ids"].tolist()


def test_m0_keeps_the_refiner_with_an_empty_read_and_never_writes(
    native, four_flow_source, gru, tmp_path
):
    source, (consumer, _) = four_flow_source, gru
    with open_session(
        native, source, gru, tmp_path / "run", admission="m0", block_rows=1
    ) as session:
        emitted = run_events(native, source, session)
        bank, episodes = bank_state(session)
        assert bank.seen == 0 and not episodes and session.snapshot()["applied"] == 12
    for index, issued in zip(EVENTS, emitted, strict=True):
        assert issued == expected_predictions(source, consumer, None, index)
        tensors = DecisionBatch.from_validated(source["batches"][index][0], dtype=consumer.dtype)
        refined = consumer.prepare(tensors, consumer.memory()).point_predictions
        assert torch.equal(refined, consumer.adapter.forward(source["batches"][index][0]).median)


def test_warmup_advances_cursors_without_emitting_or_writing(
    native, four_flow_source, gru, tmp_path
):
    source, (consumer, _) = four_flow_source, gru
    with open_session(
        native, source, gru, tmp_path / "run", phase=phase(125, 127), block_rows=1
    ) as session:
        for index in (125, 126):
            warm = session.step(source["batches"][index], [], kind="warmup")
            assert not warm.predictions
        cursors = session.fast_state()["references"]
        assert {flow: row["observed_steps"] for flow, row in cursors.items()} == {
            flow: 2 for flow in FLOWS
        }
        assert session.diagnostics()["pending"] == 0
        first = session.step(source["batches"][127], [])
        assert {p.asset: p.value for p in first.predictions} == expected_predictions(
            source, consumer, None, 127
        )
        session.step(source["batches"][128], labels(native, first.predictions, 128))
        assert session.diagnostics()["admitted"] == 4
        assert {row["observed_steps"] for row in session.fast_state()["references"].values()} == {4}


def test_settlement_and_closure_do_not_run_the_model_or_invent_episodes(
    native, four_flow_source, gru, tmp_path
):
    source = four_flow_source
    with open_session(native, source, gru, tmp_path / "run") as session:
        first = session.step(source["batches"][125], [])
        cursors = session.fast_state()
        settled = session.step(
            [], labels(native, first.predictions, 126), kind="settlement", cutoff=moment(126)
        )
        assert len(settled.applied) == 4 and not settled.predictions
        assert session.fast_state() == cursors and session.diagnostics()["admitted"] == 4
        second = session.step(source["batches"][127], [])
        closed = session.step([], [], kind="settlement", cutoff=moment(201), close_phase=True)
        assert len(closed.finalized) == len(second.predictions) == 4 and not closed.applied
        assert session.diagnostics()["admitted"] == 4 and session.diagnostics()["pending"] == 0
        final = session.snapshot()
    with open_session(native, source, gru, tmp_path / "run", resume=True) as restored:
        assert restored.snapshot() == final


@pytest.mark.parametrize("point", ["record_written", "before_commit", "committed"])
def test_cut_and_resume_reproduce_the_next_emission_and_bank_exactly(
    native, four_flow_source, gru, tmp_path, point
):
    source = four_flow_source
    with open_session(native, source, gru, tmp_path / "reference") as reference:
        expected = run_events(native, source, reference)
        expected_bank, expected_state = bank_payload(reference), reference.snapshot()
    with open_session(native, source, gru, tmp_path / "run") as run:
        first = run.step(source["batches"][125], [])
        second = run.step(source["batches"][126], labels(native, first.predictions, 126))

        def fail(boundary):
            if boundary == getattr(native.Boundary, point):
                raise RuntimeError("Corte de fixture GRU")

        with pytest.raises(RuntimeError, match="Corte de fixture"):
            run.step(source["batches"][127], labels(native, second.predictions, 127), fault=fail)
    with open_session(native, source, gru, tmp_path / "run", resume=True) as restored:
        if point != "committed":
            third = restored.step(source["batches"][127], labels(native, second.predictions, 127))
            assert {p.asset: p.value for p in third.predictions} == expected[2]
        pending = [p for p in restored._executor.pending() if p.decision_at == moment(127)]
        last = restored.step(source["batches"][128], labels(native, pending, 128))
        assert {p.asset: p.value for p in last.predictions} == expected[3]
        same_bank(bank_payload(restored), expected_bank)
        assert restored.snapshot() == expected_state


@pytest.mark.parametrize("problem", ["future", "nan"])
def test_future_or_nonfinite_labels_never_reach_the_bank(
    native, four_flow_source, gru, tmp_path, problem
):
    source = four_flow_source
    with open_session(native, source, gru, tmp_path / "run") as session:
        first = session.step(source["batches"][125], [])
        before = session.snapshot()
        latest = (tmp_path / "run/latest.json").read_bytes()
        feedback = (
            labels(native, first.predictions, 127)
            if problem == "future"
            else labels(native, first.predictions, 126, float("nan"))
        )
        with pytest.raises((ValueError, RuntimeError)):
            session.step(source["batches"][126], feedback)
    assert (tmp_path / "run/latest.json").read_bytes() == latest
    with open_session(native, source, gru, tmp_path / "run", resume=True) as restored:
        assert restored.snapshot() == before
        assert bank_state(restored)[0].seen == 0


def test_full_capacity_follows_the_native_v2_reservoir_for_the_same_ids(
    native, four_flow_source, gru, tmp_path
):
    source = four_flow_source
    with open_session(native, source, gru, tmp_path / "run", capacity=3) as session:
        run_events(native, source, session)
        bank, _ = bank_state(session)
    scope = native.MemoryScope()
    scope.world, scope.partition, scope.fold, scope.representation = "x", "validation", "0", "y"
    reference = native.EpisodicMemory(scope, 73, 3, 2)
    for identifier in range(1, bank.seen + 1):
        reference.write(record(native, identifier), 2 * bank.seen + 1)
    assert [r.id for r in bank.records()] == [r.id for r in reference.retained_records()]


def test_recovery_rejects_two_episodes_of_the_same_decision(
    native, four_flow_source, gru, tmp_path
):
    source = four_flow_source
    with open_session(native, source, gru, tmp_path / "run", capacity=16) as session:
        run_events(native, source, session, EVENTS[:2])
        bundle = session._bundle(session.snapshot()["state"])
        payload = copy.deepcopy(session._read(bundle["bank"], "bank"))
        first, second = sorted(payload["episodes"])[:2]
        rows = payload["snapshot"]["ids"].tolist()
        a, b = rows.index(first), rows.index(second)
        payload["snapshot"]["times"][b] = payload["snapshot"]["times"][a]
        payload["snapshot"]["labels"][b] = payload["snapshot"]["labels"][a]
        payload["episodes"][second] = dict(payload["episodes"][first])
        with pytest.raises(ValueError, match="repite"):
            session._bank(session._stage(payload, "bank"))


def test_construction_rejects_foreign_codecs_banks_and_admissions(
    native, four_flow_source, gru, tmp_path
):
    source = four_flow_source
    consumer, codec = gru
    other_keys = create(source, key_seed=45)[1]
    fp32_codec = create(source, dtype=torch.float32)[1]
    invalid = [
        (dict(), (consumer, other_keys), "no comparten"),
        (dict(), (consumer, fp32_codec), "no comparten"),
        (dict(), (consumer, source["codec"]), "admisión"),
        (
            dict(retention=RetentionConfig(capacity=4, frontier=1, new_candidates=2)),
            (consumer, codec),
            "admisión",
        ),
        (dict(admission="m2"), (consumer, codec), "admisión"),
        (dict(admission="m3"), (consumer, codec), "admisión"),
        (dict(block_rows=257), (consumer, codec), "lote"),
    ]
    for position, (options, pair, message) in enumerate(invalid):
        with pytest.raises(ValueError, match=message):
            open_session(native, source, pair, tmp_path / str(position), **options)


def test_gru_sessions_have_their_own_identity_and_digest_features(
    native, four_flow_source, gru, tmp_path
):
    source = four_flow_source
    with open_session(native, source, gru, tmp_path / "gru") as session:
        # El ejecutor exige exactamente la anchura declarada en cada observación.
        session.step(source["batches"][125], [])
        rows = session._rows(source["batches"][126])
        widths = {len(session._binding.features(row)) for row in rows}
        contract, model_id = session.contract_id, session.model_id
        assert session._binding.kind == "candidate_gru"
    assert widths == {session._binding.feature_width} == {16}
    other = create(source, refinements=2)
    with open_session(native, source, other, tmp_path / "k2") as session:
        assert (session.contract_id, session.model_id) != (contract, model_id)


def test_the_consumer_rejects_unfrozen_or_substituted_computation(four_flow_source, gru):
    consumer, _ = gru
    parameter = consumer.model.named_parameters()["head_bias"]
    parameter.requires_grad_(True)
    with pytest.raises(ValueError, match="gradientes"):
        consumer.verify()
    parameter.requires_grad_(False)
    consumer.prepare = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="sustituid"):
        consumer.verify()
    del consumer.prepare
    with torch.no_grad():
        parameter.add_(1.0)
    with pytest.raises(ValueError, match="pesos"):
        consumer.verify()
    consumer.verify(strong=False)


def test_input_availability_before_the_decision_is_kept_in_episode_times(
    native, four_flow_source, gru, tmp_path
):
    delay = 3_600_000_000
    source = earlier_inputs(four_flow_source, delay)
    with open_session(native, source, gru, tmp_path / "run") as session:
        run_events(native, source, session, EVENTS[:2])
        bank, episodes = bank_state(session)
    view = bank.read_view()
    for row, identifier in enumerate(view["ids"].tolist()):
        decision, available, maturity = view["times"][row].tolist()
        episode = episodes[identifier]
        assert (decision, available) == (episode["prediction_at"], decision - delay)
        assert available == episode["input_available_at"] and maturity == moment(126)
