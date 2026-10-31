"""S2 en el ejecutor financiero con etiquetas manuales y sin estimación ni aprendizaje."""

import hashlib
import importlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_financial_session_gru import create as create_gru
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.memory.candidate_bank import CandidateBankConfig
from mars_titan.memory.failure_ledger import FailureLedgerConfig
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


def api():
    return importlib.import_module("mars_titan.memory.failure_session")


def consumer(source):
    model = (
        FinancialPredictor(
            FinancialConfig(source["spec"], variant="mac_online", hidden_size=32, seed=42),
            device="cpu",
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    return FrozenFinancialConsumer(model)


def policy(source, engine, tmp_path, *, mode="confidence_only", strength=0.5):
    catalog = tmp_path / "fixture-catalog.csv"
    indicators = source["spec"].representation["macro_indicators"]
    catalog.write_text("id,category\n" + "".join(f"{name},rates\n" for name in indicators))
    return api().FailureSessionConfig(
        ledger=FailureLedgerConfig(
            threshold=0.01,
            threshold_manifest_id="e" * 64,
            threshold_fitted_until=moment(100),
            source_manifest_id=engine.model_id,
            source_fitted_until=moment(100),
            capacity=8,
            max_pending=32,
            max_flows=8,
            cache_keys=True,
            strength=strength,
        ),
        macro_catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        catalog=catalog,
        mode=mode,
    )


def open_session(
    native, source, engine, path, config, *, resume=False, phase=None, codec=None, retention=None
):
    return FinancialSession(
        path,
        native=native,
        consumer=engine,
        codec=codec or source["codec"],
        prefixes=source["prefixes"],
        retention=retention
        or RetentionConfig(
            capacity=4,
            frontier=2,
            new_candidates=4,
        ),
        phase=phase
        or FinancialPhase("validation", moment(125), moment(125), moment(200), moment(201)),
        world="failure_fixture",
        fold="0",
        admission="m0",
        block_rows=2,
        failure_memory=config,
        resume=resume,
    )


def book(session):
    bundle = session._bundle(session.snapshot()["state"])
    return api().restore(session, bundle)


def test_confidence_only_registers_the_actual_emission_and_recovers_the_next_update(
    native,
    four_flow_source,
    tmp_path,
):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path)
    path = tmp_path / "run"
    with open_session(native, source, engine, path, config) as run:
        first = run.step(source["batches"][125], [])
        assert book(run).counts()["pending"] == 4
        feedback = [native.Feedback(p.id, 0, moment(126), p.value + 1.0) for p in first.predictions]
        second = run.step(source["batches"][126], feedback)
        assert book(run).counts()["admitted"] == 4
        snapshot = run.snapshot()
        ledger = book(run).export()
    with open_session(native, source, engine, path, config, resume=True) as restored:
        assert restored.snapshot() == snapshot
        assert book(restored).export() == ledger
        feedback = [
            native.Feedback(p.id, 0, moment(127), p.value + 2.0) for p in second.predictions
        ]
        restored.step(source["batches"][127], feedback)
        assert book(restored).counts()["resolved"] == 8
        assert book(restored).counts()["admitted"] == 8


@pytest.mark.parametrize("mode,strength", [("confidence_only", 0.5), ("shrink_point", 0.0)])
def test_disabling_attenuation_preserves_all_emitted_points(
    native, four_flow_source, tmp_path, mode, strength
):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path, mode=mode, strength=strength)
    outputs = []
    for name, selected in (("base", None), ("failure", config)):
        previous, emitted = [], []
        with open_session(native, source, engine, tmp_path / name, selected) as run:
            for index in (125, 126, 127):
                labels = [native.Feedback(p.id, 0, moment(index), p.value + 1.0) for p in previous]
                step = run.step(source["batches"][index], labels)
                emitted.append({p.asset: p.value for p in step.predictions})
                previous = step.predictions
        outputs.append(emitted)
    assert outputs[0] == outputs[1]


def test_shrinking_still_scores_the_final_point_that_was_actually_issued(
    native,
    four_flow_source,
    tmp_path,
):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path, mode="shrink_point")
    with open_session(native, source, engine, tmp_path / "shrink", config) as run:
        previous = []
        for index in (125, 126, 127, 128):
            labels = [native.Feedback(p.id, 0, moment(index), p.value + 2.0) for p in previous]
            step = run.step(source["batches"][index], labels)
            previous = step.predictions
        for record in book(run).export()["failures"]:
            emitted = {
                row["context"]["context_id"]: row["prediction"]
                for row in book(run).export()["emissions"]
            }[record["context_id"]]
            assert record["target"] - emitted == pytest.approx(2.0)
        bundle = run._bundle(run.snapshot()["state"])
        diagnostics = run._read(bundle["control"], "control")["failure_memory"]
        assert any(value < 1 for value in diagnostics["confidence"])
        actual = [p.value for p in sorted(step.predictions, key=lambda p: p.asset)]
        expected = [
            point * factor
            for point, factor in zip(
                diagnostics["source_points"], diagnostics["confidence"], strict=True
            )
        ]
        assert actual == expected


def test_asset_order_preserves_emissions_and_the_recovered_book(native, four_flow_source, tmp_path):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path)
    results = []
    for reverse in (False, True):
        previous, values = [], []
        with open_session(native, source, engine, tmp_path / str(reverse), config) as run:
            for index in (125, 126, 127):
                labels = [native.Feedback(p.id, 0, moment(index), p.value + 1.0) for p in previous]
                batches = source["batches"][index]
                step = run.step(list(reversed(batches)) if reverse else batches, labels)
                previous = step.predictions
                values.append({p.asset: p.value for p in previous})
            results.append((values, book(run).export()))
    assert results[0] == results[1]


def test_warmup_before_the_fit_boundary_does_not_create_out_of_sample_emissions(
    native,
    four_flow_source,
    tmp_path,
):
    from dataclasses import replace

    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path)
    config = replace(
        config,
        ledger=replace(
            config.ledger, source_fitted_until=moment(126), threshold_fitted_until=moment(126)
        ),
    )
    phase = FinancialPhase("validation", moment(125), moment(127), moment(200), moment(201))
    with open_session(native, source, engine, tmp_path / "warm", config, phase=phase) as run:
        run.step(source["batches"][125], [], kind="warmup")
        assert book(run).clock is None
        assert book(run).counts()["issued"] == 0
        run.step(source["batches"][127], [])
        assert book(run).counts()["issued"] == 4


def test_the_gru_keeps_its_own_codec_and_recovers_the_same_pending_book(
    native,
    four_flow_source,
    tmp_path,
):
    source = four_flow_source
    engine, codec = create_gru(source)
    config = policy(source, engine, tmp_path)
    options = dict(codec=codec, retention=CandidateBankConfig(capacity=4))
    path = tmp_path / "gru"
    with open_session(native, source, engine, path, config, **options) as run:
        first = run.step(source["batches"][125], [])
        labels = [native.Feedback(p.id, 0, moment(126), p.value + 1.0) for p in first.predictions]
        run.step(source["batches"][126], labels)
        saved = book(run).export()
        assert run._failure_producer.layout.key_width == 128
    with open_session(native, source, engine, path, config, resume=True, **options) as restored:
        assert book(restored).export() == saved


def test_constant_attenuation_has_its_own_declared_control(native, four_flow_source, tmp_path):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path, mode="constant_shrink")
    outputs = []
    for name, selected in (("base", None), ("constant", config)):
        with open_session(native, source, engine, tmp_path / name, selected) as run:
            step = run.step(source["batches"][125], [])
            outputs.append({p.asset: p.value for p in step.predictions})
    assert outputs[1] == {key: value * 0.75 for key, value in outputs[0].items()}


def test_settlement_drops_finalized_pending_without_turning_it_into_a_failure(
    native,
    four_flow_source,
    tmp_path,
):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path)
    with open_session(native, source, engine, tmp_path / "closed", config) as run:
        run.step(source["batches"][125], [])
        run.step([], [], kind="settlement", cutoff=moment(201), close_phase=True)
        assert book(run).counts() == dict(issued=4, resolved=0, censored=4, admitted=0, pending=0)


@pytest.mark.parametrize("bad_state", [[], {"emissions": None}, {"emissions": [None]}])
def test_malformed_wire_states_fail_before_decoding_contexts(
    native,
    four_flow_source,
    tmp_path,
    bad_state,
):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path)
    with open_session(native, source, engine, tmp_path / "malformed", config) as run:
        run.step(source["batches"][125], [])
        bundle = run._bundle(run.snapshot()["state"])
        reference = run._stage(
            dict(encoding="context_uint8_tensors_v1", state=bad_state), "failure_memory"
        )
        with pytest.raises(ValueError):
            api().restore(run, dict(bundle, failure_memory=reference))
        assert book(run).counts()["pending"] == 4


def test_terminal_boundary_can_close_the_book_without_opening_a_2024_prediction(
    native,
    four_flow_source,
    tmp_path,
):
    from mars_titan.models.titans.financial_inputs import FINAL_TEST_US

    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path)
    phase = FinancialPhase("validation", moment(125), moment(125), moment(200), FINAL_TEST_US)
    path = tmp_path / "terminal-boundary"
    with open_session(native, source, engine, path, config, phase=phase) as run:
        run.step(source["batches"][125], [])
        run.step([], [], kind="settlement", cutoff=FINAL_TEST_US, close_phase=True)
        assert book(run).clock == FINAL_TEST_US
        saved = book(run).export()
    with open_session(native, source, engine, path, config, phase=phase, resume=True) as restored:
        assert book(restored).export() == saved


@pytest.mark.parametrize(
    "field", ["source_manifest_id", "source_fitted_until", "threshold_fitted_until"]
)
def test_policy_provenance_cannot_describe_another_model_or_a_future_fit(
    native,
    four_flow_source,
    tmp_path,
    field,
):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path)
    value = "0" * 64 if field == "source_manifest_id" else moment(125)
    config = replace(config, ledger=replace(config.ledger, **{field: value}))
    with pytest.raises(ValueError):
        open_session(native, source, engine, tmp_path / "rejected", config)


@pytest.mark.parametrize(
    "change", [dict(mode="unknown"), dict(constant_confidence=True), dict(constant_confidence=1.1)]
)
def test_policy_configuration_rejects_implicit_or_out_of_range_controls(
    native,
    four_flow_source,
    tmp_path,
    change,
):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path)
    with pytest.raises(ValueError):
        replace(config, **change)


@pytest.mark.parametrize("edit", ["dtype", "size", "encoding"])
def test_wire_buffers_keep_their_exact_type_shape_and_encoding(
    native,
    four_flow_source,
    tmp_path,
    edit,
):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path)
    with open_session(native, source, engine, tmp_path / "bad-buffer", config) as run:
        run.step(source["batches"][125], [])
        bundle = run._bundle(run.snapshot()["state"])
        payload = run._read(bundle["failure_memory"], "failure_memory")
        context = payload["state"]["emissions"][0]["context"]
        if edit == "dtype":
            context["key_inputs"] = context["key_inputs"].to(torch.float32)
        elif edit == "size":
            context["key_inputs"] = context["key_inputs"][:-1].clone()
        else:
            payload["encoding"] = "unsupported"
        reference = run._stage(payload, "failure_memory")
        with pytest.raises(ValueError, match="representación|codificación"):
            api().restore(run, dict(bundle, failure_memory=reference))
        assert book(run).counts()["pending"] == 4


def test_mature_feedback_must_match_the_point_registered_by_the_native_executor(
    native,
    four_flow_source,
    tmp_path,
):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path)
    with open_session(native, source, engine, tmp_path / "emission-mismatch", config) as run:
        step = run.step(source["batches"][125], [])
        prediction = step.predictions[0]
        ledger = book(run)
        ledger.begin(moment(126))
        key = prediction.asset, prediction.decision_at
        ledger._pending[key] = replace(ledger._pending[key], prediction=prediction.value + 1.0)
        reference = api().stage(run, ledger)
        bundle = run._bundle(run.snapshot()["state"])
        outcome = SimpleNamespace(
            prediction=prediction, label=SimpleNamespace(available_at=moment(126), value=0.1)
        )
        with pytest.raises(ValueError, match="realmente emitida"):
            api().resolve(
                run, dict(bundle, failure_memory=reference, cutoff=moment(126)), [outcome], [], []
            )
        assert book(run)._pending[key].prediction == prediction.value


def test_all_assets_of_one_decision_share_the_snapshot_before_feedback(
    native,
    four_flow_source,
    tmp_path,
):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path)
    with open_session(native, source, engine, tmp_path / "shared-cutoff", config) as run:
        first = run.step(source["batches"][125], [])
        labels = [native.Feedback(p.id, 0, moment(126), p.value + 1.0) for p in first.predictions]
        run.step(source["batches"][126], labels)
        bundle = run._bundle(run.snapshot()["state"])
        control = run._read(bundle["control"], "control")["failure_memory"]
        assert control["eligible"] == [0, 0, 0, 0]
        assert book(run).counts()["admitted"] == 4
        run.step(source["batches"][127], [])
        bundle = run._bundle(run.snapshot()["state"])
        control = run._read(bundle["control"], "control")["failure_memory"]
        assert control["eligible"] == [4, 4, 4, 4]


def test_failure_before_commit_keeps_the_confirmed_book_and_replays_the_same_update(
    native,
    four_flow_source,
    tmp_path,
):
    source = four_flow_source
    engine = consumer(source)
    config = policy(source, engine, tmp_path)
    path = tmp_path / "interrupted"
    with open_session(native, source, engine, path, config) as run:
        first = run.step(source["batches"][125], [])
        before = run.snapshot()
        previous_book = book(run).export()
        labels = [native.Feedback(p.id, 0, moment(126), p.value + 1.0) for p in first.predictions]

        def fail(boundary):
            if boundary == native.Boundary.before_commit:
                raise RuntimeError("Corte técnico antes de publicar S2")

        with pytest.raises(RuntimeError, match="Corte técnico"):
            run.step(source["batches"][126], labels, fault=fail)
    with open_session(native, source, engine, path, config, resume=True) as restored:
        assert restored.snapshot() == before
        assert book(restored).export() == previous_book
        actual = restored.step(source["batches"][126], labels)
        assert book(restored).counts()["admitted"] == 4
        replayed = [p.value for p in actual.predictions]
    with open_session(native, source, engine, tmp_path / "reference", config) as reference:
        first = reference.step(source["batches"][125], [])
        labels = [native.Feedback(p.id, 0, moment(126), p.value + 1.0) for p in first.predictions]
        expected = reference.step(source["batches"][126], labels)
        assert replayed == [p.value for p in expected.predictions]
